#!/bin/sh
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# run-intelligence-migrations.sh — Apply omniintelligence database migrations
#
# Creates the omniintelligence database if absent and applies all pending SQL
# migrations in order (000–023), tracking applied migrations in a
# schema_migrations table within the omniintelligence database.
#
# This script is run by the intelligence-migration service (docker-compose
# runtime profile) as a one-shot init container before intelligence-api starts.
#
# Ticket: OMN-4082 (PIPELINE AUDIT GAP-4 — intelligence migration wiring)
#
# Environment:
#   POSTGRES_USER     (default: postgres)
#   POSTGRES_PASSWORD (required)
#   POSTGRES_HOST     (default: localhost)
#   POSTGRES_PORT     (default: 5432)
#   MIGRATIONS_DIR    (default: /migrations/intelligence)
#   PG_WAIT_RETRIES   (default: 30 — see section 0)

set -e

PGUSER="${POSTGRES_USER:-postgres}"
PGHOST="${POSTGRES_HOST:-postgres}"
PGPORT="${POSTGRES_PORT:-5432}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-/migrations/intelligence}"
PG_WAIT_RETRIES="${PG_WAIT_RETRIES:-30}"

export PGPASSWORD="${POSTGRES_PASSWORD}"

# ---------------------------------------------------------------------------
# 0. Wait for Postgres to accept connections (first-boot initdb race guard)
# ---------------------------------------------------------------------------
# OMN-17150 defect 1. This section is a port of run-forward-migrations.sh's own
# section 0 (OMN-13062) — same env var, same 2s interval, same fail-loud abort —
# and porting it IS the fix. The two one-shots start at the same instant behind
# the same `depends_on: postgres: {condition: service_healthy}`, and only ONE of
# them had this loop. That asymmetry, not the compose wiring, is why
# forward-migration survives a cold boot and this script did not.
#
# WHY service_healthy IS NOT ENOUGH. On a fresh volume the postgres image runs a
# TEMPORARY server for its initdb/init-script phase, and the compose healthcheck
# (`pg_isready` over the local unix socket) answers TRUE against that temporary
# server. Dependents are released; the temporary server is then stopped and the
# real one started, and every connection in that window is refused. Measured on
# omnibase-infra-lakshman 2026-08-31, deterministic across three clean boots:
# postgres started 16:31:52, this container started 16:31:57.879, died 150ms
# later on `Connection refused`, and postgres first reported healthy at 16:32:27.
# The compose-side half is fixed too (the healthcheck now probes TCP, which the
# temporary server does not listen on), but a one-shot that dies on the first
# refusal is fragile no matter how good the upstream signal is. The retry is what
# makes it correct rather than lucky.
#
# The first statement of section 1 below ends in `2>/dev/null || true`, so without
# this loop a refusal was swallowed, DB_EXISTS came back empty, and the script
# proceeded to CREATE DATABASE and exited 2. `restart: "no"` meant no retry, and
# omninode-runtime's `service_completed_successfully` dependency then held the
# entire runtime tier behind a container that had already given up. The documented
# workaround was "run `up -d` a second time"; this removes the need for it.
#
# Probes `postgres` (the always-present maintenance database), not
# `omniintelligence`, which section 1 may still have to create.
echo "[intelligence-migration] Waiting for Postgres to accept connections..."
retries=0
until psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres -c "SELECT 1" >/dev/null 2>&1; do
  retries=$((retries + 1))
  if [ "$retries" -ge "$PG_WAIT_RETRIES" ]; then
    echo "[intelligence-migration] ERROR: Postgres not ready after ${PG_WAIT_RETRIES} retries. Aborting." >&2
    exit 1
  fi
  echo "[intelligence-migration]   postgres not ready (attempt ${retries}/${PG_WAIT_RETRIES}), retrying in 2s..."
  sleep 2
done
echo "[intelligence-migration] Postgres is ready."

# ---------------------------------------------------------------------------
# 1. Create the omniintelligence database if it does not exist
# ---------------------------------------------------------------------------
echo "[intelligence-migration] Ensuring omniintelligence database exists..."

DB_EXISTS=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres \
  -tAc "SELECT 1 FROM pg_database WHERE datname = 'omniintelligence'" 2>/dev/null || true)

if [ "$DB_EXISTS" != "1" ]; then
  echo "[intelligence-migration] Creating database omniintelligence..."
  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres \
    -c "CREATE DATABASE omniintelligence OWNER \"${PGUSER}\";"
  echo "[intelligence-migration] Database created."
else
  echo "[intelligence-migration] Database omniintelligence already exists."
fi

# ---------------------------------------------------------------------------
# 2. Create migration tracking table (idempotent)
# ---------------------------------------------------------------------------
echo "[intelligence-migration] Ensuring schema_migrations table exists..."

psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d omniintelligence -c "
CREATE TABLE IF NOT EXISTS schema_migrations (
    id              SERIAL PRIMARY KEY,
    migration_name  VARCHAR(255) NOT NULL UNIQUE,
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum        VARCHAR(64)
);
"

# ---------------------------------------------------------------------------
# 3. Apply pending migrations in sorted order
# ---------------------------------------------------------------------------
echo "[intelligence-migration] Scanning ${MIGRATIONS_DIR} for pending migrations..."

APPLIED=0
SKIPPED=0

for migration_file in $(ls "${MIGRATIONS_DIR}"/*.sql | sort); do
  migration_name=$(basename "$migration_file" .sql)

  # Check if already applied
  already_applied=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d omniintelligence \
    -tAc "SELECT 1 FROM schema_migrations WHERE migration_name = '${migration_name}'" 2>/dev/null || true)

  if [ "$already_applied" = "1" ]; then
    echo "[intelligence-migration]   skip  ${migration_name} (already applied)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "[intelligence-migration]   apply ${migration_name}..."

  # Apply migration then record in tracking table (psql exits non-zero on error)
  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d omniintelligence \
    -v ON_ERROR_STOP=1 -f "$migration_file"

  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d omniintelligence \
    -c "INSERT INTO schema_migrations (migration_name) VALUES ('${migration_name}') ON CONFLICT DO NOTHING;"

  echo "[intelligence-migration]   done  ${migration_name}"
  APPLIED=$((APPLIED + 1))
done

# ---------------------------------------------------------------------------
# 4. Provision omnibase_infra-owned cross-repo tables before fingerprint stamp
# ---------------------------------------------------------------------------
# PluginIntelligence uses omnibase_infra's idempotency store against the
# omniintelligence database. The table must exist before the runtime entrypoint
# stamps expected_schema_fingerprint; otherwise the plugin creates it lazily
# during startup and immediately invalidates the stamped fingerprint.
echo "[intelligence-migration] Ensuring cross-repo idempotency table exists..."

psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d omniintelligence -v ON_ERROR_STOP=1 <<'EOSQL'
CREATE TABLE IF NOT EXISTS idempotency_records (
    id UUID PRIMARY KEY,
    domain VARCHAR(255),
    message_id UUID NOT NULL,
    correlation_id UUID,
    processed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    UNIQUE (domain, message_id)
);

CREATE INDEX IF NOT EXISTS idx_idempotency_records_processed_at
    ON idempotency_records(processed_at);

CREATE INDEX IF NOT EXISTS idx_idempotency_records_domain
    ON idempotency_records(domain);

CREATE INDEX IF NOT EXISTS idx_idempotency_records_correlation_id
    ON idempotency_records(correlation_id)
    WHERE correlation_id IS NOT NULL;
EOSQL

echo "[intelligence-migration] Cross-repo idempotency table ready."
echo "[intelligence-migration] Complete: ${APPLIED} applied, ${SKIPPED} skipped."
