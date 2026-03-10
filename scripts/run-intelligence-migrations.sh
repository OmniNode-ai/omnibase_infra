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

set -e

PGUSER="${POSTGRES_USER:-postgres}"
PGHOST="${POSTGRES_HOST:-postgres}"
PGPORT="${POSTGRES_PORT:-5432}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-/migrations/intelligence}"

export PGPASSWORD="${POSTGRES_PASSWORD}"

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

echo "[intelligence-migration] Complete: ${APPLIED} applied, ${SKIPPED} skipped."
