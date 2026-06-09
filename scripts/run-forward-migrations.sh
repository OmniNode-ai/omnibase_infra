#!/bin/sh
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# run-forward-migrations.sh — Apply omnibase_infra forward migrations on warm Postgres volumes
#
# Tracks applied migrations in public.schema_migrations and applies any
# pending files from /migrations/forward in sorted order. Safe to run on
# both fresh volumes (no-op for files already applied via docker-entrypoint-initdb.d)
# and warm volumes (applies any new files not yet recorded).
#
# This script is run by the forward-migration compose service before migration-gate
# checks the sentinel flag. It replaces the docker-entrypoint-initdb.d-only
# mechanism for keeping warm Postgres volumes up-to-date with new migrations.
#
# Ticket: OMN-4175 (Forward migration runner for warm Postgres volumes)
#
# ---------------------------------------------------------------------------
# Node-owned migration auto-discovery (OMN-12559)
# ---------------------------------------------------------------------------
# omnimarket projection nodes ship SQL under
# src/omnimarket/nodes/<node>/migrations/*.sql. Those files are vendored into
# this repo under ${MIGRATIONS_DIR}/nodes/<node>/ (kept in sync by
# scripts/sync-node-migrations.sh) so a clean clone reproduces the views with
# NO manual copy and NO manual renumber.
#
# Each node migration is tracked under a NAMESPACED migration_id of the form
#   node:<node>:<filename>
# This is a separate identity space from the flat infra sequence
# (tracked as docker/<filename>). Because the namespace is distinct, a node
# migration numbered e.g. 076 NEVER collides with infra's flat 076 file —
# the renumber-as-operational-pattern is eliminated.
#
# Environment:
#   POSTGRES_USER     (default: postgres)
#   POSTGRES_PASSWORD (required)
#   POSTGRES_HOST     (default: postgres)
#   POSTGRES_PORT     (default: 5432)
#   POSTGRES_DB       (default: omnibase_infra)
#   MIGRATIONS_DIR    (default: /migrations/forward)
#   NODE_MIGRATIONS_DIR (default: ${MIGRATIONS_DIR}/nodes)
#   NODE_POSTGRES_DB  (default: POSTGRES_DB; compose sets omnidash_analytics)

set -e

PGUSER="${POSTGRES_USER:-postgres}"
PGHOST="${POSTGRES_HOST:-postgres}"
PGPORT="${POSTGRES_PORT:-5432}"
PGDB="${POSTGRES_DB:-omnibase_infra}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-/migrations/forward}"
NODE_MIGRATIONS_DIR="${NODE_MIGRATIONS_DIR:-${MIGRATIONS_DIR}/nodes}"
NODE_PGDB="${NODE_POSTGRES_DB:-${PGDB}}"

export PGPASSWORD="${POSTGRES_PASSWORD}"

validate_database_identifier() {
  database="$1"
  if ! printf '%s' "$database" | grep -Eq '^[A-Za-z_][A-Za-z0-9_-]*$'; then
    echo "[forward-migration] invalid database identifier in migration directive: ${database}" >&2
    exit 1
  fi
}

ensure_directive_database() {
  migration_file="$1"
  database="$(
    sed -n -E 's/^--[[:space:]]*onex-create-database:[[:space:]]*([A-Za-z_][A-Za-z0-9_-]*)[[:space:]]*$/\1/p' "$migration_file" \
      | head -1
  )"
  if [ -z "$database" ]; then
    return 0
  fi
  validate_database_identifier "$database"
  echo "[forward-migration]   ensure database ${database}..."
  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" -v ON_ERROR_STOP=1 <<-EOSQL
    SELECT 'CREATE DATABASE "$database"'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
EOSQL
}

# ---------------------------------------------------------------------------
# 1. Ensure schema_migrations tracking table exists (idempotent)
# ---------------------------------------------------------------------------
echo "[forward-migration] Ensuring schema_migrations table exists..."

psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" -c "
CREATE TABLE IF NOT EXISTS public.schema_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum     TEXT NOT NULL,
    source_set   TEXT NOT NULL
);
"

# ---------------------------------------------------------------------------
# 2. Apply pending migrations in sorted order
# ---------------------------------------------------------------------------
echo "[forward-migration] Scanning ${MIGRATIONS_DIR} for pending migrations..."

APPLIED=0
SKIPPED=0

for migration_file in $(ls "${MIGRATIONS_DIR}"/*.sql | sort); do
  filename=$(basename "$migration_file")
  migration_id="docker/${filename}"

  # Check if already applied
  already_applied=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
    -tAc "SELECT 1 FROM public.schema_migrations WHERE migration_id = '${migration_id}'" 2>/dev/null || true)

  if [ "$already_applied" = "1" ]; then
    echo "[forward-migration]   skip  ${filename} (already applied)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "[forward-migration]   apply ${filename}..."

  # Apply migration then record in tracking table
  ensure_directive_database "$migration_file"
  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
    -v ON_ERROR_STOP=1 -f "$migration_file"

  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
    -c "INSERT INTO public.schema_migrations (migration_id, checksum, source_set)
        VALUES ('${migration_id}', 'applied-by-runner', 'docker')
        ON CONFLICT (migration_id) DO NOTHING;"

  echo "[forward-migration]   done  ${filename}"
  APPLIED=$((APPLIED + 1))
done

# ---------------------------------------------------------------------------
# 3. Auto-discover and apply node-owned migrations (OMN-12559)
# ---------------------------------------------------------------------------
# Walk ${NODE_MIGRATIONS_DIR}/<node>/*.sql. Within each node directory files
# are applied in sorted (lexical) order. Each file is tracked under the
# namespaced id  node:<node>:<filename>  so the infra numeric sequence is
# never collided with and no renumber is ever required.
NODE_APPLIED=0
NODE_SKIPPED=0

if [ -d "${NODE_MIGRATIONS_DIR}" ]; then
  echo "[forward-migration] Ensuring schema_migrations table exists in node projection database ${NODE_PGDB}..."

  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$NODE_PGDB" -c "
CREATE TABLE IF NOT EXISTS public.schema_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum     TEXT NOT NULL,
    source_set   TEXT NOT NULL
);
"

  echo "[forward-migration] Scanning ${NODE_MIGRATIONS_DIR} for node-owned migrations in ${NODE_PGDB}..."

  # Iterate node directories in sorted order for deterministic application.
  for node_dir in $(ls -d "${NODE_MIGRATIONS_DIR}"/*/ 2>/dev/null | sort); do
    node_name=$(basename "$node_dir")

    # Skip directories with no .sql files.
    if ! ls "${node_dir}"*.sql >/dev/null 2>&1; then
      continue
    fi

    for migration_file in $(ls "${node_dir}"*.sql | sort); do
      filename=$(basename "$migration_file")
      migration_id="node:${node_name}:${filename}"

      already_applied=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$NODE_PGDB" \
        -tAc "SELECT 1 FROM public.schema_migrations WHERE migration_id = '${migration_id}'" 2>/dev/null || true)

      if [ "$already_applied" = "1" ]; then
        echo "[forward-migration]   skip  ${migration_id} (already applied)"
        NODE_SKIPPED=$((NODE_SKIPPED + 1))
        continue
      fi

      echo "[forward-migration]   apply ${migration_id}..."

      psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$NODE_PGDB" \
        -v ON_ERROR_STOP=1 -f "$migration_file"

      psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$NODE_PGDB" \
        -c "INSERT INTO public.schema_migrations (migration_id, checksum, source_set)
            VALUES ('${migration_id}', 'applied-by-runner', 'node')
            ON CONFLICT (migration_id) DO NOTHING;"

      echo "[forward-migration]   done  ${migration_id}"
      NODE_APPLIED=$((NODE_APPLIED + 1))
    done
  done
else
  echo "[forward-migration] No node migrations dir at ${NODE_MIGRATIONS_DIR} — skipping node discovery."
fi

echo "[forward-migration] Complete: ${APPLIED} infra applied, ${SKIPPED} infra skipped; ${NODE_APPLIED} node applied, ${NODE_SKIPPED} node skipped."
