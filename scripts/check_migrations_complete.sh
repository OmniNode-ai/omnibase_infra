#!/bin/sh
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# check_migrations_complete.sh — one-shot readiness probe for the migration sentinel
#
# Returns exit 0 if the db_metadata.migrations_complete flag is TRUE and the
# required projection tables exist, exit 1 otherwise.
#
# OMN-24483: no container runs this any more. The long-running migration-gate
# container that mounted it is gone; the runtime performs the same two checks at
# boot (omnibase_infra.runtime.entrypoint_preflight.wait_for_migrations). The
# script stays as the operator-runnable form of that probe and as the reference
# the fence tests drive.
#
# Ticket: OMN-3737 (Boot-Order Migration Sentinel)
#
# Environment:
#   POSTGRES_USER     (default: postgres)
#   POSTGRES_PASSWORD (required)
#   POSTGRES_HOST     (default: localhost)
#   POSTGRES_PORT     (default: 5432)
#   POSTGRES_DB       (default: omnibase_infra)
#   NODE_POSTGRES_DB  (default: omnidash_analytics)
#   REQUIRED_PROJECTION_TABLES (default: delegation_events node_service_registry)

set -e

PGUSER="${POSTGRES_USER:-postgres}"
PGHOST="${POSTGRES_HOST:?POSTGRES_HOST required}"
PGPORT="${POSTGRES_PORT:-5432}"
PGDB="${POSTGRES_DB:-omnibase_infra}"
NODE_PGDB="${NODE_POSTGRES_DB:-omnidash_analytics}"
REQUIRED_PROJECTION_TABLES="${REQUIRED_PROJECTION_TABLES:-delegation_events node_service_registry}"

export PGPASSWORD="${POSTGRES_PASSWORD}"

result=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
  -tAc "SELECT migrations_complete FROM public.db_metadata WHERE id = TRUE" 2>/dev/null)

if [ "$result" != "t" ]; then
  exit 1
fi

for table_name in $REQUIRED_PROJECTION_TABLES; do
  table_exists=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$NODE_PGDB" \
    -tAc "SELECT to_regclass('public.${table_name}') IS NOT NULL" 2>/dev/null)
  if [ "$table_exists" != "t" ]; then
    exit 1
  fi
done

exit 0
