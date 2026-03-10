#!/bin/sh
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# check_migrations_complete.sh — Docker healthcheck for migration sentinel
#
# Returns exit 0 if the db_metadata.migrations_complete flag is TRUE,
# exit 1 otherwise. Used by docker-compose depends_on to gate runtime
# service startup until all forward migrations have been applied.
#
# Ticket: OMN-3737 (Boot-Order Migration Sentinel)
#
# Environment:
#   POSTGRES_USER     (default: postgres)
#   POSTGRES_PASSWORD (required)
#   POSTGRES_HOST     (default: localhost)
#   POSTGRES_PORT     (default: 5432)
#   POSTGRES_DB       (default: omnibase_infra)

set -e

PGUSER="${POSTGRES_USER:-postgres}"
PGHOST="${POSTGRES_HOST:-localhost}"
PGPORT="${POSTGRES_PORT:-5432}"
PGDB="${POSTGRES_DB:-omnibase_infra}"

export PGPASSWORD="${POSTGRES_PASSWORD}"

result=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
  -tAc "SELECT migrations_complete FROM public.db_metadata WHERE id = TRUE" 2>/dev/null)

if [ "$result" = "t" ]; then
  exit 0
else
  exit 1
fi
