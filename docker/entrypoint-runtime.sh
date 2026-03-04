#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
#
# ONEX Infrastructure Runtime Entrypoint
#
# This entrypoint runs the migration runner before starting the runtime kernel.
# The runner applies any pending SQL migrations and restamps the schema
# fingerprint into db_metadata (via restamp_fingerprint() inside run-migrations.py).
# Without the fingerprint stamp, the kernel's startup assertion finds
# expected_schema_fingerprint = NULL and crash-loops.
#
# Idempotent: already-applied migrations are skipped; re-stamping an
# already-stamped database safely overwrites the existing fingerprint value.
#
# Environment:
#   OMNIBASE_INFRA_DB_URL  (required) - PostgreSQL DSN for the infra database
#
# Usage (called automatically by Docker ENTRYPOINT):
#   entrypoint-runtime.sh <CMD args...>
#
# The script exec's into "$@" (the CMD) so the kernel process replaces
# the shell and receives signals directly from tini.

set -e

# =============================================================================
# Deployment Identity Banner
# =============================================================================
# Print before any service initialization so operators can immediately verify
# which code is running via: docker logs <container> | head -15
#
# RUNTIME_SOURCE_HASH and COMPOSE_PROJECT are stamped at build time from
# --build-arg values passed by deploy-runtime.sh. They default to "unknown"
# when the image is built without those args (e.g. manual docker compose up).
#
# SOURCE_DIR is the installed package location inside the container.
echo "=== OmniNode Runtime ==="
echo "RUNTIME_SOURCE_HASH=${RUNTIME_SOURCE_HASH:-unknown}"
echo "COMPOSE_PROJECT=${COMPOSE_PROJECT:-unknown}"
echo "SOURCE_DIR=/app/src"
echo "BUILD_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "========================"

if [ -n "${OMNIBASE_INFRA_DB_URL:-}" ]; then
  echo "Running migration runner..."
  python /app/scripts/run-migrations.py --db-url "${OMNIBASE_INFRA_DB_URL}" || {
    echo "WARNING: migration runner failed — continuing to start kernel"
  }
fi

echo "[entrypoint] Starting runtime kernel..."

exec "$@"
