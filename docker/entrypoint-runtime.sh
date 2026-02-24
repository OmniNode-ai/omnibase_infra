#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
#
# ONEX Infrastructure Runtime Entrypoint
#
# This entrypoint stamps the schema fingerprint into db_metadata before
# starting the runtime kernel. The fingerprint is a SHA-256 hash of the
# live database schema (columns + constraints) for tables declared in
# the schema manifest.  Without this stamp, the kernel's startup
# assertion finds expected_schema_fingerprint = NULL and crash-loops.
#
# Idempotent: re-stamping an already-stamped database safely overwrites
# the existing fingerprint value via UPDATE.
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
  echo "[entrypoint] Stamping schema fingerprint into db_metadata..."
  if python -m omnibase_infra.runtime.util_schema_fingerprint stamp; then
    echo "[entrypoint] Schema fingerprint stamped successfully."
  else
    echo "[entrypoint] WARNING: Schema fingerprint stamp failed (exit $?). Continuing startup."
    echo "[entrypoint] The kernel will detect a missing fingerprint via its health check."
  fi
else
  echo "[entrypoint] OMNIBASE_INFRA_DB_URL not set, skipping schema fingerprint stamp."
fi

echo "[entrypoint] Starting runtime kernel..."

exec "$@"
