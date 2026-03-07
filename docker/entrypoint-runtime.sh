#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
#
# ONEX Infrastructure Runtime Entrypoint
#
# This entrypoint stamps the schema fingerprint into db_metadata before
# starting the runtime kernel. The fingerprint is computed from the live
# database schema via the installed util_schema_fingerprint module.
# Without the fingerprint stamp, the kernel's startup assertion finds
# expected_schema_fingerprint = NULL and crash-loops.
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

# =============================================================================
# Schema Fingerprint Stamp
# =============================================================================
# Stamp expected_schema_fingerprint into db_metadata so the kernel's startup
# assertion can compare live schema against a known-good fingerprint.
#
# Retry logic: up to 5 attempts with 1s sleep between failures handles
# transient DB-not-ready conditions at container startup.
#
# Exit code handling from util_schema_fingerprint:
#   0 = success (fingerprint stamped)
#   2 = schema mismatch (no point retrying -- bail immediately)
#   1 = connection or general error (retry)
#
# Fail-open: kernel starts regardless of stamp outcome. The kernel's own
# fingerprint assertion will catch any real problems.

if [ -n "${OMNIBASE_INFRA_DB_URL:-}" ]; then
  # Safe log: strip scheme and userinfo, show only host:port/db
  SAFE_DSN=$(echo "${OMNIBASE_INFRA_DB_URL}" | sed 's|^[^/]*//[^@]*@||')
  echo "[entrypoint] Stamping schema fingerprint (db: ${SAFE_DSN})..."

  STAMP_OK=0
  ATTEMPT=1
  MAX_ATTEMPTS=5

  while [ "${ATTEMPT}" -le "${MAX_ATTEMPTS}" ]; do
    RC=0
    python -m omnibase_infra.runtime.util_schema_fingerprint stamp || RC=$?
    if [ "${RC}" -eq 0 ]; then
      STAMP_OK=1
      break
    fi
    if [ "${RC}" -eq 2 ]; then
      echo "[entrypoint] WARNING: fingerprint mismatch (exit 2) -- not retrying"
      break
    fi
    echo "[entrypoint] Stamp attempt ${ATTEMPT}/${MAX_ATTEMPTS} failed (exit ${RC})"
    ATTEMPT=$((ATTEMPT + 1))
    if [ "${ATTEMPT}" -le "${MAX_ATTEMPTS}" ]; then
      sleep 1
    fi
  done

  if [ "${STAMP_OK}" -eq 0 ]; then
    echo "[entrypoint] WARNING: fingerprint stamp did not succeed -- continuing to start kernel"
  fi
else
  echo "[entrypoint] OMNIBASE_INFRA_DB_URL not set -- skipping fingerprint stamp"
fi

echo "[entrypoint] Starting runtime kernel..."

exec "$@"
