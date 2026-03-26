#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
#
# ONEX Infrastructure Runtime Entrypoint
#
# This entrypoint stamps schema fingerprints into db_metadata for ALL
# databases with schema manifests before starting the runtime kernel.
# The fingerprint is computed from the live database schema via the
# installed util_schema_fingerprint module.
# Without the fingerprint stamp, the kernel's startup assertion finds
# expected_schema_fingerprint = NULL and crash-loops.
#
# Environment:
#   OMNIBASE_INFRA_DB_URL    (required) - PostgreSQL DSN for the infra database
#   OMNIINTELLIGENCE_DB_URL  (optional) - PostgreSQL DSN for the intelligence database
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
# OMN-6699: Stamp expected_schema_fingerprint into db_metadata for ALL
# databases with schema manifests, not just omnibase_infra.
#
# Previously only stamped omnibase_infra. After upgrades that add tables to
# omniintelligence (e.g. code_entities, code_relationships), the stored
# fingerprint was stale and the service failed health checks.
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

stamp_fingerprint() {
  # Stamp schema fingerprint for a single database.
  # Usage: stamp_fingerprint <manifest_name> <db_url>
  MANIFEST_NAME="$1"
  DB_URL="$2"

  # Safe log: strip scheme and userinfo, show only host:port/db
  SAFE_DSN=$(echo "${DB_URL}" | sed 's|^[^/]*//[^@]*@||')
  echo "[entrypoint] Stamping schema fingerprint for ${MANIFEST_NAME} (db: ${SAFE_DSN})..."

  STAMP_OK=0
  ATTEMPT=1
  MAX_ATTEMPTS=5

  while [ "${ATTEMPT}" -le "${MAX_ATTEMPTS}" ]; do
    RC=0
    python -m omnibase_infra.runtime.util_schema_fingerprint \
      --manifest "${MANIFEST_NAME}" --db-url "${DB_URL}" stamp || RC=$?
    if [ "${RC}" -eq 0 ]; then
      STAMP_OK=1
      echo "[entrypoint] Schema fingerprint stamped for ${MANIFEST_NAME}."
      break
    fi
    if [ "${RC}" -eq 2 ]; then
      echo "[entrypoint] WARNING: ${MANIFEST_NAME} fingerprint mismatch (exit 2) -- not retrying"
      break
    fi
    echo "[entrypoint] ${MANIFEST_NAME} stamp attempt ${ATTEMPT}/${MAX_ATTEMPTS} failed (exit ${RC})"
    ATTEMPT=$((ATTEMPT + 1))
    if [ "${ATTEMPT}" -le "${MAX_ATTEMPTS}" ]; then
      sleep 1
    fi
  done

  if [ "${STAMP_OK}" -eq 0 ]; then
    echo "[entrypoint] WARNING: ${MANIFEST_NAME} fingerprint stamp did not succeed -- continuing"
  fi
}

# Stamp omnibase_infra (primary, always required)
if [ -n "${OMNIBASE_INFRA_DB_URL:-}" ]; then
  stamp_fingerprint "omnibase_infra" "${OMNIBASE_INFRA_DB_URL}"
else
  echo "[entrypoint] OMNIBASE_INFRA_DB_URL not set -- skipping fingerprint stamp"
fi

# Stamp omniintelligence (optional, activates only when plugin DB is configured)
if [ -n "${OMNIINTELLIGENCE_DB_URL:-}" ]; then
  stamp_fingerprint "omniintelligence" "${OMNIINTELLIGENCE_DB_URL}"
else
  echo "[entrypoint] OMNIINTELLIGENCE_DB_URL not set -- skipping omniintelligence fingerprint stamp"
fi

echo "[entrypoint] Starting runtime kernel..."

exec "$@"
