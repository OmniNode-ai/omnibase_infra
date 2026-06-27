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
# Fresh Volume Bootstrap
# =============================================================================
# Docker named volumes mounted at /app/data or /app/logs hide the image-owned
# directories created during build. Fresh named volumes are commonly root-owned,
# so the non-root runtime user cannot write Bifrost contracts or runtime state
# unless ownership is repaired before dropping privileges.

if [ "$(id -u)" -eq 0 ]; then
  echo "[entrypoint] Bootstrapping runtime volume ownership..."
  install -d -o omniinfra -g omniinfra /app/data /app/data/delegation /app/logs /app/tmp
  chown -R omniinfra:omniinfra /app/data /app/logs /app/tmp
  exec gosu omniinfra "$0" "$@"
fi

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
# OMN-13666: Required vs best-effort stamp policy.
#   - The runtime's OWN database (omnibase_infra) is REQUIRED. Its db_metadata
#     row drives the kernel's startup fingerprint assertion; if the stamp cannot
#     succeed the kernel would start with a NULL/stale fingerprint and crash-loop
#     anyway, so we fail FAST and loud here with a clear cause.
#   - Secondary / non-owned databases (e.g. omniintelligence) are BEST-EFFORT.
#     The runtime DB user legitimately lacks write permission on another
#     service's db_metadata table ("permission denied for table db_metadata").
#     A failure there must NOT take the whole runtime down -- it is logged as a
#     WARNING and boot proceeds. The owning service stamps its own fingerprint.
#
# Retry logic: up to 5 attempts with 1s sleep between failures handles
# transient DB-not-ready conditions at container startup.
#
# Exit code handling from util_schema_fingerprint:
#   0 = success (fingerprint stamped)
#   2 = schema mismatch (no point retrying -- bail immediately)
#   1 = connection or general error (retry)

stamp_fingerprint() {
  # Stamp schema fingerprint for a single database.
  # Usage: stamp_fingerprint <manifest_name> <db_url> <required>
  #   required="required" -> a failed stamp aborts boot (exit 1)
  #   required="optional" -> a failed stamp warns and boot continues
  MANIFEST_NAME="$1"
  DB_URL="$2"
  REQUIRED="$3"

  # Safe log: strip scheme and userinfo, show only host:port/db
  SAFE_DSN=$(echo "${DB_URL}" | sed 's|^[^/]*//[^@]*@||')
  echo "[entrypoint] Stamping schema fingerprint for ${MANIFEST_NAME} (db: ${SAFE_DSN}, ${REQUIRED})..."

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
    if [ "${REQUIRED}" = "required" ]; then
      echo "[entrypoint] ERROR: ${MANIFEST_NAME} (PRIMARY/owned DB) fingerprint stamp failed -- aborting boot" >&2
      exit 1
    fi
    echo "[entrypoint] WARNING: ${MANIFEST_NAME} (secondary/non-owned DB) fingerprint stamp did not succeed -- continuing best-effort"
  fi
}

# Stamp omnibase_infra (PRIMARY/owned DB -- REQUIRED: failure aborts boot)
if [ -n "${OMNIBASE_INFRA_DB_URL:-}" ]; then
  stamp_fingerprint "omnibase_infra" "${OMNIBASE_INFRA_DB_URL}" "required"
else
  echo "[entrypoint] OMNIBASE_INFRA_DB_URL not set -- skipping fingerprint stamp"
fi

# Stamp omniintelligence (SECONDARY/non-owned DB -- BEST-EFFORT: failure warns)
if [ -n "${OMNIINTELLIGENCE_DB_URL:-}" ]; then
  stamp_fingerprint "omniintelligence" "${OMNIINTELLIGENCE_DB_URL}" "optional"
else
  echo "[entrypoint] OMNIINTELLIGENCE_DB_URL not set -- skipping omniintelligence fingerprint stamp"
fi

if [ -n "${BIFROST_CONTRACT_PATH:-}" ]; then
  # OMN-12945: Force re-seed from packaged source on every container restart.
  # BIFROST_FORCE_RESEED=1 bypasses the stale-volume early-return path so the
  # named-volume copy is always rebuilt from the committed packaged source
  # merged with the lane-overlay BIFROST_LOCAL_* endpoint vars. This eliminates
  # the volume-drift defect where a stale /app/data/delegation/bifrost_delegation.yaml
  # silently survived image rebuilds and exposed wrong/missing backends.
  echo "[entrypoint] Rendering Bifrost delegation contract (force-reseed)..."
  BIFROST_FORCE_RESEED=1 python -m omnibase_infra.runtime.render_bifrost_delegation_contract
fi

if [ -n "${ONEX_SECRET_RESOLVER_CONFIG_PATH:-}" ]; then
  echo "[entrypoint] Rendering secret resolver config..."
  python -m omnibase_infra.runtime.render_secret_resolver_config
fi

echo "[entrypoint] Starting runtime kernel..."

exec "$@"
