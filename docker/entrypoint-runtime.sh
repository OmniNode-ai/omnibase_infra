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
  # OMN-15807: The renderer always rebuilds from the packaged base contract and
  # the mounted typed lane overlay. No endpoint or model environment binding is
  # accepted, so a stale volume or poisoned BIFROST_LOCAL_* value cannot route.
  echo "[entrypoint] Rendering Bifrost delegation contract from typed overlay..."
  python -m omnibase_infra.runtime.render_bifrost_delegation_contract
fi

if [ -n "${ONEX_SECRET_RESOLVER_CONFIG_PATH:-}" ]; then
  echo "[entrypoint] Rendering secret resolver config..."
  python -m omnibase_infra.runtime.render_secret_resolver_config
fi

if [ -n "${DELEGATION_ROUTING_TIERS_PATH:-}" ] && [ ! -f "${DELEGATION_ROUTING_TIERS_PATH}" ]; then
  # OMN-15628 remediation: DELEGATION_ROUTING_TIERS_PATH is pinned in the k8s
  # manifest as a literal string that embeds the venv's Python minor version
  # (e.g. .../python3.12/site-packages/omnimarket/configs/routing_tiers.yaml).
  # A base-image Python version bump silently breaks that pin with no signal
  # until the routing reducer fails closed at first use. Self-heal here by
  # re-deriving the path from the installed omnimarket package's OWN
  # location, which always matches whatever Python is actually running in
  # this image -- never a hardcoded guess. This is a best-effort correction,
  # not a silent-fallback: if re-derivation also fails to find a real file,
  # the original (possibly-stale) pinned value is left untouched and the
  # routing reducer still fails closed with an attributable error, per
  # CLAUDE.md rule 8.
  echo "[entrypoint] WARNING: DELEGATION_ROUTING_TIERS_PATH=${DELEGATION_ROUTING_TIERS_PATH} does not exist -- attempting to re-derive from the installed omnimarket package"
  RESOLVED_TIERS_PATH=$(python -c "import pathlib, omnimarket; print(pathlib.Path(omnimarket.__file__).resolve().parent / 'configs' / 'routing_tiers.yaml')" 2>/dev/null) || RESOLVED_TIERS_PATH=""
  if [ -n "${RESOLVED_TIERS_PATH}" ] && [ -f "${RESOLVED_TIERS_PATH}" ]; then
    echo "[entrypoint] Re-derived DELEGATION_ROUTING_TIERS_PATH=${RESOLVED_TIERS_PATH}"
    export DELEGATION_ROUTING_TIERS_PATH="${RESOLVED_TIERS_PATH}"
  else
    echo "[entrypoint] WARNING: could not re-derive a valid routing_tiers.yaml path -- leaving DELEGATION_ROUTING_TIERS_PATH as pinned; the routing reducer fails closed with an attributable error if it truly does not exist (OMN-15628)"
  fi
fi

echo "[entrypoint] Starting runtime kernel..."

exec "$@"
