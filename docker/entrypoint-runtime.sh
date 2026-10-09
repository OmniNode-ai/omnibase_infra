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
  # The runtime-owned secret store (OMN-17099): a named volume mounted under the
  # runtime user's home is root-owned when fresh, and the process below runs as
  # omniinfra, so it could neither read nor create the store until repaired.
  install -d -o omniinfra -g omniinfra /home/omniinfra/.omninode /home/omniinfra/.omninode/delegation
  # The lane overlay under .omninode/config is a read-only mount the profile's
  # seed already owns (OMN-19747); chowning it fails on a read-only filesystem.
  find /home/omniinfra/.omninode -path /home/omniinfra/.omninode/config -prune \
    -o -exec chown omniinfra:omniinfra {} +
  exec gosu omniinfra "$0" "$@"
fi

# =============================================================================
# Deployment Identity Banner
# =============================================================================
# Print before any service initialization so operators can immediately verify
# which code is running via: docker logs <container> | head -15
#
# RUNTIME_SOURCE_HASH and COMPOSE_PROJECT are stamped at build time from
# --build-arg values passed by onex-runtime-deploy. They default to "unknown"
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
# Boot Preflight + Kernel Start -- ONE Python interpreter (OMN-17372)
# =============================================================================
# This script used to launch FOUR separate cold Python interpreters before the
# kernel ever started -- the schema-fingerprint stamp, the Bifrost render, the
# secret-resolver render, and then the kernel exec -- each paying a full cold
# `import omnibase_infra`. Measured on the onex-dev runtime container
# (2026-09-06, container start 21:34:27Z): 141.3 s + 51.9 s + 50.2 s + 48.6 s =
# 292.0 s of the 551 s that elapsed before the first subscription was even
# attempted. All four steps are now one warm process.
#
# Everything those steps did lives in
# `omnibase_infra.runtime.entrypoint_preflight`, in the same order, with the
# same env guards, the same 5-attempt/1s stamp retry, the same required
# (omnibase_infra) vs best-effort (omniintelligence) policy (OMN-13666), the
# same OMN-15628 DELEGATION_ROUTING_TIERS_PATH self-heal, and the same exit
# codes. It then starts the CMD: the packaged `onex-runtime` console script is
# called IN-PROCESS so the kernel inherits the warm import; any other CMD is
# `os.execvp`-ed, exactly as `exec "$@"` did.
#
# `exec` here keeps the Python process as the direct child of tini, so signals
# still reach the kernel without a shell in between.

exec python -m omnibase_infra.runtime.entrypoint_preflight "$@"
