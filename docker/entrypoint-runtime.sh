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

if [ -n "${OMNIBASE_INFRA_DB_URL:-}" ]; then
  echo "[entrypoint] Stamping schema fingerprint into db_metadata..."
  python -m omnibase_infra.runtime.util_schema_fingerprint stamp
  echo "[entrypoint] Schema fingerprint stamped successfully."
else
  echo "[entrypoint] OMNIBASE_INFRA_DB_URL not set, skipping schema fingerprint stamp."
fi

echo "[entrypoint] Starting runtime kernel..."

exec "$@"
