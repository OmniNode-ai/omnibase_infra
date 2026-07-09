#!/usr/bin/env bash
# devpi-server entrypoint for the OMN-14027 PyPI pull-through cache.
#
# Initialises the server directory on first boot (idempotent), then runs
# devpi-server in the foreground bound to all interfaces so the runner fleet can
# reach the root/pypi pull-through index at http://<host>:3141/root/pypi/+simple/.
set -euo pipefail

SERVERDIR="${DEVPI_SERVERDIR:-/devpi/server}"
HOST="${DEVPI_HOST:-0.0.0.0}"
PORT="${DEVPI_PORT:-3141}"

# devpi-init writes the state marker under $SERVERDIR/.serverversion. Only
# initialise when that marker is absent so container restarts reuse the cache.
if [[ ! -f "${SERVERDIR}/.serverversion" ]]; then
    echo "[pypi-cache] initialising devpi server dir at ${SERVERDIR}"
    devpi-init --serverdir "${SERVERDIR}"
fi

echo "[pypi-cache] starting devpi-server on ${HOST}:${PORT} (serverdir=${SERVERDIR})"
# --restrict-modify root: only the root user may create/modify indexes; the
# fleet is a read-only consumer of the pull-through root/pypi index.
exec devpi-server \
    --host "${HOST}" \
    --port "${PORT}" \
    --serverdir "${SERVERDIR}" \
    --restrict-modify root
