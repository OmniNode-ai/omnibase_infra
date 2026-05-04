#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Wait for a running Keycloak to become ready, then reconcile its clients
# from desired-clients.json via seed-keycloak-clients.py. Idempotent.
#
# Env vars (all optional):
#   KC_URL              Keycloak base URL (default: http://localhost:28080)
#   KC_REALM            Realm to seed (default: omninode)
#   OMNIBASE_ENV_FILE   Path to env file with KEYCLOAK_ADMIN_* vars
#                       (default: $HOME/.omnibase/.env)
#   KC_READY_TIMEOUT    Max seconds to wait for Keycloak readiness (default: 90)
#   KC_CONTAINER_NAME   Container name for diagnostics on timeout
#                       (default: omnibase-infra-keycloak)
#
# Required in OMNIBASE_ENV_FILE (or already in environment):
#   KEYCLOAK_ADMIN_USERNAME
#   KEYCLOAK_ADMIN_PASSWORD
#
# This script is the canonical entry point for seeding a local Keycloak from
# the omnibase platform installer. The top-level omnibase Makefile delegates
# `make seed-keycloak` here so all Docker/Keycloak knowledge stays in
# omnibase_infra.

set -euo pipefail

KC_URL="${KC_URL:-http://localhost:28080}"
KC_REALM="${KC_REALM:-omninode}"
OMNIBASE_ENV_FILE="${OMNIBASE_ENV_FILE:-${HOME}/.omnibase/.env}"
KC_READY_TIMEOUT="${KC_READY_TIMEOUT:-90}"
KC_CONTAINER_NAME="${KC_CONTAINER_NAME:-omnibase-infra-keycloak}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${OMNIBASE_ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${OMNIBASE_ENV_FILE}"
  set +a
fi

: "${KEYCLOAK_ADMIN_USERNAME:?KEYCLOAK_ADMIN_USERNAME must be set in ${OMNIBASE_ENV_FILE} or environment}"
: "${KEYCLOAK_ADMIN_PASSWORD:?KEYCLOAK_ADMIN_PASSWORD must be set in ${OMNIBASE_ENV_FILE} or environment}"

echo "==> Waiting for Keycloak at ${KC_URL} to become ready (max ${KC_READY_TIMEOUT}s)..."
attempts=$((KC_READY_TIMEOUT / 3))
i=0
until curl -fsS -m 3 "${KC_URL}/realms/${KC_REALM}/.well-known/openid-configuration" > /dev/null 2>&1; do
  i=$((i + 1))
  if [ "${i}" -gt "${attempts}" ]; then
    echo "ERROR: Keycloak at ${KC_URL} did not become ready within ${KC_READY_TIMEOUT}s." >&2
    if command -v docker > /dev/null 2>&1; then
      echo "--- docker ps (${KC_CONTAINER_NAME}) ---" >&2
      docker ps -a --filter "name=${KC_CONTAINER_NAME}" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' >&2 || true
      echo "--- last 40 lines of ${KC_CONTAINER_NAME} logs ---" >&2
      docker logs --tail 40 "${KC_CONTAINER_NAME}" >&2 2>&1 || true
    fi
    exit 1
  fi
  echo "   Keycloak not ready — retrying in 3s... (${i}/${attempts})"
  sleep 3
done

echo "==> Keycloak ready. Running client reconciler..."

# Caller-side gate for --reset-bootstrap-admin: only when targeting localhost.
# The callee (seed-keycloak-clients.py) has its own localhost guard for
# defense-in-depth, but gating here too prevents the flag from ever being
# passed against a remote Keycloak.
reset_flag=()
case "${KC_URL}" in
  http://localhost:* | http://127.0.0.1:*)
    reset_flag=(--reset-bootstrap-admin)
    ;;
esac

cd "${REPO_ROOT}"
exec uv run python scripts/seed-keycloak-clients.py \
  --kc-url "${KC_URL}" \
  --realm "${KC_REALM}" \
  --admin-username "${KEYCLOAK_ADMIN_USERNAME}" \
  --admin-password "${KEYCLOAK_ADMIN_PASSWORD}" \
  ${reset_flag[@]+"${reset_flag[@]}"} \
  --config "${REPO_ROOT}/docker/keycloak/desired-clients.json"
