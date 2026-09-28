#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Start the disposable sim clone only after its declared shape is verified.
set -euo pipefail

readonly PROJECT="omnibase-infra-sim-preflight"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly ROOT
readonly MIGRATION_PROFILE="${ROOT}/docker/sim-preflight-migration-profile.json"
readonly RUNTIME_SOURCE_PINS="${ROOT}/docker/sim-preflight-runtime-source-pins.json"

require() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is required" >&2
    exit 64
  fi
}

if [[ "$#" -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 64
fi
if [[ "${COMPOSE_PROJECT_NAME:-}" != "" && "${COMPOSE_PROJECT_NAME}" != "${PROJECT}" ]]; then
  echo "refusing unexpected COMPOSE_PROJECT_NAME" >&2
  exit 64
fi
for name in SIM_PREFLIGHT_ENV_FILE SIM_202_RUNTIME_IMAGE SIM_PREFLIGHT_IMAGE_PROVENANCE_PATH; do
  require "${name}"
done
if [[ ! -f "${SIM_PREFLIGHT_ENV_FILE}" ]]; then
  echo "SIM_PREFLIGHT_ENV_FILE must name a private regular file" >&2
  exit 64
fi
if [[ "$(stat -f '%Lp' "${SIM_PREFLIGHT_ENV_FILE}")" != "600" ]]; then
  echo "SIM_PREFLIGHT_ENV_FILE must have mode 600" >&2
  exit 64
fi

compose_files=(
  -f "${ROOT}/docker/docker-compose.dogfood.yml"
  -f "${ROOT}/docker/docker-compose.sim-202.yml"
  -f "${ROOT}/docker/docker-compose.sim-preflight.yml"
)
compose_profiles=(--profile dogfood)
start_services=()
start_receipt="sim-preflight-clone-started"
compose_verifier=("${ROOT}/scripts/runtime_build/summarize_sim_preflight_compose.py")
compose_verifier_args=()
case "${SIM_PREFLIGHT_GRAPH_OVERLAY:-false}" in
  true)
    for name in SIM_PREFLIGHT_GRAPH_RUNTIME_CONFIG_FILE SIM_PREFLIGHT_GRAPH_GATEWAY_KEYMAP_FILE SIM_PREFLIGHT_GRAPH_TERMINAL_PRIVATE_KEY_FILE; do
      require "${name}"
      if [[ "${!name}" != /* || ! -f "${!name}" ]]; then
        echo "${name} must name an existing absolute regular file" >&2
        exit 64
      fi
    done
    for name in SIM_PREFLIGHT_GRAPH_RUNTIME_CONFIG_FILE SIM_PREFLIGHT_GRAPH_TERMINAL_PRIVATE_KEY_FILE; do
      if [[ "$(stat -f '%Lp' "${!name}")" != "600" ]]; then
        echo "${name} must have mode 600" >&2
        exit 64
      fi
    done
    compose_files+=(-f "${ROOT}/docker/docker-compose.sim-preflight-graph.yml")
    ;;
  false) ;;
  *)
    echo "SIM_PREFLIGHT_GRAPH_OVERLAY must be true or false" >&2
    exit 64
    ;;
esac

case "${SIM_PREFLIGHT_ISOLATED_AUTH:-false}" in
  true)
    if [[ "${SIM_PREFLIGHT_GRAPH_OVERLAY:-false}" != true ]]; then
      echo "isolated authentication requires the graph overlay" >&2
      exit 64
    fi
    for name in SIM_PREFLIGHT_GRAPH_MAIN_RUNTIME_CONFIG_FILE SIM_PREFLIGHT_KEYCLOAK_REALM_FILE SIM_PREFLIGHT_GATEWAY_CATALOG_FILE SIM_PREFLIGHT_GATEWAY_SIGNING_PRIVATE_KEY_FILE SIM_PREFLIGHT_GRAPH_TERMINAL_PUBLIC_KEY_FILE; do
      require "${name}"
      if [[ "${!name}" != /* || ! -f "${!name}" || "$(stat -f '%Lp' "${!name}")" != "600" ]]; then
        echo "${name} must name an absolute private mode-600 file" >&2
        exit 64
      fi
    done
    compose_files+=(
      -f "${ROOT}/docker/docker-compose.sim-preflight-auth.yml"
      -f "${ROOT}/docker/docker-compose.sim-preflight-isolated.yml"
    )
    compose_profiles+=(--profile sim-preflight-auth)
    # Prove real-realm authorization before the graph route becomes live.
    # This first stage starts neither Gateway nor either runtime. Their later
    # activation requires the auth proof and restricted-image readback.
    start_services=(keycloak cloud-migration redpanda valkey)
    start_receipt="sim-preflight-auth-infrastructure-started"
    compose_verifier=(-m scripts.runtime_build.verify_sim_preflight_isolation)
    compose_verifier_args=(--private-dir "$(dirname "${SIM_PREFLIGHT_ENV_FILE}")" --check-ports)
    cd "${ROOT}"
    UV_NO_SYNC=1 uv run --project "${ROOT}" python "${compose_verifier[@]}" \
      --main-config "${SIM_PREFLIGHT_GRAPH_MAIN_RUNTIME_CONFIG_FILE}" \
      --effects-config "${SIM_PREFLIGHT_GRAPH_RUNTIME_CONFIG_FILE}" >/dev/null
    ;;
  false) ;;
  *) echo "SIM_PREFLIGHT_ISOLATED_AUTH must be true or false" >&2; exit 64 ;;
esac

export COMPOSE_PROJECT_NAME="${PROJECT}"
cd "${ROOT}"
if ! docker compose -p "${PROJECT}" --env-file "${SIM_PREFLIGHT_ENV_FILE}" \
  "${compose_files[@]}" \
  "${compose_profiles[@]}" config --format json 2>/dev/null |
  UV_NO_SYNC=1 uv run python \
    "${compose_verifier[@]}" "${compose_verifier_args[@]}" \
    >/dev/null 2>&1; then
  echo "sim-preflight compose config validation failed" >&2
  exit 65
fi
UV_NO_SYNC=1 uv run python \
  "${ROOT}/scripts/runtime_build/verify_sim_preflight_migration_profile.py" \
  --profile "${MIGRATION_PROFILE}" \
  --migrations-dir "${ROOT}/docker/migrations/forward" >/dev/null
UV_NO_SYNC=1 uv run python \
  "${ROOT}/scripts/runtime_build/verify_sim_preflight_runtime_source_pins.py" \
  --profile "${RUNTIME_SOURCE_PINS}" >/dev/null
UV_NO_SYNC=1 uv run python \
  "${ROOT}/scripts/runtime_build/verify_sim_preflight_image_provenance.py" \
  --provenance "${SIM_PREFLIGHT_IMAGE_PROVENANCE_PATH}" \
  --image-ref "${SIM_202_RUNTIME_IMAGE}" --source-pins "${RUNTIME_SOURCE_PINS}" >/dev/null

if [[ -n "$(docker compose -p "${PROJECT}" --env-file "${SIM_PREFLIGHT_ENV_FILE}" \
  "${compose_files[@]}" \
  "${compose_profiles[@]}" ps -a -q)" ]]; then
  echo "refusing to restart an existing sim-preflight project" >&2
  exit 65
fi
if [[ -n "$(docker volume ls --filter "label=com.docker.compose.project=${PROJECT}" -q)" ]]; then
  echo "refusing to reuse existing sim-preflight volumes" >&2
  exit 65
fi

docker compose -p "${PROJECT}" --env-file "${SIM_PREFLIGHT_ENV_FILE}" \
  "${compose_files[@]}" \
  "${compose_profiles[@]}" up -d --no-build "${start_services[@]}"
printf '%s\n' "${start_receipt}"
