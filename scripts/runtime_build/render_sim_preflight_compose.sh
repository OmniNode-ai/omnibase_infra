#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Read-only, exact-project compose renderer for OMN-19728 only.
set -euo pipefail

readonly PROJECT="omnibase-infra-sim-preflight"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly ROOT

if [[ "${COMPOSE_PROJECT_NAME:-}" != "" && "${COMPOSE_PROJECT_NAME}" != "${PROJECT}" ]]; then
  echo "refusing unexpected COMPOSE_PROJECT_NAME" >&2
  exit 64
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 (renders config only; no lifecycle actions)" >&2
  exit 64
fi

export COMPOSE_PROJECT_NAME="${PROJECT}"
cd "${ROOT}"
compose_files=(
  -f "${ROOT}/docker/docker-compose.dogfood.yml"
  -f "${ROOT}/docker/docker-compose.sim-202.yml"
  -f "${ROOT}/docker/docker-compose.sim-preflight.yml"
)
case "${SIM_PREFLIGHT_GRAPH_OVERLAY:-false}" in
  true)
    for name in SIM_PREFLIGHT_GRAPH_RUNTIME_CONFIG_FILE SIM_PREFLIGHT_GRAPH_GATEWAY_KEYMAP_FILE SIM_PREFLIGHT_GRAPH_TERMINAL_PRIVATE_KEY_FILE; do
      if [[ -z "${!name:-}" || "${!name}" != /* || ! -f "${!name}" ]]; then
        echo "${name} must name an existing absolute regular file" >&2
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
if summary="$(docker compose -p "${PROJECT}" \
  --env-file /dev/null \
  "${compose_files[@]}" \
  --profile dogfood config --format json 2>/dev/null | \
  uv run --no-sync --project "${ROOT}" python \
    "${ROOT}/scripts/runtime_build/summarize_sim_preflight_compose.py" 2>/dev/null)"; then
  printf '%s\n' "${summary}"
else
  echo "sim-preflight compose config failed (details suppressed)" >&2
  exit 1
fi
