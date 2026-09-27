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
if summary="$(docker compose -p "${PROJECT}" \
  --env-file /dev/null \
  -f "${ROOT}/docker/docker-compose.dogfood.yml" \
  -f "${ROOT}/docker/docker-compose.sim-202.yml" \
  -f "${ROOT}/docker/docker-compose.sim-preflight.yml" \
  --profile dogfood config --format json 2>/dev/null | \
  uv run --project "${ROOT}" python \
    "${ROOT}/scripts/runtime_build/summarize_sim_preflight_compose.py" 2>/dev/null)"; then
  printf '%s\n' "${summary}"
else
  echo "sim-preflight compose config failed (details suppressed)" >&2
  exit 1
fi
