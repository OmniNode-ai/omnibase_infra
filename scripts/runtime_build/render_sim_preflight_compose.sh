#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Read-only, exact-project compose renderer for OMN-19728 only.
set -euo pipefail

readonly PROJECT="omnibase-infra-sim-preflight"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly ROOT

if [[ "${COMPOSE_PROJECT_NAME:-}" != "" && "${COMPOSE_PROJECT_NAME}" != "${PROJECT}" ]]; then
  echo "refusing COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME}; expected ${PROJECT}" >&2
  exit 64
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 (renders config only; no lifecycle actions)" >&2
  exit 64
fi

export COMPOSE_PROJECT_NAME="${PROJECT}"
exec docker compose -p "${PROJECT}" \
  -f "${ROOT}/docker/docker-compose.dogfood.yml" \
  -f "${ROOT}/docker/docker-compose.sim-202.yml" \
  -f "${ROOT}/docker/docker-compose.sim-preflight.yml" \
  --profile dogfood config --format json
