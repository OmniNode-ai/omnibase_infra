#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Validate a self-hosted runner against the OMN-12567 versioned image contract
# (OMN-12568 acceptance). Run this on a *freshly recreated* runner — the point
# is to prove the image contract survives recreation, not warm-runner state.
#
# This is a READ-ONLY assertion. It does not recreate, restart, register, or
# mutate any runner; it only inspects the runner it runs on and exits non-zero
# if the versioned-image contract is not satisfied.
#
# What it asserts (via scripts/ci/validate_runner_image.py):
#   1. image identity      — baked OMNI_RUNNER_IMAGE_IDENTITY matches the lock's
#                            recorded identity_digest (no drift, not "unbound").
#   2. zero `uv sync`       — the prebuilt shared CI env is baked + UV_NO_SYNC=1,
#                            so the happy path resolves zero `uv sync`.
#   3. Receipt-Gate ready   — gh/git/jq/python3/uv are present on the runner.
#
# Usage (on the runner host or inside the runner container):
#   scripts/ci/validate_runner_image.sh                 # human report
#   scripts/ci/validate_runner_image.sh --json          # machine-readable
#   OMNI_RUNNER_CONTAINER=omninode-runner-1 \
#     scripts/ci/validate_runner_image.sh               # inspect a named container
#
# When OMNI_RUNNER_CONTAINER is set and docker is available, the script reads the
# baked identity from the container's image label instead of the ambient env, so
# it can validate a runner container from the host without entering it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VALIDATOR="${SCRIPT_DIR}/validate_runner_image.py"

python_bin="$(command -v python3 || true)"
if [[ -z "${python_bin}" ]]; then
  echo "::error::python3 not found on runner; cannot validate image contract" >&2
  exit 1
fi

# Resolve the baked image identity. Priority:
#   1. OMNI_RUNNER_IMAGE_IDENTITY already in the env (inside the runner image).
#   2. The image label of a named container (OMNI_RUNNER_CONTAINER) via docker.
baked_identity="${OMNI_RUNNER_IMAGE_IDENTITY:-}"
if [[ -z "${baked_identity}" && -n "${OMNI_RUNNER_CONTAINER:-}" ]] \
  && command -v docker >/dev/null 2>&1; then
  image_ref="$(docker inspect --format '{{.Config.Image}}' "${OMNI_RUNNER_CONTAINER}" 2>/dev/null || true)"
  if [[ -n "${image_ref}" ]]; then
    baked_identity="$(
      docker inspect --format \
        '{{ index .Config.Labels "org.omninode.runner.image.identity" }}' \
        "${image_ref}" 2>/dev/null || true
    )"
  fi
fi
export OMNI_RUNNER_IMAGE_IDENTITY="${baked_identity}"

echo "[validate-runner-image] OMN-12568 acceptance check on $(hostname)"
echo "[validate-runner-image] baked identity: ${baked_identity:-<unset>}"
echo "[validate-runner-image] env root: ${OMNI_CI_ENV_ROOT:-/home/runner/.cache/omni/ci-envs}"

exec "${python_bin}" "${VALIDATOR}" \
  --repo-root "${REPO_ROOT}" \
  ${OMNI_CI_ENV_ROOT:+--env-root "${OMNI_CI_ENV_ROOT}"} \
  "$@"
