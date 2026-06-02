#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Build the versioned omnibase_infra self-hosted runner image with its
# repo-specific prebuilt shared CI env baked in (OMN-12567).
#
# The runner image "version" is a binding, not a label. Before building, this
# script verifies that docker/runners/runner-image.lock.json is consistent with
# the live repo inputs (manifest, python, uv, shared-env), then stamps the bound
# identity into the image via build args. A drifted lock fails the build fast.
#
# Usage:
#   scripts/ci/build_runner_image.sh [--tag <image:tag>] [--no-verify]
#
# The actual `docker build`/`docker push`/fleet rollout is a gated live runtime
# step; run it on the runner host. This script assembles the build context and
# (by default, when docker is available) builds locally.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNERS_DIR="${REPO_ROOT}/docker/runners"
LOCK_FILE="${RUNNERS_DIR}/runner-image.lock.json"
CONTEXT_DIR="${RUNNERS_DIR}/prebuilt-env-context"

IMAGE_TAG="omninode-runner:latest"
VERIFY=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) IMAGE_TAG="$2"; shift 2 ;;
    --no-verify) VERIFY=false; shift ;;
    -h|--help)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "[build-runner-image] unknown arg: $1" >&2; exit 2 ;;
  esac
done

python_bin="$(command -v python3)"

if "${VERIFY}"; then
  echo "[build-runner-image] verifying bound identity is current..."
  PYTHONPATH="${REPO_ROOT}/scripts/ci" "${python_bin}" \
    "${SCRIPT_DIR}/runner_image_identity.py" \
    --repo-root "${REPO_ROOT}" --lock-file "${LOCK_FILE}" --mode verify
fi

IMAGE_VERSION="$("${python_bin}" -c \
  "import json,sys; print(json.load(open('${LOCK_FILE}'))['image_version'])")"
IMAGE_IDENTITY="$("${python_bin}" -c \
  "import json,sys; print(json.load(open('${LOCK_FILE}'))['identity_digest'])")"
UV_VERSION="$("${python_bin}" -c \
  "import json,sys; print(json.load(open('${LOCK_FILE}'))['uv_version'])")"

echo "[build-runner-image] staging prebuilt env build context..."
rm -rf "${CONTEXT_DIR}"
mkdir -p "${CONTEXT_DIR}/ci"
mkdir -p "${CONTEXT_DIR}/.github/actions/setup-python-uv"
cp "${REPO_ROOT}/pyproject.toml" "${CONTEXT_DIR}/pyproject.toml"
cp "${REPO_ROOT}/uv.lock" "${CONTEXT_DIR}/uv.lock"
cp "${REPO_ROOT}/.github/actions/setup-python-uv/action.yml" \
  "${CONTEXT_DIR}/.github/actions/setup-python-uv/action.yml"
cp "${REPO_ROOT}/scripts/ci/ci_env_digest.py" "${CONTEXT_DIR}/ci/ci_env_digest.py"
cp "${REPO_ROOT}/scripts/ci/ensure_ci_env.sh" "${CONTEXT_DIR}/ci/ensure_ci_env.sh"

echo "[build-runner-image] runner image v${IMAGE_VERSION} identity ${IMAGE_IDENTITY}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[build-runner-image] docker not available; context staged at ${CONTEXT_DIR}"
  echo "[build-runner-image] run on the runner host:"
  echo "  docker build -t ${IMAGE_TAG} \\"
  echo "    --build-arg UV_VERSION=${UV_VERSION} \\"
  echo "    --build-arg OMNI_RUNNER_IMAGE_VERSION=${IMAGE_VERSION} \\"
  echo "    --build-arg OMNI_RUNNER_IMAGE_IDENTITY=${IMAGE_IDENTITY} \\"
  echo "    ${RUNNERS_DIR}"
  exit 0
fi

docker build -t "${IMAGE_TAG}" \
  --build-arg "UV_VERSION=${UV_VERSION}" \
  --build-arg "OMNI_RUNNER_IMAGE_VERSION=${IMAGE_VERSION}" \
  --build-arg "OMNI_RUNNER_IMAGE_IDENTITY=${IMAGE_IDENTITY}" \
  "${RUNNERS_DIR}"

echo "[build-runner-image] built ${IMAGE_TAG} (runner image v${IMAGE_VERSION})"
