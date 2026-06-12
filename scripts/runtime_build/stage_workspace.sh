#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Stage sibling repos from OMNI_HOME into the Docker build context before
# a workspace-mode build.  Must be called from the repo root (build context).
#
# Usage:
#   OMNI_HOME=/data/omninode/omni_home \
#     bash docker/runtime_build/stage_workspace.sh
#
# On success, creates:
#   workspace/sibling-repos/<repo-name>/  (rsync'd working tree copy)
#
# Exit codes:
#   0  all sibling repos staged successfully
#   1  OMNI_HOME not set
#   2  one or more sibling repos missing from OMNI_HOME
set -euo pipefail

SIBLING_REPOS=(
    "omnibase_compat"
    "onex_change_control"
    "omnimarket"
)

if [[ -z "${OMNI_HOME:-}" ]]; then
    echo "ERROR: OMNI_HOME must be set for workspace-mode build" >&2
    exit 1
fi

STAGING_DIR="workspace/sibling-repos"
mkdir -p "${STAGING_DIR}"

missing=()
for repo in "${SIBLING_REPOS[@]}"; do
    src="${OMNI_HOME}/${repo}"
    if [[ ! -d "${src}" ]]; then
        missing+=("${src}")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: missing sibling repos required for workspace mode:" >&2
    for m in "${missing[@]}"; do
        echo "  ${m}" >&2
    done
    exit 2
fi

for repo in "${SIBLING_REPOS[@]}"; do
    src="${OMNI_HOME}/${repo}"
    dst="${STAGING_DIR}/${repo}"
    echo "staging: ${src} -> ${dst}"
    rsync -a --delete \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.venv' \
        --exclude='*.egg-info' \
        "${src}/" "${dst}/"
    # Record the source HEAD SHA so the lock-pin preflight and provenance can
    # identify exactly which commit was vendored. rsync drops .git, so without
    # this marker the staged tree has no recoverable SHA (OMN-12987).
    if git -C "${src}" rev-parse HEAD >/dev/null 2>&1; then
        git -C "${src}" rev-parse HEAD > "${dst}/.build-sha"
    else
        echo "ERROR: cannot resolve HEAD SHA for ${src}; refusing to stage an unverifiable tree" >&2
        exit 3
    fi
done

echo "workspace staging complete: ${#SIBLING_REPOS[@]} repos staged to ${STAGING_DIR}"
