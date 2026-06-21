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
#   workspace/sibling-pin-comparison.json (expected-vs-actual pin proof)
#
# Sibling-pin preflight (OMN-12977, OMN-13403):
#   Before staging, the consuming repo's uv.lock is the pin authority. The
#   build vendors the canonical OMNI_HOME clones of omnibase_infra / omnibase_core
#   / siblings; if any clone's version mismatches the lock, or the clone is BEHIND
#   the locked SHA (stale), the build ABORTS rather than silently vendoring a
#   stale sibling. This is the recurrence guard for the 2026-06-11 stability crash
#   where a 13-day-stale infra 0.37.0-dev (pre-OMN-12501 guard) was vendored
#   against an omnimarket dev lock pinning infra 0.38.1.
#   OMN-13403 exception: a clone whose version EXACTLY matches the lock and whose
#   HEAD is a strict git DESCENDANT of the locked SHA (clone strictly AHEAD) is
#   recorded as a non-fatal clone-ahead note and does NOT abort -- this is the
#   unavoidable steady state for a real runtime sibling like onex_change_control
#   whose dev branch advances on every receipt PR, so an exact lock pin can never
#   durably converge. Set CONSUMER_LOCK to the consuming repo's uv.lock (default:
#   ${OMNI_HOME}/omnimarket/uv.lock). Set ALLOW_SIBLING_PIN_DRIFT=1 ONLY with an
#   explicit operator decision (e.g. an intentional forward rebuild ahead of a
#   lock bump); the override is recorded in the provenance artifact, never silent.
#
# Exit codes:
#   0  all sibling repos staged successfully
#   1  OMNI_HOME not set
#   2  one or more sibling repos missing from OMNI_HOME
#   3  sibling pin drift from the consuming lock (preflight abort)
set -euo pipefail

# OMN-13405: omnibase_core is staged FIRST so the Dockerfile workspace branch can
# install the dev-HEAD core (which carries enum modules not yet in the released
# 0.45.0 wheel pinned by omnibase_infra/uv.lock, e.g. enum_correction_failure_axis
# added by OMN-13234/OMN-12846). Without this, `uv sync` installs the lock-pinned
# enum-LESS core wheel and omnimarket (installed --no-deps) imports a missing enum,
# crash-looping projection-api + the runtime kernel. Order matters: core must be
# staged/installed before compat/occ/omnimarket so it is the resolved core for all.
SIBLING_REPOS=(
    "omnibase_core"
    "omnibase_compat"
    "onex_change_control"
    "omnimarket"
)

if [[ -z "${OMNI_HOME:-}" ]]; then
    echo "ERROR: OMNI_HOME must be set for workspace-mode build" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Sibling-pin preflight (OMN-12977): the consuming repo's uv.lock is authority.
# ---------------------------------------------------------------------------
CONSUMER_LOCK="${CONSUMER_LOCK:-${OMNI_HOME}/omnimarket/uv.lock}"
PIN_COMPARISON_OUT="workspace/sibling-pin-comparison.json"

# Foundation + sibling packages the build vendors, mapped to OMNI_HOME clone dirs.
PREFLIGHT_REPO_ARGS=(
    --repo "omnibase-infra=${OMNI_HOME}/omnibase_infra"
    --repo "omnibase-core=${OMNI_HOME}/omnibase_core"
    --repo "omnibase-spi=${OMNI_HOME}/omnibase_spi"
    --repo "omnibase-compat=${OMNI_HOME}/omnibase_compat"
    --repo "onex-change-control=${OMNI_HOME}/onex_change_control"
    --repo "omnimarket=${OMNI_HOME}/omnimarket"
)

preflight_extra=()
if [[ "${ALLOW_SIBLING_PIN_DRIFT:-0}" == "1" ]]; then
    preflight_extra+=(--allow-drift)
    echo "WARNING: ALLOW_SIBLING_PIN_DRIFT=1 -- pin drift will be recorded, not fatal" >&2
fi

if [[ -f "${CONSUMER_LOCK}" ]]; then
    mkdir -p "$(dirname "${PIN_COMPARISON_OUT}")"
    if ! python3 "${SCRIPT_DIR}/check_sibling_lock_pins.py" \
        --lock "${CONSUMER_LOCK}" \
        "${PREFLIGHT_REPO_ARGS[@]}" \
        --output "${PIN_COMPARISON_OUT}" \
        "${preflight_extra[@]}"; then
        echo "ERROR: sibling-pin preflight failed against ${CONSUMER_LOCK}" >&2
        echo "       canonical clones drift from the lock; sync clones to the" >&2
        echo "       locked SHAs (or set ALLOW_SIBLING_PIN_DRIFT=1 with an" >&2
        echo "       explicit operator decision) before rebuilding (OMN-12977)." >&2
        exit 3
    fi
else
    echo "ERROR: consuming lock not found: ${CONSUMER_LOCK}" >&2
    echo "       set CONSUMER_LOCK to the consuming repo's uv.lock (OMN-12977)." >&2
    exit 3
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
