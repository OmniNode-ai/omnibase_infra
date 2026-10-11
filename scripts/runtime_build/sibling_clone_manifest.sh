#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# sibling_clone_manifest.sh -- OMN-15137: single source of truth for every
# sibling repo that must have a live clone under the deploy runner's OMNI_HOME.
#
# Root cause this file closes: ensure_runner_clones.sh (RUNNER_CLONE_REPOS)
# and stage_workspace.sh's sibling-pin preflight (PREFLIGHT_REPO_ARGS, which
# maps 1:1 onto check_sibling_lock_pins.py's DEFAULT_PACKAGE_REPO_DIRS) each
# hardcoded their OWN independent copy of "the sibling repo set" as two
# separately maintained bash arrays. omnibase_spi was added to the preflight's
# set (OMN-12977) but never mirrored into the clone-provisioning set, so
# check_sibling_lock_pins.py referenced a clone directory
# (OMNI_HOME/omnibase_spi) that ensure_runner_clones.sh never created -- a
# silent coverage gap that surfaced only 3 hops deep into the pipeline
# (OMN-15137: "ERROR: cannot resolve clone pin for omnibase-spi: missing
# pyproject.toml"), after two unrelated defects (OMN-15122, OMN-15131) were
# fixed and a run finally reached this step for the first time.
#
# Both consuming scripts source THIS file instead of hardcoding their own
# list, so the two can never drift apart again:
#   - ensure_runner_clones.sh builds RUNNER_CLONE_REPOS from
#     SIBLING_CLONE_MANIFEST.
#   - stage_workspace.sh builds PREFLIGHT_REPO_ARGS from
#     SIBLING_CLONE_MANIFEST + SIBLING_CLONE_MANIFEST_DIST_NAMES.
# tests/scripts/test_sibling_clone_manifest_parity.py additionally asserts
# this file's directory set is IDENTICAL to check_sibling_lock_pins.py's
# DEFAULT_PACKAGE_REPO_DIRS values (the Python side's own canonical mapping),
# so a future 7th sibling added on one side and forgotten on the other fails
# CI immediately instead of failing 3 deploy hops deep on a real runner.
#
# NOT the same list as stage_workspace.sh's SIBLING_REPOS (the narrower
# subset actually vendored as SOURCE into the runtime image via rsync):
# omnibase_infra is the Docker build context itself (never vendored as a
# "sibling"), and omnibase_spi is installed from the published wheel via
# `uv sync` (never staged from a local source tree) -- both still need a
# live OMNI_HOME clone so the pin-preflight can read their pyproject.toml
# version + git HEAD SHA against the consuming lock file.
#
# Arrays below are INDEX-ALIGNED: SIBLING_CLONE_MANIFEST[i] is the OMNI_HOME
# directory name; SIBLING_CLONE_MANIFEST_DIST_NAMES[i] is the corresponding
# uv.lock distribution (package) name for that same repo.
# shellcheck disable=SC2034  # consumed by scripts that `source` this file
SIBLING_CLONE_MANIFEST=(
    "omnibase_infra"
    "omnibase_core"
    "omnibase_spi"
    "omnibase_compat"
    "omnimarket"
)

# shellcheck disable=SC2034  # consumed by scripts that `source` this file
SIBLING_CLONE_MANIFEST_DIST_NAMES=(
    "omnibase-infra"
    "omnibase-core"
    "omnibase-spi"
    "omnibase-compat"
    "omnimarket"
)

# ---------------------------------------------------------------------------
# OMN-19072: every OTHER named sibling set lives here too.
#
# Six scripts in this directory used to keep literal lists of their own. One of
# them, cut-lab-ref.sh, carried a comment saying it mirrored stage_workspace.sh
# plus the build context, which was false in both directions: it added
# onex_change_control and it omitted omnibase_spi, the very repo whose absence
# caused OMN-15137. A reader took that list for the clone set and wrote a
# runbook that failed on first use. The sets below are the only place a sibling
# repo is spelled under scripts/runtime_build/;
# tests/scripts/test_sibling_clone_manifest_parity.py fails on a literal
# sibling array anywhere else in the directory and pins each relation below.
# ---------------------------------------------------------------------------

# Siblings vendored as SOURCE into the runtime image (rsync'd by
# stage_workspace.sh, clean-checked-out to DEPLOY_REF by its RT-1 step, and
# staged by prepr_verify_lane.sh), in install order: omnibase_core FIRST so
# the dev-HEAD core is the resolved core for everything after it (OMN-13405).
# A strict subset of SIBLING_CLONE_MANIFEST: omnibase_infra is the Docker build
# context itself and omnibase_spi is installed from the published wheel. It is
# also the key set of compute_workspace_provenance.py's WORKSPACE_PACKAGES,
# which runs inside the image and cannot source this file; the parity test
# holds the two equal.
# shellcheck disable=SC2034  # consumed by scripts that `source` this file
SIBLING_VENDORED_REPOS=(
    "omnibase_core"
    "omnibase_compat"
    "omnimarket"
)

# Repos the lab lane scripts TRACK although they are not build siblings: the
# lab tag cutters tag them and the lane refresh scripts move them to the
# deployed ref, but nothing vendors them and no pin preflight reads them.
# onex_change_control left the runtime image in OMN-16296, but both tag cutters
# have always tagged it and both refresh scripts refresh it.
# shellcheck disable=SC2034  # consumed by scripts that `source` this file
SIBLING_EXTRA_TRACKED_REPOS=(
    "onex_change_control"
)

# The lab tag set (cut-lab-ref.sh --cut-tag, cut_release_train_tag.sh): every
# clone the pin preflight reads, plus the extras above. Derived, never spelled,
# so a sibling added to the manifest is tagged without anyone remembering to.
# shellcheck disable=SC2034  # consumed by scripts that `source` this file
SIBLING_LAB_TAG_REPOS=(
    "${SIBLING_CLONE_MANIFEST[@]}"
    "${SIBLING_EXTRA_TRACKED_REPOS[@]}"
)

# Tag-set members the lane refresh scripts do NOT move. omnibase_spi is
# installed from the wheel its lock pins, and refresh_dev_lane.sh and
# refresh_stability_lane.sh have never checked its clone out to the deployed
# ref. OMN-19072 records that membership as it was; changing it would change
# what a lane refresh does, which is out of that ticket's scope.
# shellcheck disable=SC2034  # consumed by scripts that `source` this file
SIBLING_LANE_REFRESH_EXCLUDED_REPOS=(
    "omnibase_spi"
)

# The repos refresh_dev_lane.sh and refresh_stability_lane.sh record prior
# HEADs for and refresh: the tag set less the exclusions above.
# shellcheck disable=SC2034  # consumed by scripts that `source` this file
SIBLING_LANE_REFRESH_REPOS=()
for _sibling_manifest_repo in "${SIBLING_LAB_TAG_REPOS[@]}"; do
    _sibling_manifest_skip=false
    for _sibling_manifest_excluded in "${SIBLING_LANE_REFRESH_EXCLUDED_REPOS[@]}"; do
        if [[ "${_sibling_manifest_repo}" == "${_sibling_manifest_excluded}" ]]; then
            _sibling_manifest_skip=true
        fi
    done
    if [[ "${_sibling_manifest_skip}" == false ]]; then
        SIBLING_LANE_REFRESH_REPOS+=("${_sibling_manifest_repo}")
    fi
done
unset _sibling_manifest_repo _sibling_manifest_skip _sibling_manifest_excluded

# OMN-20637: deploy callers can have different registry roots, but RT-1's
# shared worktrees must all belong to one declared clone set. Apply this AFTER
# loading the operator env and restoring caller-owned values. An explicit bad
# declaration refuses; it never silently stages the caller's other clone set.
resolve_deploy_source_clone_root() {
    if [[ "${DEPLOY_SOURCE_CLONE_ROOT+x}" != "x" ]]; then
        return 0
    fi
    if [[ "${DEPLOY_SOURCE_CLONE_ROOT}" != /* || ! -d "${DEPLOY_SOURCE_CLONE_ROOT}" ]]; then
        echo "ERROR: DEPLOY_SOURCE_CLONE_ROOT must name an existing absolute clone-root directory" >&2
        return 64
    fi
    OMNI_HOME="$(cd "${DEPLOY_SOURCE_CLONE_ROOT}" && pwd -P)" || return 64
    export OMNI_HOME
}

# OMN-20687: where `onex-runtime-deploy` lives. It is a console script of the
# omnibase_internal project, not of omnibase_infra and not of the dispatch venv.
# It is installed into that clone's own `.venv` by its `onex-internal-clone-sync`
# timer, and nothing puts it on PATH, so a bare default of `onex-runtime-deploy`
# made every lane refresh exit 64 on a host whose PATH did not carry it. The
# clone lookup is the one omniclaude#2592 (OMN-17427) settled on for
# `onex-host-reconcile`, in the same order, so the two commands cannot be found
# in two different places.
#
# Resolution order; sets DEPLOY_RUNTIME and DEPLOY_RUNTIME_TRIED, always returns
# 0 (each caller keeps its own `command -v` refusal and prints the tried list):
#   1. DEPLOY_RUNTIME, when set: the operator named it, and it is used as given.
#   2. The omnibase_internal clone's `.venv/bin/onex-runtime-deploy`. The clone is
#      OMNIBASE_INTERNAL_HOME when set (and then no other clone is searched), else
#      the sibling of OMNI_HOME as written, then the sibling of its resolved
#      target. A symlinked OMNI_HOME can put a stale second copy beside the link
#      (h201: ~/Code/omnibase_internal beside the link, /data/omninode/
#      omnibase_internal beside the target), so the answer is the first candidate
#      that holds an executable command, not the first directory.
#   3. `onex-runtime-deploy` on PATH.
# Call it after OMNI_HOME is final (the operator env is loaded and
# DEPLOY_SOURCE_CLONE_ROOT applied). When nothing is found DEPLOY_RUNTIME is the
# bare name, so the plan the caller prints still reads.
resolve_runtime_deploy() {
    local -a homes=()
    local home resolved
    DEPLOY_RUNTIME_TRIED=""
    if [[ -n "${DEPLOY_RUNTIME:-}" ]]; then
        DEPLOY_RUNTIME_TRIED="DEPLOY_RUNTIME=${DEPLOY_RUNTIME}"
        return 0
    fi
    if [[ -n "${OMNIBASE_INTERNAL_HOME:-}" ]]; then
        homes+=("${OMNIBASE_INTERNAL_HOME%/}")
    elif [[ -n "${OMNI_HOME:-}" ]]; then
        # Lexical parents, not `/..`: the kernel resolves `link/..` through the
        # symlink, which would make the "as written" candidate the resolved one.
        home="${OMNI_HOME%/}"
        homes+=("${home%/*}/omnibase_internal")
        resolved="$(cd "${OMNI_HOME}" 2>/dev/null && pwd -P)" || resolved=""
        if [[ -n "${resolved}" && "${resolved}" != "${home}" ]]; then
            homes+=("${resolved%/*}/omnibase_internal")
        fi
    fi
    for home in ${homes[@]+"${homes[@]}"}; do
        DEPLOY_RUNTIME_TRIED+="${DEPLOY_RUNTIME_TRIED:+ }${home}/.venv/bin/onex-runtime-deploy"
        if [[ -x "${home}/.venv/bin/onex-runtime-deploy" ]]; then
            DEPLOY_RUNTIME="${home}/.venv/bin/onex-runtime-deploy"
            return 0
        fi
    done
    DEPLOY_RUNTIME_TRIED+="${DEPLOY_RUNTIME_TRIED:+ }PATH"
    DEPLOY_RUNTIME="onex-runtime-deploy"
    return 0
}
