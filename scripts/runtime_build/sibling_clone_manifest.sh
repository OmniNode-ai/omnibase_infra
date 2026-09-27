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
