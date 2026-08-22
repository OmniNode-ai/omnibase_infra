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
