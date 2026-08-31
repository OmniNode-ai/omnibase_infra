#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# reconcile_deploy_clones.sh -- OMN-17291: the standing reconciler for the
# DEPLOY-SOURCE clones.
#
# The gap this closes: OMN-17190 reconciles the clones on the operator's Mac.
# Nothing reconciled ${OMNI_HOME} on the .201 host -- the tree every lane image
# is actually built from. On 2026-08-31 the dev lane baked an omnimarket 11
# commits behind origin/dev, and omnibase_core could not advance AT ALL because
# its clone carried core.bare=true with a full working tree: `git fetch` exited
# 0 forever while `git checkout` exited 128. The staleness was invisible until
# someone diffed a deployed direct_url against origin/dev by hand.
#
# This wrapper is the .201-runnable entry point. All logic lives in
# deploy_source_ref.py's `reconcile` subcommand (stdlib-only, so it runs on the
# host without the project venv), which fetches, fast-forwards, refuses loudly
# (naming the repo) on a dirty / diverged / bare-with-working-tree clone, and
# VERIFIES HEAD landed on the fetched tip before reporting any repo as synced.
#
# movement-proof-delegated-to: scripts/runtime_build/deploy_source_ref.py
# This wrapper runs no git command of its own, so it has no surface to read
# back. Every fetch and fast-forward happens inside deploy_source_ref.py's
# `reconcile` subcommand, which re-reads HEAD after the operation and compares
# it against the fetched tip it resolved BEFORE the operation -- exit 5, "a
# fetch succeeded but HEAD did not land on the fetched tip", is that comparison
# failing. It is never the exit status of the command that was supposed to move
# the clone. That distinction is the whole OMN-17307 defect class, and it is why
# this declaration is `delegated-to` rather than a claim this file verifies
# anything itself.
#
# The repo set is sourced from sibling_clone_manifest.sh -- the same single
# source of truth ensure_runner_clones.sh and stage_workspace.sh's pin preflight
# read -- so this can never become a third hand-maintained copy of the list
# (the OMN-15137 drift this repo already paid for once).
#
# Usage:
#   OMNI_HOME=/data/omninode/omni_home bash scripts/runtime_build/reconcile_deploy_clones.sh
#
# Env:
#   OMNI_HOME           (required) root holding the deploy-source clones
#   RECONCILE_BRANCH    tracked branch to reconcile onto (default: dev)
#   RECONCILE_RECEIPT   receipt path (default: ${OMNI_HOME}/.deploy-clone-reconcile.json)
#
# Exit codes:
#   0   every manifest clone present under OMNI_HOME is at origin/<branch>
#   3   a clone refused to reconcile (dirty, diverged, or bare-with-working-tree)
#   5   a fetch succeeded but HEAD did not land on the fetched tip
#   64  precondition failure (OMNI_HOME unset/missing, git or python3 absent,
#       or no manifest clone present under OMNI_HOME)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./sibling_clone_manifest.sh
source "${SCRIPT_DIR}/sibling_clone_manifest.sh"

RECONCILE_BRANCH="${RECONCILE_BRANCH:-dev}"

log() { printf '[reconcile-deploy-clones] %s\n' "$*" >&2; }
err() { printf '[reconcile-deploy-clones] ERROR: %s\n' "$*" >&2; }

if [[ -z "${OMNI_HOME:-}" ]]; then
    err "OMNI_HOME must be set (the deploy-source clones live under it)."
    exit 64
fi
if [[ ! -d "${OMNI_HOME}" ]]; then
    err "OMNI_HOME directory does not exist: ${OMNI_HOME}"
    exit 64
fi
command -v git >/dev/null 2>&1 || { err "'git' is required but not found in PATH."; exit 64; }
command -v python3 >/dev/null 2>&1 || { err "'python3' is required but not found in PATH."; exit 64; }

RECONCILE_RECEIPT="${RECONCILE_RECEIPT:-${OMNI_HOME}/.deploy-clone-reconcile.json}"

# A clone the manifest names but that is absent here is reported, not silently
# skipped -- but it is ensure_runner_clones.sh's job to create it, not this
# script's. Absence is surfaced; only PRESENT clones are reconciled.
REPO_ARGS=()
MISSING=()
for repo in "${SIBLING_CLONE_MANIFEST[@]}"; do
    clone="${OMNI_HOME}/${repo}"
    if [[ -e "${clone}/.git" ]]; then
        REPO_ARGS+=(--repo "${repo}=${clone}")
    else
        MISSING+=("${repo}")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    log "not a git clone under ${OMNI_HOME} (skipped; provision with ensure_runner_clones.sh): ${MISSING[*]}"
fi

if [[ ${#REPO_ARGS[@]} -eq 0 ]]; then
    err "no manifest clone found under ${OMNI_HOME} -- nothing to reconcile."
    err "  Expected at least one of: ${SIBLING_CLONE_MANIFEST[*]}"
    exit 64
fi

log "reconciling deploy-source clones under ${OMNI_HOME} onto origin/${RECONCILE_BRANCH}"
exec python3 "${SCRIPT_DIR}/deploy_source_ref.py" reconcile \
    --branch "${RECONCILE_BRANCH}" \
    --output "${RECONCILE_RECEIPT}" \
    "${REPO_ARGS[@]}"
