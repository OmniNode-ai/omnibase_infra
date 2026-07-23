#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# ensure_runner_clones.sh -- OMN-14900: idempotent provisioning of the deploy
# runner's PRIVATE OMNI_HOME clones.
#
# The omninode-deploy-runner container no longer mounts the shared host
# /data/omninode/omni_home clones (the 2026-07-21 release-train failures were
# all uid-mismatch write attempts into clones the runner does not own:
# dubious-ownership, FETCH_HEAD EACCES, busy-branch worktree contention).
# Instead it gets a PRIVATE OMNI_HOME bind mount (DEPLOY_RUNNER_OMNI_HOME in
# docker/docker-compose.runners.yml) owned by the runner uid. This script
# guarantees the 5 sibling clones exist under that private OMNI_HOME and are
# operable by the CURRENT euid, so provisioning is a committed, automatic step
# inside every entry script -- never a remembered manual step on the host.
#
# Called at the top of the execute path of cut_release_train_tag.sh,
# refresh_stability_lane.sh, and refresh_dev_lane.sh. Safe to re-run: existing
# clones are left untouched (the refresh scripts already fetch; this script
# only guarantees existence + operability).
#
# All 5 repos are public, so cloning needs NO credentials. The base URL is
# overridable (RUNNER_CLONE_BASE_URL) so tests can point it at local file://
# bare fixtures.
#
# Exit codes:
#   0   all clones present and operable by the current euid
#   64  precondition failure (OMNI_HOME unset / not a directory, git missing,
#       clone failed, or a clone is not operable/writable by the current euid)

set -euo pipefail

RUNNER_CLONE_REPOS=(
    "omnibase_infra"
    "omnibase_core"
    "omnibase_compat"
    "onex_change_control"
    "omnimarket"
)

RUNNER_CLONE_BASE_URL="${RUNNER_CLONE_BASE_URL:-https://github.com/OmniNode-ai}"

log() { printf '[ensure-runner-clones] %s\n' "$*" >&2; }
err() { printf '[ensure-runner-clones] ERROR: %s\n' "$*" >&2; }

if [[ -z "${OMNI_HOME:-}" ]]; then
    err "OMNI_HOME must be set (the private runner clones live under it)."
    exit 64
fi

if [[ ! -d "${OMNI_HOME}" ]]; then
    err "OMNI_HOME directory does not exist: ${OMNI_HOME}"
    err "  Inside the deploy runner this is the DEPLOY_RUNNER_OMNI_HOME bind mount"
    err "  created by the compose root-phase init (docker/docker-compose.runners.yml)."
    err "  A missing directory means the bind mount / init did not happen -- refusing"
    err "  to mkdir it here, which would silently write into the container filesystem."
    exit 64
fi

command -v git >/dev/null 2>&1 || { err "'git' is required but not found in PATH."; exit 64; }

for repo in "${RUNNER_CLONE_REPOS[@]}"; do
    clone="${OMNI_HOME}/${repo}"
    if [[ ! -e "${clone}/.git" ]]; then
        log "clone missing: ${clone} -- cloning ${RUNNER_CLONE_BASE_URL}/${repo}.git"
        if ! git -c "safe.directory=${clone}" clone "${RUNNER_CLONE_BASE_URL}/${repo}.git" "${clone}"; then
            err "git clone failed for ${RUNNER_CLONE_BASE_URL}/${repo}.git -> ${clone}"
            exit 64
        fi
    fi
done

# Operability assertion: every clone must be a readable git repo AND writable
# by the CURRENT euid. A clone owned by a different uid is exactly the failure
# mode (mode-1 dubious ownership / mode-2 FETCH_HEAD EACCES) this private
# OMNI_HOME exists to eliminate -- fail loud, never limp into a partial run.
for repo in "${RUNNER_CLONE_REPOS[@]}"; do
    clone="${OMNI_HOME}/${repo}"
    if ! git -c "safe.directory=${clone}" -C "${clone}" rev-parse --git-dir >/dev/null 2>&1; then
        err "clone at ${clone} is not an operable git repository for uid $(id -u)."
        err "  Re-provision the private OMNI_HOME (delete the broken clone and re-run)."
        exit 64
    fi
    if [[ ! -w "${clone}/.git" ]]; then
        err "clone ${clone}/.git is not writable by uid $(id -u) (owner: $(stat -c '%u' "${clone}/.git" 2>/dev/null || stat -f '%u' "${clone}/.git"))."
        err "  The private OMNI_HOME clones must be owned by the uid running this script."
        exit 64
    fi
done

log "all ${#RUNNER_CLONE_REPOS[@]} private clones present and operable under ${OMNI_HOME}"
