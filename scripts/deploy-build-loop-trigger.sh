#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# deploy-build-loop-trigger.sh — sync the canonical onex-build-loop-trigger.sh
# and its systemd user unit/timer to the host that runs the daily trigger.
#
# OMN-15179: the trigger script + systemd units previously existed only as
# unversioned host state at ~/.local/bin/onex-build-loop-trigger.sh and
# ~/.config/systemd/user/onex-build-loop.{service,timer} on omninode-pc --
# nothing synced them from a repo checkout, so a fix landed here never
# reached the host without a manual copy. This script closes that gap the
# same way scripts/deploy-runners.sh closes it for the runner fleet: rsync
# from a local checkout, then reload the host's systemd user manager.
#
# Usage:
#   ./scripts/deploy-build-loop-trigger.sh [--dry-run] [--host HOST]
#
# Requirements:
#   - SSH access to the target host (key-based, no password prompts)
#   - rsync installed locally

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET_HOST="omninode-pc"
DRY_RUN=false

for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        --host)
            shift
            ;;
        *)
            if [[ "${arg}" != "--dry-run" ]]; then
                TARGET_HOST="${arg}"
            fi
            ;;
    esac
done

log() {
    printf '[deploy-build-loop-trigger] %s\n' "$1"
}

SCRIPT_SRC="${REPO_ROOT}/scripts/onex-build-loop-trigger.sh"
SERVICE_SRC="${REPO_ROOT}/scripts/systemd/onex-build-loop.service"
TIMER_SRC="${REPO_ROOT}/scripts/systemd/onex-build-loop.timer"

BIN_DEST="\${HOME}/.local/bin/onex-build-loop-trigger.sh"
UNIT_DEST_DIR="\${HOME}/.config/systemd/user"

if [[ "${DRY_RUN}" == true ]]; then
    log "[DRY RUN] rsync ${SCRIPT_SRC} -> ${TARGET_HOST}:${BIN_DEST}"
    log "[DRY RUN] rsync ${SERVICE_SRC} ${TIMER_SRC} -> ${TARGET_HOST}:${UNIT_DEST_DIR}/"
    log "[DRY RUN] ssh ${TARGET_HOST} systemctl --user daemon-reload"
    exit 0
fi

log "Syncing onex-build-loop-trigger.sh to ${TARGET_HOST}:~/.local/bin/"
ssh "${TARGET_HOST}" 'mkdir -p ~/.local/bin ~/.config/systemd/user'
rsync -av --checksum "${SCRIPT_SRC}" "${TARGET_HOST}:.local/bin/onex-build-loop-trigger.sh"
ssh "${TARGET_HOST}" 'chmod +x ~/.local/bin/onex-build-loop-trigger.sh'

log "Syncing systemd unit + timer to ${TARGET_HOST}:~/.config/systemd/user/"
rsync -av --checksum "${SERVICE_SRC}" "${TIMER_SRC}" "${TARGET_HOST}:.config/systemd/user/"

log "Reloading systemd user manager on ${TARGET_HOST}"
ssh "${TARGET_HOST}" 'systemctl --user daemon-reload'

log "Done. Verify with: ssh ${TARGET_HOST} systemctl --user status onex-build-loop.timer"
