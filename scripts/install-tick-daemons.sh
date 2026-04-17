#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# install-tick-daemons.sh — Deploy ONEX tick timers to .201 as systemd --user units.
#
# Usage:
#   bash scripts/install-tick-daemons.sh                   # Deploy to 192.168.86.201
#   bash scripts/install-tick-daemons.sh --host <HOST>     # Deploy to alternate host
#   bash scripts/install-tick-daemons.sh --dry-run         # Show what would be deployed
#
# Prerequisites:
#   - SSH access to TARGET_HOST as jonah (key-based)
#   - systemd --user enabled on TARGET_HOST (loginctl enable-linger jonah)
#
# Installs:
#   onex-tick-health.{service,timer}       — every hour at :03
#   onex-tick-merge-sweep.{service,timer}  — every hour at :23
#   onex-tick-overseer.{service,timer}     — every 15 minutes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_SRC="${SCRIPT_DIR}/systemd"

TARGET_HOST="${TARGET_HOST:-192.168.86.201}"
TARGET_USER="jonah"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)    TARGET_HOST="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *)         echo "Unknown flag: $1"; exit 1 ;;
    esac
done

UNITS=(
    "onex-tick-health.service"
    "onex-tick-health.timer"
    "onex-tick-merge-sweep.service"
    "onex-tick-merge-sweep.timer"
    "onex-tick-overseer.service"
    "onex-tick-overseer.timer"
)

REMOTE_SYSTEMD_DIR=".config/systemd/user"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

# shellcheck disable=SC2029
_ssh() { ssh "${TARGET_USER}@${TARGET_HOST}" "$@"; }
_scp() { scp "$@" "${TARGET_USER}@${TARGET_HOST}:${REMOTE_SYSTEMD_DIR}/"; }

log "=== ONEX Tick Daemon Installer ==="
log "Target: ${TARGET_USER}@${TARGET_HOST}"
log "Units:  ${#UNITS[@]}"
log ""

# Verify source files exist
for unit in "${UNITS[@]}"; do
    src="${SYSTEMD_SRC}/${unit}"
    [[ -f "${src}" ]] || die "Source unit not found: ${src}"
done

if [[ "${DRY_RUN}" == "true" ]]; then
    log "[DRY RUN] Would deploy to ${TARGET_USER}@${TARGET_HOST}:~/${REMOTE_SYSTEMD_DIR}/"
    for unit in "${UNITS[@]}"; do
        log "  + ${unit}"
    done
    log "[DRY RUN] Would run: systemctl --user daemon-reload && systemctl --user enable --now onex-tick-*.timer"
    exit 0
fi

# Ensure remote systemd user dir exists
log "==> Ensuring remote directory ~/.config/systemd/user/ exists"
_ssh "mkdir -p ~/${REMOTE_SYSTEMD_DIR}"

# Copy unit files
log "==> Copying unit files"
for unit in "${UNITS[@]}"; do
    _scp "${SYSTEMD_SRC}/${unit}"
    log "    copied ${unit}"
done

# Reload daemon + enable timers
log "==> Reloading systemd --user daemon"
_ssh "systemctl --user daemon-reload"

log "==> Enabling and starting timers"
_ssh "systemctl --user enable --now onex-tick-health.timer onex-tick-merge-sweep.timer onex-tick-overseer.timer"

# Verify
log "==> Verifying timer state"
_ssh "systemctl --user list-timers --all | grep onex-tick" || {
    log "WARNING: onex-tick timers not found in list-timers output" >&2
}

log ""
log "==> Enabled timers:"
_ssh "systemctl --user is-enabled onex-tick-health.timer onex-tick-merge-sweep.timer onex-tick-overseer.timer" || true

log ""
log "=== Done ==="
log "Monitor with: ssh ${TARGET_USER}@${TARGET_HOST} 'journalctl --user -u onex-tick-* -f'"
log "List timers:  ssh ${TARGET_USER}@${TARGET_HOST} 'systemctl --user list-timers --all | grep onex-tick'"
