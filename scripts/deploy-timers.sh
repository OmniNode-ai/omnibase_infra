#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# deploy-timers.sh — Install and enable ONEX systemd timer units on .201
#
# Run as: ssh jonah@192.168.86.201 'bash /opt/omninode/scripts/deploy-timers.sh'
#
# Units installed:
#   onex-build-loop.timer   — fires at 02:00, publishes to onex.cmd.build.loop-requested.v1
#   onex-build-loop.service — oneshot service triggered by timer
#   onex-nightly-sweep.timer   — fires at 03:00, publishes to onex.cmd.verification.sweep-requested.v1
#   onex-nightly-sweep.service — oneshot service triggered by timer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_SRC="$SCRIPT_DIR/systemd"
SYSTEMD_DEST="/etc/systemd/system"

UNITS=(
    "onex-build-loop.timer"
    "onex-build-loop.service"
    "onex-nightly-sweep.timer"
    "onex-nightly-sweep.service"
)

echo "==> Copying unit files to $SYSTEMD_DEST"
for unit in "${UNITS[@]}"; do
    src="$SYSTEMD_SRC/$unit"
    if [[ ! -f "$src" ]]; then
        echo "ERROR: source unit not found: $src" >&2
        exit 1
    fi
    sudo cp "$src" "$SYSTEMD_DEST/$unit"
    echo "    copied $unit"
done

echo "==> Reloading systemd daemon"
sudo systemctl daemon-reload

echo "==> Enabling and starting timers"
sudo systemctl enable --now onex-build-loop.timer
sudo systemctl enable --now onex-nightly-sweep.timer

echo "==> Verifying timer state"
systemctl list-timers --no-pager | grep -E "onex-(build-loop|nightly-sweep)" || {
    echo "WARNING: timers not found in list-timers output" >&2
}

echo "==> Done. Active timer status:"
systemctl is-enabled onex-build-loop.timer
systemctl is-enabled onex-nightly-sweep.timer
