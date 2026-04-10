#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# install-infra-watchdog.sh — Install the ONEX infra watchdog systemd timer on .201
#
# Run this script once on 192.168.86.201 after pulling the latest main.
#
# Usage:
#   bash deploy/install-infra-watchdog.sh          # Install and enable
#   bash deploy/install-infra-watchdog.sh --uninstall  # Stop, disable, and remove
#   bash deploy/install-infra-watchdog.sh --status     # Show timer/service status
#
# Prerequisites:
#   - systemd (Linux / .201)
#   - Docker installed and the running user is in the docker group
#   - omnibase_infra cloned at /home/jonah/Code/omni_home/omnibase_infra
#   - Log directory /var/log/onex (created automatically by ExecStartPre in the service)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_SRC="${SCRIPT_DIR}/infra-watchdog.service"
TIMER_SRC="${SCRIPT_DIR}/infra-watchdog.timer"
WATCHDOG_SRC="${SCRIPT_DIR}/../scripts/infra-watchdog.sh"
SERVICE_DST="/etc/systemd/system/infra-watchdog.service"
TIMER_DST="/etc/systemd/system/infra-watchdog.timer"

if [[ "${1:-}" == "--uninstall" ]]; then
    echo "Uninstalling infra-watchdog..."
    sudo systemctl stop infra-watchdog.timer infra-watchdog.service 2>/dev/null || true
    sudo systemctl disable infra-watchdog.timer 2>/dev/null || true
    sudo rm -f "$SERVICE_DST" "$TIMER_DST"
    sudo systemctl daemon-reload
    echo "Done. infra-watchdog uninstalled."
    exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
    systemctl status infra-watchdog.timer infra-watchdog.service || true
    echo ""
    echo "Recent logs:"
    journalctl -u infra-watchdog.service -n 20 --no-pager || true
    exit 0
fi

echo "Installing infra-watchdog systemd timer..."

# Ensure the watchdog script is executable
chmod +x "$WATCHDOG_SRC"

# Copy unit files
sudo cp "$SERVICE_SRC" "$SERVICE_DST"
sudo cp "$TIMER_SRC" "$TIMER_DST"

# Reload and enable the timer
sudo systemctl daemon-reload
sudo systemctl enable --now infra-watchdog.timer

echo ""
echo "Done. infra-watchdog installed and running."
echo "  Service: ${SERVICE_DST}"
echo "  Timer:   ${TIMER_DST}"
echo "  Script:  ${WATCHDOG_SRC}"
echo "  Log:     /var/log/onex/infra-watchdog.log"
echo ""
echo "Check status:     systemctl status infra-watchdog.timer"
echo "View logs:        journalctl -u infra-watchdog.service -f"
echo "Manual trigger:   sudo systemctl start infra-watchdog.service"
echo "Uninstall:        bash deploy/install-infra-watchdog.sh --uninstall"
