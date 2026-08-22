#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# install-runner-disk-guard.sh — Install the .201 OMN-16363 runner-disk-admission
# restore-guard systemd USER timer.
#
# These are systemd USER units (NOT lane containers). Installing/enabling them is
# scoped to the runtime user and does not touch any docker-compose lane. No sudo.
#
# Usage (run on 192.168.86.201 after pulling latest):
#   bash deploy/disk-gc/install-runner-disk-guard.sh            # install + enable + start timer
#   bash deploy/disk-gc/install-runner-disk-guard.sh --uninstall
#   bash deploy/disk-gc/install-runner-disk-guard.sh --status
#
# Prerequisites:
#   - systemd user manager available (loginctl enable-linger $USER if running headless)
#   - omnibase_infra cloned at $OMNI_HOME/omnibase_infra (default ~/Code/omni_home)
#   - docker CLI on PATH with access to the runner fleet's docker socket

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_SRC="${SCRIPT_DIR}/onex-runner-disk-guard.service"
TIMER_SRC="${SCRIPT_DIR}/onex-runner-disk-guard.timer"
USER_UNIT_DIR="${HOME}/.config/systemd/user"
SERVICE_DST="${USER_UNIT_DIR}/onex-runner-disk-guard.service"
TIMER_DST="${USER_UNIT_DIR}/onex-runner-disk-guard.timer"

if [[ "${1:-}" == "--uninstall" ]]; then
  echo "Uninstalling onex-runner-disk-guard user timer..."
  systemctl --user stop onex-runner-disk-guard.timer onex-runner-disk-guard.service 2>/dev/null || true
  systemctl --user disable onex-runner-disk-guard.timer 2>/dev/null || true
  rm -f "$SERVICE_DST" "$TIMER_DST"
  systemctl --user daemon-reload
  echo "Done. onex-runner-disk-guard uninstalled. (Any currently-paused runners are left as-is; restore them manually via 'docker start <name>' if needed.)"
  exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
  systemctl --user list-timers onex-runner-disk-guard.timer --no-pager || true
  echo ""
  systemctl --user status onex-runner-disk-guard.timer onex-runner-disk-guard.service --no-pager || true
  echo ""
  echo "Recent logs:"
  journalctl --user -u onex-runner-disk-guard.service -n 30 --no-pager || true
  exit 0
fi

echo "Installing onex-runner-disk-guard systemd USER timer..."

chmod +x "${SCRIPT_DIR}/../../scripts/runner-disk-admission-restore.sh" 2>/dev/null || true

mkdir -p "$USER_UNIT_DIR"
cp "$SERVICE_SRC" "$SERVICE_DST"
cp "$TIMER_SRC" "$TIMER_DST"

systemctl --user daemon-reload
systemctl --user enable --now onex-runner-disk-guard.timer

echo "Installed. Verify with:"
echo "  systemctl --user list-timers onex-runner-disk-guard.timer"
echo "  journalctl --user -u onex-runner-disk-guard.service -f"
