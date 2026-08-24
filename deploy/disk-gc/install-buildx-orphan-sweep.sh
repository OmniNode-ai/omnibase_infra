#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# install-buildx-orphan-sweep.sh — Install the .201 OMN-16406 buildx
# ephemeral-builder orphan-sweep systemd USER timer.
#
# These are systemd USER units (NOT lane containers). Installing/enabling them is
# scoped to the runtime user and does not touch any docker-compose lane. No sudo.
#
# Usage (run on 192.168.86.201 after pulling latest):
#   bash deploy/disk-gc/install-buildx-orphan-sweep.sh            # install + enable + start timer
#   bash deploy/disk-gc/install-buildx-orphan-sweep.sh --uninstall
#   bash deploy/disk-gc/install-buildx-orphan-sweep.sh --status
#
# Prerequisites:
#   - systemd user manager available (loginctl enable-linger $USER if running headless)
#   - omnibase_infra cloned at $OMNI_HOME/omnibase_infra (default ~/Code/omni_home)
#   - docker CLI on PATH with access to the runner fleet's docker socket

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_SRC="${SCRIPT_DIR}/onex-buildx-orphan-sweep.service"
TIMER_SRC="${SCRIPT_DIR}/onex-buildx-orphan-sweep.timer"
USER_UNIT_DIR="${HOME}/.config/systemd/user"
SERVICE_DST="${USER_UNIT_DIR}/onex-buildx-orphan-sweep.service"
TIMER_DST="${USER_UNIT_DIR}/onex-buildx-orphan-sweep.timer"

if [[ "${1:-}" == "--uninstall" ]]; then
  echo "Uninstalling onex-buildx-orphan-sweep user timer..."
  systemctl --user stop onex-buildx-orphan-sweep.timer onex-buildx-orphan-sweep.service 2>/dev/null || true
  systemctl --user disable onex-buildx-orphan-sweep.timer 2>/dev/null || true
  rm -f "$SERVICE_DST" "$TIMER_DST"
  systemctl --user daemon-reload
  echo "Done. onex-buildx-orphan-sweep uninstalled. (Any builder containers not yet swept are left as-is.)"
  exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
  systemctl --user list-timers onex-buildx-orphan-sweep.timer --no-pager || true
  echo ""
  systemctl --user status onex-buildx-orphan-sweep.timer onex-buildx-orphan-sweep.service --no-pager || true
  echo ""
  echo "Recent logs:"
  journalctl --user -u onex-buildx-orphan-sweep.service -n 30 --no-pager || true
  exit 0
fi

echo "Installing onex-buildx-orphan-sweep systemd USER timer..."

chmod +x "${SCRIPT_DIR}/../../scripts/buildx-orphan-sweep.sh" 2>/dev/null || true

mkdir -p "$USER_UNIT_DIR"
cp "$SERVICE_SRC" "$SERVICE_DST"
cp "$TIMER_SRC" "$TIMER_DST"

systemctl --user daemon-reload
systemctl --user enable --now onex-buildx-orphan-sweep.timer

echo "Installed. Verify with:"
echo "  systemctl --user list-timers onex-buildx-orphan-sweep.timer"
echo "  journalctl --user -u onex-buildx-orphan-sweep.service -f"
