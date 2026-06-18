#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# install-worktree-reaper.sh — Install the .201 event-sourced reaper systemd USER
# timer (OMN-13228, T4 of OMN-13008).
#
# These are systemd USER units (NOT lane containers). Installing/enabling them is
# scoped to the runtime user and does not touch any docker-compose lane. No sudo.
#
# This is the *orchestrator's deploy step* — it is NOT run by the worker that ships
# the code PR. The worker ships the unit files; the operator runs this on .201.
#
# Usage (run on 192.168.86.201 after pulling latest):
#   bash deploy/disk-gc/install-worktree-reaper.sh            # install + enable + start timer
#   bash deploy/disk-gc/install-worktree-reaper.sh --uninstall
#   bash deploy/disk-gc/install-worktree-reaper.sh --status
#
# Prerequisites:
#   - systemd user manager available (loginctl enable-linger $USER if running headless)
#   - omnibase_infra cloned at $OMNI_HOME/omnibase_infra (default ~/Code/omni_home)
#   - omniclaude cloned alongside (for prune-worktrees.sh)
#   - ONEX_PROJECTION_URL set in ~/.omnibase/.env (e.g. http://localhost:3002)
#   - python3 on PATH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_SRC="${SCRIPT_DIR}/onex-worktree-reaper.service"
TIMER_SRC="${SCRIPT_DIR}/onex-worktree-reaper.timer"
USER_UNIT_DIR="${HOME}/.config/systemd/user"
SERVICE_DST="${USER_UNIT_DIR}/onex-worktree-reaper.service"
TIMER_DST="${USER_UNIT_DIR}/onex-worktree-reaper.timer"

if [[ "${1:-}" == "--uninstall" ]]; then
  echo "Uninstalling onex-worktree-reaper user timer..."
  systemctl --user stop onex-worktree-reaper.timer onex-worktree-reaper.service 2>/dev/null || true
  systemctl --user disable onex-worktree-reaper.timer 2>/dev/null || true
  rm -f "$SERVICE_DST" "$TIMER_DST"
  systemctl --user daemon-reload
  echo "Done. onex-worktree-reaper uninstalled."
  exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
  systemctl --user list-timers onex-worktree-reaper.timer --no-pager || true
  echo ""
  systemctl --user status onex-worktree-reaper.timer onex-worktree-reaper.service --no-pager || true
  echo ""
  echo "Recent logs:"
  journalctl --user -u onex-worktree-reaper.service -n 30 --no-pager || true
  exit 0
fi

echo "Installing onex-worktree-reaper systemd USER timer..."

# Make the reaper + prune scripts executable (best-effort).
chmod +x "${SCRIPT_DIR}/../../scripts/worktree_reaper.py" 2>/dev/null || true

mkdir -p "$USER_UNIT_DIR"
cp "$SERVICE_SRC" "$SERVICE_DST"
cp "$TIMER_SRC" "$TIMER_DST"

systemctl --user daemon-reload
systemctl --user enable --now onex-worktree-reaper.timer

echo ""
echo "Done. onex-worktree-reaper installed and running (user scope)."
echo "  Service: ${SERVICE_DST}"
echo "  Timer:   ${TIMER_DST}"
echo ""
echo "Check:      systemctl --user list-timers onex-worktree-reaper.timer"
echo "Logs:       journalctl --user -u onex-worktree-reaper.service -f"
echo "Run now:    systemctl --user start onex-worktree-reaper.service"
echo "Uninstall:  bash deploy/disk-gc/install-worktree-reaper.sh --uninstall"
echo ""
echo "NOTE: requires ONEX_PROJECTION_URL in ~/.omnibase/.env (e.g. http://localhost:3002)."
echo "NOTE: if this host runs headless, enable lingering so the timer fires"
echo "without an active login session:  loginctl enable-linger \$USER"
