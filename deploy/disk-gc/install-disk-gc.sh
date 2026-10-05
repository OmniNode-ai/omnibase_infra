#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# install-disk-gc.sh — Install the lab disk-maintenance systemd USER timer (OMN-13008).
#
# These are systemd USER units (NOT lane containers). Installing/enabling them is
# scoped to the runtime user and does not touch any docker-compose lane. No sudo.
#
# Usage (run on the lab host after pulling latest):
#   bash deploy/disk-gc/install-disk-gc.sh            # install + enable + start timer
#   bash deploy/disk-gc/install-disk-gc.sh --uninstall
#   bash deploy/disk-gc/install-disk-gc.sh --status
#
# Prerequisites:
#   - systemd user manager available (loginctl enable-linger $USER if running headless)
#   - omnibase_infra cloned at $OMNI_HOME/omnibase_infra (default ~/Code/omni_home)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_SRC="${SCRIPT_DIR}/onex-disk-gc.service"
TIMER_SRC="${SCRIPT_DIR}/onex-disk-gc.timer"
USER_UNIT_DIR="${HOME}/.config/systemd/user"
SERVICE_DST="${USER_UNIT_DIR}/onex-disk-gc.service"
TIMER_DST="${USER_UNIT_DIR}/onex-disk-gc.timer"

if [[ "${1:-}" == "--uninstall" ]]; then
  echo "Uninstalling onex-disk-gc user timer..."
  systemctl --user stop onex-disk-gc.timer onex-disk-gc.service 2>/dev/null || true
  systemctl --user disable onex-disk-gc.timer 2>/dev/null || true
  rm -f "$SERVICE_DST" "$TIMER_DST"
  systemctl --user daemon-reload
  echo "Done. onex-disk-gc uninstalled."
  exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
  systemctl --user list-timers onex-disk-gc.timer --no-pager || true
  echo ""
  systemctl --user status onex-disk-gc.timer onex-disk-gc.service --no-pager || true
  echo ""
  echo "Recent logs:"
  journalctl --user -u onex-disk-gc.service -n 30 --no-pager || true
  exit 0
fi

echo "Installing onex-disk-gc systemd USER timer..."

# OMN-20210: h202 has its registry under /data, without ~/Code/omni_home.
# Validate the selected root and render it before enabling any maintenance.
REGISTRY_ROOT="${OMNI_HOME:-${HOME}/Code/omni_home}"
mkdir -p "$USER_UNIT_DIR"
python3 - "$SERVICE_SRC" "$SERVICE_DST" "$REGISTRY_ROOT" <<'PY'
import sys
from pathlib import Path

source, destination, root = map(Path, sys.argv[1:])
if not root.is_absolute() or any(c in str(root) for c in "\r\n"):
    raise SystemExit("disk-gc registry root must be an absolute, single-line path")
scripts = root / "omnibase_infra" / "scripts"
for name in ("disk-gc.sh", "docker-volume-gc.sh", "disk-watermark-check.sh"):
    if not (scripts / name).is_file():
        raise SystemExit(f"disk-gc script missing from selected registry: {scripts / name}")
# Escape systemd's quoted Environment value and literal percent specifiers.
escaped = str(root).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
destination.write_text(source.read_text().replace("%h/Code/omni_home", escaped))
PY

# Make the GC scripts executable.
chmod +x "${SCRIPT_DIR}/../../scripts/disk-gc.sh" \
         "${SCRIPT_DIR}/../../scripts/docker-volume-gc.sh" \
         "${SCRIPT_DIR}/../../scripts/disk-watermark-check.sh" \
         "${SCRIPT_DIR}/../../scripts/worktree-gc.sh" 2>/dev/null || true

cp "$TIMER_SRC" "$TIMER_DST"

systemctl --user daemon-reload
systemctl --user enable --now onex-disk-gc.timer

echo ""
echo "Done. onex-disk-gc installed and running (user scope)."
echo "  Service: ${SERVICE_DST}"
echo "  Timer:   ${TIMER_DST}"
echo ""
echo "Check:      systemctl --user list-timers onex-disk-gc.timer"
echo "Logs:       journalctl --user -u onex-disk-gc.service -f"
echo "Run now:    systemctl --user start onex-disk-gc.service"
echo "Uninstall:  bash deploy/disk-gc/install-disk-gc.sh --uninstall"
echo ""
echo "NOTE: if this host runs headless, enable lingering so the timer fires"
echo "without an active login session:  loginctl enable-linger \$USER"
