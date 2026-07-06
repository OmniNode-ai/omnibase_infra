#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# install-host-maintenance.sh — install/status wrapper for .201 host maintenance.
#
# This is the single reusable entry point for the disk-GC/worktree-reaper pair.
# It delegates to the existing installers and reports both timers together so
# closeouts can prove cleanup durability without relying on ad hoc root cron.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DISK_GC_INSTALLER="${SCRIPT_DIR}/install-disk-gc.sh"
WORKTREE_REAPER_INSTALLER="${SCRIPT_DIR}/install-worktree-reaper.sh"

usage() {
  cat <<'EOF'
Usage:
  bash deploy/disk-gc/install-host-maintenance.sh
  bash deploy/disk-gc/install-host-maintenance.sh --status
  bash deploy/disk-gc/install-host-maintenance.sh --status --json
  bash deploy/disk-gc/install-host-maintenance.sh --uninstall

Installs or reports both cleanup timers:
  - onex-disk-gc.timer
  - onex-worktree-reaper.timer
EOF
}

status_json() {
  python3 - <<'PY'
import json
import subprocess

UNITS = {
    "disk_gc": {
        "timer": "onex-disk-gc.timer",
        "service": "onex-disk-gc.service",
    },
    "worktree_reaper": {
        "timer": "onex-worktree-reaper.timer",
        "service": "onex-worktree-reaper.service",
    },
}


def systemctl_show(unit: str) -> dict[str, str]:
    fields = {
        "load_state": "unknown",
        "active_state": "unknown",
        "sub_state": "unknown",
        "unit_file_state": "unknown",
        "next_elapse": "",
        "result": "",
    }
    try:
        proc = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=LoadState,ActiveState,SubState,UnitFileState,NextElapseUSecRealtime,Result",
                "--no-pager",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        fields["error"] = "systemctl not found"
        return fields
    if proc.returncode != 0:
        fields["error"] = proc.stderr.strip() or proc.stdout.strip()
        return fields
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized = {
            "LoadState": "load_state",
            "ActiveState": "active_state",
            "SubState": "sub_state",
            "UnitFileState": "unit_file_state",
            "NextElapseUSecRealtime": "next_elapse",
            "Result": "result",
        }.get(key)
        if normalized:
            fields[normalized] = value
    return fields


payload = {
    name: {
        "timer": systemctl_show(units["timer"]),
        "service": systemctl_show(units["service"]),
    }
    for name, units in UNITS.items()
}
print(json.dumps(payload, sort_keys=True))
PY
}

status_human() {
  systemctl --user list-timers onex-disk-gc.timer onex-worktree-reaper.timer --no-pager || true
  echo ""
  systemctl --user status \
    onex-disk-gc.timer onex-disk-gc.service \
    onex-worktree-reaper.timer onex-worktree-reaper.service \
    --no-pager || true
  echo ""
  echo "JSON status:"
  status_json
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--uninstall" ]]; then
  bash "${WORKTREE_REAPER_INSTALLER}" --uninstall
  bash "${DISK_GC_INSTALLER}" --uninstall
  exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
  if [[ "${2:-}" == "--json" ]]; then
    status_json
  else
    status_human
  fi
  exit 0
fi

echo "Installing ONEX host maintenance timers..."
bash "${DISK_GC_INSTALLER}"
bash "${WORKTREE_REAPER_INSTALLER}"
echo ""
echo "Done. Host maintenance timers installed."
echo "Check: bash deploy/disk-gc/install-host-maintenance.sh --status"
echo "JSON:  bash deploy/disk-gc/install-host-maintenance.sh --status --json"
