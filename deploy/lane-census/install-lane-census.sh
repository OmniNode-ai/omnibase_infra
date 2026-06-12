#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# install-lane-census.sh — Register the lane-census reconcile on the SHARED
# onex-disk-gc.service systemd USER timer (OMN-13011).
#
# COORDINATION (OMN-13008 / PR #1952): the disk-GC PR owns onex-disk-gc.service +
# onex-disk-gc.timer. This installer does NOT create a second timer — it drops a
# systemd override (onex-disk-gc.service.d/20-lane-census.conf) that appends the
# lane-census ExecStart to that one unit, then reloads. The reconcile runs on the
# same hourly tick as the GC/watermark passes.
#
# Prerequisite: OMN-13008's `bash deploy/disk-gc/install-disk-gc.sh` must have run
# first (it installs the base onex-disk-gc.service/.timer). If the base unit is
# absent this installer fails fast and tells you to run the disk-gc installer.
#
# Usage (on 192.168.86.201, user scope, no sudo):
#   bash deploy/lane-census/install-lane-census.sh             # install drop-in + reload
#   bash deploy/lane-census/install-lane-census.sh --uninstall
#   bash deploy/lane-census/install-lane-census.sh --status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DROPIN_SRC="${SCRIPT_DIR}/onex-disk-gc.service.d/20-lane-census.conf"
USER_UNIT_DIR="${HOME}/.config/systemd/user"
BASE_SERVICE="${USER_UNIT_DIR}/onex-disk-gc.service"
DROPIN_DST_DIR="${USER_UNIT_DIR}/onex-disk-gc.service.d"
DROPIN_DST="${DROPIN_DST_DIR}/20-lane-census.conf"

if [[ "${1:-}" == "--uninstall" ]]; then
  echo "Removing lane-census drop-in..."
  rm -f "$DROPIN_DST"
  systemctl --user daemon-reload 2>/dev/null || true
  echo "Done. lane-census drop-in removed (base onex-disk-gc unit untouched)."
  exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
  echo "Effective onex-disk-gc.service (base + drop-ins):"
  systemctl --user cat onex-disk-gc.service --no-pager 2>/dev/null || \
    echo "  (base unit not installed — run deploy/disk-gc/install-disk-gc.sh first)"
  exit 0
fi

if [[ ! -f "$BASE_SERVICE" ]]; then
  echo "ERROR: base onex-disk-gc.service not installed at $BASE_SERVICE" >&2
  echo "Run OMN-13008's installer first:  bash deploy/disk-gc/install-disk-gc.sh" >&2
  exit 1
fi

echo "Installing lane-census drop-in onto the shared onex-disk-gc.service..."
chmod +x "${SCRIPT_DIR}/../../scripts/lane-census-check.sh" 2>/dev/null || true
mkdir -p "$DROPIN_DST_DIR"
cp "$DROPIN_SRC" "$DROPIN_DST"
systemctl --user daemon-reload

echo ""
echo "Done. lane-census reconcile registered on the shared onex-disk-gc timer."
echo "  Drop-in: $DROPIN_DST"
echo ""
echo "Verify:   systemctl --user cat onex-disk-gc.service   # base + lane-census ExecStart"
echo "Run now:  systemctl --user start onex-disk-gc.service"
echo "Logs:     journalctl --user -u onex-disk-gc.service -f"
echo "On-demand: bash scripts/lane-census-check.sh --json"
