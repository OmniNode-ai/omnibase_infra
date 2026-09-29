#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# install-lane-census.sh — Register the hourly lane-census pass on a lab host
# (OMN-13011; memory pass and standalone mode OMN-19959).
#
# ONE SCHEDULE PER HOST.
#   * A host that runs onex-disk-gc.service (.201): the census is a systemd
#     override (onex-disk-gc.service.d/20-lane-census.conf) that appends its
#     ExecStart to that one unit (COORDINATION OMN-13008 / PR #1952: no second
#     timer). This is the default mode, and it fails fast when the base unit is
#     absent.
#   * A host with NO onex-disk-gc.service (.202, omnipc2): `--standalone`
#     installs onex-lane-census.service + .timer instead. It REFUSES when the
#     base unit is present, so a host can never carry both schedules. Installing
#     the disk-GC unit on such a host is not an option: its GC passes run with
#     --execute from a path that host does not have and fail the unit before any
#     drop-in runs.
#
# THE MEMORY PASS (OMN-19959). Both modes run the census with --memory, which
# publishes one lane container memory event per pass through the host's
# dev-lane broker container. That container is named here with
# --broker-container and written into the installed unit as
# LANE_MEMORY_BROKER_CONTAINER. There is no default: .201's is
# omnibase-infra-redpanda and .202's is omnibase-infra-dev-202-redpanda, and a
# guessed name would publish nowhere while looking installed.
#
# THE CLONE. The shipped units name %h/Code/omni_home/omnibase_infra, which is
# the clone on .201 but not on .202 (/data/omninode/omnibase_infra there). The
# installer rewrites that path to the clone it is run from, or to --repo-root.
#
# Usage (user scope, no sudo):
#   bash deploy/lane-census/install-lane-census.sh --broker-container omnibase-infra-redpanda
#   bash deploy/lane-census/install-lane-census.sh --standalone --broker-container omnibase-infra-dev-202-redpanda
#   bash deploy/lane-census/install-lane-census.sh ... --repo-root /data/omninode/omnibase_infra
#   bash deploy/lane-census/install-lane-census.sh --uninstall
#   bash deploy/lane-census/install-lane-census.sh --status   # print effective unit
#   bash deploy/lane-census/install-lane-census.sh --verify   # exit 1 if not installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DROPIN_SRC="${SCRIPT_DIR}/onex-disk-gc.service.d/20-lane-census.conf"
STANDALONE_SERVICE_SRC="${SCRIPT_DIR}/standalone/onex-lane-census.service"
STANDALONE_TIMER_SRC="${SCRIPT_DIR}/standalone/onex-lane-census.timer"
USER_UNIT_DIR="${HOME}/.config/systemd/user"
BASE_SERVICE="${USER_UNIT_DIR}/onex-disk-gc.service"
DROPIN_DST_DIR="${USER_UNIT_DIR}/onex-disk-gc.service.d"
DROPIN_DST="${DROPIN_DST_DIR}/20-lane-census.conf"
STANDALONE_SERVICE_DST="${USER_UNIT_DIR}/onex-lane-census.service"
STANDALONE_TIMER_DST="${USER_UNIT_DIR}/onex-lane-census.timer"
SHIPPED_REPO_ROOT='%h/Code/omni_home/omnibase_infra'

MODE=""
STANDALONE=false
BROKER_CONTAINER=""
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uninstall|--verify|--status) MODE="$1"; shift ;;
    --standalone) STANDALONE=true; shift ;;
    --broker-container)
      [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: --broker-container requires a container name" >&2; exit 2; }
      BROKER_CONTAINER="$2"; shift 2 ;;
    --repo-root)
      [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: --repo-root requires a path" >&2; exit 2; }
      REPO_ROOT="$2"; shift 2 ;;
    --help|-h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$MODE" == "--uninstall" ]]; then
  echo "Removing lane-census drop-in and standalone units..."
  systemctl --user disable --now onex-lane-census.timer 2>/dev/null || true
  rm -f "$DROPIN_DST" "$STANDALONE_SERVICE_DST" "$STANDALONE_TIMER_DST"
  systemctl --user daemon-reload 2>/dev/null || true
  echo "Done. lane-census removed (base onex-disk-gc unit untouched)."
  exit 0
fi

if [[ "$MODE" == "--verify" ]]; then
  # OMN-18606 AC5. `--status` PRINTS the effective unit; a reader has to notice
  # the census ExecStart is absent. That is how D1 survived: the drop-in was
  # never installed on .201, `systemctl --user status onex-disk-gc.service`
  # reported a clean SUCCESS every hour with no census pass in the unit at all,
  # and nothing anywhere distinguished that from a healthy install. --verify
  # makes the same fact an exit code, so a probe or a tick can assert it.
  rc=0
  unit=""
  if [[ -f "$DROPIN_DST" && -f "$STANDALONE_SERVICE_DST" ]]; then
    echo "DOUBLE INSTALL: both $DROPIN_DST and $STANDALONE_SERVICE_DST exist — one schedule per host" >&2
    rc=1
  elif [[ -f "$DROPIN_DST" ]]; then
    unit="onex-disk-gc.service"
  elif [[ -f "$STANDALONE_SERVICE_DST" ]]; then
    unit="onex-lane-census.service"
    if ! systemctl --user is-enabled onex-lane-census.timer >/dev/null 2>&1; then
      echo "NOT SCHEDULED: onex-lane-census.timer is not enabled" >&2
      rc=1
    fi
  else
    echo "NOT INSTALLED: neither $DROPIN_DST nor $STANDALONE_SERVICE_DST exists — the hourly census pass is not registered" >&2
    rc=1
  fi
  if [[ -n "$unit" ]]; then
    effective="$(systemctl --user cat "$unit" --no-pager 2>/dev/null || true)"
    if ! grep -q -- "--snapshot" <<<"$effective"; then
      echo "STALE INSTALL: the effective $unit carries no --snapshot census ExecStart" >&2
      echo "  re-run: bash deploy/lane-census/install-lane-census.sh" >&2
      rc=1
    fi
    if ! grep -q -- "--memory" <<<"$effective"; then
      echo "STALE INSTALL: the effective $unit runs no --memory pass (OMN-19959)" >&2
      rc=1
    fi
    if ! grep -q "LANE_MEMORY_BROKER_CONTAINER=." <<<"$effective"; then
      echo "NO BROKER: the effective $unit names no LANE_MEMORY_BROKER_CONTAINER; every memory pass exits 6" >&2
      rc=1
    fi
  fi
  if [[ $rc -eq 0 ]]; then
    echo "OK: lane-census pass is registered on $unit, writes a snapshot and publishes the memory event"
  fi
  exit $rc
fi

if [[ "$MODE" == "--status" ]]; then
  for unit in onex-disk-gc.service onex-lane-census.service; do
    echo "Effective ${unit}:"
    systemctl --user cat "$unit" --no-pager 2>/dev/null || echo "  (not installed)"
  done
  exit 0
fi

if [[ -z "$BROKER_CONTAINER" ]]; then
  echo "ERROR: --broker-container <name> is required: the dev-lane broker container on this host" >&2
  echo "  (.201: omnibase-infra-redpanda; .202: omnibase-infra-dev-202-redpanda)" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/scripts/lane-census-check.sh" ]]; then
  echo "ERROR: ${REPO_ROOT}/scripts/lane-census-check.sh not found; pass --repo-root <omnibase_infra clone>" >&2
  exit 2
fi

# Render a shipped unit: the clone path, then the broker container as the last
# [Service] setting (every shipped unit ends inside its [Service] section or
# carries [Install] after it, so the line is inserted before [Install] if any).
render_unit() {
  local src="$1" dst="$2"
  local env_line="Environment=LANE_MEMORY_BROKER_CONTAINER=${BROKER_CONTAINER}"
  sed "s#${SHIPPED_REPO_ROOT}#${REPO_ROOT}#g" "$src" | awk -v line="$env_line" '
    /^\[Install\]/ && !done { print line; print ""; done=1 }
    { print }
    END { if (!done) print line }
  ' >"$dst"
}

mkdir -p "$USER_UNIT_DIR"
chmod +x "${REPO_ROOT}/scripts/lane-census-check.sh" 2>/dev/null || true

if [[ "$STANDALONE" == true ]]; then
  if [[ -f "$BASE_SERVICE" ]]; then
    echo "ERROR: $BASE_SERVICE exists; this host's census rides on it. Install without --standalone." >&2
    exit 1
  fi
  echo "Installing standalone onex-lane-census.service + .timer (no onex-disk-gc on this host)..."
  render_unit "$STANDALONE_SERVICE_SRC" "$STANDALONE_SERVICE_DST"
  cp "$STANDALONE_TIMER_SRC" "$STANDALONE_TIMER_DST"
  rm -f "$DROPIN_DST"
  systemctl --user daemon-reload
  systemctl --user enable --now onex-lane-census.timer
  echo ""
  echo "Done. Unit: $STANDALONE_SERVICE_DST (clone ${REPO_ROOT}, broker ${BROKER_CONTAINER})"
  echo "Run now:  systemctl --user start onex-lane-census.service"
  echo "Logs:     journalctl --user -u onex-lane-census.service -f"
  exit 0
fi

if [[ ! -f "$BASE_SERVICE" ]]; then
  echo "ERROR: base onex-disk-gc.service not installed at $BASE_SERVICE" >&2
  echo "Run OMN-13008's installer first:  bash deploy/disk-gc/install-disk-gc.sh" >&2
  echo "or, on a host that runs no disk GC, install with --standalone." >&2
  exit 1
fi

echo "Installing lane-census drop-in onto the shared onex-disk-gc.service..."
mkdir -p "$DROPIN_DST_DIR"
render_unit "$DROPIN_SRC" "$DROPIN_DST"
systemctl --user disable --now onex-lane-census.timer 2>/dev/null || true
rm -f "$STANDALONE_SERVICE_DST" "$STANDALONE_TIMER_DST"
systemctl --user daemon-reload

echo ""
echo "Done. lane-census reconcile registered on the shared onex-disk-gc timer."
echo "  Drop-in: $DROPIN_DST (clone ${REPO_ROOT}, broker ${BROKER_CONTAINER})"
echo ""
echo "Verify:   bash deploy/lane-census/install-lane-census.sh --verify"
echo "Run now:  systemctl --user start onex-disk-gc.service"
echo "Logs:     journalctl --user -u onex-disk-gc.service -f"
echo "On-demand: bash scripts/lane-census-check.sh --json"
