#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# disk-watermark-check.sh — Disk-usage watermark alert ratchet for .201 (OMN-13008).
#
# WHY: the 2026-06-11 outage had NO pre-detonation alert — /data silently filled
# until all three lanes crashed. This is the "pages before it detonates" fix.
#
# Behavior (df on the target mount):
#   usage >= warn_pct  (default 85): emit a disk-watermark bus event with
#                       severity=warning. A downstream consumer (node_runtime_sweep
#                       auto-ticket path) turns warning events into a Linear ticket.
#   usage >= crit_pct  (default 90): ALSO emit severity=critical — the loud event.
#   usage <  warn_pct : no-op (exit 0, quiet).
#
# The bus is the transport (ONEX doctrine): this script publishes the typed event
# and does not itself talk to Linear. Ticket creation is the consumer's job, which
# keeps a single auto-ticket authority (the sweep) instead of a second one here.
#
# Usage:
#   ./scripts/disk-watermark-check.sh                       # check /data, publish if over
#   ./scripts/disk-watermark-check.sh --mount /            # different mount
#   ./scripts/disk-watermark-check.sh --warn 85 --crit 90  # thresholds
#   ./scripts/disk-watermark-check.sh --dry-run            # print event, do NOT publish
#
# Exit codes: 0 ok (under warn, or published), 10 warn breached, 20 crit breached,
#             2 bad args. (Non-zero breach codes let the timer surface state in
#             `systemctl --user status`.)
#
# Runs on .201 via deploy/disk-gc.timer (shares the GC timer). Log: ~/.local/log/onex/disk-watermark.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOUNT="/data"
WARN_PCT=85
CRIT_PCT=90
DRY_RUN=false
TOPIC="onex.evt.infra.disk-watermark.v1"
LOG_FILE="${HOME}/.local/log/onex/disk-watermark.log"
HOSTNAME_TAG="$(hostname -s 2>/dev/null || echo unknown)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mount) MOUNT="$2"; shift 2 ;;
    --warn) WARN_PCT="$2"; shift 2 ;;
    --crit) CRIT_PCT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [disk-watermark] $*" | tee -a "$LOG_FILE" >&2; }

# Resolve usage percentage for the mount (fall back to / if MOUNT absent).
target="$MOUNT"
df -P "$target" >/dev/null 2>&1 || target="/"
USED_PCT="$(df -P "$target" | awk 'NR==2 {gsub(/%/,"",$5); print $5}')"
AVAIL_KB="$(df -P "$target" | awk 'NR==2 {print $4}')"

if ! [[ "$USED_PCT" =~ ^[0-9]+$ ]]; then
  log "ERROR: could not parse df usage for $target"
  exit 2
fi

log "mount=$target used=${USED_PCT}% avail_kb=${AVAIL_KB} warn=${WARN_PCT} crit=${CRIT_PCT}"

if (( USED_PCT < WARN_PCT )); then
  log "under watermark — quiet"
  exit 0
fi

SEVERITY="warning"
EXIT_CODE=10
if (( USED_PCT >= CRIT_PCT )); then
  SEVERITY="critical"
  EXIT_CODE=20
fi

# Build the typed event payload deterministically in Python so the schema is
# stable and unit-testable.
EVENT_JSON="$(
  USED_PCT="$USED_PCT" AVAIL_KB="$AVAIL_KB" MOUNT="$target" \
  SEVERITY="$SEVERITY" WARN_PCT="$WARN_PCT" CRIT_PCT="$CRIT_PCT" \
  HOSTNAME_TAG="$HOSTNAME_TAG" TOPIC="$TOPIC" \
  python3 "${SCRIPT_DIR}/disk_watermark_event.py"
)"

log "event: $EVENT_JSON"

if [[ "$DRY_RUN" == true ]]; then
  echo "$EVENT_JSON"
  log "DRY-RUN — not publishing (severity=$SEVERITY)"
  exit "$EXIT_CODE"
fi

# Publish to the bus. Prefer rpk (present on .201). The broker address MUST come
# from KAFKA_BOOTSTRAP_SERVERS — fail-fast, no localhost/default fallback (Rule 8,
# OMN-10741). The systemd unit injects it; an operator running by hand must export
# it. We never hardcode a broker address or LAN IP in source.
if [[ -z "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
  log "KAFKA_BOOTSTRAP_SERVERS unset — cannot publish. Event logged above for manual replay."
  exit "$EXIT_CODE"
fi
BOOTSTRAP="$KAFKA_BOOTSTRAP_SERVERS"
if command -v rpk >/dev/null 2>&1; then
  if echo "$EVENT_JSON" | rpk topic produce "$TOPIC" --brokers "$BOOTSTRAP" >>"$LOG_FILE" 2>&1; then
    log "published $SEVERITY event to $TOPIC via rpk"
  else
    log "FAILED to publish via rpk (broker=$BOOTSTRAP) — event logged above for manual replay"
  fi
else
  log "rpk not found — event logged above; install rpk or wire a producer to publish"
fi

exit "$EXIT_CODE"
