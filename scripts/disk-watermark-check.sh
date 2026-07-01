#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# disk-watermark-check.sh — Disk-usage watermark alert ratchet (OMN-13008, OMN-13229).
#
# WHY: the 2026-06-11 outage had NO pre-detonation alert — /data silently filled
# until all three lanes crashed. This is the "pages before it detonates" fix.
# OMN-13229 extends this to the Mac Data volume (/System/Volumes/Data) which was
# at 91% on 2026-06-18. `df /` on macOS reports the sealed system snapshot and
# is misleading — always pass --mount /System/Volumes/Data on the Mac.
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
# Publish path (in preference order):
#   1. rpk topic produce (present on .201 Redpanda hosts)
#   2. curl POST to ONEX_BUS_PUBLISH_URL (thin-publish over HTTP; used on the Mac)
#   3. Log-only if neither is available (event is durable in the log for replay)
# ONEX_BUS_PUBLISH_URL must be the full endpoint URL from the contract (Rule 6 / Rule 8):
#   e.g. http://192.168.86.201:3002/api/events   # onex-allow-internal-ip
# Never hardcode LAN IPs in source; the operator exports ONEX_BUS_PUBLISH_URL.
#
# Usage:
#   ./scripts/disk-watermark-check.sh                                       # check /data (.201)
#   ./scripts/disk-watermark-check.sh --mount /System/Volumes/Data          # Mac Data volume
#   ./scripts/disk-watermark-check.sh --warn 85 --crit 90                   # thresholds
#   ./scripts/disk-watermark-check.sh --dry-run                             # print event, no publish
#
# Exit codes: 0 ok (under warn, or published), 10 warn breached, 20 crit breached,
#             2 bad args. (Non-zero breach codes let the timer surface state in
#             `systemctl --user status`.)
#
# Runs on .201 via deploy/disk-gc.timer (shares the GC timer). Log: ~/.local/log/onex/disk-watermark.log
# Runs on Mac via the reaper daemon (T4/OMN-13228) or operator invocation.

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

# Publish to the bus.
#
# Method 1: rpk (present on .201 Redpanda hosts). Broker address from KAFKA_BOOTSTRAP_SERVERS.
# Method 2: curl thin-publish to ONEX_BUS_PUBLISH_URL (Mac or any host without rpk).
# Method 3: log-only if neither is configured (event is durable in the log for replay).
#
# The broker/URL address MUST come from env — fail-fast, no localhost/default fallback
# (Rule 8, OMN-10741). We never hardcode broker addresses or LAN IPs in source.
_published=false
if command -v rpk >/dev/null 2>&1 && [[ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
  BOOTSTRAP="$KAFKA_BOOTSTRAP_SERVERS"
  if echo "$EVENT_JSON" | rpk topic produce "$TOPIC" --brokers "$BOOTSTRAP" >>"$LOG_FILE" 2>&1; then
    log "published $SEVERITY event to $TOPIC via rpk (broker=$BOOTSTRAP)"
    _published=true
  else
    log "FAILED to publish via rpk (broker=$BOOTSTRAP) — falling through to HTTP publish"
  fi
fi

if [[ "$_published" == false ]] && [[ -n "${ONEX_BUS_PUBLISH_URL:-}" ]]; then
  # HTTP thin-publish path (Mac / hosts without rpk).
  # ONEX_BUS_PUBLISH_URL must be the complete endpoint URL from the contract.
  _http_status="$(
    curl -sS -o /dev/null -w "%{http_code}" \
      -X POST "${ONEX_BUS_PUBLISH_URL}" \
      -H "Content-Type: application/json" \
      -d "$EVENT_JSON" 2>>"$LOG_FILE"
  )"
  if [[ "$_http_status" =~ ^2 ]]; then
    log "published $SEVERITY event to $TOPIC via HTTP (url=${ONEX_BUS_PUBLISH_URL} status=${_http_status})"
    _published=true
  else
    log "FAILED to publish via HTTP (url=${ONEX_BUS_PUBLISH_URL} status=${_http_status}) — event logged above for manual replay"
  fi
fi

if [[ "$_published" == false ]]; then
  if [[ -z "${KAFKA_BOOTSTRAP_SERVERS:-}" ]] && [[ -z "${ONEX_BUS_PUBLISH_URL:-}" ]]; then
    log "KAFKA_BOOTSTRAP_SERVERS and ONEX_BUS_PUBLISH_URL both unset — event logged above for manual replay."
  fi
fi

exit "$EXIT_CODE"
