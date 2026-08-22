#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# runner-disk-admission-restore.sh — Restore runners paused by the OMN-16363
# disk-admission gate (docker/runners/runner-job-started.sh) once /data has
# recovered, using the "slope-plus-canary" criterion documented on OMN-16363:
# a fixed absolute free-space number is not a reliable restart signal (a
# 150GB/10min default set mid-incident restored too conservatively in
# practice); the criterion that actually worked live was sustained POSITIVE
# SLOPE across a canary batch — restore a small batch, confirm free space
# keeps climbing (not just holding flat) for a consecutive check with that
# batch back in rotation, then proceed to the next batch. A batch that turns
# the slope negative again is the stop signal, not a restart-from-zero signal.
#
# This mechanizes the manual incident-response playbook
# (runner-fleet-diag-1's 2026-08-21/2026-08-22 responses, documented on
# OMN-16363) into a scheduled, idempotent, no-argument-required pass driven by
# deploy/disk-gc/onex-runner-disk-guard.timer.
#
# WHAT IT DOES NOT DO. It never stops a runner — that is the job-started
# hook's disk_admission_self_pause() (OMN-16363), which pauses a runner from
# INSIDE its own container via the bind-mounted docker socket, the moment that
# runner's own consecutive-admission-failure count crosses the backoff
# threshold. This script only WATCHES the paused-marker directory and restarts
# from it — it never inspects a runner's health or decides to pause one.
#
# STATE. A per-host JSON state file tracks the free-space reading from the
# PREVIOUS tick (to compute slope) and how many consecutive climbing ticks
# above the restore floor have been observed. Reused across ticks by the
# systemd timer; safe to delete (a restart of the observation window, not of
# any runner).
#
# Usage:
#   ./scripts/runner-disk-admission-restore.sh                  # normal tick
#   ./scripts/runner-disk-admission-restore.sh --dry-run         # print, don't act
#   ./scripts/runner-disk-admission-restore.sh --mount /data
#   ./scripts/runner-disk-admission-restore.sh --pause-dir /path/to/state/disk-admission-pause
#
# Exit codes: 0 always on a clean tick (including "nothing to do"); 2 bad args;
# 3 missing deps. A restore batch failing to bring a runner back logs loudly
# but does not fail the unit — the next tick re-attempts (the marker is only
# removed on a verified `docker start` + Status=running).
#
# Runs on .201 via deploy/disk-gc/onex-runner-disk-guard.timer. Log:
# ~/.local/log/onex/runner-disk-admission-restore.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOUNT="/data"
PAUSE_DIR="${SCRIPT_DIR}/../docker/state/disk-admission-pause"
STATE_FILE="${HOME}/.local/state/onex/runner-disk-admission-restore-state.json"
LOG_FILE="${HOME}/.local/log/onex/runner-disk-admission-restore.log"
DRY_RUN=false

# Below this floor, do nothing at all — paused runners stay paused. This must
# stay comfortably below RESTORE_FLOOR_GB so a hovering-near-critical disk
# never oscillates pause/restore.
CRITICAL_FLOOR_GB="${RUNNER_DISK_GUARD_CRITICAL_FLOOR_GB:-15}"
# Must be free AND climbing for CLIMB_TICKS_REQUIRED consecutive ticks before
# the first canary batch releases.
RESTORE_FLOOR_GB="${RUNNER_DISK_GUARD_RESTORE_FLOOR_GB:-40}"
CLIMB_TICKS_REQUIRED="${RUNNER_DISK_GUARD_CLIMB_TICKS_REQUIRED:-2}"
# Batch sizes in restore order: first batch is the canary (small, cheap to
# revert-by-inaction if the slope turns negative), later batches ramp up —
# matches the 10/20/20/20 batching used live on 2026-08-22.
BATCH_SIZES="${RUNNER_DISK_GUARD_BATCH_SIZES:-10 20 20 20 20}"
DOCKER_BIN="${RUNNER_DISK_GUARD_DOCKER_BIN:-docker}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mount) MOUNT="$2"; shift 2 ;;
    --pause-dir) PAUSE_DIR="$2"; shift 2 ;;
    --state-file) STATE_FILE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$STATE_FILE")"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [runner-disk-guard] $*" | tee -a "$LOG_FILE" >&2; }

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 3; }

# ---------------------------------------------------------------------------
# Read current free space (test seam: RUNNER_DISK_GUARD_AVAIL_KB_OVERRIDE lets
# tests inject an exact reading without a real near-full filesystem).
# ---------------------------------------------------------------------------
if [[ -n "${RUNNER_DISK_GUARD_AVAIL_KB_OVERRIDE:-}" ]]; then
  avail_kb="${RUNNER_DISK_GUARD_AVAIL_KB_OVERRIDE}"
else
  target="$MOUNT"
  df -P "$target" >/dev/null 2>&1 || target="/"
  avail_kb="$(df -Pk "$target" | awk 'NR==2 {print $4}')"
fi

if ! [[ "$avail_kb" =~ ^[0-9]+$ ]]; then
  log "ERROR: could not read free space on ${MOUNT}; skipping this tick (fail-safe: no restore action)."
  exit 0
fi
avail_gb="$(( avail_kb / 1024 / 1024 ))"

# ---------------------------------------------------------------------------
# Load previous-tick state, compute slope + consecutive-climb streak.
# ---------------------------------------------------------------------------
prev_avail_kb=0
climb_streak=0
if [[ -f "$STATE_FILE" ]]; then
  prev_avail_kb="$(python3 -c "import json,sys; print(json.load(open('$STATE_FILE')).get('avail_kb', 0))" 2>/dev/null || echo 0)"
  climb_streak="$(python3 -c "import json,sys; print(json.load(open('$STATE_FILE')).get('climb_streak', 0))" 2>/dev/null || echo 0)"
fi
[[ "$prev_avail_kb" =~ ^[0-9]+$ ]] || prev_avail_kb=0
[[ "$climb_streak" =~ ^[0-9]+$ ]] || climb_streak=0

restore_floor_kb=$(( RESTORE_FLOOR_GB * 1024 * 1024 ))
critical_floor_kb=$(( CRITICAL_FLOOR_GB * 1024 * 1024 ))
is_climbing=false
[[ "$avail_kb" -gt "$prev_avail_kb" ]] && is_climbing=true

if [[ "$avail_kb" -lt "$critical_floor_kb" ]]; then
  log "CRITICAL: ${avail_gb}GB free < ${CRITICAL_FLOOR_GB}GB floor — no restore action this tick; paused runners stay paused."
  new_streak=0
elif [[ "$avail_kb" -ge "$restore_floor_kb" ]] && [[ "$is_climbing" == true ]]; then
  new_streak=$(( climb_streak + 1 ))
  log "CLIMBING: ${avail_gb}GB free >= ${RESTORE_FLOOR_GB}GB floor, up from previous tick — streak ${new_streak}/${CLIMB_TICKS_REQUIRED}."
elif [[ "$avail_kb" -ge "$restore_floor_kb" ]] && [[ "$is_climbing" != true ]]; then
  log "FLAT/DECLINING at ${avail_gb}GB free (>= floor but not climbing) — resetting climb streak; hold at current restore level."
  new_streak=0
else
  log "BELOW RESTORE FLOOR: ${avail_gb}GB free < ${RESTORE_FLOOR_GB}GB — no restore action this tick."
  new_streak=0
fi

# ---------------------------------------------------------------------------
# Enumerate paused runners (oldest paused_at first — restore in pause order).
# ---------------------------------------------------------------------------
paused_runners=()
if [[ -d "$PAUSE_DIR" ]]; then
  while IFS= read -r marker; do
    [[ -z "$marker" ]] && continue
    name="$(basename "$marker")"
    paused_at="$(awk -F= '/^paused_at=/{print $2}' "$marker" 2>/dev/null || echo "9999")"
    printf '%s\t%s\n' "$paused_at" "$name"
  done < <(find "$PAUSE_DIR" -maxdepth 1 -type f 2>/dev/null) | sort | while IFS=$'\t' read -r _ n; do
    echo "$n"
  done > "${STATE_FILE}.paused_order.tmp" 2>/dev/null || : > "${STATE_FILE}.paused_order.tmp"
  while IFS= read -r n; do
    [[ -z "$n" ]] && continue
    paused_runners+=("$n")
  done < "${STATE_FILE}.paused_order.tmp"
  rm -f "${STATE_FILE}.paused_order.tmp"
fi

restored_this_tick=()
if [[ "${#paused_runners[@]}" -eq 0 ]]; then
  log "No paused runners under ${PAUSE_DIR}."
elif [[ "$new_streak" -lt "$CLIMB_TICKS_REQUIRED" ]]; then
  log "${#paused_runners[@]} paused runner(s) waiting; climb streak ${new_streak}/${CLIMB_TICKS_REQUIRED} not yet met — no batch released this tick."
else
  # Streak requirement met: release exactly ONE batch this tick (the size of
  # the next unreleased batch in BATCH_SIZES, capped by however many are still
  # paused). Requiring the streak again next tick before the FOLLOWING batch
  # is what "slope-plus-canary" means in practice — one batch, then re-prove
  # the slope, never a single large unbatched restore.
  restored_count=0
  if [[ -f "${STATE_FILE}.batch_index" ]]; then
    batch_index="$(cat "${STATE_FILE}.batch_index" 2>/dev/null || echo 0)"
  else
    batch_index=0
  fi
  [[ "$batch_index" =~ ^[0-9]+$ ]] || batch_index=0
  read -r -a batch_sizes_arr <<< "$BATCH_SIZES"
  last_idx=$(( ${#batch_sizes_arr[@]} - 1 ))
  batch_size="${batch_sizes_arr[$last_idx]}"
  if [[ "$batch_index" -lt "${#batch_sizes_arr[@]}" ]]; then
    batch_size="${batch_sizes_arr[$batch_index]}"
  fi

  log "Climb streak satisfied (${new_streak}/${CLIMB_TICKS_REQUIRED}) — releasing batch #$((batch_index + 1)) (size ${batch_size}) of ${#paused_runners[@]} paused runner(s)."

  i=0
  for name in "${paused_runners[@]}"; do
    [[ "$i" -ge "$batch_size" ]] && break
    i=$((i + 1))
    marker="${PAUSE_DIR}/${name}"
    if [[ "$DRY_RUN" == true ]]; then
      log "DRY-RUN: would docker start ${name} and clear ${marker}"
      continue
    fi
    if ! command -v "$DOCKER_BIN" >/dev/null 2>&1; then
      log "ERROR: ${DOCKER_BIN} not found — cannot restore ${name}"
      continue
    fi
    if "$DOCKER_BIN" start "$name" >>"$LOG_FILE" 2>&1; then
      status="$("$DOCKER_BIN" inspect --format '{{.State.Status}}' "$name" 2>/dev/null || echo unknown)"
      if [[ "$status" == "running" ]]; then
        rm -f "$marker"
        restored_count=$((restored_count + 1))
        restored_this_tick+=("$name")
        log "RESTORED ${name} (Status=running); pause marker cleared."
      else
        log "WARNING: docker start ${name} returned but Status=${status} (not running) — marker kept, next tick retries."
      fi
    else
      log "FAILED to docker start ${name} — marker kept, next tick retries."
    fi
  done

  if [[ "$DRY_RUN" != true ]]; then
    echo $((batch_index + 1)) > "${STATE_FILE}.batch_index"
    # Require the streak to be re-proven before the next batch: reset to 0 so
    # the NEXT batch only releases after another CLIMB_TICKS_REQUIRED
    # consecutive climbing ticks with this batch back in rotation.
    new_streak=0
  fi
  log "Batch complete: ${restored_count}/${#paused_runners[@]} candidate(s) restored this tick."
fi

# Reset the batch index once every paused runner has been cleared, so a future
# incident starts its own batching sequence from the canary size again.
remaining_paused=0
[[ -d "$PAUSE_DIR" ]] && remaining_paused="$(find "$PAUSE_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$remaining_paused" -eq 0 ]] && [[ -f "${STATE_FILE}.batch_index" ]]; then
  rm -f "${STATE_FILE}.batch_index"
fi

if [[ "$DRY_RUN" != true ]]; then
  python3 -c "
import json
json.dump({'avail_kb': $avail_kb, 'climb_streak': $new_streak}, open('$STATE_FILE', 'w'))
"
fi

exit 0
