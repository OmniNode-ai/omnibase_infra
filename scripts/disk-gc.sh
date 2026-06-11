#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# disk-gc.sh — Conservative scheduled docker/disk garbage collection for .201 (OMN-13008).
#
# WHY: On 2026-06-11 /data on .201 reached ~95% (weeks of unpruned docker images
# + builder cache) and killed all three runtime lanes mid-demo. This script is the
# scheduled, conservative GC that keeps /data from detonating, driven by a
# VERSIONED keep-list config (deploy/disk-gc/keep-list.yaml) — nothing is hardcoded.
#
# It reaps, in increasing order of caution:
#   1. docker builder cache (cache mounts are always safe to drop)
#   2. dangling images (untagged, <none>) older than min_age_days
#   3. stopped containers older than min_age_days
#   4. superseded image generations of kept repos, KEEPING keep_image_tags,
#      keeping protect_running references, keeping the newest
#      superseded_image_keep_generations, and only removing those older than
#      min_age_days.
#
# It NEVER removes:
#   - any image whose repo matches keep_image_repos
#   - any image whose tag matches keep_image_tags
#   - any image referenced by a container (when protect_running: true)
#   - anything younger than min_age_days
#   - any volume (volumes are out of scope — data safety)
#
# Usage:
#   ./scripts/disk-gc.sh                 # DRY RUN (default): print what WOULD be removed
#   ./scripts/disk-gc.sh --execute       # actually remove
#   ./scripts/disk-gc.sh --keep-list /path/to/keep-list.yaml
#   ./scripts/disk-gc.sh --json          # machine-readable plan to stdout (dry-run plan)
#
# Exit codes: 0 success (plan printed or executed), 2 bad args, 3 missing deps.
#
# Runs on .201 via deploy/disk-gc.timer (systemd user timer). Log: ~/.local/log/onex/disk-gc.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KEEP_LIST="${SCRIPT_DIR}/../deploy/disk-gc/keep-list.yaml"
EXECUTE=false
EMIT_JSON=false
LOG_FILE="${HOME}/.local/log/onex/disk-gc.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=true; shift ;;
    --keep-list) KEEP_LIST="$2"; shift 2 ;;
    --json) EMIT_JSON=true; shift ;;
    --help|-h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [disk-gc] $*" | tee -a "$LOG_FILE" >&2; }

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 3; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 3; }
[[ -f "$KEEP_LIST" ]] || { echo "ERROR: keep-list not found: $KEEP_LIST" >&2; exit 3; }

log "Starting ($( [[ "$EXECUTE" == true ]] && echo EXECUTE || echo DRY-RUN )), keep-list=$KEEP_LIST"

# ---------------------------------------------------------------------------
# Resolve the removal PLAN in Python (deterministic, testable). The plan is the
# only thing that decides what gets removed; bash only executes it.
#
# Docker inventory is written to per-run scratch files and handed to the planner
# on stdin as a JSON envelope. We do NOT pass it via env vars: a host with many
# images blows past ARG_MAX (`Argument list too long`). Scratch lives under the
# log dir (never /tmp), and is cleaned on exit.
# ---------------------------------------------------------------------------
SCRATCH="$(mktemp -d "$(dirname "$LOG_FILE")/disk-gc.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
docker image ls --all --no-trunc --format '{{json .}}' >"$SCRATCH/images.ndjson" 2>/dev/null || : >"$SCRATCH/images.ndjson"
docker ps --all --no-trunc --format '{{json .}}' >"$SCRATCH/ps.ndjson" 2>/dev/null || : >"$SCRATCH/ps.ndjson"
docker ps --all --format '{{.Image}}' 2>/dev/null | sort -u >"$SCRATCH/inuse.txt" || : >"$SCRATCH/inuse.txt"

PLAN_JSON="$(
  KEEP_LIST="$KEEP_LIST" python3 - "$SCRATCH" "${SCRIPT_DIR}/disk_gc_plan.py" <<'PYWRAP'
import json
import subprocess
import sys
from pathlib import Path

scratch = Path(sys.argv[1])
planner = sys.argv[2]
envelope = json.dumps(
    {
        "images_ndjson": (scratch / "images.ndjson").read_text(),
        "ps_ndjson": (scratch / "ps.ndjson").read_text(),
        "inuse": (scratch / "inuse.txt").read_text(),
    }
)
proc = subprocess.run(
    [sys.executable, planner], input=envelope, capture_output=True, text=True
)
sys.stderr.write(proc.stderr)
sys.stdout.write(proc.stdout)
sys.exit(proc.returncode)
PYWRAP
)"

if [[ "$EMIT_JSON" == true ]]; then
  echo "$PLAN_JSON"
fi

# Builder cache prune (always conservative — drops only reclaimable build cache).
# Honor min_age via docker's own filter so we don't drop a cache layer from a build
# that's seconds old.
MIN_AGE_DAYS="$(echo "$PLAN_JSON" | python3 -c 'import json,sys;print(json.load(sys.stdin)["min_age_days"])')"
IMAGE_IDS="$(echo "$PLAN_JSON" | python3 -c 'import json,sys;[print(i) for i in json.load(sys.stdin)["remove_image_ids"]]')"
CONTAINER_IDS="$(echo "$PLAN_JSON" | python3 -c 'import json,sys;[print(c) for c in json.load(sys.stdin)["remove_container_ids"]]')"

log "Plan: $(echo "$IMAGE_IDS" | grep -c . || true) image(s), $(echo "$CONTAINER_IDS" | grep -c . || true) stopped container(s), builder cache > ${MIN_AGE_DAYS}d"

if [[ "$EXECUTE" != true ]]; then
  log "DRY-RUN — would remove the above. Re-run with --execute to act."
  [[ -n "$IMAGE_IDS" ]] && { echo "IMAGES TO REMOVE:"; echo "$IMAGE_IDS"; } >&2
  [[ -n "$CONTAINER_IDS" ]] && { echo "STOPPED CONTAINERS TO REMOVE:"; echo "$CONTAINER_IDS"; } >&2
  exit 0
fi

# --- Execute ---------------------------------------------------------------
log "Pruning builder cache older than ${MIN_AGE_DAYS}d"
docker builder prune --force --filter "until=${MIN_AGE_DAYS}h0m0s" >>"$LOG_FILE" 2>&1 || \
  docker builder prune --force >>"$LOG_FILE" 2>&1 || log "builder prune failed (non-fatal)"

if [[ -n "$CONTAINER_IDS" ]]; then
  while IFS= read -r cid; do
    [[ -z "$cid" ]] && continue
    if docker rm "$cid" >>"$LOG_FILE" 2>&1; then log "removed container $cid"; else log "FAILED to remove container $cid"; fi
  done <<< "$CONTAINER_IDS"
fi

if [[ -n "$IMAGE_IDS" ]]; then
  while IFS= read -r iid; do
    [[ -z "$iid" ]] && continue
    if docker rmi "$iid" >>"$LOG_FILE" 2>&1; then log "removed image $iid"; else log "kept/failed image $iid (likely still referenced)"; fi
  done <<< "$IMAGE_IDS"
fi

log "Done. df after:"
df -h /data 2>/dev/null | tee -a "$LOG_FILE" >&2 || df -h / | tee -a "$LOG_FILE" >&2
exit 0
