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

# Build the stdin JSON envelope from the scratch files (SCRATCH_DIR env tells the
# encoder where to read), then pipe it straight into the planner. Two simple
# processes, no nested subprocess, no env-var size limit.
PLAN_JSON="$(
  SCRATCH_DIR="$SCRATCH" python3 -c '
import json, os
d = os.environ["SCRATCH_DIR"]
print(json.dumps({
    "images_ndjson": open(os.path.join(d, "images.ndjson")).read(),
    "ps_ndjson": open(os.path.join(d, "ps.ndjson")).read(),
    "inuse": open(os.path.join(d, "inuse.txt")).read(),
}))
' | KEEP_LIST="$KEEP_LIST" python3 "${SCRIPT_DIR}/disk_gc_plan.py"
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
# OMN-15804: id<TAB>ref1,ref2,... — one line per removal-candidate image id, refs
# comma-joined (empty string when the id is dangling / has no repo:tag to untag).
IMAGE_REFS_TSV="$(echo "$PLAN_JSON" | python3 -c '
import json, sys
plan = json.load(sys.stdin)
refs = plan.get("remove_image_refs", {})
for iid in plan["remove_image_ids"]:
    joined = ",".join(refs.get(iid, []))
    print(f"{iid}\t{joined}")
')"

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

if [[ -n "$IMAGE_REFS_TSV" ]]; then
  while IFS=$'\t' read -r iid refs_csv; do
    [[ -z "$iid" ]] && continue

    # OMN-15804 fresh in-use re-check: the plan's protect_running decision was
    # made against a docker-inventory SNAPSHOT taken before the builder-cache
    # prune / stopped-container removal above ran. Re-derive liveness right
    # before deletion via docker's own `ancestor` filter, which resolves an
    # image id or repo:tag to every container (running OR stopped) built from
    # it — closing the snapshot-staleness + short-vs-full-id gaps a static
    # substring match against a pre-captured `docker ps` list cannot catch.
    if [[ -n "$(docker ps -a --filter "ancestor=${iid}" --format '{{.ID}}' 2>/dev/null)" ]]; then
      log "kept image $iid (fresh in-use re-check: referenced by a container)"
      continue
    fi

    # OMN-15804: untag every repo:tag ref before the final by-id remove.
    # `docker rmi <id>` alone fails "must be forced - referenced in multiple
    # repositories" whenever more than one repo:tag points at the same id —
    # this was why the native timer identified ~101 candidates and removed
    # ZERO across 3+ cycles. Untagging each ref first (never -f) drops the
    # tag; the final `docker rmi <id>` (or the last untag itself) frees the
    # underlying image once no ref remains.
    untag_failed=false
    if [[ -n "$refs_csv" ]]; then
      IFS=',' read -ra refs_arr <<< "$refs_csv"
      for ref in "${refs_arr[@]}"; do
        [[ -z "$ref" ]] && continue
        if docker rmi "$ref" >>"$LOG_FILE" 2>&1; then
          log "untagged $ref"
        else
          log "FAILED to untag $ref"
          untag_failed=true
        fi
      done
    fi

    if [[ "$untag_failed" == true ]]; then
      log "kept/failed image $iid (one or more tag refs failed to untag; not force-removing)"
      continue
    fi

    # Dangling images (no refs) or an id whose underlying layers survived
    # untagging (should not happen once every ref above succeeded, but the
    # `docker image ls` id may still resolve if this id was ALSO a parent
    # layer of another image) still need this final by-id remove.
    if docker image inspect "$iid" >/dev/null 2>&1; then
      if docker rmi "$iid" >>"$LOG_FILE" 2>&1; then
        log "removed image $iid"
      else
        log "kept/failed image $iid (likely still referenced)"
      fi
    else
      log "removed image $iid (freed by untag)"
    fi
  done <<< "$IMAGE_REFS_TSV"
fi

log "Done. df after:"
DF_OUT="$(df -h /data 2>/dev/null || df -h /)"
echo "$DF_OUT" | tee -a "$LOG_FILE" >&2

# OMN-15804: surface the watermark threshold (shared with disk-watermark-check.sh,
# no new alerting path — this is a log-line-only warning) directly in the GC
# summary so a breach is visible without cross-referencing a second log file.
DF_USED_PCT="$(echo "$DF_OUT" | awk 'NR==2 {gsub(/%/,"",$5); print $5}')"
WATERMARK_WARN_PCT=85
if [[ "$DF_USED_PCT" =~ ^[0-9]+$ ]] && [[ "$DF_USED_PCT" -ge "$WATERMARK_WARN_PCT" ]]; then
  log "WARNING: disk usage ${DF_USED_PCT}% >= watermark ${WATERMARK_WARN_PCT}% — see disk-watermark-check.sh"
fi

exit 0
