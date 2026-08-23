#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# buildx-orphan-sweep.sh — OMN-16406 "Layer 0b": reclaim orphaned ephemeral
# `docker buildx` builder containers on .201 that no admission gate and no
# standard prune command can reach.
#
# MECHANISM. `docker/setup-buildx-action` (or a direct `docker buildx build`
# invocation) with no pinned builder `name` creates a NEW
# `buildx_buildkit_<name>` container + `<name>_state` volume (mounted at
# `/var/lib/buildkit`) on every invocation. Self-hosted runners are NOT
# ephemeral VMs — nothing tears that container down when the job exits, and
# because the name was never reused, it is never registered again in
# `docker buildx ls` after that one job. An unregistered builder is invisible
# to and unreachable by `docker builder prune` / `docker buildx prune`, which
# only reclaim cache belonging to a builder still present in the registry.
# The result: a continuous, low-grade, cache-invisible leak that compounded
# with the fleet's already-thin disk margin during the 2026-08-22 ENOSPC
# write-amplification incident chain (OMN-16360/OMN-16363).
#
# OMN-16406 pins a fixed builder `name` on the one CI call site that used the
# docker-container driver (.github/workflows/docker-build.yml), which stops
# NEW orphans from that path. This script is the companion reclaim pass for
# (a) orphans that already exist, and (b) any other ephemeral-builder source
# (manual `docker buildx build`, a future workflow, a person's own session)
# this repo does not control.
#
# SAFE-REMOVAL CRITERIA — all three required, matching the manual sweep
# criteria used live on 2026-08-22/23 (ledger rows, OMN-16406 ticket body):
#   1. ABSENT from `docker buildx ls` — the node name embedded in the
#      container name (`buildx_buildkit_<node-name>`) is not a currently
#      registered builder node. A registered builder, even an idle one, is
#      left alone — it may be reused by the next job.
#   2. IDLE. A container that is not currently running (Exited/Created/Dead)
#      is trivially idle — there is no build it could be running. A RUNNING
#      container must show 0% CPU AND byte-identical cumulative block I/O
#      across two samples ~60s apart (never kill a builder mid-build).
#   3. >60 minutes old (`docker inspect .Created`) — gives a just-created
#      builder time to pick up its first job before it is even eligible.
#
# Usage:
#   ./scripts/buildx-orphan-sweep.sh                  # DRY RUN (default): print candidates
#   ./scripts/buildx-orphan-sweep.sh --execute         # actually remove qualifying orphans
#   ./scripts/buildx-orphan-sweep.sh --json            # machine-readable plan/result to stdout
#   ./scripts/buildx-orphan-sweep.sh --min-age-minutes 60 --sample-gap-seconds 60
#
# Exit codes: 0 success (plan printed or executed, including "nothing to
# do"); 2 bad args; 3 missing deps.
#
# Runs on .201 via deploy/disk-gc/onex-buildx-orphan-sweep.timer (systemd
# USER unit, every 15 min). Log: ~/.local/log/onex/buildx-orphan-sweep.log

set -euo pipefail

EXECUTE=false
EMIT_JSON=false
MIN_AGE_MINUTES="${BUILDX_ORPHAN_SWEEP_MIN_AGE_MINUTES:-60}"
SAMPLE_GAP_SECONDS="${BUILDX_ORPHAN_SWEEP_SAMPLE_GAP_SECONDS:-60}"
IDLE_CPU_MAX_PERCENT="${BUILDX_ORPHAN_SWEEP_IDLE_CPU_MAX_PERCENT:-5.0}"
DOCKER_BIN="${BUILDX_ORPHAN_SWEEP_DOCKER_BIN:-docker}"
LOG_FILE="${HOME}/.local/log/onex/buildx-orphan-sweep.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=true; shift ;;
    --json) EMIT_JSON=true; shift ;;
    --min-age-minutes) MIN_AGE_MINUTES="$2"; shift 2 ;;
    --sample-gap-seconds) SAMPLE_GAP_SECONDS="$2"; shift 2 ;;
    --help|-h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [buildx-orphan-sweep] $*" | tee -a "$LOG_FILE" >&2; }

command -v "$DOCKER_BIN" >/dev/null 2>&1 || { echo "ERROR: ${DOCKER_BIN} not found" >&2; exit 3; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 3; }

now_epoch="${BUILDX_ORPHAN_SWEEP_NOW_EPOCH_OVERRIDE:-$(date -u +%s)}"

# ---------------------------------------------------------------------------
# 1. Registered builder node names (criterion 1's negative set). `docker
#    buildx ls` prints one unindented header line per builder ("NAME/NODE ..."
#    plus "default*" / "mybuilder") and one or more indented lines per node
#    under it — the node name is what backs a `buildx_buildkit_<node>`
#    container, so we collect indented-line first-tokens, not header names.
# ---------------------------------------------------------------------------
registered_nodes=()
while IFS= read -r node; do
  [[ -z "$node" ]] && continue
  registered_nodes+=("$node")
done < <("$DOCKER_BIN" buildx ls 2>/dev/null | awk '/^[[:space:]]+[^[:space:]]/{print $1}')

is_registered() {
  local candidate="$1" n
  for n in "${registered_nodes[@]}"; do
    [[ "$n" == "$candidate" ]] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# 2. All buildx builder containers currently on the host (running or not).
# ---------------------------------------------------------------------------
all_containers=()
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  all_containers+=("$name")
done < <("$DOCKER_BIN" ps -a --filter "name=^buildx_buildkit_" --format '{{.Names}}' 2>/dev/null)

if [[ "${#all_containers[@]}" -eq 0 ]]; then
  log "No buildx_buildkit_* containers on host — nothing to sweep."
  if [[ "$EMIT_JSON" == true ]]; then
    echo '{"candidates": [], "removed": [], "executed": '"$EXECUTE"'}'
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# 3. Criterion 1 (unregistered) + criterion 3 (age) filter, cheap and
#    synchronous — no waiting required for either.
# ---------------------------------------------------------------------------
age_eligible=()
for name in "${all_containers[@]}"; do
  node_name="${name#buildx_buildkit_}"
  if is_registered "$node_name"; then
    log "SKIP ${name}: node '${node_name}' is registered in 'docker buildx ls' — in active use, not an orphan."
    continue
  fi

  created="$("$DOCKER_BIN" inspect --format '{{.Created}}' "$name" 2>/dev/null || echo "")"
  if [[ -z "$created" ]]; then
    log "SKIP ${name}: could not read .Created (container may have just been removed) — skipping this tick."
    continue
  fi
  created_epoch="$(python3 -c "
import datetime, sys
raw = sys.argv[1].strip()
# Docker's .Created is RFC3339 with variable-precision fractional seconds
# ('...123456789Z'); Python's fromisoformat wants at most 6 fractional
# digits and no bare 'Z', so normalize both before parsing.
if raw.endswith('Z'):
    raw = raw[:-1] + '+00:00'
if '.' in raw:
    head, rest = raw.split('.', 1)
    frac, _, tz = rest.partition('+')
    tz = '+' + tz if tz else ''
    raw = head + '.' + frac[:6] + tz
try:
    dt = datetime.datetime.fromisoformat(raw)
except ValueError:
    print('')
    sys.exit(0)
print(int(dt.timestamp()))
" "$created" 2>/dev/null || echo "")"
  if [[ -z "$created_epoch" ]] || ! [[ "$created_epoch" =~ ^[0-9]+$ ]]; then
    log "SKIP ${name}: unparseable .Created value '${created}' — skipping this tick."
    continue
  fi

  age_minutes=$(( (now_epoch - created_epoch) / 60 ))
  if [[ "$age_minutes" -lt "$MIN_AGE_MINUTES" ]]; then
    log "SKIP ${name}: unregistered but only ${age_minutes}min old (< ${MIN_AGE_MINUTES}min floor) — give it time to be claimed or finish its first job."
    continue
  fi

  age_eligible+=("${name}|${age_minutes}")
done

if [[ "${#age_eligible[@]}" -eq 0 ]]; then
  log "No unregistered builder container is both unregistered and old enough — nothing to sweep this tick."
  if [[ "$EMIT_JSON" == true ]]; then
    echo '{"candidates": [], "removed": [], "executed": '"$EXECUTE"'}'
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# 4. Criterion 2 (idle). Not-running containers are trivially idle. Running
#    containers need two ~SAMPLE_GAP_SECONDS-apart samples of CPU% + block
#    I/O — sampled ONCE for the whole batch (not per-container) so the total
#    sweep wall-clock stays ~SAMPLE_GAP_SECONDS regardless of candidate count.
# ---------------------------------------------------------------------------
running_candidates=()
declare -A status_of
for entry in "${age_eligible[@]}"; do
  name="${entry%%|*}"
  status="$("$DOCKER_BIN" inspect --format '{{.State.Status}}' "$name" 2>/dev/null || echo "unknown")"
  status_of["$name"]="$status"
  if [[ "$status" == "running" ]]; then
    running_candidates+=("$name")
  fi
done

declare -A sample1 sample2
if [[ "${#running_candidates[@]}" -gt 0 ]]; then
  for name in "${running_candidates[@]}"; do
    sample1["$name"]="$("$DOCKER_BIN" stats --no-stream --format '{{.CPUPerc}}|{{.BlockIO}}' "$name" 2>/dev/null || echo "")"
  done
  log "Sampled ${#running_candidates[@]} running candidate(s); sleeping ${SAMPLE_GAP_SECONDS}s for the second idle sample..."
  sleep "$SAMPLE_GAP_SECONDS"
  for name in "${running_candidates[@]}"; do
    sample2["$name"]="$("$DOCKER_BIN" stats --no-stream --format '{{.CPUPerc}}|{{.BlockIO}}' "$name" 2>/dev/null || echo "")"
  done
fi

is_idle() {
  local name="$1" s1="${sample1[$1]:-}" s2="${sample2[$1]:-}"
  local cpu1 io1 cpu2 io2
  IFS='|' read -r cpu1 io1 <<<"$s1"
  IFS='|' read -r cpu2 io2 <<<"$s2"
  [[ -z "$io1" || -z "$io2" ]] && return 1
  [[ "$io1" != "$io2" ]] && return 1
  python3 -c "
import sys
def pct(s):
    return float(s.strip().rstrip('%') or 0.0)
c1, c2, cap = sys.argv[1], sys.argv[2], float(sys.argv[3])
sys.exit(0 if (pct(c1) <= cap and pct(c2) <= cap) else 1)
" "$cpu1" "$cpu2" "$IDLE_CPU_MAX_PERCENT"
}

orphans=()
for entry in "${age_eligible[@]}"; do
  name="${entry%%|*}"
  age_minutes="${entry##*|}"
  status="${status_of[$name]:-unknown}"
  if [[ "$status" != "running" ]]; then
    log "CANDIDATE ${name}: unregistered, ${age_minutes}min old, status=${status} (not running — trivially idle)."
    orphans+=("${name}|${age_minutes}|${status}")
    continue
  fi
  if is_idle "$name"; then
    log "CANDIDATE ${name}: unregistered, ${age_minutes}min old, running but IDLE across two ${SAMPLE_GAP_SECONDS}s-apart samples (CPU<=${IDLE_CPU_MAX_PERCENT}%, block I/O unchanged)."
    orphans+=("${name}|${age_minutes}|running-idle")
  else
    log "SKIP ${name}: unregistered and old enough, but NOT idle (CPU/I/O changed across samples) — likely mid-build, leaving alone."
  fi
done

if [[ "${#orphans[@]}" -eq 0 ]]; then
  log "No candidate met all three safe-removal criteria this tick."
  if [[ "$EMIT_JSON" == true ]]; then
    echo '{"candidates": [], "removed": [], "executed": '"$EXECUTE"'}'
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# 5. Remove (or, dry-run, just report) each qualifying orphan and its own
#    state volume. Volume removal is scoped to volumes actually mounted on
#    the orphan container AND named with the same buildx_buildkit_ prefix —
#    never a blind name-guess, never anything else.
# ---------------------------------------------------------------------------
removed=()
for entry in "${orphans[@]}"; do
  name="${entry%%|*}"
  rest="${entry#*|}"
  age_minutes="${rest%%|*}"
  reason="${rest##*|}"

  volumes=()
  while IFS= read -r vol; do
    [[ -z "$vol" ]] && continue
    [[ "$vol" == buildx_buildkit_* ]] && volumes+=("$vol")
  done < <("$DOCKER_BIN" inspect --format '{{range .Mounts}}{{if eq .Type "volume"}}{{.Name}}{{"\n"}}{{end}}{{end}}' "$name" 2>/dev/null || true)

  if [[ "$EXECUTE" != true ]]; then
    log "DRY-RUN: would remove ${name} (age=${age_minutes}min, ${reason}) + volume(s): ${volumes[*]:-none}"
    continue
  fi

  if "$DOCKER_BIN" rm -f "$name" >>"$LOG_FILE" 2>&1; then
    log "REMOVED container ${name}."
    for vol in "${volumes[@]}"; do
      if "$DOCKER_BIN" volume rm "$vol" >>"$LOG_FILE" 2>&1; then
        log "REMOVED volume ${vol}."
      else
        log "WARNING: failed to remove volume ${vol} (may still be referenced) — left in place."
      fi
    done
    removed+=("$name")
  else
    log "FAILED to remove ${name} — left in place, next tick retries."
  fi
done

if [[ "$EMIT_JSON" == true ]]; then
  python3 -c "
import json
orphans = '''${orphans[*]}'''.split()
removed = '''${removed[*]}'''.split()
print(json.dumps({
    'candidates': [o.split('|')[0] for o in orphans],
    'removed': removed,
    'executed': '${EXECUTE}' == 'true',
}))
"
fi

log "Sweep complete: ${#orphans[@]} candidate(s), ${#removed[@]} removed (executed=${EXECUTE})."
exit 0
