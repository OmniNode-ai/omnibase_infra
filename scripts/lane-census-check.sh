#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# lane-census-check.sh — Reconcile declared desired-state lanes vs actual (OMN-13011).
#
# THE CLASS FIX. Nothing reconciled desired vs actual lane state, so the same
# drift kept recurring with zero signal:
#   - volume config drift (OMN-12945 family)
#   - WORKER_REPLICAS silent zero (OMN-12988 / OMN-12990)
#   - 2026-06-11: prod runtime containers + broker network silently absent, no alert
#
# This script gathers the live docker inventory, diffs it against the versioned
# lane manifest (deploy/lane-census/lane-manifest.yaml) via the pure planner
# (scripts/lane_census_plan.py), and on drift publishes a typed bus event
# (scripts/lane_census_event.py) so the sweep auto-ticket path opens a Linear
# ticket naming exactly what is missing/extra.
#
# Fail-fast, NO warn-only mode (gates-block policy): a drift on a non-optional
# lane exits non-zero. The systemd unit treats that as a maintenance signal.
#
# Usage:
#   ./scripts/lane-census-check.sh                  # reconcile ALL non-optional lanes
#   ./scripts/lane-census-check.sh --lane prod      # one lane
#   ./scripts/lane-census-check.sh --dry-run        # print event/plan, do NOT publish
#   ./scripts/lane-census-check.sh --json           # emit the plan JSON to stdout
#
# Exit codes: 0 no drift, 30 drift detected (event emitted), 2 bad args, 3 missing deps.
#
# Runs on .201 via the SHARED onex-disk-gc.timer (4th ExecStart — coordinated with
# OMN-13008 rather than a second timer). Log: ~/.local/log/onex/lane-census.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LANE=""
DRY_RUN=false
EMIT_JSON=false
LOG_FILE="${HOME}/.local/log/onex/lane-census.log"
DRIFT_TOPIC="onex.evt.infra.lane-census-drift.v1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lane) LANE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --json) EMIT_JSON=true; shift ;;
    --help|-h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [lane-census] $*" | tee -a "$LOG_FILE" >&2; }

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 3; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 3; }

HOST="${LANE_CENSUS_HOST:-$(hostname)}"
log "Starting (lane=${LANE:-ALL}, host=$HOST, $( [[ "$DRY_RUN" == true ]] && echo DRY-RUN || echo LIVE ))"

# ---------------------------------------------------------------------------
# Gather actual state. Inventory goes to per-run scratch files (never /tmp) and
# is handed to the pure planner on stdin as a JSON envelope. No env-var size
# limit, no decision logic in bash.
# ---------------------------------------------------------------------------
SCRATCH="$(mktemp -d "$(dirname "$LOG_FILE")/lane-census.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT

docker ps -a --no-trunc --format '{{json .}}' >"$SCRATCH/ps.ndjson" 2>/dev/null || : >"$SCRATCH/ps.ndjson"
docker network ls --format '{{.Name}}' >"$SCRATCH/networks.txt" 2>/dev/null || : >"$SCRATCH/networks.txt"

# Resolve the runtime tag from the deploy-agent runtime version when available
# (relaxes to the default pattern if unresolvable — see the planner).
RUNTIME_TAG="${RUNTIME_TAG:-}"

ENVELOPE_JSON="$(
  SCRATCH_DIR="$SCRATCH" LANE="$LANE" RUNTIME_TAG="$RUNTIME_TAG" python3 -c '
import json, os
d = os.environ["SCRATCH_DIR"]
lane = os.environ.get("LANE") or None
containers = []
with open(os.path.join(d, "ps.ndjson")) as fh:
    for line in fh:
        line = line.strip()
        if line:
            containers.append(json.loads(line))
networks = [n.strip() for n in open(os.path.join(d, "networks.txt")) if n.strip()]
print(json.dumps({
    "lane": lane,
    "containers": containers,
    "networks": networks,
    "runtime_tag": os.environ.get("RUNTIME_TAG") or None,
}))
'
)"

PLAN_JSON="$(echo "$ENVELOPE_JSON" | python3 "${SCRIPT_DIR}/lane_census_plan.py")"

if [[ "$EMIT_JSON" == true ]]; then
  echo "$PLAN_JSON"
fi

HAS_DRIFT="$(echo "$PLAN_JSON" | python3 -c 'import json,sys;print(json.load(sys.stdin)["has_drift"])')"

if [[ "$HAS_DRIFT" != "True" ]]; then
  log "No lane drift. Desired == actual."
  exit 0
fi

# Build the typed drift event from the plan.
EVENT_JSON="$(echo "$PLAN_JSON" | LANE_CENSUS_HOST="$HOST" python3 "${SCRIPT_DIR}/lane_census_event.py")"
log "DRIFT detected:"
# Render each finding to stderr. The event JSON is passed via env (EVENT_JSON) so
# the single-quoted heredoc body needs no shell escaping of inner Python quotes.
EVENT_JSON="$EVENT_JSON" python3 <<'PYEOF' >&2 || true
import json, os
event = json.loads(os.environ["EVENT_JSON"])
for finding in event["findings"]:
    print("  [{severity}] {kind} {container} ({lane}): {detail}".format(**finding))
PYEOF

if [[ "$DRY_RUN" == true ]]; then
  echo "$EVENT_JSON"
  log "DRY-RUN — not publishing drift event."
  exit 30
fi

# Publish to the bus. The broker address MUST come from KAFKA_BOOTSTRAP_SERVERS —
# fail-fast, no localhost/default fallback (Rule 8). The systemd unit injects it;
# an operator running by hand must export it. We never hardcode a broker / LAN IP.
if [[ -z "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
  log "KAFKA_BOOTSTRAP_SERVERS unset — cannot publish. Drift event logged above for manual replay."
  exit 30
fi
BOOTSTRAP="$KAFKA_BOOTSTRAP_SERVERS"
if command -v rpk >/dev/null 2>&1; then
  if echo "$EVENT_JSON" | rpk topic produce "$DRIFT_TOPIC" --brokers "$BOOTSTRAP" >>"$LOG_FILE" 2>&1; then
    log "published lane-census-drift event to $DRIFT_TOPIC via rpk"
  else
    log "FAILED to publish via rpk (broker=$BOOTSTRAP) — event logged above for manual replay"
  fi
else
  log "rpk not found — event logged above; install rpk or wire a producer to publish"
fi

# Fail-fast on drift (gates-block policy, no warn-only mode).
exit 30
