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
#   ./scripts/lane-census-check.sh --snapshot PATH  # write the census SNAPSHOT to PATH ('-' = stdout)
#
# THE SNAPSHOT IS NOT THE PLAN (OMN-18606). `--json` emits the PLAN document
# (keys: findings, has_drift, lanes_checked, schema_version). The staleness gate
# reads `emitted_at`, which the plan does not carry — so `--json` can never
# produce a file that gate accepts. The snapshot the gate reads is the typed
# EVENT document, and `--snapshot` is the only supported way to write it:
#
#   ./scripts/lane-census-check.sh --snapshot deploy/lane-census/census-snapshot.json
#
# `--snapshot` writes the file whether or not there is drift, and leaves the
# drift verdict and exit code below untouched.
#
# Exit codes: 0 no drift, 30 drift detected (event emitted), 2 bad args, 3 missing deps,
#             4 inventory unobservable (BOTH Engine API and bounded CLI failed —
#               fail-loud, NO drift event; deliberately distinct from 30 so a host
#               we cannot see is never reported as a host that is down. OMN-15466).
#
# Runs on .201 via the SHARED onex-disk-gc.timer (4th ExecStart — coordinated with
# OMN-13008 rather than a second timer). Log: ~/.local/log/onex/lane-census.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LANE=""
DRY_RUN=false
EMIT_JSON=false
# OMN-18606: destination for the gate-valid census snapshot document. Empty means
# "do not write one" (the pre-OMN-18606 behaviour). "-" means stdout.
SNAPSHOT_OUT=""
LOG_FILE="${HOME}/.local/log/onex/lane-census.log"
DRIFT_TOPIC="onex.evt.infra.lane-census-drift.v1"
# OMN-18769: published on EVERY run, drift or not. The drift topic stays the
# ALERT authority (a consumer opens a ticket from it); this one is the FACT
# a lane-health projection reads, and without a clean-run message that
# projection cannot tell a matching fleet from a census that stopped running.
OBSERVED_TOPIC="onex.evt.infra.lane-census-observed.v1"
# Inventory unobservable — NOT drift. See the exit-code table above (OMN-15466).
EXIT_INVENTORY_UNAVAILABLE=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lane) LANE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --json) EMIT_JSON=true; shift ;;
    --snapshot)
      [[ $# -ge 2 ]] || { echo "ERROR: --snapshot requires a path ('-' for stdout)" >&2; exit 2; }
      SNAPSHOT_OUT="$2"; shift 2 ;;
    --help|-h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# A snapshot to STDOUT must be the only document on stdout. --json prints the
# PLAN document and --dry-run prints the EVENT document; either alongside
# "--snapshot -" produces two concatenated JSON documents, which is exactly the
# `Extra data: line 2 column 1` shape that made the old recipes unusable.
if [[ "$SNAPSHOT_OUT" == "-" ]]; then
  if [[ "$EMIT_JSON" == true || "$DRY_RUN" == true ]]; then
    echo "ERROR: --snapshot - cannot be combined with --json or --dry-run (two JSON documents on stdout)" >&2
    exit 2
  fi
fi

mkdir -p "$(dirname "$LOG_FILE")"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [lane-census] $*" | tee -a "$LOG_FILE" >&2; }

# OMN-18606: resolve the interpreter EXPLICITLY rather than inheriting whatever
# `python3` happens to mean in the caller's PATH.
#
# WHY. This script's only third-party dependency is PyYAML, imported by
# lane_census_plan.py to read the lane manifest. On .201 the hourly
# onex-disk-gc drop-in runs with PATH=/usr/local/bin:/usr/bin:/bin and the
# system interpreter there carries PyYAML, so `python3` was correct for years
# and stays the default here. In GitHub Actions it is NOT: `uv sync` installs
# PyYAML into a project .venv, and the shared setup action puts neither
# .venv/bin on PATH nor VIRTUAL_ENV in the environment, so a bare `python3`
# resolves to the runner's system interpreter and dies with
# `ModuleNotFoundError: No module named 'yaml'` deep inside a sibling script.
# Both live dispatches of lane-census-refresh.yml failed exactly that way
# (runs 35265905466 and 35266125665, 2026-09-17).
#
# The caller names the interpreter; this script never guesses one. The default
# preserves the host path byte-for-byte.
LANE_CENSUS_PYTHON="${LANE_CENSUS_PYTHON:-python3}"
command -v "$LANE_CENSUS_PYTHON" >/dev/null 2>&1 || {
  echo "ERROR: interpreter '$LANE_CENSUS_PYTHON' not found (set LANE_CENSUS_PYTHON)" >&2
  exit 3
}

# Fail LOUD and EARLY on a usable interpreter that cannot import what the
# census needs. Without this the failure surfaces as a traceback from
# lane_census_plan.py with the offending interpreter never named, which is what
# made the CI failure above take a live dispatch to diagnose.
if ! "$LANE_CENSUS_PYTHON" -c 'import yaml' >/dev/null 2>&1; then
  echo "ERROR: interpreter '$LANE_CENSUS_PYTHON' cannot import PyYAML, which" >&2
  echo "       lane_census_plan.py needs to read the lane manifest." >&2
  echo "       In CI, set LANE_CENSUS_PYTHON to the project venv interpreter," >&2
  echo "       e.g. LANE_CENSUS_PYTHON=\"\$(uv run python -c 'import sys; print(sys.executable)')\"" >&2
  exit 3
fi

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 3; }

HOST="${LANE_CENSUS_HOST:-$(hostname)}"
log "Starting (lane=${LANE:-ALL}, host=$HOST, $( [[ "$DRY_RUN" == true ]] && echo DRY-RUN || echo LIVE ))"

# ---------------------------------------------------------------------------
# Gather actual state via the fail-loud collector (scripts/lane_census_inventory.py):
# Docker Engine API first, bounded docker-CLI fallback, hard failure if neither
# can see the host. No decision logic in bash.
#
# OMN-15466: this replaced `docker ps -a --format '{{json .}}' ... || : >file`,
# which (a) made the CLI request size=1, forcing daemon-side snapshotter.Usage
# per container — 90.363 s vs 0.150 s for the same inventory over the Engine API
# on .201's 111 containers — and (b) truncated the inventory to EMPTY on any
# docker failure, which the planner cannot distinguish from a genuine total
# outage (32 critical findings, published as real drift). A host we cannot
# observe must never be reported as a host that is down.
# ---------------------------------------------------------------------------
SCRATCH="$(mktemp -d "$(dirname "$LOG_FILE")/lane-census.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT

# Resolve the runtime tag from the deploy-agent runtime version when available
# (relaxes to the default pattern if unresolvable — see the planner).
RUNTIME_TAG="${RUNTIME_TAG:-}"

set +e
ENVELOPE_JSON="$(
  LANE="$LANE" RUNTIME_TAG="$RUNTIME_TAG" \
    "$LANE_CENSUS_PYTHON" "${SCRIPT_DIR}/lane_census_inventory.py" 2>"$SCRATCH/inventory.err"
)"
INVENTORY_RC=$?
set -e

if [[ $INVENTORY_RC -ne 0 ]]; then
  while IFS= read -r line; do [[ -n "$line" ]] && log "$line"; done <"$SCRATCH/inventory.err"
  log "ABORT: docker inventory could not be observed (exit $INVENTORY_RC). \
Publishing NO drift event — an unobservable host is not a drifted host."
  exit "$EXIT_INVENTORY_UNAVAILABLE"
fi

# Surface any fallback/degradation notices without changing the exit policy.
while IFS= read -r line; do [[ -n "$line" ]] && log "$line"; done <"$SCRATCH/inventory.err"

PLAN_JSON="$(echo "$ENVELOPE_JSON" | "$LANE_CENSUS_PYTHON" "${SCRIPT_DIR}/lane_census_plan.py")"

if [[ "$EMIT_JSON" == true ]]; then
  echo "$PLAN_JSON"
fi

HAS_DRIFT="$(echo "$PLAN_JSON" | "$LANE_CENSUS_PYTHON" -c 'import json,sys;print(json.load(sys.stdin)["has_drift"])')"

# OMN-18606: the census SNAPSHOT document is the typed event — the only carrier
# of `emitted_at`, which the staleness gate reads. Build it BEFORE the no-drift
# early exit below so a fleet that matches its manifest can still emit a census.
# Until this change the event was built only on drift, so a healthy fleet could
# produce no snapshot by any path and the gate became unsatisfiable seven days
# later with no available remedy. Writing the snapshot does NOT change this
# script's exit-code contract: the drift verdict below is unchanged.
if [[ -n "$SNAPSHOT_OUT" || "$HAS_DRIFT" == "True" ]]; then
  EVENT_JSON="$(echo "$PLAN_JSON" | LANE_CENSUS_HOST="$HOST" "$LANE_CENSUS_PYTHON" "${SCRIPT_DIR}/lane_census_event.py")"
fi

if [[ -n "$SNAPSHOT_OUT" ]]; then
  if [[ "$SNAPSHOT_OUT" == "-" ]]; then
    printf '%s\n' "$EVENT_JSON"
  else
    mkdir -p "$(dirname "$SNAPSHOT_OUT")"
    printf '%s\n' "$EVENT_JSON" >"$SNAPSHOT_OUT"
    log "census snapshot written: $SNAPSHOT_OUT"
  fi
fi

# OMN-18769: publish the census-OBSERVED fact BEFORE the no-drift early exit, so
# the clean case -- the one the drift topic structurally cannot carry -- reaches
# the bus. This block never changes this script's exit-code contract: a publish
# failure is logged and the run continues to its drift verdict, because a broker
# that is down is not a lane that has drifted.
OBSERVED_JSON="$(echo "$PLAN_JSON" | LANE_CENSUS_HOST="$HOST" "$LANE_CENSUS_PYTHON" "${SCRIPT_DIR}/lane_census_event.py" --observed)"

if [[ "$DRY_RUN" == true ]]; then
  log "DRY-RUN — not publishing census-observed event."
elif [[ -z "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
  # Rule 8: fail loud, never a localhost default. The systemd unit injects it.
  log "KAFKA_BOOTSTRAP_SERVERS unset — census-observed event NOT published."
elif ! command -v rpk >/dev/null 2>&1; then
  log "rpk not found — census-observed event NOT published."
elif echo "$OBSERVED_JSON" | rpk topic produce "$OBSERVED_TOPIC" --brokers "$KAFKA_BOOTSTRAP_SERVERS" >>"$LOG_FILE" 2>&1; then
  log "published lane-census-observed event to $OBSERVED_TOPIC via rpk"
else
  log "FAILED to publish census-observed via rpk (broker=$KAFKA_BOOTSTRAP_SERVERS)"
fi

if [[ "$HAS_DRIFT" != "True" ]]; then
  log "No lane drift. Desired == actual."
  exit 0
fi

log "DRIFT detected:"
# Render each finding to stderr. The event JSON is passed via env (EVENT_JSON) so
# the single-quoted heredoc body needs no shell escaping of inner Python quotes.
EVENT_JSON="$EVENT_JSON" "$LANE_CENSUS_PYTHON" <<'PYEOF' >&2 || true
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
