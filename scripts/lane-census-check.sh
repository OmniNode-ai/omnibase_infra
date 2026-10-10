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
#   ./scripts/lane-census-check.sh --observed-out PATH  # write the census-OBSERVED document (OMN-18769)
#   ./scripts/lane-census-check.sh --memory         # also run the lane container memory pass (OMN-19959)
#   ./scripts/lane-census-check.sh --memory --memory-event-out PATH  # and write its event to PATH
#
# THE MEMORY PASS (OMN-19959). With --memory the collector also reads every lane
# container's cgroup v2 memory.max, memory.peak and memory.events, and the worker
# logs of this host's runner containers, and the pass publishes ONE event on
# onex.evt.omnibase-infra.lane-container-memory.v1 through the broker container
# named by LANE_MEMORY_BROKER_CONTAINER (`rpk` is not on the lab host PATH, so the
# produce runs inside that container, as runner-monitor.sh does). The SASL pair
# is expanded inside the container from the variable NAMES in
# LANE_MEMORY_BROKER_SASL_USER_VAR / LANE_MEMORY_BROKER_SASL_PASS_VAR; no value
# ever reaches this host's argv. The previous pass's counters live in
# LANE_MEMORY_STATE_PATH and advance only after a successful publish.
#   exit 6  the memory event was not published (no broker container, or the produce failed)
#   exit 7  the memory counters or a runner worker log could not be read
#   exit 31 a lane container was OOM-killed this window, or hit its limit in two consecutive passes
# A memory code replaces a census 0 or 30; the census's own failure codes win.
# --dry-run builds and logs the memory event but neither publishes it nor advances the state.
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
#               we cannot see is never reported as a host that is down. OMN-15466),
#             5 host undeclared (LANE_CENSUS_HOST, default `hostname`, names no entry
#               of the manifest's `hosts:` registry — no plan, no event. OMN-19088),
#             8 drift detected and the drift event was NOT published (no broker
#               container named, or the produce failed). A drift that reached the
#               bus is 30; one that did not is 8, so a unit's exit code says
#               whether the alert was delivered. OMN-20798.
#
# THE DRIFT EVENT TAKES THE MEMORY PASS'S TRANSPORT (OMN-20798). Both events are
# produced inside the broker container named by LANE_MEMORY_BROKER_CONTAINER
# (`rpk` is not on a lab host's PATH, and a lab broker's external listener
# requires SASL). The earlier `rpk ... --brokers "$KAFKA_BOOTSTRAP_SERVERS"`
# branch could not run on any lab host and is gone.
#
# Host scoping (OMN-19088): only the lanes the manifest declares for this host
# are evaluated; the rest are reported in `lanes_not_applicable`, and one of
# them found RUNNING here is a `lane_on_undeclared_host` finding.
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
LOG_FILE="${LANE_CENSUS_LOG_FILE:-${HOME}/.local/log/onex/lane-census.log}"
DRIFT_TOPIC="onex.evt.infra.lane-census-drift.v1"
# OMN-18769: published on EVERY run, drift or not. The drift topic stays the
# ALERT authority (a consumer opens a ticket from it); this one is the FACT
# a lane-health projection reads, and without a clean-run message that
# projection cannot tell a matching fleet from a census that stopped running.
OBSERVED_TOPIC="onex.evt.omnibase-infra.lane-census-observed.v1"
# OMN-18769: destination for the census-OBSERVED document. Empty means "do not
# write one"; "-" means stdout. The topic the document names is OBSERVED_TOPIC
# above -- the publisher reads it off the document rather than being told, so
# the name cannot be spelled two ways.
OBSERVED_OUT=""
# Inventory unobservable — NOT drift. See the exit-code table above (OMN-15466).
EXIT_INVENTORY_UNAVAILABLE=4
# Host not declared in the manifest's hosts registry — NOT drift (OMN-19088).
EXIT_HOST_UNDECLARED=5
# OMN-19959: the lane container memory pass (`--memory`). One event per host per
# pass on its own topic -- a memory observation is not drift, so the drift
# topic's vocabulary is untouched. See scripts/lane_container_memory_event.py.
MEMORY=false
MEMORY_TOPIC="onex.evt.omnibase-infra.lane-container-memory.v1"
MEMORY_EVENT_OUT=""
# The memory event was not published: no broker container named, or the
# produce failed. Distinct from every census code (0, 30, 2, 3, 4, 5).
EXIT_MEMORY_UNPUBLISHED=6
# The memory counters or a runner worker log could not be read or built.
EXIT_MEMORY_UNOBSERVABLE=7
# A lane container was OOM-killed this window, or hit its limit in two
# consecutive passes.
EXIT_MEMORY_ALERT=31
MEMORY_RC=0
# OMN-20798: drift was found and its event did not reach the bus. Distinct from
# 30 (drift, delivered) and from every memory code.
EXIT_DRIFT_UNPUBLISHED=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lane) LANE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --json) EMIT_JSON=true; shift ;;
    --snapshot)
      [[ $# -ge 2 ]] || { echo "ERROR: --snapshot requires a path ('-' for stdout)" >&2; exit 2; }
      SNAPSHOT_OUT="$2"; shift 2 ;;
    --observed-out)
      [[ $# -ge 2 ]] || { echo "ERROR: --observed-out requires a path ('-' for stdout)" >&2; exit 2; }
      OBSERVED_OUT="$2"; shift 2 ;;
    --memory) MEMORY=true; shift ;;
    --memory-event-out)
      [[ $# -ge 2 ]] || { echo "ERROR: --memory-event-out requires a path" >&2; exit 2; }
      MEMORY_EVENT_OUT="$2"; shift 2 ;;
    --help|-h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# A snapshot to STDOUT must be the only document on stdout. --json prints the
# PLAN document and --dry-run prints the EVENT document; either alongside
# "--snapshot -" produces two concatenated JSON documents, which is exactly the
# `Extra data: line 2 column 1` shape that made the old recipes unusable.
if [[ "$OBSERVED_OUT" == "-" && ( "$SNAPSHOT_OUT" == "-" || "$EMIT_JSON" == true || "$DRY_RUN" == true ) ]]; then
  echo "ERROR: --observed-out - cannot be combined with another stdout document" >&2
  exit 2
fi

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

MEMORY_STATE="${LANE_MEMORY_STATE_PATH:-${HOME}/.local/state/onex/lane-container-memory-state.json}"
MEMORY_ARGS=()
if [[ "$MEMORY" == true ]]; then
  MEMORY_ARGS=(--memory-out "$SCRATCH/memory-observation.json" --memory-state "$MEMORY_STATE")
fi

set +e
ENVELOPE_JSON="$(
  LANE="$LANE" RUNTIME_TAG="$RUNTIME_TAG" \
    "$LANE_CENSUS_PYTHON" "${SCRIPT_DIR}/lane_census_inventory.py" ${MEMORY_ARGS[@]+"${MEMORY_ARGS[@]}"} 2>"$SCRATCH/inventory.err"
)"
INVENTORY_RC=$?
set -e

# OMN-19959: exit 7 from the collector means the census inventory on stdout is
# valid and only the memory observation failed. The census carries on; the
# memory pass reports its failure through the final exit code.
if [[ $INVENTORY_RC -eq $EXIT_MEMORY_UNOBSERVABLE && "$MEMORY" == true ]]; then
  MEMORY_RC=$EXIT_MEMORY_UNOBSERVABLE
  INVENTORY_RC=0
fi

if [[ $INVENTORY_RC -ne 0 ]]; then
  while IFS= read -r line; do [[ -n "$line" ]] && log "$line"; done <"$SCRATCH/inventory.err"
  log "ABORT: docker inventory could not be observed (exit $INVENTORY_RC). \
Publishing NO drift event — an unobservable host is not a drifted host."
  exit "$EXIT_INVENTORY_UNAVAILABLE"
fi

# Surface any fallback/degradation notices without changing the exit policy.
while IFS= read -r line; do [[ -n "$line" ]] && log "$line"; done <"$SCRATCH/inventory.err"

# ---------------------------------------------------------------------------
# Produce one document to a topic through the broker container (OMN-19959,
# OMN-20798). The SASL pair is expanded INSIDE the broker container by `sh -c`;
# this host passes only the variable names. rpk reads RPK_USER / RPK_PASS /
# RPK_SASL_MECHANISM from the environment, so no flag carries a credential.
# Returns 0 on a published document; the caller logs its own context. Requires
# LANE_MEMORY_BROKER_CONTAINER to be non-empty (callers check and report that).
# ---------------------------------------------------------------------------
broker_produce() {
  local topic="$1" file="$2"
  local broker="${LANE_MEMORY_BROKER_CONTAINER}"
  local user_var="${LANE_MEMORY_BROKER_SASL_USER_VAR:-DEV_KAFKA_SASL_USERNAME}"
  local pass_var="${LANE_MEMORY_BROKER_SASL_PASS_VAR:-DEV_KAFKA_SASL_PASSWORD}"
  local mechanism="${LANE_MEMORY_BROKER_SASL_MECHANISM:-SCRAM-SHA-256}"
  docker exec -i "$broker" sh -c \
      'RPK_USER="${'"$user_var"'}" RPK_PASS="${'"$pass_var"'}" RPK_SASL_MECHANISM="'"$mechanism"'" rpk topic produce "'"$topic"'"' \
      <"$file" >>"$LOG_FILE" 2>&1
}

# ---------------------------------------------------------------------------
# OMN-19959: the lane container memory pass (invoked after the planner, below).
# Every failure is a distinct exit code carried to the end of the script by
# finish(); none is a warning.
# ---------------------------------------------------------------------------
memory_pass() {
  local event_file="$SCRATCH/memory-event.json"
  local state_out="$SCRATCH/memory-state.json"
  local alerts_file="$SCRATCH/memory-alerts.txt"

  if ! "$LANE_CENSUS_PYTHON" "${SCRIPT_DIR}/lane_container_memory_event.py" \
      --host "$HOST" \
      --observation "$SCRATCH/memory-observation.json" \
      --state "$MEMORY_STATE" \
      --event-out "$event_file" \
      --state-out "$state_out" \
      --alerts-out "$alerts_file" 2>"$SCRATCH/memory-build.err"; then
    while IFS= read -r line; do [[ -n "$line" ]] && log "$line"; done <"$SCRATCH/memory-build.err"
    log "MEMORY: the memory event could not be built (exit $EXIT_MEMORY_UNOBSERVABLE)."
    MEMORY_RC=$EXIT_MEMORY_UNOBSERVABLE
    return 0
  fi

  if [[ -n "$MEMORY_EVENT_OUT" ]]; then
    mkdir -p "$(dirname "$MEMORY_EVENT_OUT")"
    cp "$event_file" "$MEMORY_EVENT_OUT"
  fi

  local alert_count
  alert_count="$(grep -c . "$alerts_file" || true)"
  while IFS= read -r line; do [[ -n "$line" ]] && log "MEMORY ALERT: $line"; done <"$alerts_file"
  log "MEMORY: built event for host=$HOST ($(wc -c <"$event_file" | tr -d ' ') bytes, ${alert_count} alert(s))"

  if [[ "$DRY_RUN" == true ]]; then
    log "DRY-RUN — not publishing the memory event and not advancing $MEMORY_STATE."
    [[ "$alert_count" -gt 0 ]] && MEMORY_RC=$EXIT_MEMORY_ALERT
    return 0
  fi

  local broker="${LANE_MEMORY_BROKER_CONTAINER:-}"
  if [[ -z "$broker" ]]; then
    log "MEMORY: LANE_MEMORY_BROKER_CONTAINER is unset — the memory event is NOT published (exit $EXIT_MEMORY_UNPUBLISHED)."
    MEMORY_RC=$EXIT_MEMORY_UNPUBLISHED
    return 0
  fi
  if broker_produce "$MEMORY_TOPIC" "$event_file"; then
    mkdir -p "$(dirname "$MEMORY_STATE")"
    mv "$state_out" "$MEMORY_STATE"
    log "MEMORY: published to $MEMORY_TOPIC via broker container $broker; state advanced."
  else
    log "MEMORY: produce to $MEMORY_TOPIC via broker container $broker FAILED — state NOT advanced (exit $EXIT_MEMORY_UNPUBLISHED)."
    MEMORY_RC=$EXIT_MEMORY_UNPUBLISHED
    return 0
  fi
  [[ "$alert_count" -gt 0 ]] && MEMORY_RC=$EXIT_MEMORY_ALERT
  return 0
}


# OMN-19088: the planner evaluates only the lanes the manifest declares for this
# host and reports the others not-applicable. It resolves HOST through the
# manifest's `hosts:` registry and refuses (exit 5) a host that registry does
# not declare. No plan exists for such a host, so nothing is emitted, written or
# published: a census that cannot say which host it describes describes none.
set +e
PLAN_JSON="$(echo "$ENVELOPE_JSON" | LANE_CENSUS_HOST="$HOST" "$LANE_CENSUS_PYTHON" "${SCRIPT_DIR}/lane_census_plan.py" 2>"$SCRATCH/plan.err")"
PLAN_RC=$?
set -e
if [[ $PLAN_RC -ne 0 ]]; then
  while IFS= read -r line; do [[ -n "$line" ]] && log "$line"; done <"$SCRATCH/plan.err"
  if [[ $PLAN_RC -eq $EXIT_HOST_UNDECLARED ]]; then
    log "ABORT: host '$HOST' is not declared in the lane manifest's hosts registry. \
Publishing NO census — declare the host and its lanes first."
  else
    log "ABORT: the lane planner failed (exit $PLAN_RC)."
  fi
  exit "$PLAN_RC"
fi

# OMN-19959: the memory pass runs once the planner has accepted this host, so a
# host the manifest does not declare (exit 5) publishes nothing at all, as
# OMN-19088 requires. It runs before the drift handling, so a drifted or clean
# census publishes it alike.
if [[ "$MEMORY" == true && $MEMORY_RC -eq 0 ]]; then
  memory_pass
fi

# The census's own clean (0) and drift (30) outcomes give way to a memory code,
# so a memory failure or alert is never masked by a clean fleet. The census's
# failure codes (2, 3, 4, 5) exit above, before the memory pass, and keep their
# meaning. An unpublished drift (8) is a failure of the same standing and keeps
# its code; the memory code is still logged beside it.
finish() {
  local rc="$1"
  if [[ $MEMORY_RC -ne 0 ]]; then
    if [[ $rc -eq $EXIT_DRIFT_UNPUBLISHED ]]; then
      log "exit $rc (drift unpublished); the memory pass also failed with $MEMORY_RC"
      exit "$rc"
    fi
    log "exit $MEMORY_RC (memory pass) in place of census exit $rc"
    exit "$MEMORY_RC"
  fi
  exit "$rc"
}

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

# OMN-18769: write the census-OBSERVED document BEFORE the no-drift early exit,
# so the clean case -- the one the drift topic structurally cannot carry --
# is available to the caller. This block never changes this script's exit-code
# contract: it writes a document and nothing else.
#
# THIS SCRIPT DOES NOT PUBLISH IT, and that is the correction rather than an
# omission. The first revision of OMN-18769 ended this block with
# `rpk topic produce --brokers "$KAFKA_BOOTSTRAP_SERVERS"`, and on the one host
# this script actually runs on that branch is unreachable twice over: `rpk` is
# not on the .201 host PATH (it lives inside the broker container -- the live
# log has read `rpk not found` on every run since the drift event was added),
# and the hourly drop-in sets no broker address. A publish that can only warn
# is not a mechanism, it is a comment that runs. Publication is the caller's,
# beside the declared transport: `lane-census-refresh.yml` runs
# `scripts/ci/publish_lab_fact_event.py`, which resolves protocol and mechanism
# from the checked-in lane overlay and carries the SASL credential the .201
# broker's external listener has required since OMN-18012 Phase B.
if [[ -n "$OBSERVED_OUT" ]]; then
  OBSERVED_JSON="$(echo "$PLAN_JSON" | LANE_CENSUS_HOST="$HOST" "$LANE_CENSUS_PYTHON" "${SCRIPT_DIR}/lane_census_event.py" --observed)"
  if [[ "$OBSERVED_OUT" == "-" ]]; then
    printf '%s\n' "$OBSERVED_JSON"
  else
    mkdir -p "$(dirname "$OBSERVED_OUT")"
    printf '%s\n' "$OBSERVED_JSON" >"$OBSERVED_OUT"
    log "census-observed document written: $OBSERVED_OUT"
  fi
fi

if [[ "$HAS_DRIFT" != "True" ]]; then
  log "No lane drift. Desired == actual."
  finish 0
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
  finish 30
fi

# Publish to the bus through the broker container (OMN-20798). The container is
# a deployment fact the installer writes into the unit as
# LANE_MEMORY_BROKER_CONTAINER; there is no default and no host-side `rpk`
# fallback, so a host that names none fails here instead of logging "for manual
# replay" and exiting as if the alert had been delivered.
if [[ -z "${LANE_MEMORY_BROKER_CONTAINER:-}" ]]; then
  log "DRIFT event NOT published: LANE_MEMORY_BROKER_CONTAINER is unset (exit $EXIT_DRIFT_UNPUBLISHED). Drift event logged above for manual replay."
  finish "$EXIT_DRIFT_UNPUBLISHED"
fi
printf '%s\n' "$EVENT_JSON" >"$SCRATCH/drift-event.json"
if broker_produce "$DRIFT_TOPIC" "$SCRATCH/drift-event.json"; then
  log "published lane-census-drift event to $DRIFT_TOPIC via broker container $LANE_MEMORY_BROKER_CONTAINER"
else
  log "DRIFT event NOT published: produce to $DRIFT_TOPIC via broker container $LANE_MEMORY_BROKER_CONTAINER FAILED (exit $EXIT_DRIFT_UNPUBLISHED). Drift event logged above for manual replay."
  finish "$EXIT_DRIFT_UNPUBLISHED"
fi

# Fail-fast on drift (gates-block policy, no warn-only mode).
finish 30
