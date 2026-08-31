#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# run_on_gate_runner.sh — THE sanctioned entry point for running governed
# pre-push and bounded ad-hoc test work inside the `.201` gate-runner container
# (OMN-16295 surface, landed by OMN-16752, made detachment-safe and
# slot-governed by OMN-17317).
#
# WHY THIS EXISTS
# ---------------
# docker/docker-compose.gate-runner.yml has always documented this script as
# the invocation path ("scripts/ci/run_on_gate_runner.sh sets
# UV_PROJECT_ENVIRONMENT per-invocation relative to the worktree it targets"),
# and scripts/hooks/prepush_smart_tests.sh routes operators to the gate-runner
# when the local host is over its load threshold. Before OMN-16752 the script
# did not exist at all; before OMN-17317 it only offered an ATTACHED
# `docker exec`, which is the shape that wedges (see below). Every gate-runner
# run therefore had to be hand-rolled, which is exactly the kind of
# undocumented per-operator recipe that makes a gate unreproducible.
#
# WHAT IT GUARANTEES
# ------------------
# 1. ONE venv per worktree, derived deterministically from the worktree's real
#    path, under a CONTAINER-ONLY prefix. Two worktrees of the same repo can
#    never share a venv (divergent dependency sets resolve into one another),
#    and a container venv can never collide with a host-built `.venv` whose
#    interpreter path/ABI differs. Raw `docker exec` lands on the venv ROOT
#    instead and re-syncs it under every sibling lane's live run (OMN-17222) —
#    routing through this script is what prevents that.
# 2. The SYMLINK TRAP is handled once, here. On `.201`,
#    `/home/jonah/Code/omni_home` is a symlink to `/data/omninode/omni_home`,
#    and `git worktree add` stores the REAL path in each worktree's `.git`
#    pointer file. A bind mount at the symlink path leaves every worktree's
#    `.git` pointer dangling inside the container ("fatal: not a git
#    repository"), which the hook then reports as "not inside a git worktree" —
#    a confusing failure two layers away from its cause. Every path this script
#    handles is resolved with `readlink -f` before it crosses the boundary.
# 3. DETACHED execution that cannot wedge (OMN-17317, `--detached`). An
#    ATTACHED `docker exec` couples the run's liveness to a live interactive
#    client: `pre_commit` buffers a hook's whole output and writes it in one
#    burst after the hook exits, the exec session's stdout is a 64 KiB kernel
#    pipe whose only reader is the containerd shim, and if the client detaches
#    (a dockerd bounce, an ssh drop, a session ending) nothing drains it — so
#    the write blocks FOREVER, after the suite has already finished and passed.
#    Measured live 2026-08-31: 3 h 45 m green, exit 0, then a permanent
#    `anon_pipe_write` with `git push` parked behind it and zero diagnostics.
#    `--detached` gives the launcher a run-id and a tail command instead of a
#    stream; the run's output goes to a log file on the bind mount and its
#    verdict to a receipt file. THE CALLER PATTERN IS POLL-THE-RECEIPT, NEVER
#    HOLD-THE-PIPE.
# 4. LOAD-AWARE ADMISSION against the CONTAINER's own cgroup, not the bare
#    host's loadavg (OMN-16446 finding 5: the host can read "HAS capacity" at
#    32 cores while this 4-CPU/8 GiB container is saturated). It uses the same
#    `PREPUSH_LOAD_THRESHOLD` knob and the same busy/limit ratio semantics the
#    governed selector's `host_is_fit()` uses, applied to the correct scope.
#    Refusals are TYPED and immediate; nothing is ever queued silently.
# 5. An EXCLUSIVE heavy-suite slot on the container path (OMN-17221, the
#    implementation of OMN-16968 option (a)(ii)). The slot is taken by the
#    supervisor that actually runs the suite, so it is held for exactly the
#    run's lifetime and released by the kernel on death.
# 6. NO OPERATOR-SPECIFIC PATHS. All run state lives under the target
#    worktree's own `.onex_state/`, and the slot lock lives inside the
#    container — so any account in the `docker` group (jonah, lakshman;
#    OMN-17280 contractor-access chain) can use this identically. Nothing here
#    reads `~/push-lanes`, which is mode 0750 and unreadable to a second
#    account.
#
# Usage:
#   run_on_gate_runner.sh [--sync] [--detached [--timeout SEC] [--no-slot]
#                          [--label NAME] [--wait [--wait-timeout SEC]]]
#                         <worktree-path> <command> [args...]
#   run_on_gate_runner.sh --status <run-dir>
#   run_on_gate_runner.sh --wait <run-dir> [--wait-timeout SEC]
#
#   --sync        run `uv sync --all-extras` in the worktree first (needed the
#                 first time a worktree is seen, and after a dependency change).
#   --detached    launch under scripts/ci/gate_runner_supervisor.sh inside the
#                 container, fully detached; print a run-id, a tail command and
#                 a poll command, then exit 0. THIS IS THE MODE FOR A GOVERNED
#                 PRE-PUSH OR ANY RUN LONGER THAN A FEW MINUTES.
#   --timeout     wall-clock bound for a detached run (default 14400 = 4h).
#   --no-slot     do not take the exclusive heavy-suite slot. Only legal for a
#                 BOUNDED ad-hoc run: it forces --timeout <= 1800. Recorded in
#                 the receipt as `"slot":"skipped"`; never silent.
#   --label       short name recorded in the run-id and the receipt.
#   --wait        poll the receipt to completion and exit with the run's own
#                 status. Polling, not piping — safe to interrupt at any time.
#
# Examples:
#   # governed pre-push, detached, 4h bound, exclusive slot:
#   scripts/ci/run_on_gate_runner.sh --sync --detached --label omn1234-prepush \
#     "$OMNI_HOME/omni_worktrees/OMN-1234/omnimarket" git push
#
#   # bounded ad-hoc test run, no slot, 20 minute bound:
#   scripts/ci/run_on_gate_runner.sh --detached --no-slot --timeout 1200 \
#     --label omn1234-unit "$OMNI_HOME/omni_worktrees/OMN-1234/omnimarket" \
#     uv run pytest tests/unit -q
#
#   # attached, for a fast interactive probe only (NEVER for a heavy suite):
#   scripts/ci/run_on_gate_runner.sh "$OMNI_HOME/omni_worktrees/OMN-1234/omnimarket" \
#     uv run pytest tests/unit/test_one.py -q
#
# Environment:
#   GATE_RUNNER_CONTAINER   container name (default: omninode-gate-runner)
#   GATE_RUNNER_SSH_TARGET  when set, `docker` is invoked over ssh against this
#                           target instead of locally — lets `.200` drive the
#                           `.201` container directly. Default: empty (local).
#   GATE_RUNNER_VENV_ROOT   container-only venv prefix
#                           (default: /workspace/.venv-gate-runner)
#   GATE_RUNNER_SLOT_LOCK   container path of the exclusive heavy-suite slot
#                           (default: /workspace/.gate-runner/SLOT.lock)
#   PREPUSH_LOAD_THRESHOLD  busy/limit ratio at or under which the container
#                           counts as fit (default 1.0 — the same knob and the
#                           same default the governed selector uses).
#   GATE_RUNNER_MEM_THRESHOLD  memory.current/memory.max ceiling (default 0.90)
#
# Exit codes:
#   0        attached: the command succeeded. detached: the run was LAUNCHED
#            (its own verdict lands in the receipt).
#   1..123   attached: the command's own status, propagated verbatim.
#   2        setup failure — always fail-closed. A gate that cannot run must be
#            indistinguishable from a failing gate.
#   3        REFUSED_LOAD — the container is over its admission threshold.
#   4        REFUSED_SLOT — the exclusive heavy-suite slot is already held, or a
#            foreign heavy pre-push is running on the host.
#   5        REFUSED_PROBE — admission could not be measured. Fails CLOSED.
set -euo pipefail

GATE_RUNNER_CONTAINER="${GATE_RUNNER_CONTAINER:-omninode-gate-runner}"
GATE_RUNNER_SSH_TARGET="${GATE_RUNNER_SSH_TARGET:-}"
GATE_RUNNER_VENV_ROOT="${GATE_RUNNER_VENV_ROOT:-/workspace/.venv-gate-runner}"
GATE_RUNNER_SLOT_LOCK="${GATE_RUNNER_SLOT_LOCK:-/workspace/.gate-runner/SLOT.lock}"
PREPUSH_LOAD_THRESHOLD="${PREPUSH_LOAD_THRESHOLD:-1.0}"
GATE_RUNNER_MEM_THRESHOLD="${GATE_RUNNER_MEM_THRESHOLD:-0.90}"

# Exit-code names, used verbatim in the typed refusal lines so a caller can
# grep for the reason rather than parsing prose.
readonly RC_SETUP=2
readonly RC_REFUSED_LOAD=3
readonly RC_REFUSED_SLOT=4
readonly RC_REFUSED_PROBE=5

# Default wall-clock bound for a detached run. 4h: the longest healthy
# omnibase_core full suite measured on this container is 3h45m (OMN-17317), so
# a run past 4h is not slow-but-fine, it is a fault worth aborting loudly.
DEFAULT_TIMEOUT_SECONDS=14400
# A run that declines the exclusive slot must be genuinely bounded — 30
# minutes. This is the mechanical form of "--no-slot is for ad-hoc work", so
# it cannot quietly become the way heavy suites skip serialization.
NO_SLOT_MAX_TIMEOUT_SECONDS=1800

die() {
  echo "run_on_gate_runner: $1" >&2
  if [ "$#" -gt 1 ]; then
    echo "  remediation: $2" >&2
  fi
  exit "${RC_SETUP}"
}

refuse() {
  local rc="$1" code="$2" msg="$3" remediation="$4"
  echo "run_on_gate_runner: REFUSED ${code} — ${msg}" >&2
  echo "  remediation: ${remediation}" >&2
  exit "${rc}"
}

DO_SYNC=0
DETACHED=0
WANT_SLOT=1
TIMEOUT_SECONDS=""
LABEL=""
DO_WAIT=0
WAIT_TIMEOUT_SECONDS=0
MODE="run"
STATUS_TARGET=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --sync) DO_SYNC=1; shift ;;
    --detached | -d) DETACHED=1; shift ;;
    --no-slot) WANT_SLOT=0; shift ;;
    --timeout) TIMEOUT_SECONDS="${2:-}"; shift 2 ;;
    --label) LABEL="${2:-}"; shift 2 ;;
    --wait-timeout) WAIT_TIMEOUT_SECONDS="${2:-}"; shift 2 ;;
    --status)
      MODE="status"
      STATUS_TARGET="${2:-}"
      shift 2
      ;;
    --wait)
      # `--wait <run-dir>` is the standalone poller; a bare `--wait` alongside
      # --detached means "launch, then poll".
      if [ -n "${2:-}" ] && [ "${2#-}" = "${2}" ] && [ "${MODE}" = "run" ] && [ "${DETACHED}" -eq 0 ]; then
        MODE="wait"
        STATUS_TARGET="$2"
        shift 2
      else
        DO_WAIT=1
        shift
      fi
      ;;
    --) shift; break ;;
    -*) die "unknown option '$1'" "see the usage block at the top of this script" ;;
    *) break ;;
  esac
done

# ---------------------------------------------------------------------------
# Transport. Kept as an array so no argument is ever re-split (the zsh/bash
# word-splitting class this repo's shell-hygiene gate exists for).
# ---------------------------------------------------------------------------
docker_cmd=(docker)
if [ -n "${GATE_RUNNER_SSH_TARGET}" ]; then
  docker_cmd=(ssh -n -o BatchMode=yes "${GATE_RUNNER_SSH_TARGET}" docker)
fi

require_container_running() {
  if ! "${docker_cmd[@]}" inspect -f '{{.State.Running}}' "${GATE_RUNNER_CONTAINER}" 2> /dev/null | grep -qx true; then
    die "container '${GATE_RUNNER_CONTAINER}' is not running" \
      "bring it up: OMNI_HOME=\"\$(readlink -f \"\$OMNI_HOME\")\" docker compose -f docker/docker-compose.gate-runner.yml up -d"
  fi
}

# read_receipt RUN_DIR — prints the receipt JSON, or nothing if it is absent.
read_receipt() {
  "${docker_cmd[@]}" exec "${GATE_RUNNER_CONTAINER}" cat "$1/receipt.json" 2> /dev/null || true
}

# receipt_field JSON FIELD — extracts one scalar. Deliberately not a JSON
# parser: the receipt is written by gate_runner_supervisor.sh with escaped
# scalars and a flat shape, and depending on `jq` (absent from the container)
# would make the evidence path fail in the degraded conditions the receipt
# exists to describe.
receipt_field() {
  printf '%s' "$1" | sed -n "s/.*\"$2\":\"\\([^\"]*\\)\".*/\\1/p"
}

receipt_number() {
  printf '%s' "$1" | sed -n "s/.*\"$2\":\\([0-9-][0-9]*\\).*/\\1/p"
}

# poll_receipt RUN_DIR TIMEOUT_SECONDS — polls until the receipt reaches a
# terminal status, then exits with the run's own exit code. Poll, never pipe:
# this loop holds no fd belonging to the run, so interrupting it, losing the
# ssh connection or killing the shell cannot affect the run at all.
poll_receipt() {
  local run_dir="$1" budget="$2" waited=0 json status
  while :; do
    json="$(read_receipt "${run_dir}")"
    status="$(receipt_field "${json}" status)"
    case "${status}" in
      passed | failed | timeout | refused)
        printf '%s\n' "${json}"
        exit "$(receipt_number "${json}" exit_code)"
        ;;
    esac
    if [ "${budget}" -gt 0 ] && [ "${waited}" -ge "${budget}" ]; then
      echo "run_on_gate_runner: still ${status:-unknown} after ${waited}s; the RUN IS UNAFFECTED — poll again with --status ${run_dir}" >&2
      exit 0
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

if [ "${MODE}" = "status" ]; then
  [ -n "${STATUS_TARGET}" ] || die "--status needs a run directory" "pass the run-dir printed by the detached launch"
  require_container_running
  json="$(read_receipt "${STATUS_TARGET}")"
  [ -n "${json}" ] || die "no receipt at '${STATUS_TARGET}/receipt.json'" \
    "check the run-dir path, or tail '${STATUS_TARGET}/run.log' and '${STATUS_TARGET}/heartbeat'"
  printf '%s\n' "${json}"
  exit 0
fi

if [ "${MODE}" = "wait" ]; then
  [ -n "${STATUS_TARGET}" ] || die "--wait needs a run directory" "pass the run-dir printed by the detached launch"
  require_container_running
  poll_receipt "${STATUS_TARGET}" "${WAIT_TIMEOUT_SECONDS}"
fi

if [ "$#" -lt 2 ]; then
  die "usage: run_on_gate_runner.sh [--sync] [--detached ...] <worktree-path> <command> [args...]" \
    "pass the worktree to run in, then the command to run there"
fi

WORKTREE_RAW="$1"
shift

# Resolve the symlink trap described in the header. `readlink -f` is GNU/macOS
# 12.3+; both designated hosts have it.
WORKTREE="$(readlink -f "${WORKTREE_RAW}" 2> /dev/null || true)"
[ -n "${WORKTREE}" ] || die "could not resolve worktree path '${WORKTREE_RAW}'" \
  "pass an existing path; it is resolved with 'readlink -f' before use"
[ -d "${WORKTREE}" ] || die "worktree '${WORKTREE}' is not a directory" \
  "create it first: git -C \"\$OMNI_HOME/<repo>\" worktree add <path> -b <branch>"

if [ "${DETACHED}" -eq 0 ]; then
  [ "${WANT_SLOT}" -eq 1 ] || die "--no-slot only applies to a detached run" "add --detached, or drop --no-slot"
  [ -z "${TIMEOUT_SECONDS}" ] || die "--timeout only applies to a detached run" \
    "add --detached; an attached run is bounded by the operator's own terminal, which is exactly the coupling OMN-17317 removes"
  [ "${DO_WAIT}" -eq 0 ] || die "--wait only applies to a detached run" "add --detached, or drop --wait"

  # The attached path is the wedge shape itself: its stdout IS the `docker
  # exec` pipe whose only reader is the containerd shim, so a heavy payload
  # under a client that detaches reproduces OMN-17317 exactly — through this
  # script rather than around it. Keeping "attached is for fast probes only"
  # as a COMMENT would leave the sanctioned entry point as a working way to
  # reproduce the defect it exists to remove (repo rule 5: enforcement, not
  # detection). Matched on the payload's own argv, so it cannot be argued
  # past; the remediation names the exact flags that make the run safe.
  for _arg in "$@"; do
    case "${_arg}" in
      pytest | */pytest | prepush_smart_tests.sh | */prepush_smart_tests.sh)
        die "refusing to run '${_arg}' in ATTACHED mode — this is the OMN-17317 wedge shape" \
          "heavy work must be detached so its liveness does not depend on this terminal: re-run with '--detached --timeout <sec>' and poll the receipt with '--status <run-dir>'. For a genuinely short probe use '--detached --no-slot --timeout 600'"
        ;;
      push)
        # Only when it is `git push`, never a stray argument that happens to
        # be the word "push".
        case " $* " in
          *" git push "*)
            die "refusing to run 'git push' in ATTACHED mode — the pre-push hook's buffered output is what wedges (OMN-17317)" \
              "re-run with '--detached --timeout <sec>' and poll the receipt with '--status <run-dir>'"
            ;;
        esac
        ;;
    esac
  done
fi

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-${DEFAULT_TIMEOUT_SECONDS}}"
case "${TIMEOUT_SECONDS}" in
  '' | *[!0-9]* | 0) die "--timeout must be a positive whole number of seconds, got '${TIMEOUT_SECONDS}'" "pass e.g. --timeout 14400" ;;
esac
if [ "${WANT_SLOT}" -eq 0 ] && [ "${TIMEOUT_SECONDS}" -gt "${NO_SLOT_MAX_TIMEOUT_SECONDS}" ]; then
  die "--no-slot requires --timeout <= ${NO_SLOT_MAX_TIMEOUT_SECONDS} (got ${TIMEOUT_SECONDS})" \
    "--no-slot exists for BOUNDED ad-hoc work. A run that needs longer is heavy work and must take the exclusive slot (OMN-17221)"
fi

require_container_running

# Per-worktree venv, keyed on a slug of the REAL worktree path so it is stable
# across invocations and unique across worktrees. Non-alphanumerics collapse to
# '-' so the result is always a valid single path segment.
venv_slug="$(printf '%s' "${WORKTREE}" | tr -c '[:alnum:]' '-' | sed -e 's/--*/-/g' -e 's/^-//' -e 's/-$//')"
venv_path="${GATE_RUNNER_VENV_ROOT}/${venv_slug}"

if [ "${DO_SYNC}" -eq 1 ]; then
  echo "run_on_gate_runner: syncing venv ${venv_path} for ${WORKTREE}" >&2
  "${docker_cmd[@]}" exec \
    -e "UV_PROJECT_ENVIRONMENT=${venv_path}" \
    -w "${WORKTREE}" \
    "${GATE_RUNNER_CONTAINER}" \
    uv sync --all-extras \
    || die "uv sync failed in ${WORKTREE}" "inspect the output above; the venv is at ${venv_path}"
fi

if [ "${DETACHED}" -eq 0 ]; then
  # ATTACHED. Preserved verbatim for fast interactive probes; the command's own
  # status is propagated so this stays safe to use as a gate. Anything that can
  # outlive a terminal must use --detached (OMN-17317).
  exec "${docker_cmd[@]}" exec \
    -e "UV_PROJECT_ENVIRONMENT=${venv_path}" \
    -w "${WORKTREE}" \
    "${GATE_RUNNER_CONTAINER}" \
    "$@"
fi

# ===========================================================================
# DETACHED PATH (OMN-17317)
# ===========================================================================

# --- Admission: container cgroup, not the bare host (OMN-16446 finding 5) ---
#
# The `.201` HOST has 32 cores; this container is hard-capped at 4 CPUs / 8 GiB
# by docker-compose.gate-runner.yml. Reading `/proc/loadavg` on the host — what
# the pre-push guard's `.201` capacity check does — can report "HAS capacity"
# while the container it is recommending is fully saturated, which is the
# documented container-starvation trap. This probe reads the container's own
# cgroup v2 accounting: a one-second `cpu.stat` delta against the `cpu.max`
# quota, and `memory.current` against `memory.max`.
#
# POSIX and single-quote-free: it is handed to `sh -c` inside the container.
# shellcheck disable=SC2016  # intentionally unexpanded: evaluated by the container's shell.
_CGROUP_PROBE_SH='cg=/sys/fs/cgroup
[ -r $cg/cpu.max ] || exit 1
[ -r $cg/cpu.stat ] || exit 1
q=$(cut -d" " -f1 $cg/cpu.max)
p=$(cut -d" " -f2 $cg/cpu.max)
if [ "$q" = "max" ]; then q=$(getconf _NPROCESSORS_ONLN); p=1; fi
u1=$(awk "/^usage_usec/ {print \$2}" $cg/cpu.stat)
sleep 1
u2=$(awk "/^usage_usec/ {print \$2}" $cg/cpu.stat)
mm=max
mc=0
[ -r $cg/memory.max ] && mm=$(cat $cg/memory.max)
[ -r $cg/memory.current ] && mc=$(cat $cg/memory.current)
[ -n "$q" ] && [ -n "$p" ] && [ -n "$u1" ] && [ -n "$u2" ] || exit 1
printf "%s %s %s %s %s %s\n" "$q" "$p" "$u1" "$u2" "$mm" "$mc"'

probe_raw="$("${docker_cmd[@]}" exec "${GATE_RUNNER_CONTAINER}" sh -c "${_CGROUP_PROBE_SH}" 2> /dev/null || true)"
[ -n "${probe_raw}" ] || refuse "${RC_REFUSED_PROBE}" REFUSED_PROBE \
  "could not read container '${GATE_RUNNER_CONTAINER}' cgroup accounting, so its capacity is unproven" \
  "an unmeasurable target fails CLOSED (the same posture the pre-push load guard takes). Check 'docker exec ${GATE_RUNNER_CONTAINER} cat /sys/fs/cgroup/cpu.stat'"

# A here-string, NOT `set --`. `set --` would clobber the positional parameters,
# which at this point still hold the OPERATOR'S COMMAND — caught live on `.201`
# during this ticket's own proof run, where the launched receipt recorded
# `"command":["fit","0.000","0.143"]` and the payload exited 127.
IFS=' ' read -r cpu_quota cpu_period usage1 usage2 mem_max mem_cur <<< "${probe_raw}"

admission="$(awk -v q="${cpu_quota}" -v p="${cpu_period}" -v u1="${usage1}" -v u2="${usage2}" \
  -v mm="${mem_max}" -v mc="${mem_cur}" -v thr="${PREPUSH_LOAD_THRESHOLD}" -v mthr="${GATE_RUNNER_MEM_THRESHOLD}" '
BEGIN {
  cores = (p + 0 > 0) ? (q / p) : 0
  if (cores <= 0) { print "probe 0 0"; exit }
  used = (u2 - u1) / 1000000.0
  cpu_ratio = used / cores
  mem_ratio = (mm == "max" || mm + 0 <= 0) ? 0 : (mc / mm)
  verdict = "fit"
  if (cpu_ratio > thr + 0) verdict = "cpu"
  else if (mem_ratio > mthr + 0) verdict = "mem"
  printf "%s %.3f %.3f\n", verdict, cpu_ratio, mem_ratio
}')"
IFS=' ' read -r verdict cpu_ratio mem_ratio <<< "${admission}"

case "${verdict}" in
  probe)
    refuse "${RC_REFUSED_PROBE}" REFUSED_PROBE \
      "container '${GATE_RUNNER_CONTAINER}' reports a non-positive CPU quota, so its capacity is unproven" \
      "check 'docker exec ${GATE_RUNNER_CONTAINER} cat /sys/fs/cgroup/cpu.max'"
    ;;
  cpu)
    refuse "${RC_REFUSED_LOAD}" REFUSED_LOAD \
      "container '${GATE_RUNNER_CONTAINER}' is at ${cpu_ratio}x its ${cpu_quota}/${cpu_period} CPU quota, over the ${PREPUSH_LOAD_THRESHOLD}x admission threshold" \
      "wait for the in-flight run to finish (poll its receipt), or run on another lab host per docs/runbooks/lab-prepush-host-table.md. Nothing is queued here — this refusal is immediate and typed by design"
    ;;
  mem)
    refuse "${RC_REFUSED_LOAD}" REFUSED_LOAD \
      "container '${GATE_RUNNER_CONTAINER}' is at ${mem_ratio} of its memory limit, over the ${GATE_RUNNER_MEM_THRESHOLD} admission threshold" \
      "wait for the in-flight run to finish (poll its receipt), or run on another lab host per docs/runbooks/lab-prepush-host-table.md"
    ;;
  fit) : ;;
  *)
    refuse "${RC_REFUSED_PROBE}" REFUSED_PROBE \
      "admission verdict could not be computed from the container cgroup probe" \
      "re-run; if it persists, check 'docker exec ${GATE_RUNNER_CONTAINER} cat /sys/fs/cgroup/cpu.stat'"
    ;;
esac

# --- Admission: the same foreign-heavy-run signal the host queue's gate-1 uses
# (OMN-17221 / OMN-16968(a)(ii)) ---
#
# `~/push-lanes/queue-runner.sh` gates on `pgrep -f prepush_smart_tests\.sh`,
# but it is only ever called from INSIDE the queue loop, so a container run
# that never enqueued consults nothing — which is how two heavy omnibase_core
# suites ran concurrently on 2026-08-30. The host CAN see container runs by pid
# (proven: /proc/<host-pid>/status NSpid maps to the container pid), so the
# same cheap signal works from here; no new IPC is needed. This closes the
# container half of the gap. The `~/push-lanes` queue itself is host state
# outside any repo and is unchanged.
if [ "${WANT_SLOT}" -eq 1 ]; then
  pgrep_rc=0
  if [ -n "${GATE_RUNNER_SSH_TARGET}" ]; then
    foreign="$(ssh -n -o BatchMode=yes "${GATE_RUNNER_SSH_TARGET}" "pgrep -f 'prepush_smart_tests\\.sh'" 2> /dev/null | tr '\n' ' ')" || pgrep_rc=$?
  else
    foreign="$(pgrep -f 'prepush_smart_tests\.sh' 2> /dev/null | tr '\n' ' ')" || pgrep_rc=$?
  fi
  if [ "${pgrep_rc}" -gt 1 ]; then
    refuse "${RC_REFUSED_PROBE}" REFUSED_PROBE \
      "could not probe for foreign heavy pre-push runs (pgrep exited ${pgrep_rc})" \
      "an unmeasurable slot fails CLOSED. Run this from the .201 host, or set GATE_RUNNER_SSH_TARGET to reach it"
  fi
  foreign="$(printf '%s' "${foreign}" | sed -e 's/[[:space:]]*$//')"
  if [ -n "${foreign}" ]; then
    refuse "${RC_REFUSED_SLOT}" REFUSED_SLOT \
      "a heavy pre-push run is already active on this host (pids: ${foreign}) and the heavy-suite slot is exclusive" \
      "poll the active run to completion, then re-launch. Concurrent heavy suites starve each other — measured 0.11%/min throughput collapse (OMN-16968)"
  fi
fi

# --- Resolve the container-side supervisor ---
#
# Two candidates, in order. (1) The sibling of THIS script: correct whenever
# the invoking checkout is itself under the bind mount, which is the normal
# `.201` case and is also what lets a branch prove its own supervisor before
# it merges. (2) The canonical clone's copy under the mount root: the
# cross-host case, where this script runs from a `.200` checkout whose path
# does not exist inside the container. Neither resolving fails CLOSED.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MOUNT_ROOT="$("${docker_cmd[@]}" inspect \
  -f '{{range .Mounts}}{{if eq .Source .Destination}}{{.Destination}}{{end}}{{end}}' \
  "${GATE_RUNNER_CONTAINER}" 2> /dev/null || true)"

SUPERVISOR=""
for candidate in "${SCRIPT_DIR}/gate_runner_supervisor.sh" \
  "${MOUNT_ROOT:+${MOUNT_ROOT}/omnibase_infra/scripts/ci/gate_runner_supervisor.sh}"; do
  [ -n "${candidate}" ] || continue
  if "${docker_cmd[@]}" exec "${GATE_RUNNER_CONTAINER}" test -x "${candidate}" 2> /dev/null; then
    SUPERVISOR="${candidate}"
    break
  fi
done
[ -n "${SUPERVISOR}" ] || die "the container cannot see scripts/ci/gate_runner_supervisor.sh" \
  "the supervisor must be reachable inside '${GATE_RUNNER_CONTAINER}'. Run this script from a checkout under the container's bind mount (${MOUNT_ROOT:-<none>}), or update the canonical clone at ${MOUNT_ROOT:-<mount>}/omnibase_infra"

# --- Launch ---
label_slug="$(printf '%s' "${LABEL:-run}" | tr -c '[:alnum:]' '-' | sed -e 's/--*/-/g' -e 's/^-//' -e 's/-$//')"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${label_slug:-run}-$$"
RUN_DIR="${WORKTREE}/.onex_state/gate-runner/${RUN_ID}"

supervisor_args=(--run-dir "${RUN_DIR}" --timeout "${TIMEOUT_SECONDS}" --label "${LABEL:-${label_slug}}")
if [ "${WANT_SLOT}" -eq 1 ]; then
  supervisor_args+=(--slot-lock "${GATE_RUNNER_SLOT_LOCK}")
else
  supervisor_args+=(--no-slot)
fi

# `docker exec -d` attaches no stdio at all, and `setsid` puts the supervisor
# in its own session so a torn-down exec session cannot signal it. Between the
# two, there is no pipe for the run to write into and no client whose death
# the run can notice — the OMN-17317 wedge is structurally unreachable.
"${docker_cmd[@]}" exec -d \
  -e "UV_PROJECT_ENVIRONMENT=${venv_path}" \
  -w "${WORKTREE}" \
  "${GATE_RUNNER_CONTAINER}" \
  setsid "${SUPERVISOR}" "${supervisor_args[@]}" -- "$@" \
  || die "could not launch the detached run in '${GATE_RUNNER_CONTAINER}'" \
    "check 'docker logs ${GATE_RUNNER_CONTAINER}' and that the supervisor at ${SUPERVISOR} is executable"

# `docker exec -d` returns before the supervisor has done anything, so confirm
# liveness from the receipt rather than reporting a launch we have not seen.
waited=0
receipt_json=""
while [ "${waited}" -lt 60 ]; do
  receipt_json="$(read_receipt "${RUN_DIR}")"
  [ -z "${receipt_json}" ] || break
  sleep 2
  waited=$((waited + 2))
done
[ -n "${receipt_json}" ] || die "the detached run wrote no receipt at ${RUN_DIR}/receipt.json within ${waited}s" \
  "check 'docker exec ${GATE_RUNNER_CONTAINER} ls -la ${RUN_DIR}' — the supervisor refused before it could start, or the run dir is not writable"

launch_status="$(receipt_field "${receipt_json}" status)"
if [ "${launch_status}" = "refused" ]; then
  reason="$(receipt_field "${receipt_json}" reason)"
  case "${reason}" in
    slot_held)
      refuse "${RC_REFUSED_SLOT}" REFUSED_SLOT \
        "the exclusive heavy-suite slot '${GATE_RUNNER_SLOT_LOCK}' is held by another run" \
        "poll the holding run's receipt, then re-launch. Never start a second heavy suite"
      ;;
    *)
      refuse "${RC_REFUSED_PROBE}" REFUSED_PROBE \
        "the supervisor refused to start (${reason})" \
        "read ${RUN_DIR}/run.log for the named cause"
      ;;
  esac
fi

cat >&2 <<EOF
run_on_gate_runner: LAUNCHED (detached) — the launching session's death cannot affect this run.
  run id     : ${RUN_ID}
  run dir    : ${RUN_DIR}
  worktree   : ${WORKTREE}
  slot       : $( [ "${WANT_SLOT}" -eq 1 ] && printf 'held (%s)' "${GATE_RUNNER_SLOT_LOCK}" || printf 'skipped (bounded ad-hoc run, <= %ss)' "${NO_SLOT_MAX_TIMEOUT_SECONDS}" )
  timeout    : ${TIMEOUT_SECONDS}s (loud abort, not a hang)
  admission  : cpu ${cpu_ratio}x of ${cpu_quota}/${cpu_period}, mem ${mem_ratio}, threshold ${PREPUSH_LOAD_THRESHOLD}x
  tail log   : docker exec ${GATE_RUNNER_CONTAINER} tail -f ${RUN_DIR}/run.log
  heartbeat  : docker exec ${GATE_RUNNER_CONTAINER} cat ${RUN_DIR}/heartbeat
  poll status: scripts/ci/run_on_gate_runner.sh --status ${RUN_DIR}
  block on it: scripts/ci/run_on_gate_runner.sh --wait ${RUN_DIR}
POLL THE RECEIPT, NEVER HOLD THE PIPE — an attached stream is the OMN-17317 wedge.
EOF

if [ "${DO_WAIT}" -eq 1 ]; then
  poll_receipt "${RUN_DIR}" "${WAIT_TIMEOUT_SECONDS}"
fi

exit 0
