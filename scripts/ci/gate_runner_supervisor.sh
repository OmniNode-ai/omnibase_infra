#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# gate_runner_supervisor.sh — the container-side half of the sanctioned
# gate-runner entry point (OMN-17317). Never invoked directly by an operator;
# `scripts/ci/run_on_gate_runner.sh --detached` execs it inside
# `omninode-gate-runner`.
#
# WHY THIS EXISTS — the 2026-08-31 wedge (OMN-17317)
# --------------------------------------------------
# A governed pre-push run driven by a raw attached `docker exec` deadlocks
# FOREVER, silently, AFTER the suite has already finished and passed, if the
# exec client detaches at any point during the run.
#
# `pre_commit._run_single_hook` buffers a hook's ENTIRE output and writes it
# out in one burst after the hook exits. For a fail-closed full-suite
# escalation that burst is megabytes of per-test lines. The `docker exec`
# session's stdout is a kernel pipe whose only reader is the containerd shim;
# when the client goes away the shim holds the read end open and nothing
# drains it. The pipe fills to the kernel's 64 KiB limit and every subsequent
# write(2) blocks forever — measured live: 3 h 45 m of green tests, exit 0,
# and then `anon_pipe_write` with no timeout, no log line, and `git push`
# parked in `do_wait` behind a hook that could never exit. Recovery required
# draining the shim's read fd by hand (exactly 65 538 bytes).
#
# The defect is NOT slowness and NOT the fail-closed escalation. It is that a
# governed run's liveness depended on a live interactive attach. This script
# removes that dependency by construction:
#
#   1. fds 1 and 2 are re-pointed at a log file on the bind-mounted worktree
#      as the FIRST thing that happens, before any payload byte is produced.
#      An inherited pipe — drained, undrained or severed — is then not on any
#      write path, so it cannot wedge the writer and the log is durable
#      evidence rather than terminal scrollback.
#   2. The payload is bounded in WALL-CLOCK time and a timeout is LOUD: a
#      named status in the receipt and a banner in the log, never a silent
#      block.
#   3. A heartbeat file records the timestamp AND the log's byte count on a
#      fixed interval, so a monitor can tell alive-and-progressing from
#      alive-but-producing-nothing from dead — the distinction whose absence
#      made the incident unreadable and nearly got a healthy 3 h 45 m run
#      killed.
#   4. The exit status and a summary land in a RECEIPT file. The caller
#      pattern is poll-the-receipt; nothing upstream ever holds the pipe.
#   5. The exclusive heavy-suite slot (OMN-17221 / OMN-16968(a)(ii)) is taken
#      HERE, by the process that actually runs the suite, so it is held for
#      exactly the run's lifetime and is released by the kernel on death —
#      there is no stale-lock path and no way to run heavy work through this
#      entry point without holding it.
#
# Usage (all flags required except where noted):
#   gate_runner_supervisor.sh --run-dir DIR --timeout SEC
#                             [--slot-lock PATH | --no-slot]
#                             [--label NAME] [--heartbeat-interval SEC]
#                             -- COMMAND [ARG...]
#
# Exit codes (also recorded in the receipt's `status`):
#   0..123  the payload's own status, propagated verbatim  -> passed/failed
#   124     wall-clock timeout                             -> timeout
#   2       supervisor setup failure                       -> (no receipt)
#   4       exclusive slot already held                    -> refused
#   5       a required capability could not be resolved    -> refused
#
# Fail-closed: a supervisor that cannot establish its own guarantees refuses.
# A gate that cannot run must be indistinguishable from a failing gate.
set -euo pipefail

RECEIPT_SCHEMA="onex.gate_runner.receipt.v1"

die() {
  echo "gate_runner_supervisor: $1" >&2
  if [ "$#" -gt 1 ]; then
    echo "  remediation: $2" >&2
  fi
  exit 2
}

RUN_DIR=""
TIMEOUT_SECONDS=""
SLOT_LOCK=""
SLOT_REQUESTED=1
LABEL=""
HEARTBEAT_INTERVAL="${GATE_RUNNER_HEARTBEAT_INTERVAL:-15}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="${2:-}"; shift 2 ;;
    --slot-lock) SLOT_LOCK="${2:-}"; SLOT_REQUESTED=1; shift 2 ;;
    --no-slot) SLOT_REQUESTED=0; SLOT_LOCK=""; shift ;;
    --label) LABEL="${2:-}"; shift 2 ;;
    --heartbeat-interval) HEARTBEAT_INTERVAL="${2:-}"; shift 2 ;;
    --) shift; break ;;
    *) die "unknown argument '$1'" "see the usage block at the top of this script" ;;
  esac
done

[ -n "${RUN_DIR}" ] || die "--run-dir is required" "the launcher creates the run directory and passes it here"
[ -n "${TIMEOUT_SECONDS}" ] || die "--timeout is required" "a governed run MUST be bounded in wall-clock time (OMN-17317 property 2)"
case "${TIMEOUT_SECONDS}" in
  '' | *[!0-9]*) die "--timeout must be a whole number of seconds, got '${TIMEOUT_SECONDS}'" "pass e.g. --timeout 14400" ;;
esac
case "${HEARTBEAT_INTERVAL}" in
  '' | *[!0-9]* | 0) die "--heartbeat-interval must be a positive whole number of seconds, got '${HEARTBEAT_INTERVAL}'" "pass e.g. --heartbeat-interval 15" ;;
esac
[ "$#" -gt 0 ] || die "no command given after --" "gate_runner_supervisor.sh ... -- uv run pytest tests/unit -q"
if [ "${SLOT_REQUESTED}" -eq 1 ] && [ -z "${SLOT_LOCK}" ]; then
  die "--slot-lock PATH is required unless --no-slot is given" "the launcher passes the container-wide slot lock path"
fi

mkdir -p "${RUN_DIR}" || die "could not create run directory '${RUN_DIR}'" "point --run-dir at a writable path on the bind-mounted worktree"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
LOG="${RUN_DIR}/run.log"
HEARTBEAT="${RUN_DIR}/heartbeat"
RECEIPT="${RUN_DIR}/receipt.json"

# ---------------------------------------------------------------------------
# THE FIX (OMN-17317 property 1). Everything above this line writes to the
# inherited fds and is bounded to a few hundred bytes of setup diagnostics.
# Everything below — including every byte the payload produces — goes to the
# log file. The inherited stdout/stderr are not on any write path from here
# on, so an undrained, unread or severed exec pipe cannot block this process.
# ---------------------------------------------------------------------------
: > "${LOG}" || die "could not open the run log '${LOG}' for writing" "point --run-dir at a writable path on the bind-mounted worktree"
exec >> "${LOG}" 2>&1

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
COMMAND_ARGV=("$@")

# json_escape STRING — minimal RFC 8259 string-body escaper. Deliberately
# interpreter-free: this script runs inside a container whose contract is "git
# + uv + a POSIX userland", and making the receipt writer depend on a Python
# interpreter would make the evidence path fail in exactly the degraded
# conditions the receipt exists to describe.
json_escape() {
  printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/\\t/g' -e 's/\r/\\r/g' | awk 'NR>1{printf "\\n"} {printf "%s", $0}'
}

json_argv() {
  local first=1 arg
  printf '['
  for arg in "${COMMAND_ARGV[@]}"; do
    [ "${first}" -eq 1 ] || printf ','
    first=0
    printf '"%s"' "$(json_escape "${arg}")"
  done
  printf ']'
}

# write_receipt STATUS EXIT_CODE ENDED_AT REASON — atomic (tmp + mv) so a
# poller never reads a half-written receipt.
write_receipt() {
  local status="$1" exit_code="$2" ended_at="$3" reason="$4" tmp
  tmp="${RECEIPT}.tmp.$$"
  {
    printf '{'
    printf '"schema":"%s",' "${RECEIPT_SCHEMA}"
    printf '"run_id":"%s",' "$(json_escape "$(basename "${RUN_DIR}")")"
    printf '"label":"%s",' "$(json_escape "${LABEL}")"
    printf '"status":"%s",' "${status}"
    printf '"exit_code":%s,' "${exit_code}"
    printf '"reason":"%s",' "$(json_escape "${reason}")"
    printf '"started_at":"%s",' "${STARTED_AT}"
    printf '"ended_at":%s,' "$( [ -n "${ended_at}" ] && printf '"%s"' "${ended_at}" || printf 'null' )"
    printf '"timeout_seconds":%s,' "${TIMEOUT_SECONDS}"
    printf '"slot":"%s",' "$( [ "${SLOT_REQUESTED}" -eq 1 ] && printf 'held' || printf 'skipped' )"
    printf '"slot_lock":%s,' "$( [ -n "${SLOT_LOCK}" ] && printf '"%s"' "$(json_escape "${SLOT_LOCK}")" || printf 'null' )"
    printf '"run_dir":"%s",' "$(json_escape "${RUN_DIR}")"
    printf '"log":"%s",' "$(json_escape "${LOG}")"
    printf '"heartbeat":"%s",' "$(json_escape "${HEARTBEAT}")"
    printf '"container_hostname":"%s",' "$(json_escape "$(hostname 2>/dev/null || echo unknown)")"
    printf '"supervisor_pid":%s,' "$$"
    printf '"command":%s' "$(json_argv)"
    printf '}\n'
  } > "${tmp}"
  mv -f "${tmp}" "${RECEIPT}"
}

# ---------------------------------------------------------------------------
# Exclusive heavy-suite slot (OMN-17221 / OMN-16968 option (a)(ii)).
# ---------------------------------------------------------------------------
# Held by THIS process for exactly the payload's lifetime. flock(2) is
# released by the kernel when the fd closes, so an OOM-killed or SIGKILLed
# run cannot leave a stale slot behind — the failure mode the host-side
# `~/push-lanes` queue has to reason about with pids.
if [ "${SLOT_REQUESTED}" -eq 1 ]; then
  if ! command -v flock > /dev/null 2>&1; then
    echo "gate_runner_supervisor: flock(1) is not available, so the exclusive heavy-suite slot cannot be taken." >&2
    echo "  remediation: run heavy work inside omninode-gate-runner (which has flock), or pass --no-slot for a bounded light run." >&2
    write_receipt refused 5 "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "flock_unavailable"
    exit 5
  fi
  mkdir -p "$(dirname "${SLOT_LOCK}")" 2> /dev/null || true
  if ! exec 9>> "${SLOT_LOCK}"; then
    write_receipt refused 5 "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "slot_lock_unwritable"
    exit 5
  fi
  if ! flock -n 9; then
    echo "gate_runner_supervisor: REFUSED — the exclusive heavy-suite slot '${SLOT_LOCK}' is already held by another run." >&2
    echo "  remediation: poll the holding run's receipt, or re-launch when it completes. Do NOT start a second heavy suite." >&2
    write_receipt refused 4 "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "slot_held"
    exit 4
  fi
fi

# ---------------------------------------------------------------------------
# Heartbeat (OMN-17317 property 3).
# ---------------------------------------------------------------------------
# Timestamp AND log byte count: mtime alone proves the supervisor is alive,
# the byte count proves the PAYLOAD is still producing. A run whose heartbeat
# advances while log_bytes stands still is alive-but-stalled, which is a
# different diagnosis from both "wedged" and "healthy".
#
# THE HEARTBEAT MUST DIE WITH THE SUPERVISOR. A heartbeat that outlives its
# run does not merely leak — it LIES, and it lies in the one direction that
# matters: "timestamp frozen is dead" is the whole diagnostic contract this
# file exists to establish, and an orphaned loop keeps stamping a fresh
# timestamp onto a run that is already gone. A monitor would then read a dead
# run as healthy forever, which is strictly worse than shipping no heartbeat.
#
# It is also the OMN-16995 orphan class, which this repo has already paid for
# once: nineteen leaked background loops from a unit test accumulated on `.200`
# and drove it to 1.64x-core load, at which the governed pre-push selector
# refuses every heavy escalation. Leaking that class ONTO THE GATE HOST from
# the gate-runner entry point itself would be self-defeating.
#
# `trap ... EXIT` alone does NOT close this: bash runs no EXIT trap for
# SIGTERM/SIGINT/SIGHUP, and none at all for SIGKILL. Three independent
# defenses, matching the pattern tests/unit/scripts/test_heavy_lock.py pins for
# the same class:
#   1. cleanup() reaps the loop, trapped on EXIT *and* on the three catchable
#      signals — so a terminated supervisor still reaps;
#   2. the loop checks its supervisor is still alive each tick (`kill -0 $$`;
#      `$$` inside a bash subshell is the PARENT's pid, which is exactly the
#      supervisor pid the receipt records) — so even SIGKILL, which runs no
#      trap anywhere, is survivable: the loop notices within one interval;
#   3. the loop carries its own absolute deadline, bounded by the run's own
#      wall-clock bound — so a pid-recycle race cannot produce an immortal
#      loop either.
HEARTBEAT_PID=""
heartbeat_loop() {
  local size tmp supervisor_pid deadline
  supervisor_pid="$$"
  tmp="${HEARTBEAT}.tmp.${supervisor_pid}"
  deadline=$(($(date +%s) + TIMEOUT_SECONDS + 300))
  while kill -0 "${supervisor_pid}" 2> /dev/null; do
    [ "$(date +%s)" -lt "${deadline}" ] || break
    size="$(wc -c < "${LOG}" 2> /dev/null | tr -d ' ')"
    printf '%s log_bytes=%s supervisor_pid=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${size:-0}" "${supervisor_pid}" > "${tmp}" 2> /dev/null \
      && mv -f "${tmp}" "${HEARTBEAT}" 2> /dev/null
    sleep "${HEARTBEAT_INTERVAL}"
  done
  rm -f "${tmp}" 2> /dev/null || true
}

cleanup() {
  if [ -n "${HEARTBEAT_PID}" ]; then
    kill "${HEARTBEAT_PID}" 2> /dev/null || true
    wait "${HEARTBEAT_PID}" 2> /dev/null || true
    HEARTBEAT_PID=""
  fi
}
# EXIT alone is not enough: bash skips it for SIGTERM/SIGINT/SIGHUP.
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 129' HUP

write_receipt running 0 "" "started"
heartbeat_loop &
HEARTBEAT_PID=$!

# ---------------------------------------------------------------------------
# Wall-clock bound (OMN-17317 property 2).
# ---------------------------------------------------------------------------
# `--kill-after` guarantees the bound holds even against a payload that
# ignores SIGTERM; without it "bounded" would be an intention rather than a
# property.
timeout_cmd=""
if command -v timeout > /dev/null 2>&1; then
  timeout_cmd="timeout"
elif command -v gtimeout > /dev/null 2>&1; then
  timeout_cmd="gtimeout"
else
  echo "gate_runner_supervisor: neither timeout(1) nor gtimeout(1) is available, so the run cannot be bounded in wall-clock time." >&2
  echo "  remediation: install GNU coreutils on this execution host; an unbounded governed run is the OMN-17317 defect itself." >&2
  write_receipt refused 5 "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "timeout_unavailable"
  exit 5
fi

echo "=== gate-runner run ${RUN_DIR##*/} started ${STARTED_AT} (timeout ${TIMEOUT_SECONDS}s) ==="
rc=0
"${timeout_cmd}" --signal=TERM --kill-after=60 "${TIMEOUT_SECONDS}" "${COMMAND_ARGV[@]}" || rc=$?
ENDED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

status="passed"
reason="completed"
if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
  status="timeout"
  reason="wall_clock_timeout_${TIMEOUT_SECONDS}s"
  echo "=== TIMEOUT: the run exceeded its ${TIMEOUT_SECONDS}s wall-clock bound and was terminated ==="
  echo "=== remediation: re-run with a larger --timeout, or narrow the selection; this is a LOUD abort, not a hang ==="
elif [ "${rc}" -ne 0 ]; then
  status="failed"
  reason="exit_${rc}"
fi

echo "=== gate-runner run ${RUN_DIR##*/} ${status} rc=${rc} ended ${ENDED_AT} ==="
write_receipt "${status}" "${rc}" "${ENDED_AT}" "${reason}"
exit "${rc}"
