#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# reconcile-host.sh -- the one workspace reconciler, for every machine (OMN-17307)
# ============================================================================
#
# WHAT IT OWNS, AND WHAT IT DELIBERATELY DOES NOT
# ----------------------------------------------------------------------------
# This script owns ORDERING and PROOF. It owns no repair logic at all, and that
# is the point: there is exactly one clone reconciler and exactly one venv
# reconciler in this repo, and a third implementation of either would be the
# drift this file exists to end.
#
#   clone surface  ->  scripts/runtime_build/reconcile_deploy_clones.sh  (OMN-17291)
#   venv surface   ->  scripts/reconcile-workspace-venvs.sh              (OMN-17190)
#
# Around each delegate it does the thing neither of them can do for itself:
# observe the surface BEFORE, observe it AFTER, and compare the AFTER against an
# independently-established TARGET. A delegate that exits 0 without moving
# anything fails here.
#
# WHY THAT MATTERS -- the four incidents, one shape
# ----------------------------------------------------------------------------
# Every one of them was a surface that did not move while everything that could
# have noticed reported success:
#
#   * OMN-17291, `.201`: `omnibase_core` had `core.bare=true` on a clone WITH a
#     working tree. `git fetch` exited 0 forever; `git checkout` exited 128
#     forever. A sync loop reading the fetch's status called that clone healthy
#     for as long as it existed.
#   * OMN-17291 again: the dev lane then baked an omnimarket 11 commits behind
#     `origin/dev`, because the image's source ref is derived from that clone.
#   * OMN-17190, this Mac: the CLI venv drift self-heal was real and worked --
#     and was not the code that ran, because `uv run` silently resolved a
#     DIFFERENT `onex` off PATH.
#   * OMN-16932: a delegation probe ran against a build nobody chose and
#     produced a RECEIPT. Invalid evidence is worse than no evidence, because it
#     outlives the invocation.
#
# So the rule here is absolute and has no override: a step is judged by reading
# the surface back. `scripts/reconcile_verify_movement.py` holds the verdict
# table, and its `verdict()` takes no exit status at all -- there is no argument
# that turns "the command succeeded" into "the surface moved".
#
# THE TARGET IS ESTABLISHED HERE, NOT ASKED FOR
# ----------------------------------------------------------------------------
# This script fetches each clone itself before verdicting. It does NOT trust the
# delegate to have fetched, and it does not read the delegate's own receipt for
# the answer. A verifier that takes the target from the thing it is verifying is
# not a verifier. The fetch is read-only with respect to the working tree.
#
# UNCOVERED IS A FAILURE, NOT A SKIP
# ----------------------------------------------------------------------------
# If a delegate is absent, the surface is UNCOVERED: reported, alerted, non-zero.
# Silently skipping a surface nobody reconciles is precisely the OMN-17291
# condition ("`.201` is not in the reconciler's scope at all"), and a reconciler
# that quietly covers less than it claims is worse than one that is missing.
#
# ALERT ON UNPROVABLE MOVEMENT
# ----------------------------------------------------------------------------
# A failing verdict posts on the existing Slack path and writes NO floor and NO
# success line. Reporting success for an unproven surface is the failure mode;
# staying quiet about a failure is the second-worst outcome, so it does both:
# non-zero exit AND an alert.
#
# THE FLOOR
# ----------------------------------------------------------------------------
# It stamps ${OMNI_HOME}/.onex-workspace-floor.json -- the minimum installed
# state of the DISPATCH venv that has been PROVEN on this host. `scripts/onex`
# reads it at invocation (OMN-17309) and refuses to let an evidence-producing
# command run below it. The floor never describes a state that was merely
# attempted: it is stamped only when every surface the dispatch build is made
# FROM verdicted ok (see is_dispatch_premise below), and otherwise the previous
# floor is left untouched.
#
# Scoped to the dispatch premise, not to the whole host (OMN-20111). The floor
# records nothing but dispatch-venv distributions and the omnimarket commit, so
# withholding it over a surface that does not feed that build proves nothing
# more and blocks every `onex delegate` on the host. On 2026-09-29 one canonical
# clone (omnibase_core) carried staged files for about 90 minutes, every pass
# ended FAILED on it alone, the dispatch venv had moved to a new omnimarket, and
# the floor kept the old commit, so the wrapper refused all delegation. An
# unrelated failure still fails the verdict, alerts and exits non-zero; it just
# no longer freezes the floor.
#
# A REFUSED CLONE NAMES ITS OWNER (OMN-20403)
# ----------------------------------------------------------------------------
# A clone that verdicts DID_NOT_MOVE (or UNHEALTHY) is explained, not just
# refused: the dirty tracked paths, the index mtime, and the ledger lane whose
# CLAIM last named those paths (or `unowned`), read from $ONEX_LEDGER_PATH.
# The same line is written into the receipt, so `scripts/onex` repeats it when
# it refuses with exit 3. In repair mode the second consecutive refusal of the
# same clone at the same index state and dirty-path digest saves a patch of the
# staged and dirty diff under $OMNI_HOME/.onex_state/dirty-clone-backups and
# appends exactly one MSG through onex-ledger (to the owner; `unowned`
# goes to=operator). A changed index resets the count; --check reports and
# writes nothing. A git lock file older than the clone step budget is named with
# its age and the `rm -f` that clears it: printed, never run. Nothing here
# discards, resets, stashes or cleans a clone.
#
# ----------------------------------------------------------------------------
# Usage:
#   reconcile-host.sh [--check] [--verbose] [--omni-home PATH] [--branch NAME]
#
#     --check       Observe and verdict; run NO delegate and mutate NOTHING.
#                   Fetches (to establish targets) but never checks out or syncs.
#     --verbose     Echo each collaborator command.
#     --omni-home   Registry root, overriding $OMNI_HOME.
#     --branch      Tracked branch (default: dev).
#
# Env:
#   OMNI_HOME                       required unless --omni-home (rule 8: no default)
#   ONEX_RECONCILE_CLONE_DELEGATE   override the clone reconciler (tests)
#   ONEX_RECONCILE_VENV_DELEGATE    override the venv reconciler (tests)
#   ONEX_RECONCILE_STEP_TIMEOUT_S    wall-clock budget for each delegate
#                                   (default: 1800 seconds)
#   ONEX_RECONCILE_MAX_HOLDER_AGE_S age after which a live local lock holder is
#                                   reported separately (default: 7200 seconds)
#   ONEX_RECONCILE_ALERT_CMD        command receiving the alert text on argv;
#                                   defaults to the Slack chat.postMessage path
#   ONEX_LEDGER_PATH                the rolling ledger, read for the owner of a
#                                   refused clone's paths and appended to with
#                                   the MSG; unset means the owner is `unknown`
#   ONEX_RECONCILE_RECEIPT          receipt path (default
#                                   $OMNI_HOME/.onex-workspace-reconcile.json)
#   SLACK_BOT_TOKEN, SLACK_CHANNEL_ID   default alert transport (best effort)
#
# Exit codes:
#   0  every surface verdicted MOVED or ALREADY_AT_TARGET; floor stamped
#   2  a surface FAILED verification, or a surface is UNCOVERED
#   3  INDETERMINATE configuration (no OMNI_HOME, no git, no python3)
#   4  DECLINED because a young live peer holds the lock
#   5  a delegated reconcile step exceeded its wall-clock budget
#   6  a live local process has held the lock past the maximum holder age
#
# There is NO bypass variable, and adding one would defeat the ticket.
# ----------------------------------------------------------------------------
set -uo pipefail

readonly EXIT_OK=0
readonly EXIT_FAILED=2
readonly EXIT_INDETERMINATE=3
# A DECLINE is not a verdict about the host (OMN-18608). This run reconciled
# nothing because a live peer holds the lock, so it proved nothing -- and the
# one thing it must not do is let a caller record "in sync" on its behalf. That
# is exactly what happened for 35 minutes on 2026-09-17: the tick mapped exit 0
# to `verdict="in sync"` and wrote it on runs that did no work at all.
#
# Distinct from EXIT_INDETERMINATE deliberately. Indeterminate means the
# question could not be answered and something may be wrong; a decline means a
# PEER IS ANSWERING IT RIGHT NOW, which is the normal outcome when several hook
# ticks fire at once and must not raise an alarm.
readonly EXIT_DECLINED=4
readonly EXIT_STEP_TIMEOUT=5
readonly EXIT_STALE_LIVE_HOLDER=6

MODE="repair"
VERBOSE=0
OMNI_HOME_ARG=""
BRANCH="dev"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) MODE="check" ;;
    --verbose) VERBOSE=1 ;;
    --omni-home)
      shift
      [[ $# -gt 0 ]] || { echo "reconcile-host.sh: --omni-home requires a path" >&2; exit "$EXIT_INDETERMINATE"; }
      OMNI_HOME_ARG="$1"
      ;;
    --omni-home=*) OMNI_HOME_ARG="${1#--omni-home=}" ;;
    --branch)
      shift
      [[ $# -gt 0 ]] || { echo "reconcile-host.sh: --branch requires a name" >&2; exit "$EXIT_INDETERMINATE"; }
      BRANCH="$1"
      ;;
    --branch=*) BRANCH="${1#--branch=}" ;;
    -h|--help) sed -n '2,127p' "${BASH_SOURCE[0]}"; exit "$EXIT_OK" ;;
    *) echo "reconcile-host.sh: unknown argument: $1" >&2; exit "$EXIT_INDETERMINATE" ;;
  esac
  shift
done

say() { printf '[reconcile-host] %s\n' "$*" >&2; }
trace() { [[ "$VERBOSE" -eq 1 ]] && printf '[reconcile-host]   $ %s\n' "$*" >&2; return 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"

# --------------------------------------------------------------------------- #
# Configuration: fail fast, never guess (CLAUDE.md rule 8)
# --------------------------------------------------------------------------- #
[[ -n "$OMNI_HOME_ARG" ]] && OMNI_HOME="$OMNI_HOME_ARG"

if [[ -z "${OMNI_HOME:-}" ]]; then
  say "INDETERMINATE: OMNI_HOME is not set and --omni-home was not passed."
  say "  A guessed root would reconcile some other checkout and then report"
  say "  success for a workspace nobody is running. Pass one explicitly:"
  say "    reconcile-host.sh --omni-home /path/to/omni_home"
  exit "$EXIT_INDETERMINATE"
fi
if [[ ! -d "$OMNI_HOME" ]]; then
  say "INDETERMINATE: OMNI_HOME does not exist: $OMNI_HOME"
  exit "$EXIT_INDETERMINATE"
fi

command -v git >/dev/null 2>&1 || { say "INDETERMINATE: git is not on PATH."; exit "$EXIT_INDETERMINATE"; }
PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
[[ -n "$PYTHON_BIN" ]] || { say "INDETERMINATE: python3 is not on PATH."; exit "$EXIT_INDETERMINATE"; }

VERIFIER="$SCRIPT_DIR/reconcile_verify_movement.py"
[[ -f "$VERIFIER" ]] || { say "INDETERMINATE: verifier missing at $VERIFIER"; exit "$EXIT_INDETERMINATE"; }

MANIFEST_SH="$SCRIPT_DIR/runtime_build/sibling_clone_manifest.sh"
if [[ ! -f "$MANIFEST_SH" ]]; then
  say "INDETERMINATE: clone manifest missing at $MANIFEST_SH"
  exit "$EXIT_INDETERMINATE"
fi
# shellcheck source=./runtime_build/sibling_clone_manifest.sh
source "$MANIFEST_SH"

PRIVILEGE_LIB="$SCRIPT_DIR/reconcile_privilege_lib.sh"
if [[ ! -f "$PRIVILEGE_LIB" ]]; then
  say "INDETERMINATE: privilege library missing at $PRIVILEGE_LIB"
  say "  Without it there is no way to know who owns the surfaces below, and"
  say "  writing as whoever this process happens to be is the OMN-17366 defect."
  exit "$EXIT_INDETERMINATE"
fi
# shellcheck source=./reconcile_privilege_lib.sh
source "$PRIVILEGE_LIB"

CLONE_DELEGATE="${ONEX_RECONCILE_CLONE_DELEGATE:-$SCRIPT_DIR/runtime_build/reconcile_deploy_clones.sh}"
VENV_DELEGATE="${ONEX_RECONCILE_VENV_DELEGATE:-$SCRIPT_DIR/reconcile-workspace-venvs.sh}"
STEP_TIMEOUT_SECONDS="${ONEX_RECONCILE_STEP_TIMEOUT_S:-1800}"
MAX_HOLDER_AGE_SECONDS="${ONEX_RECONCILE_MAX_HOLDER_AGE_S:-7200}"

if [[ ! "$STEP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  say "INDETERMINATE: ONEX_RECONCILE_STEP_TIMEOUT_S must be a positive integer; got '$STEP_TIMEOUT_SECONDS'."
  exit "$EXIT_INDETERMINATE"
fi
if [[ ! "$MAX_HOLDER_AGE_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  say "INDETERMINATE: ONEX_RECONCILE_MAX_HOLDER_AGE_S must be a positive integer; got '$MAX_HOLDER_AGE_SECONDS'."
  exit "$EXIT_INDETERMINATE"
fi

RECEIPT="${ONEX_RECONCILE_RECEIPT:-$OMNI_HOME/.onex-workspace-reconcile.json}"
FLOOR="$OMNI_HOME/.onex-workspace-floor.json"

# The DISPATCH venv is the composed one -- lock layer plus the omnimarket
# provider layer -- and is what `scripts/onex` execs (OMN-17819). It is read
# here because it is the interpreter a dispatch actually runs in, so it is the
# only one whose omnimarket commit answers the question this readback asks.
# The default is spelled identically in `scripts/onex` and
# `scripts/reconcile-workspace-venvs.sh`, and all three read the same override.
DISPATCH_VENV="${ONEX_DISPATCH_VENV:-$OMNI_HOME/.onex-dispatch-venv}"
# The GATE venv is the canonical clone's own project environment: lock-governed
# ONLY, and the one the OMN-15620 purity gate judges when a lane runs
# `uv run pytest` there. It is read below as a PURITY readback, never as a place
# the provider layer may live.
GATE_VENV="$OMNI_HOME/omnibase_infra/.venv"
CLI_LOCK="$OMNI_HOME/omnibase_infra/uv.lock"
MARKET_CLONE="$OMNI_HOME/omnimarket"

# --------------------------------------------------------------------------- #
# Single-writer lock
# --------------------------------------------------------------------------- #
# `mkdir` and not `flock`: macOS ships no flock(1) (memory
# reference_macos_no_flock_use_fcntl_shim), and this script has to behave
# identically on both hosts. Concurrency here is real -- several hook ticks can
# fire at once -- and without a lock they all pile onto uv's own exclusive lock,
# which is the OMN-15590 stall shape rather than a race.
#
# THE LOCK RECORDS ITS HOLDER (OMN-18608). A bare `mkdir` released only by a
# trap is leaked permanently by a SIGKILL, and a directory with nothing in it
# cannot tell a live holder from a corpse. Measured on this host on 2026-09-17:
# a session died at about 15:25Z holding a lock it had taken at 15:23Z, and
# every tick for the next 35 minutes printed "nothing to do" and exited 0 while
# `ps` showed no reconcile process anywhere. The tick's own receipt line read
# `tick=complete reconciler_exit=0 verdict="in sync"` on runs that did no work,
# so nothing in the log said the host had stopped reconciling.
#
# This is the shape `omni_home/scripts/commit_lock.py` already uses for the
# same reason: the holder is written INTO the lock, and a tick decides by
# reading it rather than by its mere existence.
LOCK_DIR="$OMNI_HOME/.onex-reconcile-host.lock"
LOCK_HOLDER="$LOCK_DIR/holder"

# How long a lock whose holder cannot be PROVEN live may sit before a tick
# reclaims it. Deliberately far longer than any real pass: the tick fires every
# 600s by default and a cold pass syncs several venvs, so an hour is "nothing
# alive would still be here", not "this is taking a while".
LOCK_STALE_SECONDS="${ONEX_RECONCILE_LOCK_STALE_SECONDS:-3600}"

# `stat` is not portable: BSD spells the mtime `-f %m`, GNU spells it `-c %Y`.
# Both are tried rather than branching on `uname`, because this script runs on
# the macOS workstation and the Linux lab host and must behave identically.
#
# THE RESULT IS VALIDATED, NOT JUST THE EXIT STATUS, and that is the whole point
# of this function rather than an inline `stat`. On GNU coreutils `-f` does not
# mean "format" at all -- it means "report on the FILESYSTEM" -- so
# `stat -f %m <path>` there succeeds, exits 0, and prints a filesystem dump
# beginning `File: ...`. A plain `||` fallback therefore never fires on Linux,
# and the dump flows into the `(( ))` below, where bash reads the bare word
# `File` as a variable name and `set -u` kills the script with
# `File: unbound variable`. That is not hypothetical: it is what CI reported on
# the Linux runner for a version of this file that had been proven green on
# macOS, where `-f %m` is correct and the bug is invisible.
#
# So a candidate is accepted only when it is entirely digits, which an epoch
# second is and a filesystem dump is not.
file_mtime() {
  local out
  # shellcheck disable=SC2086  # deliberate: each candidate is a flag PAIR
  for fmt in "-f %m" "-c %Y"; do
    out="$(stat $fmt "$1" 2>/dev/null || true)"
    case "$out" in
      "" | *[!0-9]*) continue ;;
      *) printf '%s' "$out"; return 0 ;;
    esac
  done
  return 1
}

# Seconds since the lock was taken. The holder file when it exists, else the
# directory itself -- which is the window between `mkdir` and the holder being
# written, and is correctly read as "just taken".
lock_age_seconds() {
  local target="$LOCK_DIR" mtime now
  [[ -f "$LOCK_HOLDER" ]] && target="$LOCK_HOLDER"
  mtime="$(file_mtime "$target")"
  # Belt and braces with file_mtime's own validation: nothing but digits may
  # reach the arithmetic below, because under `set -u` a stray word there is a
  # fatal unbound-variable error rather than a bad number.
  case "$mtime" in
    "" | *[!0-9]*) return 1 ;;
  esac
  now="$(date +%s)"
  printf '%s' "$(( now - mtime ))"
}

lock_holder_field() {
  local key="$1" line
  [[ -f "$LOCK_HOLDER" ]] || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      "$key"=*) printf '%s' "${line#*=}"; return 0 ;;
    esac
  done < "$LOCK_HOLDER"
  return 1
}

# Why the existing lock may be reclaimed, or empty when it may not. Printing
# the REASON rather than returning a boolean is deliberate: a reclaim that does
# not say what it overrode would hide a genuine concurrency bug behind a
# self-healing tick, which is the failure this whole file is about.
lock_reclaim_reason() {
  local age pid host
  age="$(lock_age_seconds)" || age=""
  # An unreadable age is not evidence of staleness. Fail closed: respect it.
  [[ -n "$age" ]] || return 0
  pid="$(lock_holder_field pid || true)"
  host="$(lock_holder_field host || true)"

  if [[ -z "$pid" || -z "$host" ]]; then
    # No holder record yet, or a malformed one. Under the bound this is the
    # `mkdir`-to-holder window of a peer that is very much alive.
    (( age > LOCK_STALE_SECONDS )) && \
      printf 'no readable holder record and the lock is %ss old (bound %ss)' \
        "$age" "$LOCK_STALE_SECONDS"
    return 0
  fi

  if [[ "$host" != "$(hostname)" ]]; then
    # HOST-SCOPED ON PURPOSE. A pid from another host means nothing here, and
    # `kill -0` on it would answer about an unrelated LOCAL process -- quite
    # possibly a live one, which would make a dead foreign holder look alive
    # forever. Age is the only honest signal for a foreign lock.
    (( age > LOCK_STALE_SECONDS )) && \
      printf 'holder pid %s is on host %s, not this one, and the lock is %ss old (bound %ss)' \
        "$pid" "$host" "$age" "$LOCK_STALE_SECONDS"
    return 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  printf 'holder pid %s on this host is not running, and the lock is %ss old' \
    "$pid" "$age"
}

# Populate the LIVE_HOLDER_* fields only when the existing holder is a local,
# still-running process whose RECORDED start time is beyond the operator-facing
# maximum. This is deliberately separate from reclaim: age is a reason to make
# a wedged live process loud, never authority to kill it or break its lock.
stale_live_holder() {
  local pid host started_at started_epoch now age
  LIVE_HOLDER_PID=""
  LIVE_HOLDER_HOST=""
  LIVE_HOLDER_AGE=""

  pid="$(lock_holder_field pid || true)"
  host="$(lock_holder_field host || true)"
  started_at="$(lock_holder_field started_at || true)"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$host" == "$(hostname)" ]] || return 1
  [[ -n "$started_at" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1

  started_epoch="$("$PYTHON_BIN" -c \
    'import datetime, sys; print(int(datetime.datetime.strptime(sys.argv[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.UTC).timestamp()))' \
    "$started_at" 2>/dev/null || true)"
  [[ "$started_epoch" =~ ^[0-9]+$ ]] || return 1
  now="$(date +%s)"
  age=$(( now - started_epoch ))
  (( age > MAX_HOLDER_AGE_SECONDS )) || return 1

  LIVE_HOLDER_PID="$pid"
  LIVE_HOLDER_HOST="$host"
  LIVE_HOLDER_AGE="$age"
  return 0
}

write_lock_holder() {
  printf 'pid=%s\nhost=%s\nstarted_at=%s\n' \
    "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_HOLDER"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  reclaim="$(lock_reclaim_reason)"
  if [[ -z "$reclaim" ]]; then
    if stale_live_holder; then
      say "STALE-LIVE-HOLDER: pid $LIVE_HOLDER_PID on $LIVE_HOLDER_HOST has held the lock for ${LIVE_HOLDER_AGE}s (max ${MAX_HOLDER_AGE_SECONDS}s); this run does not kill it"
      say "  Remedy: inspect pid $LIVE_HOLDER_PID on $LIVE_HOLDER_HOST and stop it only after confirming why it is stuck; the next tick will reclaim the dead holder."
      exit "$EXIT_STALE_LIVE_HOLDER"
    fi
    say "another reconcile-host is running ($LOCK_DIR); nothing to do."
    say "  held by pid $(lock_holder_field pid || printf '<unrecorded>') on" \
      "$(lock_holder_field host || printf '<unrecorded>') since" \
      "$(lock_holder_field started_at || printf '<unrecorded>')"
    say "  this run reconciled NOTHING; its exit status says so rather than"
    say "  letting a caller record the host as in sync on its behalf."
    exit "$EXIT_DECLINED"
  fi
  say "RECLAIMING a stale reconcile-host lock: $reclaim"
  rm -rf "$LOCK_DIR"
  # A peer may have reclaimed it in the same instant. Losing that race is the
  # ordinary outcome, not an error.
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    say "another reconcile-host took the lock during the reclaim; nothing to do."
    exit "$EXIT_DECLINED"
  fi
fi
write_lock_holder
# `rm -rf`, not `rmdir`: the lock is no longer an empty directory.
# A delegate runs in its own process group (run_reconcile_step), so a signal
# sent to this shell's group no longer reaches it. Stop that group here too, or
# a TERM to the holder would release the lock while its delegate kept writing.
CURRENT_STEP_PGID=""
cleanup() {
  if [[ -n "$CURRENT_STEP_PGID" ]]; then
    kill -TERM -- "-$CURRENT_STEP_PGID" 2>/dev/null || true
  fi
  rm -rf "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Run one delegate in a process group created solely for that child tree. The
# watchdog itself stays in this shell so it can release the lock through the
# existing EXIT trap. macOS has no guaranteed timeout(1), while its system Perl
# can make the group without depending on launchd's restricted PATH.
STEP_TIMED_OUT=0
run_reconcile_step() { # step-name direct|as_owner command [args...]
  local step="$1" launch_mode="$2" child_pid started now term_deadline
  shift 2
  STEP_TIMED_OUT=0

  if [[ "$launch_mode" == "as_owner" ]]; then
    # RUN_AS is empty when this process already is the owner. macOS ships
    # bash 3.2, where "${RUN_AS[@]}" of an empty array is an unbound-variable
    # error under set -u; the ${var+...} form expands to nothing instead.
    /usr/bin/perl -e 'setpgrp(0, 0) or die "setpgrp: $!\n"; exec @ARGV or die "exec: $!\n"' \
      ${RUN_AS[@]+"${RUN_AS[@]}"} "$@" &
  else
    /usr/bin/perl -e 'setpgrp(0, 0) or die "setpgrp: $!\n"; exec @ARGV or die "exec: $!\n"' "$@" &
  fi
  child_pid=$!
  CURRENT_STEP_PGID="$child_pid"
  started="$(date +%s)"

  while kill -0 "$child_pid" 2>/dev/null; do
    now="$(date +%s)"
    if (( now - started >= STEP_TIMEOUT_SECONDS )); then
      STEP_TIMED_OUT=1
      kill -TERM -- "-$child_pid" 2>/dev/null || true
      term_deadline=$(( now + 5 ))
      while kill -0 -- "-$child_pid" 2>/dev/null; do
        now="$(date +%s)"
        if (( now >= term_deadline )); then
          kill -KILL -- "-$child_pid" 2>/dev/null || true
          break
        fi
        sleep 1
      done
      wait "$child_pid" 2>/dev/null || true
      CURRENT_STEP_PGID=""
      say "TIMEOUT: $step exceeded ${STEP_TIMEOUT_SECONDS}s; killed its process group"
      return 1
    fi
    sleep 1
  done

  local rc=0
  wait "$child_pid" || rc=$?
  CURRENT_STEP_PGID=""
  return "$rc"
}

# --------------------------------------------------------------------------- #
# Alerting
# --------------------------------------------------------------------------- #
# Best effort by design: an alert that cannot be delivered must never turn a
# detected failure into a crash that hides the failure. The non-zero exit and
# the stderr report are the primary signal; Slack is the second copy.
alert() {
  local text="$1"
  if [[ -n "${ONEX_RECONCILE_ALERT_CMD:-}" ]]; then
    trace "$ONEX_RECONCILE_ALERT_CMD <text>"
    # shellcheck disable=SC2086  # deliberate: the override may carry arguments
    ${ONEX_RECONCILE_ALERT_CMD} "$text" >/dev/null 2>&1 || true
    return 0
  fi
  [[ -n "${SLACK_BOT_TOKEN:-}" && -n "${SLACK_CHANNEL_ID:-}" ]] || return 0
  command -v curl >/dev/null 2>&1 || return 0
  command -v jq >/dev/null 2>&1 || return 0
  curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
    -H 'Content-type: application/json; charset=utf-8' \
    --data "$(jq -n --arg channel "${SLACK_CHANNEL_ID}" \
      --arg text "*OmniNode workspace reconcile FAILED* ($(hostname))
${text}" '{channel:$channel,text:$text}')" >/dev/null 2>&1 || true
}

# --------------------------------------------------------------------------- #
# Verdict bookkeeping
# --------------------------------------------------------------------------- #
FAILURES=()
DISPATCH_FAILURES=()
SURFACE_LINES=()
# Index-aligned with SURFACE_LINES: the owner line of a refused clone, or empty
# (OMN-20403). A parallel array, not a fourth `|` field, because the detail text
# is free and the receipt reader below splits on `|`.
SURFACE_OWNERS=()

# Does the dispatch build depend on this surface? (OMN-20111)
#
# The dispatch venv is composed from exactly two sources: the lock layer, whose
# targets are read from $CLI_LOCK inside the omnibase_infra clone, and the
# omnimarket provider layer, installed at the omnimarket clone's HEAD. So the
# premise is every dispatch-venv readback plus those two clones, plus either
# delegate being absent (then nothing reconciles the build at all). Everything
# else -- another sibling clone, the gate venv's purity, a PATH shadow of the
# wrapper -- still fails the verdict, but says nothing about which build a
# dispatch would run, so it may not hold the floor.
#
# An allowlist of what IS premise, not a list of what is not: a surface added
# later is outside the premise until someone decides it feeds the build, and
# the cost of that default is the pre-OMN-20111 behaviour for that one surface,
# never a floor stamped over an unproven build.
is_dispatch_premise() { # surface
  case "$1" in
    venv:gate-purity) return 1 ;;
    venv:*|venv-surface|clone-surface) return 0 ;;
    clone:omnibase_infra|clone:omnimarket) return 0 ;;
    *) return 1 ;;
  esac
}

# The command that clears a failing surface, printed here and written into the
# receipt so `scripts/onex` can name it when it refuses (OMN-20111). A refusal
# that names neither the blocking surface nor its repair is a dead end, and on
# 2026-09-29 several lanes hand-wrote code for an hour rather than find it.
surface_remedy() { # surface verdict
  local rerun="bash $SCRIPT_DIR/reconcile-host.sh --omni-home $OMNI_HOME"
  case "$1" in
    clone:*)
      if [[ "$2" == "UNHEALTHY" ]]; then
        printf 'apply the repair named in the detail, then %s' "$rerun"
      else
        printf 'bash %s/omniclaude/scripts/converge-canonical-clone.sh %s --execute, then %s' \
          "$OMNI_HOME" "${1#clone:}" "$rerun"
      fi
      ;;
    venv:*)
      printf 'bash %s/reconcile-workspace-venvs.sh --omni-home %s, then %s' \
        "$SCRIPT_DIR" "$OMNI_HOME" "$rerun"
      ;;
    clone-surface|venv-surface)
      printf 'restore the missing delegate named in the detail (git -C %s/omnibase_infra status), then %s' \
        "$OMNI_HOME" "$rerun"
      ;;
    onex-path-shadow) printf 'uv tool uninstall omnibase-core, then %s' "$rerun" ;;
    canonical-guard)
      printf 'bash %s/install-canonical-clone-git-hooks.sh (readback), then bash %s/install-canonical-clone-git-hooks.sh --apply <the clones the readback lists as ok>, then %s' \
        "$SCRIPT_DIR" "$SCRIPT_DIR" "$rerun"
      ;;
    *) printf '%s' "$rerun" ;;
  esac
}

record() { # surface verdict detail
  SURFACE_LINES+=("$1|$2|$3")
  SURFACE_OWNERS+=("")
  case "$2" in
    MOVED|ALREADY_AT_TARGET) say "  $1: $2 ($3)" ;;
    *)
      say "  $1: $2 ($3)"
      FAILURES+=("$1: $2 — $3")
      is_dispatch_premise "$1" && DISPATCH_FAILURES+=("$1: $2")
      ;;
  esac
}

judge() { # surface before after target
  local surface="$1" before="${2:-}" after="${3:-}" target="${4:-}"
  local out name detail
  # The verifier emits exactly one tab-separated line on stdout:
  #   <surface>\t<VERDICT>\t<detail>
  # stderr is deliberately NOT merged: capturing 2>&1 is how a stray diagnostic
  # ends up parsed as a verdict.
  out="$("$PYTHON_BIN" "$VERIFIER" verdict --surface "$surface" \
    --before "$before" --after "$after" --target "$target" 2>/dev/null)"
  IFS=$'\t' read -r _ name detail < <(printf '%s\n' "$out")
  record "$surface" "${name:-INDETERMINATE}" "${detail:-verifier produced no verdict}"
}

# --------------------------------------------------------------------------- #
# Clone surface
# --------------------------------------------------------------------------- #
present_clones=()
for repo in "${SIBLING_CLONE_MANIFEST[@]}"; do
  [[ -e "$OMNI_HOME/$repo/.git" ]] && present_clones+=("$repo")
done

clone_head() { git -C "$1" rev-parse HEAD 2>/dev/null || true; }
clone_target() { git -C "$1" rev-parse "origin/$BRANCH" 2>/dev/null || true; }

# --------------------------------------------------------------------------- #
# Who the clone-surface writes run as (OMN-17366)
# --------------------------------------------------------------------------- #
# Planned BEFORE the first fetch, because the first fetch is already a write.
#
# `git fetch` deposits objects, refs and reflogs. Running it as root against an
# operator-owned clone is what left 1118 root-owned paths under `.201`'s five
# deploy-source clones, after which a plain operator fetch fails intermittently
# with "insufficient permission for adding an object to repository database".
#
# THIS APPLIES IN --check MODE TOO, and that is a deliberate divergence from
# reconcile-workspace-venvs.sh, which exempts its check mode on the grounds that
# a read-only probe writes nothing. True there; false here. Check mode on this
# script still fetches -- it has to, since a verifier that takes its target from
# the thing under verification is not a verifier -- so a `--check` that ran as
# the wrong user would deposit exactly the objects this ticket is about.
plan_clone_privileges() {
  local rc=0 repo owner
  rp_plan_privileges "$OMNI_HOME" || rc=$?

  case "$rc" in
    0) ;;
    1)
      say "INDETERMINATE: cannot read the owner of $OMNI_HOME."
      say "  Every fetch and checkout below writes into that tree. Without"
      say "  knowing who owns it there is no way to write as the right user,"
      say "  and writing as the wrong one leaves clones their owner can no"
      say "  longer fetch into."
      exit "$EXIT_INDETERMINATE"
      ;;
    3)
      say "INDETERMINATE: $OMNI_HOME is owned by $RP_OWNER, whose home directory"
      say "  could not be resolved. Dropping privileges without a HOME leaves"
      say "  git reading root's config and credentials as an unprivileged user."
      exit "$EXIT_INDETERMINATE"
      ;;
    *)
      say "INDETERMINATE: $OMNI_HOME is owned by $RP_OWNER, but this process runs"
      say "  as $CURRENT_USER and cannot become that user."
      say "  Fetching anyway would put $CURRENT_USER-owned objects inside"
      say "  $RP_OWNER's clones, after which $RP_OWNER's own git commands fail"
      say "  on permissions (OMN-17366). Run this as $RP_OWNER, or as root on a"
      say "  host with runuser."
      exit "$EXIT_INDETERMINATE"
      ;;
  esac

  # One delegate invocation cannot be two users at once, so a split ownership
  # set has no correct answer: running it as either owner writes into the
  # other's tree as the wrong user. Refuse rather than pick.
  for repo in "${present_clones[@]}"; do
    owner="$(rp_surface_owner "$OMNI_HOME/$repo" 2>/dev/null || true)"
    [[ -z "$owner" || "$owner" == "$RP_OWNER" ]] && continue
    say "INDETERMINATE: the clones under $OMNI_HOME do not share one owner."
    say "  $OMNI_HOME is owned by $RP_OWNER, but $repo is owned by $owner."
    say "  The clone delegate reconciles every clone in a single process, so"
    say "  whichever user it ran as would write into the other's tree as the"
    say "  wrong one — the very thing this guard exists to prevent."
    exit "$EXIT_INDETERMINATE"
  done

  [[ ${#RUN_AS[@]} -eq 0 ]] || \
    say "writing as $RP_OWNER (owner of $OMNI_HOME); this process is $CURRENT_USER"
}

# Establish targets ourselves. See the header: taking the target from the thing
# under verification is not verification.
fetch_all() {
  local repo
  for repo in "${present_clones[@]}"; do
    trace "git -C $OMNI_HOME/$repo fetch --quiet --prune origin $BRANCH"
    as_owner git -C "$OMNI_HOME/$repo" fetch --quiet --prune origin "$BRANCH" 2>/dev/null || true
  done
}

# Who owns the work in a refused clone, and what else is in the way (OMN-20403).
# Runs for a clone whose verdict just failed. The verifier reads the clone and the
# ledger, and in repair mode keeps the consecutive-refusal count, saves a patch
# of the staged diff and appends the one MSG; this script removes nothing from a
# clone and runs none of the printed commands.
explain_refused_clone() { # repo clone
  local repo="$1" clone="$2" last tag text owner_line=""
  local -a args
  last=$(( ${#SURFACE_LINES[@]} - 1 ))
  case "${SURFACE_LINES[$last]}" in
    "clone:$repo|MOVED|"*|"clone:$repo|ALREADY_AT_TARGET|"*) return 0 ;;
  esac
  args=(--clone "$clone" --repo "$repo" --state-dir "$OMNI_HOME/.onex_state"
        --ledger "${ONEX_LEDGER_PATH:-}"
        --lock-age-s "$STEP_TIMEOUT_SECONDS")
  [[ "$MODE" == "repair" ]] && args+=(--record)
  while IFS=$'\t' read -r tag text; do
    say "    $text"
    [[ "$tag" == "owner" && -z "$owner_line" ]] && owner_line="$text"
  done < <(as_owner env OMNI_HOME="$OMNI_HOME" "$PYTHON_BIN" "$VERIFIER" clone-refusal "${args[@]}" || \
             say "clone-refusal report failed for $repo (exit $?)")
  SURFACE_OWNERS[last]="${owner_line#owner: }"
}

declare -a before_heads=()
say "surfaces under $OMNI_HOME (branch $BRANCH): clones=${#present_clones[@]}"

plan_clone_privileges

fetch_all
for repo in "${present_clones[@]}"; do
  before_heads+=("$(clone_head "$OMNI_HOME/$repo")")
done

if [[ "$MODE" == "repair" ]]; then
  if [[ ! -f "$CLONE_DELEGATE" ]]; then
    # Not a skip. See the header.
    record "clone-surface" "UNCOVERED" \
      "no clone reconciler at $CLONE_DELEGATE — the deploy-source clones on this host are reconciled by nobody"
  else
    say "clone surface: delegating to $CLONE_DELEGATE"
    trace "OMNI_HOME=$OMNI_HOME RECONCILE_BRANCH=$BRANCH bash $CLONE_DELEGATE"
    # The delegate fetches AND checks out, so it is the larger of the two write
    # paths into these clones. Guarding only the fetch above would have fixed
    # the smaller half and left the damage accumulating (OMN-17366).
    if ! run_reconcile_step "clone delegate" as_owner env \
        OMNI_HOME="$OMNI_HOME" RECONCILE_BRANCH="$BRANCH" \
        bash "$CLONE_DELEGATE" >&2; then
      (( STEP_TIMED_OUT == 0 )) || exit "$EXIT_STEP_TIMEOUT"
      say "clone delegate exited non-zero; the readback below is what decides."
    fi
  fi
fi

# Re-establish targets after the delegate ran; a delegate that fetched moves
# origin/<branch>, and one that did not leaves it where we put it above.
fetch_all
idx=0
for repo in "${present_clones[@]}"; do
  clone="$OMNI_HOME/$repo"
  if health="$("$PYTHON_BIN" "$VERIFIER" clone-health --clone "$clone" 2>/dev/null)"; then
    judge "clone:$repo" "${before_heads[$idx]}" "$(clone_head "$clone")" "$(clone_target "$clone")"
  else
    IFS=$'\t' read -r _ _ health_reason < <(printf '%s\n' "$health")
    # The core.bare=true trap: fetch succeeds, checkout cannot. Reported ahead
    # of the HEAD comparison, because that comparison alone would say
    # DID_NOT_MOVE without saying WHY -- and a refusal that does not name the
    # repair is a dead end.
    record "clone:$repo" "UNHEALTHY" "${health_reason:-clone is not checkout-capable}"
  fi
  explain_refused_clone "$repo" "$clone"
  idx=$((idx + 1))
done

# --------------------------------------------------------------------------- #
# Venv surface
# --------------------------------------------------------------------------- #
site_packages() {
  local venv="$1" candidate
  for candidate in "$venv"/lib/python*/site-packages; do
    [[ -d "$candidate" ]] && { printf '%s' "$candidate"; return 0; }
  done
  return 1
}

observe_version() { # site-packages dist_prefix
  local sp="$1" dist="$2" d
  for d in "$sp/$dist"-*.dist-info; do
    [[ -d "$d" ]] || continue
    d="$(basename "$d")"
    d="${d%.dist-info}"
    printf '%s' "${d##*-}"
    return 0
  done
  return 1
}

observe_commit() { # site-packages dist_prefix
  "$PYTHON_BIN" "$VERIFIER" observe --site-packages "$1" --commit-dist "$2" 2>/dev/null |
    sed -n 's/.*"'"$2"'": "\([0-9a-f]*\)".*/\1/p' | head -1
}

SP=""
if ! SP="$(site_packages "$DISPATCH_VENV")"; then
  record "venv:dispatch" "INDETERMINATE" "no site-packages under $DISPATCH_VENV"
fi

# Governed distributions: exactly the lock-governed siblings, named from the
# same index-aligned manifest the clone loop uses, so the two surfaces can never
# disagree about which repos are in scope.
governed_dists=()
for name in "${SIBLING_CLONE_MANIFEST_DIST_NAMES[@]}"; do
  [[ "$name" == "omnimarket" ]] && continue  # composed layer, verified by commit
  governed_dists+=("$name")
done

lock_target_json=""
if [[ -f "$CLI_LOCK" ]]; then
  lock_args=()
  for name in "${governed_dists[@]}"; do lock_args+=(--dist "$name"); done
  lock_target_json="$("$PYTHON_BIN" "$VERIFIER" lock-targets --lock "$CLI_LOCK" "${lock_args[@]}" 2>/dev/null)"
fi
lock_target_for() { # dist-name
  printf '%s' "$lock_target_json" | sed -n 's/.*"'"$1"'": "\([^"]*\)".*/\1/p' | head -1
}

declare -a before_versions=()
for name in "${governed_dists[@]}"; do
  before_versions+=("$( [[ -n "$SP" ]] && observe_version "$SP" "${name//-/_}" || true )")
done
before_market_commit="$( [[ -n "$SP" ]] && observe_commit "$SP" "omnimarket" || true )"

if [[ "$MODE" == "repair" ]]; then
  if [[ ! -f "$VENV_DELEGATE" ]]; then
    record "venv-surface" "UNCOVERED" \
      "no venv reconciler at $VENV_DELEGATE — the installed layers on this host are reconciled by nobody"
  else
    say "venv surface: delegating to $VENV_DELEGATE"
    # --branch is passed through so the delegate's own clone->origin/<branch>
    # observation (OMN-17295) names the same branch this run is reconciling
    # onto. Both default to `dev`; they would only disagree when someone passes
    # --branch here, which is precisely when a silent disagreement would be
    # hardest to spot.
    trace "bash $VENV_DELEGATE --omni-home $OMNI_HOME --branch $BRANCH"
    if ! run_reconcile_step "venv delegate" direct bash "$VENV_DELEGATE" \
        --omni-home "$OMNI_HOME" --branch "$BRANCH" >&2; then
      (( STEP_TIMED_OUT == 0 )) || exit "$EXIT_STEP_TIMEOUT"
      say "venv delegate exited non-zero; the readback below is what decides."
    fi
    SP="$(site_packages "$DISPATCH_VENV" || true)"
  fi
fi

if [[ -n "$SP" ]]; then
  idx=0
  for name in "${governed_dists[@]}"; do
    target="$(lock_target_for "$name")"
    if [[ -z "$target" ]]; then
      # Not lock-governed on this host: nothing to assert, and asserting
      # something anyway would manufacture a failure out of a package the lock
      # legitimately does not pin.
      idx=$((idx + 1))
      continue
    fi
    judge "venv:$name" "${before_versions[$idx]}" "$(observe_version "$SP" "${name//-/_}" || true)" "$target"
    idx=$((idx + 1))
  done

  # omnimarket is deliberately absent from omnibase_infra's lock (the layer
  # graph puts it ABOVE infra), so its target is the canonical clone's HEAD --
  # which is also exactly what the OMN-14060 drift guard compares against.
  if [[ -d "$MARKET_CLONE/.git" ]]; then
    judge "venv:omnimarket" "$before_market_commit" \
      "$(observe_commit "$SP" "omnimarket")" "$(clone_head "$MARKET_CLONE")"
  fi
fi

# --------------------------------------------------------------------------- #
# Gate-venv purity surface (OMN-17819)
# --------------------------------------------------------------------------- #
# The repair delegate composes the provider layer into $DISPATCH_VENV and syncs
# the gate venv EXACT, which is what keeps `uv run pytest` in the canonical clone
# runnable. Read that back rather than trusting the delegate's exit code -- this
# script owns PROOF and the delegate owns REPAIR, and an undeclared `onex.nodes`
# provider in the gate venv is invisible to every exit status involved. The
# failure it prevents does not surface here at all: it surfaces later, at some
# other lane's `pytest_configure`, as a refusal with no pointer back to this run.
#
# `*.dist-info` directory names, not an interpreter start: the same
# packaging-spec-encoded observation the wrapper's floor check uses, so this
# still answers when the gate venv's own python is broken. Checked in BOTH
# modes -- in check mode it is the one leg that would otherwise let a
# "clones/venv: in sync" verdict be printed over a gate venv nobody can run
# tests in.
gate_venv_purity_check() {
  local sp d name
  sp="$(site_packages "$GATE_VENV" 2>/dev/null || true)"
  if [[ -z "$sp" ]]; then
    # Not a failure: a host with no gate venv has nothing to keep pure, and
    # manufacturing one here would fire on every fresh clone.
    return 0
  fi
  for d in "$sp"/omnimarket-*.dist-info; do
    [[ -d "$d" ]] || continue
    name="${d##*/}"
    record "venv:gate-purity" "IMPURE" \
      "$name is installed in $GATE_VENV, which is lock-governed only — every \`uv run pytest\` in $OMNI_HOME/omnibase_infra is refused by the OMN-15620 purity gate while it is there; the provider layer belongs in $DISPATCH_VENV (OMN-17819)"
    return 0
  done
  record "venv:gate-purity" "ALREADY_AT_TARGET" "no undeclared omnimarket provider in $GATE_VENV"
}
gate_venv_purity_check

# --------------------------------------------------------------------------- #
# onex CLI PATH-shadow surface (OMN-18403)
# --------------------------------------------------------------------------- #
# A `uv tool install omnibase-core` (or any other package shipping an `onex`
# console script) drops a binary at `$HOME/.local/bin/onex`. The sanctioned
# invocation is the wrapper at `$SCRIPT_DIR/onex` (`scripts/onex`), reached via
# the interactive-shell alias in `~/.zshrc` -- but that alias covers
# INTERACTIVE shells only. Every non-interactive invocation of a bare `onex`
# (a script, a hook, a cron job, `zsh -c`) resolves through PATH instead, and
# `~/.local/bin` sits ahead of nothing that would stop it. A stray tool install
# there silently outranks the wrapper for exactly the invocations most likely
# to run unattended -- this was live on this host from 2026-09-10 to
# 2026-09-15, went undetected by this reconciler for five days, and produced
# four separate "onex delegate unavailable or stale" reports before being
# found and removed by hand.
#
# This surface is deliberately checked in BOTH modes (unlike the clone-origin
# drift leg above, which is report-only in repair mode because the venv
# reconciler does not own clone convergence): a stray tool install under
# ~/.local/bin is not a surface this reconciler mutates in either mode, but it
# IS the exact class of drift a "clones/venv: in sync" verdict must not paper
# over. A fixed path is checked -- not a live `command -v onex`, which would
# depend on the caller's own PATH -- so the verdict is deterministic
# regardless of what shell state this process happens to inherit from.
#
# NOT every occupant of $shadow is drift (OMN-19810). A symlink placed there on
# purpose so `onex` is on PATH, whose fully-resolved target IS the canonical
# wrapper, is the sanctioned setup working as intended and must not fail the
# run. What still fails: a regular file (the `uv tool install` shim shape this
# surface was built to catch), and a symlink that resolves anywhere else. Both
# of those keep outranking the wrapper for every non-interactive invocation,
# same as before. `readlink -f` is used for resolution -- present as a plain
# `-f` flag on both BSD/macOS `readlink` (verified: `man readlink` on this
# host) and GNU coreutils, so no python/perl fallback is needed here.
path_onex_shadow_check() {
  local shadow wrapper resolved_shadow resolved_wrapper found
  [[ -n "${HOME:-}" ]] || return 0
  shadow="${HOME}/.local/bin/onex"
  wrapper="$SCRIPT_DIR/onex"
  [[ -e "$shadow" || -L "$shadow" ]] || return 0

  if [[ -L "$shadow" ]]; then
    resolved_shadow="$(readlink -f "$shadow" 2>/dev/null || true)"
    resolved_wrapper="$(readlink -f "$wrapper" 2>/dev/null || true)"
    if [[ -n "$resolved_shadow" && -n "$resolved_wrapper" && "$resolved_shadow" == "$resolved_wrapper" ]]; then
      record "onex-path-shadow" "ALREADY_AT_TARGET" \
        "$shadow is a symlink to the canonical wrapper ($resolved_wrapper) — not a shadow"
      return 0
    fi
    found="a symlink to ${resolved_shadow:-$(readlink "$shadow" 2>/dev/null || echo "an unresolvable target")}"
  else
    found="a regular file"
  fi

  record "onex-path-shadow" "SHADOWED" \
    "$shadow exists ($found) and outranks $wrapper for every non-interactive onex invocation (the interactive-shell alias never covers those) — fix: uv tool uninstall omnibase-core"
}
path_onex_shadow_check

# --------------------------------------------------------------------------- #
# Installed canonical-clone guard surface (OMN-17291)
# --------------------------------------------------------------------------- #
# The reference-transaction guard that decides what this reconciler may do to a
# canonical clone is a COPY, installed under $OMNI_HOME/scripts/git-hooks from
# the tracked source beside this script. Nothing proved the copy current. On
# 2026-09-26 the tracked guard learned that `checkout -B dev` on the branch HEAD
# is already on is not a branch switch (OMN-18608), and the installed copy stayed
# at the 2026-09-24 text for nine days: every tick's own `checkout --force -B dev`
# was refused by a guard that had already been fixed, and only the readback below
# the delegate kept a clone that happened to sit at its target reading in sync.
# A clone that fell behind origin/dev could not advance at all, and the tick
# failed (measured 2026-10-03T22:33Z onward).
#
# Read it back by content, never by installer exit status: the installer's own
# readback also reports clones whose core.hooksPath is deliberately not the
# shared directory, which would hold this surface red forever. This surface
# answers one question -- is the live guard byte-identical to the tracked guard
# -- and its repair is the sanctioned installer, which keeps a dated copy of what
# it replaces. It is never applied here: rewriting an enforcement surface
# unattended is not a reconciliation step.
#
# A host with no live hooks directory has no installed guard to be stale, so it
# records nothing, the same way the gate-venv check treats a host with no gate
# venv. Never a failure by absence.
canonical_guard_drift_check() {
  local src_dir="$SCRIPT_DIR/git-hooks" live_dir="$OMNI_HOME/scripts/git-hooks"
  local script drifted=() checked=0
  [[ -d "$live_dir" && -d "$src_dir" ]] || return 0
  for script in canonical_clone_guard.sh canonical_clone_paths.sh canonical_clone_ref_guard.sh; do
    [[ -f "$src_dir/$script" ]] || continue
    checked=$((checked + 1))
    if [[ ! -f "$live_dir/$script" ]] || ! cmp -s "$src_dir/$script" "$live_dir/$script"; then
      drifted+=("$script")
    fi
  done
  (( checked > 0 )) || return 0
  if (( ${#drifted[@]} > 0 )); then
    record "canonical-guard" "DRIFT" \
      "the installed canonical-clone guard at $live_dir differs from the tracked source at $src_dir: ${drifted[*]} -- a guard fix merged to dev is not in force on this host until it is installed (OMN-17291)"
    return 0
  fi
  record "canonical-guard" "ALREADY_AT_TARGET" \
    "$checked installed guard script(s) in $live_dir are byte-identical to $src_dir"
}
canonical_guard_drift_check

# --------------------------------------------------------------------------- #
# Receipt, floor, alert
# --------------------------------------------------------------------------- #
{
  printf '{\n  "schema": "onex.workspace.reconcile.v1",\n'
  printf '  "generated_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "mode": "%s",\n  "omni_home": "%s",\n  "branch": "%s",\n' "$MODE" "$OMNI_HOME" "$BRANCH"
  printf '  "surfaces": [\n'
  sep=""
  oi=0
  for line in "${SURFACE_LINES[@]}"; do
    IFS='|' read -r s v d < <(printf '%s\n' "$line")
    owner="${SURFACE_OWNERS[$oi]}"
    oi=$((oi + 1))
    premise=false
    is_dispatch_premise "$s" && premise=true
    remedy=""
    case "$v" in
      MOVED|ALREADY_AT_TARGET) ;;
      *) remedy="$(surface_remedy "$s" "$v")" ;;
    esac
    # ONE LINE PER SURFACE, and the key order is a consumed contract:
    # `scripts/onex` reads failing surfaces and their remedies back out of this
    # file with awk when it refuses (OMN-20111), because its hot path starts no
    # interpreter. Keep `surface`, `verdict`, `dispatch_premise`, `remedy` and
    # `owner` ahead of `detail` on the element's own line (owner: OMN-20403).
    printf '%s    {"surface": "%s", "verdict": "%s", "dispatch_premise": %s, "remedy": "%s", "owner": "%s", "detail": "%s"}' \
      "$sep" "$s" "$v" "$premise" "${remedy//\"/\'}" "${owner//\"/\'}" "${d//\"/\'}"
    # $'...' , not "..." (OMN-17800). Bash interprets \n only in ANSI-C quoting,
    # and this value is then handed to printf as a %s ARGUMENT, where printf does
    # not interpret escapes either -- so `sep=",\n"` wrote the literal three
    # characters `,\n` between elements and every receipt this script has ever
    # produced, on BOTH hosts, failed json.loads with "Expecting value: line 8".
    # The pre-existing receipt test missed it because its workspace yields one
    # surface, and a separator is untested until something is separated.
    sep=$',\n'
  done
  printf '\n  ],\n  "failures": %d,\n  "dispatch_premise_failures": %d\n}\n' \
    "${#FAILURES[@]}" "${#DISPATCH_FAILURES[@]}"
} | as_owner tee "$RECEIPT" >/dev/null 2>&1 || \
  say "WARNING: could not write receipt to $RECEIPT"
# `tee` rather than a `>` redirection: a redirect is performed by THIS shell, so
# it would create a root-owned receipt inside an operator-owned $OMNI_HOME even
# though every other write here drops privileges — the same defect, one file
# over, and the file an operator is most likely to want to delete (OMN-17366).

# Stamp the floor from what the dispatch venv holds NOW. Called only on a
# repair run where every dispatch-premise surface verdicted ok; see the header.
stamp_floor() {
  local name v mc
  local -a floor_args=()
  if [[ -n "$SP" ]]; then
    for name in "${governed_dists[@]}"; do
      v="$(observe_version "$SP" "${name//-/_}" || true)"
      [[ -n "$v" ]] && floor_args+=(--distribution "${name//-/_}=$v")
    done
    mc="$(observe_commit "$SP" "omnimarket")"
    [[ -n "$mc" ]] && floor_args+=(--omnimarket-commit "$mc")
  fi
  if [[ "${#floor_args[@]}" -gt 0 ]]; then
    # As the owner: the floor lives inside $OMNI_HOME, and `scripts/onex` reads it
    # on every invocation. A root-owned floor is one the operator's own reconcile
    # can no longer restamp.
    as_owner "$PYTHON_BIN" "$VERIFIER" floor --output "$FLOOR" --omni-home "$OMNI_HOME" "${floor_args[@]}" >&2
  else
    say "WARNING: nothing observable to stamp; floor left untouched."
  fi
}

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  say "VERDICT: FAILED — ${#FAILURES[@]} surface(s) could not be proven at target."
  for line in "${SURFACE_LINES[@]}"; do
    IFS='|' read -r s v d < <(printf '%s\n' "$line")
    case "$v" in
      MOVED|ALREADY_AT_TARGET) continue ;;
    esac
    say "  $s: $v — $d"
    say "    clears with: $(surface_remedy "$s" "$v")"
  done
  say "  receipt: $RECEIPT"
  if [[ "$MODE" == "repair" && "${#DISPATCH_FAILURES[@]}" -eq 0 ]]; then
    # The dispatch premise is proven even though the host is not (OMN-20111).
    stamp_floor
    say "  The dispatch premise (dispatch venv, omnibase_infra and omnimarket clones) IS"
    say "  proven, so $FLOOR was stamped: the failures above do not block onex delegate."
  else
    say "  The floor marker was NOT stamped; $FLOOR keeps whatever was last proven."
    # The `+` form: macOS /bin/bash 3.2 treats an empty array as unbound under
    # `set -u`, and a --check run with only unrelated failures has none.
    for f in ${DISPATCH_FAILURES[@]+"${DISPATCH_FAILURES[@]}"}; do
      say "  blocks onex delegate: $f"
    done
  fi
  alert "$(printf '%s\n' "${FAILURES[@]}")
receipt: $RECEIPT
host root: $OMNI_HOME"
  exit "$EXIT_FAILED"
fi

if [[ "$MODE" == "check" ]]; then
  say "VERDICT: IN_SYNC (check mode; nothing mutated, floor untouched)"
  exit "$EXIT_OK"
fi

stamp_floor

say "VERDICT: IN_SYNC — every surface proven at target."
exit "$EXIT_OK"
