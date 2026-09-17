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
# On an all-ok run it stamps ${OMNI_HOME}/.onex-workspace-floor.json -- the
# minimum installed state that has been PROVEN on this host. `scripts/onex`
# reads it at invocation (OMN-17309) and refuses to let an evidence-producing
# command run below it. A failed run leaves the previous floor untouched, so the
# floor never describes a state that was merely attempted.
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
#   ONEX_RECONCILE_ALERT_CMD        command receiving the alert text on argv;
#                                   defaults to the Slack chat.postMessage path
#   ONEX_RECONCILE_RECEIPT          receipt path (default
#                                   $OMNI_HOME/.onex-workspace-reconcile.json)
#   SLACK_BOT_TOKEN, SLACK_CHANNEL_ID   default alert transport (best effort)
#
# Exit codes:
#   0  every surface verdicted MOVED or ALREADY_AT_TARGET; floor stamped
#   2  a surface FAILED verification, or a surface is UNCOVERED
#   3  INDETERMINATE configuration (no OMNI_HOME, no git, no python3)
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
    -h|--help) sed -n '2,110p' "${BASH_SOURCE[0]}"; exit "$EXIT_OK" ;;
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

write_lock_holder() {
  printf 'pid=%s\nhost=%s\nstarted_at=%s\n' \
    "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_HOLDER"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  reclaim="$(lock_reclaim_reason)"
  if [[ -z "$reclaim" ]]; then
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
cleanup() { rm -rf "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

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
SURFACE_LINES=()

record() { # surface verdict detail
  SURFACE_LINES+=("$1|$2|$3")
  case "$2" in
    MOVED|ALREADY_AT_TARGET) say "  $1: $2 ($3)" ;;
    *) say "  $1: $2 ($3)"; FAILURES+=("$1: $2 — $3") ;;
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
  IFS=$'\t' read -r _ name detail <<<"$out"
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
    as_owner env OMNI_HOME="$OMNI_HOME" RECONCILE_BRANCH="$BRANCH" \
      bash "$CLONE_DELEGATE" >&2 || \
      say "clone delegate exited non-zero; the readback below is what decides."
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
    IFS=$'\t' read -r _ _ health_reason <<<"$health"
    # The core.bare=true trap: fetch succeeds, checkout cannot. Reported ahead
    # of the HEAD comparison, because that comparison alone would say
    # DID_NOT_MOVE without saying WHY -- and a refusal that does not name the
    # repair is a dead end.
    record "clone:$repo" "UNHEALTHY" "${health_reason:-clone is not checkout-capable}"
  fi
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
    bash "$VENV_DELEGATE" --omni-home "$OMNI_HOME" --branch "$BRANCH" >&2 || \
      say "venv delegate exited non-zero; the readback below is what decides."
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
path_onex_shadow_check() {
  local shadow wrapper
  [[ -n "${HOME:-}" ]] || return 0
  shadow="${HOME}/.local/bin/onex"
  wrapper="$SCRIPT_DIR/onex"
  [[ -e "$shadow" ]] || return 0
  record "onex-path-shadow" "SHADOWED" \
    "$shadow exists and outranks $wrapper for every non-interactive onex invocation (the interactive-shell alias never covers those) — fix: uv tool uninstall omnibase-core"
}
path_onex_shadow_check

# --------------------------------------------------------------------------- #
# Receipt, floor, alert
# --------------------------------------------------------------------------- #
{
  printf '{\n  "schema": "onex.workspace.reconcile.v1",\n'
  printf '  "generated_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "mode": "%s",\n  "omni_home": "%s",\n  "branch": "%s",\n' "$MODE" "$OMNI_HOME" "$BRANCH"
  printf '  "surfaces": [\n'
  sep=""
  for line in "${SURFACE_LINES[@]}"; do
    IFS='|' read -r s v d <<<"$line"
    printf '%s    {"surface": "%s", "verdict": "%s", "detail": "%s"}' "$sep" "$s" "$v" "${d//\"/\'}"
    # $'...' , not "..." (OMN-17800). Bash interprets \n only in ANSI-C quoting,
    # and this value is then handed to printf as a %s ARGUMENT, where printf does
    # not interpret escapes either -- so `sep=",\n"` wrote the literal three
    # characters `,\n` between elements and every receipt this script has ever
    # produced, on BOTH hosts, failed json.loads with "Expecting value: line 8".
    # The pre-existing receipt test missed it because its workspace yields one
    # surface, and a separator is untested until something is separated.
    sep=$',\n'
  done
  printf '\n  ],\n  "failures": %d\n}\n' "${#FAILURES[@]}"
} | as_owner tee "$RECEIPT" >/dev/null 2>&1 || \
  say "WARNING: could not write receipt to $RECEIPT"
# `tee` rather than a `>` redirection: a redirect is performed by THIS shell, so
# it would create a root-owned receipt inside an operator-owned $OMNI_HOME even
# though every other write here drops privileges — the same defect, one file
# over, and the file an operator is most likely to want to delete (OMN-17366).

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  say "VERDICT: FAILED — ${#FAILURES[@]} surface(s) could not be proven at target."
  for f in "${FAILURES[@]}"; do say "  $f"; done
  say "  receipt: $RECEIPT"
  say "  The floor marker was NOT stamped; $FLOOR keeps whatever was last proven."
  alert "$(printf '%s\n' "${FAILURES[@]}")
receipt: $RECEIPT
host root: $OMNI_HOME"
  exit "$EXIT_FAILED"
fi

if [[ "$MODE" == "check" ]]; then
  say "VERDICT: IN_SYNC (check mode; nothing mutated, floor untouched)"
  exit "$EXIT_OK"
fi

floor_args=()
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

say "VERDICT: IN_SYNC — every surface proven at target."
exit "$EXIT_OK"
