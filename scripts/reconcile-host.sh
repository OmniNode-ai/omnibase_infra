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

CLI_VENV="$OMNI_HOME/omnibase_infra/.venv"
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
LOCK_DIR="$OMNI_HOME/.onex-reconcile-host.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  say "another reconcile-host is running ($LOCK_DIR); nothing to do."
  exit "$EXIT_OK"
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
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
if ! SP="$(site_packages "$CLI_VENV")"; then
  record "venv:cli" "INDETERMINATE" "no site-packages under $CLI_VENV"
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
    trace "bash $VENV_DELEGATE --omni-home $OMNI_HOME"
    bash "$VENV_DELEGATE" --omni-home "$OMNI_HOME" >&2 || \
      say "venv delegate exited non-zero; the readback below is what decides."
    SP="$(site_packages "$CLI_VENV" || true)"
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
