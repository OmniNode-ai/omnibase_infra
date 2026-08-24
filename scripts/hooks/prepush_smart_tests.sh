#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Pre-push governed impacted-test selector (OMN-13973 / WS7 OMN-14655 fan-out).
#
# Runs the FAST LOCAL IMPACTED SUBSET of the unit suite once per `git push`,
# using the SAME governed selector CI uses -- scripts/ci/detect_test_paths.py +
# scripts/ci/test_selection_adjacency.yaml -- NOT a hand-typed `-k`. The selector
# is fail-closed: it escalates to the full unit suite whenever it cannot prove
# narrowing is safe (a shared module -- models/enums/runtime/errors/nodes/topics --
# a test-infrastructure change (pyproject.toml, tests/conftest.py, tests/fixtures/,
# tests/helpers/, pytest.ini), a >=6-module change, or the main branch). See root
# CLAUDE.md Rule #4.
#
# This hook is deliberately NOT byte-parity with an enforced CI context. On CI the
# selector is gated behind ENABLE_SMART_TESTS (off by default during rollout) and
# the enforced merge gate is the FULL suite. This hook is *net-new, fast local
# impacted-subset enforcement* -- a fast local mirror that is ADVISORY of the full
# CI suite, run before the push leaves the machine. It retires the "run the whole
# unit suite by hand before every push" default (CLAUDE.md Rule #4: "until
# OMN-13973 lands, the full local suite remains the fail-closed default").
#
# PER-REPO SEAM-MATCH (this is an adaptation of the omnibase_core#1451 canary,
# NOT a paste):
#   * omnibase_infra's scripts/ci/detect_test_paths.py hardcodes
#     SRC_PREFIX = "src/omnibase_infra/" and its own adjacency map; the selector
#     is invoked with the SAME flags CI uses (.github/workflows/ci.yml).
#   * infra's detect_test_paths.py main() does NOT accept a `--base-ref` argument
#     (core's does), and infra CI does not pass one either. So this wrapper
#     computes the merge-base locally for `git diff` but MUST NOT forward
#     `--base-ref` to the selector -- doing so would make the selector argparse
#     hard-error on every push. DRY parity is with infra's OWN CI invocation.
#   * The full-suite escalation runs the full UNIT suite (tests/unit/), not all of
#     tests/. Infra's tests/integration, tests/chaos, tests/replay and
#     tests/performance need a live runtime (Docker/Postgres/Kafka) and stay a
#     CI-only concern; the pre-push subset is unit-scoped by design.
#
# FAIL-LOUD (CLAUDE.md Rule #8): if the diff base, the selector, or its adjacency
# config cannot resolve, this hook HARD-ERRORS with a remediation message and a
# non-zero exit. It never degrades to a green skip -- a gate that cannot run must
# be indistinguishable from a failing gate. (Verified by the landed
# precommit-fail-loud-meta-gate, which scans this script.)
#
# Env overrides (all optional):
#   PREPUSH_BASE_REF     git ref to diff against            (default: origin/dev)
#   PREPUSH_ADJACENCY    adjacency yaml path            (default: selector built-in)
#   PREPUSH_PYTEST_ARGS  extra args appended to the pytest invocation
#   ENABLE_SMART_TESTS   set false/0/off to force the FULL suite (parity with the
#                        CI var name); default here is smart selection ON, because
#                        the whole point of the local hook is the impacted subset.
#   PREPUSH_FULL_SUITE   set non-empty to force the FULL suite.
#
# NOT an env override (OMN-16480): the host/capacity escape hatch. Any
# `PREPUSH_ALLOW_*` variable found in the environment is REJECTED at entry --
# see the rejection block below. The override is a single-use, repo+HEAD-scoped,
# TTL-bounded, receipted grant minted with
# `uv run python scripts/hooks/prepush_override_grant.py mint --reason '<why>'`.
# Note the asymmetry: the two knobs above can only make the hook run MORE tests,
# so they are not bypasses; an ALLOW override makes it accept WEAKER evidence,
# which is why it may not be ambient.

set -euo pipefail

log() { printf '[prepush-smart-tests] %s\n' "$1" >&2; }
die() {
  log "ERROR: $1"
  log "REMEDIATION: $2"
  exit 1
}

# =============================================================================
# Recursion guard (OMN-16489, F-01)
# =============================================================================
# This hook spawns pytest, and the spawned suite contains tests that exec THIS
# script again (tests/ci/test_prepush_hook_host_identity_guard.py and
# siblings). OMN-16425 proved one leaked override var turns that re-entry into
# a recursive full-suite launcher (~9h03m lost across 5 failed ~1h45m runs;
# friction report F-01) — and its fix covered the test sites, not the hook.
# The env scrub at the pytest invocations below closes the override-
# inheritance vector; this sentinel closes the re-entry class itself: a nested
# invocation refuses fail-closed before the selector resolves or any pytest
# spawns. The sentinel deliberately survives the override scrub — children
# must inherit it for this guard to hold. A test that intends to exercise this
# script's FIRST-entry behavior must strip ONEX_PREPUSH_HOOK_ACTIVE from the
# subprocess env it constructs.
if [ -n "${ONEX_PREPUSH_HOOK_ACTIVE:-}" ]; then
  die "nested invocation refused: this hook is already active in an ancestor process (ONEX_PREPUSH_HOOK_ACTIVE=${ONEX_PREPUSH_HOOK_ACTIVE}, this pid $$)" \
      "a pre-push hook run must never be spawned from inside another pre-push hook run (OMN-16425 recursion class). If a test means to exercise first-entry behavior, construct the subprocess env explicitly and strip ONEX_PREPUSH_HOOK_ACTIVE"
fi
export ONEX_PREPUSH_HOOK_ACTIVE="$$"

# =============================================================================
# Inheritable env-var gate overrides are REJECTED AT ENTRY (OMN-16480)
# =============================================================================
# This gate's escape hatch used to BE an environment variable
# (`PREPUSH_ALLOW_LOCAL_FULL_SUITE=1`). An environment variable is inherited by
# every descendant process, is bound to no repo/commit/run, never expires, and
# leaves no receipt -- so "permission to bypass the load gate once, for this
# push" was really "permission for every process this shell ever spawns to
# bypass this gate, silently". Same failure shape Rule 10 was hardened against
# for `[skip-*` tokens (OMN-9731 / OMN-13388), one layer down.
#
# Measured: on 2026-08-23 that variable leaked from an operator shell into a
# guard test's `env=dict(os.environ)` subprocess copy; this hook took its
# degraded-override branch and recursively launched another full 44,064-test
# suite, which reached the same test and recursed again -- ~9h03m, ~72% of all
# serialized suite wall-clock in that window (friction report F-01/F-04).
# Compliance was PERFECT that night: zero `[skip-*`, zero `--no-verify`. The
# damage came from the sanctioned escape path being used correctly.
#
# So the variable is no longer an arming signal in either direction: its
# presence is a HARD REFUSAL, not a bypass. That is what makes inheritance
# harmless -- a leaked override can no longer arm anything, and it surfaces
# immediately instead of silently disarming the gate for a whole process tree.
# The supported path is a single-use, repo+HEAD-scoped, TTL-bounded, receipted
# grant token: scripts/hooks/prepush_override_grant.py.
#
# Matched by PREFIX, not by one exact name, so a future
# `PREPUSH_ALLOW_SOMETHING_ELSE` cannot quietly reopen the class.
reject_inherited_env_overrides() {
  local leaked
  leaked="$(env | sed -n 's/^\(PREPUSH_ALLOW_[A-Za-z0-9_]*\)=..*/\1/p' | sort -u | tr '\n' ' ')"
  leaked="${leaked% }"
  [ -n "$leaked" ] || return 0
  die "inheritable gate-override environment variable(s) present: ${leaked} -- these are REJECTED, never honored (OMN-16480)" \
      "unset them in this shell (e.g. \`unset ${leaked%% *}\`), then, if this run genuinely must proceed on this host, mint a scoped single-use grant: \`uv run python scripts/hooks/prepush_override_grant.py mint --reason '<why>'\`. The grant is bound to this repo and this HEAD sha, expires in minutes, is consumed by the first guard that reads it (so no child process can reuse it), and appends a receipt line to .onex_state/prepush_override/receipts.jsonl"
}
reject_inherited_env_overrides

# consume_override_grant CONTEXT -- 0 when a valid single-use grant was claimed
# for this run, 1 otherwise. Delegates to the one implementation
# (scripts/hooks/prepush_override_grant.py) that the pytest-side guard also
# uses, so the two entry points can never drift apart on what a valid grant is.
# Routed through `uv run` per the OMN-14953 pinned-interpreter gate.
consume_override_grant() {
  uv run python "${REPO_ROOT}/scripts/hooks/prepush_override_grant.py" \
    consume --context "$1"
}

# =============================================================================
# .200-default host guard for the heavy (full-suite) escalation (OMN-15059)
# =============================================================================
# CLAUDE.md documents that pushes / heavy gate runs default to the `.200`
# execution host, not the local Mac -- but a rule stated only in a doc/prompt
# has zero enforcement force without a call-site mechanism (memory
# feedback_a_rule_is_not_a_mechanism). Evidence this is load-bearing: a
# 2026-07-24 session drove the local Mac to load ~55 / 93% swap running this
# exact full-suite escalation for 115+ minutes before .200 was invoked as a
# rescue instead of having been the execution target from the start. This
# guard fires ONLY on the heavy branch below (full-suite fail-closed
# escalation), never on the fast impacted-subset path -- gating every push
# would get this hook disabled within a week, which is worse than no guard.
#
# An UNRESOLVABLE hostname fails CLOSED (OMN-16489 defect 3, redesign plan
# 2026-08-24 §4 S0 item 3 / C2 — supersedes the earlier fail-open note here).
# Heavy runs are routed BY host identity; a host that cannot be identified
# cannot be routed, and proceeding on that silence is the same assumed-
# headroom failure class as the load-probe incidents below. The refusal is
# cheap (<1s, before any pytest) and names its remediation, consistent with
# this hook's fail-loud doctrine: a gate that cannot run must be
# indistinguishable from a failing gate.
PREPUSH_200_HOSTNAME="${PREPUSH_200_HOSTNAME:-stickybeatz-studio}"

# =============================================================================
# Live-load host selection (OMN-16295)
# =============================================================================
# Extends the host-IDENTITY guard below with a CAPACITY dimension: `.200`
# being the right host by IDENTITY does not mean it has headroom. Measured
# 2026-08-20: `.200` load average 32-34 against 24 cores (and, live during
# this same investigation, 56/24 -- 2.3x oversubscribed) driving an 89-93
# minute full-suite run with orphaned pytest processes left behind. Same
# failure class as the 2026-07-24 incident described below, recurring under
# concurrent-session load. OMN-16295 adds a second execution target -- a
# hard-capped gate-runner container on `.201`
# (docker/docker-compose.gate-runner.yml, this repo), selected ONLY when
# `.200` is over threshold, never the default.
#
# FAIL-CLOSED, unlike the host-IDENTITY guard's fail-open posture below --
# deliberately different, not inconsistent. An unresolvable HOSTNAME is
# ambiguous evidence about WHERE we are (fail open: don't lock a developer out
# of their own repo on a shaky read). An unresolvable LOAD reading is a
# failure to prove EITHER candidate host has capacity, and proceeding anyway
# on that silence is exactly how the 2026-07-24 / 2026-08-20 incidents
# happened -- assumed headroom that was not there. "Neither host reachable"
# refuses; it does not skip the check.
PREPUSH_201_GATE_RUNNER_HOSTNAME="${PREPUSH_201_GATE_RUNNER_HOSTNAME:-gate-runner-201}"
PREPUSH_200_SSH_TARGET="${PREPUSH_200_SSH_TARGET:-jonah@stickybeatz-studio.tail75df5e.ts.net}"  # onex-allow-internal-ip OMN-16295 reason="pre-push guard needs the real host target to probe live load"
PREPUSH_201_SSH_TARGET="${PREPUSH_201_SSH_TARGET:-jonah@192.168.86.201}"  # onex-allow-internal-ip OMN-16295 reason="pre-push guard needs the real host target to probe live load" # fallback-ok: real .201 host target, not a dev/local placeholder
# load1/cores at or under this ratio counts as "fit". 1.0 == "not
# oversubscribed" (a standard load-average heuristic); correctly reads the
# observed-fit `.201` snapshot (~0.4x, 2026-08-20) as fit and both observed
# `.200` snapshots above (1.33x and 2.3x) as over threshold.
PREPUSH_LOAD_THRESHOLD="${PREPUSH_LOAD_THRESHOLD:-1.0}"

# Cross-platform (Linux `.201` / macOS `.200`) load probe, printing
# "<load1> <nproc>". Deliberately interpreter-free: OMN-14953's pinned-
# interpreter gate (tests/ci/test_prepush_hook_pinned_interpreter.py) requires
# every python/python3 invocation under scripts/hooks/ to route through
# `uv run`, and the two ssh branches below cannot -- `.201` has no `uv` binary
# at all (probed 2026-08-20). Dropping the interpreter satisfies that gate
# rather than carving an exception out of it, and keeps interpreter startup
# off the pre-push critical path.
#
# Two portability constraints, both load-bearing:
#   1. Field extraction uses cut(1), NOT `set -- $(...)` word splitting.
#      `.200`'s remote login shell is zsh, which does not word-split unquoted
#      command substitution, so `set --` would collapse the whole line into $1
#      there while working fine on `.201`'s bash.
#   2. This snippet is handed to ssh(1) as the remote command and executed by
#      whatever login shell the remote user has, so it stays POSIX and carries
#      no single quotes (it is itself a single-quoted assignment here).
# shellcheck disable=SC2016  # intentionally unexpanded: evaluated by the local
# `sh -c` / the remote login shell, not by this script.
_PREPUSH_LOAD_PROBE_SH='n=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)
[ "$n" -gt 0 ] || exit 1
if [ -r /proc/loadavg ]; then
  l=$(cut -d" " -f1 /proc/loadavg)
else
  l=$(sysctl -n vm.loadavg 2>/dev/null | cut -d" " -f2)
fi
[ -n "$l" ] || exit 1
printf "%s %s\n" "$l" "$n"'

# Prefer GNU coreutils timeout(1); fall back to gtimeout(1) (Homebrew name on
# macOS); fall back to no wrapper at all (ssh -o ConnectTimeout already bounds
# the connection phase, and the remote command is a single fast shell probe).
_prepush_timeout_cmd() {
  if command -v timeout > /dev/null 2>&1; then
    printf 'timeout'
  elif command -v gtimeout > /dev/null 2>&1; then
    printf 'gtimeout'
  fi
}

# host_load_ratio TARGET -- prints "<load1> <nproc> <ratio>" and returns 0, or
# prints nothing and returns 1 on any read/parse/timeout failure. TARGET is
# empty for "read this host directly" or an ssh(1) target string for a
# bounded remote read. Deterministic, network-free overrides for tests (each a
# "<load1> <nproc>" pair -- the ratio is still computed from it, never
# hardcoded):
#   PREPUSH_LOAD_OVERRIDE_LOCAL   overrides the direct (TARGET="") read
#   PREPUSH_LOAD_OVERRIDE_REMOTE  overrides every ssh-target read
host_load_ratio() {
  local target="$1" raw load1 ncpu timeout_cmd
  if [ -z "$target" ]; then
    if [ -n "${PREPUSH_LOAD_OVERRIDE_LOCAL:-}" ]; then
      raw="$PREPUSH_LOAD_OVERRIDE_LOCAL"
    else
      raw="$(sh -c "$_PREPUSH_LOAD_PROBE_SH" 2> /dev/null)" || return 1
    fi
  else
    if [ -n "${PREPUSH_LOAD_OVERRIDE_REMOTE:-}" ]; then
      raw="$PREPUSH_LOAD_OVERRIDE_REMOTE"
    else
      timeout_cmd="$(_prepush_timeout_cmd)"
      if [ -n "$timeout_cmd" ]; then
        raw="$("$timeout_cmd" 6 ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
          "$target" "$_PREPUSH_LOAD_PROBE_SH" 2> /dev/null)" || return 1
      else
        raw="$(ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
          "$target" "$_PREPUSH_LOAD_PROBE_SH" 2> /dev/null)" || return 1
      fi
    fi
  fi
  [ -n "$raw" ] || return 1
  # shellcheck disable=SC2086
  set -- $raw
  load1="${1:-}"
  ncpu="${2:-}"
  [ -n "$load1" ] && [ -n "$ncpu" ] && [ "$ncpu" != "0" ] || return 1
  awk -v l="$load1" -v n="$ncpu" 'BEGIN { if (n + 0 <= 0) exit 1; printf "%s %s %.3f\n", l, n, (l / n) }'
}

# host_is_fit TARGET -- 0 if measured load1/nproc is at/under
# PREPUSH_LOAD_THRESHOLD, 1 if over threshold, 2 if the read itself failed
# (unreachable/unresolvable). Callers must not conflate 1 and 2 anywhere the
# difference is user-visible ("over capacity" vs "could not check").
host_is_fit() {
  local target="$1" ratio
  ratio="$(host_load_ratio "$target" | awk '{print $3}')" || return 2
  [ -n "$ratio" ] || return 2
  awk -v r="$ratio" -v thr="$PREPUSH_LOAD_THRESHOLD" 'BEGIN { exit !(r <= thr + 0) }'
}

guard_full_suite_host() {
  local host lc_host lc_target lc_201 heavy_what
  # OMN-15408: the caller names WHICH heavyweight run is being guarded, so the
  # refusal names the real cause. Default preserves the OMN-15059 wording for
  # the flag-driven escalation call sites, which pass no argument.
  heavy_what="${1:-heavy fail-closed full-suite escalation}"
  host="$(hostname -s 2>/dev/null || true)"
  if [ -z "$host" ]; then
    # Fail CLOSED (OMN-16489): see the routing note above PREPUSH_200_HOSTNAME.
    die "could not determine the local hostname while deciding where ${heavy_what} may run" \
        "heavy gate runs are routed by host identity (OMN-15059) and an unidentifiable host cannot be routed. Fix 'hostname -s' (macOS: 'sudo scutil --set HostName <name>'; Linux: 'hostnamectl set-hostname <name>'), or run the push from a designated gate host (.200 '${PREPUSH_200_HOSTNAME}' or the .201 gate-runner '${PREPUSH_201_GATE_RUNNER_HOSTNAME}')"
  fi
  lc_host="$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]')"
  lc_target="$(printf '%s' "$PREPUSH_200_HOSTNAME" | tr '[:upper:]' '[:lower:]')"
  lc_201="$(printf '%s' "$PREPUSH_201_GATE_RUNNER_HOSTNAME" | tr '[:upper:]' '[:lower:]')"
  if [ "$lc_host" = "$lc_target" ] || [ "$lc_host" = "$lc_201" ]; then
    # OMN-16295: identity alone is not enough -- this known-good host must
    # also have capacity right now.
    if host_is_fit ""; then
      return 0
    fi
    if consume_override_grant "degraded-capacity: ${heavy_what} on '${host}' at/over the ${PREPUSH_LOAD_THRESHOLD}x-core load threshold"; then
      log "WARNING: DEGRADED-CAPACITY OVERRIDE IN EFFECT (single-use grant consumed) -- running ${heavy_what} on '${host}' at/over the ${PREPUSH_LOAD_THRESHOLD}x-core load threshold. Treat any evidence from this run as WEAKER than a fit-host-run gate."
      return 0
    fi
    local other_target other_label other_rc other_note
    if [ "$lc_host" = "$lc_target" ]; then
      other_target="$PREPUSH_201_SSH_TARGET"
      other_label="the .201 gate-runner (${PREPUSH_201_GATE_RUNNER_HOSTNAME})"
    else
      other_target="$PREPUSH_200_SSH_TARGET"
      other_label=".200 (${PREPUSH_200_HOSTNAME})"
    fi
    other_rc=0
    host_is_fit "$other_target" || other_rc=$?
    case "$other_rc" in
      0) other_note="${other_label} currently HAS capacity -- route there instead" ;;
      2) other_note="${other_label} could not be reached to check capacity" ;;
      *) other_note="${other_label} is ALSO at/over the load threshold" ;;
    esac
    die "${heavy_what} triggered on '${host}' (the designated host by identity), but its load is at/over the ${PREPUSH_LOAD_THRESHOLD}x-core threshold" \
        "${other_note}. See docs/runbooks/200-build-lane-execution-pattern.md for the .201 gate-runner recipe, or mint a single-use grant to run here anyway (degraded evidence -- do not use as a routine bypass): uv run python scripts/hooks/prepush_override_grant.py mint --reason '<why>'"
  fi
  if consume_override_grant "degraded-host: ${heavy_what} on '${host}', not the designated .200 host '${PREPUSH_200_HOSTNAME}'"; then
    log "WARNING: DEGRADED-HOST OVERRIDE IN EFFECT (single-use grant consumed) -- running ${heavy_what} on '${host}', NOT the designated .200 host ('${PREPUSH_200_HOSTNAME}'). This host has weaker isolation/headroom than .200; treat any evidence from this run as WEAKER than a .200-run gate. See docs/runbooks/200-build-lane-execution-pattern.md."
    return 0
  fi
  die "${heavy_what} triggered on host '${host}', not the designated .200 build host ('${PREPUSH_200_HOSTNAME}')" \
      "push from .200 instead (ssh jonah@stickybeatz-studio.tail75df5e.ts.net, wrap remote commands as zsh -lc \"...\"; see docs/runbooks/200-build-lane-execution-pattern.md for the full pattern), OR mint a single-use override grant to run the full suite on this host anyway (visible, receipted, degraded-evidence override -- do not use as a routine bypass): uv run python scripts/hooks/prepush_override_grant.py mint --reason '<why>'"
}

# -----------------------------------------------------------------------------
# Heavyweight-SELECTION predicate (OMN-15408)
# -----------------------------------------------------------------------------
# The OMN-15059 guard above was wired to fire on the selector's `is_full_suite`
# FLAG. That is the wrong key: the selector routinely emits
# `is_full_suite=False` with a whole-suite path set -- the entire suite
# arriving as an "impacted subset" -- and those runs sail straight past the
# guard. Measured on host `omnibook` through real `git push` runs on
# 2026-07-29: this repo selected `is_full_suite=False` and executed 2,429 tests
# in 245s locally with the guard never invoked, and omnimarket (identical
# predicate, `paths=[ tests/ ]`) executed 13,898 tests in 506s the same way.
# The SAME selected work forced via `PREPUSH_FULL_SUITE=1` (`is_full_suite=True
# reason=feature_flag_off paths=[ tests/ ]`) WAS refused, in this repo,
# verbatim. Identical cost, opposite outcome, decided by a flag.
#
# SEAM -- what "heavyweight selection" means, exactly: the selection is
# heavyweight when the paths pytest is about to be handed COVER THE ENTIRE
# full-suite target this hook would run on a fail-closed escalation
# (`$FULL_SUITE_TARGET` -- `tests/unit/` in this repo, since infra's escalation
# is unit-scoped by design; defined next to the pytest invocation below so the
# predicate and the actual run can never drift apart). Concretely: some
# selected path is `$FULL_SUITE_TARGET` itself or a directory ANCESTOR of it
# (so a bare `tests/` selection, which is strictly MORE than the escalation
# target, is caught too). That is "the selection failed to be a proper
# narrowing" expressed against the selector's own output -- NOT a parallel cost
# model, no test counting, no timing heuristic, nothing this hook does not
# already parse.
#
# The predicate is evaluated against the RUNNABLE path set (post
# `filter_prepush_runnable_paths`), i.e. exactly the argv pytest receives --
# never the pre-filter selection, which would let a deferred integration path
# influence a local-cost decision it has no bearing on.
#
# A genuine narrow selection (`tests/unit/scripts/`, a single test module) is
# strictly below the target and stays runnable locally -- the guard must not
# brick every push from a developer's machine, only the ones that are the
# full-suite run wearing a different label.
#
# Keep this function self-contained (target passed in, no globals): it is
# extracted and EXECUTED directly by
# tests/ci/test_prepush_hook_host_identity_guard.py.
selection_is_whole_suite() {
  local target normalized_target p normalized
  target="$1"
  shift
  [ -n "$target" ] || return 1
  normalized_target="${target%/}/"
  for p in "$@"; do
    [ -n "$p" ] || continue
    normalized="${p%/}/"
    case "$normalized_target" in
      "$normalized"*) return 0 ;;
    esac
  done
  return 1
}

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || die "not inside a git worktree" \
         "run 'git push' from within the omnibase_infra repository"
cd "$REPO_ROOT"

# =============================================================================
# bash>=5 canary (OMN-15617)
# =============================================================================
# On stickybeatz-studio (.200, the rule-11a default gate host), non-interactive
# ssh sessions resolve `bash` to the system 3.2.57 shell even though a modern
# bash 5.x sits at /opt/homebrew/bin/bash -- it just is not first on PATH for
# that session class. runner-monitor.sh (exercised end-to-end by
# tests/unit/observability/runner_health/test_runner_monitor_*.py) uses
# `declare -A`, which bash 3.2 does not support, so those tests fail SILENTLY
# on every push from this host class -- a bash syntax error deep inside a
# subprocess, not a resolvable "wrong interpreter" diagnostic.
#
# Resolve a bash>=5 interpreter EXPLICITLY here, independent of PATH order,
# via the same resolver the pytest harness uses (single source of truth --
# scripts/ci/resolve_modern_bash.sh -- so the two can never drift apart), and
# export it so the harness does not have to re-discover it. Fail LOUD with a
# pointed remediation message if none is resolvable anywhere -- never a quiet
# skip, never a silent fallback to whatever "bash" happens to resolve first.
MODERN_BASH="$(bash "${REPO_ROOT}/scripts/ci/resolve_modern_bash.sh")" \
  || die "no bash>=5 interpreter resolvable on this host" \
         "install a modern bash (e.g. 'brew install bash') and/or set OMNIBASE_INFRA_BASH_BIN to its absolute path; see scripts/ci/resolve_modern_bash.sh"
export OMNIBASE_INFRA_BASH_BIN="$MODERN_BASH"
log "bash>=5 canary: resolved ${MODERN_BASH}"

BASE_REF="${PREPUSH_BASE_REF:-origin/dev}"

# Deterministic diff base: fetch the base ref best-effort so an online push gets
# an up-to-date merge-base, then REQUIRE it to resolve. Offline is tolerated ONLY
# when the ref already exists locally; an entirely unresolvable base HARD-ERRORS
# rather than silently diffing against nothing.
git fetch --quiet origin "${BASE_REF#origin/}" 2>/dev/null || true
if ! git rev-parse --verify --quiet "${BASE_REF}^{commit}" >/dev/null; then
  die "base ref '${BASE_REF}' could not be resolved" \
      "fetch it ('git fetch origin ${BASE_REF#origin/}') or set PREPUSH_BASE_REF to a resolvable ref"
fi

BASE_SHA="$(git merge-base "${BASE_REF}" HEAD 2>/dev/null)" \
  || die "no common ancestor between '${BASE_REF}' and HEAD" \
         "rebase your branch onto ${BASE_REF} so a merge-base exists"

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo HEAD)"

CHANGED_FILE="$(mktemp)"
SELECTION_FILE="$(mktemp)"
SELECTION_ERR="$(mktemp)"
trap 'rm -f "$CHANGED_FILE" "$SELECTION_FILE" "$SELECTION_ERR"' EXIT

git diff --name-only "${BASE_SHA}" HEAD > "$CHANGED_FILE"

# Feature-flag: default ON (impacted subset). Honor the CI var name and an
# explicit full-suite override. Neither knob is a silent bypass -- forcing OFF
# runs MORE tests (the whole suite), never fewer.
FLAG="on"
case "${ENABLE_SMART_TESTS:-}" in
  false | False | FALSE | 0 | off | OFF) FLAG="off" ;;
esac
if [ -n "${PREPUSH_FULL_SUITE:-}" ]; then
  FLAG="off"
fi

# DRY: invoke the EXACT module CI runs (scripts.ci.detect_test_paths) with the
# SAME flags .github/workflows/ci.yml passes -- crucially WITHOUT `--base-ref`,
# which infra's selector does not accept. Split on the optional adjacency
# override to avoid empty-array expansion under `set -u` on bash 3.2 (macOS
# system bash).
run_selector() {
  if [ -n "${PREPUSH_ADJACENCY:-}" ]; then
    uv run python -m scripts.ci.detect_test_paths \
      --changed-files-from "$CHANGED_FILE" \
      --ref-name "$BRANCH" \
      --event-name pull_request \
      --feature-flag "$FLAG" \
      --adjacency "$PREPUSH_ADJACENCY"
  else
    uv run python -m scripts.ci.detect_test_paths \
      --changed-files-from "$CHANGED_FILE" \
      --ref-name "$BRANCH" \
      --event-name pull_request \
      --feature-flag "$FLAG"
  fi
}

if ! run_selector > "$SELECTION_FILE" 2> "$SELECTION_ERR"; then
  log "selector stderr follows:"
  cat "$SELECTION_ERR" >&2 || true
  die "governed test selector failed to resolve a selection" \
      "verify scripts/ci/detect_test_paths.py + scripts/ci/test_selection_adjacency.yaml resolve under 'uv run' in this worktree"
fi

# Parse the selection with stdlib json -- fail loud on any parse error.
read_sel() {
  uv run python - "$SELECTION_FILE" "$1" << 'PY'
import json
import sys

with open(sys.argv[1]) as fh:
    data = json.load(fh)
val = data[sys.argv[2]]
if isinstance(val, list):
    print("\n".join(val))
else:
    print(val)
PY
}

IS_FULL="$(read_sel is_full_suite)" \
  || die "could not parse selector output (is_full_suite)" \
         "the selector emitted non-JSON; inspect $SELECTION_FILE"
REASON="$(read_sel full_suite_reason 2> /dev/null || true)"

# OMN-15245 SEAM: the selector now emits changed tests/integration/ paths -- a
# changed test module is never dropped by narrowing (fail-closed invariant).
# This hook is unit-scoped by design and passes --ignore=tests/integration to
# pytest below: handing pytest a path it also ignores collects nothing from it,
# and when it is the ONLY path pytest exits 5 ("no tests ran") and blocks the
# push. Filter those out here, visibly -- they are deferred to CI, which runs
# them. Keep this function self-contained (no globals): it is extracted and
# EXECUTED by tests/unit/scripts/test_prepush_smart_tests_seam.py.
filter_prepush_runnable_paths() {
  local p
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    case "$p" in
      tests/integration/*) continue ;;
    esac
    printf '%s\n' "$p"
  done
}

ALL_PATHS=()
while IFS= read -r p; do
  if [ -n "$p" ]; then
    ALL_PATHS+=("$p")
  fi
done < <(read_sel selected_paths)

PATHS=()
PATHS_STR=""
DEFERRED_STR=""
# Guard the array expansions: bash 3.2 (macOS system bash) errors on
# "${arr[@]}" for an empty array under `set -u`.
if [ "${#ALL_PATHS[@]}" -gt 0 ]; then
  while IFS= read -r p; do
    if [ -n "$p" ]; then
      PATHS+=("$p")
      PATHS_STR="${PATHS_STR}${p} "
    fi
  done < <(printf '%s\n' "${ALL_PATHS[@]}" | filter_prepush_runnable_paths)
  for p in "${ALL_PATHS[@]}"; do
    case " $PATHS_STR " in
      *" $p "*) ;;
      *) DEFERRED_STR="${DEFERRED_STR}${p} " ;;
    esac
  done
fi

log "selection: is_full_suite=${IS_FULL} reason=${REASON:-none} paths=[ ${PATHS_STR}] (feature-flag=${FLAG})"
if [ -n "$DEFERRED_STR" ]; then
  log "deferred to CI (integration needs live services; this hook is unit-scoped): [ ${DEFERRED_STR}]"
fi

# Assemble the pytest target set. tests/integration is always ignored -- it needs
# real services and stays a CI-only concern. On a fail-closed escalation we run
# the full UNIT suite (tests/unit/), NOT all of tests/, so the pre-push hook stays
# unit-scoped and service-free (infra seam-match).
RC=0
# SINGLE SOURCE OF TRUTH for "what the heavy run is" (OMN-15408): the
# fail-closed escalation runs exactly this target, and `selection_is_whole_suite`
# measures the impacted-subset selection against this same value. Changing the
# escalation target automatically moves the guard predicate with it.
FULL_SUITE_TARGET="tests/unit/"

# OMN-15071: git EXPORTS repo-scoping variables into hook processes -- a live
# `git push` from a worktree hands this hook
# `GIT_DIR=<common>/worktrees/<name>` -- and those variables OVERRIDE both `-C`
# and the cwd for every descendant `git` call (memory
# `reference_git_env_vars_override_c_and_cwd`). Tests that build throwaway
# repositories under `tmp_path` and commit into them therefore operate on THIS
# worktree instead, and fail at setup. Unset them for the test run only: the
# hook has already resolved everything it needs from git, and pytest must
# rediscover the repository from its own cwd like any ordinary invocation.
#
# This is not a latent nicety. Until OMN-15071 chained the canonical-clone
# guard into the real hook chain, `core.hooksPath` on `.200` meant this hook
# never executed there at all, so the leak had no observable effect on the
# documented default gate host. Turning the hook on without this unset would
# hand every `.200` push a fail-closed pre-push that cannot pass.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_COMMON_DIR GIT_PREFIX

# =============================================================================
# Override-inheritance sanitization (OMN-16489, F-04)
# =============================================================================
# PREPUSH_* overrides (and ENABLE_SMART_TESTS) are honored at THIS hook's
# entry only. They must never inherit into the pytest subprocess tree: a test
# down there that re-invokes this script would receive the OUTER push's bypass
# grants -- the exact mechanism that turned one sanctioned override into a
# recursive 44k-test full-suite launcher (friction report F-01/F-04, ~9h03m).
# Called inside the subshell wrapping each pytest invocation, after the
# command's own knobs have been captured into non-PREPUSH names, so the parent
# hook's variables are untouched. Only EXPORTED names can inherit, so only
# those are scrubbed. ONEX_PREPUSH_HOOK_ACTIVE deliberately survives -- the
# recursion guard above depends on children inheriting it. This stops
# inheritance ONLY. OMN-16480 (landed) separately rejects PREPUSH_ALLOW_* at
# hook entry; this scrub covers the whole PREPUSH_* prefix class one layer
# deeper, at the child boundary.
scrub_prepush_override_env() {
  local v
  for v in $(compgen -A export PREPUSH_ || true); do
    unset "$v" || true
  done
  unset ENABLE_SMART_TESTS || true
}

if [ "$IS_FULL" = "True" ] || [ "$IS_FULL" = "true" ]; then
  guard_full_suite_host
  log "running FULL unit suite (fail-closed escalation): uv run pytest ${FULL_SUITE_TARGET} --ignore=tests/integration ${PREPUSH_PYTEST_ARGS:-}"
  (
    _pytest_extra_args="${PREPUSH_PYTEST_ARGS:-}"
    scrub_prepush_override_env
    # shellcheck disable=SC2086
    exec uv run pytest "${FULL_SUITE_TARGET}" --ignore=tests/integration --tb=short ${_pytest_extra_args}
  ) || RC=$?
elif [ "${#PATHS[@]}" -gt 0 ]; then
  # OMN-15408: guard on the SELECTED WORK, not the is_full_suite flag. A
  # selection that covers the whole full-suite target is the heavy run under
  # another name and must be routed to .200 exactly as the flagged escalation is.
  if selection_is_whole_suite "$FULL_SUITE_TARGET" "${PATHS[@]}"; then
    guard_full_suite_host "whole-suite-equivalent impacted selection (is_full_suite=${IS_FULL}, selected paths [ ${PATHS_STR}] cover the entire '${FULL_SUITE_TARGET}' escalation target)"
  fi
  log "running impacted subset: uv run pytest ${PATHS_STR}--ignore=tests/integration ${PREPUSH_PYTEST_ARGS:-}"
  (
    _pytest_extra_args="${PREPUSH_PYTEST_ARGS:-}"
    scrub_prepush_override_env
    # shellcheck disable=SC2086
    exec uv run pytest "${PATHS[@]}" --ignore=tests/integration --tb=short ${_pytest_extra_args}
  ) || RC=$?
else
  log "no impacted unit tests mapped for this push (no source/test change contributed a target); nothing to run."
fi

if [ "$RC" -ne 0 ]; then
  log "ERROR: impacted tests failed (pytest exit ${RC})"
  log "REMEDIATION: fix the failing tests, then re-push. Reproduce with: uv run pytest ${PATHS_STR:-tests/unit/} --ignore=tests/integration"
  exit "$RC"
fi

log "impacted tests passed; allowing push."
exit "$RC"
