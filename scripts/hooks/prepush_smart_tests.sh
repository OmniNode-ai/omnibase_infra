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

set -euo pipefail

log() { printf '[prepush-smart-tests] %s\n' "$1" >&2; }
die() {
  log "ERROR: $1"
  log "REMEDIATION: $2"
  exit 1
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
# This is a ROUTING OPTIMIZATION, not a security control: if host identity
# cannot be determined, FAIL OPEN (let the push proceed on this host) rather
# than lock a developer out of their own repo on an ambiguous read. Do not
# "harden" this into a hard block later -- the failure mode this guard exists
# to prevent is a stalled/contended local machine, not an untrusted push.
PREPUSH_200_HOSTNAME="${PREPUSH_200_HOSTNAME:-stickybeatz-studio}"
guard_full_suite_host() {
  local host lc_host lc_target heavy_what
  # OMN-15408: the caller names WHICH heavyweight run is being guarded, so the
  # refusal names the real cause. Default preserves the OMN-15059 wording for
  # the flag-driven escalation call sites, which pass no argument.
  heavy_what="${1:-heavy fail-closed full-suite escalation}"
  host="$(hostname -s 2>/dev/null || true)"
  if [ -z "$host" ]; then
    log "WARNING: could not determine local hostname -- unable to verify this is the .200 build host; proceeding locally (fail-open: this guard is a routing optimization, not a security gate)."
    return 0
  fi
  lc_host="$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]')"
  lc_target="$(printf '%s' "$PREPUSH_200_HOSTNAME" | tr '[:upper:]' '[:lower:]')"
  if [ "$lc_host" = "$lc_target" ]; then
    return 0
  fi
  if [ -n "${PREPUSH_ALLOW_LOCAL_FULL_SUITE:-}" ]; then
    log "WARNING: DEGRADED-HOST OVERRIDE IN EFFECT (PREPUSH_ALLOW_LOCAL_FULL_SUITE set) -- running ${heavy_what} on '${host}', NOT the designated .200 host ('${PREPUSH_200_HOSTNAME}'). This host has weaker isolation/headroom than .200; treat any evidence from this run as WEAKER than a .200-run gate. See docs/runbooks/200-build-lane-execution-pattern.md."
    return 0
  fi
  die "${heavy_what} triggered on host '${host}', not the designated .200 build host ('${PREPUSH_200_HOSTNAME}')" \
      "push from .200 instead (ssh jonah@stickybeatz-studio.tail75df5e.ts.net, wrap remote commands as zsh -lc \"...\"; see docs/runbooks/200-build-lane-execution-pattern.md for the full pattern), OR set PREPUSH_ALLOW_LOCAL_FULL_SUITE=1 to run the full suite on this host anyway (visible, degraded-evidence override -- do not use as a routine bypass)"
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

if [ "$IS_FULL" = "True" ] || [ "$IS_FULL" = "true" ]; then
  guard_full_suite_host
  log "running FULL unit suite (fail-closed escalation): uv run pytest ${FULL_SUITE_TARGET} --ignore=tests/integration ${PREPUSH_PYTEST_ARGS:-}"
  # shellcheck disable=SC2086
  uv run pytest "${FULL_SUITE_TARGET}" --ignore=tests/integration --tb=short ${PREPUSH_PYTEST_ARGS:-} || RC=$?
elif [ "${#PATHS[@]}" -gt 0 ]; then
  # OMN-15408: guard on the SELECTED WORK, not the is_full_suite flag. A
  # selection that covers the whole full-suite target is the heavy run under
  # another name and must be routed to .200 exactly as the flagged escalation is.
  if selection_is_whole_suite "$FULL_SUITE_TARGET" "${PATHS[@]}"; then
    guard_full_suite_host "whole-suite-equivalent impacted selection (is_full_suite=${IS_FULL}, selected paths [ ${PATHS_STR}] cover the entire '${FULL_SUITE_TARGET}' escalation target)"
  fi
  log "running impacted subset: uv run pytest ${PATHS_STR}--ignore=tests/integration ${PREPUSH_PYTEST_ARGS:-}"
  # shellcheck disable=SC2086
  uv run pytest "${PATHS[@]}" --ignore=tests/integration --tb=short ${PREPUSH_PYTEST_ARGS:-} || RC=$?
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
