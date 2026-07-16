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
  python3 - "$SELECTION_FILE" "$1" << 'PY'
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

PATHS=()
PATHS_STR=""
while IFS= read -r p; do
  if [ -n "$p" ]; then
    PATHS+=("$p")
    PATHS_STR="${PATHS_STR}${p} "
  fi
done < <(read_sel selected_paths)

log "selection: is_full_suite=${IS_FULL} reason=${REASON:-none} paths=[ ${PATHS_STR}] (feature-flag=${FLAG})"

# Assemble the pytest target set. tests/integration is always ignored -- it needs
# real services and stays a CI-only concern. On a fail-closed escalation we run
# the full UNIT suite (tests/unit/), NOT all of tests/, so the pre-push hook stays
# unit-scoped and service-free (infra seam-match).
RC=0
if [ "$IS_FULL" = "True" ] || [ "$IS_FULL" = "true" ]; then
  log "running FULL unit suite (fail-closed escalation): uv run pytest tests/unit/ --ignore=tests/integration ${PREPUSH_PYTEST_ARGS:-}"
  # shellcheck disable=SC2086
  uv run pytest tests/unit/ --ignore=tests/integration --tb=short ${PREPUSH_PYTEST_ARGS:-} || RC=$?
elif [ "${#PATHS[@]}" -gt 0 ]; then
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
