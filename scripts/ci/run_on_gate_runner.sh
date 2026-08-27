#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# run_on_gate_runner.sh — run a command inside the `.201` gate-runner container
# against a specific worktree (OMN-16295 surface, landed by OMN-16752).
#
# WHY THIS EXISTS
# ---------------
# docker/docker-compose.gate-runner.yml has always documented this script as
# the invocation path ("scripts/ci/run_on_gate_runner.sh sets
# UV_PROJECT_ENVIRONMENT per-invocation relative to the worktree it targets"),
# and scripts/hooks/prepush_smart_tests.sh routes operators to the gate-runner
# when `.200` is over its load threshold — but the script itself was never
# committed. Every gate-runner run therefore had to be hand-rolled as
# `docker exec` + a hand-typed `UV_PROJECT_ENVIRONMENT`, which is exactly the
# kind of undocumented per-operator recipe that makes a gate unreproducible.
# OMN-16752 measured the cost: the escalation target could not be used at all
# without rediscovering the plumbing first.
#
# WHAT IT GUARANTEES
# ------------------
# 1. ONE venv per worktree, derived deterministically from the worktree's real
#    path, under a CONTAINER-ONLY prefix. Two worktrees of the same repo can
#    never share a venv (divergent dependency sets resolve into one another),
#    and a container venv can never collide with a host-built `.venv` whose
#    interpreter path/ABI differs.
# 2. The SYMLINK TRAP is handled once, here. On `.201`,
#    `/home/jonah/Code/omni_home` is a symlink to `/data/omninode/omni_home`,
#    and `git worktree add` stores the REAL path in each worktree's `.git`
#    pointer file. A bind mount at the symlink path leaves every worktree's
#    `.git` pointer dangling inside the container ("fatal: not a git
#    repository"), which the hook then reports as "not inside a git worktree" —
#    a confusing failure two layers away from its cause. Every path this script
#    handles is resolved with `readlink -f` before it crosses the boundary.
#
# Usage:
#   scripts/ci/run_on_gate_runner.sh [--sync] <worktree-path> <command> [args...]
#
#   --sync   run `uv sync --all-extras` in the worktree first (needed the first
#            time a worktree is seen, and after a dependency change).
#
# Examples:
#   scripts/ci/run_on_gate_runner.sh --sync "$OMNI_HOME/omni_worktrees/OMN-1234/omnimarket" \
#     uv run pytest tests/ -q
#   scripts/ci/run_on_gate_runner.sh "$OMNI_HOME/omni_worktrees/OMN-1234/omnimarket" \
#     uv run pytest tests/unit -q
#
# Environment:
#   GATE_RUNNER_CONTAINER   container name (default: omninode-gate-runner)
#   GATE_RUNNER_SSH_TARGET  when set, `docker` is invoked over ssh against this
#                           target instead of locally — lets `.200` drive the
#                           `.201` container directly. Default: empty (local).
#   GATE_RUNNER_VENV_ROOT   container-only venv prefix
#                           (default: /workspace/.venv-gate-runner)
#
# Exit codes: the command's own status is propagated verbatim, so this script
# is safe to use as a gate. Setup failures exit 2 and are always fail-closed —
# a gate that cannot run must be indistinguishable from a failing gate.
set -euo pipefail

GATE_RUNNER_CONTAINER="${GATE_RUNNER_CONTAINER:-omninode-gate-runner}"
GATE_RUNNER_SSH_TARGET="${GATE_RUNNER_SSH_TARGET:-}"
GATE_RUNNER_VENV_ROOT="${GATE_RUNNER_VENV_ROOT:-/workspace/.venv-gate-runner}"

die() {
  echo "run_on_gate_runner: $1" >&2
  if [ "$#" -gt 1 ]; then
    echo "  remediation: $2" >&2
  fi
  exit 2
}

DO_SYNC=0
if [ "${1:-}" = "--sync" ]; then
  DO_SYNC=1
  shift
fi

if [ "$#" -lt 2 ]; then
  die "usage: run_on_gate_runner.sh [--sync] <worktree-path> <command> [args...]" \
      "pass the worktree to run in, then the command to run there"
fi

WORKTREE_RAW="$1"
shift

# Resolve the symlink trap described in the header. `readlink -f` is GNU/macOS
# 12.3+; both designated hosts have it.
WORKTREE="$(readlink -f "${WORKTREE_RAW}" 2>/dev/null || true)"
[ -n "${WORKTREE}" ] || die "could not resolve worktree path '${WORKTREE_RAW}'" \
  "pass an existing path; it is resolved with 'readlink -f' before use"
[ -d "${WORKTREE}" ] || die "worktree '${WORKTREE}' is not a directory" \
  "create it first: git -C \"\$OMNI_HOME/<repo>\" worktree add <path> -b <branch>"

# Build the docker invocation. Kept as an array so no argument is ever re-split
# (the zsh/bash word-splitting class this repo's shell-hygiene gate exists for).
docker_cmd=(docker)
if [ -n "${GATE_RUNNER_SSH_TARGET}" ]; then
  docker_cmd=(ssh -o BatchMode=yes "${GATE_RUNNER_SSH_TARGET}" docker)
fi

if ! "${docker_cmd[@]}" inspect -f '{{.State.Running}}' "${GATE_RUNNER_CONTAINER}" 2>/dev/null | grep -qx true; then
  die "container '${GATE_RUNNER_CONTAINER}' is not running" \
      "bring it up: OMNI_HOME=\"\$(readlink -f \"\$OMNI_HOME\")\" docker compose -f docker/docker-compose.gate-runner.yml up -d"
fi

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

exec "${docker_cmd[@]}" exec \
  -e "UV_PROJECT_ENVIRONMENT=${venv_path}" \
  -w "${WORKTREE}" \
  "${GATE_RUNNER_CONTAINER}" \
  "$@"
