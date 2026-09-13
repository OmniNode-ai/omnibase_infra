#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Canonical-clone worktree-discipline guard, chained into the real hook chain
# (OMN-7018 guard + OMN-15071 chaining fix).
#
# Installed on a host by pointing `core.hooksPath` at the sibling
# `canonical-clone/` directory, whose per-hook-type symlinks all resolve to this
# script. `core.hooksPath` REPLACES git's hook lookup outright -- git never falls
# back to `$GIT_COMMON_DIR/hooks/` -- so anything this script does not explicitly
# invoke simply does not run.
#
# OMN-15071: the pre-chaining revision of this guard `exit 0`-ed for every
# worktree path, which meant that on `.200` -- the host root CLAUDE.md rule 11a
# makes the DEFAULT target for pushes and gate runs -- EVERY `git commit` in a
# worktree ran ZERO hooks and reported success. Silently: the guard printed
# nothing, so a clean commit was indistinguishable from a commit that had passed
# every gate. Concrete false negative (2026-07-30): a pattern-ratchet violation
# was correctly rejected on the Mac (no `core.hooksPath` override there) and
# committed clean on `.200`; it was caught by ordering luck, not by a gate.
#
# Behaviour:
#   1. Commits in a registry canonical clone are refused (the OMN-7018 rule:
#      canonical clones are pull/index mirrors, work happens in a worktree).
#   2. Everywhere the guard permits the operation, it CHAINS to the real hook of
#      the same type instead of returning success on its own -- the installed
#      pre-commit-framework hook when one is present, otherwise `pre-commit
#      hook-impl` directly when the repo carries a `.pre-commit-config.yaml`.
#   3. If the repo has a pre-commit config but no runnable pre-commit at all,
#      the hook FAILS CLOSED rather than reporting a vacuous success.
#
# Escape hatch (pre-existing, unchanged): `ALLOW_CANONICAL_CLONE_COMMIT=1`
# suppresses the canonical-clone refusal only. It does not skip the chain.

set -euo pipefail

hook_name="$(basename "$0")"

# --- shared repository-position helpers (OMN-16497) -------------------------
#
# resolve_path / absolutize / is_under / canonical_clone_context live beside
# this script so the ref-transaction guard in the same hooks directory answers
# "is this a canonical clone?" from the SAME code. Two copies of that answer
# would drift, and a drifted copy fails in the dangerous direction: a guard
# that misreads a canonical clone as a worktree permits exactly what it was
# installed to refuse.

_self_script="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"
while [[ -L "$_self_script" ]]; do
  _self_link="$(readlink "$_self_script")"
  case "$_self_link" in
    /*) _self_script="$_self_link" ;;
    *) _self_script="$(cd "$(dirname "$_self_script")" && pwd -P)/$_self_link" ;;
  esac
done
# shellcheck source=./canonical_clone_paths.sh
source "$(cd "$(dirname "$_self_script")" && pwd -P)/canonical_clone_paths.sh"

# --- repository facts -------------------------------------------------------

top_level=""
# shellcheck disable=SC2034  # populated by canonical_clone_context; part of its contract
git_dir=""
git_common_dir=""
# shellcheck disable=SC2034  # populated by canonical_clone_context; part of its contract
is_linked_worktree=0
omni_home=""
is_canonical_clone=0
if ! canonical_clone_context; then
  # Not inside a work tree (bare repo, or git internals) -- nothing to guard.
  exit 0
fi

# --- decision ---------------------------------------------------------------

if [[ "$is_canonical_clone" == "1" ]]; then
  if [[ "${ALLOW_CANONICAL_CLONE_COMMIT:-}" != "1" ]]; then
    cat >&2 <<EOF
ERROR: blocked $hook_name in canonical clone:
  $top_level

Canonical clones are pull/index mirrors. Create a worktree under:
  $omni_home/omni_worktrees/<ticket>/<repo>

Override only for an intentional emergency:
  ALLOW_CANONICAL_CLONE_COMMIT=1 git ...
EOF
    exit 1
  fi
fi

# --- chain to the real hook (OMN-15071) -------------------------------------
#
# Reaching here means the worktree-discipline guard permits the operation. It
# does NOT mean the operation is clean: the real hook chain still has to run,
# and `core.hooksPath` guarantees git will not run it for us.

# Hooks in this fleet shell out to `uv`, `python3` and friends. A git hook can
# be invoked from a non-login shell -- e.g. a `ssh <host> 'git commit ...'`,
# which is how agent lanes drive .200 -- whose PATH omits the package-manager
# prefixes. Without this, chaining would turn OMN-15071's silent pass into a
# blanket "Executable `uv` not found" refusal on every commit. Prefixes are
# APPENDED, so an explicitly-chosen toolchain earlier on PATH still wins.
for prefix in /opt/homebrew/bin /usr/local/bin; do
  if [[ -d "$prefix" ]] && [[ ":$PATH:" != *":$prefix:"* ]]; then
    PATH="$PATH:$prefix"
  fi
done
export PATH

self_real="$(resolve_path "$0")"
real_hook="$git_common_dir/hooks/$hook_name"

if [[ -x "$real_hook" ]] && [[ "$(resolve_path "$real_hook")" != "$self_real" ]]; then
  exec "$real_hook" "$@"
fi

# No installed hook of this type. If the repo declares a pre-commit config, the
# stage may still have hooks bound to it (`pre-commit install` REFUSES to write
# hook files while core.hooksPath is set, so "not installed" says nothing about
# "not configured"). Invoke pre-commit's own hook entry point -- the exact call
# the generated hook file makes.
if [[ ! -f "$top_level/.pre-commit-config.yaml" ]]; then
  exit 0
fi

precommit_cmd=()
if command -v pre-commit >/dev/null 2>&1; then
  precommit_cmd=("$(command -v pre-commit)")
else
  # A git hook can run under a non-login shell whose PATH omits the package
  # manager prefixes (observed on .200: PATH lacks /opt/homebrew/bin).
  for candidate in /opt/homebrew/bin/pre-commit /usr/local/bin/pre-commit; do
    if [[ -x "$candidate" ]]; then
      precommit_cmd=("$candidate")
      break
    fi
  done
fi

if [[ ${#precommit_cmd[@]} -eq 0 ]]; then
  for candidate in /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
      "$candidate" -c 'import pre_commit' >/dev/null 2>&1; then
      precommit_cmd=("$candidate" -m pre_commit)
      break
    fi
  done
fi

if [[ ${#precommit_cmd[@]} -eq 0 ]]; then
  cat >&2 <<EOF
ERROR: $hook_name cannot run the hook chain and refuses to report a vacuous pass.

  repo:        $top_level
  hooks dir:   $git_common_dir/hooks  (no executable '$hook_name')
  config:      $top_level/.pre-commit-config.yaml (present)
  pre-commit:  not resolvable on PATH or at the standard prefixes

core.hooksPath is set to this guard, so git will NOT fall back to the repo's own
hooks directory -- returning success here would silently skip every gate
(OMN-15071). Install pre-commit, or unset core.hooksPath for this repo.
EOF
  exit 1
fi

exec "${precommit_cmd[@]}" hook-impl \
  --config=.pre-commit-config.yaml \
  --hook-type="$hook_name" \
  --hook-dir "$git_common_dir/hooks" \
  -- "$@"
