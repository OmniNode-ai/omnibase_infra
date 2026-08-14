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

# --- path helpers -----------------------------------------------------------

resolve_path() {
  # Portable realpath: macOS `readlink` has no -f. Resolves symlinks in the
  # final component, which is all this script needs (the hook symlinks).
  local target="$1"
  local dir base link hops=0
  while [[ -L "$target" ]]; do
    hops=$((hops + 1))
    if [[ "$hops" -gt 32 ]]; then
      printf 'ERROR: symlink loop resolving %s\n' "$1" >&2
      exit 1
    fi
    link="$(readlink "$target")"
    case "$link" in
      /*) target="$link" ;;
      *) target="$(dirname "$target")/$link" ;;
    esac
  done
  dir="$(cd "$(dirname "$target")" && pwd -P)"
  base="$(basename "$target")"
  printf '%s/%s\n' "$dir" "$base"
}

absolutize() {
  # git may hand back a relative --git-common-dir / --git-dir.
  local p="$1"
  case "$p" in
    /*) printf '%s\n' "$p" ;;
    *) printf '%s\n' "$(cd "$p" && pwd -P)" ;;
  esac
}

is_under() {
  # is_under <candidate> <ancestor> -- strict descendant, not equal.
  local candidate="$1" ancestor="$2"
  [[ -n "$ancestor" ]] || return 1
  [[ "$candidate" == "$ancestor"/* ]]
}

# --- repository facts -------------------------------------------------------

top_level="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$top_level" ]]; then
  # Not inside a work tree (bare repo, or git internals) -- nothing to guard.
  exit 0
fi
# Compare physical paths throughout: git_common_dir below is resolved with
# `pwd -P`, so a symlinked registry root would otherwise defeat the
# canonical-clone refusal by making the two sides incomparable.
top_level="$(cd "$top_level" && pwd -P)"

git_dir="$(absolutize "$(git rev-parse --git-dir)")"
git_common_dir="$(absolutize "$(git rev-parse --git-common-dir)")"

# A LINKED worktree has --git-dir == <common>/worktrees/<name>, so it differs
# from --git-common-dir. The MAIN worktree of a clone has them equal. This test
# is path-layout independent, which is what makes the guard correct on a host
# whose registry lives somewhere other than the documented path.
if [[ "$git_dir" != "$git_common_dir" ]]; then
  is_linked_worktree=1
else
  is_linked_worktree=0
fi

# Registry root. Prefer the explicit env var (root CLAUDE.md contract); fall
# back to the clone's own position -- `<registry>/<repo>/.git` -- which holds
# for a canonical clone AND for every worktree linked to it, because both share
# the same --git-common-dir. No hardcoded absolute paths (root CLAUDE.md #6).
omni_home="${OMNI_HOME:-}"
if [[ -z "$omni_home" ]]; then
  omni_home="$(cd "$git_common_dir/../.." && pwd -P)"
fi

# --- decision ---------------------------------------------------------------

allowed=0

if [[ "$top_level" == "$omni_home" ]]; then
  # The registry meta-repo itself commits directly to its docs branch.
  allowed=1
elif is_under "$top_level" "${ONEX_WORKTREES_ROOT:-}"; then
  allowed=1
elif is_under "$top_level" "$omni_home/omni_worktrees"; then
  allowed=1
elif is_under "$top_level" "$(dirname "$omni_home")/omni_worktrees"; then
  allowed=1
elif [[ "$is_linked_worktree" == "1" ]]; then
  allowed=1
fi

if [[ "$allowed" == "0" ]] && is_under "$top_level" "$omni_home"; then
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
