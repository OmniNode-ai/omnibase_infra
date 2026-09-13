# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# shellcheck shell=bash
#
# Shared repository-position helpers for the canonical-clone hook family
# (OMN-7018 guard, OMN-15071 chaining fix, OMN-16497 ref-transaction guard).
#
# SOURCED, never executed. It defines functions and sets no policy: deciding
# what to do with a canonical clone belongs to the hook that sources this.
#
# ## Why this file exists
#
# `core.hooksPath` points every canonical clone at ONE shared directory whose
# per-hook-type symlinks resolve to the scripts beside this file. That makes the
# directory the single composition point for the whole family: a new hook TYPE
# is a new symlink pointing at its own script, never a replacement of the
# directory, and never a second installer writing over the first.
#
# The family has more than one member now, and every member has to answer the
# same question first -- "is the tree this hook was invoked in a canonical
# clone, or a worktree linked to one?" -- from the same evidence. Two copies of
# that answer would drift, and a drifted copy fails in the dangerous direction:
# a guard that misreads a canonical clone as a worktree permits exactly what it
# was installed to refuse. So the answer lives here, once.

# Portable realpath: macOS `readlink` has no -f. Resolves symlinks in the final
# component, which is all the hook symlinks need.
resolve_path() {
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

# canonical_clone_context
#
# Populates, in the caller's scope:
#   top_level           physical path of the invoking work tree ("" when there
#                       is none -- a bare repo or a git-internal invocation)
#   git_dir             absolute --git-dir
#   git_common_dir      absolute --git-common-dir
#   is_linked_worktree  1 when this is a LINKED worktree, 0 for a main one
#   omni_home           registry root
#   is_canonical_clone  1 when the invoking tree is a canonical clone that the
#                       family guards, 0 otherwise
#
# Returns 1 when there is no work tree at all, so a caller can `exit 0` early.
#
# A LINKED worktree has --git-dir == <common>/worktrees/<name>, so it differs
# from --git-common-dir; the MAIN worktree of a clone has them equal. That test
# is path-layout independent, which is what makes the family correct on a host
# whose registry lives somewhere other than the documented path.
canonical_clone_context() {
  top_level="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -z "$top_level" ]]; then
    is_canonical_clone=0
    return 1
  fi
  # Compare physical paths throughout: git_common_dir is resolved with `pwd -P`,
  # so a symlinked registry root would otherwise defeat the refusal by making
  # the two sides incomparable.
  top_level="$(cd "$top_level" && pwd -P)"

  git_dir="$(absolutize "$(git rev-parse --git-dir)")"
  git_common_dir="$(absolutize "$(git rev-parse --git-common-dir)")"

  if [[ "$git_dir" != "$git_common_dir" ]]; then
    is_linked_worktree=1
  else
    is_linked_worktree=0
  fi

  # Registry root. Prefer the explicit env var (root CLAUDE.md contract); fall
  # back to the clone's own position -- `<registry>/<repo>/.git` -- which holds
  # for a canonical clone AND for every worktree linked to it, because both
  # share the same --git-common-dir. No hardcoded absolute paths (rule 6).
  omni_home="${OMNI_HOME:-}"
  if [[ -z "$omni_home" ]]; then
    omni_home="$(cd "$git_common_dir/../.." && pwd -P)"
  fi

  local exempt=0
  if [[ "$top_level" == "$omni_home" ]]; then
    # The registry meta-repo itself commits directly to its docs branch.
    exempt=1
  elif is_under "$top_level" "${ONEX_WORKTREES_ROOT:-}"; then
    exempt=1
  elif is_under "$top_level" "$omni_home/omni_worktrees"; then
    exempt=1
  elif is_under "$top_level" "$(dirname "$omni_home")/omni_worktrees"; then
    exempt=1
  elif [[ "$is_linked_worktree" == "1" ]]; then
    exempt=1
  fi

  # shellcheck disable=SC2034  # the caller's out-parameter; read in every sourcing hook
  if [[ "$exempt" == "0" ]] && is_under "$top_level" "$omni_home"; then
    is_canonical_clone=1
  else
    is_canonical_clone=0
  fi
  return 0
}
