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
#   omni_home           registry root (the worktree root named in refusals
#                       is $omni_home/omni_worktrees)
#   registry_roots      array: every registry root whose clones are guarded,
#                       see registry_roots_context
#   is_canonical_clone  1 when the invoking tree is a canonical clone that the
#                       family guards, 0 otherwise
#   unresolved_root_note  non-empty when the clone was guarded by the position
#                       rule (no resolvable root); a refusal prints it
#
# Returns 1 when there is no work tree at all, so a caller can `exit 0` early.
# EXITS 1 (fails closed) when ONEX_REGISTRY_ROOTS or the roots file is malformed.
#
# A LINKED worktree has --git-dir == <common>/worktrees/<name>, so it differs
# from --git-common-dir; the MAIN worktree of a clone has them equal. That test
# is path-layout independent, which is what makes the family correct on a host
# whose registry lives somewhere other than the documented path.
# The roots file (OMN-19388 r2).
#
# `install-canonical-clone-git-hooks.sh --apply` writes `registry-roots` beside
# this library in the LIVE hooks directory: one `omni_home=<dir>` line and one
# `root=<dir>` line per registry root, every path absolute and physical. It is
# what makes the family hold with NO environment. The first revision read the
# roots from ONEX_REGISTRY_ROOTS alone, so an actor that never sourced the shell
# profile -- a launchd job, `bash -lc`, `env -i`, a GUI git client, any agent
# whose environment was built without it -- saw only OMNI_HOME, and a commit in
# a second-root clone was permitted without a word.
#
# The tracked source directory carries no roots file, and does not need one:
# the file only ever adds roots. Its absence means "no roots recorded", which
# the position rule in canonical_clone_context still covers.
_canonical_clone_library_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
registry_roots_file="$_canonical_clone_library_dir/registry-roots"

# read_registry_roots_file
#
# Populates, in the caller's scope, `file_omni_home` (may be empty) and the
# array `file_roots` from $registry_roots_file. Fails CLOSED (exit 1) on a file
# that exists but cannot be read, or on a line that is not `omni_home=<abs>`,
# `root=<abs>`, blank, or a `#` comment. A recorded root that no longer exists
# is skipped: there is nothing under a missing directory to guard, and a clone
# moved out of it is caught by the position rule instead.
read_registry_roots_file() {
  file_omni_home=""
  file_roots=()
  [[ -e "$registry_roots_file" ]] || return 0
  if [[ ! -r "$registry_roots_file" ]]; then
    printf 'ERROR: the registry roots file %s exists but cannot be read. Re-run install-canonical-clone-git-hooks.sh --apply.\n' "$registry_roots_file" >&2
    exit 1
  fi
  local line key value n=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    n=$((n + 1))
    case "$line" in
      "" | "#"*) continue ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    if [[ "$key" == "$line" ]] || [[ "$value" != /* ]] ||
      [[ "$key" != "omni_home" && "$key" != "root" ]]; then
      printf 'ERROR: the registry roots file %s line %s reads "%s"; every line must be omni_home=<absolute dir> or root=<absolute dir>. Re-run install-canonical-clone-git-hooks.sh --apply.\n' "$registry_roots_file" "$n" "$line" >&2
      exit 1
    fi
    if [[ "$key" == "omni_home" ]]; then
      file_omni_home="$value"
    elif [[ -d "$value" ]]; then
      file_roots+=("$(cd "$value" && pwd -P)")
    fi
  done <"$registry_roots_file"
  return 0
}

# registry_roots_context
#
# Populates, in the caller's scope, the array `registry_roots`: the physical
# path of every directory whose descendants are canonical clones (OMN-19388).
#
# It is the UNION of $omni_home, every ONEX_REGISTRY_ROOTS entry, and every root
# the roots file records. A union rather than a precedence, because the guarded
# set may only grow: an actor carrying a stale or narrower variable must not be
# able to un-guard a root the installer recorded.
#
# Fail-fast (rule 8): an ONEX_REGISTRY_ROOTS entry that is empty, relative, or
# not an existing directory is a misconfiguration, never a guess. The hook EXITS
# 1 naming the offending value, so every guarded git operation refuses loudly
# until it is fixed, rather than guarding a smaller set than the operator
# declared.
registry_roots_context() {
  registry_roots=()
  local physical_home
  if physical_home="$(cd "$omni_home" 2>/dev/null && pwd -P)"; then
    registry_roots+=("$physical_home")
  else
    registry_roots+=("$omni_home")
  fi
  registry_roots+=(${file_roots[@]+"${file_roots[@]}"})

  [[ -n "${ONEX_REGISTRY_ROOTS+set}" ]] || return 0
  local declared="$ONEX_REGISTRY_ROOTS"
  local entry physical rest="$declared"
  while :; do
    entry="${rest%%:*}"
    case "$entry" in
      /*) : ;;
      *)
        printf 'ERROR: ONEX_REGISTRY_ROOTS=%s holds the entry "%s", which is not an absolute path. Every entry must be an absolute registry-root directory, colon-separated; fix or unset the variable.\n' "$declared" "$entry" >&2
        exit 1
        ;;
    esac
    if ! physical="$(cd "$entry" 2>/dev/null && pwd -P)"; then
      printf 'ERROR: ONEX_REGISTRY_ROOTS=%s holds the entry "%s", which is not an existing directory. Every entry must be an absolute registry-root directory, colon-separated; fix or unset the variable.\n' "$declared" "$entry" >&2
      exit 1
    fi
    registry_roots+=("$physical")
    [[ "$rest" == *:* ]] || break
    rest="${rest#*:}"
  done
  return 0
}

# own_hooks_path_is_this_family
#
# True when the invoking repository's OWN config (not a global or system one)
# sets core.hooksPath to the directory this hook was invoked from. Only the
# installer writes that setting, and it writes it only into canonical clones, so
# it is the clone's own statement that it is one. A host-wide core.hooksPath
# reaching an unrelated repository does not satisfy it, and neither does a
# repository that never pointed at the family.
own_hooks_path_is_this_family() {
  local own invoked own_physical
  own="$(git config --local --type=path --get core.hooksPath 2>/dev/null)" || return 1
  [[ -n "$own" ]] || return 1
  case "$own" in
    /*) : ;;
    *) own="$top_level/$own" ;;
  esac
  own_physical="$(cd "$own" 2>/dev/null && pwd -P)" || return 1
  invoked="$(cd "$(dirname "$0")" 2>/dev/null && pwd -P)" || return 1
  [[ "$own_physical" == "$invoked" ]]
}

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
  # With no OMNI_HOME, the roots file's recorded omni_home comes before the
  # clone-position guess, so a refusal in a second-root clone names the
  # sanctioned worktree root rather than one beside that clone.
  read_registry_roots_file
  omni_home="${OMNI_HOME:-}"
  if [[ -z "$omni_home" ]]; then
    omni_home="$file_omni_home"
  fi
  if [[ -z "$omni_home" ]]; then
    omni_home="$(cd "$git_common_dir/../.." && pwd -P)"
  fi

  registry_roots_context

  local exempt=0 root
  if [[ "$top_level" == "$omni_home" ]]; then
    # The registry meta-repo itself commits directly to its docs branch.
    exempt=1
  elif is_under "$top_level" "${ONEX_WORKTREES_ROOT:-}"; then
    exempt=1
  elif is_under "$top_level" "$omni_home/omni_worktrees"; then
    exempt=1
  elif [[ "$is_linked_worktree" == "1" ]]; then
    exempt=1
  else
    for root in "${registry_roots[@]}"; do
      if is_under "$top_level" "$root/omni_worktrees"; then
        exempt=1
        break
      fi
    done
  fi

  local inside_a_root=0 is_a_root=0
  if [[ "$exempt" == "0" ]]; then
    # `is_under` is a strict-descendant test, so a root's own tree is not
    # guarded here; only the clones inside each root are.
    for root in "${registry_roots[@]}"; do
      if [[ "$top_level" == "$root" ]]; then
        is_a_root=1
      elif is_under "$top_level" "$root"; then
        inside_a_root=1
        break
      fi
    done
  fi

  # The position rule: FAIL CLOSED on a clone this hook cannot place. A main
  # work tree whose own config points at this family was put there by the
  # installer as a canonical clone. If it lies in no root the hook can resolve
  # -- the environment lacks the variable and the roots file is missing or
  # stale, or the clone was moved -- guarding it is the only safe reading.
  unresolved_root_note=""
  if [[ "$exempt" == "0" && "$is_a_root" == "0" && "$inside_a_root" == "0" ]] &&
    own_hooks_path_is_this_family; then
    inside_a_root=1
    # shellcheck disable=SC2034  # the caller's out-parameter; printed by each refusal
    unresolved_root_note="This clone sits outside every registry root this hook can resolve
(${registry_roots[*]}), yet its own core.hooksPath points at the
canonical-clone guard, so it is guarded as a canonical clone (fail closed).
Record its root: re-run install-canonical-clone-git-hooks.sh --apply with
ONEX_REGISTRY_ROOTS naming it, which writes $registry_roots_file."
  fi

  # shellcheck disable=SC2034  # the caller's out-parameter; read in every sourcing hook
  is_canonical_clone="$inside_a_root"
  return 0
}
