#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Canonical-clone REF-TRANSACTION guard (OMN-16497 layer 1).
#
# Installed as the `reference-transaction` hook in the shared canonical-clone
# hooks directory, alongside the commit/push guard. git runs this hook for
# EVERY ref update in the repository -- including the plumbing verbs no commit
# or push hook ever sees -- so it is the only surface in this fleet that
# refuses a canonical-clone mutation made by a process that is not a Claude
# Code tool call.
#
# ## Why a git hook and not the Claude PreToolUse guard
#
# `canonical-clone-guard.py` is a Claude Code PreToolUse hook. It cannot see any
# process that is not a Claude Code tool call. Measured, 2026-08-22: a Claude
# worker's Edit into the canonical omnimarket clone was DENIED at 13:59:06Z and
# correctly moved to a worktree; 52 seconds later a Codex CLI session ran
# `apply_patch` against the same path in the same clone and succeeded, because
# Codex has no such hook. The resulting dirty tree is what later pushed a lane
# into a `git update-ref` plumbing bypass. The uncovered actor class is the root
# of that incident chain, not a footnote -- so the enforcement has to live where
# git is, not where one agent's tool loop is.
#
# ## What git actually hands this hook (measured, git 2.50.1, 2026-09-13)
#
# argv[1] is the stage: `prepared`, `committed` or `aborted`. stdin carries one
# `<old-oid> <new-oid> <ref>` line per ref in the transaction. Exiting non-zero
# at `prepared` aborts the whole transaction -- `fatal: ref updates aborted by
# hook` -- and nothing is written.
#
# Two measured facts shape the policy below, and both contradict the obvious
# design:
#
#  1. `<old-oid>` IS NOT RELIABLE. It is what the transaction ASSERTED, not what
#     the ref currently holds, and it arrives as forty zeros for every verb that
#     asserted nothing -- `git update-ref <ref> <new>`, `git branch -f`,
#     `git checkout -b`. An "is this a fast-forward from old to new" test built
#     on it would read every one of those as a ref CREATION and allow it. So
#     this hook resolves the CURRENT value itself, with `git rev-parse` at the
#     `prepared` stage, before the update lands.
#
#  2. A BRANCH SWITCH IS VISIBLE, as a `HEAD` line whose new value is the literal
#     `ref:refs/heads/<name>`. That is the half the Claude guard's deny list
#     reaches only for `checkout`/`switch` spelled as a Bash tool call, and it is
#     the half the 2026-09-13 friction report asked for: the canonical clone of
#     the knowledge-base repository was found sitting on a background-fleet
#     feature branch, so every lane resolving that clone read a stale main.
#
# ## Policy
#
# Enforced ONLY when the invoking work tree is a canonical clone. Worktrees
# linked to that clone share its config, and therefore this hooksPath, so every
# lane's ordinary commit reaches this hook -- it must let them through, and the
# shared `canonical_clone_context` is what decides that.
#
#   ALLOW  refs/remotes/*, refs/tags/*, refs/notes/*   the fetch/pull path
#   ALLOW  creating a refs/heads/* ref that does not exist yet
#          -- linking a new worktree with a fresh branch is the SANCTIONED
#          escape hatch (rule 9), and creating a branch does not move this
#          clone's checked-out state. `git checkout -b` creates a branch too,
#          and is refused one line later by its HEAD symref move, which linking
#          a worktree does not make.
#   ALLOW  fast-forwarding an existing refs/heads/* ref   the `pull --ff-only` path
#   DENY   any HEAD update: a symref move is a branch switch, an oid is a detach
#   DENY   rewinding or diverting an existing refs/heads/* ref (branch -f,
#          reset --hard backwards, update-ref to an unrelated commit, a non-ff
#          `fetch <src>:<dst>`)
#   DENY   deleting any refs/heads/* ref
#   DENY   refs/stash -- stashing is a mutation the Claude guard already denies
#
# Sanctioned bypass: `ONEX_CANONICAL_CONVERGE=1`, exported by
# `converge-canonical-clone.sh` for its own re-attach and reset. That script
# preserves the working tree as patches and appends a ledger row BEFORE it moves
# anything, which is the whole difference between a convergence and the drift
# this hook exists to stop. It is a named, evidence-producing path -- not a skip
# flag -- and it is the only door, because a hook with no sanctioned door gets
# routed around.
#
# Fails CLOSED inside a canonical clone. Unlike the Claude guard, which fails
# open so a parse bug cannot freeze a session, this hook has the entire
# transaction on stdin and no session to protect: an unreadable line is a
# mutation it cannot vouch for.

set -euo pipefail

stage="${1:-}"

_self="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"
while [[ -L "$_self" ]]; do
  _link="$(readlink "$_self")"
  case "$_link" in
    /*) _self="$_link" ;;
    *) _self="$(cd "$(dirname "$_self")" && pwd -P)/$_link" ;;
  esac
done
# shellcheck source=./canonical_clone_paths.sh
source "$(cd "$(dirname "$_self")" && pwd -P)/canonical_clone_paths.sh"

# Only `prepared` can abort a transaction; `committed` and `aborted` are
# after-the-fact notifications. Reading stdin on those stages buys nothing and
# costs a git invocation on every ref of every fetch.
[[ "$stage" == "prepared" ]] || exit 0

# Declared before the call so `set -u` cannot trip on one the function does not
# reach. canonical_clone_context populates all six; only two are read here, but
# a partial declaration would be a latent unbound-variable fault the day this
# hook grows a second decision.
top_level=""
# shellcheck disable=SC2034  # populated by canonical_clone_context, read by future policy
git_dir=""
# shellcheck disable=SC2034  # populated by canonical_clone_context, read by future policy
git_common_dir=""
# shellcheck disable=SC2034  # populated by canonical_clone_context, read by future policy
is_linked_worktree=0
omni_home=""
is_canonical_clone=0
canonical_clone_context || exit 0
[[ "$is_canonical_clone" == "1" ]] || exit 0

# The sanctioned convergence path. Checked after the canonical-clone test so the
# variable means nothing anywhere else.
if [[ "${ONEX_CANONICAL_CONVERGE:-}" == "1" ]]; then
  exit 0
fi

_zero_oid='0000000000000000000000000000000000000000'

deny() {
  local ref="$1" what="$2" hint="$3"
  cat >&2 <<EOF
ERROR: refused $what in canonical clone:
  repo: $top_level
  ref:  $ref

Canonical clones are pull/index mirrors that must stay on their tracking
branch. $hint

Work belongs in a worktree linked from this clone, under:
  $omni_home/omni_worktrees/<ticket>/<repo>

To repair a clone that has already drifted, use the sanctioned path, which
preserves the working tree as patches and appends a ledger row first:
  $omni_home/omniclaude/scripts/converge-canonical-clone.sh <repo> --execute
EOF
  exit 1
}

# worktree_is_initializing
#
# True when some registered worktree admin directory has no HEAD file yet.
#
# git creates `$common/worktrees/<name>/` and only THEN writes that worktree's
# HEAD, as the very transaction this hook is being asked about. So at the
# `prepared` stage of a worktree link exactly one admin directory exists with no
# HEAD file, and at the `prepared` stage of a real branch switch every admin
# directory already has one. Measured on git 2.50.1, 2026-09-13:
#
#   linking a worktree, prepared:  wt <name> HEAD=[No such file or directory]
#   switching a branch,  prepared:  wt <name> HEAD=[ref: refs/heads/feat]
#
# This is git's initialization order, not a timing heuristic, and it is the only
# signal that separates the two -- every other one was measured identical.
#
# HONEST LIMIT, stated rather than left for someone to find: a real branch
# switch that lands in the window while a CONCURRENT actor is linking a worktree
# on the same clone reads as an initialization and is permitted. The window is
# the few milliseconds between git creating the admin directory and writing its
# HEAD. It fails OPEN there, which is the availability-safe direction and the
# enforcement-unsafe one; closing it needs a signal git does not currently emit.
worktree_is_initializing() {
  local common wt_dir
  common="$(git rev-parse --git-common-dir 2>/dev/null)" || return 1
  common="$(absolutize "$common")"
  [[ -d "$common/worktrees" ]] || return 1
  for wt_dir in "$common"/worktrees/*/; do
    [[ -d "$wt_dir" ]] || continue
    [[ -f "$wt_dir/HEAD" ]] || return 0
  done
  return 1
}

while read -r old new ref; do
  # Fail closed on a line this hook cannot parse. `old` is deliberately unused
  # in the decision (see fact 1 in the header) but its absence means the line is
  # not the documented shape.
  if [[ -z "${old:-}" || -z "${new:-}" || -z "${ref:-}" ]]; then
    printf 'ERROR: unparseable reference-transaction line in canonical clone %s; refusing the transaction.\n' "$top_level" >&2
    exit 1
  fi

  case "$ref" in
    HEAD)
      if worktree_is_initializing; then
        # Linking a new worktree writes the NEW worktree's HEAD, and git hands
        # this hook a line byte-identical to a real branch switch -- same stage,
        # same shape, and an empty environment in both cases (measured: GIT_DIR,
        # GIT_WORK_TREE, GIT_REFLOG_ACTION and GIT_INDEX_FILE are all unset for
        # both). The transaction interface cannot tell them apart, so the
        # discriminator is git's own initialization ORDER on disk. See
        # worktree_is_initializing.
        continue
      fi
      case "$new" in
        ref:*)
          deny "$ref" "a branch switch" \
            "A canonical clone left on a feature branch silently serves stale content to every lane that resolves it."
          ;;
        *)
          deny "$ref" "detaching HEAD" \
            "A detached clone tracks no branch, so it can never follow its upstream again."
          ;;
      esac
      ;;
    refs/remotes/* | refs/tags/* | refs/notes/*)
      continue
      ;;
    refs/stash)
      deny "$ref" "a stash" \
        "Stashing hides uncommitted content in a tree that is supposed to hold none."
      ;;
    refs/heads/*)
      if [[ "$new" == "$_zero_oid" ]]; then
        deny "$ref" "deleting a branch" \
          "Branch deletion in a mirror destroys the only local record of what it pointed at."
      fi
      current=""
      if ! current="$(git rev-parse --verify --quiet "$ref" 2>/dev/null)"; then
        # The ref does not exist yet: this is a creation, which is the shape a
        # newly linked worktree's branch makes. That is the sanctioned hatch.
        continue
      fi
      if [[ "$current" == "$new" ]]; then
        continue
      fi
      if git merge-base --is-ancestor "$current" "$new" 2>/dev/null; then
        # Fast-forward: the `git pull --ff-only` path.
        continue
      fi
      deny "$ref" "a non-fast-forward branch move" \
        "${current:0:12} is not an ancestor of ${new:0:12}, so this rewinds or diverts the mirror instead of advancing it."
      ;;
    refs/*)
      deny "$ref" "a ref write" \
        "Only remote-tracking refs, tags and notes move freely in a mirror."
      ;;
    *)
      # Pseudo-refs that are not under refs/ and are not HEAD: AUTO_MERGE,
      # ORIG_HEAD, MERGE_HEAD, CHERRY_PICK_HEAD. git emits AUTO_MERGE
      # transactions on every checkout (measured); they carry no branch state
      # and refusing them would break reads.
      continue
      ;;
  esac
done

exit 0
