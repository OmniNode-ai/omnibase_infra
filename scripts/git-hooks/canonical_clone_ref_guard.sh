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
#
# ## Undoing the half-apply (OMN-18358)
#
# Refusing at `prepared` is NOT a refusal on its own, because git updates the
# working tree and the index BEFORE it opens the transaction carrying the HEAD
# symref move. Aborting there refuses the LAST step of a checkout and keeps the
# rest: the clone sits on the TARGET tree with HEAD left on the old branch,
# every changed path staged, and no reflog entry, so nothing records that it
# happened. Measured on the canonical omniclaude clone 2026-09-14T07:0xZ --
# HEAD on `dev`, 420 staged paths, 0 worktree-modified, 0 untracked, and
# `git diff --cached --name-only origin/main` returning ZERO paths, meaning the
# index was byte-identical to the target while HEAD said otherwise. omnidash
# (149 staged) and omninode_infra (858 staged) carried the same signature.
# `pull-all.sh` switches `main`/`dev` inside the clone as part of its ordinary
# sanctioned sync, so from the guard's merge onward every run of it on this
# host both failed AND corrupted the clones it touched.
#
# git has no hook that runs before it touches the tree, so the refusal cannot
# move earlier. What it can do is put the tree back. At `aborted`:
#
#   1. A restore runs ONLY for an abort this hook itself caused. The `prepared`
#      deny writes a marker naming the exact `<new> <ref>` pair it refused, and
#      `aborted` restores only when the line it is handed matches. This is not
#      defensive coding: `git reset --hard` emits two `AUTO_MERGE` aborts of its
#      own (measured, git 2.50.1), and restoring on every abort would make this
#      hook reach into transactions it refused nothing in.
#   2. It restores ONLY when the half-applied state is fully explained by the
#      refused target -- the index matches the target tree exactly AND the
#      worktree matches the index. Canonical clones carry no uncommitted work by
#      rule, so that is the shape every real half-apply takes. When the premise
#      is false, the tree is LEFT ALONE and the record says so: a guard that
#      destroys a lane's in-flight work to tidy up is worse than the corruption
#      it repairs.
#   3. Every refusal and every outcome is appended to a durable record at
#      `<git-common-dir>/onex-canonical-clone-refusals.log`. git writes no
#      reflog entry for an aborted transaction, so without this the event leaves
#      no artifact at all -- which is exactly how the half-apply stayed
#      invisible for a day.
#
# The alternative considered and REJECTED: demote this hook to record-and-
# restore (permit the transaction) and rely on the Claude PreToolUse guard to
# prevent the switch. That guard cannot see a process that is not a Claude Code
# tool call -- the measured 2026-08-22 Codex `apply_patch` is the whole reason
# layer 1 exists -- so the demotion would reopen the hole for every non-Claude
# actor while still needing the restore code below. It is strictly more work for
# strictly less enforcement.

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

# Only `prepared` can abort a transaction. `committed` is an after-the-fact
# notification and reading stdin there buys nothing while costing a git
# invocation on every ref of every fetch. `aborted` is where this hook undoes
# the half-apply it caused -- see the OMN-18358 block below.
[[ "$stage" == "prepared" || "$stage" == "aborted" ]] || exit 0

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

# The durable refusal record and the same-transaction marker (OMN-18358).
#
# Both live in the git common dir rather than in `.onex_state/`: they describe
# THIS repository, they must survive a working-tree restore that is about to
# rewrite every tracked path, and a hook that writes into the work tree to
# report on the work tree can corrupt the thing it is reporting on.
_refusal_log="$git_common_dir/onex-canonical-clone-refusals.log"
_refusal_marker="$git_dir/onex-canonical-clone-refusal-pending"

# A marker older than this is stale -- its `aborted` stage never arrived (the
# process died, or git aborted a DIFFERENT transaction and never came back to
# this one). It is swept rather than honoured, so a marker left behind by one
# command can never trigger a restore during an unrelated one later.
_marker_max_age_seconds=120

_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# The invoking actor, best-effort. The whole point of layer 1 is the actors the
# Claude guard cannot see, so a record that cannot name one is worth less than
# one that names it approximately. Never fails the hook.
_actor_line() {
  local who cmd
  who="${USER:-${LOGNAME:-unknown}}"
  cmd="$(ps -o args= -p "$PPID" 2>/dev/null | head -1 | tr '|' '/' | tr -d '\n')" || cmd=""
  printf 'actor=%s ppid=%s cmd=%s' "$who" "$PPID" "${cmd:-unknown}"
}

record() {
  # record <verdict> <ref> <detail>
  local verdict="$1" ref="$2" detail="$3"
  printf '%s | %s | repo=%s | ref=%s | %s | %s\n' \
    "$(_now)" "$verdict" "$top_level" "$ref" "$detail" "$(_actor_line)" \
    >>"$_refusal_log" 2>/dev/null || true
}

deny() {
  local ref="$1" what="$2" hint="$3"
  # Resolve what the refused transaction was aiming at, while it is still
  # resolvable. A symref move names a branch; everything else names an oid
  # directly. `aborted` needs this to tell a half-applied tree apart from a tree
  # carrying content nobody asked for.
  local target=""
  case "$_deny_new" in
    ref:*) target="$(git rev-parse --verify --quiet "${_deny_new#ref:}" 2>/dev/null || true)" ;;
    "$_zero_oid" | "") target="" ;;
    *) target="$_deny_new" ;;
  esac
  record "REFUSED" "$ref" "what=$what new=$_deny_new target=${target:-none}"
  printf '%s\n%s\n%s\n' "$_deny_new" "$ref" "${target:-}" >"$_refusal_marker" 2>/dev/null || true

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

# restore_after_refusal -- the `aborted` half of the OMN-18358 fix.
#
# Runs ONLY when the marker this hook wrote at `prepared` names the same
# `<new> <ref>` pair git is now reporting as aborted. Every other abort, and
# every abort in a repository this hook refused nothing in, returns untouched.
restore_after_refusal() {
  local marker_new marker_ref marker_target matched=0 old new ref marker_age now_epoch

  [[ -f "$_refusal_marker" ]] || return 0

  # Sweep a marker whose `aborted` never arrived before reading it, so it can
  # never be honoured during an unrelated command minutes later.
  now_epoch="$(date -u +%s)"
  marker_age="$(( now_epoch - $(stat -f %m "$_refusal_marker" 2>/dev/null || stat -c %Y "$_refusal_marker" 2>/dev/null || echo "$now_epoch") ))"
  if [[ "$marker_age" -gt "$_marker_max_age_seconds" ]]; then
    rm -f "$_refusal_marker"
    return 0
  fi

  { read -r marker_new; read -r marker_ref; read -r marker_target; } <"$_refusal_marker" 2>/dev/null || return 0
  while read -r old new ref; do
    [[ -n "${ref:-}" ]] || continue
    if [[ "$new" == "$marker_new" && "$ref" == "$marker_ref" ]]; then
      matched=1
    fi
  done
  [[ "$matched" == "1" ]] || return 0
  rm -f "$_refusal_marker"

  # Nothing was half-applied: the refusal was already clean. `git checkout -b
  # <name>` with no start point takes this path, because its target tree IS
  # HEAD's tree.
  if git diff-index --quiet --cached HEAD -- 2>/dev/null; then
    record "CLEAN" "$marker_ref" "detail=index already matches HEAD; nothing to restore"
    return 0
  fi

  # An unstaged difference means content that is neither HEAD's nor the
  # target's -- a local edit git carried across the checkout. Leave it.
  if ! git diff-files --quiet 2>/dev/null; then
    record "NOT_RESTORED" "$marker_ref" \
      "reason=worktree differs from the index, so it carries content this hook cannot account for; repair with converge-canonical-clone.sh"
    return 0
  fi

  # The index must be EXACTLY the refused target's tree. Anything else -- a
  # staged local edit carried across, an unresolvable target -- is content this
  # hook did not cause and must not overwrite.
  if [[ -z "$marker_target" ]] || ! git diff-index --quiet --cached "$marker_target" -- 2>/dev/null; then
    record "NOT_RESTORED" "$marker_ref" \
      "reason=index does not match the refused target ${marker_target:-none}, so it carries content this hook cannot account for; repair with converge-canonical-clone.sh"
    return 0
  fi

  if git read-tree -u --reset HEAD 2>/dev/null; then
    record "RESTORED" "$marker_ref" "detail=worktree and index reset to HEAD after the half-applied ${marker_target:0:12}"
  else
    record "NOT_RESTORED" "$marker_ref" \
      "reason=git read-tree -u --reset HEAD failed; repair with converge-canonical-clone.sh"
  fi
  return 0
}

if [[ "$stage" == "aborted" ]]; then
  restore_after_refusal
  exit 0
fi

# `_deny_new` carries the current line's new value into deny(), which needs it
# to name the refused target in the record and the marker.
_deny_new=""

while read -r old new ref; do
  # Fail closed on a line this hook cannot parse. `old` is deliberately unused
  # in the decision (see fact 1 in the header) but its absence means the line is
  # not the documented shape.
  if [[ -z "${old:-}" || -z "${new:-}" || -z "${ref:-}" ]]; then
    printf 'ERROR: unparseable reference-transaction line in canonical clone %s; refusing the transaction.\n' "$top_level" >&2
    exit 1
  fi
  _deny_new="$new"

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
