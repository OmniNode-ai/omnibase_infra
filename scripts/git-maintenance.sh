#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail

# git-maintenance.sh — Clean up stale branches and orphaned worktrees across omni_home repos
#
# Usage:
#   git-maintenance.sh [--dry-run] [--execute] [--prune-worktrees]
#
# Modes:
#   --dry-run           Show what would be deleted (default)
#   --execute           Actually delete branches and worktrees
#   --prune-worktrees   Also clean up worktrees for completed tickets
#
# Branch deletion strategy: deletes remote branches that are fully merged into origin/main.
# This is merge-based (not age-based) — only branches whose commits are reachable from main are deleted.
# ASSUMPTION: origin/main is the canonical merge sink. Repos with different branch-flow policy
# (e.g., staged integration branches, release trains) should add protection patterns or
# perform repo-specific review before destructive execution.

OMNI_HOME="${OMNI_HOME:-/Volumes/PRO-G40/Code/omni_home}"
WORKTREE_ROOT="${WORKTREE_ROOT:-/Volumes/PRO-G40/Code/omni_worktrees}"
DRY_RUN=true
PRUNE_WORKTREES=false

# Parse args
for arg in "$@"; do
  case "$arg" in
    --execute) DRY_RUN=false ;;
    --dry-run) DRY_RUN=true ;;
    --prune-worktrees) PRUNE_WORKTREES=true ;;
  esac
done

# --- OMN-18826: worktree removal reporting -----------------------------------
#
# `git worktree remove` WITHOUT `--force` is not atomic. Its order of operations
# (verified against git 2.50.1) is: refuse if the tree is dirty; delete the
# admin directory under <clone>/.git/worktrees/<id>; THEN walk the working tree
# unlinking files, continuing past every failure and only returning non-zero at
# the end. A removal that fails partway through the walk leaves a tree with most
# of its files gone, a stale `.git` pointer, and NO admin directory. A later
# removal over that debris fails with "contains modified or untracked files",
# which points at the wrong cause entirely.
#
# This helper therefore reports what git said and what the tree looks like
# afterwards. It never prints a guessed cause, and it never suppresses stderr
# (Operating Rule 16). It does not add `--force`: the removal policy is not this
# script's to set, and a failure here is a signal, not an obstacle.
#
# Prints its report to stdout. Returns 0 on a complete removal, 1 otherwise.
remove_one_worktree() {
  local wt="$1"
  local admin_dir="" clone_dir="" head_oid="" stderr_file="" rc=0 survivors=0

  # Record what a reconstruction would need BEFORE touching the tree, because a
  # failed removal destroys the index and the admin directory that hold it.
  head_oid=$(git -C "$wt" rev-parse HEAD 2>/dev/null || echo "unknown")

  # Resolve the owning clone from the worktree's own pointer file rather than
  # from its directory name: a worktree directory is frequently named for the
  # lane ("omniclaude-freshness"), not for the repo, and `git -C <missing dir>`
  # produces its own unrelated error.
  if [ -f "$wt/.git" ]; then
    admin_dir=$(sed -n 's/^gitdir: //p' "$wt/.git" | head -1)
    case "$admin_dir" in
      */.git/worktrees/*) clone_dir="${admin_dir%%/.git/worktrees/*}" ;;
    esac
  fi
  if [ -z "$clone_dir" ]; then
    clone_dir="$OMNI_HOME/$(basename "$wt")"
  fi

  stderr_file=$(mktemp)
  git -C "$clone_dir" worktree remove "$wt" 2>"$stderr_file" || rc=$?

  if [ "$rc" -eq 0 ]; then
    rm -f "$stderr_file"
    echo "  [removed] $wt"
    return 0
  fi

  echo "  [remove-failed] $wt (exit $rc, head $head_oid)"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    echo "    git: $line"
  done < "$stderr_file"
  rm -f "$stderr_file"

  if [ ! -e "$wt" ]; then
    echo "    classification: removed-despite-error: the path no longer exists"
    return 1
  fi

  survivors=$(find "$wt" -mindepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')

  if [ -n "$admin_dir" ] && [ ! -d "$admin_dir" ]; then
    echo "    classification: half-removed: admin dir deleted, $survivors files remain"
    echo "    this tree is debris, not peer work; reconstruct from head $head_oid"
  else
    echo "    classification: not-removed: tree intact, $survivors files remain"
  fi
  return 1
}
# -----------------------------------------------------------------------------

echo "=== Git Maintenance ==="
echo "Mode: $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo 'EXECUTE')"
echo ""

TOTAL_DELETED=0

# Phase 1: Delete remote branches for merged PRs
for repo_dir in "$OMNI_HOME"/*/; do
  repo=$(basename "$repo_dir")
  [ -d "$repo_dir/.git" ] || [ -f "$repo_dir/.git" ] || continue

  echo "--- $repo ---"

  # Refresh remote refs before evaluating merged state
  git -C "$repo_dir" fetch --prune origin 2>/dev/null || { echo "  [warn] fetch failed, skipping $repo"; continue; }

  stale=0

  # Get branches with merged PRs (closed + merged)
  while IFS= read -r branch; do
    [ -z "$branch" ] && continue
    branch="${branch#origin/}"

    # Skip protected branches and long-lived naming patterns
    case "$branch" in
      main|master|develop|HEAD*|gh-pages) continue ;;
      release/*|support/*|infra/*) continue ;;  # long-lived branch patterns
    esac

    if [ "$DRY_RUN" = true ]; then
      echo "  [would delete] origin/$branch"
    else
      git -C "$repo_dir" push origin --delete "$branch" 2>/dev/null && echo "  [deleted] origin/$branch"
    fi
    stale=$((stale + 1))
  done < <(git -C "$repo_dir" branch -r --merged origin/main 2>/dev/null | grep -v 'origin/main' | grep -v 'origin/HEAD' | sed 's/^ *//')

  echo "  $stale merged branches $([ "$DRY_RUN" = true ] && echo 'would be' || echo '') deleted"
  TOTAL_DELETED=$((TOTAL_DELETED + stale))
done

# Phase 2: Clean orphaned worktrees
if [ "$PRUNE_WORKTREES" = true ] && [ -d "$WORKTREE_ROOT" ]; then
  echo ""
  echo "=== Worktree Cleanup ==="
  worktrees_removed=0

  for ticket_dir in "$WORKTREE_ROOT"/*/; do
    # The glob leaves a trailing slash, which doubles in every path built below
    # and lands in the report rows. A reported path has to be one a reader can
    # paste back, so normalise it once here.
    ticket_dir="${ticket_dir%/}"
    ticket=$(basename "$ticket_dir")
    # Check if any repo worktree in this ticket dir has uncommitted changes
    has_changes=false
    debris=false
    for wt in "$ticket_dir"/*/; do
      [ -d "$wt/.git" ] || [ -f "$wt/.git" ] || continue
      # OMN-18826: a half-deleted worktree's signature is a `.git` pointer file
      # whose `gitdir:` target no longer exists, because a failed removal
      # deletes the admin directory before it walks the tree. Such a tree is
      # debris from an earlier failed removal, not a lane's in-flight work, and
      # `git status` cannot tell you so: with its admin directory gone, status
      # fails outright and prints nothing, so the tree reads as clean.
      if [ -f "$wt/.git" ]; then
        pointer=$(sed -n 's/^gitdir: //p' "$wt/.git" | head -1)
        if [ -n "$pointer" ] && [ ! -d "$pointer" ]; then
          echo "  [debris] ${wt%/}: .git points at a missing admin dir ($pointer) — half-removed tree, not peer work"
          debris=true
          continue
        fi
      fi
      if [ -n "$(git -C "$wt" status --porcelain 2>/dev/null)" ]; then
        has_changes=true
        break
      fi
    done

    if [ "$debris" = true ]; then
      # Reported, not removed: what to do with debris is a removal-policy
      # decision this script does not own.
      echo "  [skip] $ticket (half-removed worktree present)"
    elif [ "$has_changes" = true ]; then
      echo "  [skip] $ticket (has uncommitted changes)"
    elif [ "$DRY_RUN" = true ]; then
      echo "  [would remove] $ticket"
      worktrees_removed=$((worktrees_removed + 1))
    else
      # Properly remove worktrees via git (no --force: let failures be safety signals)
      all_removed=true
      for wt in "$ticket_dir"/*/; do
        [ -d "$wt/.git" ] || [ -f "$wt/.git" ] || continue
        # Strip the trailing slash: git reports a removal target back verbatim.
        if ! remove_one_worktree "${wt%/}"; then
          all_removed=false
        fi
      done
      if [ "$all_removed" = true ]; then
        rm -rf "$ticket_dir"
        echo "  [removed] $ticket"
        worktrees_removed=$((worktrees_removed + 1))
      else
        # Saying "removed" over a failed removal is how the misreport reached the
        # report row in the first place. The per-worktree lines above carry the
        # reason; this line only records that the ticket directory still stands.
        echo "  [incomplete] $ticket (see the per-worktree lines above)"
      fi
    fi
  done
  echo "  $worktrees_removed worktrees $([ "$DRY_RUN" = true ] && echo 'would be' || echo '') removed"
fi

echo ""
echo "Total branches: $TOTAL_DELETED | Mode: $([ "$DRY_RUN" = true ] && echo 'DRY RUN (use --execute to apply)' || echo 'EXECUTED')"
