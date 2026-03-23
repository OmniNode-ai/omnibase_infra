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
    ticket=$(basename "$ticket_dir")
    # Check if any repo worktree in this ticket dir has uncommitted changes
    has_changes=false
    for wt in "$ticket_dir"/*/; do
      [ -d "$wt/.git" ] || [ -f "$wt/.git" ] || continue
      if [ -n "$(git -C "$wt" status --porcelain 2>/dev/null)" ]; then
        has_changes=true
        break
      fi
    done

    if [ "$has_changes" = true ]; then
      echo "  [skip] $ticket (has uncommitted changes)"
    elif [ "$DRY_RUN" = true ]; then
      echo "  [would remove] $ticket"
      worktrees_removed=$((worktrees_removed + 1))
    else
      # Properly remove worktrees via git (no --force: let failures be safety signals)
      all_removed=true
      for wt in "$ticket_dir"/*/; do
        [ -d "$wt/.git" ] || [ -f "$wt/.git" ] || continue
        repo_name=$(basename "$wt")
        if ! git -C "$OMNI_HOME/$repo_name" worktree remove "$wt" 2>/dev/null; then
          echo "  [warn] could not remove $wt (may have untracked files) — skipping"
          all_removed=false
        fi
      done
      if [ "$all_removed" = true ]; then
        rm -rf "$ticket_dir"
      fi
      echo "  [removed] $ticket"
      worktrees_removed=$((worktrees_removed + 1))
    fi
  done
  echo "  $worktrees_removed worktrees $([ "$DRY_RUN" = true ] && echo 'would be' || echo '') removed"
fi

echo ""
echo "Total branches: $TOTAL_DELETED | Mode: $([ "$DRY_RUN" = true ] && echo 'DRY RUN (use --execute to apply)' || echo 'EXECUTED')"
