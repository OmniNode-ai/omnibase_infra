#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# git-gc-auto.sh — Non-destructive, worktree-aware `git gc --auto` across the
# canonical clones under $OMNI_HOME, run ONLY when no git operation is active.
# (OMN-14760 / F-21)
#
# Why: canonical clones (notably onex_change_control, ~11.8k loose objects across
# ~73 linked worktrees on 2026-07-18) accumulate loose objects and re-print
# "too many unreachable loose objects; run 'git prune'" on every commit, adding
# sweep noise. `git gc --auto` compacts them safely.
#
# Safety model:
#   - Uses `git gc --auto` ONLY. Never `--aggressive`, never `--prune=now`; the
#     default 2-week grace on unreachable objects is preserved. `git gc` is
#     worktree-aware — it treats every linked worktree's HEAD and index as
#     reachable — so a raw `git prune` (which is NOT run here) is avoided.
#   - A repo is SKIPPED when any git operation is active in the main clone OR any
#     linked worktree: an `index.lock`, or an in-progress rebase / merge /
#     cherry-pick / revert / bisect. This is the "no active worktree operation is
#     running" precondition, and it also avoids racing a concurrent git process
#     — the class of hazard behind the OMN-14746/14744 GIT_DIR-inheritance data
#     loss. A merely-dirty working tree is NOT a skip reason: gc is safe on it,
#     and skipping on any transient dirty worktree would make this a no-op for
#     the very clone (OCC, 73 worktrees) it exists to maintain.
#
# Usage:
#   scripts/git-gc-auto.sh [--dry-run | --execute] [--root DIR]
#     --dry-run   (default) report what would run; mutate nothing.
#     --execute   run `git gc --auto` on eligible clones.
#     --root DIR  root holding the clones (default: $OMNI_HOME). One level deep.
#
# Intended to be scheduled off-peak (CronCreate), never mid-commit.
set -euo pipefail

DRY_RUN=true
ROOT=""

while [ "$#" -gt 0 ]; do
  case "${1}" in
    --execute) DRY_RUN=false ;;
    --dry-run) DRY_RUN=true ;;
    --root)
      shift
      ROOT="${1:-}"
      if [ -z "${ROOT}" ]; then
        echo "git-gc-auto: --root requires a directory" >&2
        exit 2
      fi
      ;;
    -h | --help)
      sed -n '30,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "git-gc-auto: unknown option: ${1}" >&2
      exit 2
      ;;
  esac
  shift
done

if [ -z "${ROOT}" ]; then
  # Fail fast on missing env rather than picking a wrong default (rule #8).
  ROOT="${OMNI_HOME:?OMNI_HOME must be set (or pass --root DIR)}"
fi

if [ ! -d "${ROOT}" ]; then
  echo "git-gc-auto: root is not a directory: ${ROOT}" >&2
  exit 2
fi

# In-progress-operation state files, relative to a git admin dir. Presence of any
# means a git operation is mid-flight and gc must not touch this object store.
_OP_MARKERS="index.lock rebase-merge rebase-apply MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD BISECT_LOG"

# active_op_reason GITDIR -> echoes the first marker found (active op), else empty.
active_op_reason() {
  local gitdir="${1}"
  local marker
  for marker in ${_OP_MARKERS}; do
    if [ -e "${gitdir}/${marker}" ]; then
      printf '%s' "${marker}"
      return 0
    fi
  done
  return 0
}

# repo_blocked_reason REPO_DIR -> echoes a human reason if any git op is active in
# the main clone or any linked worktree admin dir; empty if clear to gc.
repo_blocked_reason() {
  local repo_dir="${1}"
  local gitdir="${repo_dir}/.git"

  local reason
  reason="$(active_op_reason "${gitdir}")"
  if [ -n "${reason}" ]; then
    printf 'main clone: %s' "${reason}"
    return 0
  fi

  # Linked worktree admin dirs live under <main>/.git/worktrees/<name>/.
  local wt_admin
  if [ -d "${gitdir}/worktrees" ]; then
    for wt_admin in "${gitdir}/worktrees"/*/; do
      [ -d "${wt_admin}" ] || continue
      reason="$(active_op_reason "${wt_admin%/}")"
      if [ -n "${reason}" ]; then
        printf 'worktree %s: %s' "$(basename "${wt_admin%/}")" "${reason}"
        return 0
      fi
    done
  fi
  return 0
}

if [ "${DRY_RUN}" = true ]; then
  echo "=== git-gc-auto: DRY RUN (use --execute to apply) root=${ROOT} ==="
else
  echo "=== git-gc-auto: EXECUTE root=${ROOT} ==="
fi

gc_count=0
skip_count=0

for repo_dir in "${ROOT}"/*/; do
  repo_dir="${repo_dir%/}"
  # Only a canonical clone (real .git DIRECTORY) owns an object store to gc. A
  # `.git` FILE means this dir is itself a linked worktree — skip; its main clone
  # is handled when we reach it.
  [ -d "${repo_dir}/.git" ] || continue
  repo="$(basename "${repo_dir}")"

  reason="$(repo_blocked_reason "${repo_dir}")"
  if [ -n "${reason}" ]; then
    echo "  [skip] ${repo} (active op — ${reason})"
    skip_count=$((skip_count + 1))
    continue
  fi

  if [ "${DRY_RUN}" = true ]; then
    echo "  [would gc] ${repo} (git gc --auto)"
  else
    if git -C "${repo_dir}" gc --auto --quiet; then
      echo "  [gc] ${repo}"
    else
      echo "  [warn] ${repo} (git gc --auto returned non-zero; left intact)"
    fi
  fi
  gc_count=$((gc_count + 1))
done

echo "=== ${gc_count} clone(s) $([ "${DRY_RUN}" = true ] && echo 'would be' || echo '') gc'd, ${skip_count} skipped ==="
