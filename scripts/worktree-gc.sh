#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# worktree-gc.sh — Merge-triggered worktree GC driver (OMN-13008).
#
# WHY: stale worktrees for already-merged PRs accumulate under the worktrees root
# and (alongside docker images) filled /data on .201 on 2026-06-11. This driver
# garbage-collects them on a schedule, removing ONLY worktrees whose PR is merged
# (or whose remote branch is gone) AND that are clean AND fully pushed.
#
# It does NOT reimplement the safety logic — it drives the canonical
# omniclaude/scripts/prune-worktrees.sh, which already enforces:
#   - remove only if PR MERGED or remote branch gone
#   - SKIP if working tree dirty
#   - SKIP if unpushed commits (no-upstream defaults to SKIP, never DELETE)
#
# This is the single GC core used on BOTH machines:
#   - Mac: invoked from the merge-sweep CronCreate tick path (launchd does not fire
#     on this Mac; session crons are the working scheduler).
#   - .201: invoked from deploy/worktree-gc.timer (systemd user timer) against
#     /data/omninode/omni_worktrees.
#
# Usage:
#   ./scripts/worktree-gc.sh                          # DRY RUN (default)
#   ./scripts/worktree-gc.sh --execute                # actually prune merged worktrees
#   ./scripts/worktree-gc.sh --worktrees-root <path>  # override (default: $OMNI_HOME/omni_worktrees)
#   ./scripts/worktree-gc.sh --prune-script <path>    # override path to prune-worktrees.sh
#
# Exit codes: 0 success, 2 bad args, 3 prune script not found.
#
# Log: ~/.local/log/onex/worktree-gc.log

set -euo pipefail

EXECUTE=false
# Resolve OMNI_HOME fail-fast-ish: prefer env, else derive from this script's repo root.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"           # .../omnibase_infra
DEFAULT_OMNI_HOME="$(cd "$REPO_ROOT/.." && pwd)"    # .../omni_home (canonical registry sibling layout)
OMNI_HOME="${OMNI_HOME:-$DEFAULT_OMNI_HOME}"

WORKTREES_ROOT="${ONEX_WORKTREES_ROOT:-$OMNI_HOME/omni_worktrees}"
# prune-worktrees.sh lives in the omniclaude canonical clone next to omnibase_infra.
PRUNE_SCRIPT="${OMNI_HOME}/omniclaude/scripts/prune-worktrees.sh"
LOG_FILE="${HOME}/.local/log/onex/worktree-gc.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=true; shift ;;
    --worktrees-root) WORKTREES_ROOT="$2"; shift 2 ;;
    --prune-script) PRUNE_SCRIPT="$2"; shift 2 ;;
    --help|-h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [worktree-gc] $*" | tee -a "$LOG_FILE" >&2; }

if [[ ! -f "$PRUNE_SCRIPT" ]]; then
  log "ERROR: prune script not found at $PRUNE_SCRIPT (set --prune-script or OMNI_HOME)"
  exit 3
fi
if [[ ! -d "$WORKTREES_ROOT" ]]; then
  log "worktrees root absent: $WORKTREES_ROOT — nothing to GC"
  exit 0
fi

log "Starting ($( [[ "$EXECUTE" == true ]] && echo EXECUTE || echo DRY-RUN )), root=$WORKTREES_ROOT"

ARGS=(--worktrees-root "$WORKTREES_ROOT")
[[ "$EXECUTE" == true ]] && ARGS+=(--execute)

# Drive the canonical prune script. Its own safety checks are authoritative.
bash "$PRUNE_SCRIPT" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

log "Done."
