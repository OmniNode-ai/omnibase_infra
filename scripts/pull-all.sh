#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# pull-all.sh — Pull all omni_home canonical repos to latest main
#
# Usage:
#   ./pull-all.sh           # pull all repos
#   ./pull-all.sh omniclaude omnibase_core   # pull specific repos

set -euo pipefail

OMNI_HOME="${OMNI_HOME:-/Volumes/PRO-G40/Code/omni_home}"

REPOS=(
  omniclaude
  omnibase_core
  omnibase_infra
  omnibase_spi
  omnidash
  omniintelligence
  omnimemory
  omninode_infra
  omniweb
  onex_change_control
)

# Allow caller to override which repos to pull
if [[ $# -gt 0 ]]; then
  REPOS=("$@")
fi

RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR"' EXIT

# Fetch a single repo — writes result to a temp file for aggregation.
_pull_one() {
  local repo="$1"
  local dir="$OMNI_HOME/$repo"
  local result_file="$RESULTS_DIR/$repo"

  if [[ ! -d "$dir" ]]; then
    echo "  MISSING  $repo"
    echo "MISSING" > "$result_file"
    return
  fi

  local is_bare
  is_bare=$(git -C "$dir" rev-parse --is-bare-repository 2>/dev/null)

  if [[ "$is_bare" == "true" ]]; then
    # Prune stale worktrees before fetch — prevents "main is checked out"
    # errors from dead worktrees that still hold a ref to main.
    git -C "$dir" worktree prune 2>/dev/null || true

    # Bare clone: fetch origin main directly into the local main ref
    local before after output
    before=$(git -C "$dir" rev-parse main 2>/dev/null)
    if output=$(git -C "$dir" fetch origin main:main 2>&1); then
      after=$(git -C "$dir" rev-parse main 2>/dev/null)
      if [[ "$before" == "$after" ]]; then
        echo "  OK       $repo (already up to date)"
      else
        local commits
        commits=$(git -C "$dir" log --oneline "${before}..${after}" 2>/dev/null | wc -l | tr -d ' ')
        echo "  UPDATED  $repo (+${commits} commit(s))"
      fi
      echo "OK" > "$result_file"
    elif echo "$output" | grep -q "checked out"; then
      echo "  WARN     $repo (main checked out in a worktree, using FETCH_HEAD)"
      if git -C "$dir" fetch origin main 2>/dev/null; then
        echo "OK" > "$result_file"
      else
        echo "  FAILED   $repo (fetch failed even via FETCH_HEAD)"
        echo "FAILED" > "$result_file"
      fi
    else
      echo "  FAILED   $repo"
      echo "           $output"
      echo "FAILED" > "$result_file"
    fi
    return
  fi

  local branch
  branch=$(git -C "$dir" branch --show-current 2>/dev/null)
  if [[ "$branch" != "main" ]]; then
    echo "  SKIPPED  $repo (on branch: $branch)"
    echo "SKIPPED" > "$result_file"
    return
  fi

  local output
  if output=$(git -C "$dir" pull --ff-only 2>&1); then
    if echo "$output" | grep -q "Already up to date"; then
      echo "  OK       $repo (already up to date)"
    else
      local commits
      commits=$(git -C "$dir" log --oneline ORIG_HEAD..HEAD 2>/dev/null | wc -l | tr -d ' ')
      echo "  UPDATED  $repo (+${commits} commit(s))"
    fi
    echo "OK" > "$result_file"
  else
    echo "  FAILED   $repo"
    echo "           $output"
    echo "FAILED" > "$result_file"
  fi
}

# Launch all fetches in parallel
for repo in "${REPOS[@]}"; do
  _pull_one "$repo" &
done

wait

# Aggregate results
OK=0
FAILED=()

for repo in "${REPOS[@]}"; do
  result_file="$RESULTS_DIR/$repo"
  if [[ -f "$result_file" ]]; then
    status=$(cat "$result_file")
    case "$status" in
      OK) (( OK++ )) || true ;;
      FAILED) FAILED+=("$repo") ;;
      MISSING) FAILED+=("$repo (missing)") ;;
      # SKIPPED — don't count
    esac
  fi
done

echo ""
echo "${OK} repo(s) up to date. ${#FAILED[@]} failed."
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
