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

OK=0
FAILED=()

for repo in "${REPOS[@]}"; do
  dir="$OMNI_HOME/$repo"

  if [[ ! -d "$dir" ]]; then
    echo "  MISSING  $repo"
    FAILED+=("$repo (missing)")
    continue
  fi

  is_bare=$(git -C "$dir" rev-parse --is-bare-repository 2>/dev/null)

  if [[ "$is_bare" == "true" ]]; then
    # Bare clone: fetch origin main directly into the local main ref
    before=$(git -C "$dir" rev-parse main 2>/dev/null)
    if output=$(git -C "$dir" fetch origin main:main 2>&1); then
      after=$(git -C "$dir" rev-parse main 2>/dev/null)
      if [[ "$before" == "$after" ]]; then
        echo "  OK       $repo (already up to date)"
      else
        commits=$(git -C "$dir" log --oneline "${before}..${after}" 2>/dev/null | wc -l | tr -d ' ')
        echo "  UPDATED  $repo (+${commits} commit(s))"
      fi
      (( OK++ )) || true
    else
      echo "  FAILED   $repo"
      echo "           $output"
      FAILED+=("$repo")
    fi
    continue
  fi

  branch=$(git -C "$dir" branch --show-current 2>/dev/null)
  if [[ "$branch" != "main" ]]; then
    echo "  SKIPPED  $repo (on branch: $branch)"
    continue
  fi

  if output=$(git -C "$dir" pull --ff-only 2>&1); then
    if echo "$output" | grep -q "Already up to date"; then
      echo "  OK       $repo (already up to date)"
    else
      commits=$(git -C "$dir" log --oneline ORIG_HEAD..HEAD 2>/dev/null | wc -l | tr -d ' ')
      echo "  UPDATED  $repo (+${commits} commit(s))"
    fi
    (( OK++ )) || true
  else
    echo "  FAILED   $repo"
    echo "           $output"
    FAILED+=("$repo")
  fi
done

echo ""
echo "${OK} repo(s) up to date. ${#FAILED[@]} failed."
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
