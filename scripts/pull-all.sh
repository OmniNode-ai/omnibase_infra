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
    # Prune stale worktrees before fetch — prevents "main is checked out"
    # errors from dead worktrees that still hold a ref to main.
    git -C "$dir" worktree prune 2>/dev/null || true

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
    elif echo "$output" | grep -q "checked out"; then
      # Worktree has main checked out — force-fetch origin to FETCH_HEAD and update
      echo "  WARN     $repo (main checked out in a worktree, using FETCH_HEAD)"
      if git -C "$dir" fetch origin main 2>/dev/null; then
        (( OK++ )) || true
      else
        echo "  FAILED   $repo (fetch failed even via FETCH_HEAD)"
        FAILED+=("$repo")
      fi
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

# === Plugin cache refresh (Layer 2) ===
# When omniclaude was updated, refresh the Claude Code plugin cache.
_omniclaude_dir="$OMNI_HOME/omniclaude"
_plugin_cache="${CLAUDE_PLUGIN_ROOT:-}"
if [[ -z "${_plugin_cache}" ]]; then
  # Try default plugin cache path
  _plugin_cache=$(find "${HOME}/.claude/plugins/cache" -maxdepth 3 -name "skills" -type d 2>/dev/null | head -1)
  [[ -n "${_plugin_cache}" ]] && _plugin_cache=$(dirname "${_plugin_cache}")
fi

if [[ -n "${_plugin_cache}" && -d "${_omniclaude_dir}" && -d "${_plugin_cache}/skills" ]]; then
  _current=$(git -C "${_omniclaude_dir}" rev-parse HEAD 2>/dev/null)
  _deployed=""
  [[ -f "${_plugin_cache}/.deployed-commit" ]] && _deployed=$(cat "${_plugin_cache}/.deployed-commit" 2>/dev/null)

  if [[ "${_current}" != "${_deployed}" && -n "${_current}" ]]; then
    echo ""
    echo "Refreshing Claude Code plugin cache (${_deployed:-none} → ${_current:0:8})..."
    _tmpdir=$(mktemp -d)
    if git -C "${_omniclaude_dir}" archive HEAD plugins/onex/skills/ 2>/dev/null | tar -x -C "${_tmpdir}" 2>/dev/null; then
      cp -r "${_tmpdir}/plugins/onex/skills/"* "${_plugin_cache}/skills/" 2>/dev/null
      echo "${_current}" > "${_plugin_cache}/.deployed-commit"
      echo "Plugin cache refreshed."
    else
      echo "WARN: Plugin cache refresh failed (git archive error)."
    fi
    rm -rf "${_tmpdir}"
  fi
fi
# === End plugin cache refresh ===

echo ""
echo "${OK} repo(s) up to date. ${#FAILED[@]} failed."
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
