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
  omnibase_compat
)

# Allow caller to override which repos to pull
if [[ $# -gt 0 ]]; then
  REPOS=("$@")
fi

# === Pre-pull validation: detect bare repo corruption (OMN-7600) ===
# If core.bare=true, git pull updates refs but NOT the working tree, causing
# stale files. This is corruption in omni_home — repos must be non-bare clones.
BARE_REPOS=()
for repo in "${REPOS[@]}"; do
  dir="$OMNI_HOME/$repo"
  [[ -d "$dir" ]] || continue
  is_bare=$(git -C "$dir" rev-parse --is-bare-repository 2>/dev/null || echo "unknown")
  if [[ "$is_bare" == "true" ]]; then
    BARE_REPOS+=("$repo")
  fi
done

if [[ ${#BARE_REPOS[@]} -gt 0 ]]; then
  echo ""
  echo "ERROR: Bare repo corruption detected in omni_home!"
  echo ""
  echo "The following repos have core.bare=true, which means git pull"
  echo "updates refs but NOT the working tree — files go stale silently."
  echo ""
  for repo in "${BARE_REPOS[@]}"; do
    echo "  CORRUPT  $repo"
    echo "           Fix: git -C $OMNI_HOME/$repo config core.bare false"
    echo "           Then: git -C $OMNI_HOME/$repo reset --hard HEAD"
  done
  echo ""
  echo "Fix all corrupted repos above, then re-run pull-all.sh."
  exit 1
fi
# === End bare repo validation ===

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
