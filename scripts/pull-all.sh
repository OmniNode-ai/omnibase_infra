#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# pull-all.sh — Pull all omni_home canonical repos to latest dev and main
#
# Usage:
#   ./pull-all.sh           # pull all repos
#   ./pull-all.sh omniclaude omnibase_core   # pull specific repos

set -euo pipefail

OMNI_HOME="${OMNI_HOME:-/Volumes/PRO-G40/Code/omni_home}"

REPOS=(
  omniclaude
  omnibase_compat
  omnibase_core
  omnibase_infra
  omnibase_spi
  omnidash
  omnidash-v2
  omnigemini
  omniintelligence
  omnimarket
  omnimemory
  omninode_infra
  omniweb
  onex_change_control
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

# Switch to a branch, creating it from origin/<branch> when needed, then
# fast-forward it to the fetched remote branch.
_checkout_and_ff() {
  local dir="$1"
  local branch="$2"

  if git -C "$dir" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$dir" switch "$branch"
  else
    git -C "$dir" switch --track -c "$branch" "origin/$branch"
  fi

  git -C "$dir" merge --ff-only "origin/$branch"
}

_branch_summary() {
  local dir="$1"
  local branch="$2"
  local before="$3"
  local after

  after=$(git -C "$dir" rev-parse --verify --quiet "refs/heads/$branch" || true)
  if [[ -z "$before" ]]; then
    echo "$branch created"
  elif [[ "$before" == "$after" ]]; then
    echo "$branch already up to date"
  else
    local commits
    commits=$(git -C "$dir" rev-list --count "$before..$after" 2>/dev/null | tr -d ' ')
    echo "$branch +${commits} commit(s)"
  fi
}

_leave_on_dev() {
  local dir="$1"

  if git -C "$dir" show-ref --verify --quiet "refs/heads/dev"; then
    git -C "$dir" switch dev >/dev/null 2>&1
  elif git -C "$dir" show-ref --verify --quiet "refs/remotes/origin/dev"; then
    git -C "$dir" switch --track -c dev origin/dev >/dev/null 2>&1
  else
    return 1
  fi
}

# Pull a single repo — writes result to a temp file for aggregation.
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
  if [[ "$branch" != "main" && "$branch" != "dev" ]]; then
    echo "  SKIPPED  $repo (on branch: $branch)"
    echo "SKIPPED" > "$result_file"
    return
  fi

  local dirty
  dirty=$(git -C "$dir" status --porcelain)
  if [[ -n "$dirty" ]]; then
    echo "  FAILED   $repo (dirty worktree; refusing to switch branches)"
    echo "           Commit, stash, or remove local changes before re-running."
    echo "FAILED" > "$result_file"
    return
  fi

  local before_main before_dev output
  before_main=$(git -C "$dir" rev-parse --verify --quiet refs/heads/main || true)
  before_dev=$(git -C "$dir" rev-parse --verify --quiet refs/heads/dev || true)

  if ! output=$(git -C "$dir" fetch --prune origin main dev 2>&1); then
    echo "  FAILED   $repo (fetch main/dev)"
    echo "           $output"
    echo "FAILED" > "$result_file"
    return
  fi

  if ! output=$(_checkout_and_ff "$dir" main 2>&1); then
    echo "  FAILED   $repo (fast-forward main)"
    echo "           $output"
    if ! _leave_on_dev "$dir"; then
      echo "           WARN: could not return $repo to dev after failure."
    fi
    echo "FAILED" > "$result_file"
    return
  fi

  if output=$(_checkout_and_ff "$dir" dev 2>&1); then
    local main_summary dev_summary
    main_summary=$(_branch_summary "$dir" main "$before_main")
    dev_summary=$(_branch_summary "$dir" dev "$before_dev")
    echo "  OK       $repo ($main_summary; $dev_summary; left on dev)"
    echo "OK" > "$result_file"
  else
    echo "  FAILED   $repo (fast-forward dev)"
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
WARNED=()

for repo in "${REPOS[@]}"; do
  result_file="$RESULTS_DIR/$repo"
  if [[ -f "$result_file" ]]; then
    status=$(cat "$result_file")
    case "$status" in
      OK) (( OK++ )) || true ;;
      FAILED) FAILED+=("$repo") ;;
      MISSING) WARNED+=("$repo (not cloned — skipped)") ;;
      # SKIPPED — don't count
    esac
  fi
done

# === Plugin cache refresh (Layer 2, OMN-7369) ===
# When omniclaude was updated, refresh the Claude Code plugin cache.
#
# The cache lives at a versioned path:
#   ~/.claude/plugins/cache/omninode-tools/onex/<version>/
# Locate it by searching for the .deployed-commit marker file (more specific
# than matching a generic 'skills' directory — which also appears under
# other plugins like claude-plugins-official).
_omniclaude_dir="$OMNI_HOME/omniclaude"
_plugin_cache="${CLAUDE_PLUGIN_ROOT:-}"
if [[ -z "${_plugin_cache}" ]]; then
  # Search for the .deployed-commit marker inside the onex plugin cache tree.
  # maxdepth 5 covers: cache/omninode-tools/onex/<version>/.deployed-commit.
  # The `|| true` suffix stops `set -eo pipefail` from treating a missing
  # cache directory (find exits non-zero when the root does not exist) as
  # a fatal script error — a missing cache must be a clean no-op.
  _marker=$(find "${HOME}/.claude/plugins/cache" -maxdepth 5 -path "*/omninode-tools/onex/*" -name ".deployed-commit" -type f 2>/dev/null | head -1 || true)
  if [[ -n "${_marker}" ]]; then
    _plugin_cache=$(dirname "${_marker}")
  else
    # Fallback: marker absent (first deploy). Search for a versioned onex dir.
    _plugin_cache=$(find "${HOME}/.claude/plugins/cache" -maxdepth 4 -path "*/omninode-tools/onex/*" -type d 2>/dev/null | head -1 || true)
  fi
fi

# Compute a content hash of all plugin files under a directory.
# Excludes __pycache__, .pyc, and the marker files themselves so the hash
# is stable regardless of which directory it is computed against.
#
# The hash is computed against RELATIVE paths so a repo-side and cache-side
# computation of the same plugin tree yield the same hash (shasum emits
# `hash  path` — absolute paths would otherwise break comparability).
_plugin_content_hash() {
  local root="$1"
  ( cd "${root}" && find . -type f \
      ! -name "*.pyc" \
      ! -path "*/__pycache__/*" \
      ! -name ".deployed-commit" \
      ! -name ".content-hash" \
      -exec shasum {} \; 2>/dev/null | sort | shasum | cut -d' ' -f1 )
}

if [[ -n "${_plugin_cache}" && -d "${_omniclaude_dir}" && -d "${_plugin_cache}" ]]; then
  _current=$(git -C "${_omniclaude_dir}" rev-parse HEAD 2>/dev/null)
  _deployed=""
  [[ -f "${_plugin_cache}/.deployed-commit" ]] && _deployed=$(cat "${_plugin_cache}/.deployed-commit" 2>/dev/null)

  # Compare against repo content hash as a second signal beyond commit SHA.
  _repo_hash=""
  if [[ -d "${_omniclaude_dir}/plugins/onex" ]]; then
    _repo_hash=$(_plugin_content_hash "${_omniclaude_dir}/plugins/onex")
  fi
  _cache_hash=""
  [[ -f "${_plugin_cache}/.content-hash" ]] && _cache_hash=$(cat "${_plugin_cache}/.content-hash" 2>/dev/null)

  if [[ -n "${_current}" ]] && { [[ "${_current}" != "${_deployed}" ]] || [[ "${_repo_hash}" != "${_cache_hash}" ]]; }; then
    echo ""
    echo "Refreshing Claude Code plugin cache (${_deployed:-none} → ${_current:0:8})..."
    _tmpdir=$(mktemp -d)
    # Refresh the entire plugins/onex/ tree (hooks, skills, lib, agents,
    # runtime, scripts, docs, prompts, models, _bin, tests). Copying only
    # skills/ leaves stale code in sibling directories that schema changes
    # silently drop — the exact failure this fix prevents.
    if git -C "${_omniclaude_dir}" archive HEAD plugins/onex/ 2>/dev/null | tar -x -C "${_tmpdir}" 2>/dev/null; then
      if [[ -d "${_tmpdir}/plugins/onex" ]]; then
        # rsync with --delete would remove files the cache added (e.g.
        # __pycache__), so use cp -R of the contents to update in place.
        cp -R "${_tmpdir}/plugins/onex/." "${_plugin_cache}/"
        echo "${_current}" > "${_plugin_cache}/.deployed-commit"
        # Recompute hash against the cache after refresh and persist.
        _new_hash=$(_plugin_content_hash "${_plugin_cache}")
        echo "${_new_hash}" > "${_plugin_cache}/.content-hash"
        echo "Plugin cache refreshed (content hash ${_new_hash:0:8})."
      else
        echo "WARN: Plugin cache refresh failed (archive missing plugins/onex/)."
      fi
    else
      echo "WARN: Plugin cache refresh failed (git archive error)."
    fi
    rm -rf "${_tmpdir}"
  fi
fi
# === End plugin cache refresh ===

# === Local pre-commit hook installation (OMN-14099) ===
# Root cause of the "defects caught at the most expensive layer (CI/review/merge)
# instead of the cheapest" leak: the pre-commit git hook was never installed in
# the canonical clones, so every repo's .pre-commit-config.yaml was pure
# decoration -- commits silently skipped ALL local enforcement (hardcoded
# IPs/topics, banned constructs, URL/model-literal authority, skip tokens, ...)
# and CI became the first catch point. No `--no-verify` needed; the hook simply
# never ran. Installing it here -- in the sync entrypoint every session already
# runs before ticket work -- activates the already-written, correctly-scoped
# hooks at commit time. This is NOT a new pattern check; it closes a bypass leak.
#
# Design guarantees:
#   * Idempotent -- skips any repo whose hook is already pre-commit-managed, so
#     steady-state runs are a fast no-op.
#   * Scoped -- only touches repos that actually ship a .pre-commit-config.yaml.
#   * Fail-soft -- a hook-install problem NEVER fails the pull (hook install is a
#     convenience layer, not the sync's core job) and is left out of FAILED.
#   * Offline-safe -- the load-bearing step is writing the hook script (no
#     network); environment pre-build is best-effort so an offline machine still
#     gets commit-time enforcement (envs then install lazily at first commit).
if ! command -v pre-commit >/dev/null 2>&1; then
  echo ""
  echo "WARN: 'pre-commit' not found on PATH -- local git hooks were NOT installed."
  echo "      Install it (e.g. 'brew install pre-commit') so pattern/static"
  echo "      defects fail at commit time instead of first failing in CI."
else
  for repo in "${REPOS[@]}"; do
    _pc_dir="$OMNI_HOME/$repo"
    [[ -d "$_pc_dir" ]] || continue
    [[ -f "$_pc_dir/.pre-commit-config.yaml" ]] || continue
    (
      cd "$_pc_dir" || exit 0
      # `git rev-parse --git-path hooks` resolves to the SHARED hooks dir (the
      # common git dir), so installing in the canonical clone covers all of its
      # linked worktrees too -- worktrees do not get their own pre-commit hook.
      _hook="$(git rev-parse --git-path hooks 2>/dev/null)/pre-commit"
      if [[ -f "$_hook" ]] && grep -q "File generated by pre-commit" "$_hook" 2>/dev/null; then
        exit 0  # already pre-commit-managed -- idempotent no-op
      fi
      # Load-bearing: write the hook script (fast, no network). This alone
      # closes the leak; hook environments install lazily at first commit.
      if pre-commit install >/dev/null 2>&1; then
        # Best-effort env pre-build so the first real commit is not slow. Runs
        # at most once per repo (guarded above); a failure here is non-fatal
        # because the hook script is already written and will still fire.
        pre-commit install-hooks >/dev/null 2>&1 || true
        echo "  HOOK     $repo (pre-commit git hook installed)"
      else
        echo "  WARN     $repo (pre-commit install failed -- commit-time enforcement inactive)"
      fi
    )
  done
fi
# === End pre-commit hook installation ===

echo ""
if [[ ${#WARNED[@]} -gt 0 ]]; then
  echo "WARN: ${#WARNED[@]} repo(s) not found locally and were skipped:"
  for w in "${WARNED[@]}"; do
    echo "  WARN     $w"
  done
fi
echo "${OK} repo(s) up to date. ${#FAILED[@]} failed. ${#WARNED[@]} absent (skipped)."
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
