#!/bin/bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# verify-plugin-cache.sh — Content-hash-based plugin cache verification [OMN-7372]
#
# Compares the deployed plugin cache against the canonical repo source.
# Reports FRESH or STALE with a list of changed files.
#
# Exit codes:
#   0 = FRESH (cache matches repo)
#   1 = STALE (cache diverges from repo)
#   2 = ERROR (missing paths, bad args, etc.)
#
# Usage:
#   verify-plugin-cache.sh [OPTIONS]
#
# Options:
#   --repo-dir DIR       Path to omniclaude repo root (default: $OMNI_HOME/omniclaude)
#   --cache-dir DIR      Path to deployed plugin cache (auto-detected if omitted)
#   --json               Output results as JSON
#   --quiet              Suppress diff details, print only FRESH/STALE
#   --fix                If stale, refresh the cache from repo (rsync)
#   -h, --help           Show this help

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

REPO_DIR="${OMNI_HOME:-}/omniclaude"
CACHE_DIR=""
OUTPUT_JSON=0
QUIET=0
FIX=0

# ── Argument parsing ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)  REPO_DIR="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --json)      OUTPUT_JSON=1; shift ;;
    --quiet)     QUIET=1; shift ;;
    --fix)       FIX=1; shift ;;
    -h|--help)
      sed -n '3,/^$/{ s/^# //; s/^#$//; p }' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# ── Resolve paths ─────────────────────────────────────────────────────────────

PLUGIN_SUBDIR="plugins/onex"
REPO_PLUGIN_DIR="${REPO_DIR}/${PLUGIN_SUBDIR}"

if [[ ! -d "$REPO_PLUGIN_DIR" ]]; then
  echo "ERROR: Repo plugin directory not found: $REPO_PLUGIN_DIR" >&2
  exit 2
fi

# Auto-detect cache directory if not specified
if [[ -z "$CACHE_DIR" ]]; then
  # Strategy: look for .deployed-commit under the Claude plugin cache
  _found=$(find "${HOME}/.claude/plugins/cache" -maxdepth 5 -name ".deployed-commit" -type f 2>/dev/null | head -1)
  if [[ -n "$_found" ]]; then
    CACHE_DIR=$(dirname "$_found")
  else
    # Fallback: look for skills/ directory
    _found=$(find "${HOME}/.claude/plugins/cache" -maxdepth 5 -name "skills" -type d 2>/dev/null | head -1)
    if [[ -n "$_found" ]]; then
      CACHE_DIR=$(dirname "$_found")
    fi
  fi
  # Also try CLAUDE_PLUGIN_ROOT if set
  if [[ -z "$CACHE_DIR" && -n "${CLAUDE_PLUGIN_ROOT:-}" && -d "${CLAUDE_PLUGIN_ROOT}" ]]; then
    CACHE_DIR="${CLAUDE_PLUGIN_ROOT}"
  fi
fi

if [[ -z "$CACHE_DIR" || ! -d "$CACHE_DIR" ]]; then
  echo "ERROR: Plugin cache directory not found. Use --cache-dir to specify." >&2
  exit 2
fi

# ── Content hash computation ──────────────────────────────────────────────────

# Common find exclusions for plugin files.
# Skips .venv (can contain 44k+ files from torch/scipy/etc.), __pycache__,
# deploy metadata, and OS artifacts.
_FIND_EXCLUDES=(
  ! -path "*/.venv/*"
  ! -path "*/node_modules/*"
  ! -path "*/logs/*"
  ! -path "*/tmp/*"
  ! -name "*.pyc"
  ! -path "*/__pycache__/*"
  ! -name ".deployed-commit"
  ! -name ".content-hash"
  ! -name ".DS_Store"
  ! -name "*.log"
)

# Compute a deterministic content hash over all plugin source files.
# Concatenates sorted relative-path + file-content pairs into a single stream,
# then hashes it. This is metadata-independent (ignores mtime/permissions),
# so an rsync'd copy matches the original if file contents are identical.
compute_content_hash() {
  local dir="$1"

  (
    cd "$dir"
    find . -type f "${_FIND_EXCLUDES[@]}" 2>/dev/null \
    | LC_ALL=C sort \
    | while IFS= read -r f; do
        # Emit "path\0content" for each file so renames are detected
        printf '%s\0' "$f"
        cat "$f"
        printf '\0'
      done \
    | shasum \
    | cut -d' ' -f1
  )
}

# ── Compute hashes ────────────────────────────────────────────────────────────

REPO_HASH=$(compute_content_hash "$REPO_PLUGIN_DIR")
CACHE_HASH=$(compute_content_hash "$CACHE_DIR")

# ── Diff computation ──────────────────────────────────────────────────────────

# Build a fast file manifest: relative_path \t size_bytes
# Uses wc -c for portability (works on both macOS and Linux).
build_file_manifest() {
  local dir="$1"
  (
    cd "$dir"
    find . -type f "${_FIND_EXCLUDES[@]}" -print0 2>/dev/null \
    | while IFS= read -r -d '' f; do
        local sz
        sz=$(wc -c < "$f")
        printf '%s\t%s\n' "$f" "$sz"
      done \
    | LC_ALL=C sort
  )
}

CHANGED_FILES=()

if [[ "$REPO_HASH" != "$CACHE_HASH" ]]; then
  _tmp_repo=$(mktemp)
  _tmp_cache=$(mktemp)
  build_file_manifest "$REPO_PLUGIN_DIR" > "$_tmp_repo"
  build_file_manifest "$CACHE_DIR" > "$_tmp_cache"

  # Extract only file paths from each manifest for comparison
  _tmp_repo_paths=$(mktemp)
  _tmp_cache_paths=$(mktemp)
  cut -f1 "$_tmp_repo" | sort > "$_tmp_repo_paths"
  cut -f1 "$_tmp_cache" | sort > "$_tmp_cache_paths"

  # Files only in repo (added)
  while IFS= read -r f; do
    CHANGED_FILES+=("+ $f")
  done < <(comm -23 "$_tmp_repo_paths" "$_tmp_cache_paths")

  # Files only in cache (removed from repo)
  while IFS= read -r f; do
    CHANGED_FILES+=("- $f")
  done < <(comm -13 "$_tmp_repo_paths" "$_tmp_cache_paths")

  # Files in both but with different sizes (modified)
  while IFS= read -r f; do
    repo_size=$(grep -F "${f}	" "$_tmp_repo" | head -1 | cut -f2)
    cache_size=$(grep -F "${f}	" "$_tmp_cache" | head -1 | cut -f2)
    if [[ "$repo_size" != "$cache_size" ]]; then
      CHANGED_FILES+=("~ $f")
    fi
  done < <(comm -12 "$_tmp_repo_paths" "$_tmp_cache_paths")

  rm -f "$_tmp_repo" "$_tmp_cache" "$_tmp_repo_paths" "$_tmp_cache_paths"
fi

# ── Output ────────────────────────────────────────────────────────────────────

STATUS="FRESH"
EXIT_CODE=0
if [[ "$REPO_HASH" != "$CACHE_HASH" ]]; then
  STATUS="STALE"
  EXIT_CODE=1
fi

if [[ "$OUTPUT_JSON" -eq 1 ]]; then
  changed_json="[]"
  if [[ ${#CHANGED_FILES[@]} -gt 0 ]]; then
    changed_json=$(printf '%s\n' "${CHANGED_FILES[@]}" | jq -R . | jq -s .)
  fi
  jq -n \
    --arg status "$STATUS" \
    --arg repo_hash "$REPO_HASH" \
    --arg cache_hash "$CACHE_HASH" \
    --arg repo_dir "$REPO_PLUGIN_DIR" \
    --arg cache_dir "$CACHE_DIR" \
    --argjson changed "$changed_json" \
    '{
      status: $status,
      repo_hash: $repo_hash,
      cache_hash: $cache_hash,
      repo_dir: $repo_dir,
      cache_dir: $cache_dir,
      changed_files: $changed
    }'
elif [[ "$QUIET" -eq 1 ]]; then
  echo "$STATUS"
else
  echo "Plugin cache status: $STATUS"
  echo "  Repo hash:  ${REPO_HASH}"
  echo "  Cache hash: ${CACHE_HASH}"
  echo "  Repo dir:   ${REPO_PLUGIN_DIR}"
  echo "  Cache dir:  ${CACHE_DIR}"
  if [[ ${#CHANGED_FILES[@]} -gt 0 ]]; then
    echo ""
    echo "Changed files (${#CHANGED_FILES[@]}):"
    for f in "${CHANGED_FILES[@]}"; do
      echo "  $f"
    done
  fi
fi

# ── Optional fix ──────────────────────────────────────────────────────────────

if [[ "$FIX" -eq 1 && "$STATUS" == "STALE" ]]; then
  echo ""
  echo "Refreshing cache from repo..."
  rsync -a --delete \
    --exclude='.venv' --exclude='node_modules' \
    --exclude='logs' --exclude='tmp' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='*.log' \
    --exclude='.deployed-commit' --exclude='.content-hash' --exclude='.DS_Store' \
    "${REPO_PLUGIN_DIR}/" "${CACHE_DIR}/" 2>/dev/null

  # Update metadata
  _current_commit=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
  echo "$_current_commit" > "${CACHE_DIR}/.deployed-commit"

  NEW_HASH=$(compute_content_hash "$CACHE_DIR")
  echo "$NEW_HASH" > "${CACHE_DIR}/.content-hash"

  echo "Cache refreshed. New hash: ${NEW_HASH}"
  EXIT_CODE=0
fi

exit "$EXIT_CODE"
