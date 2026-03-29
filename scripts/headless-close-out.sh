#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# headless-close-out.sh — Decomposed close-out pipeline [OMN-6935]
#
# Replaces mega-session close-out with parallel headless claude -p invocations.
# Each task is scoped to one repo and one action, completing in <15 minutes.
#
# Pattern: docs/patterns/headless_decomposition.md (OMN-6927)
#
# Usage:
#   ./scripts/headless-close-out.sh [--repos REPO1,REPO2,...] [--skip-release] [--dry-run]

set -euo pipefail

# --- Configuration ---
OMNI_HOME="${OMNI_HOME:-/Volumes/PRO-G40/Code/omni_home}"
DEFAULT_REPOS="omnibase_core,omnibase_spi,omnibase_infra,omniclaude,omniintelligence,omnimemory"
TIMEOUT_SECONDS=900  # 15 minutes per task
RUN_ID=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
STATE_DIR="/tmp/headless-state/${RUN_ID}"

# --- Parse arguments ---
REPOS="${DEFAULT_REPOS}"
SKIP_RELEASE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repos) REPOS="$2"; shift 2 ;;
    --skip-release) SKIP_RELEASE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

IFS=',' read -ra REPO_LIST <<< "$REPOS"

echo "=== Headless Close-Out Pipeline ==="
echo "Run ID:  ${RUN_ID}"
echo "State:   ${STATE_DIR}"
echo "Repos:   ${REPO_LIST[*]}"
echo "Timeout: ${TIMEOUT_SECONDS}s per task"
echo ""

mkdir -p "${STATE_DIR}"

# Write run manifest
cat > "${STATE_DIR}/manifest.json" <<MANIFEST
{
  "schema_version": "1.0",
  "run_id": "${RUN_ID}",
  "repos": $(printf '%s\n' "${REPO_LIST[@]}" | jq -R . | jq -s .),
  "skip_release": ${SKIP_RELEASE},
  "started_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
MANIFEST

# --- Helper: run a headless claude task ---
run_headless() {
  local repo="$1"
  local stage="$2"
  local prompt="$3"
  local allowed_tools="$4"
  local output_file="${STATE_DIR}/${repo}/${stage}.json"
  local log_file="${STATE_DIR}/${repo}/${stage}.log"

  mkdir -p "${STATE_DIR}/${repo}"

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] Would run: claude -p for ${repo}/${stage}"
    echo "{\"status\": \"dry-run\", \"repo\": \"${repo}\", \"stage\": \"${stage}\"}" > "$output_file"
    return 0
  fi

  echo "[${repo}/${stage}] Starting..."
  timeout "${TIMEOUT_SECONDS}" claude -p \
    --print \
    --permission-mode auto \
    --allowedTools "${allowed_tools}" \
    "${prompt}" \
    > "$output_file" 2>"$log_file" || {
      local exit_code=$?
      echo "[${repo}/${stage}] FAILED (exit ${exit_code})"
      echo "{\"status\": \"failed\", \"exit_code\": ${exit_code}, \"repo\": \"${repo}\", \"stage\": \"${stage}\"}" > "$output_file"
      return "${exit_code}"
    }
  echo "[${repo}/${stage}] Complete"
}

# =============================================
# Stage 1: Merge Sweep (parallel across repos)
# =============================================
echo "=== Stage 1: Merge Sweep ==="
pids=()
for repo in "${REPO_LIST[@]}"; do
  run_headless "$repo" "merge-sweep" \
    "Merge-sweep for ${repo}: list all open PRs in OmniNode-ai/${repo} with passing CI checks. For each green PR, merge it. Report: {\"status\": \"success|partial|failed\", \"prs_merged\": [...], \"prs_skipped\": [...], \"prs_failed\": [...]}. Working directory: ${OMNI_HOME}/${repo}" \
    "Bash(git:*,gh:*) Read Glob Grep" &
  pids+=($!)
done

# Wait for all merge-sweeps
failures=0
for pid in "${pids[@]}"; do
  wait "$pid" || ((failures++))
done
echo "Stage 1 complete. Failures: ${failures}"
echo ""

# =============================================
# Stage 2: Release (parallel, reads Stage 1)
# =============================================
if [[ "$SKIP_RELEASE" == "false" ]]; then
  echo "=== Stage 2: Release ==="
  pids=()
  for repo in "${REPO_LIST[@]}"; do
    merge_result="${STATE_DIR}/${repo}/merge-sweep.json"
    if [[ -f "$merge_result" ]]; then
      run_headless "$repo" "release" \
        "Release ${repo}: check if there are unreleased commits since the last tag. If yes, bump the version, create a tag, and push. Read the merge-sweep result for context: $(cat "$merge_result" 2>/dev/null | head -c 2000). Report: {\"status\": \"success|skipped|failed\", \"version\": \"...\", \"tag\": \"...\"}. Working directory: ${OMNI_HOME}/${repo}" \
        "Bash(git:*,gh:*,uv:*) Read Edit Write Glob Grep" &
      pids+=($!)
    fi
  done

  failures=0
  for pid in "${pids[@]}"; do
    wait "$pid" || ((failures++))
  done
  echo "Stage 2 complete. Failures: ${failures}"
  echo ""
fi

# =============================================
# Stage 3: Summary
# =============================================
echo "=== Stage 3: Summary ==="

# Collect all stage results into summary
{
  echo "{"
  echo "  \"schema_version\": \"1.0\","
  echo "  \"run_id\": \"${RUN_ID}\","
  echo "  \"completed_at\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\","
  echo "  \"repos\": {"

  first=true
  for repo in "${REPO_LIST[@]}"; do
    if [[ "$first" == "true" ]]; then first=false; else echo ","; fi
    echo "    \"${repo}\": {"
    echo "      \"merge_sweep\": $(cat "${STATE_DIR}/${repo}/merge-sweep.json" 2>/dev/null || echo '{"status": "not_run"}'),"
    echo "      \"release\": $(cat "${STATE_DIR}/${repo}/release.json" 2>/dev/null || echo '{"status": "not_run"}')"
    printf "    }"
  done

  echo ""
  echo "  }"
  echo "}"
} > "${STATE_DIR}/summary.json"

echo ""
echo "=== Close-Out Complete ==="
echo "Summary: ${STATE_DIR}/summary.json"
echo "Run ID:  ${RUN_ID}"
