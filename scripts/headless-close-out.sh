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
#                                   [--skip-validation]
#
# Templates: scripts/headless-pipeline/TEMPLATES.md [OMN-6984]
# Validation: scripts/headless-pipeline/validate-state.sh [OMN-6988]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATE_SCRIPT="${SCRIPT_DIR}/headless-pipeline/validate-state.sh"

# --- Configuration ---
OMNI_HOME="${OMNI_HOME:-/Volumes/PRO-G40/Code/omni_home}"
DEFAULT_REPOS="omnibase_core,omnibase_spi,omnibase_infra,omniclaude,omniintelligence,omnimemory"
TIMEOUT_SECONDS=900  # 15 minutes per task
RUN_ID=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
STATE_DIR="/tmp/headless-state/${RUN_ID}"

# --- Stage Templates [OMN-6984] ---
# Per-stage command templates (tool allowlists, prompts, timeouts).
# See scripts/headless-pipeline/TEMPLATES.md for full documentation.
declare -A STAGE_TOOLS
STAGE_TOOLS[merge-sweep]="Bash(git:*,gh:*) Read Glob Grep"
STAGE_TOOLS[release]="Bash(git:*,gh:*,uv:*) Read Edit Write Glob Grep"
STAGE_TOOLS[redeploy]="Bash(git:*,docker:*,uv:*) Read Edit Write Glob Grep"
STAGE_TOOLS[ticket-close]="Bash(gh:*) Read Grep mcp__linear-server__*"
STAGE_TOOLS[integration-sweep]="Bash(git:*,uv:*,docker:*) Read Glob Grep"

declare -A STAGE_TIMEOUTS
STAGE_TIMEOUTS[merge-sweep]=900
STAGE_TIMEOUTS[release]=900
STAGE_TIMEOUTS[redeploy]=1200
STAGE_TIMEOUTS[ticket-close]=600
STAGE_TIMEOUTS[integration-sweep]=900

# get_stage_tools: Return the tool allowlist for a stage
get_stage_tools() {
  local stage="$1"
  echo "${STAGE_TOOLS[$stage]:-Bash Read Glob Grep}"
}

# get_stage_timeout: Return the timeout for a stage
get_stage_timeout() {
  local stage="$1"
  echo "${STAGE_TIMEOUTS[$stage]:-$TIMEOUT_SECONDS}"
}

# --- Parse arguments ---
REPOS="${DEFAULT_REPOS}"
SKIP_RELEASE=false
DRY_RUN=false
SKIP_VALIDATION=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repos) REPOS="$2"; shift 2 ;;
    --skip-release) SKIP_RELEASE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --skip-validation) SKIP_VALIDATION=true; shift ;;
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

  # Validate state file schema [OMN-6988]
  if [[ "$SKIP_VALIDATION" != "true" && -f "$VALIDATE_SCRIPT" ]]; then
    if ! bash "$VALIDATE_SCRIPT" "$output_file" --stage "$stage" --quiet; then
      echo "[${repo}/${stage}] WARNING: State file validation failed"
    fi
  fi
}

# =============================================
# Stage 1: Merge Sweep (parallel across repos)
# =============================================
echo "=== Stage 1: Merge Sweep ==="
pids=()
for repo in "${REPO_LIST[@]}"; do
  run_headless "$repo" "merge-sweep" \
    "Merge-sweep for ${repo}: list all open PRs in OmniNode-ai/${repo} with passing CI checks. For each green PR, merge it using 'gh pr merge --squash --auto'. Skip PRs with: failing CI, unresolved review comments, draft status, or merge conflicts. Working directory: ${OMNI_HOME}/${repo}. Report JSON: {\"schema_version\": \"1.0\", \"stage\": \"merge-sweep\", \"repo\": \"${repo}\", \"status\": \"success|partial|failed\", \"prs_merged\": [{\"number\": N, \"title\": \"...\"}], \"prs_skipped\": [{\"number\": N, \"reason\": \"...\"}], \"prs_failed\": [{\"number\": N, \"error\": \"...\"}]}" \
    "$(get_stage_tools merge-sweep)" &
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
        "Release ${repo}: check if there are unreleased commits since the last tag. If yes, bump the version in pyproject.toml, update CHANGELOG, create a git tag, and push. Follow existing release conventions. Read merge-sweep result for context: $(cat "$merge_result" 2>/dev/null | head -c 2000). Working directory: ${OMNI_HOME}/${repo}. Report JSON: {\"schema_version\": \"1.0\", \"stage\": \"release\", \"repo\": \"${repo}\", \"status\": \"success|skipped|failed\", \"version\": \"X.Y.Z\", \"tag\": \"vX.Y.Z\", \"commits_since_last_tag\": N}" \
        "$(get_stage_tools release)" &
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

# Validate all state files [OMN-6988]
if [[ "$SKIP_VALIDATION" != "true" && -f "$VALIDATE_SCRIPT" ]]; then
  echo ""
  echo "=== State File Validation ==="
  bash "$VALIDATE_SCRIPT" "${STATE_DIR}" --all || {
    echo "WARNING: Some state files failed validation. Check individual results above."
  }
fi

echo ""
echo "=== Close-Out Complete ==="
echo "Summary: ${STATE_DIR}/summary.json"
echo "Run ID:  ${RUN_ID}"
