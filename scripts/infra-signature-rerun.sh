#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# infra-signature-rerun.sh — OMN-13040 (retro B-5)
#
# Auto-rerun GitHub Actions runs whose failure logs match known infra signatures
# (disk casualty, runner network wedge) IF those runs started BEFORE the last
# known recovery timestamp for the affected runner.
#
# WHY: Without this, a PR that was red because of runner-22's disk dying stays
# red until a human notices and reruns it. The merge-sweep tick calls this script
# after its main sweep so infra-casualty reds self-heal without human involvement.
#
# Recovery logic:
#   A run is a rerun candidate when ALL of the following are true:
#     1. The run is in a failed/cancelled state (not queued/pending/in-progress).
#     2. The run's logs match one or more INFRA_SIGNATURES.
#     3. The run started BEFORE the last recorded recovery timestamp for the
#        runner named in the log (or the default recovery time if no runner-
#        specific record exists).
#     4. The run has not already been rerun in this session (tracked in state
#        to avoid thrashing).
#
# Known infra signatures (OMN-13040 + OMN-13045):
#   RUNNER-DISK:     — runner-disk-preflight annotation (disk casualty)
#   "compose services are healthy but unreachable from the runner"
#                    — OMN-13045: runner-1 Docker DNS/network wedge
#   "No space left on device"
#                    — raw OS-level disk full error
#   "DiskWriteError" — Redpanda disk-full crash variant
#   "exit code 137"  — OOM-kill (may co-occur with disk full)
#
# Recovery timestamps:
#   Stored in ${RECOVERY_STATE_FILE} as JSON: {"runner": "timestamp_iso"}
#   The merge-sweep operator or the runner-recycle script writes entries here
#   when a runner is confirmed healthy. If a runner has no entry, DEFAULT_RECOVERY
#   is used (24h ago, conservative).
#
# Usage:
#   ./scripts/infra-signature-rerun.sh
#   ./scripts/infra-signature-rerun.sh --dry-run
#   ./scripts/infra-signature-rerun.sh --repos omnibase_infra,omniclaude
#   ./scripts/infra-signature-rerun.sh --lookback-hours 48
#   ./scripts/infra-signature-rerun.sh --recovery-state /path/to/recovery.json
#
# Dependencies: gh CLI (authenticated), jq
#
# Exit codes: 0 = completed (rerun issued or no candidates found), 1 = error

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORG="OmniNode-ai"

# Repos to scan for infra-casualty runs. Caller can override with --repos.
DEFAULT_REPOS="omnibase_infra,omniclaude,omnibase_core,omnibase_spi,omnibase_compat,omniintelligence,omnimemory,omninode_infra,onex_change_control,omnimarket"

REPOS_CSV="${DEFAULT_REPOS}"
DRY_RUN=false
# How far back to look for failed runs (hours).
LOOKBACK_HOURS=24
# Max runs to inspect per repo (avoid hammering the API on large repos).
MAX_RUNS_PER_REPO=20
# Max reruns to issue in a single invocation (circuit breaker).
MAX_RERUNS=5
# Where recovery timestamps are stored.
RECOVERY_STATE_FILE="${ONEX_STATE_DIR:-.onex_state}/infra-signature-rerun/recovery-state.json"
# Per-session rerun log (avoids duplicate reruns across multiple tick cycles).
SESSION_RERUN_LOG="${ONEX_STATE_DIR:-.onex_state}/infra-signature-rerun/session-reruns.json"
LOG_FILE="${ONEX_STATE_DIR:-.onex_state}/infra-signature-rerun/rerun.log"

# ---------------------------------------------------------------------------
# Known infra failure signatures (matched against run log text)
# ---------------------------------------------------------------------------

# Each entry is a grep-E pattern. A run matching ANY of these is a candidate.
INFRA_SIGNATURES=(
  # OMN-13040: runner-disk-preflight annotation
  "RUNNER-DISK:"
  # OMN-13045: compose services healthy but unreachable (runner network wedge)
  "compose services are healthy but unreachable from the runner"
  # Raw OS disk full
  "No space left on device"
  # Redpanda disk-full crash
  "DiskWriteError"
  # OOM kill (may co-occur with disk full)
  "exit code 137"
  # Kafka metadata timeout that accompanies disk full (secondary indicator)
  "Kafka Schema Handshake FAILURE"
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --repos)
      if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "ERROR: --repos requires a non-empty value" >&2; exit 1
      fi
      REPOS_CSV="$2"; shift 2 ;;
    --repos=*)
      REPOS_CSV="${1#*=}"
      [[ -z "$REPOS_CSV" ]] && { echo "ERROR: --repos requires a non-empty value" >&2; exit 1; }
      shift ;;
    --lookback-hours)
      LOOKBACK_HOURS="${2:?--lookback-hours requires a value}"; shift 2 ;;
    --recovery-state)
      RECOVERY_STATE_FILE="${2:?--recovery-state requires a path}"; shift 2 ;;
    --help|-h)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mkdir -p "$(dirname "$RECOVERY_STATE_FILE")" "$(dirname "$SESSION_RERUN_LOG")" "$(dirname "$LOG_FILE")"

log() {
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[${ts}] [infra-signature-rerun] $*" | tee -a "$LOG_FILE" >&2
}

# Compute the lookback cutoff (ISO 8601 UTC).
cutoff_iso="$(python3 -c "
from datetime import datetime, timezone, timedelta
cutoff = datetime.now(timezone.utc) - timedelta(hours=${LOOKBACK_HOURS})
print(cutoff.strftime('%Y-%m-%dT%H:%M:%SZ'))
")"

log "Starting infra-signature rerun scan"
log "  repos: ${REPOS_CSV}"
log "  lookback: ${LOOKBACK_HOURS}h (cutoff: ${cutoff_iso})"
log "  dry_run: ${DRY_RUN}"
log "  recovery_state: ${RECOVERY_STATE_FILE}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Get recovery timestamp for a runner name.
# Returns ISO 8601 UTC, or a default 24h-ago timestamp if no record exists.
get_recovery_ts() {
  local runner_name="$1"
  if [[ -f "${RECOVERY_STATE_FILE}" ]]; then
    local ts
    ts="$(jq -r --arg r "${runner_name}" '.[$r] // ""' "${RECOVERY_STATE_FILE}" 2>/dev/null || echo "")"
    if [[ -n "$ts" ]]; then
      echo "$ts"; return 0
    fi
  fi
  # Default: 24h ago (conservative — assume recovery has happened by then)
  python3 -c "
from datetime import datetime, timezone, timedelta
print((datetime.now(timezone.utc) - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ'))
"
}

# Compare two ISO 8601 UTC timestamps. Returns 0 (true) if $1 < $2.
ts_before() {
  python3 -c "
from datetime import datetime, timezone
fmt = '%Y-%m-%dT%H:%M:%SZ'
a = datetime.strptime('$1', fmt).replace(tzinfo=timezone.utc)
b = datetime.strptime('$2', fmt).replace(tzinfo=timezone.utc)
exit(0 if a < b else 1)
"
}

# Check if a run_id has already been rerun in this session.
already_rerun() {
  local run_id="$1"
  if [[ -f "${SESSION_RERUN_LOG}" ]]; then
    jq -e --arg id "${run_id}" '.rerun_ids | index($id) != null' "${SESSION_RERUN_LOG}" >/dev/null 2>&1
    return $?
  fi
  return 1
}

# Record a rerun in the session log.
record_rerun() {
  local run_id="$1"
  local repo="$2"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -f "${SESSION_RERUN_LOG}" ]]; then
    tmp="$(mktemp)"
    jq --arg id "${run_id}" --arg repo "${repo}" --arg ts "${ts}" \
      '.rerun_ids += [$id] | .entries += [{run_id: $id, repo: $repo, rerun_at: $ts}]' \
      "${SESSION_RERUN_LOG}" > "$tmp" && mv "$tmp" "${SESSION_RERUN_LOG}"
  else
    jq -n --arg id "${run_id}" --arg repo "${repo}" --arg ts "${ts}" \
      '{rerun_ids: [$id], entries: [{run_id: $id, repo: $repo, rerun_at: $ts}]}' \
      > "${SESSION_RERUN_LOG}"
  fi
}

# ---------------------------------------------------------------------------
# Write a recovery timestamp for a runner (called by operator or recycle scripts)
# ---------------------------------------------------------------------------

record_runner_recovery() {
  local runner_name="$1"
  local ts="${2:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
  local state_file="${RECOVERY_STATE_FILE}"
  mkdir -p "$(dirname "${state_file}")"
  if [[ -f "${state_file}" ]]; then
    tmp="$(mktemp)"
    jq --arg runner "${runner_name}" --arg ts "${ts}" '.[$runner] = $ts' \
      "${state_file}" > "$tmp" && mv "$tmp" "${state_file}"
  else
    jq -n --arg runner "${runner_name}" --arg ts "${ts}" '{($runner): $ts}' \
      > "${state_file}"
  fi
  log "Recorded recovery for runner '${runner_name}' at ${ts}"
}

# If called as record-recovery, just write the state and exit.
if [[ "${1:-}" == "record-recovery" ]]; then
  runner="${2:?Usage: $0 record-recovery <runner-name> [<timestamp>]}"
  ts="${3:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
  record_runner_recovery "$runner" "$ts"
  exit 0
fi

# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

rerun_count=0
candidate_count=0
scanned_count=0

for repo_name in $(echo "${REPOS_CSV}" | tr ',' '\n'); do
  repo_name="${repo_name#"${repo_name%%[! ]*}"}"
  repo_name="${repo_name%"${repo_name##*[! ]}"}"
  [[ -z "${repo_name}" ]] && continue
  if echo "${repo_name}" | grep -qE '[^a-zA-Z0-9_.-]'; then
    log "WARN: skipping invalid repo token '${repo_name}'"
    continue
  fi

  full_repo="${ORG}/${repo_name}"
  log "Scanning ${full_repo} (last ${LOOKBACK_HOURS}h)..."

  # List recent failed/cancelled workflow runs.
  runs_json="$(gh run list \
    --repo "${full_repo}" \
    --status failure \
    --limit "${MAX_RUNS_PER_REPO}" \
    --json databaseId,name,createdAt,headBranch,headSha,workflowName,conclusion,url \
    2>>"${LOG_FILE}")" || {
    log "WARN: gh run list failed for ${full_repo} — skipping"
    continue
  }

  run_count="$(echo "${runs_json}" | jq 'length')"
  log "  ${run_count} failed runs found"

  while IFS= read -r run_entry; do
    [[ -z "${run_entry}" ]] && continue
    run_id="$(echo "${run_entry}" | jq -r '.databaseId')"
    run_name="$(echo "${run_entry}" | jq -r '.workflowName // .name')"
    run_created="$(echo "${run_entry}" | jq -r '.createdAt')"
    run_url="$(echo "${run_entry}" | jq -r '.url')"
    head_branch="$(echo "${run_entry}" | jq -r '.headBranch')"

    scanned_count=$((scanned_count + 1))

    # Skip if before the lookback window.
    if ! ts_before "${cutoff_iso}" "${run_created}" 2>/dev/null; then
      # run_created is older than cutoff — skip
      if ts_before "${run_created}" "${cutoff_iso}" 2>/dev/null; then
        continue
      fi
    fi

    # Skip if already rerun this session.
    if already_rerun "${run_id}"; then
      log "  SKIP run ${run_id} (${run_name}): already rerun this session"
      continue
    fi

    # Download the run log and check for infra signatures.
    log "  Checking run ${run_id} (${run_name}, branch=${head_branch})"
    run_log="$(gh run view "${run_id}" --repo "${full_repo}" --log 2>>"${LOG_FILE}" || echo "")"

    matched_signature=""
    for sig in "${INFRA_SIGNATURES[@]}"; do
      if echo "${run_log}" | grep -qE "${sig}" 2>/dev/null; then
        matched_signature="${sig}"
        break
      fi
    done

    if [[ -z "${matched_signature}" ]]; then
      continue
    fi

    log "  CANDIDATE run ${run_id} (${run_name}): matched signature '${matched_signature}'"
    candidate_count=$((candidate_count + 1))

    # Extract runner name from the log (looks for "runner_name=" or "Runner: <name>").
    runner_name="$(echo "${run_log}" | grep -oE '(runner_name|Runner)[:=]\s*\S+' | head -1 | grep -oE '\S+$' || echo "unknown")"
    log "  Runner identified: ${runner_name}"

    # Check if run started BEFORE the last recovery timestamp for this runner.
    recovery_ts="$(get_recovery_ts "${runner_name}")"
    log "  Recovery timestamp for '${runner_name}': ${recovery_ts}"

    if ! ts_before "${run_created}" "${recovery_ts}" 2>/dev/null; then
      log "  SKIP run ${run_id}: run started at ${run_created}, recovery was at ${recovery_ts} — run is NOT a pre-recovery casualty"
      continue
    fi

    log "  RERUN ELIGIBLE: run ${run_id} started at ${run_created} (before recovery at ${recovery_ts})"

    if (( rerun_count >= MAX_RERUNS )); then
      log "  SKIP: max reruns (${MAX_RERUNS}) reached for this invocation — will process remainder next tick"
      break 2
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
      log "  [DRY-RUN] Would rerun run ${run_id} for ${full_repo} (${run_url})"
    else
      log "  Issuing rerun for run ${run_id} (${full_repo} ${run_url})"
      if gh run rerun "${run_id}" --repo "${full_repo}" --failed >>"${LOG_FILE}" 2>&1; then
        log "  RERUN issued: ${run_id}"
        record_rerun "${run_id}" "${full_repo}"
        rerun_count=$((rerun_count + 1))
      else
        log "  WARN: gh run rerun failed for ${run_id} — skipping"
      fi
    fi

  done < <(echo "${runs_json}" | jq -c '.[]')
done

log "Scan complete: scanned=${scanned_count} candidates=${candidate_count} reruns_issued=${rerun_count} dry_run=${DRY_RUN}"
