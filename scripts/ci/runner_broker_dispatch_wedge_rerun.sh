#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# runner_broker_dispatch_wedge_rerun.sh — OMN-15776
#
# Targeted, signature-keyed auto-rerun for the proven GitHub Actions
# broker-dispatch/reconnect race (2026-08-09 ``omn15776-wedge`` ledger
# finding).
#
# MECHANISM (proven, not speculative — see the ticket comment history and
# docs/runbooks/runner-fleet-broker-dispatch-wedge.md):
#   GitHub's Actions broker dispatches a job to a self-hosted runner within
#   2-7s of that SAME runner finishing its previous (unrelated) job, exactly
#   while the runner's Runner.Listener is mid-reconnect on its broker
#   long-poll (every job completion triggers a retry/backoff storm on that
#   connection, observed 5-12s exponential backoff). The new dispatch lands
#   in that reconnect gap and is never delivered to the runner's active
#   message loop — no Runner.Worker process is ever spawned locally (so
#   local _diag logs show ZERO "Running job: <name>" entry — this is NOT a
#   crashed step 1) — while GitHub's server side records the assignment,
#   sets started_at, and independently times the orphaned assignment out at
#   a FIXED ~10m0-1s, unrelated to any declared workflow timeout-minutes or
#   any local watchdog threshold.
#
# WHY NO LOCAL FIX APPLIES: the drop happens in the GitHub Actions
# client/broker protocol path, before any local process this repo controls
# (Runner.Worker, or the existing OMN-14564 heartbeat watchdog in
# docker/runners/entrypoint.sh) can observe it — the watchdog's detection
# surface is idle-listener silence, and the Listener in this failure class is
# actively chattering through broker retries, never silent, and there is no
# Worker for the watchdog's Worker-running guard to observe either. The
# remediation therefore lives here, on the GitHub-API side of the boundary,
# not in entrypoint.sh.
#
# STRUCTURAL SIGNATURE (matched via the Jobs API, never log-text grepping —
# there is no log content to grep, because no Worker ever ran):
#   1. job.runner_name is set (self-hosted; GitHub-hosted jobs are excluded —
#      they are never serviced by this fleet and must never be touched here)
#   2. job.conclusion in {failure, cancelled}
#   3. job.steps is an empty array (no Worker spawned — distinguishes this
#      from a genuine content failure, which always has at least one step)
#   4. duration = completed_at - started_at is within
#      [WEDGE_MIN_DURATION_SECONDS, WEDGE_MAX_DURATION_SECONDS] — a tight
#      band around the observed fixed ~10m0-1s GitHub-side orphan timeout.
#      This is deliberately narrow: a job that happens to run steps=[] for
#      some other reason and complete at, say, 9 or 11 minutes is NOT this
#      signature and must not be silently relaunched.
#
# A job matching all four is a REPLAY of a dispatch that never reached local
# execution — reissuing it is not "retry on red" (which would launder a
# genuine content failure); it targets exactly the drop this mechanism
# describes, and only that.
#
# Usage:
#   ./scripts/ci/runner_broker_dispatch_wedge_rerun.sh
#   ./scripts/ci/runner_broker_dispatch_wedge_rerun.sh --dry-run
#   ./scripts/ci/runner_broker_dispatch_wedge_rerun.sh --repos omnibase_infra,omnibase_core
#   ./scripts/ci/runner_broker_dispatch_wedge_rerun.sh --state-dir /path/to/state
#
# Env (mirrors scripts/ci/runner_fleet_canary.sh's testable curl+env-override
# convention, NOT the gh CLI, so this script can be driven against a local
# HTTP stub in tests without touching the real GitHub API):
#   GITHUB_API_URL       API base (default https://api.github.com)
#   GITHUB_ORG           org name (default OmniNode-ai)
#   REPOS_CSV            comma-separated repo list to scan
#   RUNNER_GITHUB_TOKEN  bearer token (falls back to CROSS_REPO_PAT)
#   LOOKBACK_HOURS       how far back to scan runs (default 6)
#   MAX_RUNS_PER_REPO    runs listed per repo per scan (default 30)
#   WEDGE_MIN_DURATION_SECONDS / WEDGE_MAX_DURATION_SECONDS
#                        signature duration band (default 595 / 605 — a
#                        tight collar around the proven fixed ~600-601s)
#   MAX_RERUNS           circuit breaker on reruns issued per invocation
#
# Exit codes: 0 = scan completed (reruns issued or no candidates), 1 = error

set -euo pipefail

GITHUB_API_URL="${GITHUB_API_URL:-https://api.github.com}"
GITHUB_ORG="${GITHUB_ORG:-OmniNode-ai}"
REPOS_CSV="${REPOS_CSV:-omnibase_infra,omnibase_core,omniclaude,omnimarket,onex_change_control}"
LOOKBACK_HOURS="${LOOKBACK_HOURS:-6}"
MAX_RUNS_PER_REPO="${MAX_RUNS_PER_REPO:-30}"
WEDGE_MIN_DURATION_SECONDS="${WEDGE_MIN_DURATION_SECONDS:-595}"
WEDGE_MAX_DURATION_SECONDS="${WEDGE_MAX_DURATION_SECONDS:-605}"
MAX_RERUNS="${MAX_RERUNS:-10}"
STATE_DIR="${ONEX_STATE_DIR:-.onex_state}/runner-broker-dispatch-wedge-rerun"
DRY_RUN=false

TOKEN="${RUNNER_GITHUB_TOKEN:-${CROSS_REPO_PAT:-}}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --repos)
      REPOS_CSV="${2:?--repos requires a value}"; shift 2 ;;
    --state-dir)
      STATE_DIR="${2:?--state-dir requires a path}"; shift 2 ;;
    --help|-h)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${TOKEN}" ]]; then
  echo "ERROR: no API token — set RUNNER_GITHUB_TOKEN or CROSS_REPO_PAT" >&2
  exit 1
fi

mkdir -p "${STATE_DIR}"
SESSION_RERUN_LOG="${STATE_DIR}/session-reruns.json"
LOG_FILE="${STATE_DIR}/scan.log"

log() {
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[${ts}] [broker-dispatch-wedge-rerun] $*" | tee -a "${LOG_FILE}" >&2
}

api_get() {
  local path="$1"
  curl -fsS \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "${GITHUB_API_URL}${path}" 2>>"${LOG_FILE}" || echo ""
}

api_post_rerun() {
  local repo="$1"
  local job_id="$2"
  curl -fsS -X POST \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "${GITHUB_API_URL}/repos/${GITHUB_ORG}/${repo}/actions/jobs/${job_id}/rerun" \
    >>"${LOG_FILE}" 2>&1
}

already_rerun() {
  local job_id="$1"
  [[ -f "${SESSION_RERUN_LOG}" ]] || return 1
  jq -e --arg id "${job_id}" '.rerun_ids | index($id) != null' \
    "${SESSION_RERUN_LOG}" >/dev/null 2>&1
}

record_rerun() {
  local job_id="$1" repo="$2"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -f "${SESSION_RERUN_LOG}" ]]; then
    tmp="$(mktemp)"
    jq --arg id "${job_id}" --arg repo "${repo}" --arg ts "${ts}" \
      '.rerun_ids += [$id] | .entries += [{job_id: $id, repo: $repo, rerun_at: $ts}]' \
      "${SESSION_RERUN_LOG}" > "${tmp}" && mv "${tmp}" "${SESSION_RERUN_LOG}"
  else
    jq -n --arg id "${job_id}" --arg repo "${repo}" --arg ts "${ts}" \
      '{rerun_ids: [$id], entries: [{job_id: $id, repo: $repo, rerun_at: $ts}]}' \
      > "${SESSION_RERUN_LOG}"
  fi
}

cutoff_iso="$(python3 -c "
from datetime import datetime, timezone, timedelta
print((datetime.now(timezone.utc) - timedelta(hours=${LOOKBACK_HOURS})).strftime('%Y-%m-%dT%H:%M:%SZ'))
")"

log "Starting broker-dispatch-wedge scan (OMN-15776)"
log "  repos: ${REPOS_CSV} | lookback: ${LOOKBACK_HOURS}h | dry_run: ${DRY_RUN}"
log "  signature band: [${WEDGE_MIN_DURATION_SECONDS}s, ${WEDGE_MAX_DURATION_SECONDS}s], steps=[], runner_name set, conclusion in {failure,cancelled}"

rerun_count=0
candidate_count=0
scanned_count=0

for repo in $(echo "${REPOS_CSV}" | tr ',' '\n'); do
  repo="$(echo "${repo}" | xargs)"
  [[ -z "${repo}" ]] && continue

  log "Scanning ${GITHUB_ORG}/${repo} (last ${LOOKBACK_HOURS}h)..."

  runs_json="$(api_get "/repos/${GITHUB_ORG}/${repo}/actions/runs?status=completed&per_page=${MAX_RUNS_PER_REPO}")"
  if [[ -z "${runs_json}" ]]; then
    log "  WARN: could not list runs for ${repo} — skipping"
    continue
  fi

  run_ids="$(echo "${runs_json}" | jq -r --arg cutoff "${cutoff_iso}" '
    .workflow_runs[]?
    | select(.created_at >= $cutoff)
    | .id
  ' 2>/dev/null || true)"

  while IFS= read -r run_id; do
    [[ -z "${run_id}" ]] && continue

    jobs_json="$(api_get "/repos/${GITHUB_ORG}/${repo}/actions/runs/${run_id}/jobs")"
    [[ -z "${jobs_json}" ]] && continue

    while IFS= read -r job_entry; do
      [[ -z "${job_entry}" ]] && continue
      scanned_count=$((scanned_count + 1))

      job_id="$(echo "${job_entry}" | jq -r '.id')"
      job_name="$(echo "${job_entry}" | jq -r '.name')"
      runner_name="$(echo "${job_entry}" | jq -r '.runner_name // empty')"
      conclusion="$(echo "${job_entry}" | jq -r '.conclusion // empty')"
      step_count="$(echo "${job_entry}" | jq -r '(.steps // []) | length')"
      started_at="$(echo "${job_entry}" | jq -r '.started_at // empty')"
      completed_at="$(echo "${job_entry}" | jq -r '.completed_at // empty')"

      # 1. self-hosted only.
      [[ -z "${runner_name}" ]] && continue
      # 2. conclusion class.
      if [[ "${conclusion}" != "failure" && "${conclusion}" != "cancelled" ]]; then
        continue
      fi
      # 3. zero steps — no Worker ever spawned.
      [[ "${step_count}" != "0" ]] && continue
      # started_at/completed_at required to compute duration.
      [[ -z "${started_at}" || -z "${completed_at}" ]] && continue

      duration="$(python3 -c "
from datetime import datetime
fmt = '%Y-%m-%dT%H:%M:%SZ'
a = datetime.strptime('${started_at}', fmt)
b = datetime.strptime('${completed_at}', fmt)
print(int((b - a).total_seconds()))
" 2>/dev/null || echo "")"
      [[ -z "${duration}" ]] && continue

      # 4. tight band around the proven fixed ~10m0-1s server-side timeout.
      if (( duration < WEDGE_MIN_DURATION_SECONDS || duration > WEDGE_MAX_DURATION_SECONDS )); then
        continue
      fi

      log "  CANDIDATE job ${job_id} (${job_name}, runner=${runner_name}, conclusion=${conclusion}, duration=${duration}s, steps=0) — broker-dispatch-wedge signature (OMN-15776)"
      candidate_count=$((candidate_count + 1))
      echo "CANDIDATE job_id=${job_id} repo=${repo} run_id=${run_id} name=${job_name}"

      if already_rerun "${job_id}"; then
        log "  SKIP job ${job_id}: already rerun this session"
        continue
      fi

      if (( rerun_count >= MAX_RERUNS )); then
        log "  SKIP job ${job_id}: max reruns (${MAX_RERUNS}) reached for this invocation"
        continue
      fi

      if [[ "${DRY_RUN}" == "true" ]]; then
        log "  [DRY-RUN] Would rerun job ${job_id} (${repo})"
      else
        log "  Issuing targeted rerun for job ${job_id} (${repo})"
        if api_post_rerun "${repo}" "${job_id}"; then
          log "  RERUN issued: job ${job_id}"
          record_rerun "${job_id}" "${repo}"
          rerun_count=$((rerun_count + 1))
        else
          log "  WARN: rerun request failed for job ${job_id}"
        fi
      fi
    done < <(echo "${jobs_json}" | jq -c '.jobs[]?')
  done <<< "${run_ids}"
done

log "Scan complete: scanned=${scanned_count} candidates=${candidate_count} reruns_issued=${rerun_count} dry_run=${DRY_RUN}"
