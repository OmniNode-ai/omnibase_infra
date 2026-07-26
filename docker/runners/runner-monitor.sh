#!/usr/bin/env bash
# runner-monitor.sh — Self-hosted runner health monitor with Slack alerts
# Deployed to: 192.168.86.201:~/.omnibase/runners/runner-monitor.sh
# Cron: */3 * * * * (every 3 minutes)
# Ticket: OMN-13109 (silent-wedge + crash-loop detection)
#
# Checks configured omninode-runner-* containers and their GitHub Actions
# registrations. Fires a Slack alert on state transitions. Resolves silently
# when all recover. Uses a state file to prevent alert spam.
#
# Detection layers (each catches a failure mode the previous layer misses):
#
#   1. CONTAINER + REGISTRATION (legacy): container is `Up (healthy)` AND the
#      runner is `online` in the GitHub org pool. The OMN-12433 healthcheck
#      additionally proves github.com egress.
#
#   2. SILENT-WEDGE (OMN-13109): a runner can be `Up (healthy)` and `online`
#      while NOT pulling jobs — the Runner.Listener is alive and the long-poll
#      registration looks connected, but the runner is not picking up queued
#      work (last job days old). Container-only and registration-only checks
#      both PASS this state. We detect it by cross-referencing the GitHub
#      `busy` field against jobs that are QUEUED for our runner labels: if work
#      has been queued longer than WEDGE_QUEUE_AGE_SECONDS while the fleet sits
#      idle (online + not busy), the fleet is wedged.
#
#   3. OFFLINE-IDLE-REGISTRATION (OMN-13912): GitHub can report a runner
#      offline while Docker reports `Up (healthy)` and the local listener logs
#      prove it is still connected/listening or actively running jobs. The host
#      and listener are the primary truth for this class; GitHub offline is
#      recorded as degraded/corroborating evidence, not counted unhealthy, until
#      local evidence also fails.
#
#   4. CRASH-LOOP-ON-RESTART (OMN-13109): a blanket `docker restart` crash-loops
#      these runners — the entrypoint re-runs config.sh, which reports "already
#      configured", exits, and compose `restart: unless-stopped` immediately
#      restarts it. This shows up as a climbing container RestartCount and/or
#      repeated re-registration markers in `docker logs`. We detect both and
#      alert with the explicit SAFE remediation (NOT `docker restart`).
#
# SAFE remediation recipe (documented; NOT auto-executed unless explicitly
# enabled — see MONITOR_AUTO_BOUNCE below):
#
#   Bounce ONLY the specific wedged/crash-looping service via:
#     docker compose -f <compose> up -d --force-recreate <service-1> <service-N>
#   with a FRESHLY minted registration token, run DETACHED.
#   NEVER `docker restart`           → crash-loops (cached creds + baked token expired)
#   NEVER an empty service filter    → would recreate all 48 at once
#   NEVER block on the bounce        → run in background, mutex-guarded, timeout
#                                       scales with batch size (OMN-13947)

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STATE_FILE="/tmp/runner-monitor-state.json"
COMPOSE_DIR="$HOME/.omnibase/runners/docker"
COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.runners.yml"
RUNNER_FLEET_CONFIG_PATH="${RUNNER_FLEET_CONFIG_PATH:-$HOME/.omnibase/runners/config/runner_fleet.yaml}"

# Silent-wedge thresholds (OMN-13109). A fleet that is online + idle while jobs
# have been queued for longer than WEDGE_QUEUE_AGE_SECONDS is wedged. Default
# 10 minutes — long enough to ignore normal scheduling latency, short enough to
# catch the all-night wedge that starved the merge queue.
WEDGE_QUEUE_AGE_SECONDS="${WEDGE_QUEUE_AGE_SECONDS:-600}"
# Repos whose Actions queues are serviced by this self-hosted fleet. Queued-job
# age is sampled across these. Override via WEDGE_WATCH_REPOS (space-separated
# "owner/name" entries).
WEDGE_WATCH_REPOS="${WEDGE_WATCH_REPOS:-OmniNode-ai/omnibase_infra OmniNode-ai/omnibase_core OmniNode-ai/omniclaude OmniNode-ai/omnimarket}"

# Crash-loop thresholds (OMN-13109). A container whose RestartCount exceeds
# CRASHLOOP_RESTART_THRESHOLD, or whose recent logs show repeated re-registration
# markers, is crash-looping and must NOT be `docker restart`-ed.
CRASHLOOP_RESTART_THRESHOLD="${CRASHLOOP_RESTART_THRESHOLD:-5}"
CRASHLOOP_LOG_TAIL_LINES="${CRASHLOOP_LOG_TAIL_LINES:-200}"
CRASHLOOP_REREGISTER_MARKER_THRESHOLD="${CRASHLOOP_REREGISTER_MARKER_THRESHOLD:-3}"

# GitHub can report a runner offline while Docker still has a healthy listener.
# Recreating immediately can cancel an assigned job, so only auto-remediate this
# class after it persists for a grace period and local logs do not show an
# in-flight job.
OFFLINE_IDLE_RECREATE_AGE_SECONDS="${OFFLINE_IDLE_RECREATE_AGE_SECONDS:-900}"
OFFLINE_IDLE_LOG_TAIL_LINES="${OFFLINE_IDLE_LOG_TAIL_LINES:-240}"

# Auto-bounce is OFF by default: this monitor DETECTS and ALERTS. It does NOT
# mutate the live fleet unless an operator deliberately sets MONITOR_AUTO_BOUNCE=1
# in the monitor env. Even then it force-recreates only the named services with
# a fresh token, detached — never `docker restart`, never an empty filter.
# Steady state is observe-and-alert.
MONITOR_AUTO_BOUNCE="${MONITOR_AUTO_BOUNCE:-0}"

# OMN-13947: a single fixed timeout SIGTERMed `docker compose up --force-recreate`
# mid-batch under host load, leaving containers half-recreated (Status=created
# but never started, or the stale old container never actually replaced) —
# 36% of recreate attempts never completed during the 2026-07-04 incident. The
# timeout now scales with the number of containers in the batch: floor for a
# single-container bounce, a per-container budget for larger batches (a
# fleet-wide silent-wedge bounce targets all EXPECTED_RUNNERS at once), capped
# at a generous ceiling so a genuinely hung daemon still gets killed eventually.
AUTO_BOUNCE_HARD_LIMIT_SECONDS="${AUTO_BOUNCE_HARD_LIMIT_SECONDS:-120}"          # floor
AUTO_BOUNCE_PER_CONTAINER_BUDGET_SECONDS="${AUTO_BOUNCE_PER_CONTAINER_BUDGET_SECONDS:-30}"
AUTO_BOUNCE_TIMEOUT_CEILING_SECONDS="${AUTO_BOUNCE_TIMEOUT_CEILING_SECONDS:-1800}"
# Bounded retries after the compose call returns, verifying every target is
# actually Status=running (not left at Status=created) before giving up.
AUTO_BOUNCE_VERIFY_RETRY_COUNT="${AUTO_BOUNCE_VERIFY_RETRY_COUNT:-3}"
AUTO_BOUNCE_VERIFY_RETRY_SLEEP_SECONDS="${AUTO_BOUNCE_VERIFY_RETRY_SLEEP_SECONDS:-3}"
# flock mutex so the */10 cron cannot dispatch a second bounce while a prior one
# is still in flight — the race produced literal daemon errors ("removal of
# container ... is already in progress") during the incident.
AUTO_BOUNCE_LOCKFILE="${AUTO_BOUNCE_LOCKFILE:-/tmp/runner-monitor-bounce.lock}"

config_field() {
    local field="${1}"
    [[ -f "${RUNNER_FLEET_CONFIG_PATH}" ]] || {
        echo "[runner-monitor] ERROR: runner fleet config not found: ${RUNNER_FLEET_CONFIG_PATH}" >&2
        exit 1
    }
    local value
    value=$(awk -F':[[:space:]]*' -v key="${field}" '
        $1 == key {
            gsub(/^[[:space:]"]+|[[:space:]"]+$/, "", $2)
            print $2
            found=1
        }
        END { if (!found) exit 1 }
    ' "${RUNNER_FLEET_CONFIG_PATH}") || {
        echo "[runner-monitor] ERROR: missing ${field} in ${RUNNER_FLEET_CONFIG_PATH}" >&2
        exit 1
    }
    echo "${value}"
}

RUNNER_ORG="$(config_field github_org)"
RUNNER_GROUP="$(config_field runner_group)"
RUNNER_NAME_PREFIX="$(config_field runner_name_prefix)"
EXPECTED_RUNNERS="$(config_field expected_count)"
BURST_RUNNERS="$(config_field burst_count 2>/dev/null || echo "${EXPECTED_RUNNERS}")"
RUNNER_HOST="$(config_field runner_host)"

# Slack config — passed via environment (cron sources ~/.omnibase/.env)
: "${SLACK_BOT_TOKEN:?SLACK_BOT_TOKEN must be set}"
: "${SLACK_CHANNEL_ID:?SLACK_CHANNEL_ID must be set}"
: "${RUNNER_GITHUB_TOKEN:?RUNNER_GITHUB_TOKEN must be set}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[runner-monitor] $(date '+%H:%M:%S') $*"; }

slack_post() {
    local text="$1"
    local color="${2:-danger}"  # danger=red, warning=yellow, good=green
    curl -s -X POST https://slack.com/api/chat.postMessage \
        -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg channel "$SLACK_CHANNEL_ID" \
            --arg fallback "$text" \
            --arg color "$color" \
            --arg text "$text" \
            --arg footer "runner-monitor | ${RUNNER_HOST}" \
            '{
                channel: $channel,
                attachments: [{
                    color: $color,
                    text: $text,
                    footer: $footer,
                    ts: (now | floor)
                }]
            }'
        )" > /dev/null 2>&1
}

# github_api_get — fetch a GitHub API path as JSON. Empty string on failure.
# Prefer `gh api` because the deployed host already has working gh auth; fall
# back to RUNNER_GITHUB_TOKEN for environments without gh. Retry transient
# misses so a single GitHub/gh blip does not become a false runner outage.
github_api_get() {
    local path="$1"
    local attempt output
    for attempt in 1 2 3; do
        output=""
        if command -v gh >/dev/null 2>&1; then
            output=$(gh api "${path}" 2>/dev/null || true)
        else
            output=$(curl -fsS \
                -H "Authorization: Bearer ${RUNNER_GITHUB_TOKEN}" \
                -H "Accept: application/vnd.github+json" \
                "https://api.github.com${path}" 2>/dev/null || true)
        fi
        if [[ -n "${output}" ]] && jq -e 'type == "object"' <<< "${output}" >/dev/null 2>&1; then
            printf '%s\n' "${output}"
            return 0
        fi
        sleep "${attempt}"
    done
    return 0
}

# gh_api_runners — fetch the org self-hosted runner list as JSON. Empty string
# on failure (caller treats empty as github_api_failed).
gh_api_runners() {
    github_api_get "/orgs/${RUNNER_ORG}/actions/runners?per_page=100"
}

# oldest_queued_job_age_seconds — across WEDGE_WATCH_REPOS, find the OLDEST
# job whose status is "queued" and that targets a self-hosted label, then return
# its age in seconds. Returns 0 when nothing is queued (no wedge possible).
#
# This is the discriminator the legacy monitor lacked: a runner being "online"
# tells you the listener long-poll is connected, NOT that work is flowing.
# Queued work that ages out while the fleet is idle is the wedge signal.
oldest_queued_job_age_seconds() {
    local now_epoch oldest_epoch="" repo runs_json run_ids jobs_json job_created
    now_epoch=$(date -u +%s)

    for repo in ${WEDGE_WATCH_REPOS}; do
        runs_json=$(github_api_get "/repos/${repo}/actions/runs?status=queued&per_page=20")
        [[ -z "${runs_json}" ]] && continue

        run_ids=$(jq -r '.workflow_runs[]?.id // empty' <<< "${runs_json}" 2>/dev/null || true)
        [[ -z "${run_ids}" ]] && continue

        while IFS= read -r run_id; do
            [[ -z "${run_id}" ]] && continue
            jobs_json=$(github_api_get "/repos/${repo}/actions/runs/${run_id}/jobs?per_page=50")
            [[ -z "${jobs_json}" ]] && continue

            # Only count queued jobs that target a self-hosted label. Hosted
            # jobs (ubuntu-latest) are not serviced by this fleet and must not
            # trigger a wedge alert.
            while IFS= read -r job_created; do
                [[ -z "${job_created}" ]] && continue
                local job_epoch
                # GNU date (Linux .201 host) parses the ISO-8601 created_at.
                job_epoch=$(date -u -d "${job_created}" +%s 2>/dev/null || echo "")
                [[ -z "${job_epoch}" ]] && continue
                if [[ -z "${oldest_epoch}" ]] || [[ "${job_epoch}" -lt "${oldest_epoch}" ]]; then
                    oldest_epoch="${job_epoch}"
                fi
            done < <(jq -r --arg group "${RUNNER_GROUP}" '
                .jobs[]?
                | select(.status == "queued")
                | select(any(.labels[]?; . == "self-hosted" or . == $group))
                | .created_at
            ' <<< "${jobs_json}" 2>/dev/null || true)
        done <<< "${run_ids}"
    done

    if [[ -z "${oldest_epoch}" ]]; then
        echo 0
        return 0
    fi
    echo $(( now_epoch - oldest_epoch ))
}

# container_is_crashlooping — given a container name, return 0 (true) when it is
# crash-looping: RestartCount exceeds threshold OR recent logs show repeated
# re-registration markers (config.sh re-run loop). Echoes a human reason on the
# first matched signal.
container_is_crashlooping() {
    local name="${1}"
    local restart_count marker_count

    restart_count=$(docker inspect --format '{{.RestartCount}}' "${name}" 2>/dev/null || echo 0)
    if [[ "${restart_count}" =~ ^[0-9]+$ ]] && [[ "${restart_count}" -gt "${CRASHLOOP_RESTART_THRESHOLD}" ]]; then
        echo "RestartCount=${restart_count} > ${CRASHLOOP_RESTART_THRESHOLD}"
        return 0
    fi

    # Re-registration markers in the recent log tail. The entrypoint emits these
    # exact strings when it is stuck re-running config.sh / hitting max retries.
    marker_count=$(docker logs --tail "${CRASHLOOP_LOG_TAIL_LINES}" "${name}" 2>&1 \
        | grep -ciE 'already configured|Re-registering|Max retries .* reached|Registration error detected|Runner removed' \
        || true)
    if [[ "${marker_count}" =~ ^[0-9]+$ ]] && [[ "${marker_count}" -ge "${CRASHLOOP_REREGISTER_MARKER_THRESHOLD}" ]]; then
        echo "re-registration markers=${marker_count} >= ${CRASHLOOP_REREGISTER_MARKER_THRESHOLD} in last ${CRASHLOOP_LOG_TAIL_LINES} log lines"
        return 0
    fi

    return 1
}

runner_has_active_job() {
    local name="${1}"
    local logs last_running last_completed

    logs=$(docker logs --tail "${OFFLINE_IDLE_LOG_TAIL_LINES}" "${name}" 2>&1 || true)
    last_running=$(grep -n 'Running job:' <<< "${logs}" | tail -n 1 | cut -d: -f1 || true)
    [[ -z "${last_running}" ]] && return 1

    last_completed=$(grep -nE 'Job .+ completed with result:' <<< "${logs}" | tail -n 1 | cut -d: -f1 || true)
    if [[ -z "${last_completed}" ]] || [[ "${last_running}" -gt "${last_completed}" ]]; then
        return 0
    fi
    return 1
}

runner_has_local_listener_evidence() {
    local name="${1}"
    local logs

    logs=$(docker logs --tail "${OFFLINE_IDLE_LOG_TAIL_LINES}" "${name}" 2>&1 || true)
    if grep -qE 'Listening for Jobs|Runner reconnected|√ Connected to GitHub' <<< "${logs}"; then
        return 0
    fi
    if runner_has_active_job "${name}"; then
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Collect current state
# ---------------------------------------------------------------------------
declare -A current_status
declare -A docker_oom_killed
declare -A github_status
declare -A github_busy

total_found=0
healthy=0
online_count=0
busy_count=0
unhealthy_list=()
wedge_list=()
crashloop_list=()
offline_idle_bounce_list=()
offline_idle_recreate_list=()
github_degraded_list=()
# OMN-13947: containers left at Docker Status=created (docker create succeeded,
# docker start never ran — the signature of a bounce killed mid-batch) matched
# NONE of crashloop/wedge, so collect_remediation_targets() never re-selected
# them: a straggler that survived one bounce attempt was orphaned forever.
stuck_created_list=()
missing_container_list=()
github_api_failed=false
now_epoch=$(date -u +%s)
offline_first_seen_lines=""
prev_offline_first_seen_json="{}"
if [[ -f "${STATE_FILE}" ]]; then
    prev_offline_first_seen_json=$(jq -c '.offline_first_seen // {}' "${STATE_FILE}" 2>/dev/null || echo "{}")
fi

while IFS=$'\t' read -r name status; do
    [[ -z "${name}" ]] && continue
    current_status["$name"]="$status"
done < <(docker ps -a --filter "name=${RUNNER_NAME_PREFIX}" --format "{{.Names}}\t{{.Status}}" 2>/dev/null || true)

for name in "${!current_status[@]}"; do
    oom_killed="$(docker inspect --format '{{.State.OOMKilled}}' "$name" 2>/dev/null || echo "unknown")"
    docker_oom_killed["$name"]="$oom_killed"
done

github_json=$(gh_api_runners)

if [[ -z "${github_json}" ]]; then
    github_api_failed=true
    unhealthy_list+=("GITHUB_API: failed to fetch org runner status")
else
    # Capture both status (online/offline) AND busy (executing a job). The busy
    # field is the silent-wedge discriminator the legacy monitor ignored.
    while IFS=$'\t' read -r name status busy; do
        [[ -z "${name}" ]] && continue
        github_status["$name"]="$status"
        github_busy["$name"]="$busy"
    done < <(jq -r --arg prefix "${RUNNER_NAME_PREFIX}" --arg group "${RUNNER_GROUP}" '
        .runners[]
        | select(.name | startswith($prefix))
        | select(any(.labels[]; .name == $group))
        | [.name, .status, (.busy | tostring)]
        | @tsv
    ' <<< "${github_json}")
fi

# Check configured runners against Docker first. GitHub registration status is
# useful corroborating evidence, but it is not authoritative during API/status
# propagation incidents. A Docker-healthy runner with local listener evidence is
# treated as healthy even if the GitHub org API reports it offline.
for i in $(seq 1 "$EXPECTED_RUNNERS"); do
    name="${RUNNER_NAME_PREFIX}-${i}"
    total_found=$((total_found + 1))
    docker_state="${current_status[$name]:-MISSING (no container)}"

    if [[ "${docker_state}" == "MISSING (no container)" ]]; then
        missing_container_list+=("${name}: MISSING (no container)")
        unhealthy_list+=("${name}: MISSING (no container)")
        continue
    fi

    if [[ "${docker_oom_killed[$name]:-false}" == "true" ]]; then
        unhealthy_list+=("${name}: Docker OOMKilled=true while ${docker_state}")
        continue
    fi

    # OMN-13947: a container stuck at "Created" (never started) is the direct
    # fingerprint of a force-recreate batch that got SIGTERM'd mid-flight.
    # Flag it as a remediation target in its own right — it will never match
    # crashloop (no RestartCount, no logs) or wedge (fleet-wide only).
    if [[ "${docker_state}" == Created* ]]; then
        stuck_created_list+=("${name}: Docker Created but never started — orphaned mid-recreate")
    fi

    if [[ "${docker_state}" == Up* ]] && [[ "${docker_state}" != *"(healthy)"* ]] && runner_has_local_listener_evidence "${name}"; then
        github_degraded_list+=("${name}: Docker health ${docker_state} ignored; local listener evidence present")
        healthy=$((healthy + 1))
        continue
    fi

    if [[ "${docker_state}" != *"(healthy)"* ]] || [[ "${docker_state}" != Up* ]]; then
        unhealthy_list+=("${name}: Docker ${docker_state}")
        continue
    fi

    # Crash-loop check runs even on "healthy" containers: a container can be
    # "Up (healthy)" in the brief window between restart and the next config.sh
    # exit. RestartCount and the log markers expose the loop the status string
    # hides. Do NOT `docker restart` these.
    if crashloop_reason=$(container_is_crashlooping "${name}"); then
        crashloop_list+=("${name}: CRASH-LOOP (${crashloop_reason})")
        unhealthy_list+=("${name}: CRASH-LOOP (${crashloop_reason})")
        continue
    fi

    if [[ "${github_api_failed}" == true ]]; then
        healthy=$((healthy + 1))
        continue
    fi

    gh_state="${github_status[$name]:-missing}"
    if [[ "${gh_state}" != "online" ]]; then
        if runner_has_local_listener_evidence "${name}"; then
            github_degraded_list+=("${name}: GitHub ${gh_state} ignored; Docker ${docker_state} and local listener evidence present")
            healthy=$((healthy + 1))
            continue
        fi
        if [[ "${github_busy[$name]:-false}" != "true" ]]; then
            first_seen=$(jq -r --arg name "${name}" '.[$name] // empty' <<< "${prev_offline_first_seen_json}" 2>/dev/null || true)
            if [[ ! "${first_seen}" =~ ^[0-9]+$ ]]; then
                first_seen="${now_epoch}"
            fi
            offline_first_seen_lines+="${name}"$'\t'"${first_seen}"$'\n'
            offline_age=$(( now_epoch - first_seen ))
            offline_idle_bounce_list+=("${name}: OFFLINE-IDLE (${gh_state}, age=${offline_age}s, Docker ${docker_state})")
            if [[ "${offline_age}" -ge "${OFFLINE_IDLE_RECREATE_AGE_SECONDS}" ]]; then
                if runner_has_active_job "${name}"; then
                    offline_idle_bounce_list+=("${name}: OFFLINE-IDLE not auto-recreated; local logs show active job")
                else
                    offline_idle_recreate_list+=("${name}: OFFLINE-IDLE-RECREATE age=${offline_age}s >= ${OFFLINE_IDLE_RECREATE_AGE_SECONDS}s")
                fi
            fi
        fi
        unhealthy_list+=("${name}: GitHub ${gh_state} while Docker ${docker_state}")
        continue
    fi

    online_count=$((online_count + 1))
    if [[ "${github_busy[$name]:-false}" == "true" ]]; then
        busy_count=$((busy_count + 1))
    fi
    healthy=$((healthy + 1))
done

# ---------------------------------------------------------------------------
# Silent-wedge detection (OMN-13109)
# ---------------------------------------------------------------------------
# A wedged fleet PASSES every check above: containers Up (healthy), runners
# online. The tell is jobs queued for our labels aging out while NO runner is
# busy. Only evaluate when the GitHub API succeeded (we need the busy field) and
# at least one runner is online (otherwise the offline path already alerted).
queued_age=0
if [[ "${github_api_failed}" != true ]] && [[ "${online_count}" -gt 0 ]]; then
    queued_age=$(oldest_queued_job_age_seconds)
    if [[ "${queued_age}" -ge "${WEDGE_QUEUE_AGE_SECONDS}" ]] && [[ "${busy_count}" -eq 0 ]]; then
        wedge_list+=("SILENT-WEDGE: ${online_count} runners online + ${busy_count} busy, but a self-hosted job has been queued ${queued_age}s (>= ${WEDGE_QUEUE_AGE_SECONDS}s). Fleet is registered but not pulling jobs.")
        unhealthy_list+=("SILENT-WEDGE: online=${online_count} busy=0, queued-job-age=${queued_age}s")
    fi
fi

# Also check Docker socket accessibility from a healthy runner
docker_ok=true
if [[ $healthy -gt 0 ]]; then
    # Pick the first healthy runner to test Docker access
    for name in "${!current_status[@]}"; do
        status="${current_status[$name]}"
        if [[ "$status" == *"(healthy)"* ]]; then
            if ! docker exec "$name" docker info --format "{{.ServerVersion}}" > /dev/null 2>&1; then
                docker_ok=false
                unhealthy_list+=("DOCKER_SOCKET: permission denied from ${name}")
            fi
            break
        fi
    done
fi

for name in "${!current_status[@]}"; do
    if [[ ! "${name}" =~ ^${RUNNER_NAME_PREFIX}-[0-9]+$ ]]; then
        continue
    fi
    index="${name##*-}"
    if [[ "${index}" -gt "${BURST_RUNNERS}" ]]; then
        unhealthy_list+=("${name}: EXTRA Docker container beyond configured burst count ${BURST_RUNNERS}")
    fi
done

if [[ "${github_api_failed}" != true ]]; then
    for name in "${!github_status[@]}"; do
        if [[ ! "${name}" =~ ^${RUNNER_NAME_PREFIX}-[0-9]+$ ]]; then
            continue
        fi
        index="${name##*-}"
        if [[ "${index}" -gt "${BURST_RUNNERS}" ]]; then
            unhealthy_list+=("${name}: EXTRA GitHub registration beyond configured burst count ${BURST_RUNNERS}")
        fi
    done
fi

# ---------------------------------------------------------------------------
# SAFE remediation recipe (OMN-13109)
# ---------------------------------------------------------------------------
# Render the exact, copy-pasteable safe-bounce command for the affected
# services. By default this is ONLY rendered into the Slack alert (operator runs
# it). When MONITOR_AUTO_BOUNCE=1 it is executed — force-recreate of the named
# services only, fresh token, detached, 2-minute hard limit.
#
# remediation_targets: space-separated service names extracted from the
# wedge/crash-loop findings. Empty when the issue is something else (offline,
# OOM, socket) where automatic bounce is unsafe.
collect_remediation_targets() {
    local targets=()
    local entry svc
    if [[ "${#missing_container_list[@]}" -gt 0 ]]; then
        for entry in "${missing_container_list[@]}"; do
            svc="${entry%%:*}"
            [[ "${svc}" =~ ^${RUNNER_NAME_PREFIX}-[0-9]+$ ]] && targets+=("${svc}")
        done
    fi
    if [[ "${#offline_idle_recreate_list[@]}" -gt 0 ]]; then
        for entry in "${offline_idle_recreate_list[@]}"; do
            svc="${entry%%:*}"
            [[ "${svc}" =~ ^${RUNNER_NAME_PREFIX}-[0-9]+$ ]] && targets+=("${svc}")
        done
    fi
    if [[ "${#crashloop_list[@]}" -gt 0 ]]; then
        for entry in "${crashloop_list[@]}"; do
            svc="${entry%%:*}"
            [[ "${svc}" =~ ^${RUNNER_NAME_PREFIX}-[0-9]+$ ]] && targets+=("${svc}")
        done
    fi
    # OMN-13947: re-target stragglers left at Status=created by a prior bounce
    # that never completed. Without this, a container that survives one
    # auto_bounce verify-retry cycle is permanently invisible to remediation.
    if [[ "${#stuck_created_list[@]}" -gt 0 ]]; then
        for entry in "${stuck_created_list[@]}"; do
            svc="${entry%%:*}"
            [[ "${svc}" =~ ^${RUNNER_NAME_PREFIX}-[0-9]+$ ]] && targets+=("${svc}")
        done
    fi
    # A silent wedge is fleet-wide: the safe recipe bounces the whole configured
    # fleet, but STILL one explicit service list (never an empty filter).
    if [[ "${#wedge_list[@]}" -gt 0 ]]; then
        local j
        for j in $(seq 1 "${EXPECTED_RUNNERS}"); do
            targets+=("${RUNNER_NAME_PREFIX}-${j}")
        done
    fi
    # De-dupe while preserving order. Guard against empty array under set -u.
    if [[ "${#targets[@]}" -gt 0 ]]; then
        printf '%s\n' "${targets[@]}" | awk '!seen[$0]++' | tr '\n' ' '
    fi
}

# bounce_timeout_seconds — timeout for a force-recreate batch, scaled by the
# number of targets (OMN-13947). A fixed cap SIGTERMs `docker compose up`
# mid-batch under host load; scaling with batch size and keeping a generous
# ceiling means a multi-container recreate is never killed before it can
# finish, while a genuinely hung daemon still gets bounded eventually.
bounce_timeout_seconds() {
    local target_list="$1"
    local target_count timeout_seconds
    target_count=$(wc -w <<< "${target_list}")
    timeout_seconds=$(( target_count * AUTO_BOUNCE_PER_CONTAINER_BUDGET_SECONDS ))
    [[ "${timeout_seconds}" -lt "${AUTO_BOUNCE_HARD_LIMIT_SECONDS}" ]] && timeout_seconds="${AUTO_BOUNCE_HARD_LIMIT_SECONDS}"
    [[ "${timeout_seconds}" -gt "${AUTO_BOUNCE_TIMEOUT_CEILING_SECONDS}" ]] && timeout_seconds="${AUTO_BOUNCE_TIMEOUT_CEILING_SECONDS}"
    echo "${timeout_seconds}"
}

render_safe_bounce_cmd() {
    local target_list="$1"
    [[ -z "${target_list// /}" ]] && { echo ""; return 0; }
    local timeout_seconds
    timeout_seconds=$(bounce_timeout_seconds "${target_list}")
    cat <<RECIPE
# SAFE BOUNCE — force-recreate ONLY these services with a FRESH token, detached.
# NEVER 'docker restart' (crash-loops: cached creds + expired baked token).
# NEVER an empty service filter (would recreate all 48 at once).
# NEVER run this manually while an auto_bounce is in flight — check for a lock
# on ${AUTO_BOUNCE_LOCKFILE} first (concurrent recreates race the daemon).
TOKEN=\$(gh api --method POST /orgs/${RUNNER_ORG}/actions/runners/registration-token --jq .token)
RUNNER_TOKEN="\$TOKEN" timeout ${timeout_seconds} \\
  docker compose -f ${COMPOSE_FILE} up -d --force-recreate --no-deps ${target_list}
RECIPE
}

# auto_bounce — execute the safe recipe IFF MONITOR_AUTO_BOUNCE=1. Detached,
# fresh token, named services only. This is the documented remediation; it is
# gated OFF by default so this script stays observe-and-alert.
#
# OMN-13947 hardening over the original implementation:
#   1. flock mutex — the */10 cron fired unconditionally even while a prior
#      bounce was still running in the background, producing literal daemon
#      races ("removal of container ... is already in progress"). A
#      non-blocking flock probe skips this cycle instead of racing.
#   2. Batch-scaled timeout — see bounce_timeout_seconds().
#   3. Verify-start + bounded retry — `docker compose up` can return having
#      `docker create`d some targets without starting them (killed mid-batch).
#      Assert Status=running for every target after the call returns; explicit
#      `docker start` + retry for any straggler instead of leaving it orphaned.
auto_bounce() {
    local target_list="$1"
    [[ "${MONITOR_AUTO_BOUNCE}" != "1" ]] && return 0
    [[ -z "${target_list// /}" ]] && return 0

    # Non-blocking lock probe. If a prior bounce still holds the lock, skip
    # this cycle rather than dispatching a second concurrent recreate. The
    # deployed Linux host has flock; the mkdir fallback keeps local macOS tests
    # exercising the same single-flight behavior.
    local lock_kind="flock"
    if command -v flock >/dev/null 2>&1; then
        exec 9>"${AUTO_BOUNCE_LOCKFILE}"
        if ! flock -n 9; then
            log "AUTO-BOUNCE skipped: a prior bounce is still in flight (lock held on ${AUTO_BOUNCE_LOCKFILE})."
            exec 9>&-
            return 0
        fi
    else
        lock_kind="mkdir"
        if ! mkdir "${AUTO_BOUNCE_LOCKFILE}.d" 2>/dev/null; then
            log "AUTO-BOUNCE skipped: a prior bounce is still in flight (lock held on ${AUTO_BOUNCE_LOCKFILE}.d)."
            return 0
        fi
    fi

    log "AUTO-BOUNCE enabled (MONITOR_AUTO_BOUNCE=1). Force-recreating: ${target_list}"
    local token
    token=$(gh api --method POST "/orgs/${RUNNER_ORG}/actions/runners/registration-token" --jq .token 2>/dev/null || true)
    if [[ -z "${token}" ]]; then
        log "AUTO-BOUNCE aborted: could not mint a fresh registration token."
        slack_post "*[RUNNER AUTO-BOUNCE ABORTED]* could not mint registration token. Manual bounce required for: ${target_list}" "danger"
        if [[ "${lock_kind}" == "flock" ]]; then
            exec 9>&-
        else
            rmdir "${AUTO_BOUNCE_LOCKFILE}.d" 2>/dev/null || true
        fi
        return 0
    fi

    local timeout_seconds
    timeout_seconds=$(bounce_timeout_seconds "${target_list}")

    # The subshell inherits fd 9 (and therefore the flock) from this process.
    # We close our own copy right after forking so the lock is held for
    # exactly as long as the background subshell (compose + verify/retry) is
    # running, then released automatically when it exits.
    # shellcheck disable=SC2086
    (
        if [[ "${lock_kind}" == "mkdir" ]]; then
            trap 'rmdir "${AUTO_BOUNCE_LOCKFILE}.d" 2>/dev/null || true' EXIT
        fi
        RUNNER_TOKEN="${token}" timeout "${timeout_seconds}" \
            docker compose -f "${COMPOSE_FILE}" up -d --force-recreate --no-deps ${target_list} \
            >> /tmp/runner-monitor-bounce.log 2>&1

        svc=""
        for svc in ${target_list}; do
            attempt=1
            status="$(docker inspect --format '{{.State.Status}}' "${svc}" 2>/dev/null || echo missing)"
            while [[ "${status}" != "running" ]] && [[ "${attempt}" -le "${AUTO_BOUNCE_VERIFY_RETRY_COUNT}" ]]; do
                echo "[runner-monitor] $(date '+%H:%M:%S') AUTO-BOUNCE straggler: ${svc} Status=${status} (attempt ${attempt}/${AUTO_BOUNCE_VERIFY_RETRY_COUNT}) — explicit docker start." \
                    >> /tmp/runner-monitor-bounce.log
                docker start "${svc}" >> /tmp/runner-monitor-bounce.log 2>&1 || true
                sleep "${AUTO_BOUNCE_VERIFY_RETRY_SLEEP_SECONDS}"
                status="$(docker inspect --format '{{.State.Status}}' "${svc}" 2>/dev/null || echo missing)"
                attempt=$(( attempt + 1 ))
            done
            if [[ "${status}" != "running" ]]; then
                echo "[runner-monitor] $(date '+%H:%M:%S') AUTO-BOUNCE FAILED to bring ${svc} to running (final Status=${status}) after ${AUTO_BOUNCE_VERIFY_RETRY_COUNT} retries — left for next cycle (stuck_created detection will re-target it)." \
                    >> /tmp/runner-monitor-bounce.log
            fi
        done
    ) &
    if [[ "${lock_kind}" == "flock" ]]; then
        exec 9>&-
    fi
    log "AUTO-BOUNCE dispatched in background (timeout ${timeout_seconds}s for $(wc -w <<< "${target_list}") target(s), verify+retry up to ${AUTO_BOUNCE_VERIFY_RETRY_COUNT}x)."
}

# ---------------------------------------------------------------------------
# Compare with previous state and alert on transitions
# ---------------------------------------------------------------------------
prev_unhealthy_count=0
prev_alert_count=0
if [[ -f "$STATE_FILE" ]]; then
    prev_unhealthy_count=$(jq -r '.unhealthy_count // 0' "$STATE_FILE" 2>/dev/null || echo 0)
    prev_alert_count=$(jq -r '.alert_count // 0' "$STATE_FILE" 2>/dev/null || echo 0)
fi

current_unhealthy_count=${#unhealthy_list[@]}
wedge_count=${#wedge_list[@]}
crashloop_count=${#crashloop_list[@]}
stuck_created_count=${#stuck_created_list[@]}
offline_idle_recreate_count=${#offline_idle_recreate_list[@]}
missing_container_count=${#missing_container_list[@]}
offline_first_seen_json="{}"
if [[ -n "${offline_first_seen_lines}" ]]; then
    offline_first_seen_json=$(printf '%s' "${offline_first_seen_lines}" | jq -Rn '
        reduce inputs as $line ({};
            if ($line | length) == 0 then
                .
            else
                ($line | split("\t")) as $parts
                | . + {($parts[0]): ($parts[1] | tonumber)}
            end
        )
    ')
fi

remediation_targets="$(collect_remediation_targets)"
remediation_target_count=0
if [[ -n "${remediation_targets// /}" ]]; then
    remediation_target_count=$(wc -w <<< "${remediation_targets}")
fi
current_alert_count="${remediation_target_count}"
if [[ "${github_api_failed}" == true ]]; then
    current_alert_count=$((current_alert_count + 1))
fi
if [[ "${docker_ok}" != true ]]; then
    current_alert_count=$((current_alert_count + 1))
fi

# Write current state
jq -n \
    --argjson healthy "$healthy" \
    --argjson unhealthy_count "$current_unhealthy_count" \
    --argjson alert_count "$current_alert_count" \
    --argjson remediation_target_count "$remediation_target_count" \
    --argjson wedge_count "$wedge_count" \
    --argjson crashloop_count "$crashloop_count" \
    --argjson stuck_created_count "$stuck_created_count" \
    --argjson offline_idle_recreate_count "$offline_idle_recreate_count" \
    --argjson missing_container_count "$missing_container_count" \
    --argjson online "$online_count" \
    --argjson busy "$busy_count" \
    --argjson queued_age "$queued_age" \
    --argjson total "$total_found" \
    --argjson docker_ok "$docker_ok" \
    --arg timestamp "$(date -Iseconds)" \
    --arg unhealthy_names "$(printf '%s\n' "${unhealthy_list[@]}" 2>/dev/null || echo '')" \
    --arg github_degraded_names "$(printf '%s\n' "${github_degraded_list[@]}" 2>/dev/null || echo '')" \
    --arg offline_idle_bounce_names "$(printf '%s\n' "${offline_idle_bounce_list[@]}" 2>/dev/null || echo '')" \
    --arg offline_idle_recreate_names "$(printf '%s\n' "${offline_idle_recreate_list[@]}" 2>/dev/null || echo '')" \
    --argjson offline_first_seen "${offline_first_seen_json}" \
    '{
        healthy: $healthy,
        unhealthy_count: $unhealthy_count,
        alert_count: $alert_count,
        remediation_target_count: $remediation_target_count,
        wedge_count: $wedge_count,
        crashloop_count: $crashloop_count,
        stuck_created_count: $stuck_created_count,
        offline_idle_recreate_count: $offline_idle_recreate_count,
        missing_container_count: $missing_container_count,
        online: $online,
        busy: $busy,
        oldest_queued_job_age_seconds: $queued_age,
        total: $total,
        docker_ok: $docker_ok,
        timestamp: $timestamp,
        unhealthy_names: $unhealthy_names,
        github_degraded_names: $github_degraded_names,
        offline_idle_bounce_names: $offline_idle_bounce_names,
        offline_idle_recreate_names: $offline_idle_recreate_names,
        offline_first_seen: $offline_first_seen
    }' > "$STATE_FILE"

# ---------------------------------------------------------------------------
# Alert logic — only on state transitions
# ---------------------------------------------------------------------------

safe_bounce_block=""
if [[ -n "${remediation_targets// /}" ]]; then
    safe_bounce_block="$(render_safe_bounce_cmd "${remediation_targets}")"
fi

# Build the special-finding banner (wedge / crash-loop) appended to alerts.
special_findings=""
if [[ "${wedge_count}" -gt 0 ]]; then
    special_findings+="$(printf '%s\n' "${wedge_list[@]}")"$'\n'
fi
if [[ "${#offline_idle_bounce_list[@]}" -gt 0 ]]; then
    special_findings+="$(printf '%s\n' "${offline_idle_bounce_list[@]}")"$'\n'
fi
if [[ "${#offline_idle_recreate_list[@]}" -gt 0 ]]; then
    special_findings+="$(printf '%s\n' "${offline_idle_recreate_list[@]}")"$'\n'
fi
if [[ "${crashloop_count}" -gt 0 ]]; then
    special_findings+="$(printf '%s\n' "${crashloop_list[@]}")"$'\n'
fi
if [[ "${stuck_created_count}" -gt 0 ]]; then
    special_findings+="$(printf '%s\n' "${stuck_created_list[@]}")"$'\n'
fi

if [[ $current_alert_count -gt 0 ]] && [[ $prev_alert_count -eq 0 ]]; then
    # Transition: no actionable alert -> actionable repair/monitor failure.
    detail=$(printf '%s\n' "${unhealthy_list[@]}")
    msg="*[RUNNER ALERT]* ${current_alert_count} actionable runner-fleet issue(s); ${current_unhealthy_count}/${EXPECTED_RUNNERS} raw unhealthy

\`\`\`
${detail}
\`\`\`

Healthy: ${healthy}/${EXPECTED_RUNNERS} | Online: ${online_count} | Busy: ${busy_count}
Docker socket: $([ "$docker_ok" = true ] && echo 'OK' || echo 'FAILED')
Host: ${RUNNER_HOST}"
    if [[ -n "${special_findings}" ]]; then
        msg+="

*Detected failure modes (OMN-13109):*
\`\`\`
${special_findings}\`\`\`"
    fi
    if [[ -n "${safe_bounce_block}" ]]; then
        msg+="

*Safe remediation (force-recreate named services only — NEVER docker restart):*
\`\`\`
${safe_bounce_block}
\`\`\`"
    fi
    slack_post "${msg}" "danger"
    log "ALERT: ${current_alert_count} actionable issue(s), ${current_unhealthy_count} raw unhealthy (previous actionable 0). wedge=${wedge_count} crashloop=${crashloop_count} stuck_created=${stuck_created_count} offline_idle_recreate=${offline_idle_recreate_count}. Slack notified."
    auto_bounce "${remediation_targets}"

elif [[ $current_alert_count -gt 0 ]] && [[ $prev_alert_count -gt 0 ]] && [[ $current_alert_count -ne $prev_alert_count ]]; then
    # Transition: actionable issue count changed. Raw GitHub/Docker drift count
    # can churn under saturation; do not page on that noise.
    detail=$(printf '%s\n' "${unhealthy_list[@]}")
    msg="*[RUNNER UPDATE]* ${current_alert_count} actionable runner-fleet issue(s) (was ${prev_alert_count}); ${current_unhealthy_count}/${EXPECTED_RUNNERS} raw unhealthy

\`\`\`
${detail}
\`\`\`

Healthy: ${healthy}/${EXPECTED_RUNNERS} | Online: ${online_count} | Busy: ${busy_count}"
    if [[ -n "${special_findings}" ]]; then
        msg+="

*Detected failure modes (OMN-13109):*
\`\`\`
${special_findings}\`\`\`"
    fi
    if [[ -n "${safe_bounce_block}" ]]; then
        msg+="

*Safe remediation (force-recreate named services only — NEVER docker restart):*
\`\`\`
${safe_bounce_block}
\`\`\`"
    fi
    slack_post "${msg}" "warning"
    log "UPDATE: ${current_alert_count} actionable issue(s) (was ${prev_alert_count}), ${current_unhealthy_count} raw unhealthy. wedge=${wedge_count} crashloop=${crashloop_count} stuck_created=${stuck_created_count} offline_idle_recreate=${offline_idle_recreate_count}. Slack notified."
    auto_bounce "${remediation_targets}"

elif [[ $current_alert_count -eq 0 ]] && [[ $prev_alert_count -gt 0 ]]; then
    # Transition: actionable alert cleared. Raw drift may remain and is logged.
    slack_post "*[RUNNER RECOVERED]* No actionable runner-fleet issues remain

Healthy: ${healthy}/${EXPECTED_RUNNERS} | Raw unhealthy: ${current_unhealthy_count}
Online: ${online_count} | Busy: ${busy_count}
Docker socket: $([ "$docker_ok" = true ] && echo 'OK' || echo 'FAILED')
Host: ${RUNNER_HOST}" "good"
    log "RECOVERED: actionable alert count cleared; ${current_unhealthy_count} raw unhealthy remain. Slack notified."

else
    # No state change — silent
    log "OK: ${healthy}/${EXPECTED_RUNNERS} healthy, ${current_unhealthy_count} raw unhealthy, ${current_alert_count} actionable (wedge=${wedge_count} crashloop=${crashloop_count} stuck_created=${stuck_created_count} offline_idle_recreate=${offline_idle_recreate_count}, no Slack change)."
    auto_bounce "${remediation_targets}"
fi
