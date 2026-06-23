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
#   3. CRASH-LOOP-ON-RESTART (OMN-13109): a blanket `docker restart` crash-loops
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
#   NEVER block on the bounce        → 2-minute hard limit, run in background

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

# Auto-bounce is OFF by default: this monitor DETECTS and ALERTS. It does NOT
# mutate the live fleet unless an operator deliberately sets MONITOR_AUTO_BOUNCE=1
# in the monitor env. Even then it force-recreates only the named services with
# a fresh token, detached, with a 2-minute hard limit — never `docker restart`,
# never an empty filter. Steady state is observe-and-alert.
MONITOR_AUTO_BOUNCE="${MONITOR_AUTO_BOUNCE:-0}"
AUTO_BOUNCE_HARD_LIMIT_SECONDS="${AUTO_BOUNCE_HARD_LIMIT_SECONDS:-120}"

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

# gh_api_runners — fetch the org self-hosted runner list as JSON. Empty string
# on failure (caller treats empty as github_api_failed).
gh_api_runners() {
    curl -fsS \
        -H "Authorization: Bearer ${RUNNER_GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/orgs/${RUNNER_ORG}/actions/runners?per_page=100" 2>/dev/null || true
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
        runs_json=$(curl -fsS \
            -H "Authorization: Bearer ${RUNNER_GITHUB_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            "https://api.github.com/repos/${repo}/actions/runs?status=queued&per_page=20" 2>/dev/null || true)
        [[ -z "${runs_json}" ]] && continue

        run_ids=$(jq -r '.workflow_runs[]?.id // empty' <<< "${runs_json}" 2>/dev/null || true)
        [[ -z "${run_ids}" ]] && continue

        while IFS= read -r run_id; do
            [[ -z "${run_id}" ]] && continue
            jobs_json=$(curl -fsS \
                -H "Authorization: Bearer ${RUNNER_GITHUB_TOKEN}" \
                -H "Accept: application/vnd.github+json" \
                "https://api.github.com/repos/${repo}/actions/runs/${run_id}/jobs?per_page=50" 2>/dev/null || true)
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
github_api_failed=false

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

# Check configured runners against both Docker and GitHub. Docker healthy while
# GitHub says offline is degraded; that is exactly the state container-only
# health checks miss.
for i in $(seq 1 "$EXPECTED_RUNNERS"); do
    name="${RUNNER_NAME_PREFIX}-${i}"
    total_found=$((total_found + 1))
    docker_state="${current_status[$name]:-MISSING (no container)}"

    if [[ "${docker_state}" == "MISSING (no container)" ]]; then
        unhealthy_list+=("${name}: MISSING (no container)")
        continue
    fi

    if [[ "${docker_oom_killed[$name]:-false}" == "true" ]]; then
        unhealthy_list+=("${name}: Docker OOMKilled=true while ${docker_state}")
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
# OOM, socket) where the bounce recipe does not apply.
collect_remediation_targets() {
    local targets=()
    local entry svc
    if [[ "${#crashloop_list[@]}" -gt 0 ]]; then
        for entry in "${crashloop_list[@]}"; do
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

render_safe_bounce_cmd() {
    local target_list="$1"
    [[ -z "${target_list// /}" ]] && { echo ""; return 0; }
    cat <<RECIPE
# SAFE BOUNCE — force-recreate ONLY these services with a FRESH token, detached.
# NEVER 'docker restart' (crash-loops: cached creds + expired baked token).
# NEVER an empty service filter (would recreate all 48 at once).
TOKEN=\$(gh api --method POST /orgs/${RUNNER_ORG}/actions/runners/registration-token --jq .token)
RUNNER_TOKEN="\$TOKEN" timeout ${AUTO_BOUNCE_HARD_LIMIT_SECONDS} \\
  docker compose -f ${COMPOSE_FILE} up -d --force-recreate --no-deps ${target_list}
RECIPE
}

# auto_bounce — execute the safe recipe IFF MONITOR_AUTO_BOUNCE=1. Detached,
# fresh token, named services only, 2-minute hard limit. This is the documented
# remediation; it is gated OFF by default so this script stays observe-and-alert.
auto_bounce() {
    local target_list="$1"
    [[ "${MONITOR_AUTO_BOUNCE}" != "1" ]] && return 0
    [[ -z "${target_list// /}" ]] && return 0

    log "AUTO-BOUNCE enabled (MONITOR_AUTO_BOUNCE=1). Force-recreating: ${target_list}"
    local token
    token=$(gh api --method POST "/orgs/${RUNNER_ORG}/actions/runners/registration-token" --jq .token 2>/dev/null || true)
    if [[ -z "${token}" ]]; then
        log "AUTO-BOUNCE aborted: could not mint a fresh registration token."
        slack_post "*[RUNNER AUTO-BOUNCE ABORTED]* could not mint registration token. Manual bounce required for: ${target_list}" "danger"
        return 0
    fi
    # Detached, named services only, hard 2-minute limit, never blocking.
    # shellcheck disable=SC2086
    RUNNER_TOKEN="${token}" timeout "${AUTO_BOUNCE_HARD_LIMIT_SECONDS}" \
        docker compose -f "${COMPOSE_FILE}" up -d --force-recreate --no-deps ${target_list} \
        >> /tmp/runner-monitor-bounce.log 2>&1 &
    log "AUTO-BOUNCE dispatched in background (limit ${AUTO_BOUNCE_HARD_LIMIT_SECONDS}s)."
}

# ---------------------------------------------------------------------------
# Compare with previous state and alert on transitions
# ---------------------------------------------------------------------------
prev_unhealthy_count=0
if [[ -f "$STATE_FILE" ]]; then
    prev_unhealthy_count=$(jq -r '.unhealthy_count // 0' "$STATE_FILE" 2>/dev/null || echo 0)
fi

current_unhealthy_count=${#unhealthy_list[@]}
wedge_count=${#wedge_list[@]}
crashloop_count=${#crashloop_list[@]}

# Write current state
jq -n \
    --argjson healthy "$healthy" \
    --argjson unhealthy_count "$current_unhealthy_count" \
    --argjson wedge_count "$wedge_count" \
    --argjson crashloop_count "$crashloop_count" \
    --argjson online "$online_count" \
    --argjson busy "$busy_count" \
    --argjson queued_age "$queued_age" \
    --argjson total "$total_found" \
    --argjson docker_ok "$docker_ok" \
    --arg timestamp "$(date -Iseconds)" \
    --arg unhealthy_names "$(printf '%s\n' "${unhealthy_list[@]}" 2>/dev/null || echo '')" \
    '{
        healthy: $healthy,
        unhealthy_count: $unhealthy_count,
        wedge_count: $wedge_count,
        crashloop_count: $crashloop_count,
        online: $online,
        busy: $busy,
        oldest_queued_job_age_seconds: $queued_age,
        total: $total,
        docker_ok: $docker_ok,
        timestamp: $timestamp,
        unhealthy_names: $unhealthy_names
    }' > "$STATE_FILE"

# ---------------------------------------------------------------------------
# Alert logic — only on state transitions
# ---------------------------------------------------------------------------

remediation_targets="$(collect_remediation_targets)"
safe_bounce_block=""
if [[ -n "${remediation_targets// /}" ]]; then
    safe_bounce_block="$(render_safe_bounce_cmd "${remediation_targets}")"
fi

# Build the special-finding banner (wedge / crash-loop) appended to alerts.
special_findings=""
if [[ "${wedge_count}" -gt 0 ]]; then
    special_findings+="$(printf '%s\n' "${wedge_list[@]}")"$'\n'
fi
if [[ "${crashloop_count}" -gt 0 ]]; then
    special_findings+="$(printf '%s\n' "${crashloop_list[@]}")"$'\n'
fi

if [[ $current_unhealthy_count -gt 0 ]] && [[ $prev_unhealthy_count -eq 0 ]]; then
    # Transition: all-good → something broken
    detail=$(printf '%s\n' "${unhealthy_list[@]}")
    msg="*[RUNNER ALERT]* ${current_unhealthy_count}/${EXPECTED_RUNNERS} runners unhealthy

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
    log "ALERT: ${current_unhealthy_count} runners unhealthy (was 0). wedge=${wedge_count} crashloop=${crashloop_count}. Slack notified."
    auto_bounce "${remediation_targets}"

elif [[ $current_unhealthy_count -gt 0 ]] && [[ $prev_unhealthy_count -gt 0 ]] && [[ $current_unhealthy_count -ne $prev_unhealthy_count ]]; then
    # Transition: bad → worse or bad → partially recovered
    detail=$(printf '%s\n' "${unhealthy_list[@]}")
    msg="*[RUNNER UPDATE]* ${current_unhealthy_count}/${EXPECTED_RUNNERS} runners unhealthy (was ${prev_unhealthy_count})

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
    log "UPDATE: ${current_unhealthy_count} unhealthy (was ${prev_unhealthy_count}). wedge=${wedge_count} crashloop=${crashloop_count}. Slack notified."
    auto_bounce "${remediation_targets}"

elif [[ $current_unhealthy_count -eq 0 ]] && [[ $prev_unhealthy_count -gt 0 ]]; then
    # Transition: broken → all recovered
    slack_post "*[RUNNER RECOVERED]* All ${EXPECTED_RUNNERS} runners healthy

Online: ${online_count} | Busy: ${busy_count}
Docker socket: $([ "$docker_ok" = true ] && echo 'OK' || echo 'FAILED')
Host: ${RUNNER_HOST}" "good"
    log "RECOVERED: All ${EXPECTED_RUNNERS} runners healthy. Slack notified."

else
    # No state change — silent
    log "OK: ${healthy}/${EXPECTED_RUNNERS} healthy, ${current_unhealthy_count} unhealthy (wedge=${wedge_count} crashloop=${crashloop_count}, no change)."
fi
