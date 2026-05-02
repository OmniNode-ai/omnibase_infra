#!/usr/bin/env bash
# runner-monitor.sh — Self-hosted runner health monitor with Slack alerts
# Deployed to: 192.168.86.201:~/.omnibase/runners/runner-monitor.sh
# Cron: */3 * * * * (every 3 minutes)
#
# Checks configured omninode-runner-* containers and their GitHub Actions
# registrations. Fires a Slack alert on state transitions. Resolves silently
# when all recover. Uses a state file to prevent alert spam.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STATE_FILE="/tmp/runner-monitor-state.json"
COMPOSE_DIR="$HOME/.omnibase/runners/docker"
RUNNER_FLEET_CONFIG_PATH="${RUNNER_FLEET_CONFIG_PATH:-$HOME/.omnibase/runners/config/runner_fleet.yaml}"

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

# ---------------------------------------------------------------------------
# Collect current state
# ---------------------------------------------------------------------------
declare -A current_status
declare -A docker_oom_killed
declare -A github_status

total_found=0
healthy=0
unhealthy_list=()
github_api_failed=false

while IFS=$'\t' read -r name status; do
    [[ -z "${name}" ]] && continue
    current_status["$name"]="$status"
done < <(docker ps -a --filter "name=${RUNNER_NAME_PREFIX}" --format "{{.Names}}\t{{.Status}}" 2>/dev/null || true)

for name in "${!current_status[@]}"; do
    oom_killed="$(docker inspect --format '{{.State.OOMKilled}}' "$name" 2>/dev/null || echo "unknown")"
    docker_oom_killed["$name"]="$oom_killed"
done

github_json=$(curl -fsS \
    -H "Authorization: Bearer ${RUNNER_GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/orgs/${RUNNER_ORG}/actions/runners?per_page=100" 2>/dev/null || true)

if [[ -z "${github_json}" ]]; then
    github_api_failed=true
    unhealthy_list+=("GITHUB_API: failed to fetch org runner status")
else
    while IFS=$'\t' read -r name status; do
        [[ -z "${name}" ]] && continue
        github_status["$name"]="$status"
    done < <(jq -r --arg prefix "${RUNNER_NAME_PREFIX}" --arg group "${RUNNER_GROUP}" '
        .runners[]
        | select(.name | startswith($prefix))
        | select(any(.labels[]; .name == $group))
        | [.name, .status]
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

    if [[ "${github_api_failed}" == true ]]; then
        healthy=$((healthy + 1))
        continue
    fi

    gh_state="${github_status[$name]:-missing}"
    if [[ "${gh_state}" != "online" ]]; then
        unhealthy_list+=("${name}: GitHub ${gh_state} while Docker ${docker_state}")
        continue
    fi

    healthy=$((healthy + 1))
done

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
# Compare with previous state and alert on transitions
# ---------------------------------------------------------------------------
prev_unhealthy_count=0
if [[ -f "$STATE_FILE" ]]; then
    prev_unhealthy_count=$(jq -r '.unhealthy_count // 0' "$STATE_FILE" 2>/dev/null || echo 0)
fi

current_unhealthy_count=${#unhealthy_list[@]}

# Write current state
jq -n \
    --argjson healthy "$healthy" \
    --argjson unhealthy_count "$current_unhealthy_count" \
    --argjson total "$total_found" \
    --argjson docker_ok "$docker_ok" \
    --arg timestamp "$(date -Iseconds)" \
    --arg unhealthy_names "$(printf '%s\n' "${unhealthy_list[@]}" 2>/dev/null || echo '')" \
    '{
        healthy: $healthy,
        unhealthy_count: $unhealthy_count,
        total: $total,
        docker_ok: $docker_ok,
        timestamp: $timestamp,
        unhealthy_names: $unhealthy_names
    }' > "$STATE_FILE"

# ---------------------------------------------------------------------------
# Alert logic — only on state transitions
# ---------------------------------------------------------------------------

if [[ $current_unhealthy_count -gt 0 ]] && [[ $prev_unhealthy_count -eq 0 ]]; then
    # Transition: all-good → something broken
    detail=$(printf '%s\n' "${unhealthy_list[@]}")
    slack_post "*[RUNNER ALERT]* ${current_unhealthy_count}/${EXPECTED_RUNNERS} runners unhealthy

\`\`\`
${detail}
\`\`\`

Healthy: ${healthy}/${EXPECTED_RUNNERS}
Docker socket: $([ "$docker_ok" = true ] && echo 'OK' || echo 'FAILED')
Host: ${RUNNER_HOST}" "danger"
    log "ALERT: ${current_unhealthy_count} runners unhealthy (was 0). Slack notified."

elif [[ $current_unhealthy_count -gt 0 ]] && [[ $prev_unhealthy_count -gt 0 ]] && [[ $current_unhealthy_count -ne $prev_unhealthy_count ]]; then
    # Transition: bad → worse or bad → partially recovered
    detail=$(printf '%s\n' "${unhealthy_list[@]}")
    slack_post "*[RUNNER UPDATE]* ${current_unhealthy_count}/${EXPECTED_RUNNERS} runners unhealthy (was ${prev_unhealthy_count})

\`\`\`
${detail}
\`\`\`

Healthy: ${healthy}/${EXPECTED_RUNNERS}" "warning"
    log "UPDATE: ${current_unhealthy_count} unhealthy (was ${prev_unhealthy_count}). Slack notified."

elif [[ $current_unhealthy_count -eq 0 ]] && [[ $prev_unhealthy_count -gt 0 ]]; then
    # Transition: broken → all recovered
    slack_post "*[RUNNER RECOVERED]* All ${EXPECTED_RUNNERS} runners healthy

Docker socket: $([ "$docker_ok" = true ] && echo 'OK' || echo 'FAILED')
Host: ${RUNNER_HOST}" "good"
    log "RECOVERED: All ${EXPECTED_RUNNERS} runners healthy. Slack notified."

else
    # No state change — silent
    log "OK: ${healthy}/${EXPECTED_RUNNERS} healthy, ${current_unhealthy_count} unhealthy (no change)."
fi
