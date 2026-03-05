#!/usr/bin/env bash
# runner-monitor.sh — Self-hosted runner health monitor with Slack alerts
# Deployed to: 192.168.86.201:~/.omnibase/runners/runner-monitor.sh
# Cron: */3 * * * * (every 3 minutes)
#
# Checks all omninode-runner-* containers. Fires a Slack alert on state
# transitions (healthy → unhealthy/down). Resolves silently when all recover.
# Uses a state file to prevent alert spam.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXPECTED_RUNNERS=10
STATE_FILE="/tmp/runner-monitor-state.json"
COMPOSE_DIR="$HOME/.omnibase/runners/docker"
COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.runners.yml"

# Slack config — passed via environment (cron sources ~/.omnibase/.env)
: "${SLACK_BOT_TOKEN:?SLACK_BOT_TOKEN must be set}"
: "${SLACK_CHANNEL_ID:?SLACK_CHANNEL_ID must be set}"

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
            '{
                channel: $channel,
                attachments: [{
                    color: $color,
                    text: $text,
                    footer: "runner-monitor | 192.168.86.201",
                    ts: (now | floor)
                }]
            }'
        )" > /dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Collect current state
# ---------------------------------------------------------------------------
declare -A current_status

total_found=0
healthy=0
unhealthy_list=()

while IFS=$'\t' read -r name status; do
    total_found=$((total_found + 1))
    current_status["$name"]="$status"

    if [[ "$status" == *"(healthy)"* ]] && [[ "$status" == Up* ]]; then
        healthy=$((healthy + 1))
    else
        unhealthy_list+=("${name}: ${status}")
    fi
done < <(docker ps -a --filter "name=omninode-runner" --format "{{.Names}}\t{{.Status}}" 2>/dev/null || true)

# Check for missing runners (expected but not even in docker ps)
for i in $(seq 1 $EXPECTED_RUNNERS); do
    name="omninode-runner-${i}"
    if [[ -z "${current_status[$name]+x}" ]]; then
        unhealthy_list+=("${name}: MISSING (no container)")
        total_found=$((total_found + 1))  # count as expected
    fi
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
Host: 192.168.86.201" "danger"
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
Host: 192.168.86.201" "good"
    log "RECOVERED: All ${EXPECTED_RUNNERS} runners healthy. Slack notified."

else
    # No state change — silent
    log "OK: ${healthy}/${EXPECTED_RUNNERS} healthy, ${current_unhealthy_count} unhealthy (no change)."
fi
