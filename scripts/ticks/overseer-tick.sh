#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# overseer-tick.sh — Pull Linear backlog, identify gaps, dispatch headless builders.
#
# Schedule: every 15 minutes
# Output:   /data/onex/ticks/overseer-YYYYMMDD-HHmm.md
# Mirror:   rsync to $OMNI_HOME/.onex_state/ticks/ on local Mac (if ONEX_MAC_HOST set)
#
# Behaviour (idempotent):
#   1. Query Linear for In-Progress + Todo tickets in team "OmniNode"
#   2. Identify tickets with no active claim (not in claims TTL table)
#   3. Dispatch builders via `claude -p` headless for unclaimed gaps
#   4. Write tick report with in-flight count, gap count, new dispatches
#
# Claims TTL: /data/onex/overseer-claims.json
#   {ticket_id: {claimed_at: epoch, expires_at: epoch, pid: N}}
#   Expired claims (>90min) are pruned each tick.

set -euo pipefail

# shellcheck source=/dev/null
source "${HOME}/.omnibase/.env"

OMNI_HOME="${OMNI_HOME:-/home/jonah/Code/omni_home}"
TICK_DIR="/data/onex/ticks"
FRICTION_DIR="${OMNI_HOME}/.onex_state/friction"
CLAIMS_FILE="/data/onex/overseer-claims.json"
TIMESTAMP="$(date -u +%Y%m%d-%H%M)"
REPORT="${TICK_DIR}/overseer-${TIMESTAMP}.md"
ONEX_MAC_HOST="${ONEX_MAC_HOST:-}"

CLAIM_TTL_SECONDS=5400  # 90 minutes
MAX_DISPATCHES=3        # cap per tick to avoid thundering herd

mkdir -p "${TICK_DIR}" "${FRICTION_DIR}" "$(dirname "${CLAIMS_FILE}")"

# Require LINEAR_API_KEY
if [[ -z "${LINEAR_API_KEY:-}" ]]; then
    echo "ERROR: LINEAR_API_KEY not set" >&2
    exit 1
fi

# ---- Claims helpers ----

_now() { date +%s; }

_load_claims() {
    if [[ -f "${CLAIMS_FILE}" ]]; then
        cat "${CLAIMS_FILE}"
    else
        echo "{}"
    fi
}

_prune_claims() {
    local claims="$1"
    local now
    now=$(_now)
    echo "${claims}" | jq --argjson now "${now}" \
        'to_entries | map(select(.value.expires_at > $now)) | from_entries'
}

_is_claimed() {
    local claims="$1" ticket_id="$2"
    local now
    now=$(_now)
    echo "${claims}" | jq -r --arg id "${ticket_id}" --argjson now "${now}" \
        'if .[$id] and (.[$id].expires_at > $now) then "yes" else "no" end'
}

_add_claim() {
    local claims="$1" ticket_id="$2" pid="$3"
    local now expires
    now=$(_now)
    expires=$((now + CLAIM_TTL_SECONDS))
    echo "${claims}" | jq --arg id "${ticket_id}" --argjson pid "${pid}" \
        --argjson now "${now}" --argjson exp "${expires}" \
        '.[$id] = {claimed_at: $now, expires_at: $exp, pid: $pid}'
}

# ---- Linear query ----

_query_linear() {
    curl -sf "https://api.linear.app/graphql" \
        -H "Authorization: ${LINEAR_API_KEY}" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "{ issues(filter: { team: { name: { eq: \"OmniNode\" } }, state: { type: { in: [\"started\", \"unstarted\"] } } }, first: 50) { nodes { id identifier title priority state { name type } } } }"
        }' | jq '.data.issues.nodes'
}

# ---- Dispatch helpers ----

_dispatch_builder() {
    local ticket_id="$1" ticket_title="$2"
    local worktree_base="${OMNI_HOME}"

    local prompt="You are a background builder. Work on Linear ticket ${ticket_id}: \"${ticket_title}\". \
Pull the ticket details from Linear, identify which repo and files to change, create a worktree, \
implement the changes, run tests, commit, push, and open a PR. \
Working directory: ${worktree_base}. \
Follow CLAUDE.md conventions. Report completion by updating the ticket state to In Review."

    # Dispatch headless, capturing PID
    local log_file="${TICK_DIR}/builder-${ticket_id}-${TIMESTAMP}.log"
    claude --print \
        --permission-mode auto \
        "${prompt}" \
        > "${log_file}" 2>&1 &
    echo $!
}

# ---- Main ----

CLAIMS=$(_load_claims)
CLAIMS=$(_prune_claims "${CLAIMS}")

# Query Linear
LINEAR_ISSUES=$(_query_linear 2>/dev/null) || {
    echo "ERROR: Linear query failed" >&2
    cat > "${FRICTION_DIR}/overseer-tick-linear-fail-${TIMESTAMP}.md" <<EOF
# Friction: overseer-tick Linear query failed — ${TIMESTAMP}

Source: overseer-tick.sh
Linear API unreachable or LINEAR_API_KEY invalid.
EOF
    exit 1
}

IN_PROGRESS_COUNT=$(echo "${LINEAR_ISSUES}" | jq '[.[] | select(.state.type == "started")] | length')
TODO_COUNT=$(echo "${LINEAR_ISSUES}" | jq '[.[] | select(.state.type == "unstarted")] | length')

# Find unclaimed todo/in-progress tickets
UNCLAIMED=()
while IFS= read -r issue; do
    local_id=$(echo "${issue}" | jq -r '.identifier')
    local_title=$(echo "${issue}" | jq -r '.title')
    claimed=$(_is_claimed "${CLAIMS}" "${local_id}")
    if [[ "${claimed}" == "no" ]]; then
        UNCLAIMED+=("${local_id}|${local_title}")
    fi
done < <(echo "${LINEAR_ISSUES}" | jq -c '.[]')

GAP_COUNT="${#UNCLAIMED[@]}"
DISPATCHED=0
declare -a DISPATCH_LOG=()

# Dispatch up to MAX_DISPATCHES builders for unclaimed gaps
for item in "${UNCLAIMED[@]}"; do
    [[ "${DISPATCHED}" -ge "${MAX_DISPATCHES}" ]] && break
    local_id="${item%%|*}"
    local_title="${item#*|}"

    pid=$(_dispatch_builder "${local_id}" "${local_title}")
    CLAIMS=$(_add_claim "${CLAIMS}" "${local_id}" "${pid}")
    DISPATCH_LOG+=("${local_id} — ${local_title} (pid=${pid})")
    DISPATCHED=$((DISPATCHED + 1))
done

# Persist updated claims
echo "${CLAIMS}" > "${CLAIMS_FILE}"

# Write report
{
    echo "# Overseer Tick — ${TIMESTAMP}"
    echo ""
    echo "**In-progress**: ${IN_PROGRESS_COUNT}  |  **Todo**: ${TODO_COUNT}  |  **Gaps (unclaimed)**: ${GAP_COUNT}  |  **Dispatched this tick**: ${DISPATCHED}"
    echo ""

    if [[ "${#DISPATCH_LOG[@]}" -gt 0 ]]; then
        echo "## Dispatched builders"
        for item in "${DISPATCH_LOG[@]}"; do echo "- ${item}"; done
        echo ""
    fi

    if [[ "${GAP_COUNT}" -gt "${DISPATCHED}" ]]; then
        echo "## Remaining gaps (dispatch cap reached)"
        echo ""
        local_i=0
        for item in "${UNCLAIMED[@]}"; do
            [[ "${local_i}" -lt "${MAX_DISPATCHES}" ]] && { local_i=$((local_i + 1)); continue; }
            echo "- ${item%%|*} — ${item#*|}"
        done
        echo ""
    fi

    echo "_Generated by overseer-tick.sh at $(date -u +"%Y-%m-%dT%H:%M:%SZ")_"
} > "${REPORT}"

# Mirror tick reports to local Mac
if [[ -n "${ONEX_MAC_HOST}" ]]; then
    rsync -a --quiet "${TICK_DIR}/" "${ONEX_MAC_HOST}:${OMNI_HOME}/.onex_state/ticks/" 2>/dev/null || true
fi
