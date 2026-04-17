#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# merge-sweep-tick.sh — Org-wide PR sweep: arm auto-merge, update-branch, rerun stale gates.
#
# Schedule: every hour at :23
# Output:   /data/onex/ticks/merge-sweep-YYYYMMDD-HHmm.md
# Mirror:   rsync to $OMNI_HOME/.onex_state/ticks/ on local Mac (if ONEX_MAC_HOST set)
#
# Behaviour (idempotent):
#   CLEAN + unarmed  → enablePullRequestAutoMerge (SQUASH) via GraphQL
#   BEHIND           → gh pr update-branch
#   DIRTY            → flag in report, skip
#   stale-gate-fail  → gh workflow run to rerun gate
#
# Prerequisites: gh CLI authenticated, GH_TOKEN in env

set -euo pipefail

# shellcheck source=/dev/null
source "${HOME}/.omnibase/.env"

OMNI_HOME="${OMNI_HOME:-/home/jonah/Code/omni_home}"
TICK_DIR="/data/onex/ticks"
FRICTION_DIR="${OMNI_HOME}/.onex_state/friction"
TIMESTAMP="$(date -u +%Y%m%d-%H%M)"
REPORT="${TICK_DIR}/merge-sweep-${TIMESTAMP}.md"
ONEX_MAC_HOST="${ONEX_MAC_HOST:-}"

ORG="OmniNode-ai"
REPOS=(omniclaude omnibase_core omnibase_infra omnibase_spi omniintelligence omnimemory omnibase_compat)

mkdir -p "${TICK_DIR}" "${FRICTION_DIR}"

declare -a MERGED=()
declare -a ARMED=()
declare -a UPDATED=()
declare -a DIRTY=()
declare -a RERUN=()
declare -a ERRORS=()

_graphql_enable_auto_merge() {
    local pr_node_id="$1"
    gh api graphql -f query="
        mutation {
          enablePullRequestAutoMerge(input: {
            pullRequestId: \"${pr_node_id}\"
            mergeMethod: SQUASH
          }) {
            pullRequest { number autoMergeRequest { mergeMethod } }
          }
        }
    " --jq '.data.enablePullRequestAutoMerge.pullRequest.number' 2>/dev/null || echo ""
}

_sweep_repo() {
    local repo="$1"

    # Fetch open, non-draft PRs with their merge state
    local prs
    prs=$(gh pr list \
        --repo "${ORG}/${repo}" \
        --state open \
        --json number,title,mergeable,mergeStateStatus,autoMergeRequest,isDraft,nodeId,headRefName \
        2>/dev/null) || {
        ERRORS+=("${repo}: gh pr list failed")
        return
    }

    local count
    count=$(echo "${prs}" | jq 'length')
    [[ "${count}" == "0" ]] && return

    while IFS= read -r pr; do
        local num title state auto_merge is_draft node_id
        num=$(echo "${pr}"       | jq -r '.number')
        title=$(echo "${pr}"     | jq -r '.title')
        state=$(echo "${pr}"     | jq -r '.mergeStateStatus')
        auto_merge=$(echo "${pr}"| jq -r '.autoMergeRequest')
        is_draft=$(echo "${pr}"  | jq -r '.isDraft')
        node_id=$(echo "${pr}"   | jq -r '.nodeId')

        # Skip drafts
        [[ "${is_draft}" == "true" ]] && continue

        case "${state}" in
            CLEAN)
                if [[ "${auto_merge}" == "null" ]]; then
                    result=$(_graphql_enable_auto_merge "${node_id}")
                    if [[ -n "${result}" ]]; then
                        ARMED+=("${repo}#${num} — ${title}")
                    else
                        ERRORS+=("${repo}#${num}: enablePullRequestAutoMerge failed")
                    fi
                fi
                # already armed = already counted in previous run
                ;;
            BEHIND)
                gh pr update-branch "${num}" --repo "${ORG}/${repo}" 2>/dev/null && \
                    UPDATED+=("${repo}#${num} — ${title}") || \
                    ERRORS+=("${repo}#${num}: update-branch failed")
                ;;
            BLOCKED)
                # Check if it's a stale gate (all checks passed historically but re-run needed)
                # Attempt rerun of required workflows for gate-fail PRs
                local failed_checks
                failed_checks=$(gh pr checks "${num}" --repo "${ORG}/${repo}" 2>/dev/null \
                    | grep -E "^[^✓].*fail|^[^✓].*timed" | head -3 || true)
                if [[ -n "${failed_checks}" ]]; then
                    # Rerun failed workflow runs for this PR's head SHA
                    local head_sha
                    head_sha=$(gh pr view "${num}" --repo "${ORG}/${repo}" --json headRefOid --jq '.headRefOid' 2>/dev/null || echo "")
                    if [[ -n "${head_sha}" ]]; then
                        # Find failed runs and rerun
                        gh run list --repo "${ORG}/${repo}" --commit "${head_sha}" --json databaseId,conclusion \
                            2>/dev/null | jq -r '.[] | select(.conclusion == "failure" or .conclusion == "timed_out") | .databaseId' | \
                            while read -r run_id; do
                                gh run rerun "${run_id}" --repo "${ORG}/${repo}" 2>/dev/null && \
                                    RERUN+=("${repo}#${num} run ${run_id}") || true
                            done
                    fi
                else
                    DIRTY+=("${repo}#${num} — ${title} (${state}: review/conversation block)")
                fi
                ;;
            DIRTY|UNSTABLE)
                DIRTY+=("${repo}#${num} — ${title} (${state})")
                ;;
            MERGED)
                MERGED+=("${repo}#${num} — ${title}")
                ;;
        esac
    done < <(echo "${prs}" | jq -c '.[]')
}

for repo in "${REPOS[@]}"; do
    _sweep_repo "${repo}"
done

# Write report
{
    echo "# Merge-Sweep Tick — ${TIMESTAMP}"
    echo ""
    echo "**Armed for auto-merge**: ${#ARMED[@]}  |  **Updated**: ${#UPDATED[@]}  |  **Dirty/blocked**: ${#DIRTY[@]}  |  **Errors**: ${#ERRORS[@]}"
    echo ""

    if [[ "${#ARMED[@]}" -gt 0 ]]; then
        echo "## Armed (SQUASH auto-merge enabled)"
        for item in "${ARMED[@]}"; do echo "- ${item}"; done
        echo ""
    fi

    if [[ "${#UPDATED[@]}" -gt 0 ]]; then
        echo "## Updated (update-branch)"
        for item in "${UPDATED[@]}"; do echo "- ${item}"; done
        echo ""
    fi

    if [[ "${#RERUN[@]}" -gt 0 ]]; then
        echo "## Gate reruns triggered"
        for item in "${RERUN[@]}"; do echo "- ${item}"; done
        echo ""
    fi

    if [[ "${#DIRTY[@]}" -gt 0 ]]; then
        echo "## Dirty / blocked (skipped)"
        for item in "${DIRTY[@]}"; do echo "- ${item}"; done
        echo ""
    fi

    if [[ "${#ERRORS[@]}" -gt 0 ]]; then
        echo "## Errors"
        for item in "${ERRORS[@]}"; do echo "- ${item}"; done
        echo ""
    fi

    echo "_Generated by merge-sweep-tick.sh at $(date -u +"%Y-%m-%dT%H:%M:%SZ")_"
} > "${REPORT}"

# Friction on errors
if [[ "${#ERRORS[@]}" -gt 0 ]]; then
    FRICTION_FILE="${FRICTION_DIR}/merge-sweep-tick-errors-${TIMESTAMP}.md"
    {
        echo "# Friction: merge-sweep-tick errors — ${TIMESTAMP}"
        echo ""
        for item in "${ERRORS[@]}"; do echo "- ${item}"; done
    } > "${FRICTION_FILE}"
fi

# Mirror tick reports to local Mac
if [[ -n "${ONEX_MAC_HOST}" ]]; then
    rsync -a --quiet "${TICK_DIR}/" "${ONEX_MAC_HOST}:${OMNI_HOME}/.onex_state/ticks/" 2>/dev/null || true
fi
