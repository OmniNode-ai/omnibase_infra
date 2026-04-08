#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# pr-snapshot.sh — Snapshot all open PRs across OmniNode-ai repos
#
# Usage:
#   pr-snapshot.sh [--output FILE] [--json]
#
# Lists: repo, PR#, title, author, mergeable state, CI status, review decision, age

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
OMNI_HOME="${OMNI_HOME:?OMNI_HOME must be set}"
SNAPSHOT_DIR="${OMNI_HOME}/.onex_state/merge-sweep"
SNAPSHOT_FILE="${SNAPSHOT_DIR}/pr-snapshot-$(date +%Y%m%d-%H%M%S).json"

# =============================================================================
# Registry (mirrors pull-all.sh)
# =============================================================================

REPOS=(
    omniclaude
    omnibase_compat
    omnibase_core
    omnibase_infra
    omnibase_spi
    omnidash
    omnigemini
    omniintelligence
    omnimarket
    omnimemory
    omninode_infra
    omniweb
    onex_change_control
)

# =============================================================================
# Defaults
# =============================================================================

OUTPUT_FILE=""
JSON_MODE=false

# =============================================================================
# Usage
# =============================================================================

usage() {
    cat <<EOF
${SCRIPT_NAME} — Snapshot all open PRs across OmniNode-ai repos

USAGE
    ${SCRIPT_NAME} [OPTIONS]

OPTIONS
    --output FILE   Write output to FILE (default: stdout for table, auto-named for --json)
    --json          Output structured JSON instead of table
    --help          Show this help message and exit

OUTPUT
    Table (default): repo, PR#, title, author, mergeable, CI, review decision, age
    JSON (--json):   Written to ${SNAPSHOT_DIR}/pr-snapshot-YYYYMMDD-HHMMSS.json

EXAMPLES
    ${SCRIPT_NAME}
    ${SCRIPT_NAME} --json
    ${SCRIPT_NAME} --json --output /tmp/prs.json
EOF
    exit 0
}

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            if [[ -z "${2:-}" || "${2:0:1}" == "-" ]]; then
                printf '[pr-snapshot] ERROR: --output requires a value\n' >&2
                exit 1
            fi
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --json)
            JSON_MODE=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            printf '[pr-snapshot] ERROR: Unknown option: %s\n' "$1" >&2
            printf '[pr-snapshot] Run %s --help for usage.\n' "${SCRIPT_NAME}" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Prerequisites
# =============================================================================

if ! command -v gh &>/dev/null; then
    printf '[pr-snapshot] ERROR: gh (GitHub CLI) is required but not found in PATH.\n' >&2
    exit 1
fi

if ! command -v jq &>/dev/null; then
    printf '[pr-snapshot] ERROR: jq is required but not found in PATH.\n' >&2
    exit 1
fi

# =============================================================================
# Helpers
# =============================================================================

# age_days: compute age in whole days from ISO 8601 date string
age_days() {
    local created_at="$1"
    local now
    now="$(date -u +%s)"
    local then
    # macOS date: -j -f format; GNU date: -d
    if date --version &>/dev/null 2>&1; then
        then="$(date -d "${created_at}" +%s 2>/dev/null || echo "${now}")"
    else
        then="$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "${created_at}" +%s 2>/dev/null || echo "${now}")"
    fi
    echo $(( (now - then) / 86400 ))
}

# ci_status: derive pass/fail/pending from statusCheckRollup array
ci_status_from_rollup() {
    local rollup_json="$1"
    if [[ "${rollup_json}" == "null" || "${rollup_json}" == "[]" ]]; then
        echo "none"
        return
    fi
    local conclusions
    conclusions="$(printf '%s' "${rollup_json}" | jq -r '.[].conclusion // .[].state // "PENDING"' 2>/dev/null | sort -u || true)"
    if echo "${conclusions}" | grep -q "FAILURE\|ERROR\|TIMED_OUT"; then
        echo "fail"
    elif echo "${conclusions}" | grep -q "PENDING\|IN_PROGRESS\|null"; then
        echo "pending"
    elif echo "${conclusions}" | grep -q "SUCCESS\|NEUTRAL\|SKIPPED"; then
        echo "pass"
    else
        echo "unknown"
    fi
}

# =============================================================================
# Collect PRs
# =============================================================================

printf '[pr-snapshot] Collecting open PRs across %d repos...\n' "${#REPOS[@]}" >&2

ALL_PRS="[]"

for repo in "${REPOS[@]}"; do
    printf '[pr-snapshot]   %s...\n' "${repo}" >&2

    raw="$(gh pr list \
        --repo "OmniNode-ai/${repo}" \
        --json "number,title,author,mergeable,statusCheckRollup,reviewDecision,createdAt,headRefName" \
        --limit 50 \
        2>/dev/null || echo "[]")"

    if [[ "${raw}" == "null" || "${raw}" == "" ]]; then
        raw="[]"
    fi

    # Annotate each PR with repo name and computed fields
    enriched="$(printf '%s' "${raw}" | jq --arg repo "${repo}" '
        map(. + {
            repo: $repo,
            ci_status: (
                if (.statusCheckRollup | length) == 0 then "none"
                elif (.statusCheckRollup | map(.conclusion // .state // "PENDING") | any(. == "FAILURE" or . == "ERROR" or . == "TIMED_OUT")) then "fail"
                elif (.statusCheckRollup | map(.conclusion // .state // "PENDING") | any(. == "PENDING" or . == "IN_PROGRESS")) then "pending"
                elif (.statusCheckRollup | map(.conclusion // .state // "PENDING") | all(. == "SUCCESS" or . == "NEUTRAL" or . == "SKIPPED")) then "pass"
                else "unknown"
                end
            ),
            age_days: (
                (now - (.createdAt | fromdateiso8601)) / 86400 | floor
            ),
            author_login: (.author.login // "unknown")
        })
    ' 2>/dev/null || echo "[]")"

    ALL_PRS="$(printf '%s\n%s' "${ALL_PRS}" "${enriched}" | jq -s 'add')"

    # Rate limit: 1s between repos
    sleep 1
done

# =============================================================================
# Output
# =============================================================================

if [[ "${JSON_MODE}" == true ]]; then
    # Resolve output path
    if [[ -n "${OUTPUT_FILE}" ]]; then
        OUT="${OUTPUT_FILE}"
        mkdir -p "$(dirname "${OUTPUT_FILE}")"
    else
        OUT="${SNAPSHOT_FILE}"
        mkdir -p "${SNAPSHOT_DIR}"
    fi

    printf '%s' "${ALL_PRS}" | jq '{
        snapshot_at: (now | strftime("%Y-%m-%dT%H:%M:%SZ")),
        total_prs: length,
        prs: .
    }' > "${OUT}"

    printf '[pr-snapshot] Written to %s\n' "${OUT}" >&2
    printf '[pr-snapshot] Total PRs: %s\n' "$(jq '.total_prs' "${OUT}")" >&2
else
    # Table output
    {
        printf '%-22s  %-5s  %-9s  %-7s  %-8s  %-16s  %-5s  %s\n' \
            "REPO" "PR#" "MERGEABLE" "CI" "REVIEW" "AUTHOR" "AGE" "TITLE"
        printf '%s\n' "$(printf '%.0s-' {1..110})"

        printf '%s' "${ALL_PRS}" | jq -r '.[] |
            [
                .repo,
                (.number | tostring),
                (.mergeable // "UNKNOWN"),
                .ci_status,
                (.reviewDecision // "NONE"),
                .author_login,
                ((.age_days | tostring) + "d"),
                .title
            ] | @tsv
        ' | while IFS=$'\t' read -r repo num mergeable ci review author age title; do
            printf '%-22s  %-5s  %-9s  %-7s  %-8s  %-16s  %-5s  %s\n' \
                "${repo}" "${num}" "${mergeable}" "${ci}" "${review}" "${author}" "${age}" "${title:0:60}"
        done
    } | if [[ -n "${OUTPUT_FILE}" ]]; then
        tee "${OUTPUT_FILE}"
    else
        cat
    fi

    total="$(printf '%s' "${ALL_PRS}" | jq 'length')"
    printf '\n[pr-snapshot] Total open PRs: %s\n' "${total}" >&2
fi
