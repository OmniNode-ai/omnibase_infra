#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# runner_fleet_canary.sh — scheduled fleet-status canary (OMN-13915)
#
# Compares the GitHub org self-hosted runner registry (the AUTHORITATIVE view
# of whether runners are serving jobs) against the expected fleet size declared
# in config/runner_fleet.yaml, and FAILS LOUDLY when the offline count crosses
# a threshold — BEFORE queued CI runs pile up.
#
# Why this exists: on 2026-07-03 the org API showed 37/48 runners offline while
# every runner container on .201 reported "Up (healthy)". Docker-side checks
# (healthcheck, runner-monitor cron on the host itself) share fate with the
# host; this canary runs on GitHub-hosted compute so it stays alive when the
# fleet — or the whole .201 host — is dead.
#
# Enforcement surface: .github/workflows/runner-fleet-canary.yml runs this on a
# 15-minute schedule on ubuntu-latest. A threshold breach fails the workflow
# run (red X + owner notification). This is not an opt-in script.
#
# Required env:
#   RUNNER_FLEET_STATUS_TOKEN or CROSS_REPO_PAT — token able to read
#     GET /orgs/{org}/actions/runners (classic PAT: admin:org read;
#     fine-grained: org "Self-hosted runners" read).
# Optional env:
#   RUNNER_CANARY_MAX_OFFLINE   max offline runners tolerated (default 5)
#   RUNNER_FLEET_CONFIG_PATH    path to runner_fleet.yaml (default config/runner_fleet.yaml)
#   GITHUB_API_URL              API base (set by Actions; default https://api.github.com)
#   GITHUB_STEP_SUMMARY         if set, a markdown summary is appended
#   SLACK_BOT_TOKEN + SLACK_CHANNEL_ID  best-effort Slack alert on breach

set -euo pipefail

RUNNER_FLEET_CONFIG_PATH="${RUNNER_FLEET_CONFIG_PATH:-config/runner_fleet.yaml}"
RUNNER_CANARY_MAX_OFFLINE="${RUNNER_CANARY_MAX_OFFLINE:-5}"
GITHUB_API_URL="${GITHUB_API_URL:-https://api.github.com}"

log() { echo "[fleet-canary] $*"; }

fail() {
    echo "[fleet-canary] FAIL: $*" >&2
    exit 1
}

# --- config (same awk extraction contract as runner-monitor.sh) -------------
config_field() {
    local field="${1}"
    [[ -f "${RUNNER_FLEET_CONFIG_PATH}" ]] || fail "runner fleet config not found: ${RUNNER_FLEET_CONFIG_PATH}"
    local value
    value=$(awk -F':[[:space:]]*' -v key="${field}" '
        $1 == key {
            gsub(/^[[:space:]"]+|[[:space:]"]+$/, "", $2)
            print $2
            found=1
        }
        END { if (!found) exit 1 }
    ' "${RUNNER_FLEET_CONFIG_PATH}") || fail "missing ${field} in ${RUNNER_FLEET_CONFIG_PATH}"
    echo "${value}"
}

RUNNER_ORG="$(config_field github_org)"
RUNNER_GROUP="$(config_field runner_group)"
RUNNER_NAME_PREFIX="$(config_field runner_name_prefix)"
EXPECTED_RUNNERS="$(config_field expected_count)"

# --- token selection (fail-closed: no token => red run, not a silent skip) --
TOKEN="${RUNNER_FLEET_STATUS_TOKEN:-${CROSS_REPO_PAT:-}}"
[[ -n "${TOKEN}" ]] || fail "no API token: set the RUNNER_FLEET_STATUS_TOKEN secret (org-runner read scope) or CROSS_REPO_PAT"

# --- fetch org runner registry (paginated, bounded retries) ------------------
fetch_page() {
    local page="${1}"
    local attempt
    for attempt in 1 2 3; do
        if curl -fsS \
            -H "Authorization: Bearer ${TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            "${GITHUB_API_URL}/orgs/${RUNNER_ORG}/actions/runners?per_page=100&page=${page}"; then
            return 0
        fi
        sleep $((attempt * 5))
    done
    return 1
}

all_runners="[]"
page=1
while :; do
    page_json=$(fetch_page "${page}") || fail "GitHub org runners API unreachable after retries (fail-closed: cannot prove fleet health)"
    page_runners=$(jq '.runners' <<< "${page_json}")
    count=$(jq 'length' <<< "${page_runners}")
    all_runners=$(jq -s '.[0] + .[1]' <(echo "${all_runners}") <(echo "${page_runners}"))
    [[ "${count}" -lt 100 ]] && break
    page=$((page + 1))
    [[ "${page}" -gt 10 ]] && break  # hard cap: 1000 runners
done

# --- classify the fleet (name prefix + runner-group label) -------------------
fleet=$(jq --arg prefix "${RUNNER_NAME_PREFIX}" --arg group "${RUNNER_GROUP}" '
    [ .[]
      | select(.name | startswith($prefix))
      | select(any(.labels[]; .name == $group))
    ]' <<< "${all_runners}")

total_registered=$(jq 'length' <<< "${fleet}")
online_count=$(jq '[ .[] | select(.status == "online") ] | length' <<< "${fleet}")
offline_count=$(jq '[ .[] | select(.status != "online") ] | length' <<< "${fleet}")
missing_count=$(( EXPECTED_RUNNERS - total_registered ))
[[ "${missing_count}" -lt 0 ]] && missing_count=0
# A runner that dropped its registration entirely is offline in every way that
# matters — count it against the same threshold.
effective_offline=$(( offline_count + missing_count ))

offline_names=$(jq -r '[ .[] | select(.status != "online") | .name ] | join(", ")' <<< "${fleet}")

log "expected=${EXPECTED_RUNNERS} registered=${total_registered} online=${online_count} offline=${offline_count} missing=${missing_count} threshold=${RUNNER_CANARY_MAX_OFFLINE}"

summary() {
    cat <<EOF
## Runner fleet canary (OMN-13915)

| Metric | Value |
|--------|-------|
| Expected fleet size | ${EXPECTED_RUNNERS} |
| Registered (org API) | ${total_registered} |
| Online | ${online_count} |
| Offline | ${offline_count} |
| Missing registrations | ${missing_count} |
| Offline threshold | ${RUNNER_CANARY_MAX_OFFLINE} |

Offline runners: ${offline_names:-none}
EOF
}

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    summary >> "${GITHUB_STEP_SUMMARY}"
fi

slack_alert() {
    [[ -n "${SLACK_BOT_TOKEN:-}" && -n "${SLACK_CHANNEL_ID:-}" ]] || return 0
    curl -s -X POST https://slack.com/api/chat.postMessage \
        -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg channel "${SLACK_CHANNEL_ID}" \
            --arg text "*[RUNNER FLEET CANARY]* ${effective_offline}/${EXPECTED_RUNNERS} runners offline-or-missing (online=${online_count}, threshold=${RUNNER_CANARY_MAX_OFFLINE}). Offline: ${offline_names:-none}. Docker 'Up (healthy)' is NOT sufficient evidence — see OMN-13915 runbook." \
            '{channel: $channel, text: $text}')" > /dev/null 2>&1 || true
}

if [[ "${effective_offline}" -gt "${RUNNER_CANARY_MAX_OFFLINE}" ]]; then
    slack_alert
    fail "${effective_offline}/${EXPECTED_RUNNERS} runners offline-or-missing (> ${RUNNER_CANARY_MAX_OFFLINE}). Offline: ${offline_names:-<registrations missing>}. The fleet is degrading silently — do NOT trust Docker 'Up (healthy)'. See docs/runbooks/runner-fleet-listener-liveness.md"
fi

log "OK: fleet within threshold."
