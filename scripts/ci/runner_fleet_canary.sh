#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# runner_fleet_canary.sh — scheduled fleet-status canary (OMN-13915, OMN-16030)
#
# Compares the GitHub org self-hosted runner registry against the expected fleet
# size declared in config/runner_fleet.yaml, and FAILS LOUDLY on the signals that
# actually prove the fleet stopped serving jobs — BEFORE queued CI runs pile up.
#
# Why this exists: on 2026-07-03 the org API showed 37/48 runners offline while
# every runner container on .201 reported "Up (healthy)". Docker-side checks
# (healthcheck, runner-monitor cron on the host itself) share fate with the
# host; this canary runs on GitHub-hosted compute so it stays alive when the
# fleet — or the whole .201 host — is dead.
#
# OMN-16030 — WHY `status == "offline"` ALONE IS NOT A LIVENESS SIGNAL.
# This script previously treated the org REST `status` field as "the
# AUTHORITATIVE view of whether runners are serving jobs" and failed whenever
# more than RUNNER_CANARY_MAX_OFFLINE runners reported offline. On 2026-08-14
# that assumption was measured and falsified against the 72-runner fleet:
#
#   * Runners reporting `offline` were concurrently executing jobs. runner-51
#     completed a job at 10:24:22Z and runner-67 at 10:25:20Z while both were
#     labelled offline; runner-30 held an in-progress job while labelled
#     offline; `ps` inside runner-23 showed Runner.Worker running `uv sync`
#     while the registry reported it offline+busy.
#   * Over a 2h40m window the 13 persistently-offline-labelled runners served
#     153 jobs (mean 11.8/runner) versus 14.7/runner for online-labelled ones —
#     ~80% of nominal throughput, not zero.
#   * `missing` was 0 in every sample all day and RestartCount was 0 on all 72
#     containers: nothing ever actually de-registered or crashed.
#   * The offline count correlates POSITIVELY with concurrent job count
#     (Pearson r=+0.55, n=7) — it reads worst exactly when the fleet is
#     busiest, which is the opposite of a liveness signal.
#
# Mechanism: these runners use the Actions V2 broker flow (`useV2Flow: true`,
# serverUrlV2 = broker.actions.githubusercontent.com). The org REST `status`
# field reflects broker-session bookkeeping that goes stale under load; it is
# not a heartbeat. A transient stale read is not a dead listener.
#
# Cost of getting this wrong: a persistently-red canary is indistinguishable
# from a real outage, so it trains operators to ignore it AND it halts landing
# sweeps on a false alarm (observed 2026-08-14: two sweeps halted, and a
# proposed "recovery" would have force-recreated healthy runners, killing
# in-flight jobs and wiping the warm tool cache).
#
# What this canary fails on now — signals that cannot be produced by a stale
# read, in descending order of certainty:
#   1. missing > 0            — a runner lost its REGISTRATION. Unambiguous.
#   2. offline fraction >= RUNNER_CANARY_MASS_OFFLINE_PCT — mass listener death
#      (the 2026-07-03 mode was 77%). Load-induced staleness has never exceeded
#      ~22% in measurement, so this band separates the two cleanly.
# A `busy` runner is counted ALIVE regardless of `status`: it is demonstrably
# executing a job, which is the thing the canary exists to protect.
# Anything between the advisory threshold and the mass-outage threshold is a
# WARNING (visible in the step summary + Slack) on a GREEN run — loud enough to
# investigate, not loud enough to block landing.
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
#   RUNNER_CANARY_MAX_OFFLINE   offline count above which the run WARNS (default 5).
#                               Advisory only — see OMN-16030 note above.
#   RUNNER_CANARY_MASS_OFFLINE_PCT  percent of the fleet reporting offline-and-not-busy
#                               at which the run FAILS (default 50).
#   RUNNER_FLEET_CONFIG_PATH    path to runner_fleet.yaml (default config/runner_fleet.yaml)
#   GITHUB_API_URL              API base (set by Actions; default https://api.github.com)
#   GITHUB_STEP_SUMMARY         if set, a markdown summary is appended
#   SLACK_BOT_TOKEN + SLACK_CHANNEL_ID  best-effort Slack alert on breach

set -euo pipefail

RUNNER_FLEET_CONFIG_PATH="${RUNNER_FLEET_CONFIG_PATH:-config/runner_fleet.yaml}"
RUNNER_CANARY_MAX_OFFLINE="${RUNNER_CANARY_MAX_OFFLINE:-5}"
RUNNER_CANARY_MASS_OFFLINE_PCT="${RUNNER_CANARY_MASS_OFFLINE_PCT:-50}"
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

# OMN-16030: a runner reporting offline while `busy` is demonstrably executing a
# job — the registry read is stale, the listener is not dead. Only offline AND
# not-busy runners are candidates for "actually unreachable".
offline_idle_count=$(jq '[ .[] | select(.status != "online") | select(.busy != true) ] | length' <<< "${fleet}")
offline_busy_count=$(( offline_count - offline_idle_count ))

# Mass-outage fraction is computed against offline-and-not-busy plus lost
# registrations — the two states a stale read cannot manufacture.
unreachable=$(( offline_idle_count + missing_count ))
mass_threshold=$(( EXPECTED_RUNNERS * RUNNER_CANARY_MASS_OFFLINE_PCT / 100 ))

offline_names=$(jq -r '[ .[] | select(.status != "online") | .name ] | join(", ")' <<< "${fleet}")

log "expected=${EXPECTED_RUNNERS} registered=${total_registered} online=${online_count} offline=${offline_count} offline_but_busy=${offline_busy_count} offline_idle=${offline_idle_count} missing=${missing_count} unreachable=${unreachable} warn_threshold=${RUNNER_CANARY_MAX_OFFLINE} fail_threshold=${mass_threshold}"

summary() {
    cat <<EOF
## Runner fleet canary (OMN-13915, OMN-16030)

| Metric | Value |
|--------|-------|
| Expected fleet size | ${EXPECTED_RUNNERS} |
| Registered (org API) | ${total_registered} |
| Online | ${online_count} |
| Offline (reported) | ${offline_count} |
| — of which busy (alive, stale read) | ${offline_busy_count} |
| — of which idle (unreachable candidates) | ${offline_idle_count} |
| Missing registrations | ${missing_count} |
| **Unreachable (offline-idle + missing)** | **${unreachable}** |
| Warn above | ${RUNNER_CANARY_MAX_OFFLINE} |
| **Fail at or above** | **${mass_threshold}** (${RUNNER_CANARY_MASS_OFFLINE_PCT}% of fleet) |

Offline-reporting runners: ${offline_names:-none}

> A runner reporting \`offline\` is NOT proof it is dead. These runners use the
> Actions V2 broker flow, whose REST \`status\` goes stale under load — measured
> 2026-08-14, offline-labelled runners served ~80% of nominal job throughput
> (OMN-16030). Only lost registrations and mass offline-idle fail this gate.
EOF
}

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    summary >> "${GITHUB_STEP_SUMMARY}"
fi

slack_alert() {
    local severity="${1}"
    local detail="${2}"
    [[ -n "${SLACK_BOT_TOKEN:-}" && -n "${SLACK_CHANNEL_ID:-}" ]] || return 0
    curl -s -X POST https://slack.com/api/chat.postMessage \
        -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg channel "${SLACK_CHANNEL_ID}" \
            --arg text "*[RUNNER FLEET CANARY — ${severity}]* ${detail} (expected=${EXPECTED_RUNNERS} online=${online_count} offline=${offline_count} offline_but_busy=${offline_busy_count} offline_idle=${offline_idle_count} missing=${missing_count}). Offline: ${offline_names:-none}. See docs/runbooks/runner-fleet-listener-liveness.md" \
            '{channel: $channel, text: $text}')" > /dev/null 2>&1 || true
}

# --- FAIL 1: lost registrations. A stale status read cannot remove a runner
# from the registry, so this is unambiguous evidence of real fleet loss.
if [[ "${missing_count}" -gt 0 ]]; then
    slack_alert "FAIL" "${missing_count} runner registration(s) LOST (registered=${total_registered}/${EXPECTED_RUNNERS})"
    fail "${missing_count} runner registration(s) missing (registered=${total_registered}, expected=${EXPECTED_RUNNERS}). A runner dropped its registration entirely — this is real fleet loss, not a stale status read. See docs/runbooks/runner-fleet-listener-liveness.md"
fi

# --- FAIL 2: mass listener death (the 2026-07-03 mode, 77% of fleet).
if [[ "${unreachable}" -ge "${mass_threshold}" ]]; then
    slack_alert "FAIL" "${unreachable}/${EXPECTED_RUNNERS} runners offline-and-idle (>= ${mass_threshold})"
    fail "${unreachable}/${EXPECTED_RUNNERS} runners offline-and-not-busy (>= ${mass_threshold} = ${RUNNER_CANARY_MASS_OFFLINE_PCT}% of fleet). At this scale it is no longer explainable as broker-status staleness — treat as mass listener death. Offline: ${offline_names:-none}. Do NOT trust Docker 'Up (healthy)'. See docs/runbooks/runner-fleet-listener-liveness.md"
fi

# --- WARN: elevated but within the band that measurement attributes to V2
# broker-status staleness. Visible, not blocking. See OMN-16030.
if [[ "${unreachable}" -gt "${RUNNER_CANARY_MAX_OFFLINE}" ]]; then
    slack_alert "WARN" "${unreachable}/${EXPECTED_RUNNERS} runners offline-and-idle (> ${RUNNER_CANARY_MAX_OFFLINE}, below fail threshold ${mass_threshold})"
    log "WARN: ${unreachable}/${EXPECTED_RUNNERS} offline-and-idle — above the advisory threshold (${RUNNER_CANARY_MAX_OFFLINE}) but below the mass-outage threshold (${mass_threshold})."
    log "WARN: this band is attributed to V2 broker-status staleness (OMN-16030). Confirm with actual job throughput before any restart — offline-labelled runners are usually still serving jobs."
    log "OK: fleet serving; not failing on a status-staleness signal."
    exit 0
fi

log "OK: fleet within threshold."
