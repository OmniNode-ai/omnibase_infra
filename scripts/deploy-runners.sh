#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# deploy-runners.sh
# Single-command deploy entry point for OmniNode self-hosted GitHub Actions runners
# Ticket: OMN-3277 / Epic: OMN-3273
#
# Usage:
#   ./scripts/deploy-runners.sh [--dry-run] [--skip-build] [--soft]
#
# What it does (in order):
#   1. Fetch a fresh GitHub Actions registration token (valid 1 hour)
#   2. Base64-encode token for safe SSH passing
#   3. Rsync runner artifacts to 192.168.86.201:~/.omnibase/runners/
#   4. Deploy via SSH: docker compose up -d --build --force-recreate --remove-orphans
#   5. Install docker prune cron idempotently (build cache + untagged images, tee)
#   6. Install runner health monitor cron (Slack alerts on state transitions)
#   7. Poll GitHub API until all 10 runners online (max 5 min, 15s interval)
#   8. Retry once with fresh token if poll times out
#   9. Print stale runner report (offline runners with no host container)
#
# --soft mode (entrypoint-only update, no recreate):
#   1. Rsync runner artifacts to host
#   2. Rebuild the Docker image (for future containers)
#   3. docker cp the new entrypoint into each running container
#   4. docker restart each container (preserves filesystem + cached credentials)
#   Skips: registration token fetch, force-recreate, cron installs
#   Use when: updating entrypoint logic without needing fresh registration
#
# Requirements:
#   - gh CLI authenticated with org admin scope
#   - SSH access to 192.168.86.201 (key-based, no password prompts)
#   - rsync installed locally
#
# See also: docker/runners/Dockerfile, docker/docker-compose.runners.yml

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNNER_HOST="192.168.86.201"
# Remote path on the CI host — NOT local $HOME (which differs between macOS and Linux)
RUNNER_HOST_DIR="/home/jonah/.omnibase/runners"
RUNNER_ORG="OmniNode-ai"
RUNNER_GROUP="omnibase-ci"
RUNNER_NAME_PREFIX="omninode-runner"
RUNNER_COUNT=10
COMPOSE_FILE="docker/docker-compose.runners.yml"
POLL_MAX_SECONDS=300
POLL_INTERVAL_SECONDS=15
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Artifacts to sync to the host (relative to repo root)
SYNC_PATHS=(
    "docker/runners/Dockerfile"
    "docker/runners/entrypoint.sh"
    "docker/runners/runner-monitor.sh"
    "docker/docker-compose.runners.yml"
)

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

DRY_RUN=false
SKIP_BUILD=false
SOFT_DEPLOY=false

for arg in "$@"; do
    case "${arg}" in
        --dry-run)    DRY_RUN=true ;;
        --skip-build) SKIP_BUILD=true ;;
        --soft)       SOFT_DEPLOY=true ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--skip-build] [--soft]"
            echo "  --dry-run     Print actions without executing remote commands"
            echo "  --skip-build  Skip docker build (use existing image)"
            echo "  --soft        Update entrypoint in-place without destroying containers"
            echo "                (preserves cached credentials, skips registration token)"
            exit 0
            ;;
        *)
            echo "[deploy-runners] Unknown argument: ${arg}" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[deploy-runners] $*"; }
warn() { echo "[deploy-runners] WARN: $*" >&2; }
err()  { echo "[deploy-runners] ERROR: $*" >&2; exit 1; }

run_ssh() {
    # Run a command on the runner host via SSH
    # Usage: run_ssh "command string"
    if "${DRY_RUN}"; then
        log "[DRY RUN] ssh ${RUNNER_HOST}: $1"
        return 0
    fi
    ssh "${RUNNER_HOST}" "$1"
}

run_local() {
    # Run a local command (suppressed in dry-run)
    if "${DRY_RUN}"; then
        log "[DRY RUN] local: $*"
        return 0
    fi
    "$@"
}

# ---------------------------------------------------------------------------
# Step 1: Fetch registration token
# ---------------------------------------------------------------------------

fetch_registration_token() {
    log "Fetching GitHub Actions registration token for org ${RUNNER_ORG}..."
    local token
    token=$(gh api "/orgs/${RUNNER_ORG}/actions/runners/registration-token" --jq .token)
    if [[ -z "${token}" ]]; then
        err "Failed to fetch registration token. Check gh auth and org admin permissions."
    fi
    echo "${token}"
}

# ---------------------------------------------------------------------------
# Step 2: Base64-encode token for safe SSH passing
# ---------------------------------------------------------------------------

encode_token() {
    local token="${1}"
    echo -n "${token}" | base64
}

# ---------------------------------------------------------------------------
# Step 3: Rsync artifacts to host
# ---------------------------------------------------------------------------

rsync_artifacts() {
    log "Rsyncing runner artifacts to ${RUNNER_HOST}:${RUNNER_HOST_DIR}/ ..."

    # Ensure remote directory structure exists
    run_ssh "mkdir -p ${RUNNER_HOST_DIR}/docker/runners ${RUNNER_HOST_DIR}/docker"

    if "${DRY_RUN}"; then
        log "[DRY RUN] rsync ${SYNC_PATHS[*]} -> ${RUNNER_HOST}:${RUNNER_HOST_DIR}/"
        return 0
    fi

    # Sync Dockerfile, entrypoint, and monitor into docker/runners/
    rsync -av --checksum \
        "${REPO_ROOT}/docker/runners/Dockerfile" \
        "${REPO_ROOT}/docker/runners/entrypoint.sh" \
        "${REPO_ROOT}/docker/runners/runner-monitor.sh" \
        "${RUNNER_HOST}:${RUNNER_HOST_DIR}/docker/runners/"

    # Sync compose file into docker/
    rsync -av --checksum \
        "${REPO_ROOT}/docker/docker-compose.runners.yml" \
        "${RUNNER_HOST}:${RUNNER_HOST_DIR}/docker/"

    log "Rsync complete."
}

# ---------------------------------------------------------------------------
# Step 4: Deploy via SSH
# ---------------------------------------------------------------------------

deploy_runners() {
    local token_b64="${1}"

    local compose_cmd="docker compose -f ${RUNNER_HOST_DIR}/docker/docker-compose.runners.yml"

    if "${SKIP_BUILD}"; then
        local up_flags="--force-recreate --remove-orphans"
    else
        local up_flags="--build --force-recreate --remove-orphans"
    fi

    log "Deploying runners on ${RUNNER_HOST} (force-recreate ensures fresh env)..."

    # Decode token on remote side to avoid shell metacharacter issues
    run_ssh "
        set -euo pipefail
        RUNNER_TOKEN=\$(echo '${token_b64}' | base64 -d)
        export RUNNER_TOKEN
        cd ${RUNNER_HOST_DIR}
        ${compose_cmd} up -d ${up_flags}
    "

    log "Docker compose deploy complete."
}

# ---------------------------------------------------------------------------
# Step 5: Install prune cron idempotently
# ---------------------------------------------------------------------------

install_prune_cron() {
    log "Installing docker prune cron on ${RUNNER_HOST} (idempotent tee, not append)..."

    # Two weekly prune jobs (Sunday):
    #   03:00 — Build cache prune, only when disk > 70%, retain 14 days (336h)
    #   04:00 — Untagged image prune, retain 14 days (336h)
    # Weekly (not twice-weekly) to avoid cache thrash from Docker builds.
    local cron_content
    cron_content='# Build cache prune (Sunday 03:00) — only when disk > 70%, retain 14 days
0 3 * * 0 root USAGE=$(df --output=pcent /var/lib/docker | tail -1 | tr -d '"'"' %'"'"'); [ "${USAGE:-0}" -ge 70 ] && docker builder prune -f --filter '"'"'until=336h'"'"'
# Untagged image prune (Sunday 04:00) — retain 14 days
0 4 * * 0 root docker image prune -f --filter '"'"'until=336h'"'"''

    run_ssh "echo '${cron_content}' | sudo tee /etc/cron.d/docker-prune > /dev/null && echo '[deploy-runners] Prune cron installed (idempotent).'"
}

# ---------------------------------------------------------------------------
# Step 6: Install runner health monitor cron
# ---------------------------------------------------------------------------
# Deploys the runner-monitor.sh script with a cron that runs every 3 minutes.
# Fires Slack alerts on state transitions (healthy→unhealthy, recovery).
# Requires SLACK_BOT_TOKEN and SLACK_CHANNEL_ID in ~/.omnibase/.env.

install_monitor_cron() {
    log "Installing runner health monitor on ${RUNNER_HOST}..."

    # Source local .env to get Slack credentials
    local slack_bot_token=""
    local slack_channel_id=""
    if [[ -f "${HOME}/.omnibase/.env" ]]; then
        # shellcheck disable=SC1091
        set -a && source "${HOME}/.omnibase/.env" && set +a
        slack_bot_token="${SLACK_BOT_TOKEN:-}"
        slack_channel_id="${SLACK_CHANNEL_ID:-}"
    fi

    if [[ -z "${slack_bot_token}" ]] || [[ -z "${slack_channel_id}" ]]; then
        warn "SLACK_BOT_TOKEN or SLACK_CHANNEL_ID not set in ~/.omnibase/.env"
        warn "Skipping monitor cron install. Monitor script is deployed but cron won't work without credentials."
        return 0
    fi

    # Make monitor executable on remote
    run_ssh "chmod +x ${RUNNER_HOST_DIR}/docker/runners/runner-monitor.sh"

    # Deploy Slack credentials to a separate env file (not in compose or main .env)
    if "${DRY_RUN}"; then
        log "[DRY RUN] Would write .monitor-env with Slack credentials"
    else
        # Write credentials via SSH to avoid them appearing in rsync'd files
        ssh "${RUNNER_HOST}" "cat > ${RUNNER_HOST_DIR}/.monitor-env" <<ENVEOF
SLACK_BOT_TOKEN=${slack_bot_token}
SLACK_CHANNEL_ID=${slack_channel_id}
ENVEOF
        ssh "${RUNNER_HOST}" "chmod 600 ${RUNNER_HOST_DIR}/.monitor-env"
    fi

    # Install cron idempotently: replace any existing runner-monitor line
    local monitor_script="${RUNNER_HOST_DIR}/docker/runners/runner-monitor.sh"
    local monitor_env="${RUNNER_HOST_DIR}/.monitor-env"
    local cron_line="*/3 * * * * set -a && source ${monitor_env} && set +a && ${monitor_script} >> /tmp/runner-monitor.log 2>&1"

    run_ssh "
        EXISTING=\$(crontab -l 2>/dev/null || true)
        echo \"\${EXISTING}\" | grep -v 'runner-monitor' | { cat; echo '${cron_line}'; } | crontab -
    "

    log "Runner health monitor cron installed (every 3 minutes)."
}

# ---------------------------------------------------------------------------
# Step 7: Poll GitHub API until all 10 runners are online and validated
# ---------------------------------------------------------------------------

poll_runners_online() {
    log "Polling GitHub API for ${RUNNER_COUNT} online runners in group '${RUNNER_GROUP}'..."
    log "Max wait: ${POLL_MAX_SECONDS}s, interval: ${POLL_INTERVAL_SECONDS}s"

    if "${DRY_RUN}"; then
        log "[DRY RUN] Would poll until ${RUNNER_COUNT} runners online."
        return 0
    fi

    local elapsed=0
    while true; do
        local online
        online=$(gh api "/orgs/${RUNNER_ORG}/actions/runners" --jq "
          [.runners[] |
           select(.name | startswith(\"${RUNNER_NAME_PREFIX}\")) |
           select(.status == \"online\") |
           select(.runner_group_name == \"${RUNNER_GROUP}\") |
           select(any(.labels[]; .name == \"${RUNNER_GROUP}\"))] | length
        ")

        log "Online runners validated: ${online}/${RUNNER_COUNT} (${elapsed}s elapsed)"

        if [[ "${online}" -ge "${RUNNER_COUNT}" ]]; then
            log "All ${RUNNER_COUNT} runners online and validated."
            return 0
        fi

        if [[ ${elapsed} -ge ${POLL_MAX_SECONDS} ]]; then
            warn "Poll timed out after ${POLL_MAX_SECONDS}s. Only ${online}/${RUNNER_COUNT} runners online."
            return 1
        fi

        sleep "${POLL_INTERVAL_SECONDS}"
        elapsed=$((elapsed + POLL_INTERVAL_SECONDS))
    done
}

# ---------------------------------------------------------------------------
# Soft deploy: update entrypoint in-place, restart containers
# ---------------------------------------------------------------------------
# Preserves container filesystems (and cached runner credentials).
# Use when updating entrypoint logic without needing fresh registration.
#
# Flow: rsync → rebuild image → docker cp entrypoint → docker restart
# Skips: registration token, force-recreate, cron installs

soft_deploy() {
    log "=== Soft deploy (entrypoint-only update) ==="

    rsync_artifacts

    # Rebuild the image so future containers (from force-recreate deploys) use it
    log "Rebuilding runner image on ${RUNNER_HOST}..."
    run_ssh "cd ${RUNNER_HOST_DIR} && docker build -t omninode-runner:latest docker/runners/"

    # Copy the new entrypoint into each running container and restart
    log "Updating entrypoint in ${RUNNER_COUNT} running containers..."
    for i in $(seq 1 "${RUNNER_COUNT}"); do
        local container="${RUNNER_NAME_PREFIX}-${i}"
        log "  Updating ${container}..."

        # docker cp overwrites the file inside the container's writable layer
        run_ssh "docker cp ${RUNNER_HOST_DIR}/docker/runners/entrypoint.sh ${container}:/usr/local/bin/entrypoint.sh"
        run_ssh "docker restart ${container}"
    done

    log "Soft deploy complete. Waiting for runners to come online..."
}

# ---------------------------------------------------------------------------
# Step 8: Retry once with fresh token if poll fails
# ---------------------------------------------------------------------------

deploy_with_retry() {
    local attempt=1

    while true; do
        log "=== Deploy attempt ${attempt} ==="

        local token
        token=$(fetch_registration_token)
        local token_b64
        token_b64=$(encode_token "${token}")

        rsync_artifacts
        deploy_runners "${token_b64}"
        install_prune_cron
        install_monitor_cron

        if poll_runners_online; then
            log "Deploy succeeded on attempt ${attempt}."
            return 0
        fi

        if [[ ${attempt} -ge 2 ]]; then
            err "Deploy failed after ${attempt} attempts. Check runner logs on ${RUNNER_HOST}."
        fi

        warn "Retrying deploy with fresh token (attempt $((attempt + 1)))..."
        attempt=$((attempt + 1))
    done
}

# ---------------------------------------------------------------------------
# Step 9: Stale runner report
# ---------------------------------------------------------------------------
# Reports offline GitHub runners with no matching container on the host.
# ACTION IS MANUAL — do NOT auto-delete runners.
# Reason: GitHub API has no reliable age gate; auto-delete risks removing
# runners that are between jobs or restarting.

print_stale_runner_report() {
    log "=== Stale Runner Report ==="
    log "Checking for offline runners with no matching container..."

    if "${DRY_RUN}"; then
        log "[DRY RUN] Would query GitHub API and host containers for stale runner report."
        return 0
    fi

    # Get all offline runners matching our prefix
    local offline_runners
    offline_runners=$(gh api "/orgs/${RUNNER_ORG}/actions/runners" --jq "
      [.runners[] |
       select(.name | startswith(\"${RUNNER_NAME_PREFIX}\")) |
       select(.status == \"offline\")] |
      .[] | {id: .id, name: .name, status: .status}
    " 2>/dev/null || echo "")

    if [[ -z "${offline_runners}" ]]; then
        log "No offline runners found. All runners appear healthy."
        return 0
    fi

    # Get running containers on the host
    local host_containers
    host_containers=$(run_ssh "docker ps --format '{{.Names}}'" 2>/dev/null || echo "")

    local stale_found=false

    while IFS= read -r runner_json; do
        [[ -z "${runner_json}" ]] && continue
        local runner_id runner_name
        runner_id=$(echo "${runner_json}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['id'])")
        runner_name=$(echo "${runner_json}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['name'])")

        # Check if a matching container exists on the host
        if ! echo "${host_containers}" | grep -q "^${runner_name}$"; then
            stale_found=true
            warn "Stale runner detected: ${runner_name} (id=${runner_id}) — offline, no host container"
            warn "  To remove manually:"
            warn "    gh api /orgs/${RUNNER_ORG}/actions/runners/${runner_id} --method DELETE"
        fi
    done <<< "${offline_runners}"

    if ! "${stale_found}"; then
        log "No stale runners found (all offline runners have matching containers)."
    else
        warn "Stale runners reported above require MANUAL deletion."
        warn "Do not auto-delete: runners may be restarting between jobs."
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    log "Starting deploy-runners.sh (dry_run=${DRY_RUN}, skip_build=${SKIP_BUILD}, soft=${SOFT_DEPLOY})"
    log "Target host: ${RUNNER_HOST} | Org: ${RUNNER_ORG} | Group: ${RUNNER_GROUP}"
    log "Runner count: ${RUNNER_COUNT} | Compose file: ${COMPOSE_FILE}"

    if "${DRY_RUN}"; then
        log "[DRY RUN MODE] No remote commands will be executed."
    fi

    if "${SOFT_DEPLOY}"; then
        soft_deploy
        poll_runners_online || warn "Not all runners came online. Check logs on ${RUNNER_HOST}."
    else
        deploy_with_retry
    fi

    print_stale_runner_report

    log "=== deploy-runners.sh complete ==="
}

main "$@"
