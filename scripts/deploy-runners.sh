#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# deploy-runners.sh
# Single-command deploy entry point for OmniNode self-hosted GitHub Actions runners
# Ticket: OMN-3277 / Epic: OMN-3273
#
# Usage:
#   ./scripts/deploy-runners.sh [--dry-run] [--skip-build] [--soft] [--rolling [--limit=N] [--only=NAME]]
#   ./scripts/deploy-runners.sh --add=<service>[,<service>...] [--dry-run] [--skip-build]
#
# What it does (in order):
#   1. Fetch a fresh GitHub Actions registration token (valid 1 hour)
#   2. Base64-encode token for safe SSH passing
#   3. Rsync runner artifacts to 192.168.86.201:~/.omnibase/runners/
#   4. Build the versioned runner image via scripts/ci/build_runner_image.sh
#   5. Deploy via SSH: docker compose up -d --force-recreate --remove-orphans
#   6. Install docker prune cron idempotently (build cache + untagged images, tee)
#   7. Install runner health monitor cron (Slack alerts on state transitions)
#   8. Poll GitHub API until the configured runner fleet is online
#      (max 5 min, 15s interval)
#   9. Retry once with fresh token if poll times out
#   10. Print stale runner report (offline runners with no host container)
#
# --soft mode (entrypoint-only update, no recreate):
#   1. Rsync runner artifacts to host
#   2. Rebuild the Docker image (for future containers)
#   3. docker cp the new entrypoint into each running container
#   4. docker restart each container (preserves filesystem + cached credentials)
#   Skips: registration token fetch, force-recreate, cron installs
#   Use when: updating entrypoint logic without needing fresh registration
#
# --rolling mode (one runner at a time, busy-checked, fail-closed):
#   1. Rsync runner artifacts to host
#   2. For each fleet service in turn: confirm it is idle on TWO independent
#      signals, recreate that ONE service, wait for it to come back online
#   3. Skip (and retry later) any runner executing a job; HALT if a recreated
#      runner does not return
#   Skips: image build, cron installs, --remove-orphans
#   Use when: a container-env change must reach live runners. Env is frozen at
#   container creation, so a recreate is the only way; --soft cannot carry one.
#   Cost: roughly 60-100s per runner, serial by nature (see the mode's own
#   comment block below).
#   --limit=N stops after N successful recreates: --limit=1 is the canary
#   step, proven before the rest of the fleet is touched.
#   --only=NAME converges exactly one fleet service. A runner that is busy
#   through every retry pass is reported and left alone, which is correct --
#   but without a way to come back for it later, the residual would have to be
#   cleared by a hand-typed compose call, i.e. by the exact recipe this mode
#   exists to replace.
#   With --dry-run the READ-ONLY busy probes still run -- a rolling dry run is
#   how you see which runners are executing jobs -- but nothing is recreated.
#
# --add=<service>[,...] mode (additive stand-up of declared NON-pool services):
#   1. Rsync runner artifacts to host (the compose file must reach the host
#      before any service can be created from it)
#   2. Build the image unless --skip-build
#   3. docker compose up -d --no-deps <service>...  -- with NEITHER
#      --force-recreate NOR --remove-orphans
#   4. Wait for exactly those runners to come online
#   Skips: the fleet-wide poll, every cron install, the stale report.
#   Use when: standing up or converging ONE non-general-pool runner
#   (omninode-deploy-runner, omninode-verify-runner-1, the customer-plane pair).
#
#   WHY A FOURTH MODE (OMN-18408). None of the three above can create a
#   container that does not exist yet without collateral. The default path
#   force-recreates the WHOLE project: ~3h serialised across the 88-runner
#   fleet (OMN-18188), with the monitor's auto-bounce cron racing anything left
#   offline past 300s. --soft never creates a container at all. And --rolling
#   --only=NAME refuses this twice by design -- fleet_services() enumerates only
#   ${RUNNER_NAME_PREFIX}-N, and roll_one_runner() requires the target to be
#   online and idle first, which a container that does not exist can never be.
#   Those refusals are correct and are NOT relaxed; this is a separate verb.
#
#   Additive by construction: no --force-recreate, so running it against an
#   already-converged runner is a no-op rather than an outage, and no
#   --remove-orphans, which compose evaluates against the whole project even
#   when services are named.
#
#   REFUSED, deliberately: a general-pool ${RUNNER_NAME_PREFIX}-N service (those
#   belong to the fleet and --rolling paths, which bracket recreates with the
#   toolcache seeding and the two-signal busy check this mode has neither of);
#   a service the compose file does not declare (fail closed on a typo BEFORE
#   the rsync touches the host); and --add combined with --soft or --rolling.
#
# Requirements:
#   - gh CLI authenticated with org admin scope
#   - SSH access to 192.168.86.201 (key-based, no password prompts)
#   - rsync installed locally
#
# See also: docker/runners/Dockerfile, docker/docker-compose.runners.yml,
# scripts/ci/build_runner_image.sh
#
# Trap for on-host operators (OMN-15142): this script is written to run from
# an OPERATOR WORKSTATION that SSHes into the runner host (see run_ssh/
# run_local below). It cannot be run directly ON the runner host targeting
# itself via its own Tailscale MagicDNS hostname -- self-ssh fails "Host key
# verification failed" (no known_hosts entry for the box's own hostname).
# Run the rsync/build/compose steps directly on-host instead in that case.
#
# CONSOLIDATED REQUIRED ENV VARS for docker-compose.runners.yml's
# omninode-deploy-runner service (OMN-15142 -- previously undocumented in one
# place; compose interpolates the ENTIRE file before selecting services, so
# ANY `docker compose -f docker-compose.runners.yml ...` invocation --
# including one scoped to the shared omninode-runner-N fleet -- fails
# closed if these are unset). This script does NOT set any of them; it only
# handles RUNNER_TOKEN for the shared fleet. Anyone standing up
# omninode-deploy-runner fresh (or running any compose command against this
# file at all) must export all three first:
#   - DEPLOY_RUNNER_OMNI_HOME       Private, runner-uid-owned OMNI_HOME clone
#                                   tree (e.g. /data/omninode/runner_omni_home).
#                                   Fail-fast `:?` in compose.
#   - DEPLOY_RUNNER_TOKEN           ORG registration token for OmniNode-ai
#                                   (mint via
#                                   `gh api -X POST orgs/OmniNode-ai/actions/runners/registration-token`,
#                                   valid 1h). OMN-18386 moved this runner from
#                                   a repository-scoped registration on
#                                   omnibase_infra to an org-scoped one in the
#                                   `omnibase-deploy` runner group, so a REPO
#                                   token no longer works here.
#                                   NOT `:?`-guarded in compose --
#                                   an unset value silently becomes an empty
#                                   RUNNER_TOKEN and fails later at container
#                                   registration, not at `up -d` interpolation
#                                   time like the other two.
#   - DEPLOY_RUNNER_OPERATOR_ENV_FILE  Host path of the operator env file
#                                   (e.g. /home/<operator>/.omnibase/.env).
#                                   Fail-fast `:?` in compose.
#
# FOURTH AND FIFTH REQUIRED VARS, satisfied by a FILE rather than an export
# (OMN-18415). Both belong to the same call path and both fail closed: the
# reviewer signs every local LLM request and refuses to send one outside a
# declared trust boundary, so provisioning either alone just moves the failure
# a few seconds later with the same empty log.
#   - LLM_ENDPOINT_CIDR_ALLOWLIST   Network boundary for local LLM calls (not a
#                                   secret). The transport refuses to default
#                                   it -- a boundary nobody declared is not a
#                                   boundary -- so compose refuses too.
#   - LOCAL_LLM_SHARED_SECRET       HMAC signing key for the local LLM
#                                   inference endpoint, consumed by the
#                                   omninode-runner-N fleet (the Hostile
#                                   Review Gate's reviewer fails closed
#                                   without it). Fail-fast `:?` in compose,
#                                   like the two above -- but nothing needs to
#                                   export it, because compose reads it from
#                                   `docker/.env` in the compose PROJECT
#                                   DIRECTORY (the directory of the first `-f`
#                                   file), which resolves the same way for the
#                                   runner-monitor auto-bounce cron as for
#                                   this script. That file is host-generated,
#                                   mode 600, and is deliberately absent from
#                                   SYNC_PATHS below so an rsync from this
#                                   repo can never overwrite or blank it --
#                                   the same rule the lab credentials
#                                   directory follows.
# See knowledge-base-internal:runbooks/omnibase-infra-release-train-lab.md for the full recreate procedure.

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER_FLEET_CONFIG="${RUNNER_FLEET_CONFIG_PATH:-${RUNNER_FLEET_CONFIG:-${REPO_ROOT}/config/runner_fleet.yaml}}"

runner_config_field() {
    local field="${1}"
    [[ -f "${RUNNER_FLEET_CONFIG}" ]] || {
        echo "[deploy-runners] ERROR: runner fleet config not found: ${RUNNER_FLEET_CONFIG}" >&2
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
    ' "${RUNNER_FLEET_CONFIG}") || {
        echo "[deploy-runners] ERROR: missing ${field} in ${RUNNER_FLEET_CONFIG}" >&2
        exit 1
    }
    echo "${value}"
}

RUNNER_HOST="$(runner_config_field runner_host)"
# Remote path on the CI host — NOT local $HOME (which differs between macOS and Linux)
RUNNER_HOST_DIR="/home/jonah/.omnibase/runners"
RUNNER_ORG="$(runner_config_field github_org)"
RUNNER_GROUP="$(runner_config_field runner_group)"
RUNNER_NAME_PREFIX="$(runner_config_field runner_name_prefix)"
RUNNER_COUNT="$(runner_config_field expected_count)"
COMPOSE_FILE="docker/docker-compose.runners.yml"
POLL_MAX_SECONDS=300
POLL_INTERVAL_SECONDS=15

# Artifacts to sync to the host (relative to repo root).
#
# NOTE (OMN-15114 follow-up): scripts/ci/check_runner_host_artifact_freshness.py
# is deliberately NOT in this list. That checker's whole job is to diff a
# local (assumed-current) repo checkout against its rsynced copy on the
# runner host over ssh -- it is installed as a LOCAL cron on the
# operator/dev machine (see install_host_artifact_freshness_cron below,
# "same model as step 10/11") and never executes on the runner host itself.
# Adding it here made the checker report its own absence as drift on every
# host that had not yet received it -- a self-referential false positive
# fixed by removal, not by syncing it. scripts/ci/check_runner_fleet_image_drift.py
# is different and correctly stays synced: it docker-execs into containers
# running ON the host, so it must exist there.
SYNC_PATHS=(
    "${RUNNER_FLEET_CONFIG}"
    ".github/actions/setup-python-uv/action.yml"
    "pyproject.toml"
    "uv.lock"
    "docker/runners/Dockerfile"
    "docker/runners/runner-image.lock.json"
    "docker/runners/entrypoint.sh"
    "docker/runners/runner-job-started.sh"
    "docker/runners/runner-monitor.sh"
    "docker/runners/healthcheck.sh"
    "docker/runners/model-review-healthcheck.sh"
    "docker/runners/model-review-observation.json"
    # OMN-15142: docker/runners/Dockerfile does `COPY omni-curl
    # /usr/local/bin/omni-curl` -- both the built binary shim and its source
    # script must be synced or a rebuild against a fresh/empty deployment dir
    # fails with "/omni-curl": not found at the COPY layer.
    "docker/runners/omni-curl"
    "docker/runners/omni-curl.sh"
    "docker/docker-compose.runners.yml"
    "docker/docker-compose.model-review-canary.yml"
    "docker/compose-overrides.list"
    "scripts/ci/build_runner_image.sh"
    "scripts/ci/ci_env_digest.py"
    "scripts/ci/ensure_ci_env.sh"
    "scripts/ci/runner_image_identity.py"
    "scripts/ci/check_runner_fleet_image_drift.py"
)

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

DRY_RUN=false
SKIP_BUILD=false
SOFT_DEPLOY=false
ROLLING_DEPLOY=false
# 0 means the whole fleet; --limit=N stops after N successful recreates.
ROLL_LIMIT=0
# Empty means the whole fleet; --only=<service> converges exactly one runner.
ROLL_ONLY=""
# OMN-18408. Space-separated, declared, NON-general-pool compose services to
# stand up additively. Empty means this mode is off.
ADD_SERVICES=""

for arg in "$@"; do
    case "${arg}" in
        --dry-run)    DRY_RUN=true ;;
        --skip-build) SKIP_BUILD=true ;;
        --soft)       SOFT_DEPLOY=true ;;
        --rolling)    ROLLING_DEPLOY=true ;;
        --limit=*)    ROLL_LIMIT="${arg#*=}" ;;
        --only=*)     ROLL_ONLY="${arg#*=}" ;;
        --add=*)      ADD_SERVICES="${arg#*=}" ; ADD_SERVICES="${ADD_SERVICES//,/ }" ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--skip-build] [--soft] [--rolling]"
            echo "  --dry-run     Print actions without executing remote commands"
            echo "  --skip-build  Skip docker build (use existing image)"
            echo "  --soft        Update entrypoint in-place without destroying containers"
            echo "                (preserves cached credentials, skips registration token)"
            echo "  --rolling     Recreate the fleet ONE runner at a time, skipping any"
            echo "                runner executing a job. The only supported way to apply"
            echo "                a container-env change (env is frozen at creation)."
            echo "  --limit=N     With --rolling: stop after N successful recreates."
            echo "                --limit=1 is the canary step of a fleet roll."
            echo "  --only=NAME   With --rolling: converge exactly one fleet service,"
            echo "                for a runner that stayed busy through every pass."
            echo "  --add=LIST    Additively stand up the named DECLARED non-general-pool"
            echo "                services: 'up -d --no-deps LIST' with NEITHER"
            echo "                --force-recreate NOR --remove-orphans, no cron installs,"
            echo "                and a wait scoped to those runners. Refuses a"
            echo "                ${RUNNER_NAME_PREFIX}-N service (use --rolling), an"
            echo "                undeclared service, and --soft/--rolling."
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
    log "Fetching GitHub Actions registration token for org ${RUNNER_ORG}..." >&2
    # Separate declaration from assignment so set -e catches gh api failures
    local token
    token=$(gh api --method POST "/orgs/${RUNNER_ORG}/actions/runners/registration-token" --jq .token) \
        || err "gh api failed. Check gh auth and org admin permissions (need admin:org scope)."
    if [[ -z "${token}" ]]; then
        err "Registration token is empty. Check gh auth and org admin permissions."
    fi
    echo "${token}"
}

# ---------------------------------------------------------------------------
# Step 2: Base64-encode token for safe SSH passing
# ---------------------------------------------------------------------------

encode_token() {
    local token="${1}"
    # -w 0 prevents line wrapping (macOS base64 wraps at 76 chars by default)
    echo -n "${token}" | base64 -w 0 2>/dev/null || echo -n "${token}" | base64 -b 0 2>/dev/null || echo -n "${token}" | base64
}

# ---------------------------------------------------------------------------
# Step 3: Rsync artifacts to host
# ---------------------------------------------------------------------------

rsync_artifacts() {
    log "Rsyncing runner artifacts to ${RUNNER_HOST}:${RUNNER_HOST_DIR}/ ..."

    # Ensure remote directory structure exists
    run_ssh "mkdir -p ${RUNNER_HOST_DIR}/config ${RUNNER_HOST_DIR}/docker/runners ${RUNNER_HOST_DIR}/docker ${RUNNER_HOST_DIR}/scripts/ci ${RUNNER_HOST_DIR}/.github/actions/setup-python-uv"

    if "${DRY_RUN}"; then
        log "[DRY RUN] rsync ${SYNC_PATHS[*]} -> ${RUNNER_HOST}:${RUNNER_HOST_DIR}/"
        return 0
    fi

    # Sync fleet config, versioned image contract inputs, Dockerfile, entrypoint,
    # and monitor. The runner image build is contract-bound; do not rely on
    # whatever files happen to be present on the host from an earlier deploy.
    rsync -av --checksum \
        "${RUNNER_FLEET_CONFIG}" \
        "${RUNNER_HOST}:${RUNNER_HOST_DIR}/config/runner_fleet.yaml"

    rsync -av --checksum \
        "${REPO_ROOT}/pyproject.toml" \
        "${REPO_ROOT}/uv.lock" \
        "${RUNNER_HOST}:${RUNNER_HOST_DIR}/"

    rsync -av --checksum \
        "${REPO_ROOT}/.github/actions/setup-python-uv/action.yml" \
        "${RUNNER_HOST}:${RUNNER_HOST_DIR}/.github/actions/setup-python-uv/action.yml"

    rsync -av --checksum \
        "${REPO_ROOT}/docker/runners/Dockerfile" \
        "${REPO_ROOT}/docker/runners/runner-image.lock.json" \
        "${REPO_ROOT}/docker/runners/entrypoint.sh" \
        "${REPO_ROOT}/docker/runners/runner-job-started.sh" \
        "${REPO_ROOT}/docker/runners/runner-monitor.sh" \
        "${REPO_ROOT}/docker/runners/healthcheck.sh" \
        "${REPO_ROOT}/docker/runners/model-review-healthcheck.sh" \
        "${REPO_ROOT}/docker/runners/omni-curl" \
        "${REPO_ROOT}/docker/runners/omni-curl.sh" \
        "${RUNNER_HOST}:${RUNNER_HOST_DIR}/docker/runners/"

    rsync -av --checksum \
        "${REPO_ROOT}/scripts/ci/build_runner_image.sh" \
        "${REPO_ROOT}/scripts/ci/ci_env_digest.py" \
        "${REPO_ROOT}/scripts/ci/ensure_ci_env.sh" \
        "${REPO_ROOT}/scripts/ci/runner_image_identity.py" \
        "${RUNNER_HOST}:${RUNNER_HOST_DIR}/scripts/ci/"

    # Sync compose file into docker/
    rsync -av --checksum \
        "${REPO_ROOT}/docker/docker-compose.runners.yml" \
        "${REPO_ROOT}/docker/docker-compose.model-review-canary.yml" \
        "${REPO_ROOT}/docker/compose-overrides.list" \
        "${RUNNER_HOST}:${RUNNER_HOST_DIR}/docker/"

    log "Rsync complete."
}

# ---------------------------------------------------------------------------
# Step 4: Deploy via SSH
# ---------------------------------------------------------------------------

deploy_runners() {
    local token_b64="${1}"
    local remote_token_b64="${token_b64}"

    # Keep the inactive candidate overlay on the normal deploy path as well as
    # the monitor repair path. Its defaults preserve the generic fleet, while
    # dropping the file here would make a later authorized activation vanish on
    # the next force-recreate.
    local compose_cmd="docker compose -f ${RUNNER_HOST_DIR}/docker/docker-compose.runners.yml -f ${RUNNER_HOST_DIR}/docker/docker-compose.model-review-canary.yml"

    local up_flags="--force-recreate --remove-orphans"

    log "Deploying runners on ${RUNNER_HOST} (force-recreate ensures fresh env)..."

    if "${DRY_RUN}"; then
        remote_token_b64="<redacted-token-b64>"
    fi

    # Decode token on remote side to avoid shell metacharacter issues
    run_ssh "
        set -euo pipefail
        RUNNER_TOKEN=\$(echo '${remote_token_b64}' | base64 -d)
        export RUNNER_TOKEN
        cd ${RUNNER_HOST_DIR}
        if [ '${SKIP_BUILD}' = 'false' ]; then
            bash scripts/ci/build_runner_image.sh --tag omninode-runner:latest
        fi
        ${compose_cmd} up -d ${up_flags}
    "

    log "Docker compose deploy complete."
}

# ---------------------------------------------------------------------------
# Step 5: Install prune cron idempotently
# ---------------------------------------------------------------------------

install_prune_cron() {
    log "Skipping legacy docker-prune cron install."
    log "Host cleanup is owned by deploy/disk-gc/install-host-maintenance.sh."
    log "Run that installer on ${RUNNER_HOST} to install onex-disk-gc and onex-worktree-reaper timers."
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
    local runner_github_token="${RUNNER_GITHUB_TOKEN:-${GH_PAT:-${GITHUB_TOKEN:-}}}"
    if [[ -f "${HOME}/.omnibase/.env" ]]; then
        # shellcheck disable=SC1091
        set +u
        set -a
        source "${HOME}/.omnibase/.env"
        set +a
        set -u
        slack_bot_token="${SLACK_BOT_TOKEN:-}"
        slack_channel_id="${SLACK_CHANNEL_ID:-}"
        runner_github_token="${RUNNER_GITHUB_TOKEN:-${GH_PAT:-${GITHUB_TOKEN:-${runner_github_token}}}}"
    fi

    if [[ -z "${slack_bot_token}" ]] || [[ -z "${slack_channel_id}" ]]; then
        warn "SLACK_BOT_TOKEN or SLACK_CHANNEL_ID not set in ~/.omnibase/.env"
        warn "Skipping monitor cron install. Monitor script is deployed but cron won't work without credentials."
        return 0
    fi
    if [[ -z "${runner_github_token}" ]]; then
        runner_github_token="$(gh auth token 2>/dev/null || true)"
    fi
    if [[ -z "${runner_github_token}" ]]; then
        warn "RUNNER_GITHUB_TOKEN/GH_PAT/GITHUB_TOKEN not set and gh auth token unavailable"
        warn "Skipping monitor cron install. GitHub-aware monitor requires org runner API access."
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
RUNNER_GITHUB_TOKEN=${runner_github_token}
RUNNER_FLEET_CONFIG_PATH=${RUNNER_HOST_DIR}/config/runner_fleet.yaml
ENVEOF
        ssh "${RUNNER_HOST}" "chmod 600 ${RUNNER_HOST_DIR}/.monitor-env"
    fi

    # Install cron idempotently: replace any existing runner monitor/repair line
    local monitor_script="${RUNNER_HOST_DIR}/docker/runners/runner-monitor.sh"
    local monitor_env="${RUNNER_HOST_DIR}/.monitor-env"
    # Cron uses /bin/sh by default on the runner host; bare `source` fails there
    # before credentials load, silently disabling Slack alerts when no MTA exists.
    # Force bash and redirect the whole monitor invocation so setup failures are
    # visible in /tmp/runner-monitor.log.
    local monitor_cron_line="*/3 * * * * /bin/bash -lc 'set -a; source ${monitor_env}; set +a; ${monitor_script}' >> /tmp/runner-monitor.log 2>&1 # runner-monitor-alert"
    local repair_cron_line="*/10 * * * * /bin/bash -lc 'set -a; source ${monitor_env}; set +a; MONITOR_AUTO_BOUNCE=1 OFFLINE_IDLE_RECREATE_AGE_SECONDS=600 ${monitor_script}' >> /tmp/runner-repair.log 2>&1 # runner-repair-check"

    run_ssh "
        EXISTING=\$(crontab -l 2>/dev/null || true)
        echo \"\${EXISTING}\" | grep -Ev 'runner-monitor|runner-repair-check' | { cat; echo '${monitor_cron_line}'; echo '${repair_cron_line}'; } | crontab -
    "

    log "Runner health monitor cron installed (alerts every 3 minutes, repair every 10 minutes)."
}

# ---------------------------------------------------------------------------
# Step 7: Poll GitHub API until the configured runner fleet is online
# and validated
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
        online=$(
            gh api --paginate "/orgs/${RUNNER_ORG}/actions/runners?per_page=100" |
                jq -s --arg prefix "${RUNNER_NAME_PREFIX}" --arg group "${RUNNER_GROUP}" '
                  [.[].runners[] |
                   select(.name | startswith($prefix)) |
                   select(.status == "online") |
                   select(any(.labels[]; .name == $group))] | length
                '
        )

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

    # Rebuild the image so future containers (from force-recreate deploys) use it.
    log "Rebuilding versioned runner image on ${RUNNER_HOST}..."
    run_ssh "cd ${RUNNER_HOST_DIR} && bash scripts/ci/build_runner_image.sh --tag omninode-runner:latest"

    # Ensure entrypoint is executable on host (docker cp preserves source permissions)
    run_ssh "chmod +x ${RUNNER_HOST_DIR}/docker/runners/entrypoint.sh"

    # Copy the new entrypoint into each container and restart.
    # docker cp works on both running and stopped containers.
    # docker stop first to avoid the container restarting mid-copy.
    log "Updating entrypoint in ${RUNNER_COUNT} containers..."
    for i in $(seq 1 "${RUNNER_COUNT}"); do
        local container="${RUNNER_NAME_PREFIX}-${i}"
        log "  Updating ${container}..."
        run_ssh "docker stop ${container} 2>/dev/null || true"
        run_ssh "docker cp ${RUNNER_HOST_DIR}/docker/runners/entrypoint.sh ${container}:/usr/local/bin/entrypoint.sh"
        run_ssh "docker start ${container}"
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
        if "${DRY_RUN}"; then
            token="dry-run-token"
        else
            token=$(fetch_registration_token)
        fi
        local token_b64
        token_b64=$(encode_token "${token}")

        rsync_artifacts
        deploy_runners "${token_b64}"
        install_prune_cron
        install_monitor_cron
        install_health_cron
        install_network_janitor_cron
        install_host_artifact_freshness_cron

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
    offline_runners=$(
        gh api --paginate "/orgs/${RUNNER_ORG}/actions/runners?per_page=100" |
            jq -s --arg prefix "${RUNNER_NAME_PREFIX}" '
              [.[].runners[] |
               select(.name | startswith($prefix)) |
               select(.status == "offline")] |
              .[] | {id: .id, name: .name, status: .status}
            ' 2>/dev/null || echo ""
    )

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
# Step 10: Install local runner health check cron (OMN-6083)
# ---------------------------------------------------------------------------
# Installs a LOCAL cron (on this dev machine) that runs the runner health
# CLI every 3 minutes. The CLI SSHes to RUNNER_HOST for Docker status.
# Uses marker-based idempotence: exactly one entry managed via
# '# runner-health-check' comment, never clobbers unrelated cron entries.

install_health_cron() {
    log "Installing LOCAL runner health check cron..."

    if "${DRY_RUN}"; then
        log "[DRY RUN] Would install local runner-health-check cron (every 3 min)."
        return 0
    fi

    # Determine the repo root (this script lives in scripts/)
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    local cron_line="*/3 * * * * set -a && . ~/.omnibase/.env && set +a && cd ${repo_root} && PYTHONPATH=${repo_root}/src RUNNER_FLEET_CONFIG_PATH=${RUNNER_FLEET_CONFIG} RUNNER_HEALTH_HOST=${RUNNER_HOST} uv run python -m omnibase_infra.observability.runner_health.cli_runner_health --emit --alert >> /tmp/runner-health.log 2>&1 # runner-health-check"

    # Filter out any existing runner-health-check line, then append new one
    local existing
    existing=$(crontab -l 2>/dev/null || true)
    echo "${existing}" | grep -v 'runner-health-check' | { cat; echo "${cron_line}"; } | crontab -

    log "Runner health check cron installed locally (every 3 minutes)."
}

# ---------------------------------------------------------------------------
# Step 11: Install bounded Docker network janitor cron (OMN-12566)
# ---------------------------------------------------------------------------
# Installs a LOCAL cron (on this dev machine) that runs the bounded Docker
# network janitor + subnet-pool collection every 15 minutes. The CLI SSHes to
# RUNNER_HOST to inspect networks, classify them against the declared
# ownership contract, emit pool occupancy to the durable network-pool-status
# topic, and alert before subnet-pool exhaustion.
#
# Reclaim default: DRY-RUN. The cron runs WITHOUT --reclaim so the steady
# state is observe-and-alert only. Enabling destructive reclaim on the live
# fleet (adding --reclaim) is a deliberate, separately-approved live step —
# not flipped on automatically by deploy.
#
# Marker-based idempotence via '# network-janitor-check'.

install_network_janitor_cron() {
    log "Installing LOCAL Docker network janitor cron (dry-run)..."

    if "${DRY_RUN}"; then
        log "[DRY RUN] Would install local network-janitor-check cron (every 15 min, dry-run)."
        return 0
    fi

    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    # NOTE: no --reclaim — observe + alert only. Live reclaim is gated.
    local cron_line="*/15 * * * * set -a && . ~/.omnibase/.env && set +a && cd ${repo_root} && PYTHONPATH=${repo_root}/src RUNNER_FLEET_CONFIG_PATH=${RUNNER_FLEET_CONFIG} RUNNER_HEALTH_HOST=${RUNNER_HOST} uv run python -m omnibase_infra.observability.runner_health.cli_runner_health --network --emit --alert >> /tmp/network-janitor.log 2>&1 # network-janitor-check"

    local existing
    existing=$(crontab -l 2>/dev/null || true)
    echo "${existing}" | grep -v 'network-janitor-check' | { cat; echo "${cron_line}"; } | crontab -

    log "Network janitor cron installed locally (every 15 minutes, dry-run)."
}

# ---------------------------------------------------------------------------
# Step 12: Install local runner host artifact-freshness cron (OMN-15114)
# ---------------------------------------------------------------------------
# Installs a LOCAL cron (on this dev machine, same model as step 10/11) that
# runs scripts/ci/check_runner_host_artifact_freshness.py every 30 minutes.
# That script diffs every SYNC_PATHS entry (sha256) between this repo
# checkout and RUNNER_HOST's rsynced copy under RUNNER_HOST_DIR, and reports
# any path that has drifted.
#
# Root cause this closes: the SYNC_PATHS rsync above only runs as part of
# this script's full (disruptive, force-recreate-all) pipeline, which
# operators routinely avoid for a small fix by rebuilding the image and
# recreating containers directly on the host instead. That leaves the
# host-staged checkout free to drift indefinitely with zero signal -- the
# checked-in runner-image.lock.json sat at image_version 5 on the host for
# 19 days after origin/dev moved to image_version 6 (OMN-15114) before
# anything noticed, exactly the "merged fix silently never lands on the
# artifact that matters" defect class OMN-15104 exists to close, one layer
# further out (host checkout, not container).
#
# Marker-based idempotence via '# runner-host-artifact-freshness-check'.

install_host_artifact_freshness_cron() {
    log "Installing LOCAL runner host artifact-freshness cron..."

    if "${DRY_RUN}"; then
        log "[DRY RUN] Would install local runner-host-artifact-freshness-check cron (every 30 min)."
        return 0
    fi

    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    local cron_line="*/30 * * * * cd ${repo_root} && uv run python scripts/ci/check_runner_host_artifact_freshness.py --runner-host ${RUNNER_HOST} --runner-host-dir ${RUNNER_HOST_DIR} >> /tmp/runner-host-artifact-freshness.log 2>&1 # runner-host-artifact-freshness-check"

    local existing
    existing=$(crontab -l 2>/dev/null || true)
    echo "${existing}" | grep -v 'runner-host-artifact-freshness-check' | { cat; echo "${cron_line}"; } | crontab -

    log "Runner host artifact-freshness cron installed locally (every 30 minutes)."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rolling deploy: recreate the fleet one runner at a time, never a busy one
# ---------------------------------------------------------------------------
# Additive stand-up of declared non-general-pool services (OMN-18408)
# ---------------------------------------------------------------------------

declared_compose_services() {
    # Every `  <name>:` service key in the compose file. Derived from the file
    # rather than from a list here, so a service added to compose is usable
    # immediately and a typo is refused against reality.
    grep -E '^  [a-z0-9][a-z0-9_.-]*:$' "${REPO_ROOT}/${COMPOSE_FILE}" | tr -d ' :'
}

validate_add_services() {
    local declared name
    declared="$(declared_compose_services)"
    for name in ${ADD_SERVICES}; do
        if [[ "${name}" =~ ^${RUNNER_NAME_PREFIX}-[0-9]+$ ]]; then
            err "--add refuses the general-pool service '${name}'. General-pool runners belong to the default path and to --rolling, which bracket a recreate with the toolcache seeding and the two-signal busy check --add has neither of. Use --rolling --only=${name}."
        fi
        if ! printf '%s\n' "${declared}" | grep -qx "${name}"; then
            err "--add: '${name}' is not a service declared in ${COMPOSE_FILE}. Refusing locally so a typo never reaches the host."
        fi
    done
}

add_services_deploy() {
    log "=== Additive deploy of: ${ADD_SERVICES} ==="
    validate_add_services

    local reg_handle token_b64
    if "${DRY_RUN}"; then
        reg_handle="dry-run-handle"
    else
        # A brand-new container has NO cached registration to restore from, so
        # unlike --rolling this path does need a real handle.
        reg_handle="$(fetch_registration_token)"
    fi
    token_b64=$(encode_token "${reg_handle}")

    rsync_artifacts

    if ! "${SKIP_BUILD}"; then
        run_ssh "
            set -euo pipefail
            cd ${RUNNER_HOST_DIR}
            bash scripts/ci/build_runner_image.sh --tag omninode-runner:latest
        "
    fi

    local compose_cmd="docker compose -f ${RUNNER_HOST_DIR}/docker/docker-compose.runners.yml -f ${RUNNER_HOST_DIR}/docker/docker-compose.model-review-canary.yml"
    local remote_token_b64="${token_b64}"
    if "${DRY_RUN}"; then
        remote_token_b64="<redacted-token-b64>"
    fi

    # No --force-recreate: additive means a converged container is left running.
    # No --remove-orphans: compose evaluates it against the WHOLE project even
    # when services are named, so it would delete containers this call never
    # looked at.
    # CONVERGE, per service, on the runner's LIVE registration state.
    #
    # `up -d` alone does NOT fix a container that was created with a bad
    # registration handle: compose finds the container present, starts it, and
    # reports success while the runner re-runs the same failing registration in
    # a restart loop. That is exactly what happened standing this runner up --
    # the container held a 14-character value where a registration token is 29,
    # and two successive `--add` runs both reported "Started" and changed
    # nothing, because neither recreated it.
    #
    # So an OFFLINE runner is recreated and an ONLINE one is never touched:
    #
    #   online  -> plain `up -d`. A registered runner may be mid-job, and its
    #              cached credentials are the thing a recreate would discard.
    #   not online -> `--force-recreate`. A runner that is not registered holds
    #              nothing worth preserving and takes no jobs, so recreating it
    #              costs nothing and is the only thing that re-reads the env.
    #
    # `unknown` counts as not-online, deliberately: a container that does not
    # exist yet reports exactly that, and so does one whose registration failed.
    # Both want the same treatment. This is the one place `--add` is allowed to
    # recreate, and it can never reach a general-pool runner -- those are
    # refused at flag-validation time, before this runs.
    local name state flags
    for name in ${ADD_SERVICES}; do
        flags="--no-deps"
        if "${DRY_RUN}"; then
            state="$(github_runner_state "${name}" 2>/dev/null || echo "unknown unknown")"
        else
            state="$(github_runner_state "${name}")"
        fi
        if [[ "${state%% *}" == "online" ]]; then
            log "  ${name} is already registered and online -- converging without recreate."
        else
            log "  ${name} is not registered (state: ${state%% *}) -- recreating so it re-reads its env."
            flags="${flags} --force-recreate"
        fi

        run_ssh "
        set -euo pipefail
        RUNNER_TOKEN=\$(echo '${remote_token_b64}' | base64 -d)
        export RUNNER_TOKEN
        # The non-general-pool services interpolate DEPLOY_RUNNER_TOKEN, not
        # RUNNER_TOKEN. Both are ORG registration tokens for the same org.
        DEPLOY_RUNNER_TOKEN=\"\${RUNNER_TOKEN}\"
        export DEPLOY_RUNNER_TOKEN
        cd ${RUNNER_HOST_DIR}
        ${compose_cmd} up -d ${flags} ${name}
    "
    done

    # Service name == container_name == RUNNER_NAME for every non-pool runner
    # in the compose file, which is what makes this an identity, not a lookup.
    local name rc=0
    for name in ${ADD_SERVICES}; do
        if "${DRY_RUN}"; then
            log "[DRY RUN] would wait for ${name} to come online."
            continue
        fi
        if wait_for_runner_online "${name}"; then
            log "  ${name} is online."
        else
            warn "  ${name} did not come online within ${ROLL_ONLINE_MAX_SECONDS}s."
            rc=1
        fi
    done
    return "${rc}"
}

# ---------------------------------------------------------------------------
#
# WHY THIS MODE EXISTS (OMN-18415). Container ENV is frozen at creation, so any
# change to the `&runner-env` block reaches a live runner only through a
# RECREATE. The default path above recreates every service in one compose call,
# which takes the whole fleet down; `--soft` does not recreate at all and so
# cannot carry an env change. Neither is usable for an env roll, and the
# procedure that is -- one service at a time, skipping runners that are
# executing someone else's job -- was until now a hand-typed loop, which is
# exactly the shape that produces a killed live job when somebody's busy check
# is subtly fail-open.
#
# WHAT IT COSTS. Roughly 60-100s per runner (stop, create, start, then the
# listener's reconnect to GitHub), measured over two full rolls. Batching buys
# nothing: `docker compose up -d --force-recreate` serialises container stop and
# start regardless of how many services are named, so a batch only puts more
# runners down at once. Serial keeps the rest of the fleet available throughout.
#
# THE BUSY CHECK FAILS CLOSED, and that is the load-bearing property. Two
# independent signals are consulted -- GitHub's own `busy` flag (authoritative)
# and the container's process list -- and a runner is touched only when BOTH
# say idle. An API error, a missing runner, a failed `docker top`, or any
# disagreement between the two yields UNKNOWN, and UNKNOWN skips. A fail-open
# `docker top` check is what killed a live job once already.
#
# A runner that does not come back online HALTS the roll rather than cascading:
# the remaining runners keep the old env, which is a partial roll, not an
# outage.
ROLL_ONLINE_MAX_SECONDS="${ROLL_ONLINE_MAX_SECONDS:-240}"
ROLL_ONLINE_INTERVAL_SECONDS="${ROLL_ONLINE_INTERVAL_SECONDS:-10}"
ROLL_SKIP_RETRY_PASSES="${ROLL_SKIP_RETRY_PASSES:-2}"

fleet_services() {
    # Enumerate fleet services from the compose file itself, then cross-check
    # the count against config/runner_fleet.yaml. Deriving the list from
    # `seq 1 ${RUNNER_COUNT}` would silently invent service names if the two
    # ever disagreed; this fails closed instead.
    local names
    names=$(grep -E "^  ${RUNNER_NAME_PREFIX}-[0-9]+:$" "${REPO_ROOT}/${COMPOSE_FILE}" \
        | tr -d ' :' || true)
    local count
    count=$(printf '%s\n' "${names}" | grep -c . || true)
    if [[ "${count}" -ne "${RUNNER_COUNT}" ]]; then
        err "compose declares ${count} ${RUNNER_NAME_PREFIX}-N services but config/runner_fleet.yaml expects ${RUNNER_COUNT}; refusing to roll against a disagreeing fleet definition."
    fi
    printf '%s\n' "${names}"
}

github_runner_state() {
    # Echo "<status> <busy>" for one runner name, or "unknown unknown".
    local name="${1}"
    local row
    row=$(gh api --paginate "/orgs/${RUNNER_ORG}/actions/runners?per_page=100" 2>/dev/null |
        jq -rs --arg name "${name}" '
          [.[].runners[] | select(.name == $name)][0]
          | if . == null then "unknown unknown"
            else "\(.status) \(.busy | tostring)" end
        ' 2>/dev/null) || row=""
    if [[ -z "${row}" ]]; then
        echo "unknown unknown"
        return 0
    fi
    echo "${row}"
}

runner_is_idle() {
    # Fail CLOSED: returns 0 (idle, safe to recreate) only when GitHub says
    # online+not-busy AND the container's process list carries no job worker.
    # Anything else -- API failure, unknown runner, docker failure, the two
    # signals disagreeing -- returns non-zero and the caller skips.
    local name="${1}"
    local state status busy
    state=$(github_runner_state "${name}")
    status="${state%% *}"
    busy="${state##* }"
    if [[ "${status}" != "online" ]] || [[ "${busy}" != "false" ]]; then
        return 1
    fi

    local worker_lines
    if ! worker_lines=$(ssh "${RUNNER_HOST}" "docker top ${name} 2>/dev/null | grep -c 'Runner.Worker' || true" 2>/dev/null); then
        return 1
    fi
    worker_lines="${worker_lines//[$'\r\n ']/}"
    # An empty answer means `docker top` produced nothing we can read. That is
    # UNKNOWN, not idle.
    [[ -n "${worker_lines}" ]] || return 1
    [[ "${worker_lines}" == "0" ]] || return 1
    return 0
}

wait_for_runner_online() {
    local name="${1}"
    local elapsed=0
    while [[ "${elapsed}" -lt "${ROLL_ONLINE_MAX_SECONDS}" ]]; do
        sleep "${ROLL_ONLINE_INTERVAL_SECONDS}"
        elapsed=$((elapsed + ROLL_ONLINE_INTERVAL_SECONDS))
        local state
        state=$(github_runner_state "${name}")
        if [[ "${state%% *}" == "online" ]]; then
            log "  ${name} back online after ~${elapsed}s."
            return 0
        fi
    done
    return 1
}

roll_one_runner() {
    local name="${1}"
    local compose_cmd="docker compose -f ${RUNNER_HOST_DIR}/docker/docker-compose.runners.yml -f ${RUNNER_HOST_DIR}/docker/docker-compose.model-review-canary.yml"

    # Re-check immediately before stopping: a job can start between the
    # selection pass and this call.
    if ! runner_is_idle "${name}"; then
        log "  ${name} became busy (or its state is unknown) -- skipped."
        return 2
    fi

    log "  Recreating ${name} ..."
    if "${DRY_RUN}"; then
        log "[DRY RUN] would run: ${compose_cmd} up -d --force-recreate --no-deps ${name}"
        return 0
    fi

    # ONE service name, always. No --remove-orphans on this path: a rolling
    # call names a single service, and an orphan sweep during a partial roll
    # would delete containers this pass has not reached yet.
    # A roll asks GitHub for nothing, deliberately. A recreate restores the
    # runner's cached registration from its per-runner named volume, so the
    # container never re-registers and needs no registration handle at all;
    # RUNNER_TOKEN is exported empty purely so compose interpolates
    # deterministically instead of warning. A runner that nonetheless fails to
    # come back halts the roll, and recovering it is then a deliberate
    # operator step rather than something this path does silently.
    ssh "${RUNNER_HOST}" "
        set -euo pipefail
        export RUNNER_TOKEN=''
        cd ${RUNNER_HOST_DIR}
        ${compose_cmd} up -d --force-recreate --no-deps --no-build ${name}
    " || return 1

    wait_for_runner_online "${name}" || return 1
    return 0
}

rolling_deploy() {
    log "=== Rolling deploy (one runner at a time, busy-checked, fail-closed) ==="
    rsync_artifacts

    local services
    services=$(fleet_services)
    local total
    total=$(printf '%s\n' "${services}" | grep -c .)
    log "Rolling ${total} fleet services on ${RUNNER_HOST}. Expect roughly $((total * 100 / 60)) minutes."

    local pending=() name
    while IFS= read -r name; do
        [[ -n "${name}" ]] || continue
        pending+=("${name}")
    done <<< "${services}"

    if [[ -n "${ROLL_ONLY}" ]]; then
        # Fail closed on an unknown name: silently rolling nothing and printing
        # "Rolled 0/0" reads exactly like a converged fleet.
        local found=false
        for name in "${pending[@]}"; do
            [[ "${name}" == "${ROLL_ONLY}" ]] && found=true
        done
        "${found}" || err "--only=${ROLL_ONLY} is not a fleet service declared in ${COMPOSE_FILE}."
        pending=("${ROLL_ONLY}")
        total=1
        log "Converging a single runner: ${ROLL_ONLY}."
    fi

    local pass=0 done_count=0 failed="" rc
    while [[ "${#pending[@]}" -gt 0 ]] && [[ "${pass}" -le "${ROLL_SKIP_RETRY_PASSES}" ]]; do
        [[ "${pass}" -eq 0 ]] || log "--- retry pass ${pass} for ${#pending[@]} runner(s) that were busy earlier ---"
        local next=()
        for name in "${pending[@]}"; do
            rc=0
            roll_one_runner "${name}" || rc=$?
            case "${rc}" in
                0) done_count=$((done_count + 1))
                   log "  [${done_count}/${total}] ${name} done."
                   # --limit=N is the canary step: stop after N successful
                   # recreates so the change can be proven on one runner before
                   # the remaining 59 are touched.
                   if [[ "${ROLL_LIMIT}" -gt 0 ]] && [[ "${done_count}" -ge "${ROLL_LIMIT}" ]]; then
                       log "Reached --limit=${ROLL_LIMIT}; stopping. The rest of the fleet keeps its previous container env until the next roll."
                       return 0
                   fi ;;
                2) next+=("${name}") ;;
                *) failed="${name}"; break ;;
            esac
        done
        [[ -z "${failed}" ]] || break
        pending=(${next[@]+"${next[@]}"})
        pass=$((pass + 1))
    done

    if [[ -n "${failed}" ]]; then
        err "HALTED at ${failed}: it did not come back online within ${ROLL_ONLINE_MAX_SECONDS}s. ${done_count}/${total} rolled; the rest still carry the previous container env. Investigate that container before resuming -- do NOT retry blindly, a wedged removal spawns another wedged container."
    fi

    log "Rolled ${done_count}/${total} runners."
    if [[ "${#pending[@]}" -gt 0 ]]; then
        warn "Still busy after ${ROLL_SKIP_RETRY_PASSES} retry pass(es), NOT rolled: ${pending[*]}"
        warn "Re-run with --rolling to converge; a runner already rolled is recreated again, which is idempotent."
    fi
}

main() {
    log "Starting deploy-runners.sh (dry_run=${DRY_RUN}, skip_build=${SKIP_BUILD}, soft=${SOFT_DEPLOY}, rolling=${ROLLING_DEPLOY}, add=${ADD_SERVICES:-<none>})"
    log "Target host: ${RUNNER_HOST} | Org: ${RUNNER_ORG} | Group: ${RUNNER_GROUP}"
    log "Runner count: ${RUNNER_COUNT} | Compose file: ${COMPOSE_FILE}"

    if "${DRY_RUN}"; then
        log "[DRY RUN MODE] No remote commands will be executed."
    fi

    if "${SOFT_DEPLOY}" && "${ROLLING_DEPLOY}"; then
        err "--soft and --rolling are mutually exclusive: --soft never recreates a container, so it cannot carry a container-env change, which is the only reason to roll."
    fi

    if [[ -n "${ADD_SERVICES}" ]]; then
        # OMN-18408. Refused in combination rather than resolved by precedence:
        # each of the three is a different answer to "what happens to the
        # containers already running", and silently picking one is how an
        # operator gets a fleet roll they did not ask for.
        if "${SOFT_DEPLOY}" || "${ROLLING_DEPLOY}"; then
            err "--add is mutually exclusive with --soft and --rolling: --add creates or converges named non-pool services additively, --soft rewrites entrypoints in every running container, and --rolling recreates general-pool runners one at a time."
        fi
        add_services_deploy || err "Additive deploy of '${ADD_SERVICES}' did not come online. Check container logs on ${RUNNER_HOST}."
        log "=== deploy-runners.sh complete (added: ${ADD_SERVICES}) ==="
        return 0
    fi

    if "${ROLLING_DEPLOY}"; then
        rolling_deploy
    elif "${SOFT_DEPLOY}"; then
        soft_deploy
        poll_runners_online || warn "Not all runners came online. Check logs on ${RUNNER_HOST}."
    else
        deploy_with_retry
    fi

    print_stale_runner_report

    log "=== deploy-runners.sh complete ==="
}

main "$@"
