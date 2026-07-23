#!/usr/bin/env bash
# GitHub Actions runner entrypoint with bounded re-registration
# Ticket: OMN-3275 / Epic: OMN-3273
#
# Re-registration policy:
#   - Max 3 retries with exponential backoff: 20s / 40s / 80s
#   - Only re-register on known "not registered" error strings
#   - Unknown errors exit immediately (do not retry; surface the real error)
#   - After max retries: sleep 5m then exit 1
#     → compose restart: unless-stopped will recover automatically
#
# Credential cache:
#   - Cache key = SHA256(RUNNER_LABELS + GITHUB_ORG_URL)
#   - Avoids re-registration if version hash matches (token reuse across restarts)
#   - Cache stored at /home/runner/.runner-creds/<cache-key>

set -euo pipefail

# ---------------------------------------------------------------------------
# Docker socket GID fix (runs as root, before dropping to runner)
# ---------------------------------------------------------------------------
# When /var/run/docker.sock is bind-mounted from the host, its GID may not
# match the container's 'docker' group GID. This block detects the socket's
# GID and adjusts the container's docker group to match, giving the runner
# user access to the Docker daemon.

_fix_docker_socket_gid() {
    local socket="/var/run/docker.sock"
    if [[ ! -S "${socket}" ]]; then
        echo "[entrypoint] No Docker socket at ${socket} — skipping GID fix"
        return 0
    fi

    local host_gid
    host_gid=$(stat -c '%g' "${socket}" 2>/dev/null || echo "")
    if [[ -z "${host_gid}" ]]; then
        echo "[entrypoint] Could not determine Docker socket GID — skipping"
        return 0
    fi

    local container_gid
    container_gid=$(getent group docker | cut -d: -f3 2>/dev/null || echo "")

    if [[ "${host_gid}" == "${container_gid}" ]]; then
        echo "[entrypoint] Docker socket GID (${host_gid}) matches container docker group"
        return 0
    fi

    echo "[entrypoint] Docker socket GID mismatch: socket=${host_gid}, container docker group=${container_gid}"
    echo "[entrypoint] Adjusting container docker group GID to ${host_gid}"
    groupmod -g "${host_gid}" docker
    echo "[entrypoint] Docker group GID updated to ${host_gid}"
}

if [[ "$(id -u)" -eq 0 ]]; then
    _fix_docker_socket_gid
fi

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------
# RUNNER_TOKEN is optional — only required for first-time registration.
# After initial setup, cached credentials are used and the token is not needed.
: "${RUNNER_NAME:?RUNNER_NAME must be set}"
: "${RUNNER_LABELS:?RUNNER_LABELS must be set}"
: "${GITHUB_ORG_URL:?GITHUB_ORG_URL must be set}"

# OMN-14900: no-colon default -- an EXPLICITLY EMPTY RUNNER_GROUP is the
# opt-out signal for repository-scoped registration (config.sh hard-fails on
# --runnergroup at repo scope, which would brick re-registration after a
# container recreate). Unset still defaults to the org fleet group.
RUNNER_GROUP="${RUNNER_GROUP-omnibase-ci}"
RUNNER_WORK_DIR="${RUNNER_WORK_DIR:-_work}"

MAX_RETRIES=3
BACKOFF_SECONDS=(20 40 80)
CRED_CACHE_DIR="/home/runner/.runner-creds"
LOG_FILE="${LOG_FILE:-/tmp/runner-listener.log}"

# ---------------------------------------------------------------------------
# Listener watchdog (OMN-13915)
# ---------------------------------------------------------------------------
# Incident: Runner.Listener died inside 37/48 containers while the run.sh
# wrapper tree stayed alive, so the entrypoint never saw an exit code and the
# container sat "Up (healthy)" for four days. The watchdog closes that gap:
# while run.sh is running, assert a bin/Runner.Listener process exists. After
# LISTENER_SUPERVISE_MISSES consecutive misses (the grace window covers runner
# self-update, which briefly restarts the listener), kill the wrapper tree so
# the main loop restarts the runner with the cached in-place credentials.
# Bounded by LISTENER_RESTART_MAX to surface a genuinely crash-looping
# listener as a container exit (compose restart policy + runner-monitor alert)
# instead of hiding it behind unbounded silent restarts.
LISTENER_SUPERVISE_INTERVAL="${LISTENER_SUPERVISE_INTERVAL:-60}"
LISTENER_SUPERVISE_MISSES="${LISTENER_SUPERVISE_MISSES:-5}"
LISTENER_RESTART_MAX="${LISTENER_RESTART_MAX:-50}"
# LISTENER_PGREP_PATTERN is derived from RUNNER_HOME below (after RUNNER_HOME
# is resolved) so it matches THIS runner home's listener binary only.

# ---------------------------------------------------------------------------
# Hung-listener heartbeat watchdog (OMN-14564)
# ---------------------------------------------------------------------------
# Incident (2026-07-16..23): 11/64 runners went GitHub-offline for ~6 days
# with the Runner.Listener process STILL ALIVE — deadlocked inside the
# AAD/OAuth token-refresh HTTP call while acknowledging a broker job
# assignment (terminal _diag line: "AAD Correlation ID for this token
# request: Unknown"). A hung listener passes the OMN-13915 process-existence
# watchdog forever, never exits (so run.sh never respawns it), and only the
# Docker healthcheck's _diag heartbeat layer flagged it — with nothing acting
# on the signal. This watchdog turns that same detection into remediation:
# when the listener process exists but the newest _diag *.log is older than
# RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS (same tunable + condition as
# healthcheck.sh layer 2) for LISTENER_HEARTBEAT_MISSES consecutive supervise
# ticks, kill the listener explicitly and recycle the wrapper tree. The
# recycle NEVER fires while a Runner.Worker is executing a job.
RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS="${RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS:-900}"
LISTENER_HEARTBEAT_MISSES="${LISTENER_HEARTBEAT_MISSES:-3}"

# ---------------------------------------------------------------------------
# Credential cache helpers
# ---------------------------------------------------------------------------

_cache_key() {
    echo -n "${RUNNER_LABELS}:${GITHUB_ORG_URL}" | sha256sum | awk '{print $1}'
}

_restore_cached_creds() {
    local key
    key=$(_cache_key)
    local cache_file="${CRED_CACHE_DIR}/${key}"
    if [[ -d "${cache_file}" ]]; then
        echo "[entrypoint] Restoring cached runner credentials (key=${key:0:12}...)"
        local restored=0
        for f in .runner .credentials .credentials_rsaparams; do
            if [[ -f "${cache_file}/${f}" ]]; then
                cp "${cache_file}/${f}" "${RUNNER_HOME}/${f}"
                chown runner:runner "${RUNNER_HOME}/${f}"
                restored=$((restored + 1))
            fi
        done
        if [[ -f "${RUNNER_HOME}/.runner" && -f "${RUNNER_HOME}/.credentials" ]]; then
            return 0
        fi
        rm -f -- \
            "${RUNNER_HOME}/.runner" \
            "${RUNNER_HOME}/.credentials" \
            "${RUNNER_HOME}/.credentials_rsaparams"
        echo "[entrypoint] Credential cache is incomplete; falling back to registration."
    fi
    return 1
}

_save_creds() {
    local key
    key=$(_cache_key)
    local cache_file="${CRED_CACHE_DIR}/${key}"
    mkdir -p "${cache_file}"
    # Store the registration credential files
    for f in .runner .credentials .credentials_rsaparams; do
        if [[ -f "${RUNNER_HOME}/${f}" ]]; then
            cp "${RUNNER_HOME}/${f}" "${cache_file}/${f}"
            chown runner:runner "${cache_file}/${f}"
        fi
    done
    echo "[entrypoint] Runner credentials cached (key=${key:0:12}...)"
}

_clear_cached_creds() {
    local key
    key=$(_cache_key)
    local cache_file="${CRED_CACHE_DIR}/${key}"
    if [[ -d "${cache_file}" ]]; then
        rm -rf "${cache_file}"
        echo "[entrypoint] Cleared stale credential cache (key=${key:0:12}...)"
    fi
}

# ---------------------------------------------------------------------------
# Known registration error detection
# These strings indicate the runner token is stale or the runner was removed
# from the org — a re-registration is needed.
# ---------------------------------------------------------------------------

_is_registration_error() {
    local log_content="${1}"
    local known_patterns=(
        "not registered"
        "HTTP 401"
        "Failed to get session"
        "Unable to connect to server"
        "unauthorized"
        "invalid token"
        "runner registration token"
        "RegistrationError"
    )
    for pattern in "${known_patterns[@]}"; do
        if echo "${log_content}" | grep -qi "${pattern}"; then
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Privilege de-escalation helper
# ---------------------------------------------------------------------------
# If running as root (default after Dockerfile change), use gosu to run
# commands as the 'runner' user. If already running as runner, execute directly.

_as_runner() {
    if [[ "$(id -u)" -eq 0 ]]; then
        gosu runner "$@"
    else
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_register() {
    echo "[entrypoint] Registering runner: ${RUNNER_NAME} @ ${GITHUB_ORG_URL}"
    echo "[entrypoint] Labels: ${RUNNER_LABELS} | Group: ${RUNNER_GROUP:-<none: repo-scoped>}"

    local config_args=(
        --url "${GITHUB_ORG_URL}"
        --token "${RUNNER_TOKEN}"
        --name "${RUNNER_NAME}"
        --labels "${RUNNER_LABELS}"
        --work "${RUNNER_WORK_DIR}"
        --unattended
        --disableupdate
        --replace
    )
    # OMN-14900: --runnergroup only when a group is actually set. Repo-scoped
    # registration (GITHUB_ORG_URL pointing at a repository, as the deploy
    # runner does) rejects --runnergroup outright.
    if [[ -n "${RUNNER_GROUP}" ]]; then
        config_args+=(--runnergroup "${RUNNER_GROUP}")
    fi

    _as_runner "${RUNNER_HOME}/config.sh" "${config_args[@]}"
}

_deregister() {
    if [[ -z "${RUNNER_TOKEN:-}" ]]; then
        echo "[entrypoint] Skipping de-registration (no RUNNER_TOKEN available)"
        return 0
    fi
    echo "[entrypoint] Attempting graceful de-registration..."
    _as_runner "${RUNNER_HOME}/config.sh" remove --token "${RUNNER_TOKEN}" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Main entrypoint loop
# ---------------------------------------------------------------------------

RUNNER_HOME="${RUNNER_HOME:-/home/runner/actions-runner}"
cd "${RUNNER_HOME}"

shutdown_requested=0
_request_shutdown() {
    shutdown_requested=1
    echo "[entrypoint] Shutdown requested; runner listener will not be relaunched."
}
trap _request_shutdown TERM INT

# Watchdog pattern (OMN-13915): match THIS runner home's listener binary path.
# Dots escaped for pgrep's ERE matching.
LISTENER_PGREP_PATTERN="${LISTENER_PGREP_PATTERN:-${RUNNER_HOME//./\\.}/bin/Runner\.Listener}"
# Worker pattern (OMN-14564): a Runner.Worker process means a job is executing
# — the heartbeat watchdog must NEVER recycle mid-job.
WORKER_PGREP_PATTERN="${WORKER_PGREP_PATTERN:-${RUNNER_HOME//./\\.}/bin/Runner\.Worker}"

# OMN-14564: mirror of healthcheck.sh layer 2 — returns 0 (stale) when no
# _diag *.log was modified within RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS. A
# missing _diag directory under a "live" listener is the same divergence and
# also reads as stale; the LISTENER_HEARTBEAT_MISSES grace window covers
# first-registration startup before the listener writes its first log.
_listener_heartbeat_stale() {
    local diag_dir="${RUNNER_HOME}/_diag"
    local max_age_minutes=$(( (RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS + 59) / 60 ))
    local fresh_file
    fresh_file=$(find "${diag_dir}" -type f -name '*.log' -mmin "-${max_age_minutes}" -print 2>/dev/null | head -n 1)
    [[ -z "${fresh_file}" ]]
}

# Kill the wrapper tree AND the listener binary itself (TERM, grace, KILL).
# The explicit listener pkill is load-bearing for OMN-14564: a listener
# deadlocked in its token-refresh HTTP call ignores the wrapper-tree TERM,
# and a surviving hung listener would collide with the respawned listener's
# broker session.
_recycle_runner_tree() {
    local pid="${1}"
    kill -TERM "${pid}" 2>/dev/null || true
    pkill -TERM -f "run-helper" 2>/dev/null || true
    pkill -TERM -f "${LISTENER_PGREP_PATTERN}" 2>/dev/null || true
    sleep 10
    kill -KILL "${pid}" 2>/dev/null || true
    pkill -KILL -f "run-helper" 2>/dev/null || true
    pkill -KILL -f "${LISTENER_PGREP_PATTERN}" 2>/dev/null || true
}

# Check for credentials in priority order:
# 1. In-place (container restart — files already in RUNNER_HOME)
# 2. Volume cache (fresh container — restore from mounted volume)
# 3. Fresh registration (first-time setup — requires RUNNER_TOKEN)
if [[ -f "${RUNNER_HOME}/.runner" && -f "${RUNNER_HOME}/.credentials" ]]; then
    echo "[entrypoint] Found in-place credentials — skipping registration (container restart)"
elif _restore_cached_creds; then
    echo "[entrypoint] Restored credentials from volume cache — skipping registration"
else
    if [[ -z "${RUNNER_TOKEN:-}" ]]; then
        echo "[entrypoint] ERROR: No credentials found and RUNNER_TOKEN is not set."
        echo "[entrypoint] For first-time registration, set RUNNER_TOKEN in the environment."
        echo "[entrypoint] Generate a token at: https://github.com/organizations/OmniNode-ai/settings/actions/runners/new"
        echo "[entrypoint] Token is valid for 1 hour. After registration, cached credentials are used."
        exit 1
    fi
    echo "[entrypoint] No cached credentials found — registering with RUNNER_TOKEN..."
    _register
    _save_creds
fi

attempt=0
listener_restarts=0
while true; do
    echo "[entrypoint] Starting runner (attempt $((attempt + 1)))"
    set +e
    _as_runner "${RUNNER_HOME}/run.sh" > >(tee "${LOG_FILE}") 2>&1 &
    runner_pid=$!

    # Watchdog: run.sh alive but no Runner.Listener process = the OMN-13915
    # zombie mode. Recycle the wrapper tree so this loop restarts the runner.
    # Listener process alive but _diag heartbeat stale (and no job running)
    # = the OMN-14564 hung-listener mode; same recycle, plus an explicit
    # listener kill because a hung listener never exits on its own.
    supervised_kill=0
    misses=0
    hb_misses=0
    while kill -0 "${runner_pid}" 2>/dev/null; do
        sleep "${LISTENER_SUPERVISE_INTERVAL}"
        kill -0 "${runner_pid}" 2>/dev/null || break
        if pgrep -f "${LISTENER_PGREP_PATTERN}" >/dev/null 2>&1; then
            misses=0
            # OMN-14564: process existence is NOT liveness. A worker process
            # means a job is executing — never recycle mid-job, whatever the
            # heartbeat says (the Docker healthcheck still surfaces it).
            if pgrep -f "${WORKER_PGREP_PATTERN}" >/dev/null 2>&1; then
                hb_misses=0
                continue
            fi
            if _listener_heartbeat_stale; then
                hb_misses=$((hb_misses + 1))
                echo "[entrypoint] WATCHDOG: listener process alive but no _diag heartbeat within ${RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS}s (miss ${hb_misses}/${LISTENER_HEARTBEAT_MISSES}) — OMN-14564 hung-listener mode"
                if [[ ${hb_misses} -ge ${LISTENER_HEARTBEAT_MISSES} ]]; then
                    echo "[entrypoint] WATCHDOG: listener hung (alive but silent) — killing listener and recycling runner wrapper tree (OMN-14564)"
                    supervised_kill=1
                    _recycle_runner_tree "${runner_pid}"
                    break
                fi
            else
                hb_misses=0
            fi
            continue
        fi
        misses=$((misses + 1))
        echo "[entrypoint] WATCHDOG: run.sh alive (pid ${runner_pid}) but no Runner.Listener process (miss ${misses}/${LISTENER_SUPERVISE_MISSES})"
        if [[ ${misses} -ge ${LISTENER_SUPERVISE_MISSES} ]]; then
            echo "[entrypoint] WATCHDOG: listener dead-in-container — recycling runner wrapper tree (OMN-13915)"
            supervised_kill=1
            _recycle_runner_tree "${runner_pid}"
            break
        fi
    done

    wait "${runner_pid}"
    exit_code=$?
    set -e

    log_content=$(cat "${LOG_FILE}" 2>/dev/null || echo "")

    if [[ ${shutdown_requested} -eq 1 ]]; then
        echo "[entrypoint] Runner exited during shutdown; stopping entrypoint loop."
        exit 0
    fi

    if [[ ${supervised_kill} -eq 1 ]]; then
        listener_restarts=$((listener_restarts + 1))
        if [[ ${listener_restarts} -gt ${LISTENER_RESTART_MAX} ]]; then
            echo "[entrypoint] WATCHDOG: listener died ${listener_restarts} times (> ${LISTENER_RESTART_MAX}). Exiting so the container restart policy + runner-monitor surface it."
            exit 1
        fi
        echo "[entrypoint] WATCHDOG: restarting runner after listener death (restart ${listener_restarts}/${LISTENER_RESTART_MAX})"
        sleep 10
        continue
    fi

    if [[ ${exit_code} -eq 0 ]]; then
        # GitHub runner exits 0 even when registration is server-side deleted
        # ("no retry needed" from its perspective). Check log for this case.
        if echo "${log_content}" | grep -qiE "registration has been deleted|Not configured"; then
            echo "[entrypoint] Runner registration was deleted by GitHub (exit 0 but stale)."
            echo "[entrypoint] Clearing cached credentials and will re-register on next attempt."
            _clear_cached_creds
            # Remove in-place credentials so fresh registration can proceed
            rm -f "${RUNNER_HOME}/.runner" "${RUNNER_HOME}/.credentials" "${RUNNER_HOME}/.credentials_rsaparams"
            if [[ -z "${RUNNER_TOKEN:-}" ]]; then
                echo "[entrypoint] ERROR: Cannot re-register — RUNNER_TOKEN is not set."
                echo "[entrypoint] Generate a token at: https://github.com/organizations/OmniNode-ai/settings/actions/runners/new"
                exit 1
            fi
            # Fall through to the registration retry loop below
        else
            echo "[entrypoint] Runner exited cleanly (exit 0). Relaunching listener after short backoff."
            sleep 5
            attempt=0
            continue
        fi
    fi

    echo "[entrypoint] Runner exited with code ${exit_code}"

    if ! _is_registration_error "${log_content}"; then
        echo "[entrypoint] Unknown error (not a registration error). Exiting immediately."
        echo "[entrypoint] Last log output:"
        tail -20 "${LOG_FILE}" 2>/dev/null || true
        exit "${exit_code}"
    fi

    echo "[entrypoint] Registration error detected."
    _clear_cached_creds

    if [[ ${attempt} -ge ${MAX_RETRIES} ]]; then
        echo "[entrypoint] Max retries (${MAX_RETRIES}) reached. Sleeping 5m before exit 1."
        echo "[entrypoint] Compose restart: unless-stopped will recover."
        sleep 300
        exit 1
    fi

    backoff=${BACKOFF_SECONDS[${attempt}]}
    echo "[entrypoint] Re-registering in ${backoff}s (retry $((attempt + 1))/${MAX_RETRIES})..."
    sleep "${backoff}"

    _deregister
    _register
    _save_creds

    attempt=$((attempt + 1))
done

echo "[entrypoint] Exiting."
