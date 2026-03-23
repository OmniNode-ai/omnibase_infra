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

RUNNER_GROUP="${RUNNER_GROUP:-omnibase-ci}"
RUNNER_WORK_DIR="${RUNNER_WORK_DIR:-_work}"

MAX_RETRIES=3
BACKOFF_SECONDS=(20 40 80)
CRED_CACHE_DIR="/home/runner/.runner-creds"
LOG_FILE="/tmp/runner-listener.log"

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
        cp -r "${cache_file}/"* "${RUNNER_HOME}/" 2>/dev/null || true
        return 0
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
        "Runner.Listener"
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
    echo "[entrypoint] Labels: ${RUNNER_LABELS} | Group: ${RUNNER_GROUP}"

    _as_runner "${RUNNER_HOME}/config.sh" \
        --url "${GITHUB_ORG_URL}" \
        --token "${RUNNER_TOKEN}" \
        --name "${RUNNER_NAME}" \
        --labels "${RUNNER_LABELS}" \
        --runnergroup "${RUNNER_GROUP}" \
        --work "${RUNNER_WORK_DIR}" \
        --unattended \
        --replace
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
while true; do
    echo "[entrypoint] Starting runner (attempt $((attempt + 1)))"
    set +e
    _as_runner "${RUNNER_HOME}/run.sh" 2>&1 | tee "${LOG_FILE}"
    exit_code=${PIPESTATUS[0]}
    set -e

    log_content=$(cat "${LOG_FILE}" 2>/dev/null || echo "")

    if [[ ${exit_code} -eq 0 ]]; then
        # GitHub runner exits 0 even when registration is server-side deleted
        # ("no retry needed" from its perspective). Check log for this case.
        if echo "${log_content}" | grep -qi "registration has been deleted"; then
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
            echo "[entrypoint] Runner exited cleanly (exit 0). Exiting."
            break
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
