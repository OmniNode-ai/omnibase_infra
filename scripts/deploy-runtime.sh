#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
#
# deploy-runtime.sh -- Stable runtime deployment for omnibase_infra
#
# Rsyncs the current repository to a versioned deployment root
# (~/.omnibase/infra/deployed/{version}/{git-sha}/), then runs
# docker compose from that stable location. This eliminates the
# directory-derived compose project name collision that occurs when
# multiple repo copies (omnibase_infra2, omnibase_infra4, etc.) all
# share the same compose project name.
#
# Pattern: real rsync copies (not symlinks), versioned directories,
# dry-run by default.
#
# Usage:
#   ./scripts/deploy-runtime.sh                   # Dry-run preview
#   ./scripts/deploy-runtime.sh --execute         # Deploy + build
#   ./scripts/deploy-runtime.sh --execute --restart  # Deploy + build + restart
#   ./scripts/deploy-runtime.sh --print-compose-cmd  # Show compose commands
#   ./scripts/deploy-runtime.sh --help            # Full usage

set -euo pipefail

# =============================================================================
# Constants
# =============================================================================

SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_NAME
readonly SCRIPT_VERSION="1.0.0"

# Deployment root -- all versioned deployments live under this tree
readonly DEPLOY_ROOT="${HOME}/.omnibase/infra"
readonly REGISTRY_FILE="${DEPLOY_ROOT}/registry.json"
readonly LOCK_DIR="${DEPLOY_ROOT}/.deploy.lock"

# Maximum number of deployed versions to retain. Older deployments are pruned
# after each successful deployment. The currently active deployment (tracked in
# registry.json) is never removed regardless of age.
readonly MAX_DEPLOYMENTS="${MAX_DEPLOYMENTS:-5}"

# Runtime services to restart (excludes infrastructure: postgres, redpanda, valkey)
readonly RUNTIME_SERVICES=(
    omninode-runtime
    runtime-effects
    runtime-worker
    agent-actions-consumer
)

# Minimum Docker Compose version (nested variable expansion support)
readonly MIN_COMPOSE_VERSION="2.20"

# Health check parameters
readonly HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8085/health}"
readonly HEALTH_CHECK_RETRIES=15
readonly HEALTH_CHECK_INTERVAL=4

# =============================================================================
# Defaults
# =============================================================================

MODE="dry-run"           # dry-run | execute
FORCE=false
RESTART=false
# Set after rsync to enable automatic cleanup of orphaned deployment directories
# on failure. If this is non-empty and the deployment directory is NOT the active
# deployment in registry.json, the trap handler will remove it.
DEPLOY_DIR_TO_CLEANUP=""
# Default is hardcoded and safe; any changes must comply with ^[a-zA-Z0-9_-]+$ (see parse_args).
COMPOSE_PROFILE="runtime"
PRINT_COMPOSE_CMD=false

# =============================================================================
# Logging
# =============================================================================

log_info() {
    printf '[deploy] %s\n' "$*"
}

log_warn() {
    printf '[deploy] WARNING: %s\n' "$*" >&2
}

log_error() {
    printf '[deploy] ERROR: %s\n' "$*" >&2
}

log_step() {
    printf '\n[deploy] === %s ===\n' "$*"
}

log_cmd() {
    printf '[deploy]   > %s\n' "$*"
}

# =============================================================================
# Usage
# =============================================================================

usage() {
    cat <<EOF
${SCRIPT_NAME} v${SCRIPT_VERSION} -- Stable runtime deployment for omnibase_infra

Rsyncs the current repo to ~/.omnibase/infra/deployed/{version}/{git-sha}/,
then runs docker compose from that stable location.

USAGE
    ${SCRIPT_NAME} [OPTIONS]

OPTIONS
    (none)              Dry-run mode (default). Preview what would be deployed.
    --execute           Actually deploy: rsync, write registry, build images.
    --force             Required to overwrite an existing version+sha directory.
    --restart           Restart runtime containers after build (requires --execute).
    --profile <name>    Docker compose profile (default: runtime).
    --print-compose-cmd Print exact compose commands without executing, then exit.
    --help              Show this help message and exit.

DEPLOYMENT ROOT
    ~/.omnibase/infra/
    +-- .deploy.lock/                       mkdir-based concurrency guard
    +-- registry.json                       tracks active deployment
    +-- deployed/
        +-- {version}/
            +-- {git-sha}/                  immutable build directory
                +-- pyproject.toml
                +-- poetry.lock
                +-- src/omnibase_infra/
                +-- contracts/
                +-- docker/
                    +-- docker-compose.infra.yml
                    +-- Dockerfile.runtime
                    +-- entrypoint-runtime.sh
                    +-- .env                preserved across deploys
                    +-- .env.local          preserved (user overrides)
                    +-- certs/              preserved (TLS certs)
                    +-- migrations/forward/

EXAMPLES
    # Preview what would be deployed
    ${SCRIPT_NAME}

    # Deploy and build images
    ${SCRIPT_NAME} --execute

    # Deploy, build, and restart containers
    ${SCRIPT_NAME} --execute --restart

    # Redeploy same version+sha (overwrite)
    ${SCRIPT_NAME} --execute --force

    # Print compose commands for manual use
    ${SCRIPT_NAME} --print-compose-cmd

    # Check registry
    cat ~/.omnibase/infra/registry.json | jq .

    # Verify image labels match deployed SHA
    docker inspect omninode-runtime \\
        --format='{{index .Config.Labels "org.opencontainers.image.revision"}}'
EOF
    exit 0
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --execute)
                MODE="execute"
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --restart)
                RESTART=true
                shift
                ;;
            --profile)
                if [[ -z "${2:-}" || "${2:0:1}" == "-" ]]; then
                    log_error "--profile requires a value"
                    exit 1
                fi
                # Validate profile name: only alphanumeric, hyphens, and underscores
                # are allowed to prevent invalid compose project names.
                if [[ ! "$2" =~ ^[a-zA-Z0-9_-]+$ ]]; then
                    log_error "--profile value must contain only alphanumeric characters, hyphens, and underscores."
                    log_error "  Got: '$2'"
                    exit 1
                fi
                COMPOSE_PROFILE="$2"
                shift 2
                ;;
            --print-compose-cmd)
                PRINT_COMPOSE_CMD=true
                shift
                ;;
            --help|-h)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                log_error "Run '${SCRIPT_NAME} --help' for usage."
                exit 1
                ;;
        esac
    done

    # Validate flag combinations
    if [[ "${RESTART}" == true && "${MODE}" != "execute" ]]; then
        log_error "--restart requires --execute"
        exit 1
    fi
}

# =============================================================================
# Prerequisites
# =============================================================================

check_command() {
    local cmd="$1"
    local purpose="$2"
    if ! command -v "${cmd}" &>/dev/null; then
        log_error "'${cmd}' is required (${purpose}) but not found in PATH."
        exit 1
    fi
}

check_compose_version() {
    local version_output
    version_output="$(docker compose version --short 2>/dev/null || true)"

    if [[ -z "${version_output}" ]]; then
        log_error "docker compose plugin not found. Install Docker Compose v2.20+."
        exit 1
    fi

    # Strip leading 'v' if present
    version_output="${version_output#v}"

    # Compare major.minor
    local major minor
    major="$(echo "${version_output}" | cut -d. -f1)"
    minor="$(echo "${version_output}" | cut -d. -f2)"
    local req_major req_minor
    req_major="$(echo "${MIN_COMPOSE_VERSION}" | cut -d. -f1)"
    req_minor="$(echo "${MIN_COMPOSE_VERSION}" | cut -d. -f2)"

    if (( major < req_major || (major == req_major && minor < req_minor) )); then
        log_error "Docker Compose ${MIN_COMPOSE_VERSION}+ required (found ${version_output})."
        log_error "Nested variable expansion requires Compose >= ${MIN_COMPOSE_VERSION}."
        exit 1
    fi

    log_info "Docker Compose version: ${version_output}"
}

validate_prerequisites() {
    log_step "Validate Prerequisites"

    check_command rsync   "file synchronization"
    check_command docker  "container runtime"
    check_command jq      "JSON processing"
    check_command git     "version control"
    check_command curl    "deployment verification"

    check_compose_version
}

# =============================================================================
# Repository Validation
# =============================================================================

resolve_repo_root() {
    # Walk up from script location to find pyproject.toml
    local dir
    dir="$(cd "$(dirname "$0")" && pwd)"

    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/pyproject.toml" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done

    log_error "Cannot find repository root (no pyproject.toml found above script)."
    exit 1
}

validate_repo_structure() {
    local repo_root="$1"
    local missing=()

    [[ -f "${repo_root}/pyproject.toml" ]]                          || missing+=("pyproject.toml")
    [[ -f "${repo_root}/poetry.lock" ]]                             || missing+=("poetry.lock")
    [[ -d "${repo_root}/src/omnibase_infra" ]]                      || missing+=("src/omnibase_infra/")
    [[ -d "${repo_root}/contracts" ]]                                || missing+=("contracts/")
    [[ -d "${repo_root}/docker" ]]                                   || missing+=("docker/")
    [[ -f "${repo_root}/docker/docker-compose.infra.yml" ]]         || missing+=("docker/docker-compose.infra.yml")
    [[ -f "${repo_root}/docker/Dockerfile.runtime" ]]               || missing+=("docker/Dockerfile.runtime")
    [[ -f "${repo_root}/docker/entrypoint-runtime.sh" ]]            || missing+=("docker/entrypoint-runtime.sh")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Repository structure validation failed. Missing:"
        for item in "${missing[@]}"; do
            log_error "  - ${item}"
        done
        exit 1
    fi

    log_info "Repository structure validated."
}

# =============================================================================
# Identity -- version + git SHA
# =============================================================================

read_version() {
    local repo_root="$1"
    local version

    # Extract version from the [tool.poetry] section of pyproject.toml.
    # A naive grep -m1 '^version' could match a version key in any TOML
    # section (e.g. a dependency table).  This awk approach activates only
    # inside [tool.poetry] and deactivates when the next section header
    # is reached, ensuring we read the project version.
    version="$(awk '
        /^\[tool\.poetry\]/ { in_section=1; next }
        /^\[/               { in_section=0 }
        in_section && /^version[[:space:]]*=/ {
            gsub(/.*=[[:space:]]*"/, "");
            gsub(/".*/, "");
            print;
            exit
        }
    ' "${repo_root}/pyproject.toml")"

    if [[ -z "${version}" ]]; then
        log_error "Could not read version from pyproject.toml [tool.poetry] section"
        exit 1
    fi

    echo "${version}"
}

read_git_sha() {
    local repo_root="$1"
    local sha

    # 12-char SHA prefix for deployment directory naming. 7-char prefixes risk
    # collisions even in moderately sized repos; 12 chars is effectively unique.
    sha="$(git -C "${repo_root}" rev-parse --short=12 HEAD 2>/dev/null || true)"

    if [[ -z "${sha}" ]]; then
        log_error "Could not determine git SHA. Is this a git repository?"
        exit 1
    fi

    echo "${sha}"
}

check_git_dirty() {
    local repo_root="$1"
    if [[ -n "$(git -C "${repo_root}" status --porcelain 2>/dev/null || true)" ]]; then
        log_warn "Working tree has uncommitted or untracked changes."
        log_warn "The deployed SHA will not match the actual file contents."
    fi
}

# =============================================================================
# Concurrency Lock
# =============================================================================

acquire_lock() {
    mkdir -p "${DEPLOY_ROOT}"

    local pid_file="${LOCK_DIR}/pid"

    # Use mkdir for atomic, cross-platform locking (works on macOS + Linux).
    # mkdir is atomic on all POSIX systems -- it either creates the directory
    # or fails if it already exists, with no race window.
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
        # Lock acquired -- write PID immediately to avoid a window where the
        # lock directory exists but has no PID file (Issue: if the script is
        # killed between mkdir and PID write, subsequent runs cannot verify
        # the lock owner and refuse to proceed).
        echo $$ > "${pid_file}"
    else
        # Lock directory exists -- check for stale lock by verifying the
        # owning PID is still alive.
        if [[ -f "${pid_file}" ]]; then
            local lock_pid
            lock_pid="$(cat "${pid_file}" 2>/dev/null || true)"
            # Validate PID is numeric before using it in kill -0.
            # A corrupted or empty PID file is treated as a stale lock.
            if [[ -n "${lock_pid}" ]] && ! [[ "${lock_pid}" =~ ^[0-9]+$ ]]; then
                log_warn "Stale lock detected (PID file contains non-numeric value: '${lock_pid}')."
                log_warn "Treating as corrupted lock and cleaning up..."
                lock_pid=""
            fi
            if [[ -z "${lock_pid}" ]] || ! kill -0 "${lock_pid}" 2>/dev/null; then
                if [[ -n "${lock_pid}" ]]; then
                    log_warn "Stale lock detected (PID ${lock_pid} is no longer running)."
                fi
                log_warn "Cleaning up stale lock and re-acquiring..."
                # Re-read the PID file before removing the lock directory.
                # Between the initial stale check and this point, another
                # process may have legitimately acquired the lock. If the
                # PID file now contains a live process, abort cleanup.
                local recheck_pid
                recheck_pid="$(cat "${pid_file}" 2>/dev/null || true)"
                if [[ -n "${recheck_pid}" ]] && [[ "${recheck_pid}" =~ ^[0-9]+$ ]] \
                        && kill -0 "${recheck_pid}" 2>/dev/null; then
                    log_error "Lock was re-acquired by PID ${recheck_pid} during stale cleanup."
                    log_error "A concurrent deployment is legitimately running. Exiting."
                    exit 2
                fi
                rm -rf "${LOCK_DIR}"
                # Retry mkdir in a short loop to handle the race between rm
                # and mkdir where another process could acquire the lock.
                local lock_acquired=false
                local retry
                for retry in 1 2 3; do
                    if mkdir "${LOCK_DIR}" 2>/dev/null; then
                        # Write PID immediately after acquiring the lock to
                        # eliminate the window where the lock exists without
                        # a PID file.
                        echo $$ > "${pid_file}"
                        lock_acquired=true
                        break
                    fi
                    # Another process grabbed the lock between our rm and mkdir.
                    # Brief sleep before retrying to avoid tight spin.
                    log_warn "Lock contention on retry ${retry}/3, waiting..."
                    sleep 1
                done
                if [[ "${lock_acquired}" != true ]]; then
                    log_error "Another process acquired the lock during stale cleanup."
                    log_error "A concurrent deployment is legitimately running. Exiting."
                    exit 2
                fi
                # Fall through to set up traps and continue
            else
                log_error "Another deployment is in progress (locked by PID ${lock_pid})."
                log_error "If the previous deployment crashed, remove the lock manually:"
                log_error "  rm -rf ${LOCK_DIR}"
                exit 2
            fi
        else
            # Lock directory exists but has no PID file. This happens when the
            # script was killed (e.g., SIGKILL) between mkdir and PID write.
            # Treat as a stale lock and attempt recovery, same as a dead PID.
            log_warn "Lock directory exists but has no PID file (likely interrupted deployment)."
            log_warn "Treating as stale lock and cleaning up..."
            rm -rf "${LOCK_DIR}"
            local lock_acquired=false
            local retry
            for retry in 1 2 3; do
                if mkdir "${LOCK_DIR}" 2>/dev/null; then
                    echo $$ > "${pid_file}"
                    lock_acquired=true
                    break
                fi
                log_warn "Lock contention on retry ${retry}/3, waiting..."
                sleep 1
            done
            if [[ "${lock_acquired}" != true ]]; then
                log_error "Another process acquired the lock during stale cleanup."
                log_error "A concurrent deployment is legitimately running. Exiting."
                exit 2
            fi
        fi
    fi

    # Ensure lock is released on exit (normal, error, or signal).
    # EXIT handles cleanup for normal/error exits.
    # INT/TERM/HUP must explicitly exit after cleanup so the script
    # does not continue executing after receiving a termination signal.
    #
    # ASSUMPTION: acquire_lock() is only called during execute mode (see main()).
    # Dry-run and --print-compose-cmd exit before reaching this code.
    # These traps REPLACE (not chain) any existing EXIT/INT/TERM/HUP traps;
    # this is acceptable because no prior traps are set in this script.
    trap 'cleanup_on_exit' EXIT
    trap 'cleanup_on_exit; exit 1' INT TERM HUP

    log_info "Acquired deployment lock (PID $$)."
}

# =============================================================================
# Cleanup -- partial deployment rollback + lock release
# =============================================================================

cleanup_on_exit() {
    # Remove orphaned deployment directory on failure. If DEPLOY_DIR_TO_CLEANUP
    # is set and registry.json does NOT point to it, the deployment was partial
    # and should be removed.
    if [[ -n "${DEPLOY_DIR_TO_CLEANUP}" && -d "${DEPLOY_DIR_TO_CLEANUP}" ]]; then
        local active_path=""
        if [[ -f "${REGISTRY_FILE}" ]]; then
            active_path="$(jq -r '.deploy_path // empty' "${REGISTRY_FILE}" 2>/dev/null || true)"
        fi
        if [[ "${active_path}" != "${DEPLOY_DIR_TO_CLEANUP}" ]]; then
            log_warn "Cleaning up partial deployment: ${DEPLOY_DIR_TO_CLEANUP}"
            rm -rf "${DEPLOY_DIR_TO_CLEANUP}" 2>/dev/null || true
            # Remove empty parent version directory if this was the only SHA
            local parent_dir
            parent_dir="$(dirname "${DEPLOY_DIR_TO_CLEANUP}")"
            if [[ -d "${parent_dir}" ]] && [[ -z "$(ls -A "${parent_dir}" 2>/dev/null)" ]]; then
                rmdir "${parent_dir}" 2>/dev/null || true
            fi
        fi
    fi

    # Release concurrency lock
    rm -rf "${LOCK_DIR}" 2>/dev/null || true
}

# =============================================================================
# Prune -- remove old deployments beyond retention limit
# =============================================================================

prune_old_deployments() {
    local deployed_root="${DEPLOY_ROOT}/deployed"

    if [[ ! -d "${deployed_root}" ]]; then
        return 0
    fi

    log_step "Prune Old Deployments"

    # Determine active deployment path from registry
    local active_path=""
    if [[ -f "${REGISTRY_FILE}" ]]; then
        active_path="$(jq -r '.deploy_path // empty' "${REGISTRY_FILE}" 2>/dev/null || true)"
    fi

    # Collect all deployment directories (version/sha pairs) sorted by
    # modification time, newest first. Each entry is a full path like
    # ~/.omnibase/infra/deployed/1.2.3/abc123def456/
    local all_deployments=()
    local version_dir
    for version_dir in "${deployed_root}"/*/; do
        [[ -d "${version_dir}" ]] || continue
        local sha_dir
        for sha_dir in "${version_dir}"*/; do
            [[ -d "${sha_dir}" ]] || continue
            all_deployments+=("${sha_dir%/}")
        done
    done

    # Sort by modification time (newest first) using stat.
    # macOS stat uses -f '%m' for epoch; GNU stat uses -c '%Y'.
    local sorted_deployments=()
    if stat -f '%m' / >/dev/null 2>&1; then
        # macOS (BSD stat)
        while IFS= read -r line; do
            sorted_deployments+=("${line}")
        done < <(
            for d in "${all_deployments[@]}"; do
                printf '%s %s\n' "$(stat -f '%m' "${d}")" "${d}"
            done | sort -rn | awk '{print $2}'
        )
    else
        # Linux (GNU stat)
        while IFS= read -r line; do
            sorted_deployments+=("${line}")
        done < <(
            for d in "${all_deployments[@]}"; do
                printf '%s %s\n' "$(stat -c '%Y' "${d}")" "${d}"
            done | sort -rn | awk '{print $2}'
        )
    fi

    local total="${#sorted_deployments[@]}"
    if (( total <= MAX_DEPLOYMENTS )); then
        log_info "Deployment count (${total}) within retention limit (${MAX_DEPLOYMENTS}). No pruning needed."
        return 0
    fi

    log_info "Found ${total} deployments, retention limit is ${MAX_DEPLOYMENTS}. Pruning..."

    local kept=0
    local pruned=0
    for deploy_dir in "${sorted_deployments[@]}"; do
        if (( kept < MAX_DEPLOYMENTS )); then
            kept=$((kept + 1))
            continue
        fi

        # Never remove the currently active deployment
        if [[ "${deploy_dir}" == "${active_path}" ]]; then
            log_info "  Skipping active deployment: ${deploy_dir}"
            continue
        fi

        log_info "  Removing old deployment: ${deploy_dir}"
        rm -rf "${deploy_dir}"
        pruned=$((pruned + 1))

        # Remove empty parent version directory
        local parent_dir
        parent_dir="$(dirname "${deploy_dir}")"
        if [[ -d "${parent_dir}" ]] && [[ -z "$(ls -A "${parent_dir}" 2>/dev/null)" ]]; then
            log_info "  Removing empty version directory: ${parent_dir}"
            rmdir "${parent_dir}" 2>/dev/null || true
        fi
    done

    log_info "Pruned ${pruned} old deployment(s). Kept ${kept}."
}

# =============================================================================
# Guard -- refuse to overwrite unless --force
# =============================================================================

guard_existing_deployment() {
    local deploy_target="$1"

    if [[ -d "${deploy_target}" ]]; then
        if [[ "${FORCE}" == true ]]; then
            log_warn "====================================================="
            log_warn "OVERWRITING existing deployment at:"
            log_warn "  ${deploy_target}"
            log_warn "====================================================="
        else
            log_error "Deployment directory already exists:"
            log_error "  ${deploy_target}"
            log_error ""
            log_error "This version+sha has already been deployed."
            log_error "To overwrite, re-run with --force:"
            log_error "  ${SCRIPT_NAME} --execute --force"
            exit 1
        fi
    fi
}

# =============================================================================
# Preview
# =============================================================================

count_files() {
    local dir="$1"
    if [[ -d "${dir}" ]]; then
        # -maxdepth 5: prevent runaway traversal in deeply nested trees
        # -type f: matches only regular files (symlinks are excluded by default
        #   since find does not follow them without -L)
        find "${dir}" -maxdepth 5 -type f | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

show_preview() {
    local repo_root="$1"
    local version="$2"
    local git_sha="$3"
    local deploy_target="$4"
    local compose_project="$5"

    log_step "Deployment Preview"

    log_info "Source repository:    ${repo_root}"
    log_info "Version:             ${version}"
    log_info "Git SHA:             ${git_sha}"
    log_info "Deploy target:       ${deploy_target}"
    log_info "Compose project:     ${compose_project}"
    log_info "Compose profile:     ${COMPOSE_PROFILE}"
    log_info "Mode:                ${MODE}"
    log_info "Force overwrite:     ${FORCE}"
    log_info "Restart containers:  ${RESTART}"
    log_info ""
    log_info "File counts (source):"
    log_info "  src/omnibase_infra/  $(count_files "${repo_root}/src/omnibase_infra") files"
    log_info "  contracts/           $(count_files "${repo_root}/contracts") files"
    log_info "  docker/              $(count_files "${repo_root}/docker") files"

    # .env strategy
    if [[ -d "${deploy_target}" && -f "${deploy_target}/docker/.env" ]]; then
        log_info "  .env strategy:       preserve existing"
    elif [[ -f "${repo_root}/docker/.env" ]]; then
        log_info "  .env strategy:       copy from repo docker/.env"
    elif [[ -f "${repo_root}/docker/.env.example" ]]; then
        log_info "  .env strategy:       copy from .env.example (WARNING: edit before use)"
    else
        log_info "  .env strategy:       none available (WARNING: compose will fail)"
    fi
}

# =============================================================================
# Sync -- rsync repository to deployment target
# =============================================================================

sync_files() {
    local repo_root="$1"
    local deploy_target="$2"

    log_step "Sync Files"

    mkdir -p "${deploy_target}/docker"

    # 1. Root files (pyproject.toml, poetry.lock, README.md, LICENSE)
    log_info "Syncing root files..."
    log_cmd "rsync pyproject.toml, poetry.lock, README.md, LICENSE"
    rsync -a \
        "${repo_root}/pyproject.toml" \
        "${repo_root}/poetry.lock" \
        "${deploy_target}/"

    # Copy README.md and LICENSE if they exist (optional files)
    for f in README.md LICENSE; do
        if [[ -f "${repo_root}/${f}" ]]; then
            rsync -a "${repo_root}/${f}" "${deploy_target}/"
        fi
    done

    # 2. Source code
    log_info "Syncing src/ directory..."
    log_cmd "rsync -a --delete src/ -> deployed"
    rsync -a --delete \
        "${repo_root}/src/" "${deploy_target}/src/"

    # 3. Contracts
    log_info "Syncing contracts/..."
    log_cmd "rsync -a --delete contracts/ -> deployed"
    rsync -a --delete \
        "${repo_root}/contracts/" "${deploy_target}/contracts/"

    # 4. Docker files -- with preserve allowlist
    #    .env, .env.local, certs/, overrides/ survive --delete
    #    Excludes use a leading '/' to anchor them to the transfer root (docker/),
    #    so only top-level .env and .env.local are excluded; nested .env files in
    #    subdirectories are synced normally.
    log_info "Syncing docker/ (preserving .env, .env.local, certs/, overrides/)..."
    log_cmd "rsync -a --delete --exclude='/.env' --exclude='/.env.local' --exclude='/certs/' --exclude='/overrides/' docker/ -> deployed"
    # Note: -a preserves source permissions, but the .env file is excluded from
    # rsync and instead copied via install -m 600 in setup_env() for restricted perms.
    rsync -a --delete \
        --exclude='/.env' \
        --exclude='/.env.local' \
        --exclude='/certs/' \
        --exclude='/overrides/' \
        "${repo_root}/docker/" "${deploy_target}/docker/"

    log_info "Sync complete."
}

# =============================================================================
# Env Setup -- ensure .env exists in deployment target
# =============================================================================

setup_env() {
    local repo_root="$1"
    local deploy_target="$2"
    local docker_dir="${deploy_target}/docker"

    log_step "Environment Setup"

    if [[ -f "${docker_dir}/.env" ]]; then
        log_info ".env already exists in deployment -- preserving."
        return 0
    fi

    # Try to copy from repo's docker/.env
    # Use install -m 600 to atomically create the file with correct permissions,
    # avoiding a race window where the file briefly has default (world-readable) perms.
    if [[ -f "${repo_root}/docker/.env" ]]; then
        log_info "Copying .env from source repo docker/.env"
        install -m 600 "${repo_root}/docker/.env" "${docker_dir}/.env"
        return 0
    fi

    # Fall back to .env.example
    if [[ -f "${repo_root}/docker/.env.example" ]]; then
        log_warn "No .env found. Copying .env.example as .env."
        log_warn "You MUST edit ${docker_dir}/.env before running containers."
        log_warn "At minimum, set POSTGRES_PASSWORD to a secure value."
        install -m 600 "${repo_root}/docker/.env.example" "${docker_dir}/.env"
        return 0
    fi

    log_warn "No .env or .env.example found. Docker compose may fail without it."
}

# =============================================================================
# Sanity Check -- validate compose can resolve all paths
# =============================================================================

sanity_check() {
    local deploy_target="$1"
    local compose_project="$2"
    local compose_file="${deploy_target}/docker/docker-compose.infra.yml"

    log_step "Post-Sync Sanity Check"

    log_info "Validating compose configuration from deployed directory..."
    log_cmd "docker compose -p ${compose_project} -f ${compose_file} config --quiet"

    local env_file_args=()
    if [[ -f "${deploy_target}/docker/.env" ]]; then
        env_file_args=(--env-file "${deploy_target}/docker/.env")
    fi

    local config_output
    if ! config_output="$(docker compose \
        -p "${compose_project}" \
        -f "${compose_file}" \
        ${env_file_args[@]+"${env_file_args[@]}"} \
        config --quiet 2>&1)"; then
        log_error "Compose configuration validation failed."
        if [[ -n "${config_output}" ]]; then
            log_error "Compose output:"
            while IFS= read -r line; do
                log_error "  ${line}"
            done <<< "${config_output}"
        fi
        log_error "The deployed directory structure may be incomplete."
        log_error "Check that src/, contracts/, and docker/ are properly synced."
        exit 1
    fi

    log_info "Compose configuration is valid."
}

# =============================================================================
# Registry -- atomic write of deployment metadata
# =============================================================================

write_registry() {
    local version="$1"
    local git_sha="$2"
    local deploy_target="$3"
    local repo_root="$4"
    local compose_project="$5"

    log_step "Write Registry"

    local deployed_at
    deployed_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    local tmp_file="${REGISTRY_FILE}.tmp"

    # Restrict temp file permissions to 600 (owner-only read/write) to prevent
    # other users from reading deployment metadata while the file is being written.
    local old_umask
    old_umask="$(umask)"
    umask 077

    jq -n \
        --arg active_version "${version}" \
        --arg git_sha "${git_sha}" \
        --arg deploy_path "${deploy_target}" \
        --arg source_repo "${repo_root}" \
        --arg deployed_at "${deployed_at}" \
        --arg compose_project "${compose_project}" \
        --arg profile "${COMPOSE_PROFILE}" \
        '{
            active_version: $active_version,
            git_sha: $git_sha,
            deploy_path: $deploy_path,
            source_repo: $source_repo,
            deployed_at: $deployed_at,
            compose_project: $compose_project,
            profile: $profile
        }' > "${tmp_file}"

    # Restore original umask before continuing
    umask "${old_umask}"

    # Atomic rename
    mv "${tmp_file}" "${REGISTRY_FILE}"

    log_info "Registry written: ${REGISTRY_FILE}"
    log_info "  version:         ${version}"
    log_info "  git_sha:         ${git_sha}"
    log_info "  deployed_at:     ${deployed_at}"
    log_info "  compose_project: ${compose_project}"
}

# =============================================================================
# Build -- docker compose build with VCS_REF label
# =============================================================================

build_images() {
    local deploy_target="$1"
    local compose_project="$2"
    local git_sha="$3"
    local compose_file="${deploy_target}/docker/docker-compose.infra.yml"

    log_step "Build Images"

    local build_date
    build_date="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    local env_file_args=()
    if [[ -f "${deploy_target}/docker/.env" ]]; then
        env_file_args=(--env-file "${deploy_target}/docker/.env")
    fi

    local cmd=(
        docker compose
        -p "${compose_project}"
        -f "${compose_file}"
        ${env_file_args[@]+"${env_file_args[@]}"}
        --profile "${COMPOSE_PROFILE}"
        build
        --build-arg "VCS_REF=${git_sha}"
        --build-arg "BUILD_DATE=${build_date}"
    )

    log_info "Building images with VCS_REF=${git_sha}..."
    log_cmd "${cmd[*]}"

    "${cmd[@]}"

    log_info "Image build complete."
}

# =============================================================================
# Restart -- bring up runtime services only
# =============================================================================

restart_services() {
    local deploy_target="$1"
    local compose_project="$2"
    local compose_file="${deploy_target}/docker/docker-compose.infra.yml"

    log_step "Restart Runtime Services"

    local env_file_args=()
    if [[ -f "${deploy_target}/docker/.env" ]]; then
        env_file_args=(--env-file "${deploy_target}/docker/.env")
    fi

    local cmd=(
        docker compose
        -p "${compose_project}"
        -f "${compose_file}"
        ${env_file_args[@]+"${env_file_args[@]}"}
        --profile "${COMPOSE_PROFILE}"
        up -d --no-deps --force-recreate
        "${RUNTIME_SERVICES[@]}"
    )

    log_info "Restarting services: ${RUNTIME_SERVICES[*]}"
    log_cmd "${cmd[*]}"

    "${cmd[@]}"

    log_info "Services restarted."
}

# =============================================================================
# Verify -- health check + label inspection + log sentinels
# =============================================================================

verify_deployment() {
    local git_sha="$1"
    local compose_project="$2"

    log_step "Verify Deployment"

    # 1. Health endpoint
    log_info "Checking health endpoint (${HEALTH_CHECK_URL})..."
    local attempt=0
    local healthy=false

    while (( attempt < HEALTH_CHECK_RETRIES )); do
        attempt=$((attempt + 1))
        if curl -sf --connect-timeout 2 --max-time 5 "${HEALTH_CHECK_URL}" >/dev/null 2>&1; then
            healthy=true
            break
        fi
        log_info "  Attempt ${attempt}/${HEALTH_CHECK_RETRIES} -- waiting ${HEALTH_CHECK_INTERVAL}s..."
        sleep "${HEALTH_CHECK_INTERVAL}"
    done

    if [[ "${healthy}" == true ]]; then
        log_info "Health check passed."
    else
        log_warn "Health check failed after ${HEALTH_CHECK_RETRIES} attempts."
        log_warn "The service may still be starting. Check manually:"
        log_warn "  curl ${HEALTH_CHECK_URL}"
    fi

    # 2. Resolve runtime container ID (supports dynamic compose project names)
    log_info "Checking image labels for VCS_REF..."
    local container_id
    container_id="$(docker ps -q --filter "name=${compose_project}-omninode-runtime" | head -1)"
    if [[ -z "${container_id}" ]]; then
        container_id="$(docker ps -q --filter "name=omninode-runtime" | head -1)"
    fi

    if [[ -z "${container_id}" ]]; then
        log_warn "Could not resolve container ID for omninode-runtime; skipping label/log checks."
        return 0
    fi

    # 3. Image label verification
    local label
    label="$(docker inspect "${container_id}" \
        --format='{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || true)"

    if [[ "${label}" == "${git_sha}" ]]; then
        log_info "Image label matches: org.opencontainers.image.revision=${label}"
    elif [[ -n "${label}" ]]; then
        log_warn "Image label mismatch:"
        log_warn "  Expected: ${git_sha}"
        log_warn "  Found:    ${label}"
        log_warn "The running container may be from a previous build."
    else
        log_warn "Could not read image label (container may not exist yet)."
    fi

    # 4. Log sentinel: entrypoint ran
    log_info "Checking log sentinels..."
    local logs
    logs="$(docker logs "${container_id}" 2>&1 | tail -50 || true)"

    if echo "${logs}" | grep -q "Schema fingerprint stamped"; then
        log_info "Sentinel found: 'Schema fingerprint stamped' (entrypoint ran)."
    else
        log_warn "Sentinel not found: 'Schema fingerprint stamped'"
        log_warn "The entrypoint may not have completed yet."
    fi
}

# =============================================================================
# Print Compose Commands
# =============================================================================

print_compose_commands() {
    local deploy_target="$1"
    local compose_project="$2"
    local git_sha="$3"
    local compose_file="${deploy_target}/docker/docker-compose.infra.yml"
    local env_file="${deploy_target}/docker/.env"

    # Only include --env-file in printed commands when the .env file exists,
    # matching the conditional behavior used by build_images and restart_services.
    local env_file_line=""
    if [[ -f "${env_file}" ]]; then
        env_file_line="    --env-file ${env_file} \\"
    fi

    log_step "Compose Commands"

    log_info "These are the exact commands this script would run from the deployed directory."
    if [[ -z "${env_file_line}" ]]; then
        log_warn "No .env file found at ${env_file} -- --env-file omitted from commands."
    fi
    log_info ""
    log_info "Build:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    [[ -n "${env_file_line}" ]] && log_info "${env_file_line}"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    build \\"
    log_info "    --build-arg VCS_REF=${git_sha} \\"
    log_info "    --build-arg BUILD_DATE=\$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")"
    log_info ""
    log_info "Restart runtime services:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    [[ -n "${env_file_line}" ]] && log_info "${env_file_line}"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    up -d --no-deps --force-recreate \\"
    log_info "    ${RUNTIME_SERVICES[*]}"
    log_info ""
    log_info "Full stack up (infra + runtime):"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    [[ -n "${env_file_line}" ]] && log_info "${env_file_line}"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    up -d"
    log_info ""
    log_info "Stop all:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    [[ -n "${env_file_line}" ]] && log_info "${env_file_line}"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    down"
    log_info ""
    log_info "Logs:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    [[ -n "${env_file_line}" ]] && log_info "${env_file_line}"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    logs -f"
    log_info ""
    log_info "Status:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    [[ -n "${env_file_line}" ]] && log_info "${env_file_line}"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    ps"
}

# =============================================================================
# Summary
# =============================================================================

show_summary() {
    local deploy_target="$1"
    local version="$2"
    local git_sha="$3"
    local compose_project="$4"

    log_step "Deployment Summary"

    log_info "Deploy path:       ${deploy_target}"
    log_info "Version:           ${version}"
    log_info "Git SHA:           ${git_sha}"
    log_info "Compose project:   ${compose_project}"
    log_info "Profile:           ${COMPOSE_PROFILE}"
    log_info "Registry:          ${REGISTRY_FILE}"
    log_info ""
    log_info "Next steps:"

    # Only include --env-file in printed commands when the .env file exists,
    # matching the conditional behavior used by build_images, restart_services,
    # and print_compose_commands.
    local env_file="${deploy_target}/docker/.env"
    local env_file_line=""
    if [[ -f "${env_file}" ]]; then
        env_file_line="      --env-file ${env_file} \\"
    fi

    if [[ "${RESTART}" == false ]]; then
        log_info "  To start containers, run:"
        log_info "    docker compose \\"
        log_info "      -p ${compose_project} \\"
        log_info "      -f ${deploy_target}/docker/docker-compose.infra.yml \\"
        [[ -n "${env_file_line}" ]] && log_info "${env_file_line}"
        log_info "      --profile ${COMPOSE_PROFILE} \\"
        log_info "      up -d"
    else
        log_info "  Containers are running. Check status:"
        log_info "    docker compose \\"
        log_info "      -p ${compose_project} \\"
        log_info "      -f ${deploy_target}/docker/docker-compose.infra.yml \\"
        [[ -n "${env_file_line}" ]] && log_info "${env_file_line}"
        log_info "      --profile ${COMPOSE_PROFILE} \\"
        log_info "      ps"
    fi

    log_info ""
    log_info "  Verify deployment:"
    log_info "    cat ${REGISTRY_FILE} | jq ."
    log_info "    docker inspect omninode-runtime --format='{{index .Config.Labels \"org.opencontainers.image.revision\"}}'"
}

# =============================================================================
# Main
# =============================================================================

main() {
    parse_args "$@"

    # Phase 1: Validate prerequisites
    validate_prerequisites

    # Resolve repository root
    local repo_root
    repo_root="$(resolve_repo_root)"
    log_info "Repository root: ${repo_root}"

    # Validate repo structure
    validate_repo_structure "${repo_root}"

    # Phase 2: Identity -- version + git SHA
    log_step "Build Identity"
    local version git_sha
    version="$(read_version "${repo_root}")"
    git_sha="$(read_git_sha "${repo_root}")"

    # Validate version format before using it in path construction.
    # A malformed version could create unexpected directory structures.
    # Policy: only stable release versions (MAJOR.MINOR.PATCH) are allowed for
    # deployment. Pre-release suffixes (e.g., 1.2.3-rc.1, 1.2.3-beta) are
    # intentionally rejected to ensure only tested releases reach production.
    if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_error "Invalid version format: '${version}'"
        log_error "Expected semantic version (e.g., 1.2.3). Check pyproject.toml [tool.poetry] version."
        exit 1
    fi

    # Validate git SHA format before using it in directory path construction.
    # A malformed SHA could create unexpected directory structures or indicate
    # a corrupted git state.
    if [[ ! "${git_sha}" =~ ^[0-9a-f]+$ ]]; then
        log_error "Invalid git SHA format: '${git_sha}'"
        log_error "Expected lowercase hex string. Is the git repository corrupted?"
        exit 1
    fi

    log_info "Version: ${version}"
    log_info "Git SHA: ${git_sha}"
    check_git_dirty "${repo_root}"

    # Compute paths
    local deploy_target="${DEPLOY_ROOT}/deployed/${version}/${git_sha}"
    local compose_project="omnibase-infra-${COMPOSE_PROFILE}"

    # --print-compose-cmd: show commands and exit
    if [[ "${PRINT_COMPOSE_CMD}" == true ]]; then
        print_compose_commands "${deploy_target}" "${compose_project}" "${git_sha}"
        exit 0
    fi

    # Phase 3: Preview
    show_preview "${repo_root}" "${version}" "${git_sha}" "${deploy_target}" "${compose_project}"

    # Dry-run mode: stop here
    if [[ "${MODE}" == "dry-run" ]]; then
        log_step "Dry Run Complete"
        log_info "No changes were made. To deploy, re-run with --execute:"
        log_info "  ${SCRIPT_NAME} --execute"
        exit 0
    fi

    # =========================================================================
    # Execute mode from here
    # =========================================================================

    # Phase 4: Lock
    acquire_lock

    # Phase 5: Guard
    guard_existing_deployment "${deploy_target}"

    # Phase 6: Sync
    sync_files "${repo_root}" "${deploy_target}"

    # Mark deployment directory for cleanup on failure. If registry write or
    # build fails after rsync, cleanup_on_exit() will remove this orphaned
    # directory (unless registry.json already points to it).
    DEPLOY_DIR_TO_CLEANUP="${deploy_target}"

    # Phase 7: Env setup
    setup_env "${repo_root}" "${deploy_target}"

    # Phase 8: Sanity check
    sanity_check "${deploy_target}" "${compose_project}"

    # Phase 9: Registry
    write_registry "${version}" "${git_sha}" "${deploy_target}" "${repo_root}" "${compose_project}"

    # Registry now points to this deployment -- disable partial cleanup
    DEPLOY_DIR_TO_CLEANUP=""

    # Phase 10: Build
    build_images "${deploy_target}" "${compose_project}" "${git_sha}"

    # Phase 11: Restart (optional)
    if [[ "${RESTART}" == true ]]; then
        restart_services "${deploy_target}" "${compose_project}"
    fi

    # Phase 12: Verify (only with --restart)
    if [[ "${RESTART}" == true ]]; then
        verify_deployment "${git_sha}" "${compose_project}"
    fi

    # Phase 13: Summary
    show_summary "${deploy_target}" "${version}" "${git_sha}" "${compose_project}"

    # Phase 14: Prune old deployments
    prune_old_deployments
}

main "$@"
