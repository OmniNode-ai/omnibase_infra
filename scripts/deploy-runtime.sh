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
readonly LOCK_FILE="${DEPLOY_ROOT}/.deploy.lock"

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
readonly HEALTH_CHECK_URL="http://localhost:8085/health"
readonly HEALTH_CHECK_RETRIES=15
readonly HEALTH_CHECK_INTERVAL=4

# =============================================================================
# Defaults
# =============================================================================

MODE="dry-run"           # dry-run | execute
FORCE=false
RESTART=false
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
    +-- .deploy.lock                        flock concurrency guard
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
                if [[ -z "${2:-}" ]]; then
                    log_error "--profile requires a value"
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

    # Extract version from pyproject.toml using grep + sed (no Python dependency)
    # Use -E for extended regex; [[:space:]] for BSD sed compatibility (\s not supported)
    version="$(grep -E '^version[[:space:]]*=' "${repo_root}/pyproject.toml" | head -1 | sed -E 's/.*=[[:space:]]*"([^"]+)".*/\1/')"

    if [[ -z "${version}" ]]; then
        log_error "Could not read version from pyproject.toml"
        exit 1
    fi

    echo "${version}"
}

read_git_sha() {
    local repo_root="$1"
    local sha

    sha="$(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || true)"

    if [[ -z "${sha}" ]]; then
        log_error "Could not determine git SHA. Is this a git repository?"
        exit 1
    fi

    echo "${sha}"
}

check_git_dirty() {
    local repo_root="$1"
    if ! git -C "${repo_root}" diff --quiet HEAD 2>/dev/null; then
        log_warn "Working tree has uncommitted changes."
        log_warn "The deployed SHA will not match the actual file contents."
    fi
}

# =============================================================================
# Concurrency Lock
# =============================================================================

acquire_lock() {
    mkdir -p "${DEPLOY_ROOT}"

    # Open lock file on fd 200
    exec 200>"${LOCK_FILE}"

    if ! flock -n 200; then
        log_error "Another deployment is in progress (locked by ${LOCK_FILE})."
        log_error "If the previous deployment crashed, remove the lock manually:"
        log_error "  rm ${LOCK_FILE}"
        exit 2
    fi

    log_info "Acquired deployment lock."
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
        find "${dir}" -type f | wc -l | tr -d ' '
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
    log_info "Syncing src/omnibase_infra/..."
    log_cmd "rsync -a --delete src/omnibase_infra/ -> deployed"
    rsync -a --delete \
        "${repo_root}/src/" "${deploy_target}/src/"

    # 3. Contracts
    log_info "Syncing contracts/..."
    log_cmd "rsync -a --delete contracts/ -> deployed"
    rsync -a --delete \
        "${repo_root}/contracts/" "${deploy_target}/contracts/"

    # 4. Docker files -- with preserve allowlist
    #    .env, .env.local, certs/, overrides/ survive --delete
    log_info "Syncing docker/ (preserving .env, .env.local, certs/, overrides/)..."
    log_cmd "rsync -a --delete --exclude='.env' --exclude='.env.local' --exclude='certs/' --exclude='overrides/' docker/ -> deployed"
    rsync -a --delete \
        --exclude='.env' \
        --exclude='.env.local' \
        --exclude='certs/' \
        --exclude='overrides/' \
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
    if [[ -f "${repo_root}/docker/.env" ]]; then
        log_info "Copying .env from source repo docker/.env"
        cp "${repo_root}/docker/.env" "${docker_dir}/.env"
        return 0
    fi

    # Fall back to .env.example
    if [[ -f "${repo_root}/docker/.env.example" ]]; then
        log_warn "No .env found. Copying .env.example as .env."
        log_warn "You MUST edit ${docker_dir}/.env before running containers."
        log_warn "At minimum, set POSTGRES_PASSWORD to a secure value."
        cp "${repo_root}/docker/.env.example" "${docker_dir}/.env"
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

    if ! docker compose \
        -p "${compose_project}" \
        -f "${compose_file}" \
        --env-file "${deploy_target}/docker/.env" \
        config --quiet 2>&1; then
        log_error "Compose configuration validation failed."
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

    local cmd=(
        docker compose
        -p "${compose_project}"
        -f "${compose_file}"
        --env-file "${deploy_target}/docker/.env"
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

    local cmd=(
        docker compose
        -p "${compose_project}"
        -f "${compose_file}"
        --env-file "${deploy_target}/docker/.env"
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
        if curl -sf "${HEALTH_CHECK_URL}" >/dev/null 2>&1; then
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

    # 2. Image label verification
    log_info "Checking image labels for VCS_REF..."
    local label
    label="$(docker inspect omninode-runtime \
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

    # 3. Log sentinel: entrypoint ran
    log_info "Checking log sentinels..."
    local logs
    logs="$(docker logs omninode-runtime 2>&1 | tail -50 || true)"

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

    log_step "Compose Commands"

    log_info "These are the exact commands this script would run from the deployed directory."
    log_info ""
    log_info "Build:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    log_info "    --env-file ${env_file} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    build \\"
    log_info "    --build-arg VCS_REF=${git_sha} \\"
    log_info "    --build-arg BUILD_DATE=\$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")"
    log_info ""
    log_info "Restart runtime services:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    log_info "    --env-file ${env_file} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    up -d --no-deps --force-recreate \\"
    log_info "    ${RUNTIME_SERVICES[*]}"
    log_info ""
    log_info "Full stack up (infra + runtime):"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    log_info "    --env-file ${env_file} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    up -d"
    log_info ""
    log_info "Stop all:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    log_info "    --env-file ${env_file} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    down"
    log_info ""
    log_info "Logs:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    log_info "    --env-file ${env_file} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    logs -f"
    log_info ""
    log_info "Status:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    -f ${compose_file} \\"
    log_info "    --env-file ${env_file} \\"
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

    if [[ "${RESTART}" == false ]]; then
        log_info "  To start containers, run:"
        log_info "    docker compose \\"
        log_info "      -p ${compose_project} \\"
        log_info "      -f ${deploy_target}/docker/docker-compose.infra.yml \\"
        log_info "      --env-file ${deploy_target}/docker/.env \\"
        log_info "      --profile ${COMPOSE_PROFILE} \\"
        log_info "      up -d"
    else
        log_info "  Containers are running. Check status:"
        log_info "    docker compose \\"
        log_info "      -p ${compose_project} \\"
        log_info "      -f ${deploy_target}/docker/docker-compose.infra.yml \\"
        log_info "      --env-file ${deploy_target}/docker/.env \\"
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

    # Phase 7: Env setup
    setup_env "${repo_root}" "${deploy_target}"

    # Phase 8: Sanity check
    sanity_check "${deploy_target}" "${compose_project}"

    # Phase 9: Registry
    write_registry "${version}" "${git_sha}" "${deploy_target}" "${repo_root}" "${compose_project}"

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
}

main "$@"
