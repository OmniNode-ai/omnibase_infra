#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# deploy-gateway.sh -- Sanctioned deploy path for the .201 omninode-gateway lane
# (OMN-15521).
#
# Before this script, the compose project `omninode-gateway` (the standalone
# operator-edge forwarder at /opt/omninode/gateway, systemd unit
# onex-gateway-forwarder) had no repo-resident deploy path at all -- it was
# stood up by hand-copying docker/docker-compose.gateway.yml and
# docker/gateway/beta-gateway-canary.yaml into a root-owned directory and
# building/running compose there directly. That left the lane invisible to
# deploy-runtime.sh (whose -p omnibase-infra scope never touches it), stamped
# with no org.opencontainers.image.revision / com.omninode.build_source
# labels, and with no recorded rollback target.
#
# This script builds the gateway-forwarder image FROM THIS REPO CHECKOUT (the
# same src/omnibase_infra/ tree deploy-runtime.sh's dev lane builds from -- pull
# to the merged-dev tip first), stamps the same OCI provenance labels every
# omnibase-infra runtime container carries, pins the running container to the
# resulting image DIGEST (never a moving :latest tag -- the existing systemd
# unit's ExecStartPre already asserts GATEWAY_IMAGE is a real sha256 digest),
# and records a rollback target the same way the omnibase-infra lane does via
# registry.json.
#
# Scope: this is the .201 gateway lane ONLY (compose project
# `omninode-gateway`). It does not touch the omnibase-infra runtime lane, does
# not run migrations, and does not restart any RUNTIME_SERVICES container.
#
# Usage:
#   ./scripts/deploy-gateway.sh                    # Dry-run preview (default)
#   ./scripts/deploy-gateway.sh --execute           # Build + deploy + reload
#   ./scripts/deploy-gateway.sh --print-compose-cmd # Show the exact build command
#   ./scripts/deploy-gateway.sh --help              # Full usage
#
# Runbook: docs/runbooks/gateway-lane-deploy.md

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_NAME
readonly SCRIPT_VERSION="1.0.0"

readonly COMPOSE_PROJECT="omninode-gateway"
readonly SERVICE_NAME="gateway-forwarder"
readonly CONTAINER_NAME="omninode-gateway-forwarder"
readonly SYSTEMD_UNIT="onex-gateway-forwarder"

# Host paths the running lane reads from. Overridable so tests can point them
# at a scratch directory instead of the real root-owned locations.
GATEWAY_HOST_DIR="${GATEWAY_HOST_DIR:-/opt/omninode/gateway}"
GATEWAY_ENV_FILE="${GATEWAY_ENV_FILE:-/etc/omninode/gateway/gateway.env}"
GATEWAY_REGISTRY_DIR="${GATEWAY_REGISTRY_DIR:-${HOME}/.omnibase/gateway}"
readonly GATEWAY_REGISTRY_FILE="${GATEWAY_REGISTRY_DIR}/registry.json"

# Build-time image tag. The running lane is pinned to a resolved digest (see
# resolve_image_digest), never to this moving tag -- it exists only so
# `docker compose build` has something to name the image it produces.
readonly BUILD_IMAGE_TAG="docker-gateway-forwarder:build"

# =============================================================================
# Logging
# =============================================================================

log_info() { printf '[deploy-gateway] %s\n' "$*"; }
log_warn() { printf '[deploy-gateway] WARNING: %s\n' "$*" >&2; }
log_error() { printf '[deploy-gateway] ERROR: %s\n' "$*" >&2; }
log_step() { printf '\n[deploy-gateway] === %s ===\n' "$*"; }
log_cmd() { printf '[deploy-gateway]   > %s\n' "$*"; }

# =============================================================================
# Usage
# =============================================================================

usage() {
    cat <<EOF
${SCRIPT_NAME} v${SCRIPT_VERSION} -- Sanctioned deploy path for the .201
omninode-gateway lane (OMN-15521)

Builds the gateway-forwarder image from THIS repo checkout, stamps OCI
provenance labels, pins the running container to the resulting image digest,
and records a rollback target. Run from the canonical omnibase_infra clone on
the host where the lane's containers live (.201) -- same convention as
scripts/deploy-runtime.sh and scripts/runtime_build/refresh_stability_lane.sh.

USAGE
    ${SCRIPT_NAME} [OPTIONS]

OPTIONS
    (none)              Dry-run mode (default). Preview without mutating anything.
    --execute           Actually build the image, sync host files, update
                         gateway.env, and reload the systemd unit.
    --print-compose-cmd Print the exact 'docker compose build' command and exit.
    --skip-reload       With --execute: build + sync + write gateway.env, but do
                         NOT reload the systemd unit (leaves the running
                         container on the previous digest until a manual reload).
    --help               Show this help message and exit.

REQUIRED ENVIRONMENT (--execute only)
    Sourced from ${GATEWAY_ENV_FILE} (override via GATEWAY_ENV_FILE): the
    AWS Roles Anywhere / TPM / container-UID variables the compose file
    requires (GATEWAY_AWS_PROFILE, GATEWAY_AWS_CONFIG_FILE,
    GATEWAY_AWS_CERTIFICATE_FILE, GATEWAY_AWS_PRIVATE_KEY_FILE,
    GATEWAY_AWS_SIGNING_HELPER_FILE, GATEWAY_TPM_DEVICE, GATEWAY_TPM_GROUP_ID,
    GATEWAY_CONTAINER_UID, GATEWAY_CONTAINER_GID). This script does not invent
    these -- it reads the same file the systemd unit already reads.

WHAT --execute DOES, IN ORDER
    1. Resolve repo root, version (pyproject.toml), and git SHA (HEAD).
    2. Build the image with the same OCI provenance build-args every
       omnibase-infra runtime container gets (VCS_REF, RUNTIME_VERSION,
       BUILD_DATE, COMPOSE_PROJECT, RUNTIME_SOURCE_HASH, PROMOTION_CLASS,
       NON_MAIN_LINEAGE).
    3. Resolve the built image's digest (sha256:<64 hex>).
    4. Sync docker/docker-compose.gateway.yml and
       docker/gateway/beta-gateway-canary.yaml from this checkout into
       ${GATEWAY_HOST_DIR} (root-owned, mode 0444 -- same posture the files
       already have), replacing the hand-copied originals.
    5. Rewrite ${GATEWAY_ENV_FILE}'s GATEWAY_IMAGE= line to the new digest,
       preserving every other key untouched.
    6. Record the previous digest + this deploy's identity in
       ${GATEWAY_REGISTRY_FILE} (rollback target).
    7. 'systemctl reload ${SYSTEMD_UNIT}' (force-recreates the container on
       the new digest; requires sudo) unless --skip-reload.
    8. Verify: image labels are non-empty, and log whether the container is
       running the new digest.

ROLLBACK
    ${GATEWAY_REGISTRY_FILE} records "previous_digest". To roll back:
      sudo sed -i "s|^GATEWAY_IMAGE=.*|GATEWAY_IMAGE=<previous_digest>|" ${GATEWAY_ENV_FILE}
      sudo systemctl reload ${SYSTEMD_UNIT}
    (mirrors the omnibase-infra lane's manual rollback-via-registry.json
    pattern -- deploy-runtime.sh has no automated --rollback flag either.)

EXAMPLES
    # Preview what would be built and deployed
    ${SCRIPT_NAME}

    # Print the exact build command
    ${SCRIPT_NAME} --print-compose-cmd

    # Build + deploy + reload
    ${SCRIPT_NAME} --execute

    # Verify after deploy
    docker inspect ${CONTAINER_NAME} \\
      --format='rev={{index .Config.Labels "org.opencontainers.image.revision"}} src={{index .Config.Labels "com.omninode.build_source"}}'
    docker exec ${CONTAINER_NAME} ls /app/src/omnibase_infra/nodes/node_bus_forwarder_effect/services/
    docker exec ${CONTAINER_NAME} ls /app/src/omnibase_infra/idempotency/
    diff ${GATEWAY_HOST_DIR}/docker-compose.gateway.yml docker/docker-compose.gateway.yml
EOF
    exit 0
}

# =============================================================================
# Argument parsing
# =============================================================================

MODE="dry-run"
PRINT_COMPOSE_CMD=false
SKIP_RELOAD=false

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --execute)
                MODE="execute"
                shift
                ;;
            --print-compose-cmd)
                PRINT_COMPOSE_CMD=true
                shift
                ;;
            --skip-reload)
                SKIP_RELOAD=true
                shift
                ;;
            --help | -h)
                usage
                ;;
            *)
                log_error "Unknown argument: $1"
                log_error "Run '${SCRIPT_NAME} --help' for usage."
                exit 64
                ;;
        esac
    done
}

# =============================================================================
# Identity
# =============================================================================

resolve_repo_root() {
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
    [[ -f "${repo_root}/docker/docker-compose.gateway.yml" ]] || missing+=("docker/docker-compose.gateway.yml")
    [[ -f "${repo_root}/docker/gateway/beta-gateway-canary.yaml" ]] || missing+=("docker/gateway/beta-gateway-canary.yaml")
    [[ -f "${repo_root}/docker/Dockerfile.runtime" ]] || missing+=("docker/Dockerfile.runtime")
    [[ -d "${repo_root}/src/omnibase_infra/nodes/node_bus_forwarder_effect/services" ]] || missing+=("src/omnibase_infra/nodes/node_bus_forwarder_effect/services/")
    [[ -d "${repo_root}/src/omnibase_infra/idempotency" ]] || missing+=("src/omnibase_infra/idempotency/")
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Repository structure validation failed. Missing:"
        for item in "${missing[@]}"; do
            log_error "  - ${item}"
        done
        exit 1
    fi
}

read_version() {
    local repo_root="$1"
    local version
    version="$(awk '
        /^\[project\]/ { in_section=1; next }
        /^\[/          { in_section=0 }
        in_section && /^version[[:space:]]*=/ {
            gsub(/.*=[[:space:]]*"/, "");
            gsub(/".*/, "");
            print;
            exit
        }
    ' "${repo_root}/pyproject.toml")"
    if [[ -z "${version}" ]]; then
        log_error "Could not read version from pyproject.toml [project] section"
        exit 1
    fi
    echo "${version}"
}

read_git_sha() {
    local repo_root="$1"
    local sha
    sha="$(git -C "${repo_root}" rev-parse --short=12 HEAD 2>/dev/null || true)"
    if [[ -z "${sha}" ]]; then
        log_error "Could not determine git SHA. Is this a git repository?"
        exit 1
    fi
    echo "${sha}"
}

check_git_dirty() {
    local repo_root="$1"
    local status_output
    status_output="$(git -C "${repo_root}" status --porcelain 2>/dev/null || true)"
    if [[ -n "${status_output}" ]]; then
        log_warn "Working tree has uncommitted changes."
        log_warn "The deployed SHA will not match the actual file contents."
    fi
}

# =============================================================================
# Build
# =============================================================================

# resolve_build_args REPO_ROOT GIT_SHA VERSION -- prints the --build-arg argv
# entries (one per line) so both build_image() and print_compose_commands()
# stay in lockstep instead of maintaining two copies of the same list.
resolve_build_args() {
    local repo_root="$1" git_sha="$2" version="$3"
    local build_date
    build_date="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    local build_source="${BUILD_SOURCE:-release}"
    local expected_build_source="${EXPECTED_BUILD_SOURCE:-${build_source}}"
    local promotion_class="clean-main"
    local non_main_lineage="false"
    if [[ "${build_source}" == "workspace" ]]; then
        promotion_class="stability-candidate"
        non_main_lineage="true"
    fi
    cat <<EOF
GIT_SHA=${git_sha}
VCS_REF=${git_sha}
RUNTIME_VERSION=${version}
BUILD_DATE=${build_date}
RUNTIME_SOURCE_HASH=${git_sha}
COMPOSE_PROJECT=${COMPOSE_PROJECT}
BUILD_SOURCE=${build_source}
EXPECTED_BUILD_SOURCE=${expected_build_source}
PROMOTION_CLASS=${promotion_class}
NON_MAIN_LINEAGE=${non_main_lineage}
OMNI_HOME=${OMNI_HOME:-}
EOF
    unset repo_root
}

build_compose_cmd() {
    # build_compose_cmd REPO_ROOT GIT_SHA VERSION -- prints (one token per
    # line, for _read_lines_into_array consumption) the full
    # 'docker compose ... build' argv.
    local repo_root="$1" git_sha="$2" version="$3"
    printf 'docker\ncompose\n-p\n%s\n-f\ndocker/docker-compose.gateway.yml\nbuild\n--progress=plain\n' "${COMPOSE_PROJECT}"
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] || continue
        printf -- '--build-arg\n%s\n' "${arg}"
    done < <(resolve_build_args "${repo_root}" "${git_sha}" "${version}")
    printf '%s\n' "${SERVICE_NAME}"
}

print_compose_commands() {
    # Portable stand-in for bash 4's `readarray`/`mapfile` (unavailable on
    # macOS's shipped /bin/bash 3.2 -- Apple has not updated it past 3.2 since
    # bash moved to GPLv3, and this script must run there like
    # scripts/deploy-runtime.sh and its runtime_build siblings do). Namerefs
    # (`local -n`, bash 4.3+) are equally unavailable, so this loop is
    # inlined at each call site rather than factored into a helper.
    local repo_root="$1" git_sha="$2" version="$3"
    local -a cmd=()
    local line
    while IFS= read -r line; do
        cmd+=("${line}")
    done < <(build_compose_cmd "${repo_root}" "${git_sha}" "${version}")
    log_step "Build command (compose project: ${COMPOSE_PROJECT})"
    log_cmd "GATEWAY_IMAGE=${BUILD_IMAGE_TAG} ${cmd[*]}"
}

build_image() {
    local repo_root="$1" git_sha="$2" version="$3"
    local -a cmd=()
    local line
    while IFS= read -r line; do
        cmd+=("${line}")
    done < <(build_compose_cmd "${repo_root}" "${git_sha}" "${version}")

    log_step "Build Image"
    log_info "Building ${BUILD_IMAGE_TAG} with VCS_REF=${git_sha} RUNTIME_VERSION=${version}..."
    log_cmd "${cmd[*]}"

    (cd "${repo_root}" && GATEWAY_IMAGE="${BUILD_IMAGE_TAG}" "${cmd[@]}")
}

resolve_image_digest() {
    local image_tag="$1"
    local digest
    digest="$(docker image inspect "${image_tag}" --format='{{.Id}}' 2>/dev/null || true)"
    if [[ -z "${digest}" ]]; then
        log_error "Could not resolve digest for built image ${image_tag}"
        exit 1
    fi
    echo "${digest}"
}

# =============================================================================
# Host-file sync (AC4) -- eliminates the 2026-07-29 hand-copy as the source of
# truth. Every deploy re-syncs from this checkout so the host copy can never
# silently drift from a merged commit again.
# =============================================================================

sync_host_files() {
    local repo_root="$1" host_dir="$2"
    log_step "Sync host files -> ${host_dir}"
    sudo install -d -m 0755 -o root -g root "${host_dir}"
    sudo install -d -m 0755 -o root -g root "${host_dir}/gateway"
    sudo install -m 0444 -o root -g root \
        "${repo_root}/docker/docker-compose.gateway.yml" \
        "${host_dir}/docker-compose.gateway.yml"
    sudo install -m 0444 -o root -g root \
        "${repo_root}/docker/gateway/beta-gateway-canary.yaml" \
        "${host_dir}/gateway/beta-gateway-canary.yaml"
    log_info "Synced docker-compose.gateway.yml + gateway/beta-gateway-canary.yaml"
}

verify_host_files_match() {
    local repo_root="$1" host_dir="$2"
    if ! diff -q "${host_dir}/docker-compose.gateway.yml" "${repo_root}/docker/docker-compose.gateway.yml" >/dev/null 2>&1; then
        log_error "AC4 FAILED: ${host_dir}/docker-compose.gateway.yml differs from the repo copy after sync."
        return 1
    fi
    log_info "AC4 OK: ${host_dir}/docker-compose.gateway.yml matches the repo copy."
    return 0
}

# =============================================================================
# gateway.env digest pin
# =============================================================================

current_gateway_image_digest() {
    local env_file="$1"
    if [[ ! -f "${env_file}" ]]; then
        echo ""
        return 0
    fi
    awk -F= '/^GATEWAY_IMAGE=/{print $2; exit}' "${env_file}"
}

update_gateway_env_digest() {
    local env_file="$1" new_digest="$2"
    log_step "Pin GATEWAY_IMAGE -> ${new_digest}"
    local tmp_file
    tmp_file="$(mktemp)"
    awk -v new="GATEWAY_IMAGE=${new_digest}" '
        BEGIN { done=0 }
        /^GATEWAY_IMAGE=/ { print new; done=1; next }
        { print }
        END { if (!done) print new }
    ' "${env_file}" >"${tmp_file}"
    sudo install -m 0444 -o root -g root "${tmp_file}" "${env_file}"
    rm -f "${tmp_file}"
    log_info "gateway.env now pins GATEWAY_IMAGE=${new_digest}"
}

# =============================================================================
# Registry (AC6 -- rollback target, same convention as ~/.omnibase/infra/registry.json)
# =============================================================================

write_registry() {
    local version="$1" git_sha="$2" new_digest="$3" previous_digest="$4" repo_root="$5"
    log_step "Write Registry"
    mkdir -p "${GATEWAY_REGISTRY_DIR}"
    local deployed_at
    deployed_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    local tmp_file="${GATEWAY_REGISTRY_FILE}.tmp"
    jq -n \
        --arg active_version "${version}" \
        --arg git_sha "${git_sha}" \
        --arg active_digest "${new_digest}" \
        --arg previous_digest "${previous_digest}" \
        --arg source_repo "${repo_root}" \
        --arg deployed_at "${deployed_at}" \
        --arg compose_project "${COMPOSE_PROJECT}" \
        --arg gateway_env_file "${GATEWAY_ENV_FILE}" \
        --arg host_compose_file "${GATEWAY_HOST_DIR}/docker-compose.gateway.yml" \
        '{
            active_version: $active_version,
            git_sha: $git_sha,
            active_digest: $active_digest,
            previous_digest: ($previous_digest | select(. != "") // null),
            source_repo: $source_repo,
            deployed_at: $deployed_at,
            compose_project: $compose_project,
            gateway_env_file: $gateway_env_file,
            host_compose_file: $host_compose_file,
            rollback_command: ("sudo sed -i \"s|^GATEWAY_IMAGE=.*|GATEWAY_IMAGE=" + ($previous_digest // "") + "|\" " + $gateway_env_file + " && sudo systemctl reload onex-gateway-forwarder")
        }' >"${tmp_file}"
    mv "${tmp_file}" "${GATEWAY_REGISTRY_FILE}"
    log_info "Registry written: ${GATEWAY_REGISTRY_FILE}"
    log_info "  active_digest:   ${new_digest}"
    log_info "  previous_digest: ${previous_digest:-<none recorded>}"
}

# =============================================================================
# Reload
# =============================================================================

reload_service() {
    log_step "Reload ${SYSTEMD_UNIT}"
    log_cmd "sudo systemctl reload ${SYSTEMD_UNIT}"
    sudo systemctl reload "${SYSTEMD_UNIT}"
}

# =============================================================================
# Verify (AC2 + AC3 probes)
# =============================================================================

verify_deployment() {
    log_step "Verify"
    local rev src
    rev="$(docker inspect "${CONTAINER_NAME}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || true)"
    src="$(docker inspect "${CONTAINER_NAME}" --format '{{index .Config.Labels "com.omninode.build_source"}}' 2>/dev/null || true)"
    if [[ -z "${rev}" ]]; then
        log_error "AC2 FAILED: org.opencontainers.image.revision is empty on ${CONTAINER_NAME}"
        return 1
    fi
    log_info "AC2 OK: org.opencontainers.image.revision=${rev} com.omninode.build_source=${src}"

    if docker exec "${CONTAINER_NAME}" test -f /app/src/omnibase_infra/nodes/node_bus_forwarder_effect/services/service_gateway_delivery.py \
        && docker exec "${CONTAINER_NAME}" test -f /app/src/omnibase_infra/idempotency/store_sqlite.py; then
        log_info "AC3 OK: service_gateway_delivery.py + store_sqlite.py present in the running container."
    else
        log_error "AC3 FAILED: one or both OMN-12912 files are absent from the running container."
        return 1
    fi
    return 0
}

# =============================================================================
# Main
# =============================================================================

main() {
    parse_args "$@"

    local repo_root version git_sha
    repo_root="$(resolve_repo_root)"
    validate_repo_structure "${repo_root}"
    version="$(read_version "${repo_root}")"
    git_sha="$(read_git_sha "${repo_root}")"
    check_git_dirty "${repo_root}"

    if [[ "${PRINT_COMPOSE_CMD}" == true ]]; then
        print_compose_commands "${repo_root}" "${git_sha}" "${version}"
        exit 0
    fi

    log_step "Identity"
    log_info "repo_root: ${repo_root}"
    log_info "version:   ${version}"
    log_info "git_sha:   ${git_sha}"
    log_info "compose_project: ${COMPOSE_PROJECT}"

    if [[ "${MODE}" != "execute" ]]; then
        log_step "Dry Run (default) -- no mutation performed"
        print_compose_commands "${repo_root}" "${git_sha}" "${version}"
        log_info "Would sync ${repo_root}/docker/docker-compose.gateway.yml -> ${GATEWAY_HOST_DIR}/docker-compose.gateway.yml"
        log_info "Would sync ${repo_root}/docker/gateway/beta-gateway-canary.yaml -> ${GATEWAY_HOST_DIR}/gateway/beta-gateway-canary.yaml"
        log_info "Would pin GATEWAY_IMAGE in ${GATEWAY_ENV_FILE} to the resolved build digest"
        log_info "Would write ${GATEWAY_REGISTRY_FILE}"
        if [[ "${SKIP_RELOAD}" != true ]]; then
            log_info "Would run: sudo systemctl reload ${SYSTEMD_UNIT}"
        fi
        log_info "Re-run with --execute to perform this deploy."
        exit 0
    fi

    if [[ ! -f "${GATEWAY_ENV_FILE}" ]]; then
        log_error "GATEWAY_ENV_FILE not found: ${GATEWAY_ENV_FILE}"
        log_error "This file supplies the AWS/TPM/UID variables the compose file requires."
        exit 64
    fi

    local previous_digest
    previous_digest="$(current_gateway_image_digest "${GATEWAY_ENV_FILE}")"
    log_info "Previous GATEWAY_IMAGE (rollback target): ${previous_digest:-<none recorded>}"

    set -a
    # shellcheck source=/dev/null
    source "${GATEWAY_ENV_FILE}"
    set +a

    build_image "${repo_root}" "${git_sha}" "${version}"
    local new_digest
    new_digest="$(resolve_image_digest "${BUILD_IMAGE_TAG}")"
    log_info "Built image digest: ${new_digest}"

    sync_host_files "${repo_root}" "${GATEWAY_HOST_DIR}"
    verify_host_files_match "${repo_root}" "${GATEWAY_HOST_DIR}"

    update_gateway_env_digest "${GATEWAY_ENV_FILE}" "${new_digest}"
    write_registry "${version}" "${git_sha}" "${new_digest}" "${previous_digest}" "${repo_root}"

    if [[ "${SKIP_RELOAD}" == true ]]; then
        log_warn "--skip-reload set: gateway.env is updated but the running container still has the OLD digest until you reload."
        exit 0
    fi

    reload_service
    verify_deployment
    log_step "Done"
    log_info "omninode-gateway deployed: version=${version} git_sha=${git_sha} digest=${new_digest}"
}

main "$@"
