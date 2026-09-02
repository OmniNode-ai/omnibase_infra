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
# registry.json. It also builds the gateway-dns-bastion sidecar's image
# (OMN-16449/OMN-16460) -- gateway-forwarder depends_on it, and the systemd
# unit reloads with --no-build, so a host that has never built the sidecar
# needs this script to build it, not a manual prerequisite step.
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
# Runbook: knowledge-base-internal:runbooks/omnibase-infra-gateway-lane-deploy.md

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_NAME
readonly SCRIPT_VERSION="1.0.0"

readonly COMPOSE_PROJECT="omninode-gateway"
readonly SERVICE_NAME="gateway-forwarder"
readonly CONTAINER_NAME="omninode-gateway-forwarder"
readonly SYSTEMD_UNIT="onex-gateway-forwarder"

# OMN-16449 added this sidecar to docker/docker-compose.gateway.yml --
# gateway-forwarder now depends_on it (condition: service_healthy) via a
# per-container `dns:` key. The systemd unit reloads with `--no-build`
# (docker/gateway/onex-gateway-forwarder.service), so if this script never
# builds the sidecar's image, a host that has never built it fails the
# reload outright (OMN-16460). Its Dockerfile (docker/gateway/dns-bastion/)
# takes no build args (plain `FROM alpine:3.20`), so it does not need the
# OCI-provenance build-arg machinery gateway-forwarder's build uses.
readonly SIDECAR_SERVICE_NAME="gateway-dns-bastion"

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
    GATEWAY_CONTAINER_UID, GATEWAY_CONTAINER_GID, GATEWAY_BROKER_REF_MAP_FILE).
    This script does not invent these -- it reads the same file the systemd
    unit already reads. GATEWAY_BROKER_REF_MAP_FILE (OMN-15743) points at an
    operator-supplied YAML mapping of contract cloud_broker_ref names to
    resolved bootstrap_servers strings, resolved by the forwarder process at
    startup; it replaces the previous hardcoded compose extra_hosts entry
    and is not a value this script or the repo hardcodes.

    BUILD_SOURCE=workspace additionally requires OMNI_HOME to be set (same
    convention as deploy-runtime.sh) so sibling repos can be staged from the
    operator's local clones before the build; the default BUILD_SOURCE=release
    does not stage anything.

WHAT --execute DOES, IN ORDER
    1. Resolve repo root, version (pyproject.toml), and git SHA (HEAD).
    2. Resolve the CONTAINER's currently running image (docker inspect
       --format '{{.Image}}') as the rollback target, and retag it durably as
       ${ROLLBACK_IMAGE_TAG} so it survives the build below moving the
       build tag onto the new image (a bare env-file digest is never used --
       it can go stale relative to what is actually running).
    3. If BUILD_SOURCE=workspace: stage workspace/sibling-repos/ from OMNI_HOME
       (same scripts/runtime_build/stage_workspace.sh deploy-runtime.sh uses)
       and run the OMN-12987 sibling lock-pin preflight -- skipped entirely in
       the default BUILD_SOURCE=release mode.
    4. Build the image with the same OCI provenance build-args every
       omnibase-infra runtime container gets (VCS_REF, RUNTIME_VERSION,
       BUILD_DATE, COMPOSE_PROJECT, RUNTIME_SOURCE_HASH, PROMOTION_CLASS,
       NON_MAIN_LINEAGE, OMNIBASE_COMPAT_REF, OMNIMARKET_REF).
    5. Resolve the built image's digest (sha256:<64 hex>).
    6. Build the ${SIDECAR_SERVICE_NAME} sidecar's image (OMN-16460).
       gateway-forwarder depends_on it (condition: service_healthy) and the
       systemd unit reloads with --no-build, so without this step a host
       that has never built the sidecar fails the reload outright. No
       OCI-provenance build-args -- its Dockerfile takes none.
    7. Sync docker/docker-compose.gateway.yml and
       docker/gateway/beta-gateway-canary.yaml from this checkout into
       ${GATEWAY_HOST_DIR} (root-owned, mode 0444 -- same posture the files
       already have), replacing the hand-copied originals.
    8. Rewrite ${GATEWAY_ENV_FILE}'s GATEWAY_IMAGE= line to the new digest,
       preserving every other key untouched.
    9. Record the previous digest + this deploy's identity in
       ${GATEWAY_REGISTRY_FILE} (rollback target). rollback_command is null
       when no previous image was retained (first deploy, or the previous
       image had already been pruned) -- never a command built from an empty
       digest.
    10. 'systemctl reload ${SYSTEMD_UNIT}' (force-recreates the container on
        the new digest; requires sudo) unless --skip-reload.
    11. Verify: the container is actually running the digest just built (not
        just that labels are non-empty -- a reload that silently fails to
        recreate the container is caught here instead of reporting success),
        image labels are non-empty, and the two OMN-12912 files are present.

ROLLBACK
    ${GATEWAY_REGISTRY_FILE}'s "rollback_command" field carries the exact
    restore command, already pre-filled with "previous_digest" -- the image
    this script retagged as ${ROLLBACK_IMAGE_TAG} before building, so it
    stays resolvable even after a later 'docker image prune'. Read it out of
    the registry and run it verbatim; do NOT reconstruct it by hand:
      jq -r .rollback_command ${GATEWAY_REGISTRY_FILE}
      bash -c "\$(jq -r .rollback_command ${GATEWAY_REGISTRY_FILE})"

    "rollback_command" is null when there is no rollback target -- the first
    deploy ever run against this lane, or a deploy whose previous running
    image had already been pruned before it could be retagged. There is
    nothing to roll back to in that case: deploy forward instead. Do not
    hand-build a substitution from "previous_digest" -- a JSON null printed
    through 'jq -r' renders as the literal string "null", which substitutes
    straight into ${GATEWAY_ENV_FILE}'s GATEWAY_IMAGE= line and corrupts it;
    the systemd unit's ExecStartPre digest-format assertion then refuses to
    start on the next restart/reboot.

    (mirrors the omnibase-infra lane's manual rollback-via-registry.json
    pattern -- deploy-runtime.sh has no automated --rollback flag either.)
    Full procedure: knowledge-base-internal:runbooks/omnibase-infra-gateway-lane-deploy.md

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
    [[ -f "${repo_root}/docker/gateway/dns-bastion/Dockerfile" ]] || missing+=("docker/gateway/dns-bastion/Dockerfile")
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

read_repo_ref_or_main() {
    # Same helper as deploy-runtime.sh's read_repo_ref_or_main (OMN-15521
    # remediation): resolve a full git SHA for a sibling workspace repo when
    # available, falling back to "main" (or the repo's own default via the
    # Dockerfile ARG default) when OMNI_HOME or the sibling clone is absent.
    local repo_path="$1" fallback="$2"
    local sha
    sha="$(git -C "${repo_path}" rev-parse HEAD 2>/dev/null || true)"
    if [[ -n "${sha}" ]]; then
        echo "${sha}"
    else
        echo "${fallback}"
    fi
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
    # OMN-15521 remediation: these sibling-ref build-args are NOT optional
    # extras -- deploy-runtime.sh's build_images() passes them unconditionally
    # on every build_source (scripts/deploy-runtime.sh resolve+pass at
    # build_images()). Omitting them silently falls back to the Dockerfile's
    # hardcoded ARG defaults (OMNIBASE_COMPAT_REF=v0.5.5, OMNIMARKET_REF=dev),
    # which is how the gateway image previously drifted from the
    # omnibase-infra runtime image on the same box.
    #
    # OMN-16296: ONEX_CHANGE_CONTROL_REF was one of these and is now gone --
    # onex_change_control is no longer installed into the runtime image, so the
    # ARG it fed no longer exists and there is no pin left to drift.
    local omni_home="${OMNI_HOME:-}"
    local compat_ref="main"
    local omnimarket_ref="dev"
    if [[ -n "${omni_home}" ]]; then
        compat_ref="$(read_repo_ref_or_main "${omni_home}/omnibase_compat" "main")"
        omnimarket_ref="$(read_repo_ref_or_main "${omni_home}/omnimarket" "dev")"
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
OMNI_HOME=${omni_home}
OMNIBASE_COMPAT_REF=${compat_ref}
OMNIMARKET_REF=${omnimarket_ref}
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

build_sidecar_compose_cmd() {
    # build_sidecar_compose_cmd -- prints (one token per line) the full
    # 'docker compose ... build' argv for the gateway-dns-bastion sidecar
    # (OMN-16460). No --build-arg entries: the sidecar's Dockerfile declares
    # none (plain `FROM alpine:3.20`), unlike gateway-forwarder's
    # OCI-provenance build.
    printf 'docker\ncompose\n-p\n%s\n-f\ndocker/docker-compose.gateway.yml\nbuild\n--progress=plain\n%s\n' \
        "${COMPOSE_PROJECT}" "${SIDECAR_SERVICE_NAME}"
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

    local -a sidecar_cmd=()
    while IFS= read -r line; do
        sidecar_cmd+=("${line}")
    done < <(build_sidecar_compose_cmd)
    log_step "Build command (sidecar: ${SIDECAR_SERVICE_NAME})"
    log_cmd "GATEWAY_IMAGE=${BUILD_IMAGE_TAG} ${sidecar_cmd[*]}"
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

build_sidecar_image() {
    # OMN-16460: build the gateway-dns-bastion sidecar image so the systemd
    # unit's --no-build reload has an image to satisfy gateway-forwarder's
    # depends_on. GATEWAY_IMAGE is exported the same way build_image() does
    # it -- the whole compose file is interpolated even when only building
    # this one service, so gateway-forwarder's `image:
    # "${GATEWAY_IMAGE:?...}"` line must still resolve. Using BUILD_IMAGE_TAG
    # rather than trusting gateway.env's existing value avoids depending on
    # that file already holding a valid digest (e.g. the very first deploy).
    local repo_root="$1"
    local -a cmd=()
    local line
    while IFS= read -r line; do
        cmd+=("${line}")
    done < <(build_sidecar_compose_cmd)

    log_step "Build Sidecar Image (${SIDECAR_SERVICE_NAME})"
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
# Workspace staging (BUILD_SOURCE=workspace) -- OMN-15521 remediation.
#
# resolve_build_args() already honours BUILD_SOURCE=workspace for the label
# values (promotion_class/non_main_lineage), but a prior version of this
# script never actually populated workspace/sibling-repos/ -- unlike
# scripts/deploy-runtime.sh's build_images(), which always calls
# stage_workspace_if_needed() first. docker/Dockerfile.runtime unconditionally
# COPYs workspace/sibling-repos/, so an unstaged workspace build silently used
# the committed placeholder (or whatever stale staging happened to be sitting
# in the checkout) while still stamping workspace-provenance labels. This is
# the same helper deploy-runtime.sh uses -- same underlying
# scripts/runtime_build/stage_workspace.sh and
# scripts/runtime_build/check_sibling_lock_pins.py, invoked the same way, not
# reimplemented -- adapted only to this script's own build-source resolution
# (no COLD_FULL_BRINGUP concept here).
# =============================================================================

stage_workspace_if_needed() {
    # Populate workspace/sibling-repos/ from the operator-selected OMNI_HOME so
    # Dockerfile.runtime can install exact local sibling repo contents.
    local repo_root="$1"
    local build_source omni_home stage_script
    build_source="${BUILD_SOURCE:-release}"
    if [[ "${build_source}" != "workspace" ]]; then
        return 0
    fi

    omni_home="${OMNI_HOME:-}"
    if [[ -z "${omni_home}" ]]; then
        log_error "BUILD_SOURCE=workspace requires OMNI_HOME before staging or build."
        exit 64
    fi

    stage_script="${repo_root}/scripts/runtime_build/stage_workspace.sh"
    if [[ ! -f "${stage_script}" ]]; then
        log_error "Workspace staging script not found: ${stage_script}"
        log_error "Cannot proceed with BUILD_SOURCE=workspace."
        exit 1
    fi

    log_step "Stage Workspace Sibling Repos"
    log_cmd "OMNI_HOME=${omni_home} bash ${stage_script}"
    (cd "${repo_root}" && OMNI_HOME="${omni_home}" bash "${stage_script}")

    check_sibling_lock_pins "${repo_root}" "${omni_home}"
}

check_sibling_lock_pins() {
    # Fail-fast preflight (OMN-12987, same guard deploy-runtime.sh runs): every
    # vendored sibling's version/SHA must match the consuming repo's
    # (omnimarket) uv.lock pin. A stale vendored sibling produced the
    # 2026-06-11 stability crash; this guard refuses to build against one.
    local repo_root="$1"
    local omni_home="$2"
    local guard="${repo_root}/scripts/runtime_build/check_sibling_lock_pins.py"
    if [[ ! -f "${guard}" ]]; then
        log_error "Sibling lock-pin preflight not found: ${guard}"
        log_error "Cannot verify vendored siblings match the consuming lock. Aborting."
        exit 1
    fi

    log_step "Sibling Lock-Pin Preflight (OMN-12987)"
    mkdir -p "${repo_root}/workspace/sibling-repos"
    local provenance_out="${repo_root}/workspace/sibling-repos/.sibling-lock-pins.json"
    local python_bin
    if [[ -x "${repo_root}/.venv/bin/python" ]]; then
        python_bin="${repo_root}/.venv/bin/python"
    elif command -v uv &>/dev/null; then
        python_bin="uv-run"
    elif command -v python3 &>/dev/null; then
        python_bin="python3"
    else
        log_error "No Python interpreter available to run the sibling lock-pin preflight."
        exit 1
    fi

    local lock_path="${omni_home}/omnimarket/uv.lock"
    local guard_args=(
        --lock "${lock_path}"
        --repo "omnibase-infra=${omni_home}/omnibase_infra"
        --repo "omnibase-core=${omni_home}/omnibase_core"
        --repo "omnibase-spi=${omni_home}/omnibase_spi"
        --repo "omnibase-compat=${omni_home}/omnibase_compat"
        --repo "onex-change-control=${omni_home}/onex_change_control"
        --output "${provenance_out}"
        --build-source workspace
    )
    if [[ "${ALLOW_SIBLING_PIN_DRIFT:-0}" == "1" ]]; then
        guard_args+=(--allow-drift)
        log_warn "ALLOW_SIBLING_PIN_DRIFT=1 -- passing --allow-drift to sibling lock-pin preflight (OMN-12977)"
    fi

    log_cmd "OMNI_HOME=${omni_home} ${guard} ${guard_args[*]}"
    if [[ "${python_bin}" == "uv-run" ]]; then
        if ! OMNI_HOME="${omni_home}" uv run --project "${repo_root}" python "${guard}" \
            "${guard_args[@]}"; then
            log_error "Sibling lock-pin preflight FAILED. Refusing to build a stale image."
            exit 1
        fi
    else
        if ! OMNI_HOME="${omni_home}" "${python_bin}" "${guard}" \
            "${guard_args[@]}"; then
            log_error "Sibling lock-pin preflight FAILED. Refusing to build a stale image."
            exit 1
        fi
    fi
    log_info "Sibling lock-pin preflight passed: all vendored siblings match the lock."
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
# Rollback target retention (AC6) -- OMN-15521 remediation.
#
# The rollback target must be derived from what the CONTAINER is actually
# running, never from gateway.env's GATEWAY_IMAGE= line: that line can go
# stale relative to the running container (manual edits, a previous deploy
# that wrote the file but was killed before reload, etc.), which produces a
# recorded "rollback target" that was never the last-known-good image.
#
# It must also be RETAGGED under a durable name before build_image() moves
# BUILD_IMAGE_TAG onto the freshly built image -- otherwise the previous
# image becomes untagged/dangling the instant the build succeeds and is
# eligible for collection by any routine `docker image prune`, so the
# recorded digest resolves to nothing by the time anyone needs it.
# =============================================================================

readonly ROLLBACK_IMAGE_TAG="docker-gateway-forwarder:previous"

resolve_running_container_image() {
    # Prints the image id (sha256:<64 hex>) CONTAINER_NAME is currently
    # running, or empty if the container does not exist yet (first deploy).
    # Read straight off the live container's own state -- this is the true
    # last-known-good and cannot go stale the way gateway.env's line can.
    docker inspect "${CONTAINER_NAME}" --format '{{.Image}}' 2>/dev/null || true
}

retain_previous_image() {
    # Retag the previous running image under ROLLBACK_IMAGE_TAG so it
    # survives the build moving BUILD_IMAGE_TAG onto the new image. Returns
    # non-zero (and the caller must then treat the rollback target as
    # unavailable) if the image no longer resolves locally -- `docker tag`
    # against a missing source image fails, which is exactly the existence
    # check a bare env-file digest never got.
    local previous_digest="$1"
    if [[ -z "${previous_digest}" ]]; then
        log_info "No previous running container image to retain (first deploy)."
        return 1
    fi
    if docker tag "${previous_digest}" "${ROLLBACK_IMAGE_TAG}" 2>/dev/null; then
        log_info "Retained previous image ${previous_digest} as ${ROLLBACK_IMAGE_TAG} (rollback target, survives prune)."
        return 0
    fi
    log_warn "Previous running image ${previous_digest} could not be retagged (already pruned / not local); no rollback target will be recorded for this deploy."
    return 1
}

# =============================================================================
# gateway.env digest pin
# =============================================================================

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
            rollback_command: (
                if ($previous_digest != "") then
                    ("sudo sed -i \"s|^GATEWAY_IMAGE=.*|GATEWAY_IMAGE=" + $previous_digest + "|\" " + $gateway_env_file + " && sudo systemctl reload onex-gateway-forwarder")
                else
                    null
                end
            )
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
    # verify_deployment NEW_DIGEST -- OMN-15521 remediation: the first check
    # must be that the container is actually running the digest this
    # invocation just built. Without this, a `systemctl reload` that
    # silently fails to recreate the container (e.g. a transient compose
    # error swallowed by the unit) leaves the OLD container running, which
    # still passes every other check here (non-empty labels, both files
    # present) -- the deploy reports success while nothing actually changed.
    local new_digest="$1"
    log_step "Verify"
    local running_image
    running_image="$(resolve_running_container_image)"
    if [[ "${running_image}" != "${new_digest}" ]]; then
        log_error "AC-VERIFY FAILED: ${CONTAINER_NAME} is running ${running_image:-<no container>}, expected the newly built ${new_digest}. The reload did not take effect."
        return 1
    fi
    log_info "AC-VERIFY OK: ${CONTAINER_NAME} is running the newly deployed digest ${new_digest}."

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
        log_info "Would build sidecar image (${SIDECAR_SERVICE_NAME})"
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
    previous_digest="$(resolve_running_container_image)"
    if ! retain_previous_image "${previous_digest}"; then
        previous_digest=""
    fi
    log_info "Previous running image (rollback target): ${previous_digest:-<none recorded -- first deploy or image no longer resolvable>}"

    set -a
    # shellcheck source=/dev/null
    source "${GATEWAY_ENV_FILE}"
    set +a

    stage_workspace_if_needed "${repo_root}"

    build_image "${repo_root}" "${git_sha}" "${version}"
    local new_digest
    new_digest="$(resolve_image_digest "${BUILD_IMAGE_TAG}")"
    log_info "Built image digest: ${new_digest}"

    build_sidecar_image "${repo_root}"

    sync_host_files "${repo_root}" "${GATEWAY_HOST_DIR}"
    verify_host_files_match "${repo_root}" "${GATEWAY_HOST_DIR}"

    update_gateway_env_digest "${GATEWAY_ENV_FILE}" "${new_digest}"
    write_registry "${version}" "${git_sha}" "${new_digest}" "${previous_digest}" "${repo_root}"

    if [[ "${SKIP_RELOAD}" == true ]]; then
        log_warn "--skip-reload set: gateway.env is updated but the running container still has the OLD digest until you reload."
        exit 0
    fi

    reload_service
    verify_deployment "${new_digest}"
    log_step "Done"
    log_info "omninode-gateway deployed: version=${version} git_sha=${git_sha} digest=${new_digest}"
}

main "$@"
