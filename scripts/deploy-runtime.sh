#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
#
# deploy-runtime.sh -- Stable runtime deployment for omnibase_infra
#
# Rsyncs the current repository to a versioned deployment root
# (~/.omnibase/infra/deployed/{version}/), then runs docker compose
# from that stable location. This eliminates the directory-derived
# compose project name collision that occurs when multiple repo
# copies (omnibase_infra2, omnibase_infra4, etc.) all share the
# same compose project name.
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

# Source contract-rendered runtime policy first, then operator env overrides, so
# all ${VAR} references in docker-compose.infra.yml resolve from exported shell
# environment without making Compose the owner of activation policy.
# shellcheck source=/dev/null
SCRIPT_DIR_FOR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_FOR_ENV="$(cd "${SCRIPT_DIR_FOR_ENV}/.." && pwd)"
OPERATOR_OMNI_HOME="${OMNI_HOME:-}"
OPERATOR_HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-}"
set -a
source "${REPO_ROOT_FOR_ENV}/docker/runtime-policy.env"
source "${HOME}/.omnibase/.env"
set +a
if [[ -n "${OPERATOR_OMNI_HOME}" ]]; then
    export OMNI_HOME="${OPERATOR_OMNI_HOME}"
fi
if [[ -n "${OPERATOR_HEALTH_CHECK_URL}" ]]; then
    export HEALTH_CHECK_URL="${OPERATOR_HEALTH_CHECK_URL}"
else
    unset HEALTH_CHECK_URL
fi
unset OPERATOR_OMNI_HOME
unset OPERATOR_HEALTH_CHECK_URL

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
    projection-api
    agent-actions-consumer
    skill-lifecycle-consumer
    intelligence-api
    omninode-contract-resolver
)
# Migration services refreshed (one-shot) before the --no-deps runtime restart.
# Order matters: forward-migration applies the omnibase_infra schema, then
# intelligence-migration applies the omniintelligence schema, then migration-gate
# stamps db_metadata.migrations_complete and stays up as a healthcheck keepalive.
#
# OMN-13220: intelligence-migration was MISSING here. The compose file gates
# omninode-runtime on `intelligence-migration: condition: service_completed_successfully`,
# but restart_services() uses `up -d --no-deps`, which bypasses depends_on. On a
# fresh-DB lane that left public.db_metadata for omniintelligence unstamped, so
# the runtime crash-looped. The preflight must run it explicitly.
#
# One-shot services (run-to-completion, exit 0) are listed in
# RUNTIME_MIGRATION_ONESHOTS so the preflight can `docker wait` on them; the
# keepalive migration-gate is deliberately excluded from that wait set.
readonly RUNTIME_MIGRATION_SERVICES=(
    forward-migration
    intelligence-migration
    migration-gate
)
readonly RUNTIME_MIGRATION_ONESHOTS=(
    forward-migration
    intelligence-migration
)
# Broker readiness services brought up (and waited on) before the runtime
# restart. redpanda-partition-cap raises topic_partitions_per_shard so the cold
# 1300+-topic provisioning burst on first boot does not exhaust the default
# single-shard partition ceiling (OMN-11886 / OMN-13220). Because the runtime
# restart is `--no-deps`, the compose depends_on chain (which includes
# redpanda-partition-cap as service_completed_successfully) is bypassed, so the
# preflight must apply the cap explicitly before the kernel provisions topics.
readonly BROKER_READINESS_SERVICE="redpanda"
readonly BROKER_PARTITION_CAP_SERVICE="redpanda-partition-cap"
# Core data-plane infra that the migration preflight + runtime depend on but
# that the `--no-deps` restart path never starts itself (OMN-13594). On a fully
# COLD lane (no prior containers) nothing brings postgres/valkey up before
# run_runtime_migration_preflight runs forward-migration `--no-deps`, so the
# migration's 30x2s Postgres-readiness probe exhausts -> exit 1 -> auto-rollback.
# ensure_core_infra_ready() brings these up + waits BEFORE the preflight; on a
# WARM lane `up -d --wait` on already-healthy services is an idempotent no-op.
# redpanda is intentionally excluded here -- warm_broker_topic_provisioning owns
# broker readiness (and its collision-tolerant reachability probe).
readonly CORE_INFRA_SERVICES=(
    postgres
    valkey
)
# Cold-start consumer-group join budget (OMN-13220). On a fully-cold lane the
# kernel joins a consumer group per subscribed topic; with 1300+ topics on a
# freshly-provisioned broker the default 30s per-consumer KAFKA_TIMEOUT_SECONDS
# blew on the slow group-coordinator tail and the kernel recycled before it
# reached healthy. Raise the per-consumer start budget for the restart-driven
# boot. Operator-overridable; clamped to the config field bound (le=300).
readonly COLD_START_KAFKA_TIMEOUT_SECONDS="${COLD_START_KAFKA_TIMEOUT_SECONDS:-180}"
readonly REQUIRED_PROJECTION_TABLES=(
    delegation_events
    node_service_registry
)

# Minimum Docker Compose version (nested variable expansion support)
readonly MIN_COMPOSE_VERSION="2.20"

# Health check parameters
readonly HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://${INFRA_HOST:?INFRA_HOST required}:8085/health}"
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
# When true (--prod, or ONEX_DEPLOY_LANE=prod), the prod promotion-lineage guard
# runs before any build: the source tree must be clean AND HEAD must be an
# ancestor-of/equal-to origin/main. Prevents building the prod image from a
# dirty or dev-only tree (OMN-12626, R1).
PROD_LANE=false
if [[ "${ONEX_DEPLOY_LANE:-}" == "prod" ]]; then
    PROD_LANE=true
fi
# When --force overwrites an existing deployment, the previous directory is
# moved here as a backup. On success the backup is removed; on failure
# cleanup_on_exit() restores it.
FORCE_BACKUP_DIR=""
# OMN-13364: path (relative to the deploy target) of the vendored forward-migration
# tree. The backup-restore path in cleanup_on_exit() reverts the WHOLE deployment
# tree, including freshly-built migrations, which silently regressed the deployed
# migrations to the pre-build snapshot (dropped node_projection_delegation/
# 0015_generation_corpus_acceptance.sql in the 2026-06-19 stability redeploy).
# After a restore, the freshly-synced migration tree is re-applied from this
# snapshot so the deployed migrations always match the build source (origin/dev).
readonly MIGRATION_TREE_REL_PATH="docker/migrations/forward"
# Absolute path to a preserved copy of the freshly-synced vendored migration
# tree, captured after sync_files() (so the restore can re-apply it). Empty until
# the snapshot is taken; the snapshot dir is removed on exit.
MIGRATION_TREE_SNAPSHOT_DIR=""
# Set to true only when ALL deployment phases complete successfully.
# Used by cleanup_on_exit to determine if the --force backup can be safely removed.
DEPLOYMENT_COMPLETE=false

# =============================================================================
# Logging
# =============================================================================

log_info() {
    # Print an informational log message to stdout.
    printf '[deploy] %s\n' "$*"
}

log_warn() {
    # Print a warning message to stderr.
    printf '[deploy] WARNING: %s\n' "$*" >&2
}

log_error() {
    # Print an error message to stderr.
    printf '[deploy] ERROR: %s\n' "$*" >&2
}

log_step() {
    # Print a section header for a deployment phase.
    printf '\n[deploy] === %s ===\n' "$*"
}

log_cmd() {
    # Print a command-echo line showing the command being executed.
    printf '[deploy]   > %s\n' "$*"
}

# =============================================================================
# Usage
# =============================================================================

usage() {
    # Print usage information and exit.
    cat <<EOF
${SCRIPT_NAME} v${SCRIPT_VERSION} -- Stable runtime deployment for omnibase_infra

Rsyncs the current repo to ~/.omnibase/infra/deployed/{version}/,
then runs docker compose from that stable location.

USAGE
    ${SCRIPT_NAME} [OPTIONS]

OPTIONS
    (none)              Dry-run mode (default). Preview what would be deployed.
    --execute           Actually deploy: rsync, write registry, build images.
    --force             Required to overwrite an existing version directory.
    --restart           Restart runtime containers after build (requires --execute).
    --profile <name>    Docker compose profile (default: runtime).
    --print-compose-cmd Print exact compose commands without executing, then exit.
    --prod              Enforce the prod promotion-lineage guard before build:
                        source tree must be clean AND HEAD an ancestor-of/equal-to
                        origin/main. Also honored via ONEX_DEPLOY_LANE=prod.
    --help              Show this help message and exit.

DEPLOYMENT ROOT
    ~/.omnibase/infra/
    +-- .deploy.lock/                       mkdir-based concurrency guard
    +-- registry.json                       tracks active deployment
    +-- deployed/
        +-- {version}/                      build directory
            +-- pyproject.toml
            +-- uv.lock
            +-- src/omnibase_infra/
            +-- contracts/
            +-- docker/
                +-- docker-compose.infra.yml
                +-- Dockerfile.runtime
                +-- entrypoint-runtime.sh
                +-- .env                    preserved across deploys
                +-- .env.local              preserved (user overrides)
                +-- certs/                  preserved (TLS certs)
                +-- migrations/forward/

EXAMPLES
    # Preview what would be deployed
    ${SCRIPT_NAME}

    # Deploy and build images
    ${SCRIPT_NAME} --execute

    # Deploy, build, and restart containers
    ${SCRIPT_NAME} --execute --restart

    # Redeploy same version (overwrite)
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
    # Parse command-line arguments and set global mode/flag variables.
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
            --prod)
                PROD_LANE=true
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

resolve_compose_project() {
    # Runtime compose files use fixed container names, and the deploy-agent
    # targets the canonical "omnibase-infra" project. Keep deploy-runtime on the
    # same project by default so rebuild/recreate updates the live stack instead
    # of creating a parallel profile-derived project such as
    # "omnibase-infra-runtime".
    local compose_project="${OMNIBASE_INFRA_COMPOSE_PROJECT:-omnibase-infra}"

    if [[ ! "${compose_project}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        log_error "OMNIBASE_INFRA_COMPOSE_PROJECT must contain only alphanumeric characters, hyphens, and underscores."
        log_error "  Got: '${compose_project}'"
        exit 1
    fi

    echo "${compose_project}"
}

# Compose project -> lane (overlay) mapping. The dev lane (bare omnibase-infra
# project) runs from docker-compose.infra.yml alone; every non-dev lane LAYERS
# its overlay so the overlay's container_name + project name + lane network win.
#
# OMN-13581: deploy-runtime.sh historically passed ONLY `-f infra.yml` on every
# `docker compose` call, including warm_broker_topic_provisioning's `up redpanda`
# step. The base infra compose hardcodes `container_name: omnibase-infra-redpanda`
# (the DEV name) and the dev network, so running the warmup against a non-dev
# project (e.g. omnibase-infra-stability-test) makes compose try to (re)create
# redpanda as the DEV-named container, which collides with the live dev broker,
# gets a Docker hash prefix, and lands in 'created' -- DESTROYING the lane's own
# correctly-named broker. That left the stability lane broker-less for ~3 days.
# Layering the matching overlay gives redpanda the lane-prefixed container_name +
# lane network, so the lane's broker is targeted and never displaced.
#
# This mirrors the authoritative, tested lane->compose-file mapping in
# scripts/deploy-agent/deploy_agent/executor.py (_LANE_CONFIGS): stability-test
# layers docker-compose.stability-test.yml, prod layers docker-compose.prod.yml,
# judge layers docker-compose.judge.yml. The dev project gets no overlay.
resolve_lane_overlay_filename() {
    # Echo the overlay compose FILENAME (relative to docker/) for a compose
    # project, or nothing for the bare dev project. Fails closed: an unknown
    # non-dev project aborts rather than silently running on the dev config (the
    # exact failure mode that displaced the lane broker).
    local compose_project="$1"

    # Lane = compose project suffix after the canonical "omnibase-infra" prefix.
    # omnibase-infra                -> "" (dev, no overlay)
    # omnibase-infra-stability-test -> "stability-test"
    # omnibase-infra-prod           -> "prod"
    # omnibase-infra-judge          -> "judge"
    local lane="${compose_project#omnibase-infra}"
    lane="${lane#-}"

    case "${lane}" in
        "")
            # Dev lane: infra.yml alone (fixed dev container names are correct here).
            return 0
            ;;
        stability-test|prod|judge)
            echo "docker-compose.${lane}.yml"
            return 0
            ;;
        *)
            log_error "Unknown lane '${lane}' derived from compose project '${compose_project}'."
            log_error "  deploy-runtime.sh only knows the dev / stability-test / prod / judge lanes."
            log_error "  Refusing to deploy: running a non-dev lane on the bare infra.yml config"
            log_error "  would recreate the DEV-named redpanda and displace this lane's broker"
            log_error "  (OMN-13581). Add the lane's overlay mapping before deploying it."
            exit 1
            ;;
    esac
}

resolve_compose_file_args() {
    # Populate a caller-provided array (passed by name) with the full
    # `-f <file>` token sequence for a deployment: always
    # docker-compose.infra.yml, plus the lane overlay (docker-compose.<lane>.yml)
    # for any non-dev compose project (OMN-13581).
    #
    # Usage:
    #   local -a compose_args
    #   resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"
    #   docker compose -p "${compose_project}" "${compose_args[@]}" ...
    local -n _out_args="$1"
    local deploy_target="$2"
    local compose_project="$3"

    local docker_dir="${deploy_target}/docker"
    _out_args=("-f" "${docker_dir}/docker-compose.infra.yml")

    local overlay_filename
    overlay_filename="$(resolve_lane_overlay_filename "${compose_project}")"
    if [[ -n "${overlay_filename}" ]]; then
        _out_args+=("-f" "${docker_dir}/${overlay_filename}")
    fi
}

# =============================================================================
# Prerequisites
# =============================================================================

check_command() {
    # Validate that a required command exists in PATH.
    local cmd="$1"
    local purpose="$2"
    if ! command -v "${cmd}" &>/dev/null; then
        log_error "'${cmd}' is required (${purpose}) but not found in PATH."
        exit 1
    fi
}

check_compose_version() {
    # Verify Docker Compose meets the minimum version requirement.
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

    # Validate version components are numeric before arithmetic comparison
    local component
    for component in "${major}" "${minor}" "${req_major}" "${req_minor}"; do
        if [[ ! "${component}" =~ ^[0-9]+$ ]]; then
            log_error "Non-numeric version component: '${component}' (from version '${version_output}')."
            log_error "Expected format: MAJOR.MINOR (e.g., 2.20)."
            exit 1
        fi
    done

    if (( major < req_major || (major == req_major && minor < req_minor) )); then
        log_error "Docker Compose ${MIN_COMPOSE_VERSION}+ required (found ${version_output})."
        log_error "Nested variable expansion requires Compose >= ${MIN_COMPOSE_VERSION}."
        exit 1
    fi

    log_info "Docker Compose version: ${version_output}"
}

validate_prerequisites() {
    # Check that all required external commands and Docker Compose version are available.
    log_step "Validate Prerequisites"

    check_command docker  "container runtime"
    check_command git     "version control"

    if [[ "${PRINT_COMPOSE_CMD}" == false ]]; then
        check_command rsync   "file synchronization"
        check_command jq      "JSON processing"
        check_command curl    "deployment verification"
    fi

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
    # Verify that all required files and directories exist in the repository.
    local repo_root="$1"
    local missing=()

    [[ -f "${repo_root}/pyproject.toml" ]]                          || missing+=("pyproject.toml")
    [[ -f "${repo_root}/uv.lock" ]]                                 || missing+=("uv.lock")
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

    # VirtioFS bind-mount conflict detection
    # docker-compose has two mounts:
    #   ../contracts:/app/contracts:ro
    #   ../src/omnibase_infra/nodes:/app/contracts/nodes:ro  (overlays nodes/ subdirectory)
    # When ../contracts/nodes exists but ../src/omnibase_infra/nodes does NOT exist,
    # the overlay source is missing and containers see an empty nodes/ directory.
    local parent_dir
    parent_dir="$(dirname "${repo_root}")"
    local contracts_nodes="${parent_dir}/contracts/nodes"
    local src_nodes="${repo_root}/src/omnibase_infra/nodes"

    if [[ ! -d "${contracts_nodes}" ]]; then
        log_warn "VirtioFS CHECK: ${contracts_nodes} not found (rsync may not have run yet — advisory)"
    elif [[ ! -d "${src_nodes}" ]]; then
        log_error "VirtioFS CHECK: ${contracts_nodes} exists but ${src_nodes} does not"
        log_error "  This will cause an empty bind-mount overlay at /app/contracts/nodes"
        log_error "  Fix: ensure src/omnibase_infra/nodes/ exists in the deploy root"
        exit 1
    else
        log_info "VirtioFS CHECK: both mount sources exist"
    fi
}

# =============================================================================
# Identity -- version + git SHA
# =============================================================================

read_version() {
    # Extract the project version from pyproject.toml [project] section (PEP 621).
    local repo_root="$1"
    local version

    # Extract version from the [project] section of pyproject.toml.
    # A naive grep -m1 '^version' could match a version key in any TOML
    # section (e.g. a dependency table).  This awk approach activates only
    # inside [project] and deactivates when the next section header
    # is reached, ensuring we read the project version.
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
    # Read the 12-character abbreviated git SHA of HEAD for VCS_REF labeling.
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
    # Return a full git SHA for sibling workspace repos when available. Docker
    # build args use the value in install URLs, so "main" is only a fallback.
    local repo_path="$1"
    local sha

    sha="$(git -C "${repo_path}" rev-parse HEAD 2>/dev/null || true)"
    if [[ -n "${sha}" ]]; then
        echo "${sha}"
    else
        echo "main"
    fi
}

resolve_build_source() {
    # Resolve the selected Dockerfile dependency source.
    echo "${BUILD_SOURCE:-release}"
}

resolve_expected_build_source() {
    # Default the Dockerfile assertion to the selected source. This preserves
    # release-mode behavior while allowing BUILD_SOURCE=workspace without
    # requiring operators to set a second env var by hand.
    local build_source="$1"
    echo "${EXPECTED_BUILD_SOURCE:-${build_source}}"
}

resolve_promotion_class() {
    # OMN-13669: compute PROMOTION_CLASS OCI label from build_source.
    # workspace builds are stability-candidates (non-main-lineage dev images);
    # release/clean-main builds default to clean-main.
    local build_source="$1"
    if [[ "${build_source}" == "workspace" ]]; then
        echo "stability-candidate"
    else
        echo "clean-main"
    fi
}

resolve_non_main_lineage() {
    # OMN-13669: compute NON_MAIN_LINEAGE OCI label from build_source.
    # workspace builds are non-main-lineage; release builds are not.
    local build_source="$1"
    if [[ "${build_source}" == "workspace" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

validate_build_source_config() {
    # Validate build-source selector agreement before staging or Docker build.
    local build_source expected_build_source omni_home
    build_source="$(resolve_build_source)"
    expected_build_source="$(resolve_expected_build_source "${build_source}")"
    omni_home="${OMNI_HOME:-}"

    case "${build_source}" in
        workspace|release) ;;
        *)
            log_error "Invalid BUILD_SOURCE='${build_source}'; expected workspace or release."
            exit 64
            ;;
    esac

    case "${expected_build_source}" in
        workspace|release) ;;
        *)
            log_error "Invalid EXPECTED_BUILD_SOURCE='${expected_build_source}'; expected workspace or release."
            exit 64
            ;;
    esac

    if [[ "${build_source}" != "${expected_build_source}" ]]; then
        log_error "BUILD_SOURCE selector mismatch: BUILD_SOURCE='${build_source}' EXPECTED_BUILD_SOURCE='${expected_build_source}'."
        exit 64
    fi

    if [[ "${build_source}" == "workspace" && -z "${omni_home}" ]]; then
        log_error "BUILD_SOURCE=workspace requires OMNI_HOME before staging or build."
        exit 64
    fi
}

stage_workspace_if_needed() {
    # Populate workspace/sibling-repos/ from the operator-selected OMNI_HOME so
    # Dockerfile.runtime can install exact local sibling repo contents.
    local repo_root="$1"
    local build_source omni_home stage_script
    build_source="$(resolve_build_source)"
    if [[ "${build_source}" != "workspace" ]]; then
        return 0
    fi

    omni_home="${OMNI_HOME:-}"
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
    # Fail-fast preflight (OMN-12987): every vendored sibling's version/SHA must
    # match the consuming repo's (omnimarket) uv.lock pin. The 2026-06-11
    # stability crash shipped a 13-day-stale infra 0.37.0 because the build
    # ignored the dev lock; this guard refuses to build a stale image.
    local repo_root="$1"
    local omni_home="$2"
    local guard="${repo_root}/scripts/runtime_build/check_sibling_lock_pins.py"
    if [[ ! -f "${guard}" ]]; then
        log_error "Sibling lock-pin preflight not found: ${guard}"
        log_error "Cannot verify vendored siblings match the consuming lock. Aborting."
        exit 1
    fi

    log_step "Sibling Lock-Pin Preflight (OMN-12987)"
    # Write under sibling-repos/ so it rides along with the directory the
    # Dockerfile already COPYs into the build image (no extra COPY needed).
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

    # The check_sibling_lock_pins.py interface changed under OMN-12977/12987:
    # the original single-output flag was removed in favor of --lock (required,
    # the pin authority), repeatable --repo PACKAGE=PATH (the canonical clones
    # the build vendors), and --output (where to write the comparison JSON).
    # The consuming repo's uv.lock (omnimarket) is the pin authority.
    local lock_path="${omni_home}/omnimarket/uv.lock"
    local guard_args=(
        --lock "${lock_path}"
        --repo "omnibase-infra=${omni_home}/omnibase_infra"
        --repo "omnibase-core=${omni_home}/omnibase_core"
        --repo "omnibase-spi=${omni_home}/omnibase_spi"
        --repo "omnibase-compat=${omni_home}/omnibase_compat"
        --repo "onex-change-control=${omni_home}/onex_change_control"
        --output "${provenance_out}"
    )
    # Operator override (OMN-12977): ALLOW_SIBLING_PIN_DRIFT=1 records drift in
    # the provenance artifact and proceeds instead of aborting. Never the default.
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

check_git_dirty() {
    # Warn if the working tree has uncommitted or untracked changes.
    local repo_root="$1"
    local status_output
    status_output="$(git -C "${repo_root}" status --porcelain 2>/dev/null || true)"
    if [[ -n "${status_output}" ]]; then
        log_warn "Working tree has uncommitted changes."
        log_warn "The deployed SHA will not match the actual file contents."
        # Show untracked files separately for visibility
        local untracked
        untracked="$(echo "${status_output}" | grep '^??' || true)"
        if [[ -n "${untracked}" ]]; then
            local untracked_count
            untracked_count="$(echo "${untracked}" | wc -l | tr -d ' ')"
            log_warn "  Includes ${untracked_count} untracked file(s)."
        fi
    fi
}

guard_prod_promotion_lineage() {
    # Fail-fast when building the prod lane from a dirty or non-promoted tree.
    #
    # Delegates to scripts/check_prod_promotion_lineage.py so the clean-tree +
    # ancestor-of-origin/main lineage rules are enforced by a single, tested
    # source of truth. Only runs when --prod / ONEX_DEPLOY_LANE=prod is set;
    # non-prod lanes keep the advisory check_git_dirty warning (OMN-12626, R1).
    local repo_root="$1"
    if [[ "${PROD_LANE}" != true ]]; then
        return 0
    fi

    log_step "Prod Promotion-Lineage Guard (OMN-12626)"

    local guard="${repo_root}/scripts/check_prod_promotion_lineage.py"
    if [[ ! -f "${guard}" ]]; then
        log_error "Prod promotion-lineage guard not found: ${guard}"
        log_error "Cannot build prod from an unverifiable source tree. Aborting."
        exit 1
    fi

    # Prefer the repo venv, then uv, then system python3 — fail-fast if none run.
    local python_bin=""
    if [[ -x "${repo_root}/.venv/bin/python" ]]; then
        python_bin="${repo_root}/.venv/bin/python"
    elif command -v uv &>/dev/null; then
        python_bin="uv-run"
    elif command -v python3 &>/dev/null; then
        python_bin="python3"
    else
        log_error "No Python interpreter available to run the prod lineage guard."
        exit 1
    fi

    if [[ "${python_bin}" == "uv-run" ]]; then
        if ! uv run --project "${repo_root}" python "${guard}" --repo "${repo_root}"; then
            log_error "Prod promotion-lineage guard FAILED. Refusing to build prod."
            exit 1
        fi
    else
        if ! "${python_bin}" "${guard}" --repo "${repo_root}"; then
            log_error "Prod promotion-lineage guard FAILED. Refusing to build prod."
            exit 1
        fi
    fi

    log_info "Prod promotion-lineage guard passed: source clean + promoted."
}

guard_hotpatch_ledger() {
    # Hot-patch ledger rebuild preflight (OMN-13014, retro B-1).
    #
    # In-container hot-patches (.prepatch sibling discipline) silently revert
    # on any image rebuild / force-recreate. When a hot-patch ledger exists on
    # this host, refuse to build a lane whose recorded patches have source PRs
    # not merged into the build ref, or whose containers carry unledgered
    # .prepatch files. Delegates to scripts/preflight_hotpatch_ledger.py.
    # Sole bypass: HOTPATCH_PREFLIGHT_BYPASS with a Rule-10 user-approval
    # receipt ('# skip-token-allowed: <receipt-id>'), validated by the gate.
    local repo_root="$1"
    local git_sha="$2"
    local compose_project="$3"

    log_step "Hot-Patch Ledger Preflight (OMN-13014)"

    local ledger_path="${HOTPATCH_LEDGER_PATH:-/data/omninode/hotpatch-ledger/ledger.yaml}"
    if [[ ! -f "${ledger_path}" ]]; then
        log_warn "No hot-patch ledger at ${ledger_path} — nothing recorded on this host; gate skipped."
        log_warn "If containers here carry live hot-patches, STOP and write the ledger first."
        return 0
    fi

    local gate="${repo_root}/scripts/preflight_hotpatch_ledger.py"
    if [[ ! -f "${gate}" ]]; then
        log_error "Hot-patch ledger exists at ${ledger_path} but the gate script is missing: ${gate}"
        log_error "Refusing to rebuild over recorded hot-patches without the preflight."
        exit 1
    fi

    # Lane = compose project suffix (omnibase-infra-stability-test -> stability-test);
    # the bare dev project (omnibase-infra) maps to lane 'dev'.
    local lane="${compose_project#omnibase-infra}"
    lane="${lane#-}"
    if [[ -z "${lane}" ]]; then
        lane="dev"
    fi

    # Workspace builds vendor sibling repos from OMNI_HOME clones; the gate
    # resolves each ledger row's repo build ref (clone HEAD unless overridden
    # via --build-ref) and runs git merge-base --is-ancestor per merge commit.
    local clones_root="${OMNI_HOME:-}"
    if [[ -z "${clones_root}" ]]; then
        log_error "Hot-patch ledger present but OMNI_HOME is unset."
        log_error "Cannot resolve build-input clones for the hot-patch preflight."
        exit 1
    fi

    local python_bin=""
    if [[ -x "${repo_root}/.venv/bin/python" ]]; then
        python_bin="${repo_root}/.venv/bin/python"
    elif command -v uv &>/dev/null; then
        python_bin="uv-run"
    elif command -v python3 &>/dev/null; then
        python_bin="python3"
    else
        log_error "No Python interpreter available to run the hot-patch ledger preflight."
        exit 1
    fi

    local gate_args=(
        --lane "${lane}"
        --ledger "${ledger_path}"
        --clones-root "${clones_root}"
        --build-ref "omnibase_infra=${git_sha}"
    )
    log_cmd "${gate} ${gate_args[*]}"
    if [[ "${python_bin}" == "uv-run" ]]; then
        if ! uv run --project "${repo_root}" python "${gate}" "${gate_args[@]}"; then
            log_error "Hot-patch ledger preflight FAILED. Refusing to rebuild over live hot-patches."
            exit 1
        fi
    else
        if ! "${python_bin}" "${gate}" "${gate_args[@]}"; then
            log_error "Hot-patch ledger preflight FAILED. Refusing to rebuild over live hot-patches."
            exit 1
        fi
    fi

    log_info "Hot-patch ledger preflight passed: all recorded patches merged into the build ref."
}

# =============================================================================
# Concurrency Lock
# =============================================================================

acquire_lock() {
    # Acquire a mkdir-based concurrency lock to prevent parallel deployments.
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
# Cleanup -- partial deployment rollback, --force backup restore, + lock release
# =============================================================================

cleanup_on_exit() {
    # Remove orphaned deployment directory on failure and restore --force backups.
    # If DEPLOY_DIR_TO_CLEANUP is set and registry.json does NOT point to it,
    # the deployment was partial and should be removed. If a --force backup
    # exists (FORCE_BACKUP_DIR), restore it on failure or remove it on success.
    if [[ -n "${DEPLOY_DIR_TO_CLEANUP}" && -d "${DEPLOY_DIR_TO_CLEANUP}" ]]; then
        local active_path=""
        if [[ -f "${REGISTRY_FILE}" ]]; then
            active_path="$(jq -r '.deploy_path // empty' "${REGISTRY_FILE}" 2>/dev/null || true)"
        fi
        if [[ "${active_path}" != "${DEPLOY_DIR_TO_CLEANUP}" ]]; then
            log_warn "Cleaning up partial deployment: ${DEPLOY_DIR_TO_CLEANUP}"
            rm -rf "${DEPLOY_DIR_TO_CLEANUP}" 2>/dev/null || true
        fi
    fi

    # If a --force backup exists, decide whether to restore it or clean it up
    # based on whether the full deployment completed successfully.
    if [[ -n "${FORCE_BACKUP_DIR}" && -d "${FORCE_BACKUP_DIR}" ]]; then
        # Derive the original deployment directory from the backup path.
        # Backup convention: {deploy_target}.bak -> restore to {deploy_target}
        local original_dir="${FORCE_BACKUP_DIR%.bak}"
        if [[ "${DEPLOYMENT_COMPLETE}" != "true" ]]; then
            # Deployment did not complete -- restore previous working deployment.
            # This covers both pre-registry failures (rsync/sanity) and
            # post-registry failures (build/restart/verify).
            log_warn "Restoring previous deployment from backup: ${FORCE_BACKUP_DIR}"
            rm -rf "${original_dir}" 2>/dev/null || true
            if ! mv "${FORCE_BACKUP_DIR}" "${original_dir}" 2>/dev/null; then
                log_error "================================================================="
                log_error "CRITICAL: Failed to restore previous deployment from backup!"
                log_error "Backup location: ${FORCE_BACKUP_DIR}"
                log_error "Expected restore target: ${original_dir}"
                log_error "Manual recovery required: mv '${FORCE_BACKUP_DIR}' '${original_dir}'"
                log_error "================================================================="
            else
                # OMN-13364: the restored tree carries the PRE-BUILD vendored
                # migration tree. Re-apply the freshly-synced migration tree
                # (snapshot taken after sync_files) so the deployed migrations
                # match the build source instead of silently regressing to the
                # backup's stale snapshot (which dropped a forward migration in
                # the 2026-06-19 stability redeploy).
                restore_migration_tree_after_revert "${original_dir}"
                log_warn "NOTE: registry.json may contain stale metadata (git_sha, deployed_at)"
                log_warn "from the failed deployment. Verify or re-deploy to restore consistency."
            fi
        else
            # Full deployment succeeded -- backup is stale, clean it up.
            log_info "Cleaning up stale backup: ${FORCE_BACKUP_DIR}"
            rm -rf "${FORCE_BACKUP_DIR}" 2>/dev/null || true
        fi
        FORCE_BACKUP_DIR=""
    fi

    # OMN-13364: remove the migration-tree snapshot taken after sync_files.
    if [[ -n "${MIGRATION_TREE_SNAPSHOT_DIR}" && -d "${MIGRATION_TREE_SNAPSHOT_DIR}" ]]; then
        rm -rf "${MIGRATION_TREE_SNAPSHOT_DIR}" 2>/dev/null || true
    fi
    MIGRATION_TREE_SNAPSHOT_DIR=""

    # Release concurrency lock
    rm -rf "${LOCK_DIR}" 2>/dev/null || true
}

assert_deployed_migration_tree_synced() {
    # OMN-13415: assert the deployed (bind-mounted) forward-migration tree is
    # byte-identical to the canonical clone @ the target SHA before any migration
    # runs. The stability-promotion footgun (stale 0016, missing 0018/0019) made a
    # lane look "deployed" while applying the wrong migration SQL; this gate makes
    # that drift abort the deploy instead of silently mis-migrating.
    local deploy_target="$1"
    local repo_root="$2"
    local git_sha="$3"
    local deployed_tree="${deploy_target}/${MIGRATION_TREE_REL_PATH}"

    if [[ ! -d "${deployed_tree}" ]]; then
        # No bind-mounted forward-migration tree in this deployment layout; nothing
        # to assert (matches snapshot_migration_tree's own no-tree tolerance).
        log_warn "No deployed migration tree at ${deployed_tree}; skipping sync assertion."
        return 0
    fi

    local check_script="${repo_root}/scripts/check_deployed_migration_tree_sync.py"
    if [[ ! -f "${check_script}" ]]; then
        log_error "Migration-sync gate script missing: ${check_script}"
        exit 1
    fi

    log_info "Asserting deployed migration tree == canonical clone @ ${git_sha} (OMN-13415)..."
    if ! python3 "${check_script}" \
        --deployed-tree "${deployed_tree}" \
        --clone-root "${repo_root}" \
        --ref "${git_sha}" \
        --tree-rel-path "${MIGRATION_TREE_REL_PATH}"; then
        log_error "Deployed migration tree is OUT OF SYNC with the canonical clone @ ${git_sha}."
        log_error "Aborting deploy to avoid applying a stale migration set (OMN-13415)."
        exit 1
    fi
    log_info "Deployed migration tree is in sync with the canonical clone @ ${git_sha}."
}

snapshot_migration_tree() {
    # Preserve a copy of the freshly-synced vendored forward-migration tree so a
    # later backup-restore (cleanup_on_exit) can re-apply it instead of leaving
    # the restored tree on the backup's stale, pre-build migrations (OMN-13364).
    local deploy_target="$1"
    local src_tree="${deploy_target}/${MIGRATION_TREE_REL_PATH}"

    if [[ ! -d "${src_tree}" ]]; then
        # No vendored migration tree to protect (e.g. a deployment layout that
        # does not bind-mount forward migrations). Nothing to snapshot.
        log_warn "No vendored migration tree at ${src_tree}; skipping snapshot."
        return 0
    fi

    local snapshot_dir="${deploy_target}.migrations.snapshot"
    rm -rf "${snapshot_dir}" 2>/dev/null || true
    mkdir -p "${snapshot_dir}"
    # Mirror the tree exactly so re-apply is a faithful copy of the build source.
    rsync -a --delete "${src_tree}/" "${snapshot_dir}/"
    MIGRATION_TREE_SNAPSHOT_DIR="${snapshot_dir}"
    log_info "Snapshotted vendored migration tree for restore safety: ${snapshot_dir}"
}

restore_migration_tree_after_revert() {
    # Re-apply the freshly-synced vendored migration tree onto a restored
    # deployment tree so a backup-restore never silently regresses migrations to
    # the backup's pre-build snapshot (OMN-13364).
    local restored_dir="$1"

    if [[ -z "${MIGRATION_TREE_SNAPSHOT_DIR}" || ! -d "${MIGRATION_TREE_SNAPSHOT_DIR}" ]]; then
        # The failure happened before sync_files snapshotted the tree (e.g. an
        # rsync/sanity failure). In that case nothing newer than the backup was
        # produced, so the backup's migration tree is already the correct one.
        log_warn "No migration-tree snapshot to re-apply; restored tree keeps the backup migrations."
        return 0
    fi

    local dst_tree="${restored_dir}/${MIGRATION_TREE_REL_PATH}"
    log_warn "Re-applying freshly-built vendored migration tree onto restored deployment:"
    log_warn "  ${MIGRATION_TREE_SNAPSHOT_DIR}/ -> ${dst_tree}/"
    mkdir -p "${dst_tree}"
    if rsync -a --delete "${MIGRATION_TREE_SNAPSHOT_DIR}/" "${dst_tree}/"; then
        log_warn "Migration tree re-applied: deployed migrations match the build source, not the backup."
    else
        log_error "================================================================="
        log_error "CRITICAL: Failed to re-apply the vendored migration tree after restore!"
        log_error "The restored deployment may carry STALE migrations (silent loss risk)."
        log_error "Manual recovery: rsync -a --delete '${MIGRATION_TREE_SNAPSHOT_DIR}/' '${dst_tree}/'"
        log_error "================================================================="
    fi
}

# =============================================================================
# Prune -- remove old deployments beyond retention limit
# =============================================================================

prune_old_deployments() {
    # Remove old deployment directories that exceed the retention limit.
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

    # Collect all deployment directories sorted by modification time,
    # newest first. Each entry is a full path like
    # ~/.omnibase/infra/deployed/1.2.3/
    local all_deployments=()
    local version_dir
    for version_dir in "${deployed_root}"/*/; do
        [[ -d "${version_dir}" ]] || continue
        # Skip backup directories from failed --force deploys
        [[ "$(basename "${version_dir}")" == *.bak ]] && continue
        all_deployments+=("${version_dir%/}")
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
    done

    log_info "Pruned ${pruned} old deployment(s). Kept ${kept}."
}

# =============================================================================
# Guard -- refuse to overwrite unless --force
# =============================================================================

guard_existing_deployment() {
    # Refuse to overwrite an existing deployment directory unless --force is set.
    # When --force is active, the existing directory is moved to a .bak backup
    # so it can be restored if the new deployment fails.
    local deploy_target="$1"

    if [[ -d "${deploy_target}" ]]; then
        if [[ "${FORCE}" == true ]]; then
            log_warn "====================================================="
            log_warn "OVERWRITING existing deployment at:"
            log_warn "  ${deploy_target}"
            log_warn "====================================================="

            # Back up the existing deployment so cleanup_on_exit can restore
            # it if the new deployment fails partway through.
            local backup_dir="${deploy_target}.bak"

            # Remove any leftover backup from a previous failed --force deploy
            if [[ -d "${backup_dir}" ]]; then
                log_warn "Removing stale backup: ${backup_dir}"
                rm -rf "${backup_dir}"
            fi

            log_info "Backing up existing deployment to: ${backup_dir}"
            if ! mv "${deploy_target}" "${backup_dir}"; then
                log_error "Failed to back up existing deployment."
                log_error "Cannot proceed with --force: unable to move '${deploy_target}' to '${backup_dir}'"
                exit 1
            fi
            FORCE_BACKUP_DIR="${backup_dir}"
        else
            log_error "Deployment directory already exists:"
            log_error "  ${deploy_target}"
            log_error ""
            log_error "This version has already been deployed."
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
    # Count regular files in a directory (up to 5 levels deep).
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
    # Display a summary of what would be deployed in dry-run mode.
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
    log_info "  scripts/runtime_build/ $(count_files "${repo_root}/scripts/runtime_build") files"
    log_info "  workspace/sibling-repos/ $(count_files "${repo_root}/workspace/sibling-repos") files"

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
    # Rsync repository files to the versioned deployment target directory.
    local repo_root="$1"
    local deploy_target="$2"

    log_step "Sync Files"

    mkdir -p "${deploy_target}/docker"

    # 1. Root files (pyproject.toml, uv.lock, README.md, LICENSE)
    log_info "Syncing root files..."
    log_cmd "rsync pyproject.toml, uv.lock, README.md, LICENSE"
    rsync -a \
        "${repo_root}/pyproject.toml" \
        "${repo_root}/uv.lock" \
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

    # 3. Contracts (if directory exists)
    if [[ -d "${repo_root}/contracts/" ]]; then
        log_info "Syncing contracts/..."
        log_cmd "rsync -a --delete contracts/ -> deployed"
        rsync -a --delete \
            "${repo_root}/contracts/" "${deploy_target}/contracts/"
    else
        log_info "No contracts/ directory present, skipping contracts sync."
    fi

    # 3b. Copy omnibase_core runtime contract YAMLs into contracts/runtime/
    # OMN-6698: The bind-mount (../contracts:/app/contracts:ro) in docker-compose
    # overrides the Dockerfile's baked-in contracts. The Dockerfile copies these
    # from the installed omnibase_core package (contracts/runtime_data/), but
    # the bind-mount hides them. We must copy them into the deployed contracts/
    # directory so they survive the bind-mount override.
    check_command python3 "locating omnibase_core runtime contracts"
    local core_contracts_dir
    core_contracts_dir="$(python3 -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('omnibase_core')
if spec and spec.origin:
    pkg_dir = pathlib.Path(spec.origin).parent
    runtime_data = pkg_dir / 'contracts' / 'runtime_data'
    if not runtime_data.is_dir():
        # Fallback: check sibling contracts/runtime_data directory (editable installs)
        runtime_data = pkg_dir.parent.parent / 'contracts' / 'runtime_data'
    if runtime_data.is_dir():
        print(runtime_data)
" 2>/dev/null || true)"

    if [[ -n "${core_contracts_dir}" && -d "${core_contracts_dir}" ]]; then
        log_info "Copying omnibase_core runtime contracts from ${core_contracts_dir}..."
        mkdir -p "${deploy_target}/contracts/runtime"
        local expected_core_runtime_count=5
        local yaml_count=0
        for yaml_file in "${core_contracts_dir}"/*.yaml; do
            if [[ -f "${yaml_file}" ]]; then
                cp -f "${yaml_file}" "${deploy_target}/contracts/runtime/"
                yaml_count=$((yaml_count + 1))
            fi
        done
        if (( yaml_count < expected_core_runtime_count )); then
            log_error "Expected at least ${expected_core_runtime_count} runtime contract YAMLs in ${core_contracts_dir}, found ${yaml_count}."
            log_error "Aborting deployment to avoid runtime startup failure."
            exit 1
        fi
        log_info "Copied ${yaml_count} runtime contract YAMLs from omnibase_core."
    else
        log_error "Could not locate omnibase_core runtime contracts."
        log_error "Aborting deployment to avoid runtime startup failure."
        log_error "Ensure omnibase_core is installed: uv pip install omnibase-core"
        exit 1
    fi

    # 4. Docker files -- with preserve allowlist
    #    .env, .env.local, certs/, overrides/ survive --delete
    #    Excludes use a leading '/' to anchor them to the transfer root (docker/),
    #    so only top-level .env and .env.local are excluded; nested .env files in
    #    subdirectories are synced normally.
    log_info "Syncing docker/ (preserving .env, .env.local, certs/, overrides/)..."
    log_cmd "rsync -a --delete --exclude='/.env' --exclude='/.env.local' --exclude='/certs/' --exclude='/overrides/' docker/ -> deployed"
    # Note: .env is excluded from rsync -- env vars come from the shell environment
    # (sourced from ~/.omnibase/.env at script top). No stale .env copy needed.
    rsync -a --delete \
        --exclude='/.env' \
        --exclude='/.env.local' \
        --exclude='/certs/' \
        --exclude='/overrides/' \
        "${repo_root}/docker/" "${deploy_target}/docker/"

    # 5. Runtime build context paths required by docker/Dockerfile.runtime.
    # Release-mode builds still COPY these paths, even when sibling repos are
    # represented only by the committed .gitkeep placeholder.
    stage_workspace_if_needed "${repo_root}"

    log_info "Syncing runtime build context..."
    mkdir -p "${deploy_target}/scripts" "${deploy_target}/workspace"
    log_cmd "rsync -a --delete scripts/runtime_build/ -> deployed"
    rsync -a --delete \
        "${repo_root}/scripts/runtime_build/" "${deploy_target}/scripts/runtime_build/"
    log_cmd "rsync -a --delete workspace/sibling-repos/ -> deployed"
    rsync -a --delete \
        "${repo_root}/workspace/sibling-repos/" "${deploy_target}/workspace/sibling-repos/"
    # The lock-pin preflight result (OMN-12987) lives under sibling-repos/ as
    # .sibling-lock-pins.json, so the rsync above already carries it into the
    # build context for the in-image provenance merge.

    # Carry the root-level workspace/ file that Dockerfile.runtime COPYs
    # (workspace/sibling-pin-comparison.json, line ~278). The sibling-repos/
    # rsync above only covers the subdirectory; without this the deployed build
    # context lacks the comparison file and `docker build` fails with
    # "failed to calculate checksum of ref ...:/workspace/sibling-pin-comparison.json:
    # not found" -- the bug fixed in OMN-12987 (the dev compose build only worked
    # because it runs from the repo root where the committed placeholder exists).
    # Release mode ships the committed placeholder; workspace mode ships the real
    # expected-vs-actual comparison stage_workspace.sh wrote into the repo root
    # (OMN-12977). The regression test
    # tests/scripts/test_deploy_runtime_build_context.py asserts every
    # COPY-from-workspace path the Dockerfile references is staged here, so a
    # future Dockerfile COPY without a matching rsync fails CI.
    log_cmd "rsync -a workspace/sibling-pin-comparison.json -> deployed"
    rsync -a \
        "${repo_root}/workspace/sibling-pin-comparison.json" \
        "${deploy_target}/workspace/sibling-pin-comparison.json"

    # Carry the per-repo VCS provenance file Dockerfile.runtime COPYs
    # (workspace/sibling-vcs-provenance.json, OMN-13030). Same rationale as the
    # pin-comparison file above: the sibling-repos/ rsync only covers the
    # subdirectory, so without this the deployed build context lacks the file and
    # `docker build` fails on the COPY. Release mode ships the committed
    # placeholder; workspace mode ships the real per-repo {vcs_ref, vcs_dirty,
    # vcs_branch} stage_workspace.sh wrote into the repo root.
    log_cmd "rsync -a workspace/sibling-vcs-provenance.json -> deployed"
    rsync -a \
        "${repo_root}/workspace/sibling-vcs-provenance.json" \
        "${deploy_target}/workspace/sibling-vcs-provenance.json"

    # 6. Migration scripts (bind-mounted by docker-compose.infra.yml)
    log_info "Syncing migration scripts..."
    mkdir -p "${deploy_target}/scripts"
    rsync -a \
        --include='run-forward-migrations.sh' \
        --include='check_migrations_complete.sh' \
        --include='run-intelligence-migrations.sh' \
        --exclude='*' \
        "${repo_root}/scripts/" "${deploy_target}/scripts/"

    log_info "Sync complete."
}

# =============================================================================
# Env Setup (REMOVED -- F65 / OMN-6910)
# =============================================================================
# The old setup_env() copied ~/.omnibase/.env into a stale snapshot at
# ${deploy_target}/docker/.env. Docker compose --env-file then read from
# that snapshot instead of the live shell environment. This caused env var
# changes to be silently ignored until the next full redeploy.
#
# Fix: source ~/.omnibase/.env at script top (see line 28) and let docker
# compose resolve ${VAR} references from the shell environment directly.
# No --env-file, no stale copies.

# =============================================================================
# Compose Project Collision Detection
# =============================================================================
#
# Detects whether the target compose project name is currently owned by a
# DIFFERENT deployment directory. This guards against the Feb 15 (OMN-2233)
# class of incident where multiple repo copies share the same compose project
# name, causing containers from the wrong copy to silently continue running.
#
# How it works:
#   Docker labels every container with the working directory of the compose
#   invocation via com.docker.compose.project.working_dir. We compare that
#   label against the resolved deploy target to detect cross-copy ownership.
#
# Scenarios:
#   - No running containers for the project  → no collision, safe to proceed
#   - Running containers from THIS deploy dir → already deployed, safe to proceed
#   - Running containers from a DIFFERENT dir → COLLISION, exit 1
#
# The check runs in BOTH dry-run and execute modes so operators see the
# warning even during a preview.

check_compose_project_collision() {
    local compose_project="$1"
    local deploy_target="$2"

    log_step "Compose Project Collision Check"

    # Query running containers for this compose project name.
    # Use --all (not just running) to catch stopped-but-not-removed containers
    # that still hold the project label, which would cause collisions on `up`.
    local running_dirs
    running_dirs="$(
        docker ps --all \
            --filter "label=com.docker.compose.project=${compose_project}" \
            --format '{{index .Labels "com.docker.compose.project.working_dir"}}' \
            2>/dev/null \
        | sort -u \
        | grep -v '^$' \
        || true
    )"

    if [[ -z "${running_dirs}" ]]; then
        log_info "No running containers for project '${compose_project}'. No collision."
        return 0
    fi

    log_info "Found containers for project '${compose_project}' from: ${running_dirs}"

    # Normalize paths: resolve symlinks so that ~/.omnibase and /home/... compare equal.
    local resolved_deploy_target
    resolved_deploy_target="$(cd "${deploy_target}" 2>/dev/null && pwd -P || echo "${deploy_target}")"

    local collision_detected=false
    local colliding_dirs=()

    while IFS= read -r running_dir; do
        [[ -z "${running_dir}" ]] && continue

        local resolved_running_dir
        resolved_running_dir="$(cd "${running_dir}" 2>/dev/null && pwd -P || echo "${running_dir}")"

        if [[ "${resolved_running_dir}" != "${resolved_deploy_target}" ]]; then
            collision_detected=true
            colliding_dirs+=("${running_dir}")
        fi
    done <<< "${running_dirs}"

    if [[ "${collision_detected}" == true ]]; then
        log_error "============================================================"
        log_error "COMPOSE PROJECT COLLISION DETECTED"
        log_error "============================================================"
        log_error ""
        log_error "Compose project '${compose_project}' is already running"
        log_error "from a DIFFERENT directory:"
        for dir in "${colliding_dirs[@]}"; do
            log_error "  Running from: ${dir}"
        done
        log_error "  You are in:   ${deploy_target}"
        log_error ""
        log_error "Proceeding would deploy from this copy while the other copy's"
        log_error "containers continue to own the compose project. This causes"
        log_error "silent failures where code changes have no effect."
        log_error ""
        log_error "To resolve:"
        log_error "  1. Stop containers from the other copy first:"
        log_error "     docker compose -p ${compose_project} down"
        log_error "  2. Then re-run this script."
        log_error ""
        log_error "Or, if you are certain this is the correct copy:"
        log_error "  Manually stop all containers for project '${compose_project}'"
        log_error "  and remove the stale deployment from: ${colliding_dirs[0]}"
        log_error "============================================================"
        exit 1
    fi

    log_info "Collision check passed: containers are from the expected deployment directory."
}

# =============================================================================
# Sanity Check -- validate compose can resolve all paths
# =============================================================================

sanity_check() {
    # Validate that docker compose config resolves cleanly from the deployed directory.
    local deploy_target="$1"
    local compose_project="$2"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Post-Sync Sanity Check"

    log_info "Validating compose configuration from deployed directory..."
    log_cmd "docker compose -p ${compose_project} ${compose_args[*]} config --quiet"

    local config_output
    if ! config_output="$(docker compose \
        -p "${compose_project}" \
        "${compose_args[@]}" \
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
    # Atomically write deployment metadata to registry.json.
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
    # Build Docker images with VCS_REF, BUILD_DATE, and deployment identity args.
    # RUNTIME_SOURCE_HASH and COMPOSE_PROJECT are stamped into the image so the
    # startup banner in entrypoint-runtime.sh can display them on container start.
    # This makes deployment drift visible in logs without git forensics.
    local deploy_target="$1"
    local compose_project="$2"
    local git_sha="$3"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Build Images"

    local build_date
    build_date="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    # OMN-12965: stamp org.opencontainers.image.version from pyproject so the
    # runtime image carries a real version instead of the Dockerfile placeholder
    # (0.1.0). A placeholder version degrades every proof packet.
    local runtime_version
    runtime_version="$(read_version "${deploy_target}")"
    local omni_home="${OMNI_HOME:-}"
    local build_source
    build_source="$(resolve_build_source)"
    local expected_build_source
    expected_build_source="$(resolve_expected_build_source "${build_source}")"
    # OMN-13669: stamp OCI provenance labels so the prod-promotion gate and
    # lineage guard can refuse workspace images for prod. Computed from
    # build_source: workspace => stability-candidate/true; release => clean-main/false.
    local promotion_class
    promotion_class="$(resolve_promotion_class "${build_source}")"
    local non_main_lineage
    non_main_lineage="$(resolve_non_main_lineage "${build_source}")"
    local compat_ref="main"
    local omnimarket_ref="dev"
    local occ_ref="main"
    if [[ -n "${omni_home}" ]]; then
        compat_ref="$(read_repo_ref_or_main "${omni_home}/omnibase_compat")"
        omnimarket_ref="$(read_repo_ref_or_main "${omni_home}/omnimarket")"
        occ_ref="$(read_repo_ref_or_main "${omni_home}/onex_change_control")"
    fi

    # Build timeout in seconds (default: 15 minutes). Prevents the known issue
    # where `docker compose build` hangs indefinitely after images are built.
    # Override via DOCKER_BUILD_TIMEOUT_SECONDS env var. (OMN-5462)
    local build_timeout="${DOCKER_BUILD_TIMEOUT_SECONDS:-900}"

    local cmd=(
        docker compose
        -p "${compose_project}"
        "${compose_args[@]}"
        --profile "${COMPOSE_PROFILE}"
        build
        --progress=plain
        --build-arg "GIT_SHA=${git_sha}"
        --build-arg "VCS_REF=${git_sha}"
        --build-arg "RUNTIME_VERSION=${runtime_version}"
        --build-arg "BUILD_DATE=${build_date}"
        --build-arg "RUNTIME_SOURCE_HASH=${git_sha}"
        --build-arg "COMPOSE_PROJECT=${compose_project}"
        --build-arg "BUILD_SOURCE=${build_source}"
        --build-arg "EXPECTED_BUILD_SOURCE=${expected_build_source}"
        --build-arg "PROMOTION_CLASS=${promotion_class}"
        --build-arg "NON_MAIN_LINEAGE=${non_main_lineage}"
        --build-arg "OMNI_HOME=${omni_home}"
        --build-arg "OMNIBASE_COMPAT_REF=${compat_ref}"
        --build-arg "OMNIMARKET_REF=${omnimarket_ref}"
        --build-arg "ONEX_CHANGE_CONTROL_REF=${occ_ref}"
    )

    log_info "Building images with VCS_REF=${git_sha} RUNTIME_VERSION=${runtime_version} RUNTIME_SOURCE_HASH=${git_sha} COMPOSE_PROJECT=${compose_project}..."
    log_info "Build source: BUILD_SOURCE=${build_source} EXPECTED_BUILD_SOURCE=${expected_build_source} PROMOTION_CLASS=${promotion_class} NON_MAIN_LINEAGE=${non_main_lineage} OMNI_HOME=${omni_home}"
    log_info "Plugin refs: OMNIBASE_COMPAT_REF=${compat_ref} OMNIMARKET_REF=${omnimarket_ref} ONEX_CHANGE_CONTROL_REF=${occ_ref}"
    log_info "Build timeout: ${build_timeout}s (set DOCKER_BUILD_TIMEOUT_SECONDS to override)"
    log_cmd "${cmd[*]}"

    # Use timeout to prevent indefinite hangs after build completes (OMN-5462).
    # Exit code 124 = timeout fired; we treat this as success if images exist.
    if timeout "${build_timeout}" "${cmd[@]}"; then
        log_info "Image build complete."
    elif [[ $? -eq 124 ]]; then
        log_warn "Build timed out after ${build_timeout}s — images may still be usable. Continuing."
    else
        log_error "Image build failed."
        return 1
    fi
}

# =============================================================================
# Restart -- bring up runtime services only
# =============================================================================

resolve_broker_container() {
    # Resolve the running broker container id/name for the given compose project.
    #
    # OMN-13364: the broker's fixed container_name (e.g. omnibase-infra-redpanda)
    # is NOT a reliable handle — when it collides with another project's broker,
    # Docker prefixes it with a random hash (3ed1fdb8d50b_omnibase-infra-redpanda).
    # The compose service label (com.docker.compose.service=redpanda) survives
    # the prefix, so resolve by compose project + service label instead of by an
    # exact container-name string match.
    local compose_project="$1"
    docker ps -q \
        --filter "label=com.docker.compose.project=${compose_project}" \
        --filter "label=com.docker.compose.service=${BROKER_READINESS_SERVICE}" \
        2>/dev/null \
        | head -1
}

assert_broker_reachable() {
    # Return 0 when the broker is actually reachable on the lane network.
    #
    # Keys readiness off `rpk cluster health` executed INSIDE the broker
    # container (talking to the broker on TCP/9092 over the lane network), not
    # off an exact container-name match or the compose-wait exit status. This is
    # what lets the warmup tolerate a Docker-prefixed broker name and an
    # already-present healthy broker without false-failing (OMN-13364).
    local compose_project="$1"
    local attempts="${BROKER_REACHABLE_RETRIES:-15}"
    local interval="${BROKER_REACHABLE_INTERVAL:-4}"

    local broker_container
    broker_container="$(resolve_broker_container "${compose_project}")"
    if [[ -z "${broker_container}" ]]; then
        log_error "No running broker container found for project '${compose_project}'"
        log_error "  (label com.docker.compose.service=${BROKER_READINESS_SERVICE})."
        return 1
    fi
    log_info "Resolved broker container: ${broker_container} (probing reachability)"

    local attempt=0
    while (( attempt < attempts )); do
        attempt=$((attempt + 1))
        # rpk talks to the broker on the internal listener (redpanda:9092 / TCP).
        # `cluster health` succeeding means the broker is reachable AND serving;
        # that is the readiness signal the partition-cap rpk calls below need.
        if docker exec "${broker_container}" \
            rpk cluster health -X brokers=redpanda:9092 >/dev/null 2>&1; then
            log_info "Broker reachable: rpk cluster health OK (attempt ${attempt})."
            return 0
        fi
        log_info "  Broker not ready yet (attempt ${attempt}/${attempts}) -- waiting ${interval}s..."
        sleep "${interval}"
    done

    log_error "Broker ${broker_container} did not become reachable after ${attempts} attempts."
    return 1
}

ensure_core_infra_ready() {
    # Bring up + wait for the core data-plane infra (postgres, valkey) BEFORE the
    # migration preflight + runtime restart (OMN-13594).
    #
    # The `--restart` path runs warm_broker_topic_provisioning ->
    # run_runtime_migration_preflight -> restart_services, and every one of those
    # uses `up -d --no-deps`, which bypasses the compose `depends_on` chain. On a
    # WARM lane that is fine (postgres/valkey are already up). On a fully COLD
    # lane (no prior containers) NOTHING starts postgres/valkey first, so
    # forward-migration (`--no-deps`) has no database to connect to: its 30x2s
    # readiness probe exhausts -> exit 1 -> the deploy auto-rolls back. This is
    # the exact cold-start defect OMN-13594 filed against this script.
    #
    # Bring the core infra up explicitly here and BLOCK on its healthchecks via
    # `--wait`. On a warm lane this is an idempotent no-op (up -d on a healthy
    # service does nothing, --wait returns immediately). On a cold lane it
    # creates + warms postgres/valkey so the preflight's forward-migration sees a
    # live database on its first attempt.
    local deploy_target="$1"
    local compose_project="$2"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Core Infra Readiness (cold-start guard, OMN-13594)"

    local core_up_cmd=(
        docker compose
        -p "${compose_project}"
        "${compose_args[@]}"
        --profile "${COMPOSE_PROFILE}"
        up -d --no-deps --wait
        "${CORE_INFRA_SERVICES[@]}"
    )
    log_info "Ensuring core infra healthy before preflight: ${CORE_INFRA_SERVICES[*]}"
    log_cmd "${core_up_cmd[*]}"
    if ! "${core_up_cmd[@]}"; then
        log_error "Core infra (${CORE_INFRA_SERVICES[*]}) did not become healthy."
        log_error "Migration preflight needs a live Postgres; aborting before it"
        log_error "wastes the 30x2s readiness budget and triggers a rollback (OMN-13594)."
        return 1
    fi
    log_info "Core infra healthy: ${CORE_INFRA_SERVICES[*]}."
}

warm_broker_topic_provisioning() {
    # Bring the broker + partition cap to readiness before the --no-deps runtime
    # restart so the cold-start topic-provisioning burst does not crash-loop the
    # kernel (OMN-13220). The runtime restart bypasses depends_on, so the
    # compose-declared redpanda-partition-cap gate never fires on a restart-only
    # deploy — apply it here, explicitly, before the kernel boots.
    local deploy_target="$1"
    local compose_project="$2"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Broker Topic-Provisioning Warmup"

    # 1. Ensure the broker itself is up and healthy. `up -d` is a no-op when it
    # is already running; the --wait flag blocks until the healthcheck passes so
    # the partition-cap rpk calls below do not race a still-starting broker.
    #
    # OMN-13364: the compose `up --wait` is best-effort, not the source of truth
    # for broker readiness. When the broker container_name collides with another
    # project's broker, Docker assigns a random prefix (e.g.
    # 3ed1fdb8d50b_omnibase-infra-redpanda) and/or leaves the recreate in
    # 'Created'; `up -d --wait` then errors or never reaches healthy even though
    # a healthy broker is already reachable on the lane network. Do NOT treat
    # that as a deploy failure (it would trigger the backup-restore path, which
    # reverts the freshly-built vendored migration tree). Key broker readiness
    # off ACTUAL reachability (`rpk cluster health` on TCP/9092 inside the lane)
    # via assert_broker_reachable below, not off the compose-wait exit status.
    local broker_up_cmd=(
        docker compose
        -p "${compose_project}"
        "${compose_args[@]}"
        --profile "${COMPOSE_PROFILE}"
        up -d --no-deps --wait
        "${BROKER_READINESS_SERVICE}"
    )
    log_info "Ensuring broker is healthy: ${BROKER_READINESS_SERVICE}"
    log_cmd "${broker_up_cmd[*]}"
    if ! "${broker_up_cmd[@]}"; then
        log_warn "Broker compose up --wait did not report healthy (possible"
        log_warn "name-prefix collision or already-present broker). Falling back"
        log_warn "to a direct broker-reachability probe before deciding."
    fi

    # Source of truth: probe the broker directly. Tolerates a Docker-prefixed
    # container name and an already-present healthy broker (OMN-13364).
    if ! assert_broker_reachable "${compose_project}"; then
        log_error "Broker is not reachable on the lane network after warmup."
        log_error "Cold-start topic provisioning cannot proceed; aborting."
        return 1
    fi

    # 2. Apply the partition cap (run-to-completion). force-recreate re-runs the
    # one-shot even if a prior run left an exited container behind.
    local cap_up_cmd=(
        docker compose
        -p "${compose_project}"
        "${compose_args[@]}"
        --profile "${COMPOSE_PROFILE}"
        up -d --no-deps --force-recreate
        "${BROKER_PARTITION_CAP_SERVICE}"
    )
    log_info "Applying broker partition cap: ${BROKER_PARTITION_CAP_SERVICE}"
    log_cmd "${cap_up_cmd[*]}"
    "${cap_up_cmd[@]}"

    local cap_container="${compose_project}-${BROKER_PARTITION_CAP_SERVICE}"
    local cap_wait_cmd=(docker wait "${cap_container}")
    log_cmd "${cap_wait_cmd[*]}"
    if [[ "$("${cap_wait_cmd[@]}")" != "0" ]]; then
        log_error "${BROKER_PARTITION_CAP_SERVICE} did not complete successfully."
        log_error "Broker partition cap not applied; cold-start topic provisioning may crash-loop the runtime."
        return 1
    fi
    log_info "Broker partition cap applied."
}

run_runtime_migration_preflight() {
    # Run bounded migration services before --no-deps runtime restarts.
    local deploy_target="$1"
    local compose_project="$2"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Runtime Migration Preflight"

    for service in "${RUNTIME_MIGRATION_SERVICES[@]}"; do
        local cmd=(
            docker compose
            -p "${compose_project}"
            "${compose_args[@]}"
            --profile "${COMPOSE_PROFILE}"
            up -d --no-deps --force-recreate
            "${service}"
        )
        log_info "Refreshing migration service: ${service}"
        log_cmd "${cmd[*]}"
        "${cmd[@]}"
        # One-shot migrations (forward-migration, intelligence-migration) run to
        # completion and must exit 0 before the dependent schema/runtime work
        # proceeds. migration-gate is a long-running healthcheck keepalive, NOT a
        # one-shot, so it is deliberately excluded from the wait set
        # (OMN-13220). Deriving the container name from the compose project keeps
        # the wait pointed at the lane being deployed (OMN-12987): the base
        # compose names it <compose-project>-<service> and each lane overlay
        # follows the same form (e.g. omnibase-infra-intelligence-migration for
        # dev, omnibase-infra-stability-test-intelligence-migration for stability).
        local is_oneshot=false
        local oneshot
        for oneshot in "${RUNTIME_MIGRATION_ONESHOTS[@]}"; do
            if [[ "${service}" == "${oneshot}" ]]; then
                is_oneshot=true
                break
            fi
        done
        if [[ "${is_oneshot}" == true ]]; then
            local migration_container="${compose_project}-${service}"
            local wait_cmd=(docker wait "${migration_container}")
            log_cmd "${wait_cmd[*]}"
            if [[ "$("${wait_cmd[@]}")" != "0" ]]; then
                log_error "${service} did not complete successfully."
                return 1
            fi
        fi
    done

    # Postgres follows the same lane-derivable naming as forward-migration:
    # <compose-project>-postgres (omnibase-infra-postgres for dev,
    # omnibase-infra-stability-test-postgres for stability). Deriving it keeps
    # the projection-table probe pointed at the lane being deployed instead of
    # always hitting the dev-lane postgres (OMN-12987).
    local postgres_container="${compose_project}-postgres"
    for table_name in "${REQUIRED_PROJECTION_TABLES[@]}"; do
        local check_cmd=(
            docker exec "${postgres_container}"
            psql
            -U postgres
            -d omnidash_analytics
            -tAc
            "SELECT to_regclass('public.${table_name}') IS NOT NULL"
        )
        log_info "Checking projection table: omnidash_analytics.${table_name}"
        log_cmd "${check_cmd[*]}"
        if [[ "$("${check_cmd[@]}")" != "t" ]]; then
            log_error "Missing projection table omnidash_analytics.${table_name}; aborting runtime restart."
            return 1
        fi
    done
}

restart_services() {
    # Restart runtime containers via docker compose up --force-recreate.
    local deploy_target="$1"
    local compose_project="$2"
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"

    log_step "Restart Runtime Services"

    local cmd=(
        docker compose
        -p "${compose_project}"
        "${compose_args[@]}"
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
    # Run health checks and verify image labels match the deployed SHA.
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
        log_error "Health check FAILED after ${HEALTH_CHECK_RETRIES} attempts."
        log_error "Service is not responding at ${HEALTH_CHECK_URL}"
        log_error "Check container logs: docker logs omninode-runtime"
        exit 1
    fi

    # 2. Resolve runtime container ID. Prefer the fixed runtime container name
    # used by docker-compose.infra.yml, then fall back to a compose label lookup.
    log_info "Checking image labels for VCS_REF..."
    local container_id
    container_id="$(docker ps -q --filter "name=^/omninode-runtime$" | head -1)"
    if [[ -z "${container_id}" ]]; then
        container_id="$(
            docker ps -q \
                --filter "label=com.docker.compose.project=${compose_project}" \
                --filter "label=com.docker.compose.service=omninode-runtime" \
                | head -1
        )"
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

    # OMN-12965: verify org.opencontainers.image.version is a real version, not
    # the Dockerfile placeholder (0.1.0) or blank. A placeholder/blank identity
    # degrades every proof packet (runtime SHA + image digest are required
    # citations in accepted evidence).
    local version_label
    version_label="$(docker inspect "${container_id}" \
        --format='{{index .Config.Labels "org.opencontainers.image.version"}}' 2>/dev/null || true)"
    if [[ -z "${version_label}" || "${version_label}" == "0.1.0" ]]; then
        log_error "Image version label is blank/placeholder: org.opencontainers.image.version='${version_label}'"
        log_error "Runtime image identity is degraded (OMN-12965). Rebuild with RUNTIME_VERSION from pyproject."
        exit 1
    fi
    log_info "Image version label OK: org.opencontainers.image.version=${version_label}"

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
    # Print the exact docker compose commands this script would execute.
    local deploy_target="$1"
    local compose_project="$2"
    local git_sha="$3"
    # OMN-13581: print the SAME `-f` token sequence the script executes, including
    # the lane overlay for non-dev projects, so copy-pasted operator commands do
    # not silently run on the bare infra.yml config (which displaces the broker).
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"
    local compose_f="${compose_args[*]}"
    local omni_home="${OMNI_HOME:-}"
    local build_source
    build_source="$(resolve_build_source)"
    local expected_build_source
    expected_build_source="$(resolve_expected_build_source "${build_source}")"
    # OMN-13669: stamp OCI provenance labels so the prod-promotion gate can refuse
    # workspace images for prod. Computed from build_source (workspace =>
    # stability-candidate/true; release => clean-main/false).
    local promotion_class
    promotion_class="$(resolve_promotion_class "${build_source}")"
    local non_main_lineage
    non_main_lineage="$(resolve_non_main_lineage "${build_source}")"
    local compat_ref="main"
    local omnimarket_ref="dev"
    local occ_ref="main"
    if [[ -n "${omni_home}" ]]; then
        compat_ref="$(read_repo_ref_or_main "${omni_home}/omnibase_compat")"
        omnimarket_ref="$(read_repo_ref_or_main "${omni_home}/omnimarket")"
        occ_ref="$(read_repo_ref_or_main "${omni_home}/onex_change_control")"
    fi

    log_step "Compose Commands"

    log_info "These are the exact commands this script would run from the deployed directory."
    log_info "Note: env vars resolve from shell environment (sourced from ~/.omnibase/.env)."
    log_info ""
    log_info "Build:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    build \\"
    log_info "    --build-arg VCS_REF=${git_sha} \\"
    log_info "    --build-arg BUILD_DATE=\$(date -u +\"%Y-%m-%dT%H:%M:%SZ\") \\"
    log_info "    --build-arg RUNTIME_SOURCE_HASH=${git_sha} \\"
    log_info "    --build-arg COMPOSE_PROJECT=${compose_project} \\"
    log_info "    --build-arg BUILD_SOURCE=${build_source} \\"
    log_info "    --build-arg EXPECTED_BUILD_SOURCE=${expected_build_source} \\"
    log_info "    --build-arg PROMOTION_CLASS=${promotion_class} \\"
    log_info "    --build-arg NON_MAIN_LINEAGE=${non_main_lineage} \\"
    log_info "    --build-arg OMNI_HOME=${omni_home} \\"
    log_info "    --build-arg OMNIBASE_COMPAT_REF=${compat_ref} \\"
    log_info "    --build-arg OMNIMARKET_REF=${omnimarket_ref} \\"
    log_info "    --build-arg ONEX_CHANGE_CONTROL_REF=${occ_ref}"
    log_info ""
    log_info "Restart runtime services:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    up -d --no-deps --force-recreate \\"
    log_info "    ${RUNTIME_SERVICES[*]}"
    log_info ""
    log_info "Full stack up (infra + runtime):"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    up -d"
    log_info ""
    log_info "Stop all:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    down"
    log_info ""
    log_info "Logs:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    logs -f"
    log_info ""
    log_info "Status:"
    log_info "  docker compose \\"
    log_info "    -p ${compose_project} \\"
    log_info "    ${compose_f} \\"
    log_info "    --profile ${COMPOSE_PROFILE} \\"
    log_info "    ps"
}

# =============================================================================
# Summary
# =============================================================================

show_summary() {
    # Display post-deployment summary with next-step commands.
    local deploy_target="$1"
    local version="$2"
    local git_sha="$3"
    local compose_project="$4"
    # OMN-13581: surface the lane-overlay-aware `-f` sequence in operator
    # next-step commands too, so a copy-paste does not run the lane on infra.yml.
    local -a compose_args
    resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"
    local compose_f="${compose_args[*]}"

    log_step "Deployment Summary"

    log_info "Deploy path:       ${deploy_target}"
    log_info "Version:           ${version}"
    log_info "Git SHA:           ${git_sha}"
    log_info "Compose project:   ${compose_project}"
    log_info "Profile:           ${COMPOSE_PROFILE}"
    log_info "Registry:          ${REGISTRY_FILE}"
    log_info ""
    log_info "Next steps (source ~/.omnibase/.env before running):"

    if [[ "${RESTART}" == false ]]; then
        log_info "  To start containers, run:"
        log_info "    docker compose \\"
        log_info "      -p ${compose_project} \\"
        log_info "      ${compose_f} \\"
        log_info "      --profile ${COMPOSE_PROFILE} \\"
        log_info "      up -d"
    else
        log_info "  Containers are running. Check status:"
        log_info "    docker compose \\"
        log_info "      -p ${compose_project} \\"
        log_info "      ${compose_f} \\"
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
    # Orchestrate the full deployment workflow from validation through verification.
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
        log_error "Expected semantic version (e.g., 1.2.3). Check pyproject.toml [project] version."
        exit 1
    fi

    # Validate git SHA format for VCS_REF image labeling.
    # Accept short (7+) or full (40) hex SHAs. read_git_sha uses --short=12
    # but other inputs (e.g., CI injection) may vary.
    # Normalize to lowercase first -- some CI systems produce uppercase hex.
    git_sha=$(echo "${git_sha}" | tr '[:upper:]' '[:lower:]')
    if [[ ! "${git_sha}" =~ ^[0-9a-f]{7,40}$ ]]; then
        log_warn "Could not read valid git SHA (got: '${git_sha}')."
        log_warn "The VCS_REF Docker label may be inaccurate."
        git_sha="unknown"
    fi

    log_info "Version: ${version}"
    log_info "Git SHA: ${git_sha}"
    check_git_dirty "${repo_root}"
    validate_build_source_config

    # Prod lane: hard-fail on dirty/non-promoted source before any build/deploy.
    # Runs in both dry-run and execute modes so operators see the rejection
    # during preview, not after a build starts (OMN-12626, R1).
    guard_prod_promotion_lineage "${repo_root}"

    # Prod lane: hard-fail on dirty/non-promoted source before any build/deploy.
    # Runs in both dry-run and execute modes so operators see the rejection
    # during preview, not after a build starts (OMN-12626, R1).
    guard_prod_promotion_lineage "${repo_root}"

    # Compute paths
    local deploy_target="${DEPLOY_ROOT}/deployed/${version}"
    local compose_project
    compose_project="$(resolve_compose_project)"

    # Hot-patch ledger preflight: refuse to rebuild over live in-container
    # hot-patches whose source PRs are not merged into the build ref.
    # Runs in both dry-run and execute modes (OMN-13014, retro B-1).
    guard_hotpatch_ledger "${repo_root}" "${git_sha}" "${compose_project}"

    # --print-compose-cmd: show commands and exit
    if [[ "${PRINT_COMPOSE_CMD}" == true ]]; then
        print_compose_commands "${deploy_target}" "${compose_project}" "${git_sha}"
        exit 0
    fi

    # Phase 2.5: Compose project collision check
    # Runs in both dry-run and execute modes so operators see collisions during
    # preview. Skipped only when Docker is unavailable (non-fatal in that case).
    if command -v docker &>/dev/null; then
        check_compose_project_collision "${compose_project}" "${deploy_target}"
    else
        log_warn "Docker not available -- skipping compose project collision check."
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

    # OMN-13415: assert the freshly-synced deployed (bind-mounted) forward-migration
    # tree is byte-identical to the canonical clone @ the target SHA BEFORE the
    # forward-migration phase. The stability-promotion footgun was a stale
    # bind-mounted tree (old 0016, no 0018/0019) that made the lane look "deployed"
    # while running the wrong migration SQL — caught only by an out-of-band rsync.
    # This gate makes that drift fail the deploy instead of silently mis-migrating.
    assert_deployed_migration_tree_synced "${deploy_target}" "${repo_root}" "${git_sha}"

    # OMN-13364: snapshot the freshly-synced vendored migration tree so a later
    # backup-restore (cleanup_on_exit) re-applies it instead of reverting the
    # deployed migrations to the backup's stale, pre-build snapshot.
    snapshot_migration_tree "${deploy_target}"

    # Mark deployment directory for cleanup on failure. If registry write or
    # build fails after rsync, cleanup_on_exit() will remove this orphaned
    # directory (unless registry.json already points to it).
    DEPLOY_DIR_TO_CLEANUP="${deploy_target}"

    # Phase 7: Env setup -- REMOVED (F65 / OMN-6910)
    # Shell environment is sourced at script top; no stale .env copy needed.

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
        # Raise the per-consumer Kafka consumer-start budget for the restart-driven
        # cold boot (OMN-13220). x-runtime-env reads KAFKA_TIMEOUT_SECONDS from the
        # shell environment (default 30s when unset); exporting it here propagates
        # the raised cold-start value to every runtime container compose recreates.
        # Validate + clamp to ModelKafkaEventBusConfig.timeout_seconds bounds
        # (ge=1, le=300) so an operator override cannot produce a config the
        # kernel rejects at boot.
        local cold_start_timeout="${COLD_START_KAFKA_TIMEOUT_SECONDS}"
        if [[ ! "${cold_start_timeout}" =~ ^[0-9]+$ ]]; then
            log_error "COLD_START_KAFKA_TIMEOUT_SECONDS must be a positive integer (got: '${cold_start_timeout}')."
            return 1
        fi
        if (( cold_start_timeout < 1 )); then
            cold_start_timeout=1
        elif (( cold_start_timeout > 300 )); then
            log_warn "COLD_START_KAFKA_TIMEOUT_SECONDS=${cold_start_timeout} exceeds the config max (300); clamping to 300."
            cold_start_timeout=300
        fi
        export KAFKA_TIMEOUT_SECONDS="${cold_start_timeout}"
        log_info "Cold-start Kafka consumer-start budget: KAFKA_TIMEOUT_SECONDS=${KAFKA_TIMEOUT_SECONDS}s"
        # OMN-13594: bring up + wait for postgres/valkey BEFORE the migration
        # preflight. On a cold lane the preflight's forward-migration runs
        # `--no-deps` and would otherwise hit a non-existent Postgres, exhaust its
        # readiness budget, and trigger an auto-rollback. Idempotent no-op on a
        # warm lane.
        ensure_core_infra_ready "${deploy_target}" "${compose_project}"
        warm_broker_topic_provisioning "${deploy_target}" "${compose_project}"
        run_runtime_migration_preflight "${deploy_target}" "${compose_project}"
        restart_services "${deploy_target}" "${compose_project}"
    fi

    # Phase 12: Verify (only with --restart)
    if [[ "${RESTART}" == true ]]; then
        verify_deployment "${git_sha}" "${compose_project}"
    fi

    # All phases completed successfully. Mark deployment as complete so that
    # cleanup_on_exit knows the backup can be safely removed rather than restored.
    DEPLOYMENT_COMPLETE=true

    # Remove the --force backup (if any) since the new deployment is fully
    # built and running. cleanup_on_exit would also handle this (since
    # DEPLOYMENT_COMPLETE=true), but explicit cleanup here keeps the success
    # path self-documenting.
    if [[ -n "${FORCE_BACKUP_DIR}" && -d "${FORCE_BACKUP_DIR}" ]]; then
        log_info "Removing previous deployment backup: ${FORCE_BACKUP_DIR}"
        rm -rf "${FORCE_BACKUP_DIR}"
        FORCE_BACKUP_DIR=""
    fi

    # Phase 13: Summary
    show_summary "${deploy_target}" "${version}" "${git_sha}" "${compose_project}"

    # Phase 14: Prune old deployments (non-fatal -- must not trigger rollback)
    prune_old_deployments || log_warn "Pruning old deployments failed (non-fatal)"
}

main "$@"
