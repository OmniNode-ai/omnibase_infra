#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# trigger-deploy.sh — Trigger post-release deployment
#
# Usage:
#   trigger-deploy.sh [--target local|dev|prod] [--skip-build] [--dry-run]
#
# Targets:
#   local  — Docker compose on this machine (default)
#   dev    — SSH to .201, pull, rebuild, restart
#   prod   — Kubernetes (omninode_infra manifests)

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
OMNI_HOME="${OMNI_HOME:?OMNI_HOME must be set}"
LOG_DIR="${OMNI_HOME}/.onex_state/logs"
LOG_FILE="${LOG_DIR}/deploy-$(date +%Y%m%d-%H%M%S).log"

# .201 server (dev target)
DEV_HOST="${DEV_HOST:-192.168.1.201}"
DEV_USER="${DEV_USER:-jonah}"
DEV_OMNI_HOME="${DEV_OMNI_HOME:-/home/${DEV_USER}/Code/omni_home}"

# Health check
HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8085/health}"
HEALTH_CHECK_RETRIES=15
HEALTH_CHECK_INTERVAL=4

# =============================================================================
# Defaults
# =============================================================================

TARGET="local"
SKIP_BUILD=false
DRY_RUN=false

# =============================================================================
# Logging
# =============================================================================

log_info()  { printf '[deploy] %s\n'          "$*" | tee -a "${LOG_FILE}"; }
log_warn()  { printf '[deploy] WARNING: %s\n' "$*" | tee -a "${LOG_FILE}" >&2; }
log_error() { printf '[deploy] ERROR: %s\n'   "$*" | tee -a "${LOG_FILE}" >&2; }
log_step()  { printf '\n[deploy] === %s ===\n' "$*" | tee -a "${LOG_FILE}"; }

# =============================================================================
# Usage
# =============================================================================

usage() {
    cat <<EOF
${SCRIPT_NAME} — Trigger post-release deployment

USAGE
    ${SCRIPT_NAME} [OPTIONS]

OPTIONS
    --target local|dev|prod     Deployment target (default: local)
    --skip-build                Skip Docker image rebuild
    --dry-run                   Print what would happen, then exit
    --help                      Show this help message and exit

TARGETS
    local   Docker compose on this machine (uses omnibase_infra catalog CLI)
    dev     SSH to .201 (${DEV_HOST}), pull latest, rebuild, restart
    prod    Kubernetes via omninode_infra manifests (kubectl apply)

ENVIRONMENT
    OMNI_HOME           Required — path to omni_home directory
    DEV_HOST            .201 host (default: ${DEV_HOST})
    DEV_USER            .201 SSH user (default: ${DEV_USER})
    DEV_OMNI_HOME       omni_home path on .201 (default: ${DEV_OMNI_HOME})
    HEALTH_CHECK_URL    Health endpoint to verify after deploy (default: ${HEALTH_CHECK_URL})

LOGS
    ${LOG_DIR}/deploy-YYYYMMDD-HHMMSS.log

EXAMPLES
    ${SCRIPT_NAME}
    ${SCRIPT_NAME} --target dev
    ${SCRIPT_NAME} --target local --skip-build
    ${SCRIPT_NAME} --target prod --dry-run
EOF
    exit 0
}

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            if [[ -z "${2:-}" || "${2:0:1}" == "-" ]]; then
                printf '[deploy] ERROR: --target requires a value\n' >&2
                exit 1
            fi
            case "$2" in
                local|dev|prod) TARGET="$2" ;;
                *)
                    printf '[deploy] ERROR: --target must be local, dev, or prod (got: %s)\n' "$2" >&2
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            printf '[deploy] ERROR: Unknown option: %s\n' "$1" >&2
            printf '[deploy] Run %s --help for usage.\n' "${SCRIPT_NAME}" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

mkdir -p "${LOG_DIR}"

ENV_FILE="${HOME}/.omnibase/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    printf '[deploy] ERROR: Required env file not found: %s\n' "${ENV_FILE}" >&2
    printf '[deploy] Run scripts/bootstrap-infisical.sh to create it.\n' >&2
    exit 1
fi

log_step "Deploy — target=${TARGET} skip-build=${SKIP_BUILD} dry-run=${DRY_RUN}"
log_info "Log: ${LOG_FILE}"

# =============================================================================
# Shared: Pull latest
# =============================================================================

pull_latest() {
    log_step "Pull Latest"
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "[DRY RUN] Would run: bash ${OMNI_HOME}/omnibase_infra/scripts/pull-all.sh"
        return 0
    fi
    bash "${OMNI_HOME}/omnibase_infra/scripts/pull-all.sh"
}

# =============================================================================
# Health check
# =============================================================================

run_health_check() {
    local url="$1"
    log_step "Health Check"
    log_info "Checking ${url}..."

    local attempt=0
    local healthy=false
    while (( attempt < HEALTH_CHECK_RETRIES )); do
        attempt=$(( attempt + 1 ))
        if curl -sf --connect-timeout 2 --max-time 5 "${url}" >/dev/null 2>&1; then
            healthy=true
            break
        fi
        log_info "  Attempt ${attempt}/${HEALTH_CHECK_RETRIES} — waiting ${HEALTH_CHECK_INTERVAL}s..."
        sleep "${HEALTH_CHECK_INTERVAL}"
    done

    if [[ "${healthy}" == true ]]; then
        log_info "Health check passed."
    else
        log_error "Health check FAILED after ${HEALTH_CHECK_RETRIES} attempts at ${url}"
        exit 1
    fi
}

# =============================================================================
# Target: local
# =============================================================================

deploy_local() {
    log_step "Deploy Local (Docker Compose)"

    if ! command -v docker &>/dev/null; then
        log_error "'docker' is required but not found in PATH."
        exit 1
    fi

    local infra_dir="${OMNI_HOME}/omnibase_infra"

    pull_latest

    if [[ "${SKIP_BUILD}" == false ]]; then
        log_step "Build Images"
        if [[ "${DRY_RUN}" == true ]]; then
            log_info "[DRY RUN] Would run: cd ${infra_dir} && docker compose build --no-cache"
        else
            cd "${infra_dir}"
            # shellcheck source=/dev/null
            source "${HOME}/.omnibase/.env"
            docker compose build --no-cache 2>&1 | tee -a "${LOG_FILE}"
        fi
    else
        log_info "Skipping build (--skip-build)"
    fi

    log_step "Start Containers"
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "[DRY RUN] Would run: cd ${infra_dir} && docker compose up -d"
    else
        cd "${infra_dir}"
        # shellcheck source=/dev/null
        source "${HOME}/.omnibase/.env"
        docker compose up -d 2>&1 | tee -a "${LOG_FILE}"

        log_step "Container Status"
        sleep 5
        docker compose ps 2>&1 | tee -a "${LOG_FILE}"

        run_health_check "${HEALTH_CHECK_URL}"
    fi

    log_info "Local deploy complete."
}

# =============================================================================
# Target: dev (.201)
# =============================================================================

deploy_dev() {
    log_step "Deploy Dev (${DEV_USER}@${DEV_HOST})"

    if ! command -v ssh &>/dev/null; then
        log_error "'ssh' is required but not found in PATH."
        exit 1
    fi

    # Test SSH connectivity
    if [[ "${DRY_RUN}" == false ]]; then
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${DEV_USER}@${DEV_HOST}" true 2>/dev/null; then
            log_error "Cannot connect to ${DEV_USER}@${DEV_HOST}. Check SSH access."
            exit 1
        fi
    fi

    local remote_cmd
    if [[ "${SKIP_BUILD}" == true ]]; then
        remote_cmd="
            set -euo pipefail
            cd ${DEV_OMNI_HOME}
            bash omnibase_infra/scripts/pull-all.sh
            cd omnibase_infra
            source ~/.omnibase/.env
            docker compose up -d
            sleep 5
            docker compose ps
        "
    else
        remote_cmd="
            set -euo pipefail
            cd ${DEV_OMNI_HOME}
            bash omnibase_infra/scripts/pull-all.sh
            cd omnibase_infra
            source ~/.omnibase/.env
            docker compose build --no-cache
            docker compose up -d
            sleep 5
            docker compose ps
        "
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        log_info "[DRY RUN] Would SSH to ${DEV_USER}@${DEV_HOST} and run:"
        printf '%s\n' "${remote_cmd}" | sed 's/^/    /' | tee -a "${LOG_FILE}"
    else
        log_info "Connecting to ${DEV_USER}@${DEV_HOST}..."
        ssh "${DEV_USER}@${DEV_HOST}" "${remote_cmd}" 2>&1 | tee -a "${LOG_FILE}"

        # Health check on dev host
        local dev_health_url
        dev_health_url="http://${DEV_HOST}:8085/health"
        run_health_check "${dev_health_url}"
    fi

    log_info "Dev deploy complete."
}

# =============================================================================
# Target: prod (k8s)
# =============================================================================

deploy_prod() {
    log_step "Deploy Prod (Kubernetes)"

    if ! command -v kubectl &>/dev/null; then
        log_error "'kubectl' is required but not found in PATH."
        exit 1
    fi

    local k8s_dir="${OMNI_HOME}/omninode_infra"
    if [[ ! -d "${k8s_dir}" ]]; then
        log_error "omninode_infra not found at ${k8s_dir}"
        exit 1
    fi

    pull_latest

    if [[ "${DRY_RUN}" == true ]]; then
        log_info "[DRY RUN] Would run: kubectl apply -f ${k8s_dir}/k8s/ --dry-run=client"
    else
        log_step "Apply Manifests"
        kubectl apply -f "${k8s_dir}/k8s/" 2>&1 | tee -a "${LOG_FILE}"

        log_step "Wait for Rollout"
        for deploy in $(kubectl get deployments -n omninode -o jsonpath='{.items[*].metadata.name}'); do
            kubectl rollout status deployment/"${deploy}" -n omninode --timeout=120s 2>&1 | tee -a "${LOG_FILE}" || {
                log_warn "Rollout status check failed for ${deploy} — check manually with: kubectl get pods -n omninode"
            }
        done

        log_step "Pod Status"
        kubectl get pods -n omninode 2>&1 | tee -a "${LOG_FILE}"
    fi

    log_info "Prod deploy complete."
}

# =============================================================================
# Dispatch
# =============================================================================

case "${TARGET}" in
    local) deploy_local ;;
    dev)   deploy_dev   ;;
    prod)  deploy_prod  ;;
esac

log_step "Done"
log_info "Target:   ${TARGET}"
log_info "Log:      ${LOG_FILE}"
