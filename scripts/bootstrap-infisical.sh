#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
#
# bootstrap-infisical.sh -- Codified bootstrap sequence for ONEX Infrastructure
# with Infisical secrets management.
#
# Bootstrap Startup Chain (OMN-2287):
#   Step 1: PostgreSQL starts (POSTGRES_PASSWORD from .env)
#   Step 2: Valkey starts
#   Step 3: Infisical starts (depends_on: postgres + valkey healthy)
#   Step 4: Identity provisioning (first-time only)
#   Step 5: Seed runs (populates Infisical from contracts + .env values)
#   Step 6: Runtime services start (prefetch from Infisical)
#
# Usage:
#   ./scripts/bootstrap-infisical.sh                   # Full bootstrap
#   ./scripts/bootstrap-infisical.sh --skip-seed       # Skip seed step
#   ./scripts/bootstrap-infisical.sh --skip-identity   # Skip identity setup
#   ./scripts/bootstrap-infisical.sh --dry-run         # Show what would happen
#
# Prerequisites:
#   - Docker Compose v2.20+
#   - .env file with POSTGRES_PASSWORD set
#   - docker/docker-compose.infra.yml present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.infra.yml"
ENV_FILE="${PROJECT_ROOT}/.env"

# Defaults
SKIP_SEED=false
SKIP_IDENTITY=false
DRY_RUN=false
COMPOSE_CMD="docker compose"
POSTGRES_DB="${POSTGRES_DB:-omnibase_infra}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${BLUE}=== Step $1: $2 ===${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-seed)
            SKIP_SEED=true
            shift
            ;;
        --skip-identity)
            SKIP_IDENTITY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-seed       Skip the Infisical seed step"
            echo "  --skip-identity   Skip identity provisioning"
            echo "  --dry-run         Show what would happen without executing"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate prerequisites
if [[ ! -f "${COMPOSE_FILE}" ]]; then
    log_error "Docker Compose file not found: ${COMPOSE_FILE}"
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    log_error ".env file not found: ${ENV_FILE}"
    log_error "Copy .env.example to .env and configure POSTGRES_PASSWORD"
    exit 1
fi

# Source .env for variable access
set -a
# shellcheck source=/dev/null
source "${ENV_FILE}"
set +a

if [[ -z "${POSTGRES_PASSWORD:-}" ]]; then
    log_error "POSTGRES_PASSWORD is not set in .env"
    exit 1
fi

# Verify Docker Compose version
COMPOSE_VERSION=$($COMPOSE_CMD version --short 2>/dev/null || echo "0.0.0")
log_info "Docker Compose version: ${COMPOSE_VERSION}"

run_cmd() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  [DRY-RUN] $*"
    else
        "$@"
    fi
}

# ============================================================================
# Step 1: Start PostgreSQL
# ============================================================================
log_step "1" "Start PostgreSQL (POSTGRES_PASSWORD from .env)"

run_cmd $COMPOSE_CMD -f "${COMPOSE_FILE}" up -d postgres
if [[ "${DRY_RUN}" != "true" ]]; then
    log_info "Waiting for PostgreSQL to be healthy..."
    $COMPOSE_CMD -f "${COMPOSE_FILE}" exec postgres pg_isready -U "${POSTGRES_USER:-postgres}" -d "$POSTGRES_DB" --timeout=30 || {
        # Wait and retry
        sleep 5
        $COMPOSE_CMD -f "${COMPOSE_FILE}" exec postgres pg_isready -U "${POSTGRES_USER:-postgres}" -d "$POSTGRES_DB" --timeout=30
    }
    log_info "PostgreSQL is healthy"
fi

# ============================================================================
# Step 2: Start Valkey
# ============================================================================
log_step "2" "Start Valkey (Redis-compatible cache)"

run_cmd $COMPOSE_CMD -f "${COMPOSE_FILE}" up -d valkey
if [[ "${DRY_RUN}" != "true" ]]; then
    log_info "Waiting for Valkey to be healthy..."
    sleep 3
    log_info "Valkey started"
fi

# ============================================================================
# Step 3: Start Infisical (depends on postgres + valkey)
# ============================================================================
log_step "3" "Start Infisical (secrets management)"

run_cmd $COMPOSE_CMD -f "${COMPOSE_FILE}" --profile secrets up -d infisical
if [[ "${DRY_RUN}" != "true" ]]; then
    log_info "Waiting for Infisical to be healthy..."
    # Infisical has a 60s start_period, so be patient
    max_attempts=30
    attempt=0
    while [[ $attempt -lt $max_attempts ]]; do
        if $COMPOSE_CMD -f "${COMPOSE_FILE}" exec infisical wget -q --spider http://localhost:8080/api/status 2>/dev/null; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
    if [[ $attempt -eq $max_attempts ]]; then
        log_error "Infisical failed to become healthy after ${max_attempts} attempts"
        exit 1
    fi
    log_info "Infisical is healthy"
fi

# ============================================================================
# Step 4: Identity provisioning (first-time only)
# ============================================================================
if [[ "${SKIP_IDENTITY}" != "true" ]]; then
    log_step "4" "Identity provisioning (first-time only)"

    IDENTITY_FILE="${PROJECT_ROOT}/.infisical-identity"
    if [[ -f "${IDENTITY_FILE}" ]]; then
        log_info "Identity file exists (${IDENTITY_FILE}), skipping provisioning"
    else
        log_info "Running identity setup..."
        IDENTITY_SCRIPT="${SCRIPT_DIR}/setup-infisical-identity.sh"
        if [[ -x "${IDENTITY_SCRIPT}" ]]; then
            run_cmd "${IDENTITY_SCRIPT}"
        else
            log_warn "Identity script not found or not executable: ${IDENTITY_SCRIPT}"
            log_warn "Skipping identity provisioning"
        fi
    fi
else
    log_info "Skipping identity provisioning (--skip-identity)"
fi

# ============================================================================
# Step 5: Seed Infisical from contracts + .env
# ============================================================================
if [[ "${SKIP_SEED}" != "true" ]]; then
    log_step "5" "Seed Infisical from contracts + .env values"

    SEED_SCRIPT="${SCRIPT_DIR}/seed-infisical.py"
    if [[ -f "${SEED_SCRIPT}" ]]; then
        log_info "Running seed script (dry-run first)..."
        run_cmd uv run python "${SEED_SCRIPT}" \
            --contracts-dir "${PROJECT_ROOT}/src/omnibase_infra/nodes" \
            --dry-run

        if [[ "${DRY_RUN}" != "true" ]]; then
            log_info "Executing seed (create missing keys)..."
            uv run python "${SEED_SCRIPT}" \
                --contracts-dir "${PROJECT_ROOT}/src/omnibase_infra/nodes" \
                --create-missing-keys \
                --execute
        fi
    else
        log_warn "Seed script not found: ${SEED_SCRIPT}"
        log_warn "Skipping seed step"
    fi
else
    log_info "Skipping seed (--skip-seed)"
fi

# ============================================================================
# Step 6: Start runtime services (prefetch from Infisical)
# ============================================================================
log_step "6" "Start runtime services (with config prefetch from Infisical)"

run_cmd $COMPOSE_CMD -f "${COMPOSE_FILE}" --profile runtime up -d
if [[ "${DRY_RUN}" != "true" ]]; then
    log_info "Runtime services starting..."
    sleep 5
    $COMPOSE_CMD -f "${COMPOSE_FILE}" ps
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
log_info "Bootstrap complete!"
echo ""
echo "Services:"
echo "  PostgreSQL:  localhost:${POSTGRES_EXTERNAL_PORT:-5436}"
echo "  Valkey:      localhost:${VALKEY_EXTERNAL_PORT:-16379}"
echo "  Infisical:   localhost:${INFISICAL_EXTERNAL_PORT:-8880}"
echo "  Runtime:     localhost:${RUNTIME_MAIN_PORT:-8085}"
echo ""
echo "Infisical UI:  http://localhost:${INFISICAL_EXTERNAL_PORT:-8880}"
