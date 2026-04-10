#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# fire-build-loop.sh — Trigger the ONEX build loop via `onex run`
#
# Usage:
#   fire-build-loop.sh [--max-cycles N] [--max-tickets N] [--dry-run] [--no-pull]
#
# Requires: OMNI_HOME set, uv installed, omnimarket synced to main

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
OMNI_HOME="${OMNI_HOME:?OMNI_HOME must be set}"
LOG_DIR="${OMNI_HOME}/.onex_state/logs"
LOG_FILE="${LOG_DIR}/build-loop-$(date +%Y%m%d-%H%M%S).log"

# =============================================================================
# Defaults
# =============================================================================

MAX_CYCLES=""
MAX_TICKETS=""
DRY_RUN=false
NO_PULL=false

# =============================================================================
# Usage
# =============================================================================

usage() {
    cat <<EOF
${SCRIPT_NAME} — Trigger the ONEX build loop via \`onex run\`

USAGE
    ${SCRIPT_NAME} [OPTIONS]

OPTIONS
    --max-cycles N      Maximum number of build loop cycles (default: workflow default)
    --max-tickets N     Maximum tickets to process per cycle (default: workflow default)
    --dry-run           Print the command that would run, then exit
    --no-pull           Skip pulling latest main before running
    --help              Show this help message and exit

ENVIRONMENT
    OMNI_HOME           Must be set to the omni_home directory
    KAFKA_BOOTSTRAP_SERVERS, LLM endpoints, etc. — loaded from ~/.omnibase/.env

LOGS
    ${LOG_DIR}/build-loop-YYYYMMDD-HHMMSS.log

EXAMPLES
    ${SCRIPT_NAME}
    ${SCRIPT_NAME} --max-cycles 5 --max-tickets 10
    ${SCRIPT_NAME} --dry-run
EOF
    exit 0
}

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-cycles)
            if [[ -z "${2:-}" || "${2:0:1}" == "-" ]]; then
                printf '[build-loop] ERROR: --max-cycles requires a value\n' >&2
                exit 1
            fi
            if [[ ! "$2" =~ ^[0-9]+$ ]]; then
                printf '[build-loop] ERROR: --max-cycles must be a positive integer\n' >&2
                exit 1
            fi
            MAX_CYCLES="$2"
            shift 2
            ;;
        --max-tickets)
            if [[ -z "${2:-}" || "${2:0:1}" == "-" ]]; then
                printf '[build-loop] ERROR: --max-tickets requires a value\n' >&2
                exit 1
            fi
            if [[ ! "$2" =~ ^[0-9]+$ ]]; then
                printf '[build-loop] ERROR: --max-tickets must be a positive integer\n' >&2
                exit 1
            fi
            MAX_TICKETS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-pull)
            NO_PULL=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            printf '[build-loop] ERROR: Unknown option: %s\n' "$1" >&2
            printf '[build-loop] Run %s --help for usage.\n' "${SCRIPT_NAME}" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Prerequisites
# =============================================================================

if ! command -v uv &>/dev/null; then
    printf '[build-loop] ERROR: uv is required but not found in PATH.\n' >&2
    exit 1
fi

OMNIMARKET_DIR="${OMNI_HOME}/omnimarket"
if [[ ! -d "${OMNIMARKET_DIR}" ]]; then
    printf '[build-loop] ERROR: omnimarket not found at %s\n' "${OMNIMARKET_DIR}" >&2
    exit 1
fi

ENV_FILE="${HOME}/.omnibase/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    printf '[build-loop] ERROR: ~/.omnibase/.env not found. Run bootstrap-infisical.sh first.\n' >&2
    exit 1
fi

# =============================================================================
# Pull latest main (unless skipped)
# =============================================================================

if [[ "${NO_PULL}" == false ]]; then
    printf '[build-loop] Pulling latest omnimarket main...\n'
    if ! git -C "${OMNIMARKET_DIR}" pull --ff-only 2>&1; then
        printf '[build-loop] ERROR: Failed to pull latest omnimarket. Use --no-pull to skip.\n' >&2
        exit 1
    fi
fi

# =============================================================================
# Build command
# =============================================================================

ARGS=()
[[ -n "${MAX_CYCLES}" ]] && ARGS+=("--max-cycles" "${MAX_CYCLES}")
[[ -n "${MAX_TICKETS}" ]] && ARGS+=("--max-tickets" "${MAX_TICKETS}")

CMD=(uv run onex run build_loop_workflow.yaml "${ARGS[@]}")

if [[ "${DRY_RUN}" == true ]]; then
    printf '[build-loop] DRY RUN — would execute from %s:\n' "${OMNIMARKET_DIR}"
    printf '[build-loop]   %s\n' "${CMD[*]}"
    printf '[build-loop] Log would be written to: %s\n' "${LOG_FILE}"
    exit 0
fi

# =============================================================================
# Execute
# =============================================================================

mkdir -p "${LOG_DIR}"

printf '[build-loop] Starting build loop\n'
printf '[build-loop] Log: %s\n' "${LOG_FILE}"
printf '[build-loop] Command: %s\n' "${CMD[*]}"

# Source env before running so Kafka, LLM endpoints, etc. resolve
# shellcheck source=/dev/null
source "${ENV_FILE}"

cd "${OMNIMARKET_DIR}"

if "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"; then
    printf '[build-loop] Build loop completed successfully.\n'
    exit 0
else
    printf '[build-loop] ERROR: Build loop exited with failure. See log: %s\n' "${LOG_FILE}" >&2
    exit 1
fi
