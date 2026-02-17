#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
#
# setup-infisical-identity.sh -- Set up machine identities for Infisical
#
# Creates two machine identities for ONEX Infrastructure:
#   1. Runtime identity (read-only) -- used by runtime services
#   2. Admin identity (read-write)  -- used by seed scripts and operators
#
# Outputs:
#   INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET, INFISICAL_PROJECT_ID
#   Written to .infisical-identity file and printed to stdout.
#
# Prerequisites:
#   - Infisical server running and accessible
#   - Admin API access (first-time setup via Infisical UI)
#   - curl and jq installed
#
# Usage:
#   ./scripts/setup-infisical-identity.sh
#   ./scripts/setup-infisical-identity.sh --admin   # Create admin identity
#   ./scripts/setup-infisical-identity.sh --runtime  # Create runtime identity (default)
#
# NOTE: This script is a template. Machine identity creation typically
# requires the Infisical Admin API or UI. The exact API endpoints and
# payloads depend on your Infisical version and configuration.
#
# See: https://infisical.com/docs/documentation/platform/identities/machine-identities

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IDENTITY_FILE="${PROJECT_ROOT}/.infisical-identity"

# Defaults
INFISICAL_ADDR="${INFISICAL_ADDR:-http://localhost:8880}"
IDENTITY_TYPE="runtime"  # runtime (read-only) or admin (read-write)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --admin) IDENTITY_TYPE="admin"; shift ;;
        --runtime) IDENTITY_TYPE="runtime"; shift ;;
        --help|-h)
            echo "Usage: $0 [--runtime|--admin]"
            echo ""
            echo "Options:"
            echo "  --runtime   Create read-only identity for runtime services (default)"
            echo "  --admin     Create read-write identity for admin/seed operations"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Check prerequisites
for cmd in curl jq; do
    if ! command -v "$cmd" &>/dev/null; then
        log_error "$cmd is required but not installed"
        exit 1
    fi
done

# Check Infisical connectivity
log_info "Checking Infisical connectivity at ${INFISICAL_ADDR}..."
if ! curl -sf "${INFISICAL_ADDR}/api/status" >/dev/null 2>&1; then
    log_error "Cannot connect to Infisical at ${INFISICAL_ADDR}"
    log_error "Make sure Infisical is running (docker compose --profile secrets up)"
    exit 1
fi
log_info "Infisical is accessible"

# Identity creation
# NOTE: The exact API for creating machine identities depends on your
# Infisical version. The steps below outline the general process.
#
# For Infisical v0.146.0+, machine identity creation is typically done via:
#   1. Create the identity in Infisical UI or API
#   2. Generate Universal Auth credentials
#   3. Assign project membership with appropriate role
#
# The API calls below are TEMPLATES -- adjust for your version.

log_info "Creating ${IDENTITY_TYPE} machine identity..."
echo ""
echo "================================================================"
echo "  MACHINE IDENTITY SETUP"
echo "================================================================"
echo ""
echo "Identity Type: ${IDENTITY_TYPE}"
echo "  Runtime: read-only access (for runtime service config prefetch)"
echo "  Admin:   read-write access (for seed scripts and operators)"
echo ""
echo "MANUAL STEPS REQUIRED:"
echo ""
echo "1. Open Infisical UI: ${INFISICAL_ADDR}"
echo ""
echo "2. Create a project (if not exists):"
echo "   - Name: 'omnibase-infra' (or your project name)"
echo "   - Note the Project ID"
echo ""
echo "3. Create a machine identity:"
echo "   - Go to: Organization Settings > Machine Identities"
echo "   - Name: 'onex-${IDENTITY_TYPE}'"
echo "   - Auth Method: Universal Auth"
echo ""
echo "4. Add the identity to your project:"
echo "   - Go to: Project Settings > Members"
echo "   - Add the machine identity"
if [[ "${IDENTITY_TYPE}" == "runtime" ]]; then
    echo "   - Role: Viewer (read-only)"
else
    echo "   - Role: Admin (read-write)"
fi
echo ""
echo "5. Generate credentials:"
echo "   - Go to the identity > Universal Auth > Create Client Secret"
echo "   - Copy the Client ID and Client Secret"
echo ""
echo "6. Set the following environment variables in your .env file:"
echo ""
echo "   INFISICAL_CLIENT_ID=<client-id-from-step-5>"
echo "   INFISICAL_CLIENT_SECRET=<client-secret-from-step-5>"
echo "   INFISICAL_PROJECT_ID=<project-id-from-step-2>"
echo "   INFISICAL_ADDR=${INFISICAL_ADDR}"
echo ""
echo "================================================================"

# Create identity file marker
if [[ ! -f "${IDENTITY_FILE}" ]]; then
    cat > "${IDENTITY_FILE}" << IDEOF
# Infisical Machine Identity Configuration
# Generated by setup-infisical-identity.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
#
# Identity Type: ${IDENTITY_TYPE}
# Infisical URL: ${INFISICAL_ADDR}
#
# IMPORTANT: Set these values in your .env file after completing
# the manual setup steps above.
#
# INFISICAL_CLIENT_ID=
# INFISICAL_CLIENT_SECRET=
# INFISICAL_PROJECT_ID=
# INFISICAL_ADDR=${INFISICAL_ADDR}
IDEOF
    log_info "Created identity marker file: ${IDENTITY_FILE}"
    log_warn "Remember to add .infisical-identity to .gitignore"
else
    log_info "Identity marker file already exists: ${IDENTITY_FILE}"
fi

echo ""
log_info "Identity setup template complete."
log_info "Follow the manual steps above to complete provisioning."
