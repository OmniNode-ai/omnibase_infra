#!/bin/bash

###############################################################################
# Vault Initialization Script
# Purpose: Initialize Vault cluster, enable KV v2 secrets engine, create policies
# Author: OmniNode Bridge Team
# Version: 1.0.0
###############################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"
VAULT_TOKEN="${VAULT_TOKEN:-}"
DEV_MODE="${DEV_MODE:-true}"
VAULT_KEYS_DIR="${VAULT_KEYS_DIR:-${PROJECT_ROOT}/deployment/vault/keys}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required dependencies
check_dependencies() {
    local missing_deps=()

    # Check for required commands
    if ! command -v vault &> /dev/null; then
        missing_deps+=("vault")
    fi

    if ! command -v jq &> /dev/null; then
        missing_deps+=("jq")
    fi

    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi

    # Report missing dependencies
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        echo ""
        log_info "Installation instructions:"
        for dep in "${missing_deps[@]}"; do
            case $dep in
                vault)
                    echo "  - Vault CLI: brew install vault"
                    echo "    Or download from: https://www.vaultproject.io/downloads"
                    ;;
                jq)
                    echo "  - jq: brew install jq"
                    ;;
                curl)
                    echo "  - curl: brew install curl"
                    ;;
            esac
        done
        echo ""
        exit 1
    fi

    log_success "All required dependencies are installed"
}

# Check if Vault is running
check_vault_status() {
    log_info "Checking Vault status at ${VAULT_ADDR}..."

    if ! curl -s -f "${VAULT_ADDR}/v1/sys/health" > /dev/null 2>&1; then
        log_error "Vault is not responding at ${VAULT_ADDR}"
        log_error "Please ensure Vault is running: docker compose up -d vault"
        exit 1
    fi

    log_success "Vault is responding"
}

# Initialize Vault (production mode only)
initialize_vault() {
    if [ "${DEV_MODE}" == "true" ]; then
        log_info "Running in dev mode - Vault is automatically initialized"
        return 0
    fi

    log_info "Initializing Vault cluster..."

    # Check if already initialized
    if vault status 2>&1 | grep -q "Sealed.*false"; then
        log_warning "Vault is already initialized and unsealed"
        return 0
    fi

    # Initialize with 5 key shares and 3 key threshold
    mkdir -p "${VAULT_KEYS_DIR}"
    chmod 700 "${VAULT_KEYS_DIR}"

    vault operator init \
        -key-shares=5 \
        -key-threshold=3 \
        -format=json > "${VAULT_KEYS_DIR}/init-keys.json"

    if [ $? -eq 0 ]; then
        log_success "Vault initialized successfully"
        log_warning "CRITICAL: Unseal keys and root token saved to ${VAULT_KEYS_DIR}/init-keys.json"
        log_warning "CRITICAL: Securely backup this file and restrict access (chmod 600)"
        chmod 600 "${VAULT_KEYS_DIR}/init-keys.json"
    else
        log_error "Failed to initialize Vault"
        exit 1
    fi
}

# Unseal Vault (production mode only)
unseal_vault() {
    if [ "${DEV_MODE}" == "true" ]; then
        log_info "Running in dev mode - Vault is automatically unsealed"
        return 0
    fi

    log_info "Unsealing Vault..."

    if [ ! -f "${VAULT_KEYS_DIR}/init-keys.json" ]; then
        log_error "Unseal keys not found at ${VAULT_KEYS_DIR}/init-keys.json"
        exit 1
    fi

    # Extract unseal keys and unseal
    for i in 0 1 2; do
        UNSEAL_KEY=$(jq -r ".unseal_keys_b64[${i}]" "${VAULT_KEYS_DIR}/init-keys.json")
        vault operator unseal "${UNSEAL_KEY}" > /dev/null
    done

    log_success "Vault unsealed successfully"

    # Set root token for subsequent operations
    export VAULT_TOKEN=$(jq -r ".root_token" "${VAULT_KEYS_DIR}/init-keys.json")
}

# Enable KV v2 secrets engine
enable_kv_secrets() {
    log_info "Enabling KV v2 secrets engine at omninode/..."

    # Check if already enabled
    if vault secrets list | grep -q "^omninode/"; then
        log_warning "KV v2 secrets engine already enabled at omninode/"
        return 0
    fi

    vault secrets enable -path=omninode -version=2 kv

    if [ $? -eq 0 ]; then
        log_success "KV v2 secrets engine enabled at omninode/"
    else
        log_error "Failed to enable KV v2 secrets engine"
        exit 1
    fi
}

# Create Vault policies
create_policies() {
    log_info "Creating Vault policies..."

    local policies_dir="${PROJECT_ROOT}/deployment/vault/policies"

    if [ ! -d "${policies_dir}" ]; then
        log_error "Policies directory not found: ${policies_dir}"
        exit 1
    fi

    # Create bridge-nodes-read policy
    if [ -f "${policies_dir}/bridge-nodes-read.hcl" ]; then
        log_info "Creating bridge-nodes-read policy..."
        vault policy write bridge-nodes-read "${policies_dir}/bridge-nodes-read.hcl"
        log_success "Policy bridge-nodes-read created"
    else
        log_warning "Policy file not found: bridge-nodes-read.hcl"
    fi

    # Create bridge-nodes-write policy
    if [ -f "${policies_dir}/bridge-nodes-write.hcl" ]; then
        log_info "Creating bridge-nodes-write policy..."
        vault policy write bridge-nodes-write "${policies_dir}/bridge-nodes-write.hcl"
        log_success "Policy bridge-nodes-write created"
    else
        log_warning "Policy file not found: bridge-nodes-write.hcl"
    fi
}

# Generate token for bridge nodes
generate_bridge_token() {
    log_info "Generating token for bridge nodes..."

    local token_file="${VAULT_KEYS_DIR}/bridge-nodes-token.txt"
    mkdir -p "${VAULT_KEYS_DIR}"

    # Generate token with bridge-nodes-read and bridge-nodes-write policies
    vault token create \
        -policy=bridge-nodes-read \
        -policy=bridge-nodes-write \
        -period=768h \
        -display-name="bridge-nodes" \
        -format=json | jq -r '.auth.client_token' > "${token_file}"

    if [ $? -eq 0 ]; then
        chmod 600 "${token_file}"
        log_success "Bridge nodes token generated and saved to ${token_file}"
        log_info "Token: $(cat ${token_file})"
        log_warning "IMPORTANT: Update your .env file with VAULT_TOKEN=$(cat ${token_file})"
    else
        log_error "Failed to generate bridge nodes token"
        exit 1
    fi
}

# Display summary
display_summary() {
    echo ""
    log_success "========================================="
    log_success "  Vault Initialization Complete"
    log_success "========================================="
    echo ""
    log_info "Vault Address: ${VAULT_ADDR}"
    log_info "KV v2 Secrets Engine: omninode/"
    log_info "Policies Created: bridge-nodes-read, bridge-nodes-write"

    if [ "${DEV_MODE}" == "true" ]; then
        log_info "Mode: Development (auto-initialized/unsealed)"
        log_info "Root Token: ${VAULT_TOKEN}"
    else
        log_info "Mode: Production"
        log_warning "Unseal keys saved to: ${VAULT_KEYS_DIR}/init-keys.json"
        log_warning "Bridge token saved to: ${VAULT_KEYS_DIR}/bridge-nodes-token.txt"
        log_warning "CRITICAL: Backup these files securely!"
    fi

    echo ""
    log_info "Next Steps:"
    log_info "1. Update .env file with VAULT_TOKEN"
    log_info "2. Run seed_secrets.sh to populate development secrets"
    log_info "3. Restart bridge nodes: docker compose restart orchestrator reducer"
    echo ""
}

# Main execution
main() {
    log_info "Starting Vault initialization..."
    log_info "Dev Mode: ${DEV_MODE}"

    # Step 0: Check dependencies
    check_dependencies

    # Export Vault address
    export VAULT_ADDR

    # Step 1: Check Vault status
    check_vault_status

    # Step 2: Initialize Vault (production only)
    if [ "${DEV_MODE}" != "true" ]; then
        initialize_vault
        unseal_vault
    else
        # In dev mode, use the dev root token
        if [ -z "${VAULT_TOKEN}" ]; then
            log_error "VAULT_TOKEN not set for dev mode"
            log_info "For Vault dev server, the root token is typically printed on startup"
            log_info "Check docker logs: docker logs vault 2>&1 | grep 'Root Token'"
            log_info "Or set a custom token: export VAULT_TOKEN='your-dev-token'"
            exit 1
        fi
        export VAULT_TOKEN
    fi

    # Step 3: Enable KV v2 secrets engine
    enable_kv_secrets

    # Step 4: Create policies
    create_policies

    # Step 5: Generate bridge nodes token
    generate_bridge_token

    # Step 6: Display summary
    display_summary
}

# Run main function
main "$@"
