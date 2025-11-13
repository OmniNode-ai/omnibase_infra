#!/bin/bash

###############################################################################
# Vault Secrets Seeding Script
# Purpose: Seed development and staging secrets into Vault
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
ENV_FILE="${PROJECT_ROOT}/deployment/.env"
VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"
VAULT_TOKEN="${VAULT_TOKEN:-}"

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

    if ! command -v openssl &> /dev/null; then
        missing_deps+=("openssl")
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
                openssl)
                    echo "  - openssl: brew install openssl"
                    ;;
            esac
        done
        echo ""
        exit 1
    fi

    log_success "All required dependencies are installed"
}

# Load environment variables
load_env_file() {
    if [ ! -f "${ENV_FILE}" ]; then
        log_error "Environment file not found: ${ENV_FILE}"
        exit 1
    fi

    log_info "Loading environment variables from ${ENV_FILE}"

    # Source the .env file (careful with special characters)
    set -a
    source "${ENV_FILE}"
    set +a

    log_success "Environment variables loaded"
}

# Check if Vault is accessible
check_vault_status() {
    log_info "Checking Vault status at ${VAULT_ADDR}..."

    if ! curl -s -f "${VAULT_ADDR}/v1/sys/health" > /dev/null 2>&1; then
        log_error "Vault is not responding at ${VAULT_ADDR}"
        log_error "Please ensure Vault is running: docker compose up -d vault"
        exit 1
    fi

    log_success "Vault is responding"
}

# Check if Vault token is valid
check_vault_token() {
    if [ -z "${VAULT_TOKEN}" ]; then
        log_error "VAULT_TOKEN not set"
        log_error "Please run init_vault.sh first or set VAULT_TOKEN in your environment"
        exit 1
    fi

    log_info "Validating Vault token..."

    if ! vault token lookup > /dev/null 2>&1; then
        log_error "Vault token is invalid or expired"
        exit 1
    fi

    log_success "Vault token is valid"
}

# Seed PostgreSQL secrets
seed_postgres_secrets() {
    local env=$1

    log_info "Seeding PostgreSQL secrets for ${env} environment..."

    vault kv put "omninode/${env}/postgres" \
        host="${POSTGRES_HOST:-192.168.86.200}" \
        port="${POSTGRES_PORT:-5436}" \
        database="${POSTGRES_DATABASE:-omninode_bridge}" \
        username="${POSTGRES_USER:-postgres}" \
        password="${POSTGRES_PASSWORD:-changeme}" \
        max_connections="${POSTGRES_MAX_CONNECTIONS:-50}" \
        min_connections="${POSTGRES_MIN_CONNECTIONS:-10}"

    if [ $? -eq 0 ]; then
        log_success "PostgreSQL secrets seeded for ${env}"
    else
        log_error "Failed to seed PostgreSQL secrets for ${env}"
        exit 1
    fi
}

# Seed Kafka secrets
seed_kafka_secrets() {
    local env=$1

    log_info "Seeding Kafka secrets for ${env} environment..."

    vault kv put "omninode/${env}/kafka" \
        bootstrap_servers="${KAFKA_BOOTSTRAP_SERVERS:-192.168.86.200:9092}" \
        enable_idempotence="${KAFKA_ENABLE_IDEMPOTENCE:-true}" \
        acks="${KAFKA_ACKS:-all}" \
        retries="${KAFKA_RETRIES:-3}" \
        compression_type="${KAFKA_COMPRESSION_TYPE:-snappy}"

    if [ $? -eq 0 ]; then
        log_success "Kafka secrets seeded for ${env}"
    else
        log_error "Failed to seed Kafka secrets for ${env}"
        exit 1
    fi
}

# Seed Consul secrets
seed_consul_secrets() {
    local env=$1

    log_info "Seeding Consul secrets for ${env} environment..."

    vault kv put "omninode/${env}/consul" \
        host="${CONSUL_HOST:-192.168.86.200}" \
        port="${CONSUL_PORT:-28500}" \
        datacenter="${CONSUL_DATACENTER:-omninode-bridge}" \
        token="${CONSUL_TOKEN:-}"

    if [ $? -eq 0 ]; then
        log_success "Consul secrets seeded for ${env}"
    else
        log_error "Failed to seed Consul secrets for ${env}"
        exit 1
    fi
}

# Seed service configuration secrets
seed_service_config() {
    local env=$1

    log_info "Seeding service configuration for ${env} environment..."

    vault kv put "omninode/${env}/service_config" \
        log_level="${LOG_LEVEL:-info}" \
        environment="${env}" \
        service_version="1.0.0" \
        enable_metrics="true" \
        enable_tracing="true"

    if [ $? -eq 0 ]; then
        log_success "Service configuration seeded for ${env}"
    else
        log_error "Failed to seed service configuration for ${env}"
        exit 1
    fi
}

# Seed OnexTree intelligence service secrets
seed_onextree_secrets() {
    local env=$1

    log_info "Seeding OnexTree intelligence service secrets for ${env} environment..."

    vault kv put "omninode/${env}/onextree" \
        host="${ONEXTREE_HOST:-192.168.86.200}" \
        port="${ONEXTREE_PORT:-8058}" \
        api_url="${ONEXTREE_API_URL:-http://192.168.86.200:8058}" \
        timeout_seconds="30" \
        max_retries="3"

    if [ $? -eq 0 ]; then
        log_success "OnexTree secrets seeded for ${env}"
    else
        log_error "Failed to seed OnexTree secrets for ${env}"
        exit 1
    fi
}

# Seed authentication secrets
seed_auth_secrets() {
    local env=$1

    log_info "Seeding authentication secrets for ${env} environment..."

    # Generate a secure random secret key if not set
    local auth_secret_key="${AUTH_SECRET_KEY:-$(openssl rand -hex 32)}"

    vault kv put "omninode/${env}/auth" \
        secret_key="${auth_secret_key}" \
        algorithm="HS256" \
        access_token_expire_minutes="30" \
        refresh_token_expire_days="7"

    if [ $? -eq 0 ]; then
        log_success "Authentication secrets seeded for ${env}"
        if [ "${AUTH_SECRET_KEY}" == "default-secret-key-change-me" ] || [ -z "${AUTH_SECRET_KEY}" ]; then
            log_warning "Generated new random secret key for ${env}: ${auth_secret_key}"
            log_warning "Update your .env file with AUTH_SECRET_KEY=${auth_secret_key}"
        fi
    else
        log_error "Failed to seed authentication secrets for ${env}"
        exit 1
    fi
}

# Seed deployment receiver secrets
seed_deployment_secrets() {
    local env=$1

    log_info "Seeding deployment receiver secrets for ${env} environment..."

    vault kv put "omninode/${env}/deployment" \
        receiver_port="${DEPLOYMENT_RECEIVER_PORT:-8001}" \
        allowed_ip_ranges="${ALLOWED_IP_RANGES:-192.168.86.0/24,10.0.0.0/8}" \
        auth_secret_key="${AUTH_SECRET_KEY:-default-secret-key-change-me}" \
        docker_host="unix:///var/run/docker.sock"

    if [ $? -eq 0 ]; then
        log_success "Deployment receiver secrets seeded for ${env}"
    else
        log_error "Failed to seed deployment receiver secrets for ${env}"
        exit 1
    fi
}

# Seed all secrets for an environment
seed_environment() {
    local env=$1

    log_info "Seeding all secrets for ${env} environment..."
    echo ""

    seed_postgres_secrets "${env}"
    seed_kafka_secrets "${env}"
    seed_consul_secrets "${env}"
    seed_service_config "${env}"
    seed_onextree_secrets "${env}"
    seed_auth_secrets "${env}"
    seed_deployment_secrets "${env}"

    echo ""
    log_success "All secrets seeded for ${env} environment"
}

# Display seeded secrets (masked sensitive values)
display_secrets() {
    local env=$1

    log_info "Displaying seeded secrets for ${env} environment:"
    echo ""

    echo "PostgreSQL:"
    vault kv get -format=json "omninode/${env}/postgres" | jq -r '.data.data | to_entries[] | select(.key != "password") | "  \(.key): \(.value)"'
    echo "  password: ********"
    echo ""

    echo "Kafka:"
    vault kv get -format=json "omninode/${env}/kafka" | jq -r '.data.data | to_entries[] | "  \(.key): \(.value)"'
    echo ""

    echo "Consul:"
    vault kv get -format=json "omninode/${env}/consul" | jq -r '.data.data | to_entries[] | "  \(.key): \(.value)"'
    echo ""

    echo "Service Config:"
    vault kv get -format=json "omninode/${env}/service_config" | jq -r '.data.data | to_entries[] | "  \(.key): \(.value)"'
    echo ""

    echo "OnexTree:"
    vault kv get -format=json "omninode/${env}/onextree" | jq -r '.data.data | to_entries[] | "  \(.key): \(.value)"'
    echo ""

    echo "Auth:"
    vault kv get -format=json "omninode/${env}/auth" | jq -r '.data.data | to_entries[] | select(.key != "secret_key") | "  \(.key): \(.value)"'
    echo "  secret_key: ********"
    echo ""

    echo "Deployment:"
    vault kv get -format=json "omninode/${env}/deployment" | jq -r '.data.data | to_entries[] | select(.key != "auth_secret_key") | "  \(.key): \(.value)"'
    echo "  auth_secret_key: ********"
    echo ""
}

# Display summary
display_summary() {
    echo ""
    log_success "========================================="
    log_success "  Vault Secrets Seeding Complete"
    log_success "========================================="
    echo ""
    log_info "Environments seeded: development, staging"
    log_info "Secret paths:"
    log_info "  - omninode/development/*"
    log_info "  - omninode/staging/*"
    echo ""
    log_info "To view secrets:"
    log_info "  vault kv get omninode/development/postgres"
    log_info "  vault kv get omninode/staging/kafka"
    echo ""
    log_info "To update a secret:"
    log_info "  vault kv put omninode/development/postgres password=new_password"
    echo ""
    log_warning "Production secrets should be managed manually or via CI/CD"
    echo ""
}

# Main execution
main() {
    log_info "Starting Vault secrets seeding..."

    # Step 0: Check dependencies
    check_dependencies

    # Export Vault address and token
    export VAULT_ADDR
    export VAULT_TOKEN

    # Step 1: Load environment file
    load_env_file

    # Step 2: Check Vault status
    check_vault_status

    # Step 3: Check Vault token
    check_vault_token

    # Step 4: Seed development environment
    seed_environment "development"

    # Step 5: Seed staging environment
    seed_environment "staging"

    # Step 6: Display seeded secrets (development only)
    echo ""
    display_secrets "development"

    # Step 7: Display summary
    display_summary
}

# Run main function
main "$@"
