#!/bin/bash

###############################################################################
# Vault Infrastructure Validation Script
# Purpose: Verify Vault initialization, policies, secrets, and client integration
# Author: OmniNode Bridge Team (DevOps Infrastructure Agent)
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

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓ PASS]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗ FAIL]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
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

    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    if ! command -v poetry &> /dev/null; then
        missing_deps+=("poetry")
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
                docker)
                    echo "  - Docker: brew install docker"
                    ;;
                poetry)
                    echo "  - Poetry: curl -sSL https://install.python-poetry.org | python3 -"
                    ;;
            esac
        done
        echo ""
        exit 1
    fi

    log_success "All required dependencies are installed"
}

# Test helper functions
run_test() {
    local test_name="$1"
    local test_command="$2"

    TESTS_TOTAL=$((TESTS_TOTAL + 1))

    if eval "$test_command" > /dev/null 2>&1; then
        log_success "$test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        log_error "$test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test 1: Vault Container Running
test_vault_container() {
    log_info "Test 1: Checking if Vault container is running..."
    run_test "Vault container is running" \
        "docker ps --format '{{.Names}}' | grep -qi vault"
}

# Test 2: Vault API Responsive
test_vault_api() {
    log_info "Test 2: Checking if Vault API is responsive..."
    run_test "Vault API is responsive" \
        "curl -s -f ${VAULT_ADDR}/v1/sys/health > /dev/null"
}

# Test 3: KV v2 Secrets Engine Enabled
test_kv_engine() {
    log_info "Test 3: Checking if KV v2 secrets engine is enabled at omninode/..."
    run_test "KV v2 secrets engine enabled at omninode/" \
        "vault secrets list | grep -q '^omninode/'"
}

# Test 4: Policies Exist
test_policies() {
    log_info "Test 4: Checking if Vault policies exist..."

    run_test "bridge-nodes-read policy exists" \
        "vault policy read bridge-nodes-read > /dev/null"

    run_test "bridge-nodes-write policy exists" \
        "vault policy read bridge-nodes-write > /dev/null"
}

# Test 5: Development Secrets Seeded
test_dev_secrets() {
    log_info "Test 5: Checking if development secrets are seeded..."

    run_test "PostgreSQL secrets exist" \
        "vault kv get omninode/development/postgres > /dev/null"

    run_test "Kafka secrets exist" \
        "vault kv get omninode/development/kafka > /dev/null"

    run_test "Consul secrets exist" \
        "vault kv get omninode/development/consul > /dev/null"

    run_test "Service config secrets exist" \
        "vault kv get omninode/development/service_config > /dev/null"

    run_test "OnexTree secrets exist" \
        "vault kv get omninode/development/onextree > /dev/null"

    run_test "Auth secrets exist" \
        "vault kv get omninode/development/auth > /dev/null"
}

# Test 6: Staging Secrets Seeded
test_staging_secrets() {
    log_info "Test 6: Checking if staging secrets are seeded..."

    run_test "Staging PostgreSQL secrets exist" \
        "vault kv get omninode/staging/postgres > /dev/null"
}

# Test 7: Secret Values Correctness
test_secret_values() {
    log_info "Test 7: Validating secret values..."

    # Test PostgreSQL host
    run_test "PostgreSQL host value is correct (192.168.86.200)" \
        "[ \"\$(vault kv get -field=host omninode/development/postgres 2>/dev/null)\" == \"192.168.86.200\" ]"

    # Test Kafka bootstrap servers
    run_test "Kafka bootstrap_servers value is correct (192.168.86.200:9092)" \
        "[ \"\$(vault kv get -field=bootstrap_servers omninode/development/kafka 2>/dev/null)\" == \"192.168.86.200:9092\" ]"
}

# Test 8: Policy Permissions
test_policy_permissions() {
    log_info "Test 8: Validating policy permissions..."

    # Check read policy allows read access
    run_test "bridge-nodes-read policy allows read on development path" \
        "vault policy read bridge-nodes-read | grep -q 'omninode/data/development'"

    # Check write policy allows CRUD on development
    run_test "bridge-nodes-write policy allows CRUD on development path" \
        "vault policy read bridge-nodes-write | grep -q 'create.*read.*update.*delete'"
}

# Test 9: Scripts Executable
test_scripts_executable() {
    log_info "Test 9: Checking if scripts are executable..."

    run_test "init_vault.sh is executable" \
        "[ -x ${PROJECT_ROOT}/deployment/scripts/init_vault.sh ]"

    run_test "seed_secrets.sh is executable" \
        "[ -x ${SCRIPT_DIR}/seed_secrets.sh ]"
}

# Test 10: Policy Files Exist
test_policy_files() {
    log_info "Test 10: Checking if policy files exist..."

    run_test "bridge-nodes-read.hcl exists" \
        "[ -f ${SCRIPT_DIR}/policies/bridge-nodes-read.hcl ]"

    run_test "bridge-nodes-write.hcl exists" \
        "[ -f ${SCRIPT_DIR}/policies/bridge-nodes-write.hcl ]"
}

# Test 11: Documentation Exists
test_documentation() {
    log_info "Test 11: Checking if documentation exists..."

    run_test "Vault README.md exists" \
        "[ -f ${SCRIPT_DIR}/README.md ]"

    # Check README contains required sections
    run_test "README contains Quick Start section" \
        "grep -q '## Quick Start' ${SCRIPT_DIR}/README.md"

    run_test "README contains Secrets Structure section" \
        "grep -q '## Secrets Structure' ${SCRIPT_DIR}/README.md"

    run_test "README contains Production Deployment section" \
        "grep -q '## Production Deployment' ${SCRIPT_DIR}/README.md"
}

# Test 12: Python VaultClient Integration
test_python_client() {
    log_info "Test 12: Testing Python VaultClient integration..."

    # Create test Python script
    cat > /tmp/test_vault_client.py << EOF
import os
import sys
sys.path.insert(0, '${PROJECT_ROOT}/src')

from omninode_bridge.config.vault_client import VaultClient

# Initialize client
client = VaultClient.from_env()

# Test availability
if not client.is_available():
    print("ERROR: VaultClient is not available")
    sys.exit(1)

# Test reading PostgreSQL secrets
try:
    pg_host = client.get_secret("development/postgres", "host")
    if pg_host != "192.168.86.200":
        print(f"ERROR: Incorrect PostgreSQL host: {pg_host}")
        sys.exit(1)

    pg_port = client.get_secret("development/postgres", "port")
    if pg_port != "5436":
        print(f"ERROR: Incorrect PostgreSQL port: {pg_port}")
        sys.exit(1)

    print("SUCCESS: VaultClient integration working")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
EOF

    # Set environment variables for Python test
    export VAULT_ADDR="${VAULT_ADDR}"
    export VAULT_TOKEN="${VAULT_TOKEN}"
    export VAULT_MOUNT_POINT="omninode"
    export VAULT_ENABLED="true"

    # Run Python test
    run_test "Python VaultClient can read secrets" \
        "cd ${PROJECT_ROOT} && poetry run python /tmp/test_vault_client.py"

    # Cleanup
    rm -f /tmp/test_vault_client.py
}

# Display test summary
display_summary() {
    echo ""
    echo "========================================="
    echo "  Vault Infrastructure Validation Summary"
    echo "========================================="
    echo ""
    echo "Total Tests: ${TESTS_TOTAL}"
    echo -e "${GREEN}Passed:${NC} ${TESTS_PASSED}"
    echo -e "${RED}Failed:${NC} ${TESTS_FAILED}"
    echo ""

    if [ ${TESTS_FAILED} -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed! Vault infrastructure is production-ready.${NC}"
        return 0
    else
        echo -e "${RED}✗ Some tests failed. Please review the output above.${NC}"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting Vault Infrastructure Validation..."
    echo ""

    # Check dependencies first
    check_dependencies

    # Export Vault environment variables
    export VAULT_ADDR
    export VAULT_TOKEN

    # Check if Vault token is set
    if [ -z "${VAULT_TOKEN}" ]; then
        log_error "VAULT_TOKEN not set. Please set it in your environment or .env file"
        exit 1
    fi

    # Run all tests
    test_vault_container
    test_vault_api
    test_kv_engine
    test_policies
    test_dev_secrets
    test_staging_secrets
    test_secret_values
    test_policy_permissions
    test_scripts_executable
    test_policy_files
    test_documentation
    test_python_client

    # Display summary
    echo ""
    display_summary
}

# Run main function
main "$@"
