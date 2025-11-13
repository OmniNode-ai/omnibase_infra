#!/bin/bash
#
# Remote Integration Test Runner for OmniNode Bridge
#
# This script runs integration tests against a remote distributed system.
# It configures the test environment to use remote services (Kafka, PostgreSQL, Consul)
# instead of local Docker containers.
#
# Usage:
#   ./run-integration-tests-remote.sh [options]
#
# Options:
#   -h, --help           Show this help message
#   -v, --verbose        Enable verbose output
#   -k PATTERN           Run only tests matching PATTERN
#   --coverage           Generate coverage report
#   --no-verify          Skip service verification
#   --remote-host HOST   Remote system host (default: 192.168.86.200)
#
# Environment:
#   Remote system must have all services running and accessible:
#   - Kafka/RedPanda on port 29102
#   - PostgreSQL on port 5436
#   - Consul on port 28500
#   - Application services (orchestrator, reducer, etc.)
#
# Examples:
#   # Run all integration tests against remote system
#   ./run-integration-tests-remote.sh
#
#   # Run specific test pattern with verbose output
#   ./run-integration-tests-remote.sh -v -k test_kafka
#
#   # Run with coverage report
#   ./run-integration-tests-remote.sh --coverage
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default remote host
REMOTE_HOST="${REMOTE_HOST:-192.168.86.200}"

# Test options
VERBOSE=false
TEST_PATTERN=""
GENERATE_COVERAGE=false
VERIFY_SERVICES=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}$1${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

show_help() {
    head -n 30 "$0" | grep '^#' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -k)
            TEST_PATTERN="$2"
            shift 2
            ;;
        --coverage)
            GENERATE_COVERAGE=true
            shift
            ;;
        --no-verify)
            VERIFY_SERVICES=false
            shift
            ;;
        --remote-host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# ============================================================================
# Environment Setup
# ============================================================================

print_header "Remote Integration Test Setup"

print_info "Remote System: $REMOTE_HOST"
print_info "Test Mode: remote"
print_info "Working Directory: $SCRIPT_DIR"

# Export environment variables for remote testing
export TEST_MODE=remote
export KAFKA_BOOTSTRAP_SERVERS="$REMOTE_HOST:29102"
export POSTGRES_HOST="$REMOTE_HOST"
export POSTGRES_PORT=5436
export POSTGRES_DATABASE=omninode_bridge
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=omninode_remote_2024_secure
export CONSUL_HOST="$REMOTE_HOST"
export CONSUL_PORT=28500

# Service URLs
export METADATA_STAMPING_URL="http://$REMOTE_HOST:8057"
export ONEXTREE_URL="http://$REMOTE_HOST:8058"
export HOOK_RECEIVER_URL="http://$REMOTE_HOST:8001"
export ORCHESTRATOR_URL="http://$REMOTE_HOST:8060"
export REDUCER_URL="http://$REMOTE_HOST:8061"

print_success "Environment configured for remote testing"

# ============================================================================
# Service Verification
# ============================================================================

if [ "$VERIFY_SERVICES" = true ]; then
    print_header "Verifying Remote Services"

    # Check Kafka
    print_info "Checking Kafka/RedPanda on $REMOTE_HOST:29102..."
    if timeout 5 bash -c "</dev/tcp/$REMOTE_HOST/29102" 2>/dev/null; then
        print_success "Kafka/RedPanda accessible"
    else
        print_error "Kafka/RedPanda not accessible on $REMOTE_HOST:29102"
        print_warning "Tests may fail if Kafka is required"
    fi

    # Check PostgreSQL
    print_info "Checking PostgreSQL on $REMOTE_HOST:5436..."
    if timeout 5 bash -c "</dev/tcp/$REMOTE_HOST/5436" 2>/dev/null; then
        print_success "PostgreSQL accessible"
    else
        print_error "PostgreSQL not accessible on $REMOTE_HOST:5436"
        print_warning "Tests may fail if database is required"
    fi

    # Check Consul
    print_info "Checking Consul on $REMOTE_HOST:28500..."
    if timeout 5 bash -c "</dev/tcp/$REMOTE_HOST/28500" 2>/dev/null; then
        print_success "Consul accessible"
    else
        print_error "Consul not accessible on $REMOTE_HOST:28500"
        print_warning "Tests may fail if Consul is required"
    fi

    # Check Orchestrator
    print_info "Checking Orchestrator on $REMOTE_HOST:8060..."
    if timeout 5 bash -c "</dev/tcp/$REMOTE_HOST/8060" 2>/dev/null; then
        print_success "Orchestrator accessible"
    else
        print_warning "Orchestrator not accessible on $REMOTE_HOST:8060"
    fi

    print_success "Service verification complete"
fi

# ============================================================================
# Run Tests
# ============================================================================

print_header "Running Integration Tests"

# Build pytest command
PYTEST_CMD="poetry run pytest tests/integration/"

# Add verbose flag
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add test pattern
if [ -n "$TEST_PATTERN" ]; then
    PYTEST_CMD="$PYTEST_CMD -k $TEST_PATTERN"
fi

# Add coverage
if [ "$GENERATE_COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src/omninode_bridge --cov-report=html --cov-report=term"
fi

# Add markers for integration tests
PYTEST_CMD="$PYTEST_CMD -m integration"

# Add traceback option
PYTEST_CMD="$PYTEST_CMD --tb=short"

print_info "Running: $PYTEST_CMD"
echo ""

# Run tests
if eval "$PYTEST_CMD"; then
    print_success "All tests passed!"
    EXIT_CODE=0
else
    print_error "Some tests failed"
    EXIT_CODE=1
fi

# ============================================================================
# Coverage Report
# ============================================================================

if [ "$GENERATE_COVERAGE" = true ] && [ $EXIT_CODE -eq 0 ]; then
    print_header "Coverage Report"
    print_info "HTML coverage report generated at: htmlcov/index.html"

    # Try to open coverage report in browser
    if command -v open &> /dev/null; then
        print_info "Opening coverage report in browser..."
        open htmlcov/index.html 2>/dev/null || true
    fi
fi

# ============================================================================
# Summary
# ============================================================================

print_header "Test Summary"

if [ $EXIT_CODE -eq 0 ]; then
    print_success "Integration tests completed successfully"
    print_info "Remote system: $REMOTE_HOST"
    print_info "Tests run against distributed infrastructure"
else
    print_error "Integration tests failed"
    print_info "Check output above for details"
    print_info "Remote system: $REMOTE_HOST"
fi

echo ""

exit $EXIT_CODE
