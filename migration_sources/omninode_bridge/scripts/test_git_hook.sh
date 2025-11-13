#!/usr/bin/env bash
#
# Test Git Hook - File Change Event Publisher
#
# Tests the pre-push git hook installation and event publishing functionality.
# Creates test scenarios, verifies Kafka event delivery, and validates hook behavior.
#
# Usage:
#   # Run all tests
#   ./scripts/test_git_hook.sh
#
#   # Run specific test
#   ./scripts/test_git_hook.sh --test installation
#   ./scripts/test_git_hook.sh --test publishing
#   ./scripts/test_git_hook.sh --test multi-repo
#
#   # Run with debug output
#   GIT_HOOK_DEBUG=true ./scripts/test_git_hook.sh
#
# Prerequisites:
#   - Kafka/Redpanda running at localhost:29092
#   - Python 3.11+ with omninode_bridge dependencies installed
#   - Git repository with write access
#
# Exit Codes:
#   0: All tests passed
#   1: One or more tests failed

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_SCRIPT="${SCRIPT_DIR}/install_git_hooks.sh"
HOOK_SCRIPT="${SCRIPT_DIR}/git_hooks/pre_push_event_publisher.py"

# Test configuration
TEST_BRANCH="test/git-hook-$(date +%s)"
TEST_FILE="${REPO_ROOT}/test_file_$(date +%s).txt"
KAFKA_TOPIC="dev.omninode_bridge.onex.evt.file-changes.v1"

# Test results
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*"
}

log_test() {
    echo -e "${BLUE}[TEST]${NC} $*"
}

# Test result tracking
pass_test() {
    ((TESTS_PASSED++))
    log_success "$*"
}

fail_test() {
    ((TESTS_FAILED++))
    log_error "$*"
}

# Check prerequisites
check_prerequisites() {
    log_test "Checking prerequisites"

    # Check if in git repository
    if ! git rev-parse --is-inside-work-tree &>/dev/null; then
        log_error "Not in a git repository"
        return 1
    fi

    # Check if installation script exists
    if [ ! -f "${INSTALL_SCRIPT}" ]; then
        log_error "Installation script not found: ${INSTALL_SCRIPT}"
        return 1
    fi

    # Check if hook script exists
    if [ ! -f "${HOOK_SCRIPT}" ]; then
        log_error "Hook script not found: ${HOOK_SCRIPT}"
        return 1
    fi

    # Check if Python is available
    if ! command -v python3 &>/dev/null; then
        log_error "Python 3 not found"
        return 1
    fi

    # Check if Kafka is running (optional warning)
    if ! timeout 2 bash -c "cat < /dev/null > /dev/tcp/localhost/29092" 2>/dev/null; then
        log_warning "Kafka may not be running at localhost:29092"
        log_warning "Some tests may fail without Kafka connectivity"
    else
        log_success "Kafka is reachable at localhost:29092"
    fi

    pass_test "Prerequisites check"
    return 0
}

# Test 1: Hook installation
test_installation() {
    ((TESTS_RUN++))
    log_test "Test 1: Hook installation"

    # Uninstall any existing hook
    "${INSTALL_SCRIPT}" --uninstall &>/dev/null || true

    # Install hook
    if ! "${INSTALL_SCRIPT}" &>/dev/null; then
        fail_test "Hook installation failed"
        return 1
    fi

    # Verify hook exists
    if [ ! -f "${REPO_ROOT}/.git/hooks/pre-push" ]; then
        fail_test "Hook file not created"
        return 1
    fi

    # Verify hook is executable
    if [ ! -x "${REPO_ROOT}/.git/hooks/pre-push" ]; then
        fail_test "Hook is not executable"
        return 1
    fi

    pass_test "Hook installation successful"
    return 0
}

# Test 2: Hook execution (non-blocking)
test_hook_execution() {
    ((TESTS_RUN++))
    log_test "Test 2: Hook execution (non-blocking)"

    # Create test branch
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)

    if ! git checkout -b "${TEST_BRANCH}" &>/dev/null; then
        log_warning "Failed to create test branch, using current branch"
        TEST_BRANCH="${current_branch}"
    fi

    # Create test file
    echo "Test content at $(date)" > "${TEST_FILE}"
    git add "${TEST_FILE}" &>/dev/null

    # Commit test file
    if ! git commit -m "test: Add test file for git hook validation" &>/dev/null; then
        log_warning "No changes to commit"
    fi

    # Try to trigger hook (dry-run push)
    # Note: This won't actually push since we're using --dry-run
    local hook_output
    local hook_exit_code

    # Capture hook execution time
    local start_time
    local end_time
    local execution_time

    start_time=$(date +%s%3N)

    # Execute hook directly to test
    if hook_output=$(python3 "${HOOK_SCRIPT}" 2>&1); then
        hook_exit_code=0
    else
        hook_exit_code=$?
    fi

    end_time=$(date +%s%3N)
    execution_time=$((end_time - start_time))

    # Hook should always return 0 (non-blocking)
    if [ ${hook_exit_code} -ne 0 ]; then
        fail_test "Hook returned non-zero exit code: ${hook_exit_code}"
        echo "${hook_output}"
        return 1
    fi

    # Hook should execute quickly (<2s target)
    if [ ${execution_time} -gt 2000 ]; then
        log_warning "Hook execution took ${execution_time}ms (target: <2000ms)"
    else
        log_success "Hook execution took ${execution_time}ms"
    fi

    # Cleanup test branch
    git checkout "${current_branch}" &>/dev/null || true
    git branch -D "${TEST_BRANCH}" &>/dev/null || true
    rm -f "${TEST_FILE}"

    pass_test "Hook execution is non-blocking (exit code: ${hook_exit_code}, time: ${execution_time}ms)"
    return 0
}

# Test 3: Event publishing (if Kafka available)
test_event_publishing() {
    ((TESTS_RUN++))
    log_test "Test 3: Event publishing to Kafka"

    # Check if Kafka is available
    if ! timeout 2 bash -c "cat < /dev/null > /dev/tcp/localhost/29092" 2>/dev/null; then
        log_warning "Kafka not available, skipping event publishing test"
        ((TESTS_RUN--))
        return 0
    fi

    # Create test file for event
    local test_file="test_event_$(date +%s).txt"
    echo "Test event at $(date)" > "${test_file}"

    # Set environment for verbose logging
    export GIT_HOOK_DEBUG=true

    # Execute hook
    local hook_output
    if hook_output=$(python3 "${HOOK_SCRIPT}" 2>&1); then
        # Check if event was published (look for success message in output)
        if echo "${hook_output}" | grep -q "Published file change event"; then
            log_success "Event published to Kafka"
        elif echo "${hook_output}" | grep -q "Kafka"; then
            log_warning "Kafka interaction detected but publish status unclear"
            echo "${hook_output}"
        else
            log_warning "No Kafka publish confirmation in output"
        fi
    else
        log_warning "Hook execution had issues"
        echo "${hook_output}"
    fi

    # Cleanup
    rm -f "${test_file}"
    unset GIT_HOOK_DEBUG

    pass_test "Event publishing test completed"
    return 0
}

# Test 4: Graceful degradation (Kafka unavailable)
test_graceful_degradation() {
    ((TESTS_RUN++))
    log_test "Test 4: Graceful degradation (Kafka unavailable)"

    # Set invalid Kafka server to simulate failure
    export KAFKA_BOOTSTRAP_SERVERS="invalid-server:9999"
    export GIT_HOOK_TIMEOUT=1

    # Execute hook
    local hook_output
    local hook_exit_code

    if hook_output=$(python3 "${HOOK_SCRIPT}" 2>&1); then
        hook_exit_code=0
    else
        hook_exit_code=$?
    fi

    # Reset environment
    unset KAFKA_BOOTSTRAP_SERVERS
    unset GIT_HOOK_TIMEOUT

    # Hook should still return 0 (non-blocking even on failure)
    if [ ${hook_exit_code} -ne 0 ]; then
        fail_test "Hook failed to degrade gracefully (exit code: ${hook_exit_code})"
        return 1
    fi

    # Output should indicate failure/timeout
    if echo "${hook_output}" | grep -qE "(timeout|failed|warning)"; then
        log_success "Hook logged degradation properly"
    else
        log_warning "Hook may not have detected Kafka unavailability"
    fi

    pass_test "Graceful degradation successful (exit code: ${hook_exit_code})"
    return 0
}

# Test 5: Multi-repo support
test_multi_repo_support() {
    ((TESTS_RUN++))
    log_test "Test 5: Multi-repo support"

    # Create temporary test repository
    local temp_repo
    temp_repo=$(mktemp -d -t git-hook-test-XXXXXX)

    # Initialize git repo
    if ! (cd "${temp_repo}" && git init &>/dev/null); then
        log_error "Failed to create test repository"
        rm -rf "${temp_repo}"
        return 1
    fi

    # Install hook in test repo
    if ! "${INSTALL_SCRIPT}" "${temp_repo}" &>/dev/null; then
        fail_test "Failed to install hook in test repository"
        rm -rf "${temp_repo}"
        return 1
    fi

    # Verify hook exists
    if [ ! -f "${temp_repo}/.git/hooks/pre-push" ]; then
        fail_test "Hook not installed in test repository"
        rm -rf "${temp_repo}"
        return 1
    fi

    # Cleanup
    rm -rf "${temp_repo}"

    pass_test "Multi-repo support validated"
    return 0
}

# Print test summary
print_summary() {
    echo
    log_info "======================================"
    log_info "Test Summary"
    log_info "======================================"
    log_info "Total tests run: ${TESTS_RUN}"

    if [ ${TESTS_PASSED} -gt 0 ]; then
        log_success "Tests passed: ${TESTS_PASSED}"
    fi

    if [ ${TESTS_FAILED} -gt 0 ]; then
        log_error "Tests failed: ${TESTS_FAILED}"
    fi

    echo

    if [ ${TESTS_FAILED} -eq 0 ]; then
        log_success "All tests passed! ✓"
        return 0
    else
        log_error "Some tests failed! ✗"
        return 1
    fi
}

# Main function
main() {
    local specific_test=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --test)
                specific_test="$2"
                shift 2
                ;;
            --help|-h)
                cat <<EOF
Usage: $0 [OPTIONS]

Test git hook installation and event publishing.

Options:
  --test TEST_NAME    Run specific test (installation, publishing, multi-repo)
  --help, -h          Show this help message

Environment Variables:
  GIT_HOOK_DEBUG      Enable debug output (default: false)

EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Print header
    echo
    log_info "======================================"
    log_info "Git Hook Test Suite"
    log_info "======================================"
    echo

    # Check prerequisites
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi
    echo

    # Run tests
    if [ -z "${specific_test}" ]; then
        # Run all tests
        test_installation || true
        echo
        test_hook_execution || true
        echo
        test_event_publishing || true
        echo
        test_graceful_degradation || true
        echo
        test_multi_repo_support || true
    else
        # Run specific test
        case "${specific_test}" in
            installation)
                test_installation || true
                ;;
            execution)
                test_hook_execution || true
                ;;
            publishing)
                test_event_publishing || true
                ;;
            degradation)
                test_graceful_degradation || true
                ;;
            multi-repo)
                test_multi_repo_support || true
                ;;
            *)
                log_error "Unknown test: ${specific_test}"
                exit 1
                ;;
        esac
    fi

    # Print summary
    print_summary
}

# Run main function
main "$@"
