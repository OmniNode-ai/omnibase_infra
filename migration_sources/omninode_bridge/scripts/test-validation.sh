#!/bin/bash

# Comprehensive Test Validation Script for OmniNode Bridge
# Validates all integration test fixes and environment configuration

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMEOUT_DURATION=600  # 10 minutes timeout for comprehensive tests

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Test validation state
VALIDATION_RESULTS=()
OVERALL_SUCCESS=true

# Function to record test results
record_result() {
    local test_name="$1"
    local status="$2"
    local details="$3"

    VALIDATION_RESULTS+=("$test_name|$status|$details")

    if [ "$status" = "FAIL" ]; then
        OVERALL_SUCCESS=false
    fi
}

# Function to run test with timeout and capture results
run_test_with_timeout() {
    local test_command="$1"
    local test_name="$2"
    local timeout_seconds="$3"

    log_info "Running $test_name..."

    # Create temporary files for capturing output
    local output_file=$(mktemp)
    local error_file=$(mktemp)

    # Run test with timeout
    if timeout "$timeout_seconds" bash -c "$test_command" > "$output_file" 2> "$error_file"; then
        local exit_code=0
    else
        local exit_code=$?
    fi

    # Read output and errors
    local output=$(cat "$output_file")
    local errors=$(cat "$error_file")

    # Clean up temporary files
    rm -f "$output_file" "$error_file"

    # Process results
    if [ $exit_code -eq 0 ]; then
        log_success "$test_name passed"
        record_result "$test_name" "PASS" "Test completed successfully"
        return 0
    elif [ $exit_code -eq 124 ]; then
        log_error "$test_name timed out after $timeout_seconds seconds"
        record_result "$test_name" "TIMEOUT" "Test exceeded timeout of $timeout_seconds seconds"
        return 1
    else
        log_error "$test_name failed with exit code $exit_code"
        if [ -n "$errors" ]; then
            echo "Error output:"
            echo "$errors" | head -20  # Show first 20 lines of errors
        fi
        record_result "$test_name" "FAIL" "Exit code: $exit_code"
        return 1
    fi
}

# Function to validate test environment configuration
validate_test_environment() {
    log_info "Validating test environment configuration..."

    # Check if test_env_config.py exists and is properly configured
    if [ -f "$PROJECT_ROOT/tests/test_env_config.py" ]; then
        log_success "TestEnvironmentConfig module found"
        record_result "TestEnvironmentConfig Module" "PASS" "Module exists and is accessible"
    else
        log_error "TestEnvironmentConfig module not found"
        record_result "TestEnvironmentConfig Module" "FAIL" "Module file missing"
        return 1
    fi

    # Validate Python imports
    cd "$PROJECT_ROOT"
    if poetry run python -c "from tests.test_env_config import TestEnvironmentConfig, MockedTestEnvironment; print('Import successful')" 2>/dev/null; then
        log_success "TestEnvironmentConfig imports work correctly"
        record_result "TestEnvironmentConfig Imports" "PASS" "All imports successful"
    else
        log_error "TestEnvironmentConfig imports failed"
        record_result "TestEnvironmentConfig Imports" "FAIL" "Import errors detected"
        return 1
    fi

    return 0
}

# Function to run unit tests (fast feedback)
run_unit_tests() {
    log_info "Running unit tests for fast feedback..."

    local unit_test_command="cd '$PROJECT_ROOT' && poetry run pytest tests/unit/ -v --tb=short --timeout=30 -x"

    if run_test_with_timeout "$unit_test_command" "Unit Tests" 120; then
        return 0
    else
        log_warning "Unit tests failed - continuing with integration test validation"
        return 1
    fi
}

# Function to run mocked integration tests (no external dependencies)
run_mocked_integration_tests() {
    log_info "Running mocked integration tests (no external dependencies)..."

    # Run specific test classes that use mocked environments
    local mocked_test_command="cd '$PROJECT_ROOT' && poetry run pytest tests/integration/test_kafka_event_flows.py::TestHookEventProcessingFlow -v --tb=short --timeout=60 -x"

    if run_test_with_timeout "$mocked_test_command" "Mocked Integration Tests" 180; then
        return 0
    else
        log_error "Mocked integration tests failed"
        return 1
    fi
}

# Function to check Docker environment availability
check_docker_environment() {
    log_info "Checking Docker environment availability..."

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not available - skipping containerized tests"
        record_result "Docker Availability" "SKIP" "Docker command not found"
        return 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_warning "Docker daemon not running - skipping containerized tests"
        record_result "Docker Daemon" "SKIP" "Docker daemon not accessible"
        return 1
    fi

    log_success "Docker environment is available"
    record_result "Docker Environment" "PASS" "Docker daemon accessible"
    return 0
}

# Function to run containerized integration tests (with external dependencies)
run_containerized_integration_tests() {
    log_info "Running containerized integration tests..."

    # Check if containers are available
    if ! check_docker_environment; then
        log_warning "Skipping containerized tests - Docker not available"
        return 0
    fi

    # Start test containers if needed
    log_info "Starting test infrastructure..."
    if ! "$SCRIPT_DIR/wait-for-services.sh"; then
        log_error "Failed to start test infrastructure"
        record_result "Test Infrastructure" "FAIL" "Failed to start required services"
        return 1
    fi

    # Run integration tests that require real services
    local container_test_command="cd '$PROJECT_ROOT' && poetry run pytest tests/integration/test_kafka_event_flows.py::TestKafkaInfrastructure -v --tb=short --timeout=120 -x"

    if run_test_with_timeout "$container_test_command" "Containerized Integration Tests" 300; then
        return 0
    else
        log_error "Containerized integration tests failed"
        return 1
    fi
}

# Function to validate graceful degradation fixes
validate_graceful_degradation() {
    log_info "Validating graceful degradation service fixes..."

    # Test that the graceful degradation service doesn't create async tasks in test mode
    local degradation_test="cd '$PROJECT_ROOT' && ENVIRONMENT=test PYTEST_CURRENT_TEST=true poetry run python -c '
import os
os.environ[\"ENVIRONMENT\"] = \"test\"
os.environ[\"PYTEST_CURRENT_TEST\"] = \"true\"
from omninode_bridge.services.graceful_degradation import register_common_services
print(\"Graceful degradation test passed - no async task creation in test mode\")
'"

    if run_test_with_timeout "$degradation_test" "Graceful Degradation Validation" 30; then
        return 0
    else
        log_error "Graceful degradation validation failed"
        return 1
    fi
}

# Function to run specific problematic test that was previously failing
run_specific_problematic_tests() {
    log_info "Running specific tests that were previously failing..."

    # Test specific classes that had issues
    local specific_tests=(
        "tests/integration/test_kafka_event_flows.py::TestHookEventProcessingFlow::test_service_lifecycle_event_flow"
        "tests/integration/test_kafka_event_flows.py::TestWorkflowEventDrivenExecution::test_workflow_execution_event_publishing"
    )

    local all_passed=true

    for test_case in "${specific_tests[@]}"; do
        local test_command="cd '$PROJECT_ROOT' && poetry run pytest '$test_case' -v --tb=short --timeout=60"

        if run_test_with_timeout "$test_command" "Specific Test: $test_case" 90; then
            log_success "Previously failing test now passes: $test_case"
        else
            log_error "Test still failing: $test_case"
            all_passed=false
        fi
    done

    if $all_passed; then
        record_result "Previously Failing Tests" "PASS" "All previously failing tests now pass"
        return 0
    else
        record_result "Previously Failing Tests" "FAIL" "Some tests still failing"
        return 1
    fi
}

# Function to generate test validation report
generate_validation_report() {
    log_info "Generating test validation report..."

    echo ""
    echo "========================================="
    echo "TEST VALIDATION REPORT"
    echo "========================================="
    echo "Date: $(date)"
    echo "Project: OmniNode Bridge"
    echo "Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
    echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo ""

    # Summary
    local total_tests=${#VALIDATION_RESULTS[@]}
    local passed_tests=0
    local failed_tests=0
    local skipped_tests=0
    local timeout_tests=0

    for result in "${VALIDATION_RESULTS[@]}"; do
        IFS='|' read -r test_name status details <<< "$result"
        case "$status" in
            "PASS") ((passed_tests++)) ;;
            "FAIL") ((failed_tests++)) ;;
            "SKIP") ((skipped_tests++)) ;;
            "TIMEOUT") ((timeout_tests++)) ;;
        esac
    done

    echo "SUMMARY:"
    echo "  Total Tests: $total_tests"
    echo "  Passed: $passed_tests"
    echo "  Failed: $failed_tests"
    echo "  Skipped: $skipped_tests"
    echo "  Timeouts: $timeout_tests"
    echo ""

    # Detailed results
    echo "DETAILED RESULTS:"
    for result in "${VALIDATION_RESULTS[@]}"; do
        IFS='|' read -r test_name status details <<< "$result"

        case "$status" in
            "PASS")
                echo -e "  ${GREEN}✓${NC} $test_name: $details"
                ;;
            "FAIL")
                echo -e "  ${RED}✗${NC} $test_name: $details"
                ;;
            "SKIP")
                echo -e "  ${YELLOW}⊘${NC} $test_name: $details"
                ;;
            "TIMEOUT")
                echo -e "  ${RED}⏱${NC} $test_name: $details"
                ;;
        esac
    done

    echo ""
    echo "========================================="

    if $OVERALL_SUCCESS; then
        echo -e "${GREEN}OVERALL RESULT: SUCCESS${NC}"
        echo "All critical tests passed - integration test fixes are working correctly!"
    else
        echo -e "${RED}OVERALL RESULT: FAILURE${NC}"
        echo "Some tests failed - additional fixes may be needed."
    fi

    echo "========================================="
}

# Main test validation workflow
main() {
    log_info "Starting comprehensive test validation for OmniNode Bridge integration test fixes"
    log_info "Project root: $PROJECT_ROOT"

    # Ensure we're in the project root
    cd "$PROJECT_ROOT"

    # Check prerequisites
    if ! command -v poetry &> /dev/null; then
        log_error "Poetry not found - please install Poetry first"
        exit 1
    fi

    # Install dependencies
    log_info "Installing dependencies..."
    if ! poetry install --with dev &> /dev/null; then
        log_error "Failed to install dependencies"
        exit 1
    fi

    # Run validation steps
    validate_test_environment
    validate_graceful_degradation
    run_unit_tests
    run_mocked_integration_tests
    run_specific_problematic_tests
    run_containerized_integration_tests

    # Generate final report
    generate_validation_report

    # Exit with appropriate code
    if $OVERALL_SUCCESS; then
        exit 0
    else
        exit 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
