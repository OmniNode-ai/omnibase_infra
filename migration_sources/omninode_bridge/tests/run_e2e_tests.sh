#!/bin/bash
#
# End-to-End Test Runner for Two-Way Registration Pattern
#
# This script:
# 1. Starts test environment (Docker Compose)
# 2. Waits for all services to be healthy
# 3. Runs E2E tests
# 4. Runs load tests (optional)
# 5. Collects logs and metrics
# 6. Cleans up test environment
#
# Usage:
#   ./run_e2e_tests.sh                 # Run E2E tests only
#   ./run_e2e_tests.sh --with-load     # Run E2E + load tests
#   ./run_e2e_tests.sh --no-cleanup    # Keep environment running after tests
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_COMPOSE_FILE="$SCRIPT_DIR/docker-compose.test.yml"
LOG_DIR="$SCRIPT_DIR/logs"
TEST_RESULTS_DIR="$SCRIPT_DIR/results"

# Flags
RUN_LOAD_TESTS=false
NO_CLEANUP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-load)
            RUN_LOAD_TESTS=true
            shift
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --with-load      Run load tests in addition to E2E tests"
            echo "  --no-cleanup     Keep Docker environment running after tests"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Function to print colored messages
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

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    log_success "Docker is running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed. Please install it and try again."
        exit 1
    fi
    log_success "docker-compose is available"
}

# Function to create directories
setup_directories() {
    log_info "Creating test directories..."
    mkdir -p "$LOG_DIR"
    mkdir -p "$TEST_RESULTS_DIR"
    log_success "Directories created"
}

# Function to start test environment
start_environment() {
    log_info "Starting test environment..."
    log_info "Docker Compose file: $DOCKER_COMPOSE_FILE"

    # Build and start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d --build

    log_success "Test environment started"
}

# Function to wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to be healthy..."

    local max_wait=120  # 2 minutes
    local elapsed=0
    local check_interval=5

    services=("postgres-test" "redpanda-test" "consul-test" "orchestrator-test" "reducer-test" "registry-test")

    while [ $elapsed -lt $max_wait ]; do
        all_healthy=true

        for service in "${services[@]}"; do
            container_name="registration-test-${service%-test}"
            health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "not_found")

            if [ "$health_status" != "healthy" ]; then
                all_healthy=false
                log_info "Waiting for $service to be healthy (current: $health_status)..."
                break
            fi
        done

        if [ "$all_healthy" = true ]; then
            log_success "All services are healthy"
            return 0
        fi

        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done

    log_error "Services did not become healthy within ${max_wait} seconds"
    log_info "Collecting logs for debugging..."
    collect_logs
    return 1
}

# Function to run E2E tests
run_e2e_tests() {
    log_info "Running E2E tests..."

    cd "$PROJECT_ROOT"

    # Run pytest with E2E tests
    if poetry run pytest tests/integration/test_two_way_registration_e2e.py \
        -v \
        --tb=short \
        --junitxml="$TEST_RESULTS_DIR/junit-e2e.xml" \
        --html="$TEST_RESULTS_DIR/report-e2e.html" \
        --self-contained-html \
        2>&1 | tee "$LOG_DIR/e2e_tests.log"; then
        log_success "E2E tests passed"
        return 0
    else
        log_error "E2E tests failed"
        return 1
    fi
}

# Function to run load tests
run_load_tests() {
    log_info "Running load tests..."

    cd "$PROJECT_ROOT"

    # Run pytest with load tests
    if poetry run pytest tests/load/test_introspection_load.py \
        -v \
        -s \
        -m load \
        --tb=short \
        --junitxml="$TEST_RESULTS_DIR/junit-load.xml" \
        --html="$TEST_RESULTS_DIR/report-load.html" \
        --self-contained-html \
        2>&1 | tee "$LOG_DIR/load_tests.log"; then
        log_success "Load tests passed"
        return 0
    else
        log_error "Load tests failed"
        return 1
    fi
}

# Function to collect logs from containers
collect_logs() {
    log_info "Collecting container logs..."

    containers=("registration-test-postgres" "registration-test-redpanda" "registration-test-consul" "registration-test-orchestrator" "registration-test-reducer" "registration-test-registry")

    for container in "${containers[@]}"; do
        log_file="$LOG_DIR/${container}.log"
        if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
            docker logs "$container" > "$log_file" 2>&1
            log_info "Saved logs for $container to $log_file"
        else
            log_warning "Container $container not found"
        fi
    done

    log_success "Logs collected"
}

# Function to collect metrics
collect_metrics() {
    log_info "Collecting metrics..."

    # Get registry metrics
    if curl -s http://localhost:8062/metrics > "$TEST_RESULTS_DIR/registry_metrics.txt" 2>&1; then
        log_info "Collected registry metrics"
    else
        log_warning "Failed to collect registry metrics"
    fi

    # Get orchestrator metrics
    if curl -s http://localhost:8060/metrics > "$TEST_RESULTS_DIR/orchestrator_metrics.txt" 2>&1; then
        log_info "Collected orchestrator metrics"
    else
        log_warning "Failed to collect orchestrator metrics"
    fi

    # Get reducer metrics
    if curl -s http://localhost:8061/metrics > "$TEST_RESULTS_DIR/reducer_metrics.txt" 2>&1; then
        log_info "Collected reducer metrics"
    else
        log_warning "Failed to collect reducer metrics"
    fi

    log_success "Metrics collected"
}

# Function to stop and cleanup environment
cleanup_environment() {
    if [ "$NO_CLEANUP" = true ]; then
        log_warning "Skipping cleanup (--no-cleanup flag set)"
        log_info "To manually stop environment, run:"
        log_info "  docker-compose -f $DOCKER_COMPOSE_FILE down -v"
        return 0
    fi

    log_info "Stopping and cleaning up test environment..."

    # Stop and remove containers, networks, and volumes
    docker-compose -f "$DOCKER_COMPOSE_FILE" down -v

    log_success "Environment cleaned up"
}

# Function to generate test report
generate_report() {
    log_info "Generating test report..."

    report_file="$TEST_RESULTS_DIR/summary.txt"

    cat > "$report_file" << EOF
========================================
Two-Way Registration E2E Test Summary
========================================
Date: $(date)
Test Environment: Docker Compose
========================================

Test Results:
EOF

    if [ -f "$TEST_RESULTS_DIR/junit-e2e.xml" ]; then
        echo "  E2E Tests: COMPLETED" >> "$report_file"
    else
        echo "  E2E Tests: FAILED" >> "$report_file"
    fi

    if [ "$RUN_LOAD_TESTS" = true ] && [ -f "$TEST_RESULTS_DIR/junit-load.xml" ]; then
        echo "  Load Tests: COMPLETED" >> "$report_file"
    elif [ "$RUN_LOAD_TESTS" = true ]; then
        echo "  Load Tests: FAILED" >> "$report_file"
    else
        echo "  Load Tests: SKIPPED" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

Artifacts:
  - Logs: $LOG_DIR
  - Test Results: $TEST_RESULTS_DIR
  - HTML Reports: $TEST_RESULTS_DIR/report-*.html

========================================
EOF

    cat "$report_file"
    log_success "Test report generated: $report_file"
}

# Main execution
main() {
    log_info "=== Two-Way Registration E2E Test Runner ==="
    log_info "Run with --help for usage information"
    echo ""

    # Pre-flight checks
    check_docker
    check_docker_compose
    setup_directories

    # Start environment
    start_environment

    # Wait for services
    if ! wait_for_services; then
        log_error "Failed to start test environment"
        cleanup_environment
        exit 1
    fi

    # Run tests
    test_failed=false

    if ! run_e2e_tests; then
        test_failed=true
    fi

    if [ "$RUN_LOAD_TESTS" = true ]; then
        if ! run_load_tests; then
            test_failed=true
        fi
    fi

    # Collect logs and metrics
    collect_logs
    collect_metrics

    # Generate report
    generate_report

    # Cleanup
    cleanup_environment

    # Exit with appropriate code
    if [ "$test_failed" = true ]; then
        log_error "Some tests failed. Check logs and reports for details."
        exit 1
    else
        log_success "All tests passed!"
        exit 0
    fi
}

# Run main function
main
