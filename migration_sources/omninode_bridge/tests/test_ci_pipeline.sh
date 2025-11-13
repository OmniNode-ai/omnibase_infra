#!/bin/bash
#
# CI Pipeline Test Script
# Tests the fixed CI pipeline locally to validate all components work together
#
# Usage: ./test_ci_pipeline.sh [component]
# Components: health-checks, docker-compose, artifacts, full-pipeline
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_COMPOSE_FILE="$SCRIPT_DIR/docker-compose.test.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to show test header
show_test_header() {
    echo ""
    echo "========================================"
    echo "Testing: $1"
    echo "========================================"
    echo ""
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

# Function to test health check script
test_health_checks() {
    show_test_header "Health Check Script"

    log_info "Testing health check script..."

    # Test the script exists and is executable
    if [ ! -f "$SCRIPT_DIR/wait_for_services.sh" ]; then
        log_error "Health check script not found: $SCRIPT_DIR/wait_for_services.sh"
        return 1
    fi

    if [ ! -x "$SCRIPT_DIR/wait_for_services.sh" ]; then
        log_error "Health check script is not executable"
        return 1
    fi

    log_success "Health check script exists and is executable"

    # Test help function
    if "$SCRIPT_DIR/wait_for_services.sh" --help > /dev/null 2>&1; then
        log_success "Health check script help works"
    else
        log_warning "Health check script help not available"
    fi

    return 0
}

# Function to test Docker Compose configuration
test_docker_compose() {
    show_test_header "Docker Compose Configuration"

    log_info "Testing Docker Compose configuration..."

    # Test configuration file exists
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
        return 1
    fi

    log_success "Docker Compose configuration file exists"

    # Test configuration validation
    if docker-compose -f "$DOCKER_COMPOSE_FILE" config > /dev/null 2>&1; then
        log_success "Docker Compose configuration is valid"
    else
        log_error "Docker Compose configuration is invalid"
        return 1
    fi

    # Test entrypoint script exists
    if [ ! -f "$PROJECT_ROOT/docker/bridge-nodes/entrypoint.sh" ]; then
        log_error "Entrypoint script not found"
        return 1
    fi

    log_success "Entrypoint script exists"

    # Test entrypoint script is executable
    if [ -x "$PROJECT_ROOT/docker/bridge-nodes/entrypoint.sh" ]; then
        log_success "Entrypoint script is executable"
    else
        log_error "Entrypoint script is not executable"
        return 1
    fi

    return 0
}

# Function to test Docker build
test_docker_build() {
    show_test_header "Docker Build"

    log_info "Testing Docker build for orchestrator..."
    if docker build -f "$PROJECT_ROOT/docker/bridge-nodes/Dockerfile.orchestrator" -t test-orchestrator "$PROJECT_ROOT" > /dev/null 2>&1; then
        log_success "Orchestrator Docker build successful"
    else
        log_error "Orchestrator Docker build failed"
        return 1
    fi

    log_info "Testing Docker build for reducer..."
    if docker build -f "$PROJECT_ROOT/docker/bridge-nodes/Dockerfile.reducer" -t test-reducer "$PROJECT_ROOT" > /dev/null 2>&1; then
        log_success "Reducer Docker build successful"
    else
        log_error "Reducer Docker build failed"
        return 1
    fi

    log_info "Testing Docker build for registry..."
    if docker build -f "$PROJECT_ROOT/docker/bridge-nodes/Dockerfile.registry" -t test-registry "$PROJECT_ROOT" > /dev/null 2>&1; then
        log_success "Registry Docker build successful"
    else
        log_error "Registry Docker build failed"
        return 1
    fi

    # Clean up test images
    docker rmi test-orchestrator test-reducer test-registry 2>/dev/null || true

    return 0
}

# Function to test service startup
test_service_startup() {
    show_test_header "Service Startup"

    log_info "Starting test environment..."

    # Start infrastructure services only ( faster test )
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d postgres-test redpanda-test consul-test

    log_info "Waiting for infrastructure services..."
    sleep 30

    # Check infrastructure services
    log_info "Checking infrastructure services..."

    if docker exec registration-test-postgres pg_isready -U test -d bridge_test >/dev/null 2>&1; then
        log_success "PostgreSQL is healthy"
    else
        log_error "PostgreSQL is not healthy"
        return 1
    fi

    if docker exec registration-test-redpanda rpk cluster info >/dev/null 2>&1; then
        log_success "RedPanda is healthy"
    else
        log_error "RedPanda is not healthy"
        return 1
    fi

    if docker exec registration-test-consul consul members >/dev/null 2>&1; then
        log_success "Consul is healthy"
    else
        log_error "Consul is not healthy"
        return 1
    fi

    log_success "All infrastructure services are healthy"

    # Stop infrastructure services
    log_info "Stopping infrastructure services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down

    return 0
}

# Function to test health check wait script
test_health_wait_script() {
    show_test_header "Health Wait Script"

    log_info "Testing health wait script with all services..."

    # Start all services (infrastructure + bridge nodes)
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d

    # Test the health wait script
    if timeout 180 "$SCRIPT_DIR/wait_for_services.sh" 120; then
        log_success "Health wait script works correctly"
    else
        log_error "Health wait script failed or timed out"
        # Collect debug info
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs > /tmp/health_test_debug.log
        log_error "Debug logs saved to /tmp/health_test_debug.log"
        return 1
    fi

    # Stop services
    docker-compose -f "$DOCKER_COMPOSE_FILE" down

    return 0
}

# Function to test artifact generation
test_artifact_generation() {
    show_test_header "Artifact Generation"

    log_info "Testing pytest artifact generation..."

    cd "$PROJECT_ROOT"

    # Create test directories
    mkdir -p test-results test-logs

    # Run a simple test to check artifact generation
    if poetry run pytest tests/integration/test_two_way_registration_e2e.py \
        -v \
        --tb=short \
        --junitxml=test-results/junit-e2e.xml \
        --html=test-results/report-e2e.html \
        --self-contained-html \
        --cov=src/omninode_bridge/nodes \
        --cov-report=xml:test-results/coverage-e2e.xml \
        --cov-report=html:test-results/htmlcov-e2e \
        --override-ini="addopts=" \
        --maxfail=1 \
        --collect-only; then

        log_success "Pytest artifact collection test passed"
    else
        log_error "Pytest artifact collection test failed"
        return 1
    fi

    # Check if artifacts would be generated ( test collection only )
    log_info "Checking artifact paths..."

    # These should exist after collect-only
    if [ -f "test-results/junit-e2e.xml" ]; then
        log_success "JUnit XML artifact path is accessible"
    else
        log_warning "JUnit XML artifact not generated in collect-only mode (expected)"
    fi

    # Clean up
    rm -rf test-results test-logs

    return 0
}

# Function to test full pipeline
test_full_pipeline() {
    show_test_header "Full Pipeline Test"

    log_info "Testing full CI pipeline simulation..."

    # Test 1: Health checks
    if ! test_health_checks; then
        log_error "Health checks test failed"
        return 1
    fi

    # Test 2: Docker Compose configuration
    if ! test_docker_compose; then
        log_error "Docker Compose configuration test failed"
        return 1
    fi

    # Test 3: Docker build
    if ! test_docker_build; then
        log_error "Docker build test failed"
        return 1
    fi

    # Test 4: Service startup
    if ! test_service_startup; then
        log_error "Service startup test failed"
        return 1
    fi

    # Test 5: Health wait script
    if ! test_health_wait_script; then
        log_error "Health wait script test failed"
        return 1
    fi

    # Test 6: Artifact generation
    if ! test_artifact_generation; then
        log_error "Artifact generation test failed"
        return 1
    fi

    log_success "All pipeline tests passed!"
    return 0
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [component]"
    echo ""
    echo "Components:"
    echo "  health-checks      Test health check script"
    echo "  docker-compose     Test Docker Compose configuration"
    echo "  docker-build       Test Docker builds"
    echo "  service-startup    Test service startup"
    echo "  health-wait        Test health wait script"
    echo "  artifacts          Test artifact generation"
    echo "  full-pipeline      Test all components (default)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Test full pipeline"
    echo "  $0 health-checks      # Test only health checks"
    echo "  $0 docker-compose     # Test only Docker Compose"
}

# Main execution
main() {
    local component="${1:-full-pipeline}"

    log_info "=== CI Pipeline Test Script ==="
    log_info "Component: $component"
    echo ""

    # Pre-flight checks
    check_docker
    check_docker_compose

    # Run selected test
    case "$component" in
        "health-checks")
            test_health_checks
            ;;
        "docker-compose")
            test_docker_compose
            ;;
        "docker-build")
            test_docker_build
            ;;
        "service-startup")
            test_service_startup
            ;;
        "health-wait")
            test_health_wait_script
            ;;
        "artifacts")
            test_artifact_generation
            ;;
        "full-pipeline")
            test_full_pipeline
            ;;
        "--help"|"help"|"-h")
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown component: $component"
            show_usage
            exit 1
            ;;
    esac

    local result=$?
    if [ $result -eq 0 ]; then
        echo ""
        log_success "✅ Test completed successfully!"
        exit 0
    else
        echo ""
        log_error "❌ Test failed!"
        exit 1
    fi
}

# Run main function
main "$@"
