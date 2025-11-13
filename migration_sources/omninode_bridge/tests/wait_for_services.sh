#!/bin/bash
#
# Service Health Check Script for CI/CD
# Replaces hardcoded sleep with proper health validation
#
# Usage: ./wait_for_services.sh [max_wait_seconds]
#

set -e

# Configuration
MAX_WAIT=${1:-120}  # Default 2 minutes
CHECK_INTERVAL=5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

# Function to check PostgreSQL health
check_postgres() {
    log_info "Checking PostgreSQL health..."
    if docker exec registration-test-postgres pg_isready -U test -d bridge_test >/dev/null 2>&1; then
        log_success "PostgreSQL is healthy"
        return 0
    else
        log_warning "PostgreSQL is not ready yet"
        return 1
    fi
}

# Function to check RedPanda health
check_redpanda() {
    log_info "Checking RedPanda health..."
    if docker exec registration-test-redpanda rpk cluster info >/dev/null 2>&1; then
        log_success "RedPanda is healthy"
        return 0
    else
        log_warning "RedPanda is not ready yet"
        return 1
    fi
}

# Function to check Consul health
check_consul() {
    log_info "Checking Consul health..."
    if docker exec registration-test-consul consul members >/dev/null 2>&1; then
        log_success "Consul is healthy"
        return 0
    else
        log_warning "Consul is not ready yet"
        return 1
    fi
}

# Function to check bridge nodes health
check_bridge_nodes() {
    log_info "Checking bridge nodes health..."

    # Check orchestrator
    if docker exec registration-test-orchestrator /app/entrypoint.sh health-check orchestrator >/dev/null 2>&1; then
        log_success "Orchestrator is healthy"
    else
        log_warning "Orchestrator is not ready yet"
        return 1
    fi

    # Check reducer
    if docker exec registration-test-reducer /app/entrypoint.sh health-check reducer >/dev/null 2>&1; then
        log_success "Reducer is healthy"
    else
        log_warning "Reducer is not ready yet"
        return 1
    fi

    # Check registry
    if docker exec registration-test-registry /app/entrypoint.sh health-check registry >/dev/null 2>&1; then
        log_success "Registry is healthy"
    else
        log_warning "Registry is not ready yet"
        return 1
    fi

    return 0
}

# Function to show service status
show_service_status() {
    log_info "Current service status:"
    docker compose -f "$DOCKER_COMPOSE_FILE" ps
    echo ""
}

# Function to collect logs for debugging
collect_debug_logs() {
    log_info "Collecting debug logs..."
    mkdir -p debug-logs

    # Collect logs from all services
    for service in postgres redpanda consul orchestrator reducer registry; do
        container="registration-test-$service"
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            docker logs "$container" > "debug-logs/${service}.log" 2>&1
            log_info "Collected logs for $service"
        fi
    done

    # Show last few lines from each service for quick debugging
    echo ""
    log_error "=== LAST 10 LINES FROM EACH SERVICE ==="
    for service in postgres redpanda consul orchestrator reducer registry; do
        container="registration-test-$service"
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            echo ""
            log_info "--- $container ---"
            docker logs --tail 10 "$container" 2>&1 || true
        fi
    done
}

# Main health check loop
main() {
    log_info "=== Service Health Check for CI/CD ==="
    log_info "Maximum wait time: ${MAX_WAIT} seconds"
    log_info "Check interval: ${CHECK_INTERVAL} seconds"
    echo ""

    local elapsed=0

    while [ $elapsed -lt $MAX_WAIT ]; do
        log_info "Health check attempt $((elapsed / CHECK_INTERVAL + 1))..."

        # Show current status
        show_service_status

        # Check all services
        local all_healthy=true

        if ! check_postgres; then
            all_healthy=false
        fi

        if ! check_redpanda; then
            all_healthy=false
        fi

        if ! check_consul; then
            all_healthy=false
        fi

        # Only check bridge nodes if infrastructure is healthy
        if $all_healthy; then
            if ! check_bridge_nodes; then
                all_healthy=false
            fi
        fi

        if $all_healthy; then
            echo ""
            log_success "All services are healthy!"
            log_success "Total wait time: ${elapsed} seconds"
            exit 0
        fi

        echo ""
        log_info "Waiting ${CHECK_INTERVAL} seconds before next check..."
        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))
    done

    # Timeout reached
    echo ""
    log_error "Services did not become healthy within ${MAX_WAIT} seconds"

    # Collect debug information
    collect_debug_logs

    log_error "=== HEALTH CHECK FAILED ==="
    log_error "Please check the debug-logs directory for detailed service logs"

    exit 1
}

# Run main function
main "$@"
