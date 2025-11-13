#!/bin/bash
# Enhanced service health check script for CI/CD environments
# Ensures all required services are ready before running integration tests

set -euo pipefail

# Configuration
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_DATABASE="${POSTGRES_DATABASE:-omninode_bridge_test}"
KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"

MAX_WAIT_TIME=300  # 5 minutes maximum wait
HEALTH_CHECK_INTERVAL=2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to wait with timeout
wait_with_timeout() {
    local timeout=$1
    local check_command=$2
    local service_name=$3

    log_info "Waiting for $service_name to be ready (timeout: ${timeout}s)..."

    local count=0
    while [ $count -lt $timeout ]; do
        if eval "$check_command" >/dev/null 2>&1; then
            log_info "$service_name is ready!"
            return 0
        fi

        if [ $((count % 10)) -eq 0 ]; then
            log_info "Still waiting for $service_name... (${count}s elapsed)"
        fi

        sleep $HEALTH_CHECK_INTERVAL
        count=$((count + HEALTH_CHECK_INTERVAL))
    done

    log_error "$service_name failed to become ready within ${timeout}s"
    return 1
}

# PostgreSQL health check
check_postgres() {
    log_info "=== PostgreSQL Health Check ==="

    # Check if PostgreSQL is accepting connections
    if ! wait_with_timeout 60 "pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER" "PostgreSQL connection"; then
        return 1
    fi

    # Check if we can actually connect and query
    log_info "Testing PostgreSQL connectivity..."
    if ! PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres -c "SELECT 1;" >/dev/null 2>&1; then
        log_error "Cannot connect to PostgreSQL database"
        return 1
    fi

    # Check if test database exists, create if not
    log_info "Ensuring test database exists..."
    if ! PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$POSTGRES_DATABASE'" | grep -q 1; then
        log_info "Creating test database: $POSTGRES_DATABASE"
        PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE $POSTGRES_DATABASE;"
    fi

    # Test database connection
    if ! PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -c "SELECT 1;" >/dev/null 2>&1; then
        log_error "Cannot connect to test database: $POSTGRES_DATABASE"
        return 1
    fi

    log_info "PostgreSQL is fully ready and accessible"
    return 0
}

# Kafka health check
check_kafka() {
    log_info "=== Kafka/RedPanda Health Check ==="

    # Install kafkacat/kcat if not available (for testing Kafka connectivity)
    if ! command -v kcat >/dev/null 2>&1 && ! command -v kafkacat >/dev/null 2>&1; then
        log_warn "kcat/kafkacat not available, installing..."
        apt-get update >/dev/null 2>&1 && apt-get install -y kafkacat >/dev/null 2>&1 || true
    fi

    # Set kafkacat command (try both names)
    KAFKACAT_CMD=""
    if command -v kcat >/dev/null 2>&1; then
        KAFKACAT_CMD="kcat"
    elif command -v kafkacat >/dev/null 2>&1; then
        KAFKACAT_CMD="kafkacat"
    fi

    # Test Kafka broker connectivity with basic check
    log_info "Testing Kafka broker connectivity..."

    # Check if Kafka is listening on the port
    if ! wait_with_timeout 60 "nc -z ${KAFKA_BOOTSTRAP_SERVERS//:/ }" "Kafka port"; then
        return 1
    fi

    # Additional wait for Kafka to be fully initialized
    log_info "Waiting for Kafka to be fully initialized..."
    sleep 15

    # Test broker metadata retrieval (if kafkacat is available)
    if [ -n "$KAFKACAT_CMD" ]; then
        log_info "Testing Kafka broker metadata retrieval..."
        if ! timeout 30 $KAFKACAT_CMD -b "$KAFKA_BOOTSTRAP_SERVERS" -L >/dev/null 2>&1; then
            log_warn "Kafka metadata retrieval failed, but continuing..."
        else
            log_info "Kafka broker metadata retrieved successfully"
        fi

        # Test topic creation ability
        log_info "Testing topic creation capability..."
        test_topic="test-health-check-$(date +%s)"
        if timeout 20 $KAFKACAT_CMD -b "$KAFKA_BOOTSTRAP_SERVERS" -t "$test_topic" -P -o0 <<<'test' >/dev/null 2>&1; then
            log_info "Kafka topic creation test successful"
        else
            log_warn "Kafka topic creation test failed, but continuing..."
        fi
    else
        log_warn "kafkacat not available, skipping advanced Kafka checks"
    fi

    log_info "Kafka is ready (basic connectivity confirmed)"
    return 0
}

# Database migration check
check_migrations() {
    log_info "=== Database Migration Check ==="

    # Set environment variables for alembic
    export POSTGRES_HOST POSTGRES_PORT POSTGRES_DATABASE POSTGRES_USER POSTGRES_PASSWORD

    log_info "Running database migrations..."
    if ! poetry run alembic upgrade head; then
        log_error "Database migrations failed"
        return 1
    fi

    log_info "Database migrations completed successfully"
    return 0
}

# Environment validation
validate_environment() {
    log_info "=== Environment Validation ==="

    # Check required environment variables
    required_vars=("POSTGRES_HOST" "POSTGRES_PORT" "POSTGRES_USER" "POSTGRES_PASSWORD" "POSTGRES_DATABASE" "KAFKA_BOOTSTRAP_SERVERS")

    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log_error "Required environment variable $var is not set"
            return 1
        fi
    done

    # Check required tools
    required_tools=("pg_isready" "psql" "poetry")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_error "Required tool '$tool' is not available"
            return 1
        fi
    done

    log_info "Environment validation passed"
    return 0
}

# Service connectivity test
test_service_connectivity() {
    log_info "=== Service Connectivity Test ==="

    # Test PostgreSQL query performance
    log_info "Testing PostgreSQL query performance..."
    local pg_start_time=$(date +%s%N)
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -c "SELECT count(*) FROM information_schema.tables;" >/dev/null 2>&1
    local pg_end_time=$(date +%s%N)
    local pg_duration=$(( (pg_end_time - pg_start_time) / 1000000 ))
    log_info "PostgreSQL query took ${pg_duration}ms"

    if [ $pg_duration -gt 5000 ]; then
        log_warn "PostgreSQL queries are slow (${pg_duration}ms), tests may timeout"
    fi

    log_info "Service connectivity tests completed"
    return 0
}

# Main execution
main() {
    local start_time=$(date +%s)
    log_info "Starting comprehensive service health checks..."

    # Validate environment first
    if ! validate_environment; then
        log_error "Environment validation failed"
        exit 1
    fi

    # Check PostgreSQL
    if ! check_postgres; then
        log_error "PostgreSQL health check failed"
        exit 1
    fi

    # Check Kafka
    if ! check_kafka; then
        log_error "Kafka health check failed"
        exit 1
    fi

    # Run database migrations
    if ! check_migrations; then
        log_error "Database migration check failed"
        exit 1
    fi

    # Test service connectivity
    if ! test_service_connectivity; then
        log_error "Service connectivity test failed"
        exit 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_info "All services are ready and healthy! (Total time: ${duration}s)"
    log_info "Ready to run integration tests"

    return 0
}

# Execute main function
main "$@"
