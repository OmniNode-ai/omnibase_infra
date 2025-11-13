#!/bin/bash

################################################################################
# PostgreSQL Extension Setup Script for OmniNode Bridge
#
# Purpose:
#   Creates required PostgreSQL extensions (uuid-ossp, pg_stat_statements)
#   with proper error handling and verification.
#
# Required Extensions:
#   - uuid-ossp: UUID generation functions (REQUIRED)
#   - pg_stat_statements: Query performance tracking (RECOMMENDED)
#
# Prerequisites:
#   - PostgreSQL 15+ installed
#   - postgresql-contrib package installed
#   - Superuser database credentials
#   - Target database created
#
# Usage:
#   bash deployment/scripts/setup_postgres_extensions.sh
#
# Environment Variables:
#   POSTGRES_USER      - PostgreSQL superuser (default: postgres)
#   POSTGRES_DB        - Target database (default: omninode_bridge)
#   POSTGRES_HOST      - Database host (default: localhost)
#   POSTGRES_PORT      - Database port (default: 5432)
#   POSTGRES_PASSWORD  - Superuser password (optional, will prompt if not set)
#
# Exit Codes:
#   0 - Success
#   1 - Missing dependencies or prerequisites
#   2 - Database connection failure
#   3 - Extension creation failure
#
# Example:
#   export POSTGRES_HOST=db.example.com
#   export POSTGRES_PORT=5432
#   export POSTGRES_DB=omninode_bridge
#   export POSTGRES_USER=postgres
#   bash deployment/scripts/setup_postgres_extensions.sh
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

################################################################################
# Configuration
################################################################################

POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-omninode_bridge}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

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

print_separator() {
    echo "================================================================================"
}

check_psql_installed() {
    if ! command -v psql &> /dev/null; then
        log_error "psql command not found. Please install PostgreSQL client."
        log_info "Ubuntu/Debian: sudo apt-get install postgresql-client"
        log_info "RHEL/CentOS: sudo yum install postgresql"
        log_info "macOS: brew install postgresql"
        exit 1
    fi
}

check_database_exists() {
    log_info "Checking if database '$POSTGRES_DB' exists..."
    if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -lqt | cut -d \| -f 1 | grep -qw "$POSTGRES_DB"; then
        log_success "Database '$POSTGRES_DB' exists"
        return 0
    else
        log_error "Database '$POSTGRES_DB' does not exist"
        log_info "Create it with: createdb -U $POSTGRES_USER -h $POSTGRES_HOST -p $POSTGRES_PORT $POSTGRES_DB"
        exit 1
    fi
}

test_connection() {
    log_info "Testing database connection..."
    if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" > /dev/null 2>&1; then
        log_success "Database connection successful"
        return 0
    else
        log_error "Failed to connect to database"
        log_info "Host: $POSTGRES_HOST"
        log_info "Port: $POSTGRES_PORT"
        log_info "Database: $POSTGRES_DB"
        log_info "User: $POSTGRES_USER"
        exit 2
    fi
}

create_extension() {
    local extension_name=$1
    local required=$2

    log_info "Creating extension: $extension_name..."

    # Check if extension already exists
    local exists=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT COUNT(*) FROM pg_extension WHERE extname='$extension_name';")

    if [ "$exists" -eq 1 ]; then
        log_warning "Extension '$extension_name' already exists, skipping creation"
        return 0
    fi

    # Attempt to create extension
    if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS \"$extension_name\";" > /dev/null 2>&1; then
        log_success "Extension '$extension_name' created successfully"
        return 0
    else
        if [ "$required" = "true" ]; then
            log_error "Failed to create REQUIRED extension: $extension_name"
            log_info "This extension is required for OmniNode Bridge to function"
            exit 3
        else
            log_warning "Failed to create OPTIONAL extension: $extension_name"
            log_info "This extension is recommended but not required"
            return 1
        fi
    fi
}

verify_extension() {
    local extension_name=$1

    log_info "Verifying extension: $extension_name..."

    local result=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT extname, extversion FROM pg_extension WHERE extname='$extension_name';")

    if [ -n "$result" ]; then
        local version=$(echo "$result" | cut -d'|' -f2)
        log_success "Extension '$extension_name' verified (version: $version)"
        return 0
    else
        log_error "Extension '$extension_name' verification failed"
        return 1
    fi
}

test_uuid_generation() {
    log_info "Testing UUID generation function..."

    local uuid=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT uuid_generate_v4();")

    if [ -n "$uuid" ]; then
        log_success "UUID generation test passed: $uuid"
        return 0
    else
        log_error "UUID generation test failed"
        return 1
    fi
}

print_summary() {
    print_separator
    log_info "Extension Setup Summary"
    print_separator

    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" << EOF
SELECT
    extname AS "Extension Name",
    extversion AS "Version",
    CASE
        WHEN extname IN ('uuid-ossp', 'pg_stat_statements') THEN 'Required/Recommended'
        ELSE 'Optional'
    END AS "Status"
FROM pg_extension
WHERE extname IN ('uuid-ossp', 'pg_stat_statements')
ORDER BY extname;
EOF

    print_separator
}

################################################################################
# Main Execution
################################################################################

main() {
    print_separator
    log_info "OmniNode Bridge - PostgreSQL Extension Setup"
    print_separator

    log_info "Configuration:"
    log_info "  Host: $POSTGRES_HOST"
    log_info "  Port: $POSTGRES_PORT"
    log_info "  Database: $POSTGRES_DB"
    log_info "  User: $POSTGRES_USER"
    print_separator

    # Pre-flight checks
    check_psql_installed
    check_database_exists
    test_connection

    print_separator

    # Create required extension: uuid-ossp
    log_info "Step 1: Creating REQUIRED extension (uuid-ossp)..."
    create_extension "uuid-ossp" "true"
    verify_extension "uuid-ossp"

    print_separator

    # Create recommended extension: pg_stat_statements
    log_info "Step 2: Creating RECOMMENDED extension (pg_stat_statements)..."
    create_extension "pg_stat_statements" "false"

    if [ $? -eq 0 ]; then
        verify_extension "pg_stat_statements"
    else
        log_warning "Skipping pg_stat_statements verification (extension creation failed)"
    fi

    print_separator

    # Test UUID generation
    test_uuid_generation

    print_separator

    # Print summary
    print_summary

    log_success "PostgreSQL extension setup completed successfully!"
    print_separator

    log_info "Next steps:"
    log_info "  1. Run database migrations: bash migrations/run_migrations.sh"
    log_info "  2. Verify migration success: psql -U $POSTGRES_USER -d $POSTGRES_DB -c '\\dt'"
    log_info "  3. Start application: docker compose up -d"

    print_separator

    exit 0
}

################################################################################
# Execute Main Function
################################################################################

main "$@"
