#!/bin/bash

# Database Adapter Migration Helper Script
# Usage: ./apply_migrations.sh [up|down|status]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default PostgreSQL connection settings
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGDATABASE="${PGDATABASE:-omninode_bridge}"
PGUSER="${PGUSER:-postgres}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Function to check PostgreSQL connection
check_connection() {
    print_info "Checking PostgreSQL connection..."
    if ! psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c '\q' 2>/dev/null; then
        print_error "Failed to connect to PostgreSQL"
        print_info "Connection details: $PGUSER@$PGHOST:$PGPORT/$PGDATABASE"
        exit 1
    fi
    print_success "Connected to PostgreSQL"
}

# Function to apply UP migrations
apply_up_migrations() {
    print_info "Applying UP migrations..."

    # Migration order: 1 -> 6
    for i in {1..6}; do
        migration_file=$(ls -1 "$SCRIPT_DIR" | grep "^00${i}_create_" | head -1)

        if [ -z "$migration_file" ]; then
            print_error "Migration file 00${i}_create_*.sql not found"
            exit 1
        fi

        print_info "Applying migration: $migration_file"
        if psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -f "$SCRIPT_DIR/$migration_file" > /dev/null 2>&1; then
            print_success "Applied: $migration_file"
        else
            print_error "Failed to apply: $migration_file"
            exit 1
        fi
    done

    print_success "All UP migrations applied successfully"
}

# Function to apply DOWN migrations
apply_down_migrations() {
    print_info "Applying DOWN migrations (rollback)..."

    # Rollback order: 6 -> 1 (reverse)
    for i in {6..1}; do
        migration_file=$(ls -1 "$SCRIPT_DIR" | grep "^00${i}_drop_" | head -1)

        if [ -z "$migration_file" ]; then
            print_error "Migration file 00${i}_drop_*.sql not found"
            exit 1
        fi

        print_info "Applying rollback: $migration_file"
        if psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -f "$SCRIPT_DIR/$migration_file" > /dev/null 2>&1; then
            print_success "Applied rollback: $migration_file"
        else
            print_error "Failed to apply rollback: $migration_file"
            exit 1
        fi
    done

    print_success "All DOWN migrations applied successfully"
}

# Function to show migration status
show_status() {
    print_info "Checking migration status..."
    echo ""

    # List of expected tables
    tables=("workflow_executions" "workflow_steps" "fsm_transitions" "bridge_states" "node_registrations" "metadata_stamps")

    for table in "${tables[@]}"; do
        if psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "\dt $table" 2>/dev/null | grep -q "$table"; then
            print_success "Table exists: $table"
        else
            print_error "Table missing: $table"
        fi
    done

    echo ""
    print_info "Total tables in database:"
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "\dt" 2>/dev/null | grep "public |" || echo "No tables found"
}

# Main script logic
main() {
    local command="${1:-help}"

    case "$command" in
        up)
            check_connection
            apply_up_migrations
            show_status
            ;;
        down)
            check_connection
            apply_down_migrations
            show_status
            ;;
        status)
            check_connection
            show_status
            ;;
        help|*)
            echo "Database Adapter Migration Helper"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  up      - Apply all UP migrations (create tables)"
            echo "  down    - Apply all DOWN migrations (drop tables)"
            echo "  status  - Show migration status"
            echo "  help    - Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  PGHOST     - PostgreSQL host (default: localhost)"
            echo "  PGPORT     - PostgreSQL port (default: 5432)"
            echo "  PGDATABASE - Database name (default: omninode_bridge)"
            echo "  PGUSER     - Database user (default: postgres)"
            echo "  PGPASSWORD - Database password"
            echo ""
            echo "Examples:"
            echo "  $0 up                           # Apply migrations"
            echo "  $0 down                         # Rollback migrations"
            echo "  PGDATABASE=test $0 status       # Check status in test database"
            ;;
    esac
}

# Run main function with all arguments
main "$@"
