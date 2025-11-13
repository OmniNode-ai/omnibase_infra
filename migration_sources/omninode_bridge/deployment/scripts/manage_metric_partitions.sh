#!/bin/bash

################################################################################
# Metric Partitions Management Script for OmniNode Bridge
#
# Purpose:
#   Automates creation, maintenance, and expiration of monthly partitions for
#   agent metrics tables with proper error handling and verification.
#
# Managed Partitioned Tables:
#   - agent_routing_metrics
#   - agent_state_metrics
#   - agent_coordination_metrics
#   - agent_workflow_metrics
#   - agent_quorum_metrics
#
# Features:
#   - Create future partitions (default: 3 months ahead)
#   - Drop expired partitions (default: 90 days retention)
#   - Verify existing partitions
#   - List all partitions with statistics
#   - Dry-run mode for testing
#
# Usage:
#   bash deployment/scripts/manage_metric_partitions.sh [COMMAND] [OPTIONS]
#
# Commands:
#   create     - Create future partitions (default: 3 months ahead)
#   drop       - Drop expired partitions (default: >90 days old)
#   verify     - Verify existing partitions and report status
#   list       - List all partitions with row counts and sizes
#   stats      - Show partition statistics and health metrics
#   help       - Show this help message
#
# Options:
#   --months=N         - Number of months ahead to create (default: 3)
#   --retention=N      - Retention period in days (default: 90)
#   --dry-run          - Show what would be done without making changes
#   --verbose          - Enable verbose logging
#
# Environment Variables:
#   POSTGRES_HOST      - Database host (default: localhost)
#   POSTGRES_PORT      - Database port (default: 5432)
#   POSTGRES_DB        - Target database (default: omninode_bridge)
#   POSTGRES_USER      - Database user (default: postgres)
#   POSTGRES_PASSWORD  - Database password (optional, will prompt if not set)
#
# Exit Codes:
#   0 - Success
#   1 - Invalid command or options
#   2 - Database connection failure
#   3 - Partition operation failure
#
# Examples:
#   # Create partitions for next 3 months
#   bash deployment/scripts/manage_metric_partitions.sh create
#
#   # Create partitions for next 6 months
#   bash deployment/scripts/manage_metric_partitions.sh create --months=6
#
#   # Drop partitions older than 90 days (dry-run)
#   bash deployment/scripts/manage_metric_partitions.sh drop --dry-run
#
#   # Drop partitions older than 60 days
#   bash deployment/scripts/manage_metric_partitions.sh drop --retention=60
#
#   # Verify all partitions
#   bash deployment/scripts/manage_metric_partitions.sh verify
#
#   # List all partitions with statistics
#   bash deployment/scripts/manage_metric_partitions.sh list
#
# Cron Setup (recommended):
#   # Run monthly on 1st at 2 AM to create partitions and clean old ones
#   0 2 1 * * /path/to/manage_metric_partitions.sh create && /path/to/manage_metric_partitions.sh drop
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

# Default settings
DEFAULT_MONTHS_AHEAD=3
DEFAULT_RETENTION_DAYS=90
DRY_RUN=false
VERBOSE=false

# Partitioned tables
TABLES=(
    "agent_routing_metrics"
    "agent_state_metrics"
    "agent_coordination_metrics"
    "agent_workflow_metrics"
    "agent_quorum_metrics"
)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

log_debug() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}[DEBUG]${NC} $1"
    fi
}

print_separator() {
    echo "================================================================================"
}

show_help() {
    cat << EOF
Usage: $(basename "$0") [COMMAND] [OPTIONS]

Commands:
  create     - Create future partitions (default: 3 months ahead)
  drop       - Drop expired partitions (default: >90 days old)
  verify     - Verify existing partitions and report status
  list       - List all partitions with row counts and sizes
  stats      - Show partition statistics and health metrics
  help       - Show this help message

Options:
  --months=N         - Number of months ahead to create (default: 3)
  --retention=N      - Retention period in days (default: 90)
  --dry-run          - Show what would be done without making changes
  --verbose          - Enable verbose logging

Environment Variables:
  POSTGRES_HOST      - Database host (default: localhost)
  POSTGRES_PORT      - Database port (default: 5432)
  POSTGRES_DB        - Target database (default: omninode_bridge)
  POSTGRES_USER      - Database user (default: postgres)
  POSTGRES_PASSWORD  - Database password (optional)

Examples:
  # Create partitions for next 3 months
  $(basename "$0") create

  # Create partitions for next 6 months
  $(basename "$0") create --months=6

  # Drop partitions older than 90 days (dry-run)
  $(basename "$0") drop --dry-run

  # Verify all partitions
  $(basename "$0") verify

  # List all partitions with statistics
  $(basename "$0") list

Cron Setup:
  # Run monthly on 1st at 2 AM
  0 2 1 * * /path/to/manage_metric_partitions.sh create && /path/to/manage_metric_partitions.sh drop

EOF
    exit 0
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

test_connection() {
    log_debug "Testing database connection..."
    if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" > /dev/null 2>&1; then
        log_debug "Database connection successful"
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

################################################################################
# Partition Operations
################################################################################

create_partitions() {
    local months_ahead=${1:-$DEFAULT_MONTHS_AHEAD}

    log_info "Creating partitions for next $months_ahead months..."
    print_separator

    local created_count=0
    local skipped_count=0

    for table in "${TABLES[@]}"; do
        log_info "Processing table: $table"

        for i in $(seq 0 $((months_ahead - 1))); do
            # Calculate partition date (first day of month)
            local partition_start=$(date -u -d "now + $i months" +%Y-%m-01)
            local partition_year=$(date -u -d "$partition_start" +%Y)
            local partition_month=$(date -u -d "$partition_start" +%m)
            local partition_name="${table}_${partition_year}_${partition_month}"

            # Calculate partition end date (first day of next month)
            local partition_end=$(date -u -d "$partition_start + 1 month" +%Y-%m-01)

            log_debug "  Checking partition: $partition_name"
            log_debug "    Range: [$partition_start, $partition_end)"

            # Check if partition already exists
            local exists=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
                "SELECT COUNT(*) FROM pg_tables WHERE tablename='$partition_name';")

            if [ "$exists" -eq 1 ]; then
                log_debug "  ✓ Partition $partition_name already exists"
                ((skipped_count++))
            else
                if [ "$DRY_RUN" = true ]; then
                    log_info "  [DRY-RUN] Would create partition: $partition_name"
                    log_debug "    SQL: CREATE TABLE $partition_name PARTITION OF $table FOR VALUES FROM ('$partition_start') TO ('$partition_end');"
                else
                    log_info "  Creating partition: $partition_name"

                    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
                        "CREATE TABLE IF NOT EXISTS $partition_name PARTITION OF $table FOR VALUES FROM ('$partition_start') TO ('$partition_end');" > /dev/null 2>&1

                    if [ $? -eq 0 ]; then
                        log_success "  ✓ Partition $partition_name created successfully"
                        ((created_count++))
                    else
                        log_error "  ✗ Failed to create partition: $partition_name"
                        exit 3
                    fi
                fi
            fi
        done
        echo ""
    done

    print_separator
    log_info "Summary:"
    log_success "  Created: $created_count partitions"
    log_info "  Skipped (already exist): $skipped_count partitions"
    print_separator
}

drop_expired_partitions() {
    local retention_days=${1:-$DEFAULT_RETENTION_DAYS}

    log_info "Dropping partitions older than $retention_days days..."
    print_separator

    local dropped_count=0
    local kept_count=0

    # Calculate cutoff date
    local cutoff_date=$(date -u -d "now - $retention_days days" +%Y-%m-01)
    log_info "Cutoff date: $cutoff_date (partitions before this will be dropped)"
    echo ""

    for table in "${TABLES[@]}"; do
        log_info "Processing table: $table"

        # Get all partitions for this table
        local partitions=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
            "SELECT tablename FROM pg_tables WHERE tablename LIKE '${table}_%' AND schemaname = 'public' ORDER BY tablename;")

        while IFS= read -r partition_name; do
            if [ -z "$partition_name" ]; then
                continue
            fi

            # Extract date from partition name (format: table_YYYY_MM)
            local partition_date=$(echo "$partition_name" | grep -oP '\d{4}_\d{2}$' | tr '_' '-')
            partition_date="${partition_date}-01"

            log_debug "  Checking partition: $partition_name (date: $partition_date)"

            # Compare dates
            if [[ "$partition_date" < "$cutoff_date" ]]; then
                if [ "$DRY_RUN" = true ]; then
                    log_warning "  [DRY-RUN] Would drop expired partition: $partition_name"
                else
                    log_warning "  Dropping expired partition: $partition_name"

                    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
                        "DROP TABLE IF EXISTS $partition_name;" > /dev/null 2>&1

                    if [ $? -eq 0 ]; then
                        log_success "  ✓ Partition $partition_name dropped successfully"
                        ((dropped_count++))
                    else
                        log_error "  ✗ Failed to drop partition: $partition_name"
                        exit 3
                    fi
                fi
            else
                log_debug "  ✓ Partition $partition_name is within retention period"
                ((kept_count++))
            fi
        done <<< "$partitions"

        echo ""
    done

    print_separator
    log_info "Summary:"
    if [ "$DRY_RUN" = true ]; then
        log_warning "  Would drop: $dropped_count partitions"
    else
        log_warning "  Dropped: $dropped_count partitions"
    fi
    log_info "  Kept: $kept_count partitions"
    print_separator
}

verify_partitions() {
    log_info "Verifying partition health..."
    print_separator

    local total_partitions=0
    local healthy_partitions=0
    local issues_found=0

    for table in "${TABLES[@]}"; do
        log_info "Table: $table"

        # Get partition count
        local partition_count=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
            "SELECT COUNT(*) FROM pg_tables WHERE tablename LIKE '${table}_%' AND schemaname = 'public';")

        ((total_partitions += partition_count))

        log_info "  Total partitions: $partition_count"

        # Check for gaps in partition coverage
        local partitions=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
            "SELECT tablename FROM pg_tables WHERE tablename LIKE '${table}_%' AND schemaname = 'public' ORDER BY tablename;")

        local prev_date=""
        local gap_found=false

        while IFS= read -r partition_name; do
            if [ -z "$partition_name" ]; then
                continue
            fi

            local partition_date=$(echo "$partition_name" | grep -oP '\d{4}_\d{2}$' | tr '_' '-')
            partition_date="${partition_date}-01"

            if [ -n "$prev_date" ]; then
                local expected_date=$(date -u -d "$prev_date + 1 month" +%Y-%m-01)
                if [[ "$partition_date" != "$expected_date" ]]; then
                    log_warning "  ⚠ Gap detected between $prev_date and $partition_date"
                    gap_found=true
                    ((issues_found++))
                fi
            fi

            prev_date=$partition_date
            ((healthy_partitions++))
        done <<< "$partitions"

        if [ "$gap_found" = false ]; then
            log_success "  ✓ No gaps detected in partition coverage"
        fi

        echo ""
    done

    print_separator
    log_info "Verification Summary:"
    log_info "  Total partitions: $total_partitions"
    log_success "  Healthy: $healthy_partitions"
    if [ $issues_found -gt 0 ]; then
        log_warning "  Issues found: $issues_found"
    else
        log_success "  Issues found: 0"
    fi
    print_separator
}

list_partitions() {
    log_info "Listing all partitions with statistics..."
    print_separator

    for table in "${TABLES[@]}"; do
        log_info "Table: $table"
        echo ""

        psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" << EOF
SELECT
    c.relname AS partition_name,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS partition_size,
    COALESCE(s.n_live_tup, 0) AS row_count,
    pg_size_pretty(pg_indexes_size(c.oid)) AS index_size
FROM pg_class c
LEFT JOIN pg_stat_user_tables s ON s.relname = c.relname
WHERE c.relname LIKE '${table}_%'
    AND c.relkind = 'r'
ORDER BY c.relname;
EOF

        echo ""
    done

    print_separator
}

show_stats() {
    log_info "Partition Statistics and Health Metrics"
    print_separator

    # Overall statistics
    local total_partitions=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
        "SELECT COUNT(*) FROM pg_tables WHERE tablename LIKE 'agent_%_metrics_%' AND schemaname = 'public';")

    local total_size=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
        "SELECT pg_size_pretty(SUM(pg_total_relation_size(c.oid)))
         FROM pg_class c
         WHERE c.relname LIKE 'agent_%_metrics_%' AND c.relkind = 'r';")

    local total_rows=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
        "SELECT SUM(COALESCE(s.n_live_tup, 0))
         FROM pg_class c
         LEFT JOIN pg_stat_user_tables s ON s.relname = c.relname
         WHERE c.relname LIKE 'agent_%_metrics_%' AND c.relkind = 'r';")

    log_info "Overall Statistics:"
    log_info "  Total partitions: $total_partitions"
    log_info "  Total size: $total_size"
    log_info "  Total rows: $total_rows"
    echo ""

    # Per-table statistics
    log_info "Per-Table Statistics:"
    echo ""

    for table in "${TABLES[@]}"; do
        local table_partitions=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
            "SELECT COUNT(*) FROM pg_tables WHERE tablename LIKE '${table}_%' AND schemaname = 'public';")

        local table_size=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
            "SELECT pg_size_pretty(SUM(pg_total_relation_size(c.oid)))
             FROM pg_class c
             WHERE c.relname LIKE '${table}_%' AND c.relkind = 'r';")

        local table_rows=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
            "SELECT SUM(COALESCE(s.n_live_tup, 0))
             FROM pg_class c
             LEFT JOIN pg_stat_user_tables s ON s.relname = c.relname
             WHERE c.relname LIKE '${table}_%' AND c.relkind = 'r';")

        log_info "  $table:"
        log_info "    Partitions: $table_partitions"
        log_info "    Size: $table_size"
        log_info "    Rows: $table_rows"
        echo ""
    done

    # Coverage analysis
    log_info "Coverage Analysis:"

    local oldest_partition=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
        "SELECT MIN(tablename) FROM pg_tables WHERE tablename LIKE 'agent_%_metrics_%' AND schemaname = 'public';")

    local newest_partition=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
        "SELECT MAX(tablename) FROM pg_tables WHERE tablename LIKE 'agent_%_metrics_%' AND schemaname = 'public';")

    log_info "  Oldest partition: $oldest_partition"
    log_info "  Newest partition: $newest_partition"

    print_separator
}

################################################################################
# Main Execution
################################################################################

main() {
    # Parse command
    COMMAND="${1:-help}"
    shift || true

    # Parse options
    MONTHS_AHEAD=$DEFAULT_MONTHS_AHEAD
    RETENTION_DAYS=$DEFAULT_RETENTION_DAYS

    while [[ $# -gt 0 ]]; do
        case $1 in
            --months=*)
                MONTHS_AHEAD="${1#*=}"
                shift
                ;;
            --retention=*)
                RETENTION_DAYS="${1#*=}"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                ;;
        esac
    done

    # Show configuration
    if [ "$VERBOSE" = true ]; then
        print_separator
        log_info "Configuration:"
        log_info "  Host: $POSTGRES_HOST"
        log_info "  Port: $POSTGRES_PORT"
        log_info "  Database: $POSTGRES_DB"
        log_info "  User: $POSTGRES_USER"
        log_info "  Dry-run: $DRY_RUN"
        print_separator
    fi

    # Show help without pre-flight checks
    if [[ "$COMMAND" == "help" || "$COMMAND" == "--help" || "$COMMAND" == "-h" ]]; then
        show_help
    fi

    # Pre-flight checks for all other commands
    check_psql_installed
    test_connection

    # Execute command
    case $COMMAND in
        create)
            create_partitions "$MONTHS_AHEAD"
            ;;
        drop)
            drop_expired_partitions "$RETENTION_DAYS"
            ;;
        verify)
            verify_partitions
            ;;
        list)
            list_partitions
            ;;
        stats)
            show_stats
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            echo ""
            show_help
            ;;
    esac

    exit 0
}

################################################################################
# Execute Main Function
################################################################################

main "$@"
