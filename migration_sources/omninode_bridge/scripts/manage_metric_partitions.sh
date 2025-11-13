#!/bin/bash
# ================================================================
# Metric Partitions Management Script for OmniNode Bridge
# ================================================================
# Purpose: Automated management of time-based partitions for agent metrics tables
# Author: OmniNode Team
# Date: 2025-11-07
#
# This script manages partitions for the 5 partitioned metrics tables created
# in migration 014:
#   - agent_routing_metrics
#   - agent_state_metrics
#   - agent_coordination_metrics
#   - agent_workflow_metrics
#   - agent_quorum_metrics
#
# Usage:
#   ./manage_metric_partitions.sh <command> [options]
#
# Commands:
#   create-next           Create next month's partitions for all tables
#   create-future N       Create partitions for next N months
#   drop-old              Drop partitions older than retention policy
#   list                  List all current partitions
#   status                Show partition status and recommendations
#   --help                Show this help message
#
# Options:
#   --dry-run             Show SQL without executing (default: false)
#   --retention-months N  Retention period in months (default: 3)
#   --table NAME          Operate on specific table only (default: all)
#
# Examples:
#   # Create next month's partitions (dry run)
#   ./manage_metric_partitions.sh create-next --dry-run
#
#   # Create partitions for next 6 months
#   ./manage_metric_partitions.sh create-future 6
#
#   # Drop old partitions (keeping 3 months)
#   ./manage_metric_partitions.sh drop-old --retention-months 3
#
#   # List all partitions
#   ./manage_metric_partitions.sh list
#
#   # Check partition status
#   ./manage_metric_partitions.sh status
#
#   # Create partitions for specific table only
#   ./manage_metric_partitions.sh create-next --table agent_routing_metrics
#
# Environment Variables:
#   POSTGRES_HOST         PostgreSQL host (default: from .env)
#   POSTGRES_PORT         PostgreSQL port (default: from .env)
#   POSTGRES_USER         PostgreSQL user (default: from .env)
#   POSTGRES_DATABASE     PostgreSQL database (default: from .env)
#   POSTGRES_PASSWORD     PostgreSQL password (required)
#
# Recommended Cron Schedule:
#   # Run on 1st of each month at 2 AM
#   0 2 1 * * /path/to/manage_metric_partitions.sh create-next
#
#   # Check and drop old partitions weekly
#   0 3 * * 0 /path/to/manage_metric_partitions.sh drop-old --retention-months 3
#
# ================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
DRY_RUN=false
RETENTION_MONTHS=3
TARGET_TABLE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Partitioned tables from migration 014
PARTITIONED_TABLES=(
    "agent_routing_metrics"
    "agent_state_metrics"
    "agent_coordination_metrics"
    "agent_workflow_metrics"
    "agent_quorum_metrics"
)

# ================================================================
# Helper Functions
# ================================================================

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_debug() {
    echo -e "${CYAN}🔍 $1${NC}"
}

show_help() {
    # Extract and display the usage documentation from header comments
    sed -n '/^# Usage:/,/^# ================================================================/p' "$0" | sed 's/^# //g'
    exit 0
}

# ================================================================
# Environment Setup
# ================================================================

load_environment() {
    log_info "Loading environment configuration..."

    # Try to load from .env file
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        log_debug "Loading from $PROJECT_ROOT/.env"
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
    else
        log_warning ".env file not found at $PROJECT_ROOT/.env"
    fi

    # Check required environment variables
    if [[ -z "$POSTGRES_HOST" ]]; then
        log_error "POSTGRES_HOST not set"
        exit 1
    fi

    if [[ -z "$POSTGRES_PORT" ]]; then
        log_error "POSTGRES_PORT not set"
        exit 1
    fi

    if [[ -z "$POSTGRES_USER" ]]; then
        log_error "POSTGRES_USER not set"
        exit 1
    fi

    if [[ -z "$POSTGRES_DATABASE" ]]; then
        log_error "POSTGRES_DATABASE not set"
        exit 1
    fi

    if [[ -z "$POSTGRES_PASSWORD" ]]; then
        log_error "POSTGRES_PASSWORD not set"
        exit 1
    fi

    log_success "Environment loaded: ${POSTGRES_USER}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DATABASE}"
}

# ================================================================
# Database Functions
# ================================================================

execute_sql() {
    local sql="$1"
    local description="$2"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: $description"
        echo -e "${CYAN}$sql${NC}"
        echo ""
        return 0
    fi

    log_debug "Executing: $description"

    if PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DATABASE" \
        -v ON_ERROR_STOP=1 \
        -c "$sql" > /dev/null 2>&1; then
        log_success "$description"
        return 0
    else
        log_error "Failed: $description"
        return 1
    fi
}

query_sql() {
    local sql="$1"

    PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DATABASE" \
        -t \
        -A \
        -c "$sql" 2>/dev/null
}

# ================================================================
# Date Calculation Functions
# ================================================================

# Get first day of next month in YYYY-MM-DD format
get_next_month() {
    local months_ahead="${1:-1}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        date -v +${months_ahead}m -v 1d '+%Y-%m-01'
    else
        # Linux
        date -d "$(date +%Y-%m-01) +${months_ahead} month" '+%Y-%m-01'
    fi
}

# Get first day of month after base date
get_month_after() {
    local base_date="$1"
    local months_ahead="${2:-1}"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - need to ensure base_date is in correct format
        date -j -f "%Y-%m-%d" "$base_date" -v +${months_ahead}m '+%Y-%m-%d' 2>/dev/null || \
        date -v +${months_ahead}m -v 1d '+%Y-%m-01'
    else
        # Linux
        date -d "$base_date +${months_ahead} month" '+%Y-%m-%d'
    fi
}

# Format YYYY-MM-DD to YYYY_MM for partition naming
format_partition_name() {
    local date="$1"
    echo "$date" | sed 's/-/_/g' | cut -d_ -f1-2
}

# Get retention cutoff date
get_retention_cutoff() {
    local months="$1"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        date -v -${months}m -v 1d '+%Y-%m-01'
    else
        # Linux
        date -d "$(date +%Y-%m-01) -${months} month" '+%Y-%m-01'
    fi
}

# ================================================================
# Partition Management Functions
# ================================================================

get_target_tables() {
    if [[ -n "$TARGET_TABLE" ]]; then
        echo "$TARGET_TABLE"
    else
        printf '%s\n' "${PARTITIONED_TABLES[@]}"
    fi
}

partition_exists() {
    local table_name="$1"
    local partition_name="$2"

    local count=$(query_sql "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public' AND tablename = '${partition_name}';")
    [[ "$count" -gt 0 ]]
}

create_partition() {
    local table_name="$1"
    local start_date="$2"
    local end_date="$3"

    local partition_suffix=$(format_partition_name "$start_date")
    local partition_name="${table_name}_${partition_suffix}"

    # Check if partition already exists
    if partition_exists "$table_name" "$partition_name"; then
        log_warning "Partition $partition_name already exists, skipping"
        return 0
    fi

    local sql="CREATE TABLE IF NOT EXISTS ${partition_name} PARTITION OF ${table_name}
    FOR VALUES FROM ('${start_date}') TO ('${end_date}');"

    execute_sql "$sql" "Create partition: $partition_name ($start_date to $end_date)"
}

drop_partition() {
    local partition_name="$1"
    local reason="$2"

    local sql="DROP TABLE IF EXISTS ${partition_name};"
    execute_sql "$sql" "Drop partition: $partition_name ($reason)"
}

list_partitions() {
    local table_name="${1:-}"

    log_info "Listing partitions..."
    echo ""

    local where_clause=""
    if [[ -n "$table_name" ]]; then
        where_clause="WHERE parent.relname = '${table_name}'"
    else
        # Filter to only our managed tables
        local table_list=$(printf "'%s'," "${PARTITIONED_TABLES[@]}")
        table_list="${table_list%,}"  # Remove trailing comma
        where_clause="WHERE parent.relname IN (${table_list})"
    fi

    local sql="
    SELECT
        parent.relname as parent_table,
        child.relname as partition_name,
        pg_get_expr(child.relpartbound, child.oid) as partition_bounds
    FROM pg_inherits
    JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
    JOIN pg_class child ON pg_inherits.inhrelid = child.oid
    ${where_clause}
    ORDER BY parent.relname, child.relname;
    "

    PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DATABASE" \
        -c "$sql"

    echo ""
}

create_next_partitions() {
    log_info "Creating next month's partitions..."
    echo ""

    local start_date=$(get_next_month 1)
    local end_date=$(get_next_month 2)

    log_debug "Date range: $start_date to $end_date"
    echo ""

    local success_count=0
    local skip_count=0
    local fail_count=0

    while IFS= read -r table_name; do
        if create_partition "$table_name" "$start_date" "$end_date"; then
            if partition_exists "$table_name" "${table_name}_$(format_partition_name "$start_date")"; then
                ((success_count++))
            else
                ((skip_count++))
            fi
        else
            ((fail_count++))
        fi
    done < <(get_target_tables)

    echo ""
    log_info "Summary:"
    log_success "Created: $success_count partitions"
    if [[ $skip_count -gt 0 ]]; then
        log_warning "Skipped: $skip_count partitions (already exist)"
    fi
    if [[ $fail_count -gt 0 ]]; then
        log_error "Failed: $fail_count partitions"
        exit 1
    fi
}

create_future_partitions() {
    local months="$1"

    if [[ ! "$months" =~ ^[0-9]+$ ]] || [[ $months -lt 1 ]]; then
        log_error "Invalid month count: $months (must be positive integer)"
        exit 1
    fi

    log_info "Creating partitions for next $months months..."
    echo ""

    local total_success=0
    local total_skip=0
    local total_fail=0

    for ((i=1; i<=months; i++)); do
        local start_date=$(get_next_month $i)
        local end_date=$(get_next_month $((i+1)))

        log_info "Creating partitions for month $i: $start_date to $end_date"

        while IFS= read -r table_name; do
            if create_partition "$table_name" "$start_date" "$end_date"; then
                if partition_exists "$table_name" "${table_name}_$(format_partition_name "$start_date")"; then
                    ((total_success++))
                else
                    ((total_skip++))
                fi
            else
                ((total_fail++))
            fi
        done < <(get_target_tables)

        echo ""
    done

    log_info "Summary:"
    log_success "Created: $total_success partitions"
    if [[ $total_skip -gt 0 ]]; then
        log_warning "Skipped: $total_skip partitions (already exist)"
    fi
    if [[ $total_fail -gt 0 ]]; then
        log_error "Failed: $total_fail partitions"
        exit 1
    fi
}

drop_old_partitions() {
    log_info "Dropping partitions older than $RETENTION_MONTHS months..."
    echo ""

    local cutoff_date=$(get_retention_cutoff $RETENTION_MONTHS)
    log_debug "Retention cutoff date: $cutoff_date"
    echo ""

    local drop_count=0
    local keep_count=0

    while IFS= read -r table_name; do
        # Get all partitions for this table
        local sql="
        SELECT child.relname
        FROM pg_inherits
        JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
        JOIN pg_class child ON pg_inherits.inhrelid = child.oid
        WHERE parent.relname = '${table_name}'
        ORDER BY child.relname;
        "

        while IFS= read -r partition_name; do
            # Extract date from partition name (e.g., agent_routing_metrics_2025_11 -> 2025-11-01)
            if [[ $partition_name =~ ([0-9]{4})_([0-9]{2})$ ]]; then
                local partition_date="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}-01"

                # Compare dates (convert to seconds since epoch)
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    local partition_epoch=$(date -j -f "%Y-%m-%d" "$partition_date" +%s 2>/dev/null || echo 0)
                    local cutoff_epoch=$(date -j -f "%Y-%m-%d" "$cutoff_date" +%s 2>/dev/null || echo 0)
                else
                    local partition_epoch=$(date -d "$partition_date" +%s 2>/dev/null || echo 0)
                    local cutoff_epoch=$(date -d "$cutoff_date" +%s 2>/dev/null || echo 0)
                fi

                if [[ $partition_epoch -lt $cutoff_epoch ]]; then
                    drop_partition "$partition_name" "older than $cutoff_date"
                    ((drop_count++))
                else
                    log_debug "Keeping partition: $partition_name ($partition_date >= $cutoff_date)"
                    ((keep_count++))
                fi
            else
                log_warning "Skipping partition with unexpected name format: $partition_name"
            fi
        done < <(query_sql "$sql")
    done < <(get_target_tables)

    echo ""
    log_info "Summary:"
    log_success "Dropped: $drop_count partitions"
    log_info "Kept: $keep_count partitions"
}

show_status() {
    log_info "Partition Status Report"
    echo ""

    # Show current partitions
    log_info "Current Partitions:"
    list_partitions

    # Check for missing future partitions
    log_info "Future Partition Recommendations:"
    echo ""

    local next_month=$(get_next_month 1)
    local month_after=$(get_next_month 2)
    local next_partition_suffix=$(format_partition_name "$next_month")

    local missing_next=0
    local tables_checked=0

    while IFS= read -r table_name; do
        ((tables_checked++))
        local partition_name="${table_name}_${next_partition_suffix}"
        if ! partition_exists "$table_name" "$partition_name"; then
            if [[ $missing_next -eq 0 ]]; then
                log_warning "Missing partitions for next month ($next_month):"
            fi
            echo "  - $partition_name"
            ((missing_next++))
        fi
    done < <(get_target_tables)

    if [[ $missing_next -eq 0 ]]; then
        log_success "All tables have partitions for next month ($next_month)"
    else
        echo ""
        log_warning "Run './manage_metric_partitions.sh create-next' to create missing partitions"
    fi

    echo ""

    # Check for old partitions
    log_info "Retention Policy Check (${RETENTION_MONTHS} months):"
    echo ""

    local cutoff_date=$(get_retention_cutoff $RETENTION_MONTHS)
    local old_partitions=0

    while IFS= read -r table_name; do
        local sql="
        SELECT child.relname
        FROM pg_inherits
        JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
        JOIN pg_class child ON pg_inherits.inhrelid = child.oid
        WHERE parent.relname = '${table_name}'
        ORDER BY child.relname;
        "

        while IFS= read -r partition_name; do
            if [[ $partition_name =~ ([0-9]{4})_([0-9]{2})$ ]]; then
                local partition_date="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}-01"

                if [[ "$OSTYPE" == "darwin"* ]]; then
                    local partition_epoch=$(date -j -f "%Y-%m-%d" "$partition_date" +%s 2>/dev/null || echo 0)
                    local cutoff_epoch=$(date -j -f "%Y-%m-%d" "$cutoff_date" +%s 2>/dev/null || echo 0)
                else
                    local partition_epoch=$(date -d "$partition_date" +%s 2>/dev/null || echo 0)
                    local cutoff_epoch=$(date -d "$cutoff_date" +%s 2>/dev/null || echo 0)
                fi

                if [[ $partition_epoch -lt $cutoff_epoch ]]; then
                    if [[ $old_partitions -eq 0 ]]; then
                        log_warning "Partitions older than retention policy (before $cutoff_date):"
                    fi
                    echo "  - $partition_name ($partition_date)"
                    ((old_partitions++))
                fi
            fi
        done < <(query_sql "$sql")
    done < <(get_target_tables)

    if [[ $old_partitions -eq 0 ]]; then
        log_success "No partitions older than retention policy"
    else
        echo ""
        log_warning "Run './manage_metric_partitions.sh drop-old --retention-months $RETENTION_MONTHS' to remove old partitions"
    fi

    echo ""
}

# ================================================================
# Main Command Processing
# ================================================================

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        create-next|create-future|drop-old|list|status)
            COMMAND="$1"
            shift
            if [[ "$COMMAND" == "create-future" ]]; then
                MONTHS_AHEAD="$1"
                shift
            fi
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --retention-months)
            RETENTION_MONTHS="$2"
            shift 2
            ;;
        --table)
            TARGET_TABLE="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if no command provided
if [[ -z "$COMMAND" ]]; then
    show_help
fi

# Validate target table if specified
if [[ -n "$TARGET_TABLE" ]]; then
    TABLE_VALID=false
    for table in "${PARTITIONED_TABLES[@]}"; do
        if [[ "$table" == "$TARGET_TABLE" ]]; then
            TABLE_VALID=true
            break
        fi
    done

    if [[ "$TABLE_VALID" == "false" ]]; then
        log_error "Invalid table name: $TARGET_TABLE"
        echo "Valid tables: ${PARTITIONED_TABLES[*]}"
        exit 1
    fi
fi

# Load environment
load_environment

# Show dry-run mode if enabled
if [[ "$DRY_RUN" == "true" ]]; then
    log_warning "DRY RUN MODE - No changes will be made"
    echo ""
fi

# Execute command
case "$COMMAND" in
    create-next)
        create_next_partitions
        ;;
    create-future)
        if [[ -z "$MONTHS_AHEAD" ]]; then
            log_error "create-future requires number of months"
            echo "Usage: $0 create-future <months>"
            exit 1
        fi
        create_future_partitions "$MONTHS_AHEAD"
        ;;
    drop-old)
        drop_old_partitions
        ;;
    list)
        list_partitions "$TARGET_TABLE"
        ;;
    status)
        show_status
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        ;;
esac

log_success "Operation completed successfully"
