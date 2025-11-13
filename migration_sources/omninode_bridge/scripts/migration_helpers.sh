#!/bin/bash
# Database Migration Helper Scripts for OmniNode Bridge
# Provides convenient commands for common migration operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=${ENVIRONMENT:-development}
DATABASE_URL=${OMNINODE_BRIDGE_DATABASE_URL:-""}

# Helper functions
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

check_requirements() {
    log_info "Checking requirements..."

    # Check if we're in the right directory
    if [[ ! -f "alembic.ini" ]]; then
        log_error "alembic.ini not found. Please run from project root directory."
        exit 1
    fi

    # Check if Python dependencies are installed
    if ! python -c "import alembic" 2>/dev/null; then
        log_error "Alembic not installed. Run: poetry install"
        exit 1
    fi

    # Check if database URL is configured
    if [[ -z "$DATABASE_URL" ]]; then
        log_warning "DATABASE_URL not set. Using individual environment variables."
    fi

    log_success "Requirements check passed"
}

show_current_status() {
    log_info "Current migration status:"
    echo "Environment: $ENVIRONMENT"
    echo "Database URL: ${DATABASE_URL:0:50}..."
    echo ""

    log_info "Current revision:"
    alembic current --verbose
    echo ""

    log_info "Pending migrations:"
    alembic show head
}

create_migration() {
    local message="$1"
    if [[ -z "$message" ]]; then
        log_error "Migration message required"
        echo "Usage: create_migration \"Description of changes\""
        exit 1
    fi

    log_info "Creating new migration: $message"
    alembic revision --autogenerate -m "$message"
    log_success "Migration created successfully"
}

validate_and_migrate() {
    local target="${1:-head}"
    local dry_run="${2:-false}"

    log_info "Starting validation and migration process..."

    # Validate current setup
    check_requirements

    # Show current status
    show_current_status

    # Validate migration scripts
    log_info "Validating migration scripts..."
    if ! alembic check; then
        log_error "Migration validation failed"
        exit 1
    fi
    log_success "Migration scripts validated"

    # Create backup in production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Creating production backup..."
        python scripts/db_migrate.py backup
        if [[ $? -ne 0 ]]; then
            log_error "Backup failed - aborting migration"
            exit 1
        fi
        log_success "Backup created"
    fi

    # Apply migrations
    if [[ "$dry_run" == "true" ]]; then
        log_info "Dry run - showing SQL that would be executed:"
        python scripts/db_migrate.py migrate --target "$target" --dry-run
    else
        log_info "Applying migrations to $target..."
        python scripts/db_migrate.py migrate --target "$target"
        if [[ $? -eq 0 ]]; then
            log_success "Migrations applied successfully"
        else
            log_error "Migration failed"
            exit 1
        fi
    fi
}

rollback_migration() {
    local target="$1"
    if [[ -z "$target" ]]; then
        log_error "Target revision required for rollback"
        echo "Usage: rollback_migration <revision_id>"
        exit 1
    fi

    log_warning "Rolling back to revision: $target"

    # Extra confirmation for production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo -n "This is a PRODUCTION rollback. Are you sure? (yes/no): "
        read confirmation
        if [[ "$confirmation" != "yes" ]]; then
            log_info "Rollback cancelled"
            exit 0
        fi
    fi

    python scripts/db_migrate.py rollback --target "$target"
    if [[ $? -eq 0 ]]; then
        log_success "Rollback completed successfully"
    else
        log_error "Rollback failed"
        exit 1
    fi
}

emergency_restore() {
    local backup_path="$1"
    if [[ -z "$backup_path" ]]; then
        log_error "Backup path required for emergency restore"
        echo "Usage: emergency_restore <backup_file_path>"
        exit 1
    fi

    log_warning "EMERGENCY RESTORE from: $backup_path"

    # Multiple confirmations for production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo -n "This will REPLACE ALL DATA in production database. Type 'RESTORE' to confirm: "
        read confirmation
        if [[ "$confirmation" != "RESTORE" ]]; then
            log_info "Emergency restore cancelled"
            exit 0
        fi
    fi

    python scripts/db_migrate.py restore --backup-path "$backup_path"
    if [[ $? -eq 0 ]]; then
        log_success "Emergency restore completed"
    else
        log_error "Emergency restore failed"
        exit 1
    fi
}

show_migration_history() {
    log_info "Migration history:"
    alembic history --verbose
}

# Main command processing
case "$1" in
    "status")
        show_current_status
        ;;
    "create")
        create_migration "$2"
        ;;
    "migrate")
        validate_and_migrate "$2" "$3"
        ;;
    "rollback")
        rollback_migration "$2"
        ;;
    "restore")
        emergency_restore "$2"
        ;;
    "history")
        show_migration_history
        ;;
    "validate")
        check_requirements
        alembic check
        log_success "Validation completed"
        ;;
    *)
        echo "OmniNode Bridge Migration Helper"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  status              Show current migration status"
        echo "  create <message>    Create new migration with autogenerate"
        echo "  migrate [target]    Validate and apply migrations (default: head)"
        echo "  rollback <revision> Rollback to specific revision"
        echo "  restore <backup>    Emergency restore from backup"
        echo "  history             Show migration history"
        echo "  validate            Validate migration scripts"
        echo ""
        echo "Examples:"
        echo "  $0 status"
        echo "  $0 create \"Add user authentication table\""
        echo "  $0 migrate"
        echo "  $0 migrate 001  # Migrate to specific revision"
        echo "  $0 rollback 001"
        echo "  $0 restore backups/database/backup_20250101_120000.sql"
        echo ""
        echo "Environment Variables:"
        echo "  ENVIRONMENT                    - deployment environment (development/staging/production)"
        echo "  OMNINODE_BRIDGE_DATABASE_URL   - full database URL"
        echo "  POSTGRES_*                     - individual database connection parameters"
        exit 1
        ;;
esac
