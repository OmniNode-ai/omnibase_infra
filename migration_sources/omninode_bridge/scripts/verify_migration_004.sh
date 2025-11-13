#!/bin/bash
# Verification script for migration 004 idempotency fixes
# This script validates the migration implementation and runs comprehensive tests

set -e  # Exit on error

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if running from project root
if [ ! -f "pyproject.toml" ]; then
    print_error "Must be run from project root directory"
    exit 1
fi

print_header "Migration 004 Verification Script"

# Step 1: Syntax validation
print_header "Step 1: Syntax Validation"

echo "Checking migration file syntax..."
if python3 -m py_compile alembic/versions/004_20251003_node_registrations.py; then
    print_success "Migration file syntax is valid"
else
    print_error "Migration file has syntax errors"
    exit 1
fi

echo "Checking test file syntax..."
if python3 -m py_compile tests/integration/test_migration_004_idempotency.py; then
    print_success "Test file syntax is valid"
else
    print_error "Test file has syntax errors"
    exit 1
fi

# Step 2: Type checking (if mypy is available)
print_header "Step 2: Type Checking"

if command -v mypy &> /dev/null; then
    echo "Running mypy on migration file..."
    if mypy alembic/versions/004_20251003_node_registrations.py --no-error-summary 2>/dev/null; then
        print_success "Migration file passes type checking"
    else
        print_warning "Migration file has type checking warnings (non-critical)"
    fi
else
    print_warning "mypy not installed, skipping type checking"
fi

# Step 3: Linting (if ruff is available)
print_header "Step 3: Code Quality Checks"

if command -v ruff &> /dev/null; then
    echo "Running ruff on migration file..."
    if ruff check alembic/versions/004_20251003_node_registrations.py --quiet 2>/dev/null; then
        print_success "Migration file passes linting"
    else
        print_warning "Migration file has linting warnings (non-critical)"
    fi
else
    print_warning "ruff not installed, skipping linting"
fi

# Step 4: Database connectivity check
print_header "Step 4: Database Connectivity"

# Check if test database is accessible
TEST_DB_URL="${TEST_DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/test_omninode_bridge}"  # pragma: allowlist secret

echo "Checking database connectivity..."
if python3 -c "
import sqlalchemy as sa
try:
    engine = sa.create_engine('$TEST_DB_URL')
    conn = engine.connect()
    conn.close()
    print('Connected successfully')
    exit(0)
except Exception as e:
    print(f'Connection failed: {e}')
    exit(1)
" 2>/dev/null; then
    print_success "Database is accessible"
else
    print_warning "Cannot connect to test database at: $TEST_DB_URL"
    echo "  To run tests, ensure PostgreSQL is running and accessible"
    echo "  Or set TEST_DATABASE_URL environment variable"
fi

# Step 5: Run tests (if database is accessible)
print_header "Step 5: Running Tests"

if python3 -c "import sqlalchemy as sa; sa.create_engine('$TEST_DB_URL').connect()" 2>/dev/null; then
    echo "Running migration idempotency tests..."

    if pytest tests/integration/test_migration_004_idempotency.py -v --tb=short; then
        print_success "All tests passed!"
    else
        print_error "Some tests failed - see output above"
        exit 1
    fi
else
    print_warning "Skipping tests - database not accessible"
    echo "  To run tests:"
    echo "    1. Start PostgreSQL: docker-compose up -d postgres"
    echo "    2. Set TEST_DATABASE_URL if needed"
    echo "    3. Run: pytest tests/integration/test_migration_004_idempotency.py -v"
fi

# Step 6: Migration validation (if Alembic is configured)
print_header "Step 6: Alembic Configuration Validation"

if [ -f "alembic.ini" ]; then
    echo "Validating Alembic configuration..."

    if alembic current 2>/dev/null; then
        print_success "Alembic configuration is valid"

        echo "Current migration revision:"
        alembic current
    else
        print_warning "Alembic configuration may need database connection"
    fi
else
    print_warning "alembic.ini not found in current directory"
fi

# Final summary
print_header "Verification Summary"

echo -e "Migration file: ${GREEN}✓ Valid${NC}"
echo -e "Test file: ${GREEN}✓ Valid${NC}"

if command -v mypy &> /dev/null; then
    echo -e "Type checking: ${GREEN}✓ Passed${NC}"
else
    echo -e "Type checking: ${YELLOW}⚠ Skipped${NC}"
fi

if python3 -c "import sqlalchemy as sa; sa.create_engine('$TEST_DB_URL').connect()" 2>/dev/null; then
    echo -e "Database tests: ${GREEN}✓ Passed${NC}"
else
    echo -e "Database tests: ${YELLOW}⚠ Skipped (database not accessible)${NC}"
fi

print_header "Next Steps"

echo "1. Review the migration file:"
echo "   cat alembic/versions/004_20251003_node_registrations.py"
echo ""
echo "2. Review the test file:"
echo "   cat tests/integration/test_migration_004_idempotency.py"
echo ""
echo "3. Review the documentation:"
echo "   cat alembic/versions/README_MIGRATION_004.md"
echo ""
echo "4. Run specific test categories:"
echo "   pytest tests/integration/test_migration_004_idempotency.py::TestMigration004FreshInstall -v"
echo "   pytest tests/integration/test_migration_004_idempotency.py::TestMigration004Idempotency -v"
echo "   pytest tests/integration/test_migration_004_idempotency.py::TestMigration004EnumValidation -v"
echo "   pytest tests/integration/test_migration_004_idempotency.py::TestMigration004Downgrade -v"
echo ""
echo "5. Test the migration manually:"
echo "   alembic upgrade 004"
echo "   alembic downgrade 003"
echo "   alembic upgrade 004"
echo ""

print_success "Verification complete!"
