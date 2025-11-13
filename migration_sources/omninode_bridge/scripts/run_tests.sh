#!/bin/bash

# OmniNode Bridge Test Runner Script
# Provides easy commands for running different test categories with coverage

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
COVERAGE_MIN=70
VERBOSE=false
PARALLEL=false
GENERATE_BADGE=false
OUTPUT_DIR="test-reports"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if poetry is available
check_poetry() {
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is not installed or not in PATH"
        print_info "Install Poetry: curl -sSL https://install.python-poetry.org | python3 -"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing dependencies with Poetry..."
    poetry install --no-interaction
}

# Function to run specific test category
run_tests() {
    local category="$1"
    local extra_args="${2:-}"

    print_info "Running $category tests..."

    case "$category" in
        "unit")
            poetry run pytest tests/unit/ \
                --cov=src/omninode_bridge \
                --cov-report=term-missing \
                --cov-report=html:htmlcov/unit \
                --cov-report=xml:coverage-unit.xml \
                --cov-fail-under=$COVERAGE_MIN \
                --cov-config=.coveragerc \
                --verbose \
                $extra_args
            ;;
        "integration")
            poetry run pytest tests/integration/ \
                --verbose \
                --timeout=300 \
                $extra_args
            ;;
        "security")
            poetry run pytest tests/test_security*.py \
                --verbose \
                --timeout=300 \
                $extra_args
            ;;
        "performance")
            poetry run pytest tests/performance/ \
                --verbose \
                --timeout=600 \
                $extra_args
            ;;
        "error")
            poetry run pytest tests/test_error_scenarios.py \
                --verbose \
                --timeout=300 \
                $extra_args
            ;;
        "all")
            poetry run pytest tests/ \
                --cov=src/omninode_bridge \
                --cov-report=term-missing \
                --cov-report=html:htmlcov/comprehensive \
                --cov-report=xml:coverage-comprehensive.xml \
                --cov-report=json:coverage-comprehensive.json \
                --cov-fail-under=$COVERAGE_MIN \
                --cov-config=.coveragerc \
                --verbose \
                $extra_args
            ;;
        *)
            print_error "Unknown test category: $category"
            print_info "Available categories: unit, integration, security, performance, error, all"
            exit 1
            ;;
    esac
}

# Function to generate coverage badge
generate_badge() {
    if [ "$GENERATE_BADGE" = true ]; then
        print_info "Generating coverage badge..."
        if poetry run coverage-badge -o coverage-badge.svg; then
            print_success "Coverage badge generated: coverage-badge.svg"
        else
            print_warning "Failed to generate coverage badge"
        fi
    fi
}

# Function to run linting and formatting checks
run_lint() {
    print_info "Running linting and formatting checks..."

    print_info "Checking code formatting with Black..."
    if poetry run black --check .; then
        print_success "Black formatting check passed"
    else
        print_error "Black formatting check failed"
        print_info "Run 'poetry run black .' to fix formatting issues"
        return 1
    fi

    print_info "Checking import sorting with isort..."
    if poetry run isort --check-only .; then
        print_success "isort check passed"
    else
        print_error "isort check failed"
        print_info "Run 'poetry run isort .' to fix import sorting"
        return 1
    fi

    print_info "Running Ruff linting..."
    if poetry run ruff check .; then
        print_success "Ruff linting passed"
    else
        print_error "Ruff linting failed"
        return 1
    fi

    print_info "Running MyPy type checking..."
    if poetry run mypy src/omninode_bridge; then
        print_success "MyPy type checking passed"
    else
        print_error "MyPy type checking failed"
        return 1
    fi
}

# Function to run comprehensive test suite
run_comprehensive() {
    print_info "Running comprehensive test suite..."

    # Run linting first
    if ! run_lint; then
        print_error "Linting checks failed. Fix issues before running tests."
        exit 1
    fi

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Run all test categories
    local categories=("unit" "integration" "security" "performance" "error")
    local failed_categories=()

    for category in "${categories[@]}"; do
        print_info "Running $category tests..."
        if run_tests "$category" "--tb=short"; then
            print_success "$category tests passed"
        else
            print_error "$category tests failed"
            failed_categories+=("$category")
        fi
        echo ""
    done

    # Run final comprehensive coverage
    print_info "Running final comprehensive coverage analysis..."
    if run_tests "all" "--tb=short"; then
        print_success "Comprehensive coverage analysis completed"
    else
        print_error "Comprehensive coverage analysis failed"
        failed_categories+=("comprehensive")
    fi

    # Generate badge
    generate_badge

    # Report results
    if [ ${#failed_categories[@]} -eq 0 ]; then
        print_success "All test categories passed! 🎉"
        print_info "Coverage reports generated in htmlcov/"
        return 0
    else
        print_error "Failed test categories: ${failed_categories[*]}"
        return 1
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
OmniNode Bridge Test Runner

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    unit                    Run unit tests with coverage
    integration            Run integration tests
    security               Run security tests
    performance            Run performance tests
    error                  Run error scenario tests
    all                    Run all tests with comprehensive coverage
    comprehensive          Run complete test suite with linting
    lint                   Run linting and formatting checks only
    help                   Show this help message

Options:
    --min-coverage N       Minimum coverage percentage (default: 70)
    --verbose              Enable verbose output
    --parallel             Enable parallel test execution
    --badge                Generate coverage badge
    --output-dir DIR       Output directory for reports (default: test-reports)

Examples:
    $0 unit                           # Run unit tests
    $0 all --badge                    # Run all tests and generate badge
    $0 comprehensive --min-coverage 80  # Run full suite with 80% coverage requirement
    $0 lint                           # Run only linting checks

Environment Variables:
    COVERAGE_MIN           Minimum coverage percentage
    PYTEST_ARGS           Additional arguments to pass to pytest

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --min-coverage)
            COVERAGE_MIN="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --badge)
            GENERATE_BADGE=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        help|--help|-h)
            show_usage
            exit 0
            ;;
        unit|integration|security|performance|error|all|comprehensive|lint)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set default command if none provided
if [ -z "${COMMAND:-}" ]; then
    COMMAND="help"
fi

# Main execution
print_info "OmniNode Bridge Test Runner"
print_info "Project root: $PROJECT_ROOT"
print_info "Coverage minimum: $COVERAGE_MIN%"
echo ""

# Check prerequisites
check_poetry

# Install dependencies
install_dependencies

# Execute command
case "$COMMAND" in
    "unit"|"integration"|"security"|"performance"|"error"|"all")
        extra_args=""
        if [ "$VERBOSE" = true ]; then
            extra_args="$extra_args -v"
        fi
        if [ -n "${PYTEST_ARGS:-}" ]; then
            extra_args="$extra_args $PYTEST_ARGS"
        fi

        if run_tests "$COMMAND" "$extra_args"; then
            generate_badge
            print_success "Tests completed successfully!"
            exit 0
        else
            print_error "Tests failed!"
            exit 1
        fi
        ;;
    "comprehensive")
        if run_comprehensive; then
            print_success "Comprehensive test suite completed successfully! 🎉"
            exit 0
        else
            print_error "Comprehensive test suite failed!"
            exit 1
        fi
        ;;
    "lint")
        if run_lint; then
            print_success "All linting checks passed! ✅"
            exit 0
        else
            print_error "Linting checks failed! ❌"
            exit 1
        fi
        ;;
    "help")
        show_usage
        exit 0
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
