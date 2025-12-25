#!/usr/bin/env bash
# Architecture Invariant Verification Script
# OMN-255: Verify omnibase_core does not contain infrastructure dependencies
#
# This script checks that omnibase_core maintains proper layer separation
# by not importing infrastructure-specific packages like kafka, httpx, asyncpg.
#
# Usage:
#   ./scripts/check_architecture.sh [OPTIONS]
#
# Options:
#   --help, -h      Show this help message
#   --verbose, -v   Show detailed output
#   --path PATH     Specify custom omnibase_core path
#   --no-color      Disable colored output

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Forbidden imports that indicate layer violation
FORBIDDEN_IMPORTS=(
    "kafka"
    "httpx"
    "asyncpg"
    "aiohttp"
    "redis"
    "psycopg"
    "psycopg2"
    "consul"
    "hvac"
    "aiokafka"
    "confluent_kafka"
)

# File patterns to exclude from checking
EXCLUDE_PATTERNS=(
    "requirements*.txt"
    "pyproject.toml"
    "setup.py"
    "setup.cfg"
    "*.md"
    "*.rst"
    "*.json"
    "*.yaml"
    "*.yml"
    "Makefile"
    "*.lock"
)

# Directory patterns to exclude
EXCLUDE_DIRS=(
    ".git"
    "__pycache__"
    ".pytest_cache"
    ".mypy_cache"
    "*.egg-info"
    ".tox"
    ".venv"
    "venv"
    "node_modules"
)

# =============================================================================
# Color Output
# =============================================================================

# Default: enable colors if stdout is a TTY
USE_COLOR=true
if [[ ! -t 1 ]]; then
    USE_COLOR=false
fi

# Color codes
setup_colors() {
    if [[ "$USE_COLOR" == "true" ]]; then
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        YELLOW='\033[0;33m'
        BLUE='\033[0;34m'
        BOLD='\033[1m'
        NC='\033[0m'  # No Color
    else
        RED=''
        GREEN=''
        YELLOW=''
        BLUE=''
        BOLD=''
        NC=''
    fi
}

# =============================================================================
# Utility Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BOLD}===============================================${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BOLD}===============================================${NC}"
    echo ""
}

print_pass() {
    echo -e "  ${GREEN}[PASS]${NC} $1"
}

print_fail() {
    echo -e "  ${RED}[FAIL]${NC} $1"
}

print_info() {
    echo -e "  ${BLUE}[INFO]${NC} $1"
}

print_warn() {
    echo -e "  ${YELLOW}[WARN]${NC} $1"
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat << 'EOF'
Architecture Invariant Verification Script
OMN-255: Verify omnibase_core does not contain infrastructure dependencies

USAGE:
    ./scripts/check_architecture.sh [OPTIONS]

DESCRIPTION:
    This script verifies that omnibase_core maintains proper layer separation
    by checking for forbidden infrastructure imports. The core layer should
    not depend on infrastructure-specific packages.

OPTIONS:
    --help, -h      Show this help message and exit
    --verbose, -v   Show detailed output including files scanned
    --path PATH     Specify custom omnibase_core path (default: auto-detect)
    --no-color      Disable colored output

FORBIDDEN IMPORTS:
    - kafka           (Kafka client library - belongs in infra layer)
    - httpx           (HTTP client library - belongs in infra layer)
    - asyncpg         (PostgreSQL async driver - belongs in infra layer)
    - aiohttp         (Async HTTP client - belongs in infra layer)
    - redis           (Redis client library - belongs in infra layer)
    - psycopg         (PostgreSQL driver - belongs in infra layer)
    - psycopg2        (PostgreSQL driver - belongs in infra layer)
    - consul          (Consul client library - belongs in infra layer)
    - hvac            (Vault client library - belongs in infra layer)
    - aiokafka        (Async Kafka client - belongs in infra layer)
    - confluent_kafka (Confluent Kafka client - belongs in infra layer)

EXIT CODES:
    0   All checks passed - no violations found
    1   Architecture violation detected
    2   Script error (path not found, invalid arguments, etc.)

EXAMPLES:
    # Run with auto-detected omnibase_core path
    ./scripts/check_architecture.sh

    # Run with verbose output
    ./scripts/check_architecture.sh --verbose

    # Run with custom path
    ./scripts/check_architecture.sh --path /path/to/omnibase_core

    # Run in CI (no colors)
    ./scripts/check_architecture.sh --no-color

EOF
}

# =============================================================================
# Path Detection
# =============================================================================

find_omnibase_core_path() {
    local custom_path="${1:-}"

    # If custom path provided, use it
    if [[ -n "$custom_path" ]]; then
        if [[ -d "$custom_path" ]]; then
            echo "$custom_path"
            return 0
        else
            echo "ERROR: Specified path does not exist: $custom_path" >&2
            return 2
        fi
    fi

    # Try to find installed package using Python
    local python_path
    python_path=$(python3 -c "import omnibase_core; import os; print(os.path.dirname(omnibase_core.__file__))" 2>/dev/null) || true

    if [[ -n "$python_path" && -d "$python_path" ]]; then
        echo "$python_path"
        return 0
    fi

    # Try common local paths
    local local_paths=(
        "./src/omnibase_core"
        "../omnibase_core/src/omnibase_core"
        "../omnibase_core"
    )

    for path in "${local_paths[@]}"; do
        if [[ -d "$path" ]]; then
            echo "$(cd "$path" && pwd)"
            return 0
        fi
    done

    echo "ERROR: Could not find omnibase_core. Use --path to specify location." >&2
    return 2
}

# =============================================================================
# Check Functions
# =============================================================================

build_grep_excludes() {
    local excludes=""

    # Add file pattern excludes
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        excludes="$excludes --exclude=$pattern"
    done

    # Add directory excludes
    for dir in "${EXCLUDE_DIRS[@]}"; do
        excludes="$excludes --exclude-dir=$dir"
    done

    echo "$excludes"
}

check_import() {
    local import_name="$1"
    local search_path="$2"
    local verbose="$3"
    local excludes

    excludes=$(build_grep_excludes)

    # Build grep command
    # Looking for:
    # - import kafka
    # - from kafka import ...
    # - import kafka.something
    # - from kafka.something import ...
    local pattern="^[[:space:]]*(from[[:space:]]+${import_name}[[:space:].]+import|import[[:space:]]+${import_name}([[:space:]]|$|\.))"

    # Run grep and capture output
    local violations
    # shellcheck disable=SC2086
    violations=$(grep -rn --include="*.py" $excludes -E "$pattern" "$search_path" 2>/dev/null) || true

    if [[ -n "$violations" ]]; then
        print_fail "Found '${import_name}' imports:"
        echo ""
        echo "$violations" | while IFS= read -r line; do
            echo "    $line"
        done
        echo ""
        return 1
    else
        print_pass "No '${import_name}' imports found"
        return 0
    fi
}

count_python_files() {
    local search_path="$1"
    local count
    count=$(find "$search_path" -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "$count"
}

# =============================================================================
# Main
# =============================================================================

main() {
    local verbose=false
    local custom_path=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                show_help
                exit 0
                ;;
            --verbose|-v)
                verbose=true
                shift
                ;;
            --path)
                if [[ -z "${2:-}" ]]; then
                    echo "ERROR: --path requires a value" >&2
                    exit 2
                fi
                custom_path="$2"
                shift 2
                ;;
            --no-color)
                USE_COLOR=false
                shift
                ;;
            *)
                echo "ERROR: Unknown option: $1" >&2
                echo "Use --help for usage information" >&2
                exit 2
                ;;
        esac
    done

    # Setup colors after parsing --no-color
    setup_colors

    # Find omnibase_core path
    local core_path
    core_path=$(find_omnibase_core_path "$custom_path") || exit 2

    print_header "Architecture Invariant Verification"

    echo "Target: $core_path"

    if [[ "$verbose" == "true" ]]; then
        local file_count
        file_count=$(count_python_files "$core_path")
        print_info "Found $file_count Python files to scan"
    fi

    echo ""
    echo "Checking omnibase_core for forbidden imports..."
    echo ""

    # Run checks
    local has_violations=false

    for import_name in "${FORBIDDEN_IMPORTS[@]}"; do
        if ! check_import "$import_name" "$core_path" "$verbose"; then
            has_violations=true
        fi
    done

    echo ""

    # Summary
    if [[ "$has_violations" == "true" ]]; then
        print_header "ARCHITECTURE VIOLATION DETECTED"
        echo -e "${RED}${BOLD}omnibase_core contains infrastructure dependencies!${NC}"
        echo ""
        echo "The core layer must not import infrastructure-specific packages."
        echo "These imports should be moved to omnibase_infra or removed."
        echo ""
        echo "Exit code: 1"
        exit 1
    else
        print_header "All checks passed!"
        echo -e "${GREEN}omnibase_core maintains proper layer separation.${NC}"
        echo ""
        echo "Exit code: 0"
        exit 0
    fi
}

# Run main function
main "$@"
