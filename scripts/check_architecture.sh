#!/usr/bin/env bash
# Architecture Invariant Verification Script
# OMN-255: Verify omnibase_core does not contain infrastructure dependencies
#
# This script checks that omnibase_core maintains proper layer separation
# by not importing infrastructure-specific packages like kafka, httpx, asyncpg.
#
# =============================================================================
# LIMITATIONS - IMPORTANT
# =============================================================================
#
# This script uses grep-based pattern matching which has inherent limitations:
#
# 1. INLINE IMPORTS NOT DETECTED:
#    Imports inside functions or methods are NOT detected by this script:
#
#        def my_function():
#            import kafka          # NOT DETECTED!
#            from httpx import X   # NOT DETECTED!
#
#    This is a fundamental limitation of line-based grep matching - it cannot
#    understand Python code structure or scope.
#
# 2. FALSE NEGATIVES (may miss):
#    - Imports constructed with __import__() or importlib.import_module()
#    - Imports hidden behind conditional logic (if/else)
#    - String-based import references in configuration files
#
# 3. FALSE POSITIVES (may incorrectly flag):
#    - Commented imports (partially mitigated by regex)
#    - Imports mentioned in docstrings (partially mitigated)
#    - Variable names matching import patterns (e.g., kafka_topic)
#
# FOR COMPREHENSIVE ANALYSIS, use the Python tests instead:
#     pytest tests/ci/test_architecture_compliance.py
#
# The Python tests use AST parsing and detect ALL imports including inline ones.
#
# =============================================================================
#
# Usage:
#   ./scripts/check_architecture.sh [OPTIONS]
#
# Options:
#   --help, -h      Show this help message
#   --verbose, -v   Show detailed output
#   --path PATH     Specify custom omnibase_core path
#   --no-color      Disable colored output
#   --json          Output results in JSON format

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# =============================================================================
# Forbidden Imports Configuration
# =============================================================================
#
# SYNCHRONIZATION REQUIREMENT:
# This list MUST match the Python tests in tests/ci/test_architecture_compliance.py.
# Both tools check the same forbidden imports for consistency.
#
# The Python tests have two sources:
#   1. test_no_infra_import_in_core() - Parametrized tests for each import
#   2. test_comprehensive_infra_scan() - Bulk scan for all imports
#
# aiohttp and redis have xfail markers in Python (tracked by Linear tickets)
# but are still enforced here to catch violations early.
#
# When adding/removing imports, update BOTH:
#   - This array (FORBIDDEN_IMPORTS)
#   - tests/ci/test_architecture_compliance.py (both test methods)
#
# NOTE: All imports listed here will cause a hard failure if detected.
# For known issues tracked in Linear, see KNOWN_ISSUES below.
#
FORBIDDEN_IMPORTS=(
    "kafka"              # Event streaming client
    "httpx"              # HTTP client library
    "asyncpg"            # PostgreSQL async driver
    "aiohttp"            # Known issue: OMN-1015 (xfail in Python tests)
    "redis"              # Known issue: OMN-1295 (xfail in Python tests)
    "psycopg"            # PostgreSQL driver (v3)
    "psycopg2"           # PostgreSQL driver (v2)
    "consul"             # Consul client library
    "hvac"               # Vault client library
    "aiokafka"           # Async Kafka client
    "confluent_kafka"    # Confluent Kafka client
)

# Known issues with Linear ticket references
# These are included in FORBIDDEN_IMPORTS for completeness, but have
# corresponding xfail markers in the Python test suite.
# Format: "import_name|ticket_id|description"
#
# NOTE: This array is intentionally defined for documentation purposes and
# future use (e.g., reporting known issues with ticket links). The data is
# also referenced in the --help output. Suppress shellcheck unused warning.
# shellcheck disable=SC2034
KNOWN_ISSUES=(
    "aiohttp|OMN-1015|async HTTP client usage in core - needs migration to infra"
    "redis|OMN-1295|Redis client usage in core - needs migration to infra"
)

# File patterns to exclude from checking (quoted for shellcheck compliance)
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

# Directory patterns to exclude (quoted for shellcheck compliance)
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
    if [[ "${USE_COLOR}" == "true" ]]; then
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
# JSON Output Support
# =============================================================================

OUTPUT_JSON=false
declare -a JSON_VIOLATIONS=()
declare -a JSON_PASSED=()
declare -a JSON_EXCLUDED_PATTERNS=()
declare -a JSON_EXCLUDED_DIRS=()
JSON_TARGET=""
JSON_FILE_COUNT=0
JSON_EXIT_CODE=0

json_escape() {
    local str="$1"
    # Escape backslashes, double quotes, and control characters
    str="${str//\\/\\\\}"
    str="${str//\"/\\\"}"
    str="${str//$'\n'/\\n}"
    str="${str//$'\r'/\\r}"
    str="${str//$'\t'/\\t}"
    printf '%s' "$str"
}

json_add_violation() {
    local import_name="$1"
    local violations="$2"
    local escaped_import
    local escaped_violations
    escaped_import=$(json_escape "$import_name")
    escaped_violations=$(json_escape "$violations")
    JSON_VIOLATIONS+=("{\"import\":\"${escaped_import}\",\"violations\":\"${escaped_violations}\"}")
}

json_add_passed() {
    local import_name="$1"
    local escaped_import
    escaped_import=$(json_escape "$import_name")
    JSON_PASSED+=("\"${escaped_import}\"")
}

output_json() {
    local violations_json=""
    local passed_json=""
    local excluded_patterns_json=""
    local excluded_dirs_json=""

    # Build violations array
    if [[ ${#JSON_VIOLATIONS[@]} -gt 0 ]]; then
        violations_json=$(printf '%s,' "${JSON_VIOLATIONS[@]}")
        violations_json="[${violations_json%,}]"
    else
        violations_json="[]"
    fi

    # Build passed array
    if [[ ${#JSON_PASSED[@]} -gt 0 ]]; then
        passed_json=$(printf '%s,' "${JSON_PASSED[@]}")
        passed_json="[${passed_json%,}]"
    else
        passed_json="[]"
    fi

    # Build excluded patterns arrays for debugging
    local pattern
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        local escaped
        escaped=$(json_escape "$pattern")
        excluded_patterns_json="${excluded_patterns_json}\"${escaped}\","
    done
    excluded_patterns_json="[${excluded_patterns_json%,}]"

    for pattern in "${EXCLUDE_DIRS[@]}"; do
        local escaped
        escaped=$(json_escape "$pattern")
        excluded_dirs_json="${excluded_dirs_json}\"${escaped}\","
    done
    excluded_dirs_json="[${excluded_dirs_json%,}]"

    local escaped_target
    escaped_target=$(json_escape "$JSON_TARGET")

    cat << EOF
{
  "success": $(if [[ ${JSON_EXIT_CODE} -eq 0 ]]; then echo "true"; else echo "false"; fi),
  "exit_code": ${JSON_EXIT_CODE},
  "target": "${escaped_target}",
  "files_scanned": ${JSON_FILE_COUNT},
  "violations": ${violations_json},
  "passed": ${passed_json},
  "forbidden_imports_checked": ${#FORBIDDEN_IMPORTS[@]},
  "excluded_file_patterns": ${excluded_patterns_json},
  "excluded_directories": ${excluded_dirs_json}
}
EOF
}

# =============================================================================
# Utility Functions
# =============================================================================

print_header() {
    if [[ "${OUTPUT_JSON}" == "true" ]]; then
        return
    fi
    echo ""
    echo -e "${BOLD}===============================================${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BOLD}===============================================${NC}"
    echo ""
}

print_pass() {
    if [[ "${OUTPUT_JSON}" == "true" ]]; then
        return
    fi
    echo -e "  ${GREEN}[PASS]${NC} $1"
}

print_fail() {
    if [[ "${OUTPUT_JSON}" == "true" ]]; then
        return
    fi
    echo -e "  ${RED}[FAIL]${NC} $1"
}

print_info() {
    if [[ "${OUTPUT_JSON}" == "true" ]]; then
        return
    fi
    echo -e "  ${BLUE}[INFO]${NC} $1"
}

# shellcheck disable=SC2317  # Function defined for future use in verbose mode
print_warn() {
    if [[ "${OUTPUT_JSON}" == "true" ]]; then
        return
    fi
    echo -e "  ${YELLOW}[WARN]${NC} $1"
}

print_skip() {
    if [[ "${OUTPUT_JSON}" == "true" ]]; then
        return
    fi
    echo -e "  ${YELLOW}[SKIP]${NC} $1"
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
    --json          Output results in JSON format (useful for CI integration)

FORBIDDEN IMPORTS:
    - kafka           (Kafka client library - belongs in infra layer)
    - httpx           (HTTP client library - belongs in infra layer)
    - asyncpg         (PostgreSQL async driver - belongs in infra layer)
    - aiohttp         (Async HTTP client - belongs in infra layer) [*]
    - redis           (Redis client library - belongs in infra layer) [*]
    - psycopg         (PostgreSQL driver - belongs in infra layer)
    - psycopg2        (PostgreSQL driver - belongs in infra layer)
    - consul          (Consul client library - belongs in infra layer)
    - hvac            (Vault client library - belongs in infra layer)
    - aiokafka        (Async Kafka client - belongs in infra layer)
    - confluent_kafka (Confluent Kafka client - belongs in infra layer)

    [*] Known issues with tracking tickets - see KNOWN ISSUES below.

KNOWN ISSUES:
    Some imports have known violations that are tracked in Linear tickets.
    The Python tests (tests/ci/test_architecture_compliance.py) use xfail
    markers for these, but this script enforces the invariant strictly.

    - aiohttp: OMN-1015 - async HTTP client needs migration to infra
    - redis:   OMN-1295 - Redis client needs migration to infra

    These violations will still cause the script to fail (exit code 1),
    but they are documented for visibility. Fix them by addressing the
    corresponding Linear tickets.

EXIT CODES:
    0   All checks passed - no violations found
    1   Architecture violation detected
    2   Script error (path not found, invalid arguments, etc.)

JSON OUTPUT:
    When using --json, the output format is:
    {
      "success": true|false,
      "exit_code": 0|1|2,
      "target": "/path/to/omnibase_core",
      "files_scanned": 123,
      "violations": [{"import": "kafka", "violations": "file:line: ..."}],
      "passed": ["httpx", "asyncpg", ...],
      "forbidden_imports_checked": 11,
      "excluded_file_patterns": ["*.md", "*.yaml", ...],
      "excluded_directories": [".git", "__pycache__", ...]
    }

    The excluded_file_patterns and excluded_directories fields help debug
    cases where expected files are not being scanned.

EXAMPLES:
    # Run with auto-detected omnibase_core path
    ./scripts/check_architecture.sh

    # Run with verbose output
    ./scripts/check_architecture.sh --verbose

    # Run with custom path
    ./scripts/check_architecture.sh --path /path/to/omnibase_core

    # Run in CI (no colors)
    ./scripts/check_architecture.sh --no-color

    # Run with JSON output for programmatic consumption
    ./scripts/check_architecture.sh --json

LIMITATIONS:
    This script uses grep-based pattern matching, which has significant
    limitations compared to AST-based Python analysis:

    *** CRITICAL: INLINE IMPORTS NOT DETECTED ***

    This script CANNOT detect imports inside functions or methods:

        def my_function():
            import kafka  # NOT DETECTED!
            from httpx import Client  # NOT DETECTED!

    Inline imports are common patterns to:
      - Avoid circular import issues
      - Lazy-load heavy dependencies
      - Conditionally import based on runtime conditions

    For code using inline imports, use the Python tests instead.

    1. False Negatives (May Miss):
       - INLINE IMPORTS inside functions/methods (see above)
       - Imports constructed dynamically at runtime
       - Imports hidden behind conditional logic (if/else)
       - Imports using __import__() or importlib.import_module()
       - String-based import references in configuration

    2. False Positives (May Incorrectly Flag):
       - Commented imports (partially mitigated by regex)
       - Imports in docstrings (grep cannot parse multiline strings)
       - Variable names matching import patterns (e.g., kafka_topic)

    3. Known Issues (tracked in Linear):
       - aiohttp: OMN-1015 - async HTTP client needs migration
       - redis:   OMN-1295 - Redis client needs migration

    RECOMMENDED: For comprehensive AST-based analysis, use:
        pytest tests/ci/test_architecture_compliance.py

    The Python tests provide:
       - Proper AST parsing of ALL imports (including inline)
       - Multiline docstring handling
       - xfail markers for known issues
       - More accurate detection with fewer false positives

COMPARISON WITH PYTHON TESTS:
    This bash script is designed for quick CI checks. The Python tests in
    tests/ci/test_architecture_compliance.py provide more thorough analysis.

    Both tools check the same forbidden imports list. Discrepancies should
    be reported as bugs.

EOF
}

# =============================================================================
# Path Detection
# =============================================================================

find_omnibase_core_path() {
    local custom_path="${1:-}"

    # If custom path provided, use it
    if [[ -n "${custom_path}" ]]; then
        if [[ -d "${custom_path}" ]]; then
            echo "${custom_path}"
            return 0
        else
            echo "ERROR: Specified path does not exist: ${custom_path}" >&2
            return 2
        fi
    fi

    # Try to find installed package using Python
    local python_path
    python_path=$(python3 -c "import omnibase_core; import os; print(os.path.dirname(omnibase_core.__file__))" 2>/dev/null) || true

    if [[ -n "${python_path}" && -d "${python_path}" ]]; then
        echo "${python_path}"
        return 0
    fi

    # Try common local paths
    local local_paths=(
        "./src/omnibase_core"
        "../omnibase_core/src/omnibase_core"
        "../omnibase_core"
    )

    for path in "${local_paths[@]}"; do
        if [[ -d "${path}" ]]; then
            (cd "${path}" && pwd)
            return 0
        fi
    done

    echo "ERROR: Could not find omnibase_core. Use --path to specify location." >&2
    return 2
}

# =============================================================================
# Check Functions
# =============================================================================

# Build array of grep exclude arguments
# This approach avoids shellcheck SC2086 by using proper array expansion
declare -a GREP_EXCLUDE_ARGS=()

build_grep_excludes() {
    GREP_EXCLUDE_ARGS=()

    # Add file pattern excludes
    # Array iteration with quotes preserves glob patterns for grep
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        GREP_EXCLUDE_ARGS+=("--exclude=${pattern}")
    done

    # Add directory excludes
    for dir in "${EXCLUDE_DIRS[@]}"; do
        GREP_EXCLUDE_ARGS+=("--exclude-dir=${dir}")
    done
}

# Escape special regex characters in a string for use with grep -E (ERE)
#
# This function escapes all characters that have special meaning in Extended
# Regular Expressions (ERE), making them match literally.
#
# Characters escaped: . * + ? ^ $ [ ] { } ( ) | \
#
# IMPORTANT: The character class in sed must have specific ordering:
#   - ] must come FIRST inside [] to be literal (or escaped)
#   - [ can be anywhere in the class
#   - \ must be doubled in the replacement to produce a single backslash
#
# Example: "kafka.producer" -> "kafka\.producer"
#
escape_regex() {
    local input="$1"
    # Escape ERE special characters for use in grep -E patterns
    # Character class: ][\.^$*+?{}()|
    #   - ] first (POSIX requirement for literal ])
    #   - [ anywhere
    #   - Other special chars follow
    # shellcheck disable=SC1003  # This is intentional escaping for sed
    printf '%s' "${input}" | sed -e 's/[][\^$.*+?{}()|]/\\&/g'
}

check_import() {
    local import_name="$1"
    local search_path="$2"
    local verbose="$3"

    # Build exclude arguments array
    build_grep_excludes

    # Escape special regex characters in import name
    local escaped_import
    escaped_import=$(escape_regex "${import_name}")

    # Build grep pattern
    # Looking for:
    # - import kafka
    # - from kafka import ...
    # - import kafka.something
    # - from kafka.something import ...
    # Pattern explanation:
    #   Part 1: from kafka... import - matches "from kafka" or "from kafka.submodule" imports
    #           (\.[[:alnum:]_]+)* ensures only dot-separated submodules are matched
    #           This prevents matching "kafka_utils" when checking for "kafka"
    #   Part 2: import kafka... - matches "import kafka" or "import kafka.something"
    #           (\.[[:alnum:]_]+)* same submodule handling
    #           ([[:space:],]|$) handles end of import (space, comma for multi-import, or EOL)
    local pattern="^[[:space:]]*(from[[:space:]]+${escaped_import}(\\.[[:alnum:]_]+)*[[:space:]]+import|import[[:space:]]+${escaped_import}(\\.[[:alnum:]_]+)*([[:space:],]|$))"

    # Run grep and capture output
    # Using array expansion "${GREP_EXCLUDE_ARGS[@]}" for shellcheck compliance
    local violations
    violations=$(grep -rn --include="*.py" "${GREP_EXCLUDE_ARGS[@]}" -E "${pattern}" "${search_path}" 2>/dev/null) || true

    if [[ -n "${violations}" ]]; then
        if [[ "${OUTPUT_JSON}" == "true" ]]; then
            json_add_violation "${import_name}" "${violations}"
        else
            print_fail "Found '${import_name}' imports:"
            echo ""
            echo "${violations}" | while IFS= read -r line; do
                echo "    ${line}"
            done
            echo ""
        fi
        return 1
    else
        if [[ "${OUTPUT_JSON}" == "true" ]]; then
            json_add_passed "${import_name}"
        else
            print_pass "No '${import_name}' imports found"
        fi
        return 0
    fi
}

count_python_files() {
    local search_path="$1"
    local count
    count=$(find "${search_path}" -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "${count}"
}

# =============================================================================
# Known Issues Reporting
# =============================================================================

report_known_issues() {
    local has_violations="$1"

    if [[ "${has_violations}" != "true" ]]; then
        return
    fi

    # Check which known issues were found in violations
    local found_known_issues=false

    for issue in "${KNOWN_ISSUES[@]}"; do
        IFS='|' read -r import_name ticket_id description <<< "${issue}"

        # Check if this import was found in violations (check JSON_VIOLATIONS array)
        local found=false
        if [[ "${OUTPUT_JSON}" == "true" ]]; then
            for v in "${JSON_VIOLATIONS[@]:-}"; do
                if [[ "${v}" == *"\"${import_name}\""* ]]; then
                    found=true
                    break
                fi
            done
        else
            # For text output, we've already printed violations
            # This function is called after check_import, so we re-check
            # Escape the import name for use in grep pattern
            local escaped_name
            escaped_name=$(escape_regex "${import_name}")
            if grep -rq --include="*.py" -E "^[[:space:]]*(from[[:space:]]+${escaped_name}[[:space:].]+import|import[[:space:]]+${escaped_name}([[:space:]]|$|\\.))" "${JSON_TARGET}" 2>/dev/null; then
                found=true
            fi
        fi

        if [[ "${found}" == "true" ]]; then
            if [[ "${found_known_issues}" == "false" ]]; then
                found_known_issues=true
                echo ""
                echo "============================================================"
                echo "KNOWN ISSUES (tracked in Linear)"
                echo "============================================================"
            fi
            echo ""
            echo "  ${import_name}:"
            echo "    Ticket: ${ticket_id}"
            echo "    Description: ${description}"
            echo "    Link: https://linear.app/onex/issue/${ticket_id}"
        fi
    done

    if [[ "${found_known_issues}" == "true" ]]; then
        echo ""
        echo "------------------------------------------------------------"
        echo "These violations are known and tracked. Fix by resolving the"
        echo "corresponding Linear tickets listed above."
        echo "============================================================"
        echo ""
    fi
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
            --json)
                OUTPUT_JSON=true
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
    if ! core_path=$(find_omnibase_core_path "${custom_path}"); then
        if [[ "${OUTPUT_JSON}" == "true" ]]; then
            JSON_EXIT_CODE=2
            JSON_TARGET="(not found)"
            output_json
        fi
        exit 2
    fi

    JSON_TARGET="${core_path}"

    print_header "Architecture Invariant Verification"

    if [[ "${OUTPUT_JSON}" != "true" ]]; then
        echo "Target: ${core_path}"
    fi

    # Always show file count for CI debugging
    local file_count
    file_count=$(count_python_files "${core_path}")
    JSON_FILE_COUNT="${file_count}"
    print_info "Found ${file_count} Python files to scan"

    # Handle case where no Python files found
    if [[ "${file_count}" -eq 0 ]]; then
        print_skip "No Python files found in target directory: ${core_path}"
        print_skip "Reason: Directory may be empty or contain no .py files"
        if [[ "${OUTPUT_JSON}" == "true" ]]; then
            JSON_EXIT_CODE=0
            output_json
        fi
        exit 0
    fi

    # Always show what's being excluded for CI debugging
    # This helps diagnose issues where expected files are not being scanned
    print_skip "Excluding file patterns: ${EXCLUDE_PATTERNS[*]}"
    print_skip "Excluding directories: ${EXCLUDE_DIRS[*]}"
    print_info "Checking ${#FORBIDDEN_IMPORTS[@]} forbidden import patterns"

    if [[ "${verbose}" == "true" ]]; then
        print_info "Verbose mode enabled - showing all check details"
    fi

    if [[ "${OUTPUT_JSON}" != "true" ]]; then
        echo ""
        echo "Checking omnibase_core for forbidden imports..."
        echo ""
    fi

    # Run checks
    local has_violations=false

    for import_name in "${FORBIDDEN_IMPORTS[@]}"; do
        if ! check_import "${import_name}" "${core_path}" "${verbose}"; then
            has_violations=true
        fi
    done

    if [[ "${OUTPUT_JSON}" != "true" ]]; then
        echo ""
    fi

    # Summary
    if [[ "${has_violations}" == "true" ]]; then
        JSON_EXIT_CODE=1
        if [[ "${OUTPUT_JSON}" == "true" ]]; then
            output_json
        else
            # Report known issues with ticket links (text output only)
            report_known_issues "${has_violations}"

            print_header "ARCHITECTURE VIOLATION DETECTED"
            echo -e "${RED}${BOLD}omnibase_core contains infrastructure dependencies!${NC}"
            echo ""
            echo "The core layer must not import infrastructure-specific packages."
            echo "These imports should be moved to omnibase_infra or removed."
            echo ""
            echo "Exit code: 1"
        fi
        exit 1
    else
        JSON_EXIT_CODE=0
        if [[ "${OUTPUT_JSON}" == "true" ]]; then
            output_json
        else
            print_header "All checks passed!"
            echo -e "${GREEN}omnibase_core maintains proper layer separation.${NC}"
            echo ""
            echo "Exit code: 0"
        fi
        exit 0
    fi
}

# Run main function
main "$@"
