#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
#
# E2E Test Diagnostic Automation Script
#
# This script automates the diagnostic steps from E2E_DIAGNOSTIC_MANUAL.md
# to quickly identify event processing pipeline failures.
#
# Usage:
#   ./scripts/diagnose_e2e.sh [--rebuild] [--full-report]
#
# Options:
#   --rebuild      Rebuild Docker container before testing
#   --full-report  Generate comprehensive diagnostic report
#   --help         Show this help message
#
# Portability:
#   This script is designed to work on both GNU (Linux) and BSD (macOS) systems.
#   All grep options used are POSIX-compatible or widely supported:
#     -E (extended regex), -i (case insensitive), -v (invert), -q (quiet)
#     -A (after context) is supported by both GNU and BSD grep
#   Character classes use POSIX syntax: [[:space:]] instead of \s
#
# Output:
#   - Test results printed to console
#   - Logs saved to /tmp/e2e_diagnostic_*
#   - Scenario identification printed
#   - Recommended fixes displayed
#
# Example:
#   # Quick diagnostic (use existing container)
#   ./scripts/diagnose_e2e.sh
#
#   # Full diagnostic with rebuild
#   ./scripts/diagnose_e2e.sh --rebuild --full-report

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker"
COMPOSE_FILE="${DOCKER_DIR}/docker-compose.e2e.yml"

# Log file locations
LOG_DIR="/tmp"
STARTUP_LOG="${LOG_DIR}/runtime_startup.log"
TEST_LOG="${LOG_DIR}/test_output.log"
FULL_LOG="${LOG_DIR}/runtime_full_logs.log"
REPORT_FILE="${LOG_DIR}/e2e_diagnostic_report.md"

# Test configuration
TEST_FILE="tests/integration/registration/e2e/test_runtime_e2e.py"
TEST_NAME="TestRuntimeE2EFlow::test_introspection_event_processed_by_runtime"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Command-line Argument Parsing
# =============================================================================

REBUILD=false
FULL_REPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --full-report)
            FULL_REPORT=true
            shift
            ;;
        --help)
            # Note: Use grouped grep with || true inside pipeline to handle pipefail safely
            head -n 30 "${BASH_SOURCE[0]}" | { grep "^#" || true; } | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

section_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$*${NC}"
    echo -e "${BLUE}========================================${NC}"
}

check_required_log() {
    local log_file="$1"
    local pattern="$2"
    local description="$3"

    if grep -qi "$pattern" "$log_file"; then
        log_success "✓ $description"
        return 0
    else
        log_error "✗ $description"
        return 1
    fi
}

# =============================================================================
# Step 1: Container Rebuild (Optional)
# =============================================================================

if [ "$REBUILD" = true ]; then
    section_header "Step 1: Rebuilding Runtime Container"

    cd "$DOCKER_DIR"

    log_info "Stopping existing runtime container..."
    docker compose -f docker-compose.e2e.yml down runtime 2>/dev/null || true

    log_info "Rebuilding runtime image (this may take 2-3 minutes)..."
    if DOCKER_BUILDKIT=1 docker compose -f docker-compose.e2e.yml build runtime; then
        log_success "Container rebuilt successfully"
    else
        log_error "Container rebuild failed"
        exit 1
    fi

    log_info "Starting infrastructure + runtime..."
    if docker compose -f docker-compose.e2e.yml --profile runtime up -d; then
        log_success "Containers started"
    else
        log_error "Failed to start containers"
        exit 1
    fi

    log_info "Waiting for services to become healthy (30 seconds)..."
    sleep 30
else
    section_header "Step 1: Verifying Existing Container"

    log_info "Checking if runtime container is running..."
    # Note: Docker Compose v1 uses "Up", v2 uses "running" - check both case-insensitively
    # Note: Use grouped grep with || true to handle pipefail safely when no matches
    container_status=$(docker compose -f "$COMPOSE_FILE" ps runtime 2>/dev/null || true)
    if echo "$container_status" | { grep -Eiq "(Up|running)" || false; }; then
        log_success "Runtime container is running"
    else
        log_warning "Runtime container not running - use --rebuild to start it"
        exit 1
    fi
fi

# =============================================================================
# Step 2: Health Check
# =============================================================================

section_header "Step 2: Container Health Verification"

log_info "Checking container status..."
docker compose -f "$COMPOSE_FILE" ps runtime

log_info "Testing health endpoint..."
if curl -sf http://localhost:8085/health >/dev/null 2>&1; then
    log_success "Health endpoint responding"
else
    log_error "Health endpoint not responding"
    log_info "Checking container logs for errors..."
    docker compose -f "$COMPOSE_FILE" logs --tail=50 runtime
    exit 1
fi

# =============================================================================
# Step 3: Capture Startup Logs
# =============================================================================

section_header "Step 3: Capturing Startup Logs"

log_info "Saving startup logs to $STARTUP_LOG..."
docker compose -f "$COMPOSE_FILE" logs runtime > "$STARTUP_LOG"
log_success "Startup logs captured ($(wc -l < "$STARTUP_LOG") lines)"

log_info "Verifying required startup messages..."
check_required_log "$STARTUP_LOG" "container wiring complete" "Container wiring complete"
check_required_log "$STARTUP_LOG" "HandlerNodeIntrospected resolved" "Handler resolved"
check_required_log "$STARTUP_LOG" "dispatcher created and wired" "Dispatcher created"
check_required_log "$STARTUP_LOG" "consumer started successfully" "Consumer started"
check_required_log "$STARTUP_LOG" "ONEX Runtime Kernel" "Kernel banner displayed"

# Check for startup errors (store in variable to handle empty results safely)
# Exclude false positives (case-insensitive to match the case-insensitive search):
#   - WARNING lines (not errors)
#   - Metric names: *_error_count, *_errors_total, errors_total (prometheus metrics)
#   - Method/function references: record_error, on_error, handle_error, error_handler (code references)
#   - Class names: ErrorHandler, ErrorContext, OnexError (type references, not actual errors)
#   - Zero counts: error_count=0, "error_count": 0, "errors": 0 (no actual errors)
#   - Debug log patterns: has_error=False, error=None (status checks, not errors)
# Note: Use -Eiv (case-insensitive exclusion) to match -Ei search pattern
# Note: Use -E (ERE) for portability across GNU and BSD grep (macOS)
# Note: Use intermediate file to avoid pipefail issues with empty grep results
startup_errors=""
: > /tmp/startup_errors_raw.txt  # Create/truncate temp file
if grep -Ei "error|exception" "$STARTUP_LOG" >> /tmp/startup_errors_raw.txt 2>/dev/null; then
    # Apply comprehensive exclusion filter for false positives
    startup_errors=$(grep -Eiv 'WARNING|_error_count|_errors_total|errors_total|record_error|on_error|handle_error|error_handler|ErrorHandler|ErrorContext|OnexError|"error_count":[[:space:]]*0|error_count[[:space:]]*=[[:space:]]*0|"errors":[[:space:]]*0|has_error[[:space:]]*=[[:space:]]*(False|false)|error[[:space:]]*=[[:space:]]*None' /tmp/startup_errors_raw.txt 2>/dev/null) || true
fi
if [ -n "$startup_errors" ]; then
    log_warning "Startup errors detected:"
    echo "$startup_errors" | head -10
fi

# =============================================================================
# Step 4: Run E2E Test
# =============================================================================

section_header "Step 4: Running E2E Test"

cd "$PROJECT_ROOT"

log_info "Starting test execution (this may take 30-45 seconds)..."

# Run test with verbose output
if uv run pytest \
    "${TEST_FILE}::${TEST_NAME}" \
    -v -s --log-cli-level=DEBUG \
    2>&1 | tee "$TEST_LOG"; then
    TEST_RESULT="PASSED"
    log_success "Test PASSED ✓"
else
    TEST_RESULT="FAILED"
    log_error "Test FAILED ✗"
fi

# =============================================================================
# Step 5: Capture Full Container Logs
# =============================================================================

section_header "Step 5: Capturing Container Logs During Test"

log_info "Saving full container logs to $FULL_LOG..."
docker compose -f "$COMPOSE_FILE" logs runtime > "$FULL_LOG"
log_success "Container logs captured ($(wc -l < "$FULL_LOG") lines)"

# =============================================================================
# Step 6: Analyze Logs for Scenario Identification
# =============================================================================

section_header "Step 6: Event Processing Pipeline Analysis"

log_info "Analyzing event processing pipeline..."

# Pipeline checkpoint analysis
CALLBACK_INVOKED=false
MESSAGE_PARSED=false
VALIDATION_ATTEMPTED=false
VALIDATION_SUCCEEDED=false
ENVELOPE_CREATED=false
DISPATCHER_ROUTED=false
HANDLER_SUCCEEDED=false

if grep -qi "introspection message callback invoked" "$FULL_LOG"; then
    log_success "✓ Callback invoked (message received)"
    CALLBACK_INVOKED=true
else
    log_error "✗ Callback NOT invoked (no messages received)"
fi

if grep -qi "parsing message value" "$FULL_LOG"; then
    log_success "✓ Message parsed"
    MESSAGE_PARSED=true
else
    [ "$CALLBACK_INVOKED" = true ] && log_error "✗ Message NOT parsed"
fi

if grep -qi "validating payload as ModelNodeIntrospectionEvent" "$FULL_LOG"; then
    log_success "✓ Validation attempted"
    VALIDATION_ATTEMPTED=true
else
    [ "$MESSAGE_PARSED" = true ] && log_error "✗ Validation NOT attempted"
fi

if grep -qi "introspection event parsed successfully" "$FULL_LOG"; then
    log_success "✓ Validation succeeded"
    VALIDATION_SUCCEEDED=true
else
    [ "$VALIDATION_ATTEMPTED" = true ] && log_error "✗ Validation FAILED"
fi

if grep -qi "event envelope created" "$FULL_LOG"; then
    log_success "✓ Envelope created"
    ENVELOPE_CREATED=true
else
    [ "$VALIDATION_SUCCEEDED" = true ] && log_error "✗ Envelope NOT created"
fi

if grep -qi "routing to introspection dispatcher" "$FULL_LOG"; then
    log_success "✓ Dispatcher routing attempted"
    DISPATCHER_ROUTED=true
else
    [ "$ENVELOPE_CREATED" = true ] && log_error "✗ Dispatcher routing NOT attempted"
fi

if grep -qi "introspection event processed successfully" "$FULL_LOG"; then
    log_success "✓ Handler execution succeeded"
    HANDLER_SUCCEEDED=true
else
    [ "$DISPATCHER_ROUTED" = true ] && log_error "✗ Handler execution FAILED"
fi

# =============================================================================
# Step 7: Scenario Identification
# =============================================================================

section_header "Step 7: Failure Scenario Identification"

SCENARIO="UNKNOWN"
ROOT_CAUSE=""
RECOMMENDED_FIX=""

# Determine scenario based on pipeline checkpoints
if ! grep -qi "consumer started successfully" "$STARTUP_LOG"; then
    SCENARIO="A (Consumer Never Started)"
    ROOT_CAUSE="Introspection consumer failed to start during kernel initialization"
    RECOMMENDED_FIX="Check handler wiring, event bus type, and Kafka connectivity"

elif [ "$CALLBACK_INVOKED" = false ]; then
    SCENARIO="B (No Messages Received)"
    ROOT_CAUSE="Consumer started but not receiving Kafka messages"
    RECOMMENDED_FIX="Verify topic name matches, check Kafka broker routing, reset consumer offset"

elif [ "$VALIDATION_SUCCEEDED" = false ] && [ "$VALIDATION_ATTEMPTED" = true ]; then
    SCENARIO="C (Validation Failed)"
    ROOT_CAUSE="Message received but ModelNodeIntrospectionEvent validation failed"
    RECOMMENDED_FIX="Check message schema, compare to ModelNodeIntrospectionEvent fields"

elif [ "$DISPATCHER_ROUTED" = false ] && [ "$VALIDATION_SUCCEEDED" = true ]; then
    SCENARIO="D (Envelope Creation Failed)"
    ROOT_CAUSE="Validation succeeded but envelope creation or routing failed"
    RECOMMENDED_FIX="Check ModelEventEnvelope construction logic and dispatcher availability"

elif [ "$HANDLER_SUCCEEDED" = false ] && [ "$DISPATCHER_ROUTED" = true ]; then
    SCENARIO="E (Handler Execution Failed)"
    ROOT_CAUSE="Dispatcher routed successfully but handler execution failed"
    RECOMMENDED_FIX="Check handler logic, projector write, and database connectivity"

elif [ "$HANDLER_SUCCEEDED" = true ] && [ "$TEST_RESULT" = "FAILED" ]; then
    SCENARIO="F (Projection Not Visible)"
    ROOT_CAUSE="Handler succeeded but test cannot find projection in database"
    RECOMMENDED_FIX="Check database connection consistency, entity_id matching, timing issues"

elif [ "$TEST_RESULT" = "PASSED" ]; then
    SCENARIO="SUCCESS"
    ROOT_CAUSE="All pipeline steps completed successfully"
    RECOMMENDED_FIX="No fix needed - test passed"
else
    SCENARIO="UNKNOWN"
    ROOT_CAUSE="Unable to determine failure point from logs"
    RECOMMENDED_FIX="Manual log inspection required - see E2E_DIAGNOSTIC_MANUAL.md"
fi

log_info "Identified Scenario: ${SCENARIO}"
log_info "Root Cause: ${ROOT_CAUSE}"
log_info "Recommended Fix: ${RECOMMENDED_FIX}"

# =============================================================================
# Step 8: Error Summary
# =============================================================================

section_header "Step 8: Error Summary"

log_info "Extracting errors from logs..."
# Note: Use -Eiv (case-insensitive exclusion) to match -Ei search pattern
# Note: Use -E (ERE) for portability across GNU and BSD grep (macOS)
# Note: Use intermediate file to avoid pipefail issues with empty grep results
# Exclusions match startup error detection for consistency
: > /tmp/errors.txt  # Create empty file
: > /tmp/errors_raw.txt  # Create/truncate temp file
if grep -Ei "error|exception|failed" "$FULL_LOG" >> /tmp/errors_raw.txt 2>/dev/null; then
    grep -Eiv 'WARNING|_error_count|_errors_total|errors_total|record_error|on_error|handle_error|error_handler|ErrorHandler|ErrorContext|OnexError|"error_count":[[:space:]]*0|error_count[[:space:]]*=[[:space:]]*0|"errors":[[:space:]]*0|has_error[[:space:]]*=[[:space:]]*(False|false)|error[[:space:]]*=[[:space:]]*None' /tmp/errors_raw.txt > /tmp/errors.txt 2>/dev/null || true
fi
if [ -s /tmp/errors.txt ]; then
    ERROR_COUNT=$(wc -l < /tmp/errors.txt)
    log_warning "Found $ERROR_COUNT error lines:"
    head -20 /tmp/errors.txt
    if [ "$ERROR_COUNT" -gt 20 ]; then
        log_info "... (showing first 20 of $ERROR_COUNT errors)"
    fi
else
    log_success "No errors found in logs"
fi

# =============================================================================
# Step 9: Generate Full Report (Optional)
# =============================================================================

if [ "$FULL_REPORT" = true ]; then
    section_header "Step 9: Generating Comprehensive Report"

    log_info "Creating diagnostic report at $REPORT_FILE..."

    # Pre-compute values to avoid subshell failures with set -e pipefail
    # Note: Use -E (ERE) for portability across GNU and BSD grep (macOS)
    # Note: Use intermediate files and explicit || true to handle pipefail safely

    # Startup errors - two-stage grep to avoid pipefail with empty results
    # Note: Create temp file first to ensure it exists, avoiding pipefail issues
    # Exclusions match startup error detection for consistency
    REPORT_STARTUP_ERRORS=""
    : > /tmp/report_startup_raw.txt  # Create/truncate temp file
    if grep -Ei "error|exception" "$STARTUP_LOG" >> /tmp/report_startup_raw.txt 2>/dev/null; then
        REPORT_STARTUP_ERRORS=$(grep -Eiv 'WARNING|_error_count|_errors_total|errors_total|record_error|on_error|handle_error|error_handler|ErrorHandler|ErrorContext|OnexError|"error_count":[[:space:]]*0|error_count[[:space:]]*=[[:space:]]*0|"errors":[[:space:]]*0|has_error[[:space:]]*=[[:space:]]*(False|false)|error[[:space:]]*=[[:space:]]*None' /tmp/report_startup_raw.txt 2>/dev/null | head -20) || true
    fi
    if [ -z "$REPORT_STARTUP_ERRORS" ]; then
        REPORT_STARTUP_ERRORS="No errors found"
    fi

    # Processing errors - two-stage grep to avoid pipefail with empty results
    # Note: Create temp file first to ensure it exists, avoiding pipefail issues
    # Exclusions match error summary for consistency
    REPORT_PROCESSING_ERRORS=""
    : > /tmp/report_processing_raw.txt  # Create/truncate temp file
    if grep -Ei "error|exception|failed" "$FULL_LOG" >> /tmp/report_processing_raw.txt 2>/dev/null; then
        REPORT_PROCESSING_ERRORS=$(grep -Eiv 'WARNING|_error_count|_errors_total|errors_total|record_error|on_error|handle_error|error_handler|ErrorHandler|ErrorContext|OnexError|"error_count":[[:space:]]*0|error_count[[:space:]]*=[[:space:]]*0|"errors":[[:space:]]*0|has_error[[:space:]]*=[[:space:]]*(False|false)|error[[:space:]]*=[[:space:]]*None' /tmp/report_processing_raw.txt 2>/dev/null | head -30) || true
    fi
    if [ -z "$REPORT_PROCESSING_ERRORS" ]; then
        REPORT_PROCESSING_ERRORS="No errors found"
    fi

    # Correlation IDs - two-stage grep to avoid pipefail issues with empty results
    # Note: Create temp file first to ensure it exists, avoiding pipefail with empty grep results
    REPORT_CORRELATION_IDS=""
    : > /tmp/report_correlation_raw.txt  # Create/truncate temp file
    if grep -E "correlation_id=[a-f0-9-]+" "$FULL_LOG" >> /tmp/report_correlation_raw.txt 2>/dev/null; then
        # File has content - extract unique IDs
        REPORT_CORRELATION_IDS=$(sed 's/.*correlation_id=\([a-f0-9-]*\).*/\1/' /tmp/report_correlation_raw.txt 2>/dev/null | sort -u 2>/dev/null | head -20 2>/dev/null) || true
    fi
    if [ -z "$REPORT_CORRELATION_IDS" ]; then
        REPORT_CORRELATION_IDS="No correlation IDs found"
    fi

    # Assertion details - grep with context, || true for empty results
    # Note: -A (after context) is supported by GNU and BSD grep (including macOS)
    REPORT_ASSERTION_DETAILS=$(grep -E -A 10 "PASSED|FAILED|AssertionError" "$TEST_LOG" 2>/dev/null | head -50 || true)
    [ -z "$REPORT_ASSERTION_DETAILS" ] && REPORT_ASSERTION_DETAILS="No assertion details found"

    # Callback logs - grep with context, || true for empty results
    # Note: -A (after context) is supported by GNU and BSD grep (including macOS)
    REPORT_CALLBACK_LOGS=$(grep -E -A 5 "callback invoked|parsed successfully|processed successfully" "$FULL_LOG" 2>/dev/null | head -50 || true)
    [ -z "$REPORT_CALLBACK_LOGS" ] && REPORT_CALLBACK_LOGS="No callback logs found"

    cat > "$REPORT_FILE" << EOF
# E2E Test Diagnostic Report
**Date**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Test**: ${TEST_NAME}
**Result**: ${TEST_RESULT}

---

## Summary

**Identified Scenario**: ${SCENARIO}
**Root Cause**: ${ROOT_CAUSE}
**Recommended Fix**: ${RECOMMENDED_FIX}

---

## Container Startup Status

### Required Startup Messages
- [$(grep -qi "container wiring complete" "$STARTUP_LOG" && echo "x" || echo " ")] Container wiring complete
- [$(grep -qi "HandlerNodeIntrospected resolved" "$STARTUP_LOG" && echo "x" || echo " ")] HandlerNodeIntrospected resolved
- [$(grep -qi "dispatcher created and wired" "$STARTUP_LOG" && echo "x" || echo " ")] Dispatcher created and wired
- [$(grep -qi "consumer started successfully" "$STARTUP_LOG" && echo "x" || echo " ")] Consumer started successfully
- [$(grep -qi "ONEX Runtime Kernel" "$STARTUP_LOG" && echo "x" || echo " ")] Runtime kernel banner displayed

### Startup Errors
\`\`\`
${REPORT_STARTUP_ERRORS}
\`\`\`

---

## Event Processing Pipeline Status

### Pipeline Checkpoints
- [$([ "$CALLBACK_INVOKED" = true ] && echo "x" || echo " ")] Callback invoked (message received)
- [$([ "$MESSAGE_PARSED" = true ] && echo "x" || echo " ")] Message parsed
- [$([ "$VALIDATION_ATTEMPTED" = true ] && echo "x" || echo " ")] Validation attempted
- [$([ "$VALIDATION_SUCCEEDED" = true ] && echo "x" || echo " ")] Validation succeeded
- [$([ "$ENVELOPE_CREATED" = true ] && echo "x" || echo " ")] Envelope created
- [$([ "$DISPATCHER_ROUTED" = true ] && echo "x" || echo " ")] Dispatcher routing attempted
- [$([ "$HANDLER_SUCCEEDED" = true ] && echo "x" || echo " ")] Handler execution succeeded

### Processing Errors
\`\`\`
${REPORT_PROCESSING_ERRORS}
\`\`\`

---

## Correlation IDs
\`\`\`
${REPORT_CORRELATION_IDS}
\`\`\`

---

## Evidence

### Startup Logs (Last 50 Lines)
\`\`\`
$(tail -50 "$STARTUP_LOG")
\`\`\`

### Test Output (Assertion Details)
\`\`\`
${REPORT_ASSERTION_DETAILS}
\`\`\`

### Event Processing Logs (Callback Execution)
\`\`\`
${REPORT_CALLBACK_LOGS}
\`\`\`

---

## Database State

### Recent Projections
\`\`\`
$(docker compose -f "$COMPOSE_FILE" exec -T postgres psql -U postgres -d omnibase_infra -c "SELECT entity_id, node_type, registration_phase, created_at FROM registration_projections ORDER BY created_at DESC LIMIT 10;" 2>&1 || echo "ERROR: Failed to query registration_projections table.
Possible causes:
  - PostgreSQL container 'postgres' is not running (check: docker compose ps postgres)
  - Database 'omnibase_infra' does not exist
  - Table 'registration_projections' does not exist (run migrations)
  - Network connectivity issue between containers
Run 'docker compose -f $COMPOSE_FILE logs postgres' for container logs.")
\`\`\`

---

## Recommended Fix

**Scenario**: ${SCENARIO}

**Fix Steps**:
${RECOMMENDED_FIX}

**Reference**: See \`docs/handoff/E2E_DIAGNOSTIC_MANUAL.md\` for detailed fix templates.

---

## Next Steps

1. Apply recommended fix to identified root cause
2. Rebuild container: \`docker compose -f docker/docker-compose.e2e.yml build runtime\`
3. Restart services: \`docker compose -f docker/docker-compose.e2e.yml up -d runtime\`
4. Rerun diagnostic: \`./scripts/diagnose_e2e.sh\`
5. Verify test passes consistently (run 3+ times)

---

**End of Report**
EOF

    log_success "Report generated at $REPORT_FILE"
    log_info "View with: cat $REPORT_FILE"
fi

# =============================================================================
# Final Summary
# =============================================================================

section_header "Diagnostic Summary"

echo ""
echo "Test Result:       ${TEST_RESULT}"
echo "Scenario:          ${SCENARIO}"
echo "Root Cause:        ${ROOT_CAUSE}"
echo ""
echo "Log Files:"
echo "  - Startup:       $STARTUP_LOG"
echo "  - Test Output:   $TEST_LOG"
echo "  - Full Logs:     $FULL_LOG"
[ "$FULL_REPORT" = true ] && echo "  - Report:        $REPORT_FILE"
echo ""
echo "Next Steps:"
echo "  1. Review logs above for specific errors"
echo "  2. Apply recommended fix"
echo "  3. Rerun with: ./scripts/diagnose_e2e.sh --rebuild"
echo ""

if [ "$TEST_RESULT" = "PASSED" ]; then
    log_success "All diagnostics complete - test PASSED ✓"
    exit 0
else
    log_warning "Diagnostics complete - test FAILED (see scenario above)"
    exit 1
fi
