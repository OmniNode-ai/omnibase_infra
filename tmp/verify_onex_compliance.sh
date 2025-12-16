#!/bin/bash
# Verify ONEX compliance fixes in KafkaEventBus

echo "🔍 ONEX Compliance Verification for KafkaEventBus"
echo "=================================================="
echo ""

FILE="src/omnibase_infra/event_bus/kafka_event_bus.py"

# Test 1: Error Context Usage
echo "✅ Test 1: ModelInfraErrorContext Usage"
ERROR_CONTEXTS=$(grep -c "ModelInfraErrorContext" $FILE)
echo "   Found $ERROR_CONTEXTS error contexts (expected: 9)"
if [ "$ERROR_CONTEXTS" -eq 9 ]; then
    echo "   ✓ PASSED"
else
    echo "   ✗ FAILED"
fi
echo ""

# Test 2: Required Fields in Error Contexts
echo "✅ Test 2: Error Context Required Fields"
CONTEXT_FIELDS=$(grep -A 5 "ModelInfraErrorContext" $FILE | grep -c -E "(transport_type|operation|target_name|correlation_id)")
echo "   Found $CONTEXT_FIELDS field assignments (expected: 36 = 9 contexts × 4 fields)"
if [ "$CONTEXT_FIELDS" -ge 36 ]; then
    echo "   ✓ PASSED"
else
    echo "   ✗ FAILED"
fi
echo ""

# Test 3: No Unsanitized Servers in Errors
echo "✅ Test 3: Sanitization in Error Raises"
UNSANITIZED=$(grep -n "bootstrap_servers=self._bootstrap_servers" $FILE | grep -v "AIOKafka" | wc -l)
echo "   Found $UNSANITIZED unsanitized server exposures in errors (expected: 0)"
if [ "$UNSANITIZED" -eq 0 ]; then
    echo "   ✓ PASSED"
else
    echo "   ✗ FAILED - Lines:"
    grep -n "bootstrap_servers=self._bootstrap_servers" $FILE | grep -v "AIOKafka"
fi
echo ""

# Test 4: Sanitized Servers Used
echo "✅ Test 4: Sanitized Servers Usage"
SANITIZED=$(grep -c "servers=sanitized_servers" $FILE)
echo "   Found $SANITIZED sanitized server usages (expected: 2)"
if [ "$SANITIZED" -eq 2 ]; then
    echo "   ✓ PASSED"
else
    echo "   ✗ FAILED"
fi
echo ""

# Test 5: Target Name Patterns
echo "✅ Test 5: Target Name Standardization"
BAD_TARGETS=$(grep -c 'target_name=f"kafka.{self._bootstrap_servers}"' $FILE)
echo "   Found $BAD_TARGETS target_name with bootstrap_servers (expected: 0)"
if [ "$BAD_TARGETS" -eq 0 ]; then
    echo "   ✓ PASSED"
else
    echo "   ✗ FAILED"
fi
echo ""

# Test 6: Correlation ID Propagation
echo "✅ Test 6: Correlation ID Propagation"
CORR_IDS=$(grep -A 5 "ModelInfraErrorContext" $FILE | grep -c "correlation_id=")
echo "   Found $CORR_IDS correlation_id assignments (expected: 9)"
if [ "$CORR_IDS" -eq 9 ]; then
    echo "   ✓ PASSED"
else
    echo "   ✗ FAILED"
fi
echo ""

# Test 7: Sanitization Method Exists
echo "✅ Test 7: Sanitization Method Implementation"
HAS_SANITIZE=$(grep -c "def _sanitize_bootstrap_servers" $FILE)
echo "   Found $HAS_SANITIZE sanitization method (expected: 1)"
if [ "$HAS_SANITIZE" -eq 1 ]; then
    echo "   ✓ PASSED"
else
    echo "   ✗ FAILED"
fi
echo ""

# Summary
echo "=================================================="
echo "📊 Compliance Summary:"
echo "   - Error contexts: $ERROR_CONTEXTS/9"
echo "   - Required fields: $CONTEXT_FIELDS/36"
echo "   - Unsanitized exposures: $UNSANITIZED/0 (lower is better)"
echo "   - Sanitized usages: $SANITIZED/2"
echo "   - Bad target_names: $BAD_TARGETS/0 (lower is better)"
echo "   - Correlation IDs: $CORR_IDS/9"
echo "   - Sanitization method: $HAS_SANITIZE/1"
echo ""

if [ "$ERROR_CONTEXTS" -eq 9 ] && [ "$CONTEXT_FIELDS" -ge 36 ] && [ "$UNSANITIZED" -eq 0 ] && [ "$SANITIZED" -eq 2 ] && [ "$BAD_TARGETS" -eq 0 ] && [ "$CORR_IDS" -eq 9 ] && [ "$HAS_SANITIZE" -eq 1 ]; then
    echo "🎉 ALL TESTS PASSED - ONEX COMPLIANT"
    exit 0
else
    echo "❌ SOME TESTS FAILED - REVIEW NEEDED"
    exit 1
fi
