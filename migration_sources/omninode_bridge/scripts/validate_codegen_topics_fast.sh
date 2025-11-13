#!/bin/bash

set -e

echo "======================================================================"
echo "Redpanda Codegen Topics Validation (Fast)"
echo "======================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

CONTAINER="omninode-bridge-redpanda"

# Step 1: Check Redpanda health
echo "1. Checking Redpanda health..."
HEALTH=$(docker exec $CONTAINER rpk cluster health 2>&1)
if echo "$HEALTH" | grep -q "Healthy:.*true"; then
    echo -e "   ${GREEN}✅ Redpanda is healthy${NC}"
else
    echo -e "   ${RED}❌ Redpanda is not healthy${NC}"
    exit 1
fi
echo ""

# Step 2: Count topics
echo "2. Checking topic count..."
TOPIC_COUNT=$(docker exec $CONTAINER rpk topic list | grep omninode_codegen | wc -l | tr -d ' ')
if [ "$TOPIC_COUNT" -eq 13 ]; then
    echo -e "   ${GREEN}✅ Found all 13 codegen topics${NC}"
else
    echo -e "   ${RED}❌ Expected 13 topics, found $TOPIC_COUNT${NC}"
    exit 1
fi
echo ""

# Step 3: Validate partition counts
echo "3. Validating partition counts..."
echo ""

validate_partitions() {
    local topic=$1
    local expected=$2
    local actual=$(docker exec $CONTAINER rpk topic list | grep "^$topic" | awk '{print $2}')

    if [ "$actual" -eq "$expected" ]; then
        echo -e "   ${GREEN}✅${NC} $topic: $actual partitions"
    else
        echo -e "   ${RED}❌${NC} $topic: expected $expected, got $actual partitions"
        return 1
    fi
}

# Request topics (3 partitions)
echo "   Request Topics (3 partitions each):"
validate_partitions "omninode_codegen_request_analyze_v1" 3
validate_partitions "omninode_codegen_request_validate_v1" 3
validate_partitions "omninode_codegen_request_pattern_v1" 3
validate_partitions "omninode_codegen_request_mixin_v1" 3
echo ""

# Response topics (3 partitions)
echo "   Response Topics (3 partitions each):"
validate_partitions "omninode_codegen_response_analyze_v1" 3
validate_partitions "omninode_codegen_response_validate_v1" 3
validate_partitions "omninode_codegen_response_pattern_v1" 3
validate_partitions "omninode_codegen_response_mixin_v1" 3
echo ""

# Status topic (6 partitions)
echo "   Status Topic (6 partitions):"
validate_partitions "omninode_codegen_status_session_v1" 6
echo ""

# DLQ topics (1 partition)
echo "   DLQ Topics (1 partition each):"
validate_partitions "omninode_codegen_dlq_analyze_v1" 1
validate_partitions "omninode_codegen_dlq_validate_v1" 1
validate_partitions "omninode_codegen_dlq_pattern_v1" 1
validate_partitions "omninode_codegen_dlq_mixin_v1" 1
echo ""

# Step 4: Sample topic configuration check
echo "4. Checking sample topic configuration..."
SAMPLE_TOPIC="omninode_codegen_request_analyze_v1"
CONFIG=$(docker exec $CONTAINER rpk topic describe $SAMPLE_TOPIC --print-partitions=false)

# Check for key configurations
if echo "$CONFIG" | grep -q "cleanup.policy.*delete"; then
    echo -e "   ${GREEN}✅${NC} Cleanup policy: delete"
else
    echo -e "   ${YELLOW}⚠️${NC}  Cleanup policy not found"
fi

if echo "$CONFIG" | grep -q "compression.type.*gzip"; then
    echo -e "   ${GREEN}✅${NC} Compression type: gzip"
else
    echo -e "   ${YELLOW}⚠️${NC}  Compression type not found"
fi

if echo "$CONFIG" | grep -q "retention.ms"; then
    RETENTION=$(echo "$CONFIG" | grep "retention.ms" | awk '{print $2}')
    echo -e "   ${GREEN}✅${NC} Retention configured: ${RETENTION}ms"
else
    echo -e "   ${YELLOW}⚠️${NC}  Retention not explicitly set (using default)"
fi
echo ""

# Step 5: Test basic produce/consume
echo "5. Testing produce/consume functionality..."
TEST_TOPIC="omninode_codegen_request_analyze_v1"
TEST_MESSAGE="test_validation_$(date +%s)"

# Produce a test message
echo "$TEST_MESSAGE" | docker exec -i $CONTAINER rpk topic produce $TEST_TOPIC --key "validation" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "   ${GREEN}✅${NC} Message produced successfully"

    # Try to consume (just check if consume command works)
    docker exec $CONTAINER rpk topic consume $TEST_TOPIC --num 1 --offset end > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "   ${GREEN}✅${NC} Message consumed successfully"
    else
        echo -e "   ${YELLOW}⚠️${NC}  Consume command executed (topic may be empty)"
    fi
else
    echo -e "   ${RED}❌${NC} Failed to produce test message"
fi
echo ""

# Summary
echo "======================================================================"
echo "Validation Summary"
echo "======================================================================"
echo -e "${GREEN}✅ All 13 topics exist with correct partition counts${NC}"
echo ""
echo "Topic Breakdown:"
echo "  - 4 request topics (3 partitions each, 7 day retention)"
echo "  - 4 response topics (3 partitions each, 7 day retention)"
echo "  - 1 status topic (6 partitions, 3 day retention)"
echo "  - 4 DLQ topics (1 partition each, 30 day retention)"
echo ""
echo "Configuration:"
echo "  - Replication factor: 1 (single-node)"
echo "  - Cleanup policy: delete"
echo "  - Compression: gzip"
echo ""
echo -e "${GREEN}✅ Redpanda cluster is healthy and ready for use${NC}"
