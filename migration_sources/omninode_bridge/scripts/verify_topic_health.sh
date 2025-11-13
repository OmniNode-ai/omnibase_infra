#!/bin/bash
# Verify health of potentially corrupted Redpanda topics
# This script uses kcat to check topic metadata and test read/write operations

set -e

BOOTSTRAP_SERVERS="192.168.86.200:29092"
TOPICS=(
    "dev.omninode_bridge.onex.evt.node-introspection.v1"
    "dev.omninode_bridge.onex.evt.stamp-workflow-completed.v1"
    "dev.omninode_bridge.onex.evt.stamp-workflow-failed.v1"
    "dev.omninode_bridge.onex.evt.stamp-workflow-started.v1"
    "dev.omninode_bridge.onex.evt.workflow-state-transition.v1"
    "dev.omninode_bridge.onex.evt.workflow-step-completed.v1"
)

echo "================================================================================"
echo "Redpanda Topic Health Verification Report"
echo "================================================================================"
echo "Bootstrap Servers: $BOOTSTRAP_SERVERS"
echo "Topics to verify: ${#TOPICS[@]}"
echo "Timestamp: $(date)"
echo "================================================================================"
echo ""

HEALTHY=0
CORRUPTED=0

for topic in "${TOPICS[@]}"; do
    echo "Topic: $topic"
    echo "----------------------------------------"

    # Get metadata
    metadata=$(kcat -L -b "$BOOTSTRAP_SERVERS" -t "$topic" 2>&1)

    # Extract partition count
    partition_count=$(echo "$metadata" | grep "with.*partitions:" | grep -oE '[0-9]+ partitions' | grep -oE '[0-9]+')

    # Check broker ID
    broker_id=$(echo "$metadata" | grep "from broker" | grep -oE 'broker [0-9-]+' | grep -oE '[0-9-]+')

    echo "  Partition Count: $partition_count"
    echo "  Broker ID: $broker_id"

    # Test write capability
    test_message="{\"test\":\"health_check\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
    if echo "$test_message" | kcat -P -b "$BOOTSTRAP_SERVERS" -t "$topic" 2>&1 >/dev/null; then
        echo "  Write Test: ✓ SUCCESS"
        write_ok=true
    else
        echo "  Write Test: ✗ FAILED"
        write_ok=false
    fi

    # Test read capability
    if timeout 2 kcat -C -b "$BOOTSTRAP_SERVERS" -t "$topic" -e -o beginning -c 1 2>&1 >/dev/null; then
        echo "  Read Test: ✓ SUCCESS"
        read_ok=true
    else
        echo "  Read Test: ✗ FAILED"
        read_ok=false
    fi

    # Determine health status
    if [ "$partition_count" = "0" ] || [ "$broker_id" = "-1" ] || [ "$write_ok" = false ] || [ "$read_ok" = false ]; then
        echo "  Status: ⚠ CORRUPTED or UNHEALTHY"
        ((CORRUPTED++))

        # Show detailed metadata for debugging
        echo "  Full Metadata:"
        echo "$metadata" | sed 's/^/    /'
    else
        echo "  Status: ✓ HEALTHY"
        ((HEALTHY++))
    fi

    echo ""
done

echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo "✓ Healthy Topics: $HEALTHY"
echo "⚠ Corrupted/Unhealthy Topics: $CORRUPTED"
echo "Total: ${#TOPICS[@]}"
echo "================================================================================"

if [ $CORRUPTED -gt 0 ]; then
    echo ""
    echo "⚠ REMEDIATION REQUIRED"
    echo "Corrupted topics detected. Recommended actions:"
    echo "  1. Try deleting and recreating corrupted topics"
    echo "  2. Check Redpanda logs for errors"
    echo "  3. Consider restarting Redpanda if issues persist"
    exit 1
else
    echo ""
    echo "✅ All topics are healthy!"
    exit 0
fi
