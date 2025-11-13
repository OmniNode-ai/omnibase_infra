#!/bin/bash
# Fix corrupted Redpanda topics with broker -1 metadata issue
# This script deletes and recreates topics showing broker -1 in metadata

set -e

REMOTE_HOST="192.168.86.200"
CONTAINER_NAME="omninode-bridge-redpanda"
NAMESPACE="dev"

# Topics showing broker -1 corruption
CORRUPTED_TOPICS=(
    "node-introspection"
    "stamp-workflow-completed"
    "stamp-workflow-failed"
    "workflow-state-transition"
    "workflow-step-completed"
)

echo "================================================================================"
echo "Redpanda Topic Remediation - Broker -1 Metadata Corruption Fix"
echo "================================================================================"
echo "Remote Host: $REMOTE_HOST"
echo "Container: $CONTAINER_NAME"
echo "Namespace: $NAMESPACE"
echo "Corrupted Topics: ${#CORRUPTED_TOPICS[@]}"
echo "================================================================================"
echo ""

# Function to check if we can access the remote container
check_remote_access() {
    echo "Checking remote access..."
    if ssh "$REMOTE_HOST" "docker ps -q -f name=$CONTAINER_NAME" > /dev/null 2>&1; then
        echo "✓ Remote access confirmed"
        return 0
    else
        echo "✗ Cannot access remote container"
        echo "  Please ensure:"
        echo "  1. SSH access to $REMOTE_HOST is configured"
        echo "  2. Container $CONTAINER_NAME is running"
        return 1
    fi
}

# Function to delete a topic
delete_topic() {
    local topic_slug=$1
    local full_topic="${NAMESPACE}.omninode_bridge.onex.evt.${topic_slug}.v1"

    echo "Deleting: $full_topic"

    if ssh "$REMOTE_HOST" "docker exec $CONTAINER_NAME rpk topic delete $full_topic" 2>&1 | grep -q "OK"; then
        echo "  ✓ Deleted successfully"
        return 0
    else
        echo "  ⚠ Delete failed or topic didn't exist"
        return 1
    fi
}

# Function to create a topic
create_topic() {
    local topic_slug=$1
    local full_topic="${NAMESPACE}.omninode_bridge.onex.evt.${topic_slug}.v1"

    echo "Creating: $full_topic"

    if ssh "$REMOTE_HOST" "docker exec $CONTAINER_NAME rpk topic create $full_topic -p 1 -r 1" 2>&1 | grep -q "OK"; then
        echo "  ✓ Created successfully"
        return 0
    else
        echo "  ✗ Creation failed"
        return 1
    fi
}

# Function to verify topic health
verify_topic() {
    local topic_slug=$1
    local full_topic="${NAMESPACE}.omninode_bridge.onex.evt.${topic_slug}.v1"

    echo "Verifying: $full_topic"

    local describe_output=$(ssh "$REMOTE_HOST" "docker exec $CONTAINER_NAME rpk topic describe $full_topic" 2>&1)

    # Check for partition count
    if echo "$describe_output" | grep -q "PARTITIONS.*1"; then
        echo "  ✓ Has 1 partition"
        return 0
    else
        echo "  ✗ Verification failed"
        echo "$describe_output" | sed 's/^/    /'
        return 1
    fi
}

# Main remediation workflow
main() {
    if ! check_remote_access; then
        echo ""
        echo "================================================================================"
        echo "ALTERNATIVE: Manual Remediation Steps"
        echo "================================================================================"
        echo "If SSH access is not available, run these commands on $REMOTE_HOST:"
        echo ""
        for topic_slug in "${CORRUPTED_TOPICS[@]}"; do
            full_topic="${NAMESPACE}.omninode_bridge.onex.evt.${topic_slug}.v1"
            echo "# Fix $full_topic"
            echo "docker exec $CONTAINER_NAME rpk topic delete $full_topic"
            echo "docker exec $CONTAINER_NAME rpk topic create $full_topic -p 1 -r 1"
            echo ""
        done
        exit 1
    fi

    echo ""
    echo "Step 1: Deleting corrupted topics"
    echo "----------------------------------------"
    for topic_slug in "${CORRUPTED_TOPICS[@]}"; do
        delete_topic "$topic_slug"
    done

    echo ""
    echo "Step 2: Recreating topics with correct configuration"
    echo "----------------------------------------"
    for topic_slug in "${CORRUPTED_TOPICS[@]}"; do
        create_topic "$topic_slug"
    done

    echo ""
    echo "Step 3: Verifying topic health"
    echo "----------------------------------------"
    FAILED=0
    for topic_slug in "${CORRUPTED_TOPICS[@]}"; do
        if ! verify_topic "$topic_slug"; then
            ((FAILED++))
        fi
    done

    echo ""
    echo "================================================================================"
    echo "Remediation Complete"
    echo "================================================================================"
    if [ $FAILED -eq 0 ]; then
        echo "✅ All topics successfully recreated and verified!"
        echo ""
        echo "You can verify using:"
        echo "  ./scripts/verify_topic_health.sh"
        exit 0
    else
        echo "⚠ $FAILED topics failed verification"
        echo "Please check Redpanda logs and retry"
        exit 1
    fi
}

main
