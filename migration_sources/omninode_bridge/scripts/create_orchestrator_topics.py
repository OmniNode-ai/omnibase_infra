#!/usr/bin/env python3
"""
Create all NodeBridgeOrchestrator Kafka topics with proper partitions.

This script creates all workflow event topics defined in EnumWorkflowEvent
with 1 partition and 1 replica to fix the 0-partition infrastructure issue.
"""

import subprocess
import sys

# All orchestrator workflow topics
TOPICS = [
    "stamp-workflow-started",
    "stamp-workflow-completed",
    "stamp-workflow-failed",
    "workflow-step-completed",
    "onextree-intelligence-requested",
    "onextree-intelligence-received",
    "metadata-stamp-created",
    "blake3-hash-generated",
    "workflow-state-transition",
    "node-introspection",
    "registry-request-introspection",
    "node-heartbeat",
]

NAMESPACE = "dev"
PARTITIONS = 1
REPLICAS = 1


def create_topic(topic_slug: str) -> bool:
    """Create a Kafka topic with proper partitions."""
    full_topic = f"{NAMESPACE}.omninode_bridge.onex.evt.{topic_slug}.v1"

    cmd = [
        "docker",
        "exec",
        "omninode-bridge-redpanda",
        "rpk",
        "topic",
        "create",
        full_topic,
        "-p",
        str(PARTITIONS),
        "-r",
        str(REPLICAS),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr

        if "OK" in output:
            print(f"✓ Created: {full_topic}")
            return True
        elif "TOPIC_ALREADY_EXISTS" in output:
            print(f"⚠ Exists: {full_topic} (verifying partitions...)")
            # Verify it has partitions
            return verify_topic_partitions(full_topic)
        else:
            print(f"✗ Failed: {full_topic} - {output.strip()}")
            return False

    except Exception as e:
        print(f"✗ Error creating {full_topic}: {e}")
        return False


def verify_topic_partitions(topic_name: str) -> bool:
    """Verify a topic has non-zero partitions."""
    cmd = [
        "docker",
        "exec",
        "omninode-bridge-redpanda",
        "rpk",
        "topic",
        "describe",
        topic_name,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout

        # Look for PARTITIONS line
        for line in output.split("\n"):
            if "PARTITIONS" in line and line.strip().startswith("PARTITIONS"):
                parts = line.split()
                if len(parts) >= 2:
                    partition_count = parts[1]
                    if partition_count == "0":
                        print("  ⚠ Topic exists but has 0 partitions (CORRUPTED)")
                        return False
                    else:
                        print(f"  ✓ Topic has {partition_count} partition(s)")
                        return True

        print("  ⚠ Could not verify partition count")
        return False

    except Exception as e:
        print(f"  ✗ Error verifying {topic_name}: {e}")
        return False


def main():
    """Create all orchestrator topics."""
    print("=" * 80)
    print("Creating NodeBridgeOrchestrator Kafka Topics")
    print("=" * 80)
    print(f"Namespace: {NAMESPACE}")
    print(f"Partitions: {PARTITIONS}")
    print(f"Replicas: {REPLICAS}")
    print(f"Total topics: {len(TOPICS)}")
    print("=" * 80)
    print()

    created = 0
    failed = 0

    for topic_slug in TOPICS:
        if create_topic(topic_slug):
            created += 1
        else:
            failed += 1

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Success: {created}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {len(TOPICS)}")

    if failed > 0:
        print()
        print("⚠ Some topics failed to create. Check Redpanda logs:")
        print("  docker logs omninode-bridge-redpanda | tail -50")
        sys.exit(1)
    else:
        print()
        print("✅ All topics created successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
