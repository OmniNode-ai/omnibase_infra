#!/usr/bin/env python3
"""
Fix corrupted Redpanda topics using Kafka Admin API.
This script deletes and recreates topics showing broker -1 metadata corruption.
"""

import sys
import time

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, UnknownTopicOrPartitionError

BOOTSTRAP_SERVERS = "192.168.86.200:29092"
NAMESPACE = "dev"

# Topics showing broker -1 corruption
CORRUPTED_TOPICS = [
    "node-introspection",
    "stamp-workflow-completed",
    "stamp-workflow-failed",
    "workflow-state-transition",
    "workflow-step-completed",
]


def get_full_topic_name(topic_slug: str) -> str:
    """Convert topic slug to full topic name."""
    return f"{NAMESPACE}.omninode_bridge.onex.evt.{topic_slug}.v1"


def main():
    print("=" * 80)
    print("Redpanda Topic Remediation - Using Kafka Admin API")
    print("=" * 80)
    print(f"Bootstrap Servers: {BOOTSTRAP_SERVERS}")
    print(f"Topics to fix: {len(CORRUPTED_TOPICS)}")
    print("=" * 80)
    print()

    # Create admin client
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            client_id="topic-remediation-script",
            request_timeout_ms=30000,
        )
        print("✓ Connected to Kafka cluster")
    except Exception as e:
        print(f"✗ Failed to connect to Kafka: {e}")
        sys.exit(1)

    deleted = 0
    created = 0
    failed = 0

    # Step 1: Delete corrupted topics
    print()
    print("Step 1: Deleting corrupted topics")
    print("-" * 80)

    for topic_slug in CORRUPTED_TOPICS:
        full_topic = get_full_topic_name(topic_slug)
        print(f"Deleting: {full_topic}")

        try:
            result = admin_client.delete_topics([full_topic], timeout_ms=10000)
            # Wait for deletion to complete
            time.sleep(1)
            print("  ✓ Deleted successfully")
            deleted += 1
        except UnknownTopicOrPartitionError:
            print("  ⚠ Topic doesn't exist, skipping")
        except Exception as e:
            print(f"  ⚠ Delete failed: {e}")

    # Step 2: Wait for deletion to propagate
    print()
    print("Waiting for deletions to propagate...")
    time.sleep(2)

    # Step 3: Create topics with correct configuration
    print()
    print("Step 2: Creating topics with proper configuration")
    print("-" * 80)

    new_topics = []
    for topic_slug in CORRUPTED_TOPICS:
        full_topic = get_full_topic_name(topic_slug)
        new_topics.append(
            NewTopic(
                name=full_topic,
                num_partitions=1,
                replication_factor=1,
            )
        )

    try:
        result = admin_client.create_topics(new_topics, timeout_ms=10000)
        for topic_slug in CORRUPTED_TOPICS:
            full_topic = get_full_topic_name(topic_slug)
            print(f"Creating: {full_topic}")
            print("  ✓ Created successfully")
            created += 1
    except TopicAlreadyExistsError as e:
        print(f"  ⚠ Some topics already exist: {e}")
        # Try creating individually
        for new_topic in new_topics:
            try:
                admin_client.create_topics([new_topic], timeout_ms=5000)
                print(f"Creating: {new_topic.name}")
                print("  ✓ Created successfully")
                created += 1
            except TopicAlreadyExistsError:
                print(f"Creating: {new_topic.name}")
                print("  ⚠ Already exists")
            except Exception as e:
                print(f"Creating: {new_topic.name}")
                print(f"  ✗ Failed: {e}")
                failed += 1
    except Exception as e:
        print(f"  ✗ Batch creation failed: {e}")
        failed += len(CORRUPTED_TOPICS)

    # Step 3: Verify topics
    print()
    print("Step 3: Verifying topics")
    print("-" * 80)

    try:
        all_topics = admin_client.list_topics()
        for topic_slug in CORRUPTED_TOPICS:
            full_topic = get_full_topic_name(topic_slug)
            if full_topic in all_topics:
                print(f"✓ {full_topic} exists")
            else:
                print(f"✗ {full_topic} NOT FOUND")
                failed += 1
    except Exception as e:
        print(f"⚠ Verification failed: {e}")

    # Close admin client
    admin_client.close()

    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Deleted: {deleted}")
    print(f"Created: {created}")
    print(f"Failed: {failed}")
    print("=" * 80)

    if failed == 0:
        print()
        print("✅ All topics successfully remediated!")
        print()
        print("Run verification script to confirm:")
        print("  ./scripts/verify_topic_health.sh")
        sys.exit(0)
    else:
        print()
        print("⚠ Some topics failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
