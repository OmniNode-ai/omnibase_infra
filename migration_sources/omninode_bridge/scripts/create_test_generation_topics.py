#!/usr/bin/env python3
"""
Create Kafka topics for test generation events.

Topics:
- dev.omninode-bridge.test-generation.started.v1
- dev.omninode-bridge.test-generation.completed.v1
- dev.omninode-bridge.test-generation.failed.v1
"""

import sys

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError


def create_test_generation_topics():
    """Create test generation Kafka topics."""

    # Connect to Kafka
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers="192.168.86.200:9092",
            client_id="test-generation-topics-creator",
        )
        print("✅ Connected to Kafka at 192.168.86.200:9092")
    except Exception as e:
        print(f"❌ Failed to connect to Kafka: {e}")
        sys.exit(1)

    # Define topics
    topics = [
        NewTopic(
            name="dev.omninode-bridge.test-generation.started.v1",
            num_partitions=3,
            replication_factor=1,
            topic_configs={"retention.ms": "604800000"},  # 7 days
        ),
        NewTopic(
            name="dev.omninode-bridge.test-generation.completed.v1",
            num_partitions=3,
            replication_factor=1,
            topic_configs={"retention.ms": "604800000"},  # 7 days
        ),
        NewTopic(
            name="dev.omninode-bridge.test-generation.failed.v1",
            num_partitions=3,
            replication_factor=1,
            topic_configs={"retention.ms": "2592000000"},  # 30 days
        ),
    ]

    # Create topics
    created_topics = []
    existing_topics = []
    failed_topics = []

    for topic in topics:
        try:
            admin_client.create_topics(new_topics=[topic], validate_only=False)
            created_topics.append(topic.name)
            print(f"✅ Created topic: {topic.name}")
            print(f"   - Partitions: {topic.num_partitions}")
            print(f"   - Replication: {topic.replication_factor}")
            print(f"   - Retention: {topic.topic_configs['retention.ms']}ms")
        except TopicAlreadyExistsError:
            existing_topics.append(topic.name)
            print(f" Topic already exists: {topic.name}")
        except Exception as e:
            failed_topics.append((topic.name, str(e)))
            print(f"❌ Failed to create topic {topic.name}: {e}")

    # Close admin client
    admin_client.close()

    # Summary
    print("\n" + "=" * 60)
    print("TOPIC CREATION SUMMARY")
    print("=" * 60)
    print(f"✅ Created: {len(created_topics)} topics")
    for topic in created_topics:
        print(f"   - {topic}")

    if existing_topics:
        print(f"\n Already existed: {len(existing_topics)} topics")
        for topic in existing_topics:
            print(f"   - {topic}")

    if failed_topics:
        print(f"\n❌ Failed: {len(failed_topics)} topics")
        for topic, error in failed_topics:
            print(f"   - {topic}: {error}")
        sys.exit(1)

    print("\n✅ Test generation topics ready!")
    return 0


if __name__ == "__main__":
    sys.exit(create_test_generation_topics())
