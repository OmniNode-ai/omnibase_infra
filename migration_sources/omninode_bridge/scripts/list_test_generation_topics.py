#!/usr/bin/env python3
"""List test generation Kafka topics with partition details."""


from kafka import KafkaConsumer
from kafka.admin import KafkaAdminClient


def list_topics():
    """List test generation topics with details."""

    admin_client = KafkaAdminClient(
        bootstrap_servers="192.168.86.200:9092", client_id="test-generation-lister"
    )

    # Get consumer for metadata
    consumer = KafkaConsumer(
        bootstrap_servers="192.168.86.200:9092", client_id="test-generation-consumer"
    )

    # Topics to check
    topic_names = [
        "dev.omninode-bridge.test-generation.started.v1",
        "dev.omninode-bridge.test-generation.completed.v1",
        "dev.omninode-bridge.test-generation.failed.v1",
    ]

    print("=" * 70)
    print("TEST GENERATION TOPICS")
    print("=" * 70)

    for topic_name in topic_names:
        # Get partition info from consumer metadata
        partitions = consumer.partitions_for_topic(topic_name)

        if partitions is None:
            print(f"\n❌ Topic: {topic_name}")
            print("   Status: Does not exist")
            continue

        print(f"\n✅ Topic: {topic_name}")
        print(f"   Partitions: {len(partitions)}")
        print(f"   Partition IDs: {sorted(partitions)}")

        # Expected retention
        expected_retention = {
            "started": "7 days",
            "completed": "7 days",
            "failed": "30 days",
        }

        topic_type = None
        for key in expected_retention:
            if key in topic_name:
                topic_type = key
                break

        if topic_type:
            print(f"   Expected Retention: {expected_retention[topic_type]}")

    consumer.close()
    admin_client.close()

    print("\n" + "=" * 70)
    print("✅ Topic listing complete")
    print("=" * 70)


if __name__ == "__main__":
    list_topics()
