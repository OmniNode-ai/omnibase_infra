#!/usr/bin/env python3
"""Test KafkaClient with integrated hostname patch."""

import asyncio
import os
import sys

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from omninode_bridge.services.kafka_client import KafkaClient


async def test_kafka_client_connection():
    """Test KafkaClient connection with integrated hostname patch."""
    print("Testing KafkaClient with integrated hostname patch...")

    try:
        # Create KafkaClient instance
        client = KafkaClient(
            bootstrap_servers="localhost:29092",
            timeout_seconds=5,
        )

        # Test connection
        await client.connect()
        print("✅ KafkaClient connected successfully!")

        # Test basic operations
        topics = await client.list_topics()
        print(f"📋 Topics found: {len(topics)}")
        for topic in topics[:5]:  # Show first 5 topics
            print(f"   - {topic}")

        # Test topic creation
        test_topic = "test-hostname-patch-topic"
        created = await client.create_topic(test_topic)
        print(f"📝 Topic creation result: {created}")

        await client.disconnect()
        print("🔌 KafkaClient disconnected successfully!")
        return True

    except Exception as e:
        print(f"❌ KafkaClient test failed: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_kafka_client_connection())
    if success:
        print("🎉 KafkaClient hostname patch integration successful!")
        sys.exit(0)
    else:
        print("❌ KafkaClient hostname patch integration failed!")
        sys.exit(1)
