#!/usr/bin/env python3
"""Test the Kafka hostname patch."""

import asyncio
import logging

# Import the patch (auto-applies)
from aiokafka import AIOKafkaProducer

logging.basicConfig(level=logging.INFO)


async def test_patched_connection():
    """Test aiokafka connection with hostname patch applied."""
    print("Testing aiokafka connection with hostname patch...")

    try:
        producer = AIOKafkaProducer(
            bootstrap_servers="localhost:29092",
            request_timeout_ms=5000,
            api_version="auto",
        )
        await producer.start()
        print("🎉 SUCCESS! Producer connected with hostname patch!")

        # Test sending a message
        await producer.send("test-topic", b"test-message")
        print("📤 Test message sent successfully!")

        await producer.stop()
        return True
    except Exception as e:
        print(f"❌ Producer failed even with patch: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_patched_connection())
    if success:
        print("✅ Hostname patch successfully fixes the connection issue!")
    else:
        print("❌ Hostname patch did not resolve the issue.")
