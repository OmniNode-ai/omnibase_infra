"""
Simple Kafka producer test without registry dependencies.

This script tests:
1. KafkaClient connection
2. Raw event publishing
3. Basic producer metrics
"""

import asyncio
import sys
from uuid import uuid4

from src.omninode_bridge.services.kafka_client import KafkaClient


async def test_simple_producer():
    """Test simple Kafka producer."""
    print("=" * 80)
    print("SIMPLE KAFKA PRODUCER TEST")
    print("=" * 80)

    # Initialize client
    print("\n[1/5] Initializing Kafka client...")
    client = KafkaClient(bootstrap_servers="localhost:19092")

    try:
        # Connect to Kafka
        print("[2/5] Connecting to Kafka...")
        await client.connect()

        if not client.is_connected:
            print("❌ Failed to connect to Kafka")
            return False

        print("✅ Connected to Kafka successfully")

        # Create test event data (simple dict, no Pydantic model)
        print("\n[3/5] Creating test event...")
        correlation_id = str(uuid4())
        session_id = str(uuid4())

        event_data = {
            "correlation_id": correlation_id,
            "session_id": session_id,
            "prd_content": "# Test PRD\n\nThis is a test PRD for validating Kafka publishing.",
            "analysis_type": "full",
            "test_run": True,
        }

        print(f"  - Correlation ID: {correlation_id}")
        print(f"  - Session ID: {session_id}")

        # Publish raw event (no envelope wrapping)
        print("\n[4/5] Publishing raw event...")
        success = await client.publish_raw_event(
            topic="omninode_codegen_request_analyze_v1",
            data=event_data,
            key=correlation_id,
        )

        if success:
            print("✅ Event published successfully")
        else:
            print("❌ Failed to publish event")
            return False

        # Get basic metrics
        print("\n[5/5] Checking resilience metrics...")
        metrics = await client.get_resilience_metrics()

        print("\n📊 Connection Status:")
        conn = metrics["connection_status"]
        print(f"  Connected: {conn['connected']}")
        print(f"  Bootstrap Servers: {conn['bootstrap_servers']}")

        print("\n📊 Failure Statistics:")
        stats = metrics["failure_statistics"]
        print(f"  Total Failures: {stats['total_failures']}")
        print(f"  Recent Failures (1h): {stats['recent_failures_1h']}")
        print(f"  Active Retries: {stats['active_retries']}")

        # Validation
        print("\n[6/6] Validation...")
        validation_passed = True

        if not conn["connected"]:
            print("❌ Client not connected")
            validation_passed = False
        else:
            print("✅ Client connected")

        if stats["total_failures"] > 0:
            print(f"⚠️  {stats['total_failures']} failures detected")
        else:
            print("✅ No failures")

        print("\n" + "=" * 80)
        if validation_passed:
            print("✅ SIMPLE PRODUCER TEST PASSED")
        else:
            print("❌ SIMPLE PRODUCER TEST FAILED")
        print("=" * 80)

        return validation_passed

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Disconnect
        print("\nDisconnecting from Kafka...")
        await client.disconnect()
        print("✅ Disconnected")


async def main():
    """Main test execution."""
    success = await test_simple_producer()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
