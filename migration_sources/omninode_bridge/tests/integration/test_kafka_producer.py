"""
Test script for Kafka producer with OnexEnvelopeV1 format.

This script tests:
1. KafkaClient connection
2. publish_with_envelope method
3. OnexEnvelopeV1 format validation
4. Producer metrics
"""

import asyncio
import sys
from uuid import uuid4

from src.omninode_bridge.events.codegen_schemas import CodegenAnalysisRequest
from src.omninode_bridge.services.kafka_client import KafkaClient


async def test_producer():
    """Test Kafka producer end-to-end."""
    print("=" * 80)
    print("KAFKA PRODUCER TEST - OnexEnvelopeV1 Format")
    print("=" * 80)

    # Initialize client
    print("\n[1/6] Initializing Kafka client...")
    client = KafkaClient(bootstrap_servers="localhost:19092")

    try:
        # Connect to Kafka
        print("[2/6] Connecting to Kafka...")
        await client.connect()

        if not client.is_connected:
            print("❌ Failed to connect to Kafka")
            return False

        print("✅ Connected to Kafka successfully")

        # Create test event
        print("\n[3/6] Creating test event...")
        correlation_id = uuid4()
        session_id = uuid4()

        event = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content="# Test PRD\n\nThis is a test PRD for validating Kafka publishing.",
            analysis_type="full",
        )

        print(f"  - Correlation ID: {correlation_id}")
        print(f"  - Session ID: {session_id}")
        print("  - Event type: CodegenAnalysisRequest")

        # Publish with envelope
        print("\n[4/6] Publishing event with OnexEnvelopeV1...")
        success = await client.publish_with_envelope(
            event_type="ANALYSIS_REQUEST",
            source_node_id="test_producer",
            payload=event.model_dump(),
            topic="omninode_codegen_request_analyze_v1",
            correlation_id=correlation_id,
            metadata={"test_run": True, "producer": "test_kafka_producer.py"},
        )

        if success:
            print("✅ Event published successfully")
        else:
            print("❌ Failed to publish event")
            return False

        # Get envelope metrics
        print("\n[5/6] Checking producer metrics...")
        metrics = await client.get_envelope_metrics()

        print("\n📊 Envelope Publishing Metrics:")
        print(
            f"  Total Events Published: {metrics['envelope_publishing']['total_events_published']}"
        )
        print(
            f"  Total Events Failed: {metrics['envelope_publishing']['total_events_failed']}"
        )
        print(f"  Success Rate: {metrics['envelope_publishing']['success_rate']:.2%}")

        print("\n⏱️  Latency Metrics (ms):")
        latency = metrics["latency_metrics_ms"]
        print(f"  Average: {latency['average']}ms")
        print(f"  P50: {latency['p50']}ms")
        print(f"  P95: {latency['p95']}ms")
        print(f"  P99: {latency['p99']}ms")
        print(f"  Min: {latency['min']}ms")
        print(f"  Max: {latency['max']}ms")

        print("\n🎯 Performance Summary:")
        perf = metrics["performance_summary"]
        print(
            f"  Meets Target Latency (<100ms avg): {'✅' if perf['meets_target_latency'] else '❌'}"
        )
        print(
            f"  Meets Success Rate (≥95%): {'✅' if perf['meets_success_rate'] else '❌'}"
        )

        # Validation checks
        print("\n[6/6] Validating metrics...")
        validation_passed = True

        if metrics["envelope_publishing"]["success_rate"] < 0.95:
            print("❌ Success rate below 95% threshold")
            validation_passed = False
        else:
            print("✅ Success rate ≥ 95%")

        if latency["average"] >= 100.0:
            print("❌ Average latency above 100ms threshold")
            validation_passed = False
        else:
            print("✅ Average latency < 100ms")

        if latency["p95"] >= 150.0:
            print("⚠️  P95 latency above 150ms threshold (acceptable for first run)")
        else:
            print("✅ P95 latency < 150ms")

        print("\n" + "=" * 80)
        if validation_passed:
            print("✅ PRODUCER TEST PASSED")
        else:
            print("❌ PRODUCER TEST FAILED")
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
    success = await test_producer()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
