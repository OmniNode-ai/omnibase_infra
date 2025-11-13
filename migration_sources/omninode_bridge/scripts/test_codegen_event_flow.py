#!/usr/bin/env python3
"""
Test Script for Contract-First Code Generation Event Flow

Tests the complete event infrastructure including:
1. Event schema validation (Pydantic v2)
2. Event publishing to Kafka topics
3. Event consumption and verification
4. Dead letter queue (DLQ) routing
5. Correlation ID tracking across events

Usage:
    # Start Docker services first
    docker compose -f deployment/docker-compose.yml up -d

    # Run test script
    poetry run python scripts/test_codegen_event_flow.py
"""

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omninode_bridge.events.codegen_schemas import (
    TOPIC_CODEGEN_REQUEST_ANALYZE,
    TOPIC_CODEGEN_REQUEST_ANALYZE_DLQ,
    TOPIC_CODEGEN_REQUEST_MIXIN,
    TOPIC_CODEGEN_REQUEST_PATTERN,
    TOPIC_CODEGEN_REQUEST_VALIDATE,
    TOPIC_CODEGEN_RESPONSE_ANALYZE,
    TOPIC_CODEGEN_RESPONSE_MIXIN,
    TOPIC_CODEGEN_RESPONSE_PATTERN,
    TOPIC_CODEGEN_RESPONSE_VALIDATE,
    TOPIC_CODEGEN_STATUS_SESSION,
    CodegenAnalysisRequest,
    CodegenMixinRequest,
    CodegenPatternRequest,
    CodegenPatternResponse,
    CodegenStatusEvent,
    CodegenValidationRequest,
)
from omninode_bridge.events.enums import (
    EnumAnalysisType,
    EnumNodeType,
    EnumSessionStatus,
    EnumValidationType,
)
from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import (
    ModelOnexEnvelopeV1,
)
from omninode_bridge.services.kafka_client import KafkaClient


class CodegenEventFlowTester:
    """Test harness for code generation event infrastructure."""

    def __init__(self):
        """Initialize test harness with Kafka client."""
        self.kafka_client = KafkaClient()
        self.correlation_id = uuid4()
        self.session_id = uuid4()
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
        }

    async def setup(self):
        """Setup test environment."""
        print("=" * 80)
        print("Setting up test environment...")
        print("=" * 80)

        try:
            await self.kafka_client.connect()
            print("✓ Connected to Kafka")

            # Verify topics exist
            topics = await self.kafka_client.list_topics()
            required_topics = [
                TOPIC_CODEGEN_REQUEST_ANALYZE,
                TOPIC_CODEGEN_RESPONSE_ANALYZE,
                TOPIC_CODEGEN_REQUEST_VALIDATE,
                TOPIC_CODEGEN_RESPONSE_VALIDATE,
                TOPIC_CODEGEN_REQUEST_PATTERN,
                TOPIC_CODEGEN_RESPONSE_PATTERN,
                TOPIC_CODEGEN_REQUEST_MIXIN,
                TOPIC_CODEGEN_RESPONSE_MIXIN,
                TOPIC_CODEGEN_STATUS_SESSION,
                TOPIC_CODEGEN_REQUEST_ANALYZE_DLQ,
            ]

            missing_topics = [t for t in required_topics if t not in topics]
            if missing_topics:
                print(f"⚠️  Missing topics: {missing_topics}")
                print("Run: poetry run python scripts/create_codegen_kafka_topics.py")
                return False

            print(f"✓ All {len(required_topics)} required topics exist")
            return True

        except Exception as e:
            print(f"✗ Setup failed: {e}")
            return False

    async def teardown(self):
        """Cleanup test environment."""
        print("\n" + "=" * 80)
        print("Tearing down test environment...")
        print("=" * 80)

        try:
            await self.kafka_client.disconnect()
            print("✓ Disconnected from Kafka")
        except Exception as e:
            print(f"⚠️  Teardown warning: {e}")

    def record_test_result(self, test_name: str, passed: bool, error: str = None):
        """Record test result for summary."""
        self.test_results["total_tests"] += 1
        if passed:
            self.test_results["passed"] += 1
            print(f"✓ {test_name}")
        else:
            self.test_results["failed"] += 1
            print(f"✗ {test_name}")
            if error:
                self.test_results["errors"].append({"test": test_name, "error": error})
                print(f"  Error: {error}")

    async def test_event_schema_validation(self):
        """Test 1: Event schema validation with Pydantic v2."""
        print("\n" + "=" * 80)
        print("Test 1: Event Schema Validation")
        print("=" * 80)

        # Test analysis request
        try:
            request = CodegenAnalysisRequest(
                correlation_id=self.correlation_id,
                session_id=self.session_id,
                prd_content="# Sample PRD\n\nBuild a data processing node...",
                analysis_type=EnumAnalysisType.FULL,
            )
            assert request.correlation_id == self.correlation_id
            assert request.analysis_type == EnumAnalysisType.FULL
            self.record_test_result("CodegenAnalysisRequest validation", True)
        except Exception as e:
            self.record_test_result("CodegenAnalysisRequest validation", False, str(e))

        # Test validation request
        try:
            request = CodegenValidationRequest(
                correlation_id=self.correlation_id,
                session_id=self.session_id,
                code_content="class MyNode(NodeEffect): pass",
                node_type=EnumNodeType.EFFECT,
                validation_type=EnumValidationType.FULL,
            )
            assert request.node_type == EnumNodeType.EFFECT
            self.record_test_result("CodegenValidationRequest validation", True)
        except Exception as e:
            self.record_test_result(
                "CodegenValidationRequest validation", False, str(e)
            )

        # Test pattern request
        try:
            request = CodegenPatternRequest(
                correlation_id=self.correlation_id,
                session_id=self.session_id,
                node_description="Database CRUD operations",
                node_type=EnumNodeType.EFFECT,
                limit=5,
            )
            assert request.limit == 5
            self.record_test_result("CodegenPatternRequest validation", True)
        except Exception as e:
            self.record_test_result("CodegenPatternRequest validation", False, str(e))

        # Test mixin request
        try:
            request = CodegenMixinRequest(
                correlation_id=self.correlation_id,
                session_id=self.session_id,
                requirements=["caching", "retry logic", "circuit breaker"],
                node_type=EnumNodeType.EFFECT,
            )
            assert len(request.requirements) == 3
            self.record_test_result("CodegenMixinRequest validation", True)
        except Exception as e:
            self.record_test_result("CodegenMixinRequest validation", False, str(e))

        # Test status event
        try:
            event = CodegenStatusEvent(
                session_id=self.session_id,
                status=EnumSessionStatus.PROCESSING,
                progress_percentage=50.0,
                message="Processing stage 2 of 4",
            )
            assert event.progress_percentage == 50.0
            self.record_test_result("CodegenStatusEvent validation", True)
        except Exception as e:
            self.record_test_result("CodegenStatusEvent validation", False, str(e))

    async def test_event_publishing(self):
        """Test 2: Event publishing to Kafka topics."""
        print("\n" + "=" * 80)
        print("Test 2: Event Publishing with OnexEnvelopeV1")
        print("=" * 80)

        # Test analysis request publishing
        try:
            request = CodegenAnalysisRequest(
                correlation_id=self.correlation_id,
                session_id=self.session_id,
                prd_content="# Test PRD",
                analysis_type=EnumAnalysisType.FULL,
            )

            success = await self.kafka_client.publish_with_envelope(
                event_type="CODEGEN_REQUEST_ANALYZE",
                source_node_id="test-script",
                payload=request.model_dump(),
                topic=TOPIC_CODEGEN_REQUEST_ANALYZE,
                correlation_id=self.correlation_id,
            )

            self.record_test_result(
                "Publish analysis request to Kafka",
                success,
                None if success else "Publishing failed",
            )
        except Exception as e:
            self.record_test_result("Publish analysis request to Kafka", False, str(e))

        # Test validation request publishing
        try:
            request = CodegenValidationRequest(
                correlation_id=self.correlation_id,
                session_id=self.session_id,
                code_content="class MyNode(NodeEffect): pass",
                node_type=EnumNodeType.EFFECT,
                validation_type=EnumValidationType.FULL,
            )

            success = await self.kafka_client.publish_with_envelope(
                event_type="CODEGEN_REQUEST_VALIDATE",
                source_node_id="test-script",
                payload=request.model_dump(),
                topic=TOPIC_CODEGEN_REQUEST_VALIDATE,
                correlation_id=self.correlation_id,
            )

            self.record_test_result(
                "Publish validation request to Kafka",
                success,
                None if success else "Publishing failed",
            )
        except Exception as e:
            self.record_test_result(
                "Publish validation request to Kafka", False, str(e)
            )

        # Test status event publishing
        try:
            event = CodegenStatusEvent(
                session_id=self.session_id,
                status=EnumSessionStatus.PROCESSING,
                progress_percentage=75.0,
                message="Test status update",
            )

            success = await self.kafka_client.publish_with_envelope(
                event_type="CODEGEN_STATUS_SESSION",
                source_node_id="test-script",
                payload=event.model_dump(),
                topic=TOPIC_CODEGEN_STATUS_SESSION,
                correlation_id=self.correlation_id,
            )

            self.record_test_result(
                "Publish status event to Kafka",
                success,
                None if success else "Publishing failed",
            )
        except Exception as e:
            self.record_test_result("Publish status event to Kafka", False, str(e))

    async def test_event_consumption(self):
        """Test 3: Event consumption and verification."""
        print("\n" + "=" * 80)
        print("Test 3: Event Consumption and Verification")
        print("=" * 80)

        # Test consuming analysis request
        try:
            messages = await self.kafka_client.consume_messages(
                topic=TOPIC_CODEGEN_REQUEST_ANALYZE,
                group_id="test-consumer-group",
                max_messages=1,
                timeout_ms=5000,
            )

            if messages:
                message = messages[0]
                envelope = ModelOnexEnvelopeV1.from_dict(message["value"])
                assert envelope.event_type == "CODEGEN_REQUEST_ANALYZE"
                assert envelope.correlation_id == self.correlation_id

                # Validate payload structure
                payload = envelope.payload
                assert "correlation_id" in payload
                assert "session_id" in payload
                assert "prd_content" in payload

                self.record_test_result("Consume and validate analysis request", True)
            else:
                self.record_test_result(
                    "Consume and validate analysis request",
                    False,
                    "No messages consumed",
                )
        except Exception as e:
            self.record_test_result(
                "Consume and validate analysis request", False, str(e)
            )

        # Test consuming status event
        try:
            messages = await self.kafka_client.consume_messages(
                topic=TOPIC_CODEGEN_STATUS_SESSION,
                group_id="test-consumer-group-status",
                max_messages=1,
                timeout_ms=5000,
            )

            if messages:
                message = messages[0]
                envelope = ModelOnexEnvelopeV1.from_dict(message["value"])
                assert envelope.event_type == "CODEGEN_STATUS_SESSION"

                self.record_test_result("Consume and validate status event", True)
            else:
                self.record_test_result(
                    "Consume and validate status event", False, "No messages consumed"
                )
        except Exception as e:
            self.record_test_result("Consume and validate status event", False, str(e))

    async def test_correlation_tracking(self):
        """Test 4: Correlation ID tracking across events."""
        print("\n" + "=" * 80)
        print("Test 4: Correlation ID Tracking")
        print("=" * 80)

        test_correlation_id = uuid4()

        # Publish request with correlation ID
        try:
            request = CodegenPatternRequest(
                correlation_id=test_correlation_id,
                session_id=self.session_id,
                node_description="Test pattern",
                node_type=EnumNodeType.COMPUTE,
            )

            await self.kafka_client.publish_with_envelope(
                event_type="CODEGEN_REQUEST_PATTERN",
                source_node_id="test-script",
                payload=request.model_dump(),
                topic=TOPIC_CODEGEN_REQUEST_PATTERN,
                correlation_id=test_correlation_id,
            )

            # Simulate response with same correlation ID
            response = CodegenPatternResponse(
                correlation_id=test_correlation_id,
                session_id=self.session_id,
                pattern_result=[{"pattern_id": "test", "similarity": 0.95}],
                total_matches=1,
                processing_time_ms=100,
            )

            await self.kafka_client.publish_with_envelope(
                event_type="CODEGEN_RESPONSE_PATTERN",
                source_node_id="test-script",
                payload=response.model_dump(),
                topic=TOPIC_CODEGEN_RESPONSE_PATTERN,
                correlation_id=test_correlation_id,
            )

            self.record_test_result(
                "Correlation ID tracking across request/response", True
            )
        except Exception as e:
            self.record_test_result(
                "Correlation ID tracking across request/response", False, str(e)
            )

    async def test_dlq_routing(self):
        """Test 5: Dead letter queue routing (simulated)."""
        print("\n" + "=" * 80)
        print("Test 5: Dead Letter Queue (DLQ) Routing")
        print("=" * 80)

        try:
            # Simulate a failed message by publishing directly to DLQ
            dlq_message = {
                "original_topic": TOPIC_CODEGEN_REQUEST_ANALYZE,
                "original_key": str(self.correlation_id),
                "original_data": {"test": "failed message"},
                "failure_reason": "test_dlq_routing",
                "failure_timestamp": datetime.now(UTC).isoformat(),
                "retry_attempts": 3,
            }

            success = await self.kafka_client.publish_raw_event(
                topic=TOPIC_CODEGEN_REQUEST_ANALYZE_DLQ,
                data=dlq_message,
                key=str(self.correlation_id),
            )

            if success:
                # Verify DLQ message can be consumed
                messages = await self.kafka_client.consume_messages(
                    topic=TOPIC_CODEGEN_REQUEST_ANALYZE_DLQ,
                    group_id="test-dlq-consumer",
                    max_messages=1,
                    timeout_ms=5000,
                )

                if (
                    messages
                    and messages[0]["value"]["failure_reason"] == "test_dlq_routing"
                ):
                    self.record_test_result("DLQ routing and consumption", True)
                else:
                    self.record_test_result(
                        "DLQ routing and consumption", False, "DLQ message not found"
                    )
            else:
                self.record_test_result(
                    "DLQ routing and consumption", False, "Failed to publish to DLQ"
                )
        except Exception as e:
            self.record_test_result("DLQ routing and consumption", False, str(e))

    async def test_envelope_metrics(self):
        """Test 6: Envelope publishing metrics."""
        print("\n" + "=" * 80)
        print("Test 6: Envelope Publishing Metrics")
        print("=" * 80)

        try:
            metrics = await self.kafka_client.get_envelope_metrics()

            assert "envelope_publishing" in metrics
            assert "latency_metrics_ms" in metrics
            assert metrics["envelope_publishing"]["total_events_published"] > 0

            avg_latency = metrics["latency_metrics_ms"]["average"]
            p95_latency = metrics["latency_metrics_ms"]["p95"]

            print(f"  Average latency: {avg_latency:.2f}ms")
            print(f"  P95 latency: {p95_latency:.2f}ms")
            print(
                f"  Total events published: {metrics['envelope_publishing']['total_events_published']}"
            )

            # Check performance targets
            meets_target = metrics["performance_summary"]["meets_target_latency"]

            self.record_test_result(
                "Envelope metrics collection and performance", meets_target
            )
        except Exception as e:
            self.record_test_result(
                "Envelope metrics collection and performance", False, str(e)
            )

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)

        results = self.test_results
        print(f"Total tests: {results['total_tests']}")
        print(f"✓ Passed: {results['passed']}")
        print(f"✗ Failed: {results['failed']}")

        if results["failed"] > 0:
            print("\nFailed Tests:")
            for error in results["errors"]:
                print(f"  - {error['test']}")
                print(f"    {error['error']}")

        success_rate = (
            results["passed"] / results["total_tests"] * 100
            if results["total_tests"] > 0
            else 0
        )
        print(f"\nSuccess Rate: {success_rate:.1f}%")

        if results["failed"] == 0:
            print("\n✅ All tests passed!")
            return 0
        else:
            print(f"\n⚠️  {results['failed']} test(s) failed")
            return 1


async def main():
    """Main test execution."""
    print("=" * 80)
    print("Contract-First Code Generation Event Flow Test")
    print("=" * 80)

    tester = CodegenEventFlowTester()

    # Setup
    if not await tester.setup():
        print("\n❌ Setup failed. Exiting.")
        return 1

    try:
        # Run tests
        await tester.test_event_schema_validation()
        await tester.test_event_publishing()
        await tester.test_event_consumption()
        await tester.test_correlation_tracking()
        await tester.test_dlq_routing()
        await tester.test_envelope_metrics()

        # Print summary
        return tester.print_summary()

    finally:
        await tester.teardown()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
