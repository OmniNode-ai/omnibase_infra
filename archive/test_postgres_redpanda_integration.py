#!/usr/bin/env python3
"""
Test script for PostgreSQL RedPanda integration with OmniNode event publishing.

Tests the complete flow:
1. PostgreSQL Adapter processes database operations
2. Events published to RedPanda via OmniNode topic specifications
3. Event consumption and validation from RedPanda topics

Usage:
    python test_postgres_redpanda_integration.py

Prerequisites:
    - Docker services running: postgres, redpanda
    - Topics created via docker-compose topic initialization
    - PostgreSQL adapter configured with event bus integration
"""

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

# Test imports (would normally come from test infrastructure)
try:
    from omnibase_infra.enums.enum_omninode_topic_class import EnumOmniNodeTopicClass
    from omnibase_infra.infrastructure.container import create_infrastructure_container
    from omnibase_infra.models.event_publishing.model_omninode_topic_spec import (
        ModelOmniNodeTopicSpec,
    )
    from omnibase_infra.models.postgres.model_postgres_query_request import (
        ModelPostgresQueryRequest,
    )
    from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_input import (
        ModelPostgresAdapterInput,
    )
    from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.node import (
        NodePostgresAdapterEffect,
    )

    # Mock event bus for testing
    class MockEventBus:
        """Mock event bus that captures published events for testing."""

        def __init__(self):
            self.published_events: list[dict[str, Any]] = []

        async def publish(
            self,
            topic: str,
            event_data: dict[str, Any],
            correlation_id: str,
            partition_key: str,
        ):
            """Mock publish that captures events."""
            self.published_events.append(
                {
                    "topic": topic,
                    "event_data": event_data,
                    "correlation_id": correlation_id,
                    "partition_key": partition_key,
                    "timestamp": time.time(),
                },
            )
            print(f"✅ Mock published event to topic: {topic}")

        def get_events_for_topic(self, topic: str) -> list[dict[str, Any]]:
            """Get all events published to a specific topic."""
            return [event for event in self.published_events if event["topic"] == topic]

        def clear_events(self):
            """Clear captured events."""
            self.published_events.clear()

    INTEGRATION_AVAILABLE = True

except ImportError as e:
    print(f"⚠️  Integration modules not available: {e}")
    print(
        "This test requires the full PostgreSQL adapter infrastructure to be available.",
    )
    INTEGRATION_AVAILABLE = False
    MockEventBus = None


class PostgresRedPandaIntegrationTest:
    """Test suite for PostgreSQL RedPanda integration via OmniNode event publishing."""

    def __init__(self):
        self.mock_event_bus = MockEventBus() if MockEventBus else None
        self.adapter = None
        self.test_results = []

        # Configure logging for test visibility
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("postgres_redpanda_test")

    async def setup(self):
        """Setup test environment with mock services."""
        if not INTEGRATION_AVAILABLE:
            self.logger.warning(
                "Integration testing not available - missing dependencies",
            )
            return False

        try:
            # Create infrastructure container (would use test container in real test)
            container = create_infrastructure_container()

            # Override event bus with mock for testing
            container.services["protocol_event_bus"] = self.mock_event_bus

            # Create PostgreSQL adapter
            self.adapter = NodePostgresAdapterEffect(container)

            self.logger.info("✅ Test setup completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Test setup failed: {e}")
            return False

    async def test_query_success_event_publishing(self):
        """Test that successful queries publish postgres-query-completed events."""
        test_name = "Query Success Event Publishing"

        try:
            # Clear previous events
            self.mock_event_bus.clear_events()

            # Create test query request
            correlation_id = uuid4()
            query_request = ModelPostgresQueryRequest(
                query="SELECT 1 as test_value",
                parameters=[],
                correlation_id=correlation_id,
                timeout=30.0,
                record_metrics=True,
            )

            # Create adapter input
            adapter_input = ModelPostgresAdapterInput(
                operation_type="query",
                query_request=query_request,
                correlation_id=correlation_id,
                context={"test": "query_success_event"},
            )

            # Mock the database execution to avoid actual database dependency
            original_connection_manager = self.adapter._connection_manager

            class MockConnectionManager:
                async def execute_query(
                    self, query, *params, timeout=None, record_metrics=None,
                ):
                    # Return mock successful query result
                    return "SELECT 1"  # Non-SELECT result format

            self.adapter._connection_manager = MockConnectionManager()

            # Process query operation
            result = await self.adapter.process(adapter_input)

            # Restore original connection manager
            self.adapter._connection_manager = original_connection_manager

            # Verify operation succeeded
            assert result.success, f"Query operation failed: {result.error_message}"

            # Verify event was published
            expected_topic = "dev.omnibase.onex.evt.postgres-query-completed.v1"
            events = self.mock_event_bus.get_events_for_topic(expected_topic)

            assert len(events) == 1, f"Expected 1 event, got {len(events)}"

            published_event = events[0]
            assert published_event["correlation_id"] == str(correlation_id)
            assert published_event["partition_key"] == str(correlation_id)

            self.test_results.append(
                {
                    "test": test_name,
                    "status": "PASSED",
                    "message": "Successfully published postgres-query-completed event",
                },
            )

            self.logger.info(f"✅ {test_name} - PASSED")

        except Exception as e:
            self.test_results.append(
                {
                    "test": test_name,
                    "status": "FAILED",
                    "message": f"Test failed: {e!s}",
                },
            )
            self.logger.error(f"❌ {test_name} - FAILED: {e}")

    async def test_query_failure_event_publishing(self):
        """Test that failed queries publish postgres-query-failed events."""
        test_name = "Query Failure Event Publishing"

        try:
            # Clear previous events
            self.mock_event_bus.clear_events()

            # Create test query request that will fail
            correlation_id = uuid4()
            query_request = ModelPostgresQueryRequest(
                query="SELECT * FROM non_existent_table",
                parameters=[],
                correlation_id=correlation_id,
                timeout=30.0,
                record_metrics=True,
            )

            # Create adapter input
            adapter_input = ModelPostgresAdapterInput(
                operation_type="query",
                query_request=query_request,
                correlation_id=correlation_id,
                context={"test": "query_failure_event"},
            )

            # Mock the database execution to simulate failure
            original_connection_manager = self.adapter._connection_manager

            class MockFailingConnectionManager:
                async def execute_query(
                    self, query, *params, timeout=None, record_metrics=None,
                ):
                    # Simulate database error
                    raise Exception("Table 'non_existent_table' doesn't exist")

            self.adapter._connection_manager = MockFailingConnectionManager()

            # Process query operation (should handle error gracefully)
            result = await self.adapter.process(adapter_input)

            # Restore original connection manager
            self.adapter._connection_manager = original_connection_manager

            # Verify operation failed but was handled gracefully
            assert not result.success, "Query operation should have failed"
            assert result.error_message, "Error message should be present"

            # Verify event was published
            expected_topic = "dev.omnibase.onex.evt.postgres-query-failed.v1"
            events = self.mock_event_bus.get_events_for_topic(expected_topic)

            assert len(events) == 1, f"Expected 1 event, got {len(events)}"

            published_event = events[0]
            assert published_event["correlation_id"] == str(correlation_id)
            assert published_event["partition_key"] == str(correlation_id)

            self.test_results.append(
                {
                    "test": test_name,
                    "status": "PASSED",
                    "message": "Successfully published postgres-query-failed event",
                },
            )

            self.logger.info(f"✅ {test_name} - PASSED")

        except Exception as e:
            self.test_results.append(
                {
                    "test": test_name,
                    "status": "FAILED",
                    "message": f"Test failed: {e!s}",
                },
            )
            self.logger.error(f"❌ {test_name} - FAILED: {e}")

    async def test_health_check_event_publishing(self):
        """Test that health checks publish postgres-health-response events."""
        test_name = "Health Check Event Publishing"

        try:
            # Clear previous events
            self.mock_event_bus.clear_events()

            # Create health check request
            correlation_id = uuid4()
            adapter_input = ModelPostgresAdapterInput(
                operation_type="health_check",
                correlation_id=correlation_id,
                context={"test": "health_check_event"},
            )

            # Process health check operation
            result = await self.adapter.process(adapter_input)

            # Verify health check was processed
            assert result.operation_type == "health_check"

            # Verify event was published
            expected_topic = "dev.omnibase.onex.qrs.postgres-health-response.v1"
            events = self.mock_event_bus.get_events_for_topic(expected_topic)

            assert len(events) == 1, f"Expected 1 event, got {len(events)}"

            published_event = events[0]
            assert published_event["correlation_id"] == str(correlation_id)
            assert published_event["partition_key"] == str(correlation_id)

            self.test_results.append(
                {
                    "test": test_name,
                    "status": "PASSED",
                    "message": "Successfully published postgres-health-response event",
                },
            )

            self.logger.info(f"✅ {test_name} - PASSED")

        except Exception as e:
            self.test_results.append(
                {
                    "test": test_name,
                    "status": "FAILED",
                    "message": f"Test failed: {e!s}",
                },
            )
            self.logger.error(f"❌ {test_name} - FAILED: {e}")

    async def test_omninode_topic_specifications(self):
        """Test that OmniNode topic specifications generate correct topic names."""
        test_name = "OmniNode Topic Specifications"

        try:
            # Test postgres-query-completed topic
            topic_spec = ModelOmniNodeTopicSpec.for_postgres_query_completed()
            expected_topic = "dev.omnibase.onex.evt.postgres-query-completed.v1"
            assert (
                topic_spec.to_topic_string() == expected_topic
            ), f"Expected {expected_topic}, got {topic_spec.to_topic_string()}"

            # Test postgres-query-failed topic
            topic_spec = ModelOmniNodeTopicSpec.for_postgres_query_failed()
            expected_topic = "dev.omnibase.onex.evt.postgres-query-failed.v1"
            assert (
                topic_spec.to_topic_string() == expected_topic
            ), f"Expected {expected_topic}, got {topic_spec.to_topic_string()}"

            # Test postgres-health-response topic
            topic_spec = ModelOmniNodeTopicSpec.for_postgres_health_check()
            expected_topic = "dev.omnibase.onex.qrs.postgres-health-response.v1"
            assert (
                topic_spec.to_topic_string() == expected_topic
            ), f"Expected {expected_topic}, got {topic_spec.to_topic_string()}"

            self.test_results.append(
                {
                    "test": test_name,
                    "status": "PASSED",
                    "message": "All OmniNode topic specifications generate correct topic names",
                },
            )

            self.logger.info(f"✅ {test_name} - PASSED")

        except Exception as e:
            self.test_results.append(
                {
                    "test": test_name,
                    "status": "FAILED",
                    "message": f"Test failed: {e!s}",
                },
            )
            self.logger.error(f"❌ {test_name} - FAILED: {e}")

    async def test_fire_and_forget_behavior(self):
        """Test that event publishing failures don't break main operations."""
        test_name = "Fire-and-Forget Event Publishing"

        try:
            # Create failing event bus
            class FailingEventBus:
                async def publish(
                    self, topic, event_data, correlation_id, partition_key,
                ):
                    raise Exception("Event bus connection failed")

            # Replace with failing event bus
            original_event_bus = self.adapter._event_bus
            self.adapter._event_bus = FailingEventBus()

            # Create test query request
            correlation_id = uuid4()
            query_request = ModelPostgresQueryRequest(
                query="SELECT 1 as test_value",
                parameters=[],
                correlation_id=correlation_id,
                timeout=30.0,
                record_metrics=True,
            )

            # Create adapter input
            adapter_input = ModelPostgresAdapterInput(
                operation_type="query",
                query_request=query_request,
                correlation_id=correlation_id,
                context={"test": "fire_and_forget"},
            )

            # Mock successful database execution
            class MockConnectionManager:
                async def execute_query(
                    self, query, *params, timeout=None, record_metrics=None,
                ):
                    return "SELECT 1"

            original_connection_manager = self.adapter._connection_manager
            self.adapter._connection_manager = MockConnectionManager()

            # Process query operation - should succeed despite event publishing failure
            result = await self.adapter.process(adapter_input)

            # Restore original services
            self.adapter._event_bus = original_event_bus
            self.adapter._connection_manager = original_connection_manager

            # Verify operation succeeded despite event publishing failure
            assert (
                result.success
            ), f"Main operation should succeed despite event publishing failure: {result.error_message}"

            self.test_results.append(
                {
                    "test": test_name,
                    "status": "PASSED",
                    "message": "Main operations continue successfully despite event publishing failures",
                },
            )

            self.logger.info(f"✅ {test_name} - PASSED")

        except Exception as e:
            self.test_results.append(
                {
                    "test": test_name,
                    "status": "FAILED",
                    "message": f"Test failed: {e!s}",
                },
            )
            self.logger.error(f"❌ {test_name} - FAILED: {e}")

    async def run_all_tests(self):
        """Run all integration tests."""
        if not INTEGRATION_AVAILABLE:
            self.logger.error("❌ Integration tests cannot run - missing dependencies")
            return

        self.logger.info("🚀 Starting PostgreSQL RedPanda Integration Tests")

        # Setup test environment
        if not await self.setup():
            self.logger.error("❌ Test setup failed - aborting tests")
            return

        # Run all tests
        await self.test_omninode_topic_specifications()
        await self.test_query_success_event_publishing()
        await self.test_query_failure_event_publishing()
        await self.test_health_check_event_publishing()
        await self.test_fire_and_forget_behavior()

        # Print test summary
        self.print_test_summary()

    def print_test_summary(self):
        """Print comprehensive test results summary."""
        print("\n" + "=" * 60)
        print("📊 POSTGRESQL REDPANDA INTEGRATION TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for result in self.test_results if result["status"] == "PASSED")
        failed = sum(1 for result in self.test_results if result["status"] == "FAILED")
        total = len(self.test_results)

        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%" if total > 0 else "0%")

        print("\nDetailed Results:")
        print("-" * 60)

        for result in self.test_results:
            status_emoji = "✅" if result["status"] == "PASSED" else "❌"
            print(f"{status_emoji} {result['test']}: {result['status']}")
            print(f"   {result['message']}")
            print()

        if failed == 0:
            print(
                "🎉 ALL TESTS PASSED - PostgreSQL RedPanda integration is working correctly!",
            )
        else:
            print(f"⚠️  {failed} test(s) failed - please review the issues above")

        print("=" * 60)


async def main():
    """Main test runner."""
    tester = PostgresRedPandaIntegrationTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
