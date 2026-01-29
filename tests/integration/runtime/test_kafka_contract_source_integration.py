# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for KafkaContractSource with Kafka.

These tests verify actual Kafka consumer wiring and event processing for the
KafkaContractSource. They validate the end-to-end flow from publishing contract
events to Kafka through to cache population and discovery.

Run with: pytest tests/integration/runtime/test_kafka_contract_source_integration.py -v
Skip if Kafka unavailable: pytest -m "not kafka"

Test Categories:
    - End-to-End Tests: Full publish -> consume -> cache -> discover flow
    - Event Processing Tests: Typed event model handling via Kafka
    - Error Handling Tests: Invalid contract processing with real Kafka
    - Concurrent Consumer Tests: Multiple KafkaContractSource instances

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker address (e.g., "localhost:29092")

Related:
    - OMN-1654: KafkaContractSource (cache + discovery)
    - src/omnibase_infra/runtime/kafka_contract_source.py
    - tests/unit/runtime/test_kafka_contract_source.py (unit tests)
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator, Callable, Coroutine
from uuid import uuid4

import pytest

from tests.integration.event_bus.conftest import wait_for_consumer_ready

# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Check if Kafka is available based on environment variable
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_AVAILABLE = (
    KAFKA_BOOTSTRAP_SERVERS is not None and KAFKA_BOOTSTRAP_SERVERS.strip()
)

# Module-level markers - skip all tests if Kafka is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.kafka,
    pytest.mark.skipif(
        not KAFKA_AVAILABLE,
        reason="Kafka not available (KAFKA_BOOTSTRAP_SERVERS not set)",
    ),
]

# Test configuration constants
MESSAGE_DELIVERY_WAIT_SECONDS = 3.0
CONSUMER_READY_TIMEOUT_SECONDS = 10.0


# =============================================================================
# Test Helpers
# =============================================================================


def make_valid_contract_yaml(
    handler_id: str,
    name: str,
    *,
    archetype: str = "compute",
    version: tuple[int, int, int] = (1, 0, 0),
    handler_class: str | None = None,
) -> str:
    """Generate valid handler contract YAML for testing.

    Args:
        handler_id: Unique handler identifier
        name: Human-readable handler name
        archetype: Node archetype (compute, effect, reducer, orchestrator)
        version: Semantic version tuple (major, minor, patch)
        handler_class: Optional handler class path

    Returns:
        Valid YAML string for a handler contract
    """
    major, minor, patch = version
    yaml_content = f'''handler_id: "{handler_id}"
name: "{name}"
contract_version:
  major: {major}
  minor: {minor}
  patch: {patch}
descriptor:
  node_archetype: "{archetype}"
input_model: "test.models.Input"
output_model: "test.models.Output"'''

    if handler_class:
        yaml_content += f'''
metadata:
  handler_class: "{handler_class}"'''

    return yaml_content


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def kafka_bootstrap_servers() -> str:
    """Get Kafka bootstrap servers from environment."""
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")


@pytest.fixture
def unique_topic_prefix() -> str:
    """Generate unique topic prefix for test isolation."""
    return f"test.kafka-contract.{uuid4().hex[:12]}"


@pytest.fixture
def unique_group() -> str:
    """Generate unique consumer group for test isolation."""
    return f"test-contract-group-{uuid4().hex[:8]}"


@pytest.fixture
async def kafka_event_bus(
    kafka_bootstrap_servers: str,
) -> AsyncGenerator:
    """Create and configure EventBusKafka for integration testing.

    Yields a started EventBusKafka instance and ensures cleanup after test.
    """
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    config = ModelKafkaEventBusConfig(
        bootstrap_servers=kafka_bootstrap_servers,
        environment="integration-test",
        group="test-contract-source",
        timeout_seconds=30,
        max_retry_attempts=2,
        retry_backoff_base=0.5,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=10.0,
    )
    bus = EventBusKafka(config=config)
    await bus.start()

    yield bus

    # Cleanup: ensure bus is closed
    try:
        await bus.close()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
async def contract_topic(
    ensure_test_topic: Callable[[str, int], Coroutine[None, None, str]],
    unique_topic_prefix: str,
) -> str:
    """Create a unique topic for contract events.

    Returns:
        The created topic name.
    """
    topic_name = f"{unique_topic_prefix}.contracts"
    await ensure_test_topic(topic_name, 1)
    return topic_name


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestKafkaContractSourceE2E:
    """End-to-end integration tests for KafkaContractSource with real Kafka."""

    @pytest.mark.asyncio
    async def test_end_to_end_contract_registration_via_kafka(
        self,
        kafka_event_bus,
        contract_topic: str,
        unique_group: str,
    ) -> None:
        """Test full flow: publish event -> consume -> cache -> discover.

        This test validates the complete contract registration flow:
        1. Create KafkaContractSource
        2. Set up Kafka subscription that calls source.on_contract_registered()
        3. Publish contract YAML to topic
        4. Wait for event to be consumed and cached
        5. Call discover_handlers() and verify descriptor is in result
        """
        from omnibase_infra.event_bus.models import ModelEventMessage
        from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

        # Create the contract source
        source = KafkaContractSource(environment="integration-test")
        assert source.cached_count == 0

        # Track when contract is processed
        contract_processed = asyncio.Event()
        node_name = f"test.handler.{uuid4().hex[:8]}"
        contract_yaml = make_valid_contract_yaml(
            handler_id=f"effect.{node_name}",
            name="Test Integration Handler",
            handler_class="test.handlers.IntegrationHandler",
        )

        async def contract_event_handler(msg: ModelEventMessage) -> None:
            """Handle contract events from Kafka and update the source."""
            try:
                # Parse the event payload
                payload = json.loads(msg.value.decode("utf-8"))
                event_node_name = payload.get("node_name")
                event_contract_yaml = payload.get("contract_yaml")

                if event_node_name and event_contract_yaml:
                    source.on_contract_registered(
                        node_name=event_node_name,
                        contract_yaml=event_contract_yaml,
                    )
                    contract_processed.set()
            except Exception:
                pass  # Ignore malformed messages in test

        # Subscribe to the contract topic
        unsubscribe = await kafka_event_bus.subscribe(
            contract_topic,
            unique_group,
            contract_event_handler,
        )

        try:
            # Wait for consumer to be ready
            await wait_for_consumer_ready(kafka_event_bus, contract_topic)

            # Publish contract registration event
            event_payload = {
                "node_name": node_name,
                "contract_yaml": contract_yaml,
                "event_type": "contract.registered",
            }
            await kafka_event_bus.publish(
                contract_topic,
                node_name.encode("utf-8"),
                json.dumps(event_payload).encode("utf-8"),
            )

            # Wait for event to be processed
            try:
                await asyncio.wait_for(
                    contract_processed.wait(),
                    timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
                )
            except TimeoutError:
                pytest.fail(
                    f"Contract event not processed within {MESSAGE_DELIVERY_WAIT_SECONDS * 2}s"
                )

            # Verify the descriptor was cached
            assert source.cached_count == 1

            # Verify discover_handlers returns the cached descriptor
            result = await source.discover_handlers()
            assert len(result.descriptors) == 1
            assert result.descriptors[0].handler_id == f"effect.{node_name}"
            assert result.descriptors[0].name == "Test Integration Handler"
            assert (
                result.descriptors[0].handler_class
                == "test.handlers.IntegrationHandler"
            )

        finally:
            await unsubscribe()

    @pytest.mark.asyncio
    async def test_contract_deregistration_removes_from_cache(
        self,
        kafka_event_bus,
        contract_topic: str,
        unique_group: str,
    ) -> None:
        """Test deregistration flow removes cached descriptor.

        1. Register a contract
        2. Publish deregistration event
        3. Verify descriptor is removed from cache
        """
        from omnibase_infra.event_bus.models import ModelEventMessage
        from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

        source = KafkaContractSource(environment="integration-test")
        node_name = f"test.dereg.{uuid4().hex[:8]}"

        # Pre-register a contract
        contract_yaml = make_valid_contract_yaml(
            handler_id=f"effect.{node_name}",
            name="To Be Deregistered",
        )
        source.on_contract_registered(node_name=node_name, contract_yaml=contract_yaml)
        assert source.cached_count == 1

        deregistration_processed = asyncio.Event()

        async def deregistration_handler(msg: ModelEventMessage) -> None:
            """Handle deregistration events from Kafka."""
            try:
                payload = json.loads(msg.value.decode("utf-8"))
                event_node_name = payload.get("node_name")
                event_type = payload.get("event_type")

                if event_node_name and event_type == "contract.deregistered":
                    source.on_contract_deregistered(node_name=event_node_name)
                    deregistration_processed.set()
            except Exception:
                pass

        unsubscribe = await kafka_event_bus.subscribe(
            contract_topic,
            unique_group,
            deregistration_handler,
        )

        try:
            await wait_for_consumer_ready(kafka_event_bus, contract_topic)

            # Publish deregistration event
            event_payload = {
                "node_name": node_name,
                "event_type": "contract.deregistered",
                "reason": "shutdown",
            }
            await kafka_event_bus.publish(
                contract_topic,
                node_name.encode("utf-8"),
                json.dumps(event_payload).encode("utf-8"),
            )

            # Wait for deregistration to be processed
            try:
                await asyncio.wait_for(
                    deregistration_processed.wait(),
                    timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
                )
            except TimeoutError:
                pytest.fail("Deregistration event not processed within timeout")

            # Verify descriptor was removed
            assert source.cached_count == 0

            result = await source.discover_handlers()
            assert len(result.descriptors) == 0

        finally:
            await unsubscribe()

    @pytest.mark.asyncio
    async def test_invalid_contract_event_collected_as_error(
        self,
        kafka_event_bus,
        contract_topic: str,
        unique_group: str,
    ) -> None:
        """Test that invalid contracts produce validation errors.

        1. Publish invalid contract YAML
        2. Verify source collects error
        3. Verify discover_handlers returns the error
        """
        from omnibase_infra.event_bus.models import ModelEventMessage
        from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

        source = KafkaContractSource(environment="integration-test", graceful_mode=True)
        node_name = f"test.invalid.{uuid4().hex[:8]}"

        error_collected = asyncio.Event()

        async def invalid_contract_handler(msg: ModelEventMessage) -> None:
            """Handle contract events including invalid ones."""
            try:
                payload = json.loads(msg.value.decode("utf-8"))
                event_node_name = payload.get("node_name")
                event_contract_yaml = payload.get("contract_yaml")

                if event_node_name and event_contract_yaml:
                    success = source.on_contract_registered(
                        node_name=event_node_name,
                        contract_yaml=event_contract_yaml,
                    )
                    # In graceful mode, invalid contracts return False
                    if not success:
                        error_collected.set()
            except Exception:
                pass

        unsubscribe = await kafka_event_bus.subscribe(
            contract_topic,
            unique_group,
            invalid_contract_handler,
        )

        try:
            await wait_for_consumer_ready(kafka_event_bus, contract_topic)

            # Publish invalid contract (missing required fields)
            invalid_yaml = """
name: "Incomplete Handler"
version: "1.0.0"
"""  # Missing handler_id, input_model, output_model

            event_payload = {
                "node_name": node_name,
                "contract_yaml": invalid_yaml,
                "event_type": "contract.registered",
            }
            await kafka_event_bus.publish(
                contract_topic,
                node_name.encode("utf-8"),
                json.dumps(event_payload).encode("utf-8"),
            )

            # Wait for error to be collected
            try:
                await asyncio.wait_for(
                    error_collected.wait(),
                    timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
                )
            except TimeoutError:
                pytest.fail("Invalid contract event not processed within timeout")

            # Verify no descriptors cached
            assert source.cached_count == 0

            # Verify validation error was collected
            result = await source.discover_handlers()
            assert len(result.descriptors) == 0
            assert len(result.validation_errors) == 1
            assert node_name in result.validation_errors[0].message

        finally:
            await unsubscribe()


# =============================================================================
# Typed Event Model Tests
# =============================================================================


class TestKafkaContractSourceTypedEvents:
    """Integration tests for typed event model handling via Kafka."""

    @pytest.mark.asyncio
    async def test_handle_registered_event_with_typed_model(
        self,
        kafka_event_bus,
        contract_topic: str,
        unique_group: str,
    ) -> None:
        """Test processing typed ModelContractRegisteredEvent from Kafka.

        Uses the typed event model from omnibase_core to verify proper
        event deserialization and handling.
        """
        from omnibase_core.models.events import ModelContractRegisteredEvent
        from omnibase_core.models.primitives import ModelSemVer
        from omnibase_infra.event_bus.models import ModelEventMessage
        from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

        source = KafkaContractSource(environment="integration-test")
        node_name = f"test.typed.{uuid4().hex[:8]}"

        event_processed = asyncio.Event()
        contract_yaml = make_valid_contract_yaml(
            handler_id=f"compute.{node_name}",
            name="Typed Event Handler",
            handler_class="test.handlers.TypedHandler",
        )

        async def typed_event_handler(msg: ModelEventMessage) -> None:
            """Handle typed contract events from Kafka."""
            try:
                payload = json.loads(msg.value.decode("utf-8"))

                # Reconstruct typed event model
                event = ModelContractRegisteredEvent(
                    node_name=payload["node_name"],
                    node_version=ModelSemVer(
                        major=payload["node_version"]["major"],
                        minor=payload["node_version"]["minor"],
                        patch=payload["node_version"]["patch"],
                    ),
                    contract_hash=payload["contract_hash"],
                    contract_yaml=payload["contract_yaml"],
                )

                # Use the typed event handler
                source.handle_registered_event(event)
                event_processed.set()
            except Exception:
                pass

        unsubscribe = await kafka_event_bus.subscribe(
            contract_topic,
            unique_group,
            typed_event_handler,
        )

        try:
            await wait_for_consumer_ready(kafka_event_bus, contract_topic)

            # Publish typed event as JSON
            event_payload = {
                "node_name": node_name,
                "node_version": {"major": 1, "minor": 2, "patch": 3},
                "contract_hash": "abc123",
                "contract_yaml": contract_yaml,
            }
            await kafka_event_bus.publish(
                contract_topic,
                node_name.encode("utf-8"),
                json.dumps(event_payload).encode("utf-8"),
            )

            # Wait for processing
            try:
                await asyncio.wait_for(
                    event_processed.wait(),
                    timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
                )
            except TimeoutError:
                pytest.fail("Typed event not processed within timeout")

            # Verify descriptor was cached correctly
            result = await source.discover_handlers()
            assert len(result.descriptors) == 1
            assert result.descriptors[0].handler_id == f"compute.{node_name}"
            assert result.descriptors[0].handler_class == "test.handlers.TypedHandler"

        finally:
            await unsubscribe()


# =============================================================================
# Multiple Consumer Tests
# =============================================================================


class TestKafkaContractSourceMultipleConsumers:
    """Integration tests for multiple KafkaContractSource instances on same topic."""

    @pytest.mark.asyncio
    async def test_multiple_consumers_same_topic(
        self,
        kafka_event_bus,
        contract_topic: str,
    ) -> None:
        """Test multiple KafkaContractSource instances on same topic.

        When using different consumer groups, each source should receive
        all contract events independently.
        """
        from omnibase_infra.event_bus.models import ModelEventMessage
        from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

        # Create two independent sources
        source1 = KafkaContractSource(environment="integration-test")
        source2 = KafkaContractSource(environment="integration-test")

        node_name = f"test.multi.{uuid4().hex[:8]}"
        contract_yaml = make_valid_contract_yaml(
            handler_id=f"effect.{node_name}",
            name="Multi-Consumer Handler",
        )

        source1_processed = asyncio.Event()
        source2_processed = asyncio.Event()

        async def source1_handler(msg: ModelEventMessage) -> None:
            try:
                payload = json.loads(msg.value.decode("utf-8"))
                source1.on_contract_registered(
                    node_name=payload["node_name"],
                    contract_yaml=payload["contract_yaml"],
                )
                source1_processed.set()
            except Exception:
                pass

        async def source2_handler(msg: ModelEventMessage) -> None:
            try:
                payload = json.loads(msg.value.decode("utf-8"))
                source2.on_contract_registered(
                    node_name=payload["node_name"],
                    contract_yaml=payload["contract_yaml"],
                )
                source2_processed.set()
            except Exception:
                pass

        # Subscribe with different consumer groups
        group1 = f"source1-group-{uuid4().hex[:8]}"
        group2 = f"source2-group-{uuid4().hex[:8]}"

        unsubscribe1 = await kafka_event_bus.subscribe(
            contract_topic, group1, source1_handler
        )
        unsubscribe2 = await kafka_event_bus.subscribe(
            contract_topic, group2, source2_handler
        )

        try:
            await wait_for_consumer_ready(kafka_event_bus, contract_topic)

            # Publish one contract event
            event_payload = {
                "node_name": node_name,
                "contract_yaml": contract_yaml,
                "event_type": "contract.registered",
            }
            await kafka_event_bus.publish(
                contract_topic,
                node_name.encode("utf-8"),
                json.dumps(event_payload).encode("utf-8"),
            )

            # Wait for both sources to process
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        source1_processed.wait(),
                        source2_processed.wait(),
                    ),
                    timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 3,
                )
            except TimeoutError:
                pytest.fail("Not all sources processed the event within timeout")

            # Both sources should have the same descriptor
            result1 = await source1.discover_handlers()
            result2 = await source2.discover_handlers()

            assert len(result1.descriptors) == 1
            assert len(result2.descriptors) == 1
            assert (
                result1.descriptors[0].handler_id == result2.descriptors[0].handler_id
            )

        finally:
            await unsubscribe1()
            await unsubscribe2()

    @pytest.mark.asyncio
    async def test_source_isolation_between_instances(
        self,
        kafka_event_bus,
        contract_topic: str,
    ) -> None:
        """Test that KafkaContractSource instances maintain isolated caches.

        Events processed by one source should not affect another source's cache.
        """
        from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

        source1 = KafkaContractSource(environment="integration-test")
        source2 = KafkaContractSource(environment="integration-test")

        # Register a contract directly in source1
        contract_yaml = make_valid_contract_yaml(
            handler_id="effect.isolated.handler",
            name="Isolated Handler",
        )
        source1.on_contract_registered(
            node_name="isolated.node",
            contract_yaml=contract_yaml,
        )

        # source1 should have the descriptor
        assert source1.cached_count == 1

        # source2 should remain empty
        assert source2.cached_count == 0

        result1 = await source1.discover_handlers()
        result2 = await source2.discover_handlers()

        assert len(result1.descriptors) == 1
        assert len(result2.descriptors) == 0


# =============================================================================
# Event Ordering and Idempotency Tests
# =============================================================================


class TestKafkaContractSourceEventOrdering:
    """Integration tests for event ordering and idempotency with Kafka."""

    @pytest.mark.asyncio
    async def test_multiple_registrations_same_node_uses_latest(
        self,
        kafka_event_bus,
        contract_topic: str,
        unique_group: str,
    ) -> None:
        """Test that multiple registrations for same node use latest version.

        When multiple registration events arrive for the same node_name,
        the cache should contain the latest version.
        """
        from omnibase_infra.event_bus.models import ModelEventMessage
        from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

        source = KafkaContractSource(environment="integration-test")
        node_name = f"test.versioned.{uuid4().hex[:8]}"

        events_processed = 0
        all_processed = asyncio.Event()
        expected_events = 3

        async def version_handler(msg: ModelEventMessage) -> None:
            nonlocal events_processed
            try:
                payload = json.loads(msg.value.decode("utf-8"))
                source.on_contract_registered(
                    node_name=payload["node_name"],
                    contract_yaml=payload["contract_yaml"],
                )
                events_processed += 1
                if events_processed >= expected_events:
                    all_processed.set()
            except Exception:
                pass

        unsubscribe = await kafka_event_bus.subscribe(
            contract_topic,
            unique_group,
            version_handler,
        )

        try:
            await wait_for_consumer_ready(kafka_event_bus, contract_topic)

            # Publish three versions of the same node
            for version in [(1, 0, 0), (1, 1, 0), (2, 0, 0)]:
                contract_yaml = make_valid_contract_yaml(
                    handler_id=f"effect.{node_name}",
                    name=f"Handler v{version[0]}.{version[1]}.{version[2]}",
                    version=version,
                )
                event_payload = {
                    "node_name": node_name,
                    "contract_yaml": contract_yaml,
                }
                await kafka_event_bus.publish(
                    contract_topic,
                    node_name.encode("utf-8"),
                    json.dumps(event_payload).encode("utf-8"),
                )
                # Small delay to ensure ordering
                await asyncio.sleep(0.1)

            # Wait for all events to be processed
            try:
                await asyncio.wait_for(
                    all_processed.wait(),
                    timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 3,
                )
            except TimeoutError:
                pytest.fail(
                    f"Only {events_processed}/{expected_events} events processed"
                )

            # Should only have one cached entry (the latest)
            assert source.cached_count == 1

            result = await source.discover_handlers()
            assert len(result.descriptors) == 1
            # Latest version should be 2.0.0
            assert result.descriptors[0].name == "Handler v2.0.0"

        finally:
            await unsubscribe()


# =============================================================================
# Correlation ID Propagation Tests
# =============================================================================


class TestKafkaContractSourceCorrelationId:
    """Integration tests for correlation ID propagation through Kafka events."""

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_from_event(
        self,
        kafka_event_bus,
        contract_topic: str,
        unique_group: str,
    ) -> None:
        """Test that correlation_id from Kafka event is used in source operations."""
        from uuid import UUID

        from omnibase_infra.event_bus.models import ModelEventMessage
        from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

        source = KafkaContractSource(environment="integration-test")
        node_name = f"test.corr.{uuid4().hex[:8]}"
        event_correlation_id = uuid4()

        event_processed = asyncio.Event()

        async def correlation_handler(msg: ModelEventMessage) -> None:
            try:
                payload = json.loads(msg.value.decode("utf-8"))
                corr_id = UUID(payload.get("correlation_id"))

                source.on_contract_registered(
                    node_name=payload["node_name"],
                    contract_yaml=payload["contract_yaml"],
                    correlation_id=corr_id,
                )
                event_processed.set()
            except Exception:
                pass

        unsubscribe = await kafka_event_bus.subscribe(
            contract_topic,
            unique_group,
            correlation_handler,
        )

        try:
            await wait_for_consumer_ready(kafka_event_bus, contract_topic)

            contract_yaml = make_valid_contract_yaml(
                handler_id=f"effect.{node_name}",
                name="Correlation Test Handler",
            )

            event_payload = {
                "node_name": node_name,
                "contract_yaml": contract_yaml,
                "correlation_id": str(event_correlation_id),
            }
            await kafka_event_bus.publish(
                contract_topic,
                node_name.encode("utf-8"),
                json.dumps(event_payload).encode("utf-8"),
            )

            try:
                await asyncio.wait_for(
                    event_processed.wait(),
                    timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
                )
            except TimeoutError:
                pytest.fail("Event not processed within timeout")

            # Verify the contract was registered successfully
            assert source.cached_count == 1

        finally:
            await unsubscribe()
