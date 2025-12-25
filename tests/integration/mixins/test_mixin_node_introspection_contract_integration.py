# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for contract.yaml -> ModelIntrospectionConfig -> MixinNodeIntrospection.

This test suite validates the complete workflow from contract configuration
through to mixin initialization, mimicking how a real ONEX node would use
the introspection mixin with contract-driven topic configuration.

Test Categories:
    - Contract-Driven Configuration: Parsing contract data and creating config
    - Topic Resolution: Verifying topics flow through correctly from contract
    - End-to-End Workflow: Complete node initialization with contract data
    - Multi-Channel Support: Testing nodes with multiple publish/subscribe channels

Usage:
    pytest tests/integration/mixins/test_mixin_node_introspection_contract_integration.py

Note:
    These tests do not require external services (Kafka, etc.) as they focus on
    the contract -> config -> mixin integration flow, not actual event publishing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict
from uuid import UUID, uuid4

import pytest

from omnibase_infra.mixins import MixinNodeIntrospection, ModelIntrospectionConfig
from omnibase_infra.mixins.model_introspection_config import (
    DEFAULT_HEARTBEAT_TOPIC,
    DEFAULT_INTROSPECTION_TOPIC,
    DEFAULT_REQUEST_INTROSPECTION_TOPIC,
)
from omnibase_infra.models.discovery import ModelNodeIntrospectionEvent
from omnibase_infra.models.registration import ModelNodeHeartbeatEvent

# Module-level markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

# Test timing constants (in seconds)
CACHE_TTL_WAIT = 0.15  # Wait for cache TTL expiration (TTL=0.1s + buffer)
HEARTBEAT_INTERVAL = 0.05  # Default heartbeat interval for tests
HEARTBEAT_WAIT = 0.1  # Wait for at least one heartbeat
MULTIPLE_HEARTBEAT_WAIT = 0.2  # Wait for multiple heartbeats
CACHE_EXPIRE_WAIT = 0.01  # Brief wait for cache expiration
LISTENER_SUBSCRIBE_WAIT = 0.1  # Time for listener to subscribe


# =============================================================================
# Mock Fixtures
# =============================================================================


class PublishedEventRecord(TypedDict):
    """Type for a published event record in MockEventBus.

    Represents the structure of events stored when using the publish() fallback method.
    The value is JSON-decoded from the serialized event payload.
    """

    topic: str
    key: bytes | None
    value: dict[str, object]


class MockEventBus:
    """Mock event bus for testing introspection publishing without Kafka."""

    def __init__(self) -> None:
        """Initialize mock event bus."""
        self.published_envelopes: list[
            tuple[ModelNodeIntrospectionEvent | ModelNodeHeartbeatEvent, str]
        ] = []
        self.published_events: list[PublishedEventRecord] = []
        self.subscribed_topics: list[str] = []
        self.subscribed_groups: list[str] = []

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
    ) -> None:
        """Mock publish_envelope method.

        Args:
            envelope: Event envelope to publish.
            topic: Event topic.
        """
        if isinstance(envelope, ModelNodeIntrospectionEvent | ModelNodeHeartbeatEvent):
            self.published_envelopes.append((envelope, topic))

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
    ) -> None:
        """Mock publish method (fallback).

        Args:
            topic: Event topic.
            key: Event key.
            value: Event payload as bytes.
        """
        import json

        self.published_events.append(
            {
                "topic": topic,
                "key": key,
                "value": json.loads(value.decode("utf-8")),
            }
        )

    async def subscribe(
        self,
        topic: str,
        group_id: str,
        on_message: Callable[[object], Awaitable[None]],
    ) -> Callable[[], Awaitable[None]]:
        """Mock subscribe method.

        Args:
            topic: Topic to subscribe to.
            group_id: Consumer group ID.
            on_message: Callback function for messages.

        Returns:
            An async unsubscribe function.
        """
        self.subscribed_topics.append(topic)
        self.subscribed_groups.append(group_id)

        async def unsubscribe() -> None:
            pass

        return unsubscribe


class ContractDrivenEffectNode(MixinNodeIntrospection):
    """Mock EFFECT node that initializes from contract-like data.

    This simulates how a real ONEX node would extract configuration
    from its contract.yaml and initialize introspection.
    """

    def __init__(
        self,
        contract_data: dict[str, Any],
        event_bus: MockEventBus | None = None,
    ) -> None:
        """Initialize node from contract data.

        Args:
            contract_data: Simulated contract.yaml structure.
            event_bus: Optional event bus for publishing.
        """
        self._state = "initialized"
        self.health_url = "http://localhost:8080/health"
        self.contract_data = contract_data

        # Extract configuration from contract (mimics real node behavior)
        metadata = contract_data.get("metadata", {})
        event_channels = contract_data.get("event_channels", {})

        # Build topic mappings from event_channels
        publishes = {
            ch["event_type"]: ch["topic"] for ch in event_channels.get("publishes", [])
        }
        subscribes = {
            ch["event_type"]: ch["topic"] for ch in event_channels.get("subscribes", [])
        }

        # Store the node name from contract for test verification
        self._contract_node_name = metadata.get("name", "unknown-node")

        # Build config kwargs, only including topics that are explicitly defined
        config_kwargs: dict[str, Any] = {
            "node_id": uuid4(),  # Generate unique UUID for each node instance
            "node_type": metadata.get("node_type", "EFFECT"),
            "event_bus": event_bus,
            "version": metadata.get("version", "1.0.0"),
        }

        # Only add topic overrides if they are explicitly defined in contract
        if publishes.get("introspection"):
            config_kwargs["introspection_topic"] = publishes["introspection"]
        if publishes.get("heartbeat"):
            config_kwargs["heartbeat_topic"] = publishes["heartbeat"]
        if subscribes.get("introspection_request"):
            config_kwargs["request_introspection_topic"] = subscribes[
                "introspection_request"
            ]

        # Create introspection config from contract data
        config = ModelIntrospectionConfig(**config_kwargs)

        self.initialize_introspection_from_config(config)

    async def execute_effect(
        self, operation: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock execute method for EFFECT node.

        Args:
            operation: Operation to execute.
            payload: Operation payload.

        Returns:
            Operation result.
        """
        return {"result": "ok", "operation": operation}


class ComputeNodeWithCustomTopics(MixinNodeIntrospection):
    """Mock COMPUTE node demonstrating domain-specific topic configuration."""

    def __init__(
        self,
        node_name: str,
        domain: str,
        event_bus: MockEventBus | None = None,
    ) -> None:
        """Initialize compute node with domain-specific topics.

        Args:
            node_name: Human-readable node name (for test identification).
            domain: Domain name for topic namespacing.
            event_bus: Optional event bus for publishing.
        """
        self._state = "ready"
        self._node_name = node_name  # Store for test verification

        # Create domain-specific topics
        config = ModelIntrospectionConfig(
            node_id=uuid4(),  # Generate unique UUID for each node instance
            node_type="COMPUTE",
            event_bus=event_bus,
            version="2.0.0",
            introspection_topic=f"onex.{domain}.introspection.published.v1",
            heartbeat_topic=f"onex.{domain}.heartbeat.published.v1",
            request_introspection_topic=f"onex.{domain}.introspection.requested.v1",
        )

        self.initialize_introspection_from_config(config)

    async def process_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mock compute processing.

        Args:
            data: Data to process.

        Returns:
            Processed result.
        """
        return {"processed": True, "input": data}


# =============================================================================
# Contract-Driven Configuration Tests
# =============================================================================


class TestContractToIntrospectionIntegration:
    """Test the full contract -> config -> mixin integration."""

    async def test_contract_driven_topic_configuration(self) -> None:
        """Verify topics from contract.yaml flow through to mixin."""
        # Simulate contract.yaml structure as defined in ONEX conventions
        contract_data = {
            "metadata": {
                "name": "test-effect-node",
                "version": "1.0.0",
                "node_type": "EFFECT",
            },
            "event_channels": {
                "publishes": [
                    {
                        "event_type": "introspection",
                        "topic": "onex.node.introspection.published.v1",
                    },
                    {
                        "event_type": "heartbeat",
                        "topic": "onex.node.heartbeat.published.v1",
                    },
                ],
                "subscribes": [
                    {
                        "event_type": "introspection_request",
                        "topic": "onex.registry.introspection.requested.v1",
                    },
                ],
            },
        }

        # Create node from contract data
        node = ContractDrivenEffectNode(contract_data)

        # Verify topics are set correctly from contract
        assert node._introspection_topic == "onex.node.introspection.published.v1"
        assert node._heartbeat_topic == "onex.node.heartbeat.published.v1"
        assert (
            node._request_introspection_topic
            == "onex.registry.introspection.requested.v1"
        )

        # Verify node metadata from contract
        # node_id is now a UUID (generated at initialization)
        from uuid import UUID

        assert isinstance(node._introspection_node_id, UUID)
        assert node._contract_node_name == "test-effect-node"
        assert node._introspection_node_type == "EFFECT"
        assert node._introspection_version == "1.0.0"

    async def test_contract_with_custom_domain_topics(self) -> None:
        """Verify custom domain-specific topics from contract are used."""
        # Simulate a domain-specific contract (e.g., for a payments service)
        contract_data = {
            "metadata": {
                "name": "payments-processor-node",
                "version": "2.1.0",
                "node_type": "COMPUTE",
            },
            "event_channels": {
                "publishes": [
                    {
                        "event_type": "introspection",
                        "topic": "onex.payments.introspection.published.v1",
                    },
                    {
                        "event_type": "heartbeat",
                        "topic": "onex.payments.heartbeat.published.v1",
                    },
                ],
                "subscribes": [
                    {
                        "event_type": "introspection_request",
                        "topic": "onex.payments.introspection.requested.v1",
                    },
                ],
            },
        }

        node = ContractDrivenEffectNode(contract_data)

        # Verify domain-specific topics
        assert node._introspection_topic == "onex.payments.introspection.published.v1"
        assert node._heartbeat_topic == "onex.payments.heartbeat.published.v1"
        assert (
            node._request_introspection_topic
            == "onex.payments.introspection.requested.v1"
        )

    async def test_contract_with_partial_event_channels(self) -> None:
        """Verify fallback to defaults when contract has partial event_channels."""
        # Contract with only introspection topic specified
        contract_data = {
            "metadata": {
                "name": "minimal-node",
                "version": "1.0.0",
                "node_type": "EFFECT",
            },
            "event_channels": {
                "publishes": [
                    {
                        "event_type": "introspection",
                        "topic": "onex.custom.introspection.topic.v1",
                    },
                ],
                "subscribes": [],
            },
        }

        node = ContractDrivenEffectNode(contract_data)

        # Custom topic should be set
        assert node._introspection_topic == "onex.custom.introspection.topic.v1"

        # Missing topics should fall back to defaults
        assert node._heartbeat_topic == DEFAULT_HEARTBEAT_TOPIC
        assert node._request_introspection_topic == DEFAULT_REQUEST_INTROSPECTION_TOPIC


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestEndToEndIntrospectionWorkflow:
    """Test complete introspection workflows with contract-driven configuration."""

    async def test_full_workflow_publish_introspection(self) -> None:
        """Verify complete workflow: contract -> config -> publish."""
        event_bus = MockEventBus()

        contract_data = {
            "metadata": {
                "name": "workflow-test-node",
                "version": "1.5.0",
                "node_type": "EFFECT",
            },
            "event_channels": {
                "publishes": [
                    {
                        "event_type": "introspection",
                        "topic": "onex.workflow.introspection.published.v1",
                    },
                    {
                        "event_type": "heartbeat",
                        "topic": "onex.workflow.heartbeat.published.v1",
                    },
                ],
                "subscribes": [],
            },
        }

        node = ContractDrivenEffectNode(contract_data, event_bus=event_bus)

        # Publish introspection event
        success = await node.publish_introspection(reason="startup")
        assert success is True

        # Verify event was published to correct topic from contract
        assert len(event_bus.published_envelopes) == 1
        envelope, topic = event_bus.published_envelopes[0]
        assert topic == "onex.workflow.introspection.published.v1"

        # Verify envelope content
        assert isinstance(envelope, ModelNodeIntrospectionEvent)
        # node_id is a UUID, verify it matches the node's internal ID
        assert envelope.node_id == node._introspection_node_id
        assert envelope.node_type == "EFFECT"
        assert envelope.version == "1.5.0"
        assert envelope.reason == "startup"

    async def test_introspection_data_reflects_node_capabilities(self) -> None:
        """Verify introspection data correctly reflects node capabilities."""
        contract_data = {
            "metadata": {
                "name": "capability-test-node",
                "version": "1.0.0",
                "node_type": "EFFECT",
            },
            "event_channels": {
                "publishes": [
                    {
                        "event_type": "introspection",
                        "topic": "onex.test.introspection.published.v1",
                    },
                ],
                "subscribes": [],
            },
        }

        node = ContractDrivenEffectNode(contract_data)

        # Get introspection data
        data = await node.get_introspection_data()

        # Verify node identification
        # node_id is a UUID, verify it matches the node's internal ID
        assert data.node_id == node._introspection_node_id
        assert data.node_type == "EFFECT"
        assert data.version == "1.0.0"

        # Verify capabilities were discovered
        assert "operations" in data.capabilities
        operations = data.capabilities["operations"]
        assert isinstance(operations, list)
        # The ContractDrivenEffectNode has execute_effect method
        assert "execute_effect" in operations

        # Verify protocols were discovered
        assert "protocols" in data.capabilities
        protocols = data.capabilities["protocols"]
        assert isinstance(protocols, list)
        assert "MixinNodeIntrospection" in protocols

        # Verify FSM state check
        assert "has_fsm" in data.capabilities

        # Verify endpoints were discovered
        assert "health" in data.endpoints
        assert data.endpoints["health"] == "http://localhost:8080/health"

        # Verify current state
        assert data.current_state == "initialized"

    async def test_heartbeat_uses_contract_topic(self) -> None:
        """Verify heartbeat publishing uses topic from contract."""
        event_bus = MockEventBus()

        contract_data = {
            "metadata": {
                "name": "heartbeat-test-node",
                "version": "1.0.0",
                "node_type": "EFFECT",
            },
            "event_channels": {
                "publishes": [
                    {
                        "event_type": "heartbeat",
                        "topic": "onex.custom.heartbeat.topic.v1",
                    },
                ],
                "subscribes": [],
            },
        }

        node = ContractDrivenEffectNode(contract_data, event_bus=event_bus)

        # Start heartbeat tasks with very short interval
        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.05,  # 50ms for fast test
            enable_registry_listener=False,
        )

        try:
            # Wait for at least one heartbeat
            await asyncio.sleep(CACHE_TTL_WAIT)

            # Verify heartbeat was published to correct topic
            heartbeat_topics = [
                topic
                for _, topic in event_bus.published_envelopes
                if "heartbeat" in topic.lower()
            ]
            assert len(heartbeat_topics) >= 1
            assert all(t == "onex.custom.heartbeat.topic.v1" for t in heartbeat_topics)
        finally:
            await node.stop_introspection_tasks()

    async def test_registry_listener_uses_contract_topic(self) -> None:
        """Verify registry listener subscribes to topic from contract."""
        event_bus = MockEventBus()

        contract_data = {
            "metadata": {
                "name": "listener-test-node",
                "version": "1.0.0",
                "node_type": "EFFECT",
            },
            "event_channels": {
                "publishes": [],
                "subscribes": [
                    {
                        "event_type": "introspection_request",
                        "topic": "onex.custom.registry.request.v1",
                    },
                ],
            },
        }

        node = ContractDrivenEffectNode(contract_data, event_bus=event_bus)

        # Start only registry listener
        await node.start_introspection_tasks(
            enable_heartbeat=False,
            enable_registry_listener=True,
        )

        try:
            # Give listener time to subscribe
            await asyncio.sleep(LISTENER_SUBSCRIBE_WAIT)

            # Verify subscription used correct topic from contract
            assert len(event_bus.subscribed_topics) >= 1
            assert "onex.custom.registry.request.v1" in event_bus.subscribed_topics

            # Verify group ID includes node ID (now a UUID)
            node_id_str = str(node._introspection_node_id)
            assert any(node_id_str in group for group in event_bus.subscribed_groups)
        finally:
            await node.stop_introspection_tasks()


# =============================================================================
# Multi-Domain and Multi-Channel Tests
# =============================================================================


class TestMultiDomainConfiguration:
    """Test multiple nodes with different domain configurations."""

    async def test_multiple_nodes_different_domains(self) -> None:
        """Verify multiple nodes can coexist with different domain topics."""
        payments_bus = MockEventBus()
        orders_bus = MockEventBus()

        # Create nodes for different domains
        payments_node = ComputeNodeWithCustomTopics(
            node_name="payments-compute",
            domain="payments",
            event_bus=payments_bus,
        )

        orders_node = ComputeNodeWithCustomTopics(
            node_name="orders-compute",
            domain="orders",
            event_bus=orders_bus,
        )

        # Verify each node has its own domain-specific topics
        assert (
            payments_node._introspection_topic
            == "onex.payments.introspection.published.v1"
        )
        assert (
            orders_node._introspection_topic == "onex.orders.introspection.published.v1"
        )

        assert payments_node._heartbeat_topic == "onex.payments.heartbeat.published.v1"
        assert orders_node._heartbeat_topic == "onex.orders.heartbeat.published.v1"

        # Publish from both nodes
        await payments_node.publish_introspection(reason="domain-test")
        await orders_node.publish_introspection(reason="domain-test")

        # Verify each published to its own domain topic
        assert len(payments_bus.published_envelopes) == 1
        _, payments_topic = payments_bus.published_envelopes[0]
        assert payments_topic == "onex.payments.introspection.published.v1"

        assert len(orders_bus.published_envelopes) == 1
        _, orders_topic = orders_bus.published_envelopes[0]
        assert orders_topic == "onex.orders.introspection.published.v1"

    async def test_contract_with_multiple_publish_channels(self) -> None:
        """Verify contract with additional non-introspection publish channels."""
        # Contracts often have other event channels beyond introspection
        contract_data = {
            "metadata": {
                "name": "multi-channel-node",
                "version": "1.0.0",
                "node_type": "EFFECT",
            },
            "event_channels": {
                "publishes": [
                    {
                        "event_type": "introspection",
                        "topic": "onex.multichannel.introspection.published.v1",
                    },
                    {
                        "event_type": "heartbeat",
                        "topic": "onex.multichannel.heartbeat.published.v1",
                    },
                    # Additional channels that are not introspection-related
                    {
                        "event_type": "data_processed",
                        "topic": "onex.multichannel.data.processed.v1",
                    },
                    {
                        "event_type": "error",
                        "topic": "onex.multichannel.error.published.v1",
                    },
                ],
                "subscribes": [
                    {
                        "event_type": "introspection_request",
                        "topic": "onex.multichannel.introspection.requested.v1",
                    },
                    # Additional subscribe channels
                    {
                        "event_type": "data_input",
                        "topic": "onex.multichannel.data.input.v1",
                    },
                ],
            },
        }

        node = ContractDrivenEffectNode(contract_data)

        # Introspection-specific topics should be extracted correctly
        assert (
            node._introspection_topic == "onex.multichannel.introspection.published.v1"
        )
        assert node._heartbeat_topic == "onex.multichannel.heartbeat.published.v1"
        assert (
            node._request_introspection_topic
            == "onex.multichannel.introspection.requested.v1"
        )


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestContractIntegrationEdgeCases:
    """Test edge cases in contract integration."""

    async def test_contract_with_empty_event_channels(self) -> None:
        """Verify handling of contract with no event channels."""
        contract_data = {
            "metadata": {
                "name": "no-channels-node",
                "version": "1.0.0",
                "node_type": "EFFECT",
            },
            "event_channels": {
                "publishes": [],
                "subscribes": [],
            },
        }

        node = ContractDrivenEffectNode(contract_data)

        # Should fall back to all defaults
        assert node._introspection_topic == DEFAULT_INTROSPECTION_TOPIC
        assert node._heartbeat_topic == DEFAULT_HEARTBEAT_TOPIC
        assert node._request_introspection_topic == DEFAULT_REQUEST_INTROSPECTION_TOPIC

    async def test_contract_with_missing_event_channels_key(self) -> None:
        """Verify handling of contract without event_channels key."""
        contract_data = {
            "metadata": {
                "name": "minimal-contract-node",
                "version": "1.0.0",
                "node_type": "EFFECT",
            },
            # No event_channels key at all
        }

        node = ContractDrivenEffectNode(contract_data)

        # Should fall back to all defaults
        assert node._introspection_topic == DEFAULT_INTROSPECTION_TOPIC
        assert node._heartbeat_topic == DEFAULT_HEARTBEAT_TOPIC

    async def test_introspection_cache_invalidation_workflow(self) -> None:
        """Verify cache invalidation works in contract-driven workflow."""
        contract_data = {
            "metadata": {
                "name": "cache-test-node",
                "version": "1.0.0",
                "node_type": "EFFECT",
            },
            "event_channels": {
                "publishes": [
                    {
                        "event_type": "introspection",
                        "topic": "onex.cache.introspection.published.v1",
                    },
                ],
                "subscribes": [],
            },
        }

        node = ContractDrivenEffectNode(contract_data)

        # First call populates cache
        data1 = await node.get_introspection_data()
        # node_id is a UUID, verify it matches the node's internal ID
        assert data1.node_id == node._introspection_node_id

        # Second call should return cached data (same correlation_id)
        data2 = await node.get_introspection_data()
        assert data2.correlation_id == data1.correlation_id

        # Invalidate cache (synchronous method)
        node.invalidate_introspection_cache()

        # Third call should generate fresh data (different correlation_id)
        data3 = await node.get_introspection_data()
        assert data3.correlation_id != data1.correlation_id

    async def test_correlation_id_propagation_through_workflow(self) -> None:
        """Verify correlation IDs flow through the entire workflow."""
        event_bus = MockEventBus()

        contract_data = {
            "metadata": {
                "name": "correlation-test-node",
                "version": "1.0.0",
                "node_type": "EFFECT",
            },
            "event_channels": {
                "publishes": [
                    {
                        "event_type": "introspection",
                        "topic": "onex.correlation.introspection.published.v1",
                    },
                ],
                "subscribes": [],
            },
        }

        node = ContractDrivenEffectNode(contract_data, event_bus=event_bus)

        # Publish with specific correlation ID
        test_correlation_id = uuid4()
        success = await node.publish_introspection(
            reason="correlation-test",
            correlation_id=test_correlation_id,
        )

        assert success is True
        assert len(event_bus.published_envelopes) == 1

        # Verify correlation ID was preserved
        envelope, _ = event_bus.published_envelopes[0]
        assert isinstance(envelope, ModelNodeIntrospectionEvent)
        assert envelope.correlation_id == test_correlation_id
        assert envelope.reason == "correlation-test"


# =============================================================================
# Performance and Metrics Tests
# =============================================================================


class TestContractIntegrationPerformance:
    """Test performance aspects of contract integration."""

    async def test_performance_metrics_available_after_introspection(self) -> None:
        """Verify performance metrics are captured in contract-driven workflow."""
        contract_data = {
            "metadata": {
                "name": "metrics-test-node",
                "version": "1.0.0",
                "node_type": "EFFECT",
            },
            "event_channels": {
                "publishes": [
                    {
                        "event_type": "introspection",
                        "topic": "onex.metrics.introspection.published.v1",
                    },
                ],
                "subscribes": [],
            },
        }

        node = ContractDrivenEffectNode(contract_data)

        # First call - cache miss
        await node.get_introspection_data()
        metrics1 = node.get_performance_metrics()

        assert metrics1 is not None
        assert metrics1.cache_hit is False
        assert metrics1.total_introspection_ms > 0
        assert metrics1.get_capabilities_ms >= 0
        assert metrics1.method_count > 0

        # Second call - cache hit (returns cached data without re-introspecting)
        await node.get_introspection_data()
        metrics2 = node.get_performance_metrics()

        assert metrics2 is not None
        assert metrics2.cache_hit is True

        # Cache hit performance assertion:
        # A cache hit should be faster than a cache miss because it skips the
        # expensive reflection operations (inspect.getmembers, signature analysis).
        #
        # However, in CI environments with variable load, timing can be noisy.
        # We use a robust comparison that passes if EITHER:
        # 1. Cache hit is faster than cache miss (expected behavior), OR
        # 2. Both are very fast (< 1ms), meaning timing noise dominates
        #
        # This avoids flakiness while still catching regressions where cache
        # hits become slower than cache misses (which would indicate a bug).
        cache_hit_faster = (
            metrics2.total_introspection_ms <= metrics1.total_introspection_ms
        )
        both_very_fast = (
            metrics1.total_introspection_ms < 1.0
            and metrics2.total_introspection_ms < 1.0
        )
        assert cache_hit_faster or both_very_fast, (
            f"Cache hit should be faster than cache miss. "
            f"Cache miss: {metrics1.total_introspection_ms:.3f}ms, "
            f"Cache hit: {metrics2.total_introspection_ms:.3f}ms"
        )


# =============================================================================
# Topic Validation Tests
# =============================================================================


class TestTopicValidation:
    """Test topic name validation in contract-driven configuration."""

    async def test_valid_topic_names_accepted(self) -> None:
        """Verify valid ONEX topic names are accepted."""
        valid_topics = [
            "onex.node.introspection.published.v1",
            "onex.payments.heartbeat.published.v1",
            "onex.orders.introspection.requested.v1",
            "onex.my-domain.sub_topic.published.v1",
            "onex.123.numeric.v1",
        ]

        for topic in valid_topics:
            # Should not raise
            config = ModelIntrospectionConfig(
                node_id=uuid4(),
                node_type="EFFECT",
                introspection_topic=topic,
            )
            assert config.introspection_topic == topic

    async def test_topic_starting_with_uppercase_rejected(self) -> None:
        """Verify topics starting with uppercase are rejected."""
        # Topic validation requires starting with a lowercase letter
        with pytest.raises(ValueError, match=r"must start with a lowercase letter"):
            ModelIntrospectionConfig(
                node_id=uuid4(),
                node_type="EFFECT",
                introspection_topic="Invalid.topic.v1",
            )

    async def test_invalid_topic_with_special_chars_rejected(self) -> None:
        """Verify topics with invalid characters are rejected."""
        # Use @ which is in the invalid_chars set to trigger the "invalid characters" error
        with pytest.raises(ValueError, match="invalid characters"):
            ModelIntrospectionConfig(
                node_id=uuid4(),
                node_type="EFFECT",
                introspection_topic="onex.invalid@topic.v1",
            )

    async def test_empty_topic_suffix_rejected(self) -> None:
        """Verify topic ending with a dot is rejected."""
        # Topic "onex." fails pattern validation because it ends with a dot
        # (pattern requires ending with alphanumeric character)
        with pytest.raises(ValueError, match="only lowercase alphanumeric"):
            ModelIntrospectionConfig(
                node_id=uuid4(),
                node_type="EFFECT",
                introspection_topic="onex.",
            )


# =============================================================================
# Subclass Override Tests
# =============================================================================


class TestSubclassTopicOverrides:
    """Test subclass override of default topics."""

    async def test_explicit_topic_config_overrides_defaults(self) -> None:
        """Verify explicitly configured topics override model defaults."""
        # Generate a test UUID for node identification
        test_node_id = uuid4()

        class TenantSpecificNode(MixinNodeIntrospection):
            """Node that explicitly configures tenant-specific topics."""

            def __init__(self, node_id: UUID) -> None:
                """Initialize tenant-specific node with explicit topic config."""
                self._state = "ready"
                config = ModelIntrospectionConfig(
                    node_id=node_id,
                    node_type="EFFECT",
                    introspection_topic="onex.tenant1.introspection.published.v1",
                    heartbeat_topic="onex.tenant1.heartbeat.published.v1",
                    request_introspection_topic="onex.tenant1.introspection.requested.v1",
                )
                self.initialize_introspection_from_config(config)

        node = TenantSpecificNode(node_id=test_node_id)

        # Verify explicitly configured topics are used
        assert node._introspection_topic == "onex.tenant1.introspection.published.v1"
        assert node._heartbeat_topic == "onex.tenant1.heartbeat.published.v1"
        assert (
            node._request_introspection_topic
            == "onex.tenant1.introspection.requested.v1"
        )

    async def test_partial_topic_config_uses_defaults_for_unspecified(self) -> None:
        """Verify partial config uses model defaults for unspecified topics."""
        # Generate test UUIDs for node identification
        test_node_id_1 = uuid4()
        test_node_id_2 = uuid4()

        class PartialConfigNode(MixinNodeIntrospection):
            """Node with partial topic configuration."""

            def __init__(
                self, node_id: UUID, introspection_topic: str | None = None
            ) -> None:
                """Initialize with optional topic override."""
                self._state = "ready"
                # Build config with optional topic override
                config_kwargs: dict[str, Any] = {
                    "node_id": node_id,
                    "node_type": "EFFECT",
                }
                if introspection_topic is not None:
                    config_kwargs["introspection_topic"] = introspection_topic
                config = ModelIntrospectionConfig(**config_kwargs)
                self.initialize_introspection_from_config(config)

        # Without topic override - uses model default
        node1 = PartialConfigNode(node_id=test_node_id_1)
        assert node1._introspection_topic == DEFAULT_INTROSPECTION_TOPIC

        # With topic override - uses provided value
        node2 = PartialConfigNode(
            node_id=test_node_id_2,
            introspection_topic="onex.override.introspection.published.v1",
        )
        assert node2._introspection_topic == "onex.override.introspection.published.v1"
