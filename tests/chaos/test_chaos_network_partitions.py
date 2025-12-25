# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Chaos tests for network partition scenarios (OMN-955).

This test suite validates system behavior during network partitions and
connectivity issues. It covers:

1. Event bus connection drop simulation
2. Kafka/message broker disconnect scenarios
3. Reconnection behavior after partition heals
4. Message delivery guarantees during partitions

Architecture:
    Network partitions can occur at various points:

    1. Producer-side partition: Publisher cannot reach broker
    2. Consumer-side partition: Subscriber cannot receive messages
    3. Broker partition: Broker is unreachable by all clients
    4. Split-brain: Some clients can reach broker, others cannot

    The system should:
    - Detect partitions quickly
    - Buffer messages when possible
    - Reconnect automatically when partition heals
    - Preserve message order and delivery guarantees

Test Organization:
    - TestEventBusConnectionDrop: Connection drop scenarios
    - TestPartitionDuringPublish: Partition during message publish
    - TestPartitionDuringConsume: Partition during message consumption
    - TestPartitionHealing: Reconnection after partition heals

Related Tickets:
    - OMN-955: Chaos scenario tests
    - OMN-954: Effect idempotency
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_infra.errors import InfraConnectionError, InfraUnavailableError
from tests.chaos.conftest import (
    MockEventBusWithPartition,
    NetworkPartitionSimulator,
)

# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.chaos
class TestEventBusConnectionDrop:
    """Test event bus connection drop scenarios."""

    @pytest.mark.asyncio
    async def test_connection_fails_during_partition(
        self,
        network_partition_simulator: NetworkPartitionSimulator,
        mock_event_bus_with_partition: MockEventBusWithPartition,
    ) -> None:
        """Test that connection attempts fail during active partition.

        When a network partition is active:
        - New connection attempts should fail
        - Existing connections should be invalidated
        - Connection attempts should be tracked
        """
        # Arrange - activate partition
        network_partition_simulator.start_partition()

        # Act & Assert
        with pytest.raises(InfraConnectionError, match="network partition"):
            await mock_event_bus_with_partition.start()

        # Connection attempt was tracked
        assert mock_event_bus_with_partition.connection_attempts == 1
        assert not mock_event_bus_with_partition.started

    @pytest.mark.asyncio
    async def test_connection_succeeds_without_partition(
        self,
        mock_event_bus_with_partition: MockEventBusWithPartition,
    ) -> None:
        """Test that connection succeeds when no partition is active.

        When there is no network partition:
        - Connection should succeed
        - Bus should be marked as started
        """
        # Act
        await mock_event_bus_with_partition.start()

        # Assert
        assert mock_event_bus_with_partition.started
        health = await mock_event_bus_with_partition.health_check()
        assert health["healthy"] is True

        # Cleanup
        await mock_event_bus_with_partition.close()

    @pytest.mark.asyncio
    async def test_multiple_connection_attempts_during_partition(
        self,
        network_partition_simulator: NetworkPartitionSimulator,
        mock_event_bus_with_partition: MockEventBusWithPartition,
    ) -> None:
        """Test multiple connection attempts during partition.

        Multiple connection attempts during a partition should all fail
        and be tracked.
        """
        # Arrange
        network_partition_simulator.start_partition()
        num_attempts = 3

        # Act
        for _ in range(num_attempts):
            try:
                await mock_event_bus_with_partition.start()
            except InfraConnectionError:
                pass

        # Assert
        assert mock_event_bus_with_partition.connection_attempts == num_attempts


@pytest.mark.chaos
class TestPartitionDuringPublish:
    """Test partition scenarios during message publish."""

    @pytest.mark.asyncio
    async def test_publish_fails_during_partition(
        self,
        network_partition_simulator: NetworkPartitionSimulator,
        started_event_bus_with_partition: MockEventBusWithPartition,
    ) -> None:
        """Test that publish fails when partition occurs.

        When a partition occurs during publish:
        - The publish operation should fail
        - An appropriate error should be raised
        - The message should not be recorded as published
        """
        # Arrange - bus is started, then partition occurs
        network_partition_simulator.start_partition()

        # Act & Assert
        with pytest.raises(InfraConnectionError, match="network partition"):
            await started_event_bus_with_partition.publish(
                topic="test-topic",
                key=b"test-key",
                value=b"test-value",
            )

        # Message should not be in published list
        assert len(started_event_bus_with_partition.published_messages) == 0

    @pytest.mark.asyncio
    async def test_publish_fails_when_not_started(
        self,
        mock_event_bus_with_partition: MockEventBusWithPartition,
    ) -> None:
        """Test that publish fails when bus not started.

        When attempting to publish on an unstarted bus:
        - Should raise InfraUnavailableError
        - Should not attempt network operations
        """
        # Act & Assert
        with pytest.raises(InfraUnavailableError, match="not started"):
            await mock_event_bus_with_partition.publish(
                topic="test-topic",
                key=None,
                value=b"test-value",
            )

    @pytest.mark.asyncio
    async def test_publish_succeeds_after_partition_heals(
        self,
        network_partition_simulator: NetworkPartitionSimulator,
        started_event_bus_with_partition: MockEventBusWithPartition,
    ) -> None:
        """Test that publish succeeds after partition heals.

        After a partition heals:
        - Publish operations should succeed
        - Messages should be delivered correctly
        """
        # Arrange - start partition
        network_partition_simulator.start_partition()

        # Verify publish fails during partition
        with pytest.raises(InfraConnectionError):
            await started_event_bus_with_partition.publish(
                topic="test-topic",
                key=None,
                value=b"during-partition",
            )

        # Heal partition
        network_partition_simulator.end_partition()

        # Act - publish after healing
        await started_event_bus_with_partition.publish(
            topic="test-topic",
            key=None,
            value=b"after-healing",
        )

        # Assert
        assert len(started_event_bus_with_partition.published_messages) == 1
        assert (
            started_event_bus_with_partition.published_messages[0]["value"]
            == b"after-healing"
        )

    @pytest.mark.asyncio
    async def test_concurrent_publishes_during_partition_all_fail(
        self,
        network_partition_simulator: NetworkPartitionSimulator,
        started_event_bus_with_partition: MockEventBusWithPartition,
    ) -> None:
        """Test that concurrent publishes during partition all fail.

        When multiple publish operations occur during a partition:
        - All should fail
        - None should be recorded
        """
        # Arrange
        network_partition_simulator.start_partition()
        num_concurrent = 10
        errors: list[Exception] = []
        lock = asyncio.Lock()

        async def try_publish(i: int) -> None:
            try:
                await started_event_bus_with_partition.publish(
                    topic="test-topic",
                    key=None,
                    value=f"message-{i}".encode(),
                )
            except InfraConnectionError as e:
                async with lock:
                    errors.append(e)

        # Act
        await asyncio.gather(*[try_publish(i) for i in range(num_concurrent)])

        # Assert
        assert len(errors) == num_concurrent
        assert len(started_event_bus_with_partition.published_messages) == 0


@pytest.mark.chaos
class TestPartitionDuringConsume:
    """Test partition scenarios during message consumption."""

    @pytest.mark.asyncio
    async def test_subscriber_receives_message_before_partition(
        self,
        started_event_bus_with_partition: MockEventBusWithPartition,
    ) -> None:
        """Test that subscriber receives messages before partition.

        When no partition exists:
        - Subscribers should receive published messages
        - Message content should be preserved
        """
        # Arrange
        received_messages: list[dict] = []

        async def handler(msg: dict) -> None:
            received_messages.append(msg)

        await started_event_bus_with_partition.subscribe(
            topic="test-topic",
            group="test-group",
            handler=handler,
        )

        # Act
        await started_event_bus_with_partition.publish(
            topic="test-topic",
            key=None,
            value=b"test-message",
        )

        # Assert
        assert len(received_messages) == 1
        assert received_messages[0]["value"] == b"test-message"

    @pytest.mark.asyncio
    async def test_subscription_persists_through_partition(
        self,
        network_partition_simulator: NetworkPartitionSimulator,
        started_event_bus_with_partition: MockEventBusWithPartition,
    ) -> None:
        """Test that subscriptions persist through a partition.

        Subscriptions should remain active after a partition heals:
        - Subscriber should still receive messages
        - No re-subscription should be needed
        """
        # Arrange
        received_messages: list[dict] = []

        async def handler(msg: dict) -> None:
            received_messages.append(msg)

        await started_event_bus_with_partition.subscribe(
            topic="test-topic",
            group="test-group",
            handler=handler,
        )

        # Publish before partition
        await started_event_bus_with_partition.publish(
            topic="test-topic",
            key=None,
            value=b"before-partition",
        )
        assert len(received_messages) == 1

        # Start and end partition
        network_partition_simulator.start_partition()
        network_partition_simulator.end_partition()

        # Publish after partition heals
        await started_event_bus_with_partition.publish(
            topic="test-topic",
            key=None,
            value=b"after-partition",
        )

        # Assert - subscription still works
        assert len(received_messages) == 2
        assert received_messages[1]["value"] == b"after-partition"


@pytest.mark.chaos
class TestPartitionHealing:
    """Test reconnection behavior after partition heals."""

    @pytest.mark.asyncio
    async def test_reconnection_callback_invoked_on_healing(
        self,
        network_partition_simulator: NetworkPartitionSimulator,
    ) -> None:
        """Test that reconnection callbacks are invoked when partition heals.

        When a partition heals:
        - Registered reconnection callbacks should be invoked
        - Callbacks should be invoked in order
        """
        # Arrange
        callback_invocations: list[str] = []

        callback1 = AsyncMock(side_effect=lambda: callback_invocations.append("cb1"))
        callback2 = AsyncMock(side_effect=lambda: callback_invocations.append("cb2"))

        network_partition_simulator.add_reconnection_callback(callback1)
        network_partition_simulator.add_reconnection_callback(callback2)

        # Start partition
        network_partition_simulator.start_partition()
        assert network_partition_simulator.is_partitioned

        # Act - heal partition
        await network_partition_simulator.simulate_partition_healing(duration_ms=10)

        # Assert
        assert not network_partition_simulator.is_partitioned
        assert callback_invocations == ["cb1", "cb2"]
        callback1.assert_called_once()
        callback2.assert_called_once()

    @pytest.mark.asyncio
    async def test_healing_restores_bus_operations(
        self,
        network_partition_simulator: NetworkPartitionSimulator,
    ) -> None:
        """Test that healing restores normal bus operations.

        After a partition heals:
        - Bus should be able to start
        - Publish operations should succeed
        - Health check should show healthy
        """
        # Arrange - start partition
        network_partition_simulator.start_partition()
        bus = MockEventBusWithPartition(network_partition_simulator)

        # Verify can't start during partition
        with pytest.raises(InfraConnectionError):
            await bus.start()

        # Heal partition
        network_partition_simulator.end_partition()

        # Act - start bus after healing
        await bus.start()

        # Assert
        health = await bus.health_check()
        assert health["healthy"] is True
        assert health["partitioned"] is False

        # Can publish
        await bus.publish("test-topic", None, b"test")
        assert len(bus.published_messages) == 1

        # Cleanup
        await bus.close()

    @pytest.mark.asyncio
    async def test_health_check_reflects_partition_status(
        self,
        network_partition_simulator: NetworkPartitionSimulator,
        started_event_bus_with_partition: MockEventBusWithPartition,
    ) -> None:
        """Test that health check accurately reflects partition status.

        Health check should:
        - Show healthy=False during partition
        - Show healthy=True after partition heals
        - Include partition status information
        """
        # Check healthy before partition
        health = await started_event_bus_with_partition.health_check()
        assert health["healthy"] is True
        assert health["partitioned"] is False

        # Start partition
        network_partition_simulator.start_partition()

        # Check during partition
        health = await started_event_bus_with_partition.health_check()
        assert health["healthy"] is False
        assert health["partitioned"] is True

        # Heal partition
        network_partition_simulator.end_partition()

        # Check after healing
        health = await started_event_bus_with_partition.health_check()
        assert health["healthy"] is True
        assert health["partitioned"] is False

    @pytest.mark.asyncio
    async def test_partition_timing_is_tracked(
        self,
        network_partition_simulator: NetworkPartitionSimulator,
    ) -> None:
        """Test that partition timing information is tracked.

        The simulator should track when a partition started for
        diagnostics and metrics purposes.
        """
        # Initially no partition
        assert network_partition_simulator.partition_start_time is None

        # Start partition
        network_partition_simulator.start_partition()
        assert network_partition_simulator.partition_start_time is not None
        start_time = network_partition_simulator.partition_start_time

        # Verify time is reasonable (recent)
        import asyncio

        current_time = asyncio.get_event_loop().time()
        assert current_time - start_time < 1.0  # Within 1 second

        # End partition
        network_partition_simulator.end_partition()
        assert network_partition_simulator.partition_start_time is None
