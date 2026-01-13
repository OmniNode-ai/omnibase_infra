# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Comprehensive unit tests for SnapshotPublisherRegistration.

This test suite validates:
- Publisher instantiation with AIOKafkaProducer and config
- Circuit breaker initialization and configuration
- publish_snapshot method functionality
- publish_batch method behavior (empty, success, partial failures)
- delete_snapshot tombstone publishing
- publish_from_projection with version tracking
- Version tracking mechanics (increment, independence, clearing)
- Circuit breaker integration (threshold, reset, blocking)

Test Organization:
    - TestSnapshotPublisherInitialization: Instantiation and configuration
    - TestPublishSnapshot: Single snapshot publishing
    - TestPublishBatch: Batch publishing functionality
    - TestDeleteSnapshot: Tombstone publishing
    - TestPublishFromProjection: Projection to snapshot conversion
    - TestVersionTracking: Version tracking mechanics
    - TestCircuitBreakerIntegration: Circuit breaker behavior
    - TestStartStop: Lifecycle management

Coverage Goals:
    - >90% code coverage for snapshot publisher
    - All Kafka operation paths tested
    - Error handling validated
    - Circuit breaker integration tested

Related Tickets:
    - OMN-947 (F2): Snapshot Publishing
    - OMN-944 (F1): Implement Registration Projection Schema
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelRegistrationSnapshot,
    ModelSnapshotTopicConfig,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.projectors.snapshot_publisher_registration import (
    SnapshotPublisherRegistration,
)


def create_test_projection(
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    offset: int = 100,
    domain: str = "registration",
) -> ModelRegistrationProjection:
    """Create a test projection with sensible defaults."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=uuid4(),
        domain=domain,
        current_state=state,
        node_type="effect",
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        ack_deadline=now,
        liveness_deadline=now,
        last_applied_event_id=uuid4(),
        last_applied_offset=offset,
        registered_at=now,
        updated_at=now,
    )


def create_test_snapshot(
    entity_id: UUID | None = None,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    version: int = 1,
    domain: str = "registration",
) -> ModelRegistrationSnapshot:
    """Create a test snapshot with sensible defaults."""
    now = datetime.now(UTC)
    return ModelRegistrationSnapshot(
        entity_id=entity_id or uuid4(),
        domain=domain,
        current_state=state,
        node_type="effect",
        node_name="TestNode",
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        last_state_change_at=now,
        snapshot_version=version,
        snapshot_created_at=now,
        source_projection_sequence=100,
    )


@pytest.fixture
def mock_producer() -> AsyncMock:
    """Create a mock AIOKafkaProducer."""
    producer = AsyncMock()
    producer.send_and_wait = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    return producer


@pytest.fixture
def snapshot_config() -> ModelSnapshotTopicConfig:
    """Create a snapshot topic configuration."""
    return ModelSnapshotTopicConfig(
        topic="test.registration.snapshots",
        partition_count=6,
        replication_factor=1,
        cleanup_policy="compact",
    )


@pytest.fixture
def publisher(
    mock_producer: AsyncMock, snapshot_config: ModelSnapshotTopicConfig
) -> SnapshotPublisherRegistration:
    """Create a SnapshotPublisherRegistration instance with mocked producer."""
    return SnapshotPublisherRegistration(mock_producer, snapshot_config)


@pytest.mark.unit
@pytest.mark.asyncio
class TestSnapshotPublisherInitialization:
    """Test publisher instantiation and configuration."""

    async def test_initializes_with_config(
        self,
        publisher: SnapshotPublisherRegistration,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test that publisher initializes correctly with config."""
        assert publisher._config == snapshot_config
        assert publisher._producer is not None
        assert publisher._version_tracker == {}
        assert publisher._started is False

    async def test_initializes_with_custom_version_tracker(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test initialization with custom version tracker."""
        tracker = {"registration:entity-1": 5}
        pub = SnapshotPublisherRegistration(
            mock_producer, snapshot_config, snapshot_version_tracker=tracker
        )
        assert pub._version_tracker == tracker
        assert pub._version_tracker is tracker  # Same object reference

    async def test_circuit_breaker_initialized(
        self, publisher: SnapshotPublisherRegistration
    ) -> None:
        """Test that circuit breaker is initialized correctly."""
        assert hasattr(publisher, "_circuit_breaker_lock")
        assert publisher._circuit_breaker_failures == 0
        assert publisher._circuit_breaker_open is False

    async def test_circuit_breaker_config(
        self, publisher: SnapshotPublisherRegistration
    ) -> None:
        """Test circuit breaker configuration values."""
        # Default config: threshold=5, reset_timeout=60.0
        assert publisher.circuit_breaker_threshold == 5
        assert publisher.circuit_breaker_reset_timeout == 60.0
        assert "snapshot-publisher" in publisher.service_name

    async def test_topic_property(
        self,
        publisher: SnapshotPublisherRegistration,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test topic property returns configured topic."""
        assert publisher.topic == snapshot_config.topic

    async def test_is_started_property_initially_false(
        self, publisher: SnapshotPublisherRegistration
    ) -> None:
        """Test is_started is False before start() is called."""
        assert publisher.is_started is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestStartStop:
    """Test publisher lifecycle management."""

    async def test_start_success(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test successful start."""
        await publisher.start()

        assert publisher.is_started is True
        mock_producer.start.assert_called_once()

    async def test_start_already_started(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test start when already started is a no-op."""
        await publisher.start()
        await publisher.start()  # Second call should be no-op

        # start() should only be called once
        mock_producer.start.assert_called_once()

    async def test_start_connection_error(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test start handles connection errors."""
        mock_producer.start.side_effect = Exception("Connection refused")

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.start()

        assert "Failed to start Kafka producer" in str(exc_info.value)
        assert publisher.is_started is False

    async def test_stop_success(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test successful stop."""
        await publisher.start()
        await publisher.stop()

        assert publisher.is_started is False
        mock_producer.stop.assert_called_once()

    async def test_stop_not_started(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test stop when not started is a no-op."""
        await publisher.stop()

        mock_producer.stop.assert_not_called()
        assert publisher.is_started is False

    async def test_stop_handles_error_gracefully(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test stop handles errors without raising."""
        await publisher.start()
        mock_producer.stop.side_effect = Exception("Stop failed")

        # Should not raise
        await publisher.stop()

        assert publisher.is_started is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestPublishSnapshot:
    """Test publish_snapshot method functionality."""

    async def test_publishes_snapshot_successfully(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test successful single snapshot publish via publish_snapshot."""
        projection = create_test_projection()

        await publisher.publish_snapshot(projection)

        # Should have called send_and_wait
        mock_producer.send_and_wait.assert_called_once()
        call_args = mock_producer.send_and_wait.call_args
        assert call_args[0][0] == publisher.topic  # topic
        assert call_args[1]["key"] is not None  # key should be set
        assert call_args[1]["value"] is not None  # value should be set

    async def test_publish_snapshot_with_kafka_error(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test publish with Kafka error raises InfraConnectionError."""
        mock_producer.send_and_wait.side_effect = Exception("Kafka unavailable")
        projection = create_test_projection()

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.publish_snapshot(projection)

        assert "Failed to publish snapshot" in str(exc_info.value)

    async def test_publish_snapshot_with_timeout_error(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test publish with timeout raises InfraTimeoutError."""
        mock_producer.send_and_wait.side_effect = TimeoutError("Publish timed out")
        projection = create_test_projection()

        with pytest.raises(InfraTimeoutError) as exc_info:
            await publisher.publish_snapshot(projection)

        assert "Timeout publishing snapshot" in str(exc_info.value)

    async def test_publish_snapshot_creates_correct_key_value(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test that published message has correct key and value format."""
        projection = create_test_projection()

        await publisher.publish_snapshot(projection)

        call_args = mock_producer.send_and_wait.call_args
        key = call_args[1]["key"]
        value = call_args[1]["value"]

        # Key should be bytes in format "domain:entity_id"
        assert isinstance(key, bytes)
        key_str = key.decode("utf-8")
        assert key_str.startswith(f"{projection.domain}:")

        # Value should be JSON bytes
        assert isinstance(value, bytes)
        value_dict = json.loads(value.decode("utf-8"))
        assert value_dict["domain"] == projection.domain
        assert value_dict["current_state"] == projection.current_state.value


@pytest.mark.unit
@pytest.mark.asyncio
class TestPublishSnapshotModel:
    """Test _publish_snapshot_model internal method."""

    async def test_publishes_model_successfully(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test successful snapshot model publish."""
        snapshot = create_test_snapshot()

        await publisher._publish_snapshot_model(snapshot)

        mock_producer.send_and_wait.assert_called_once()

    async def test_circuit_breaker_resets_on_success(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test circuit breaker resets after successful publish."""
        # Simulate a previous failure
        async with publisher._circuit_breaker_lock:
            publisher._circuit_breaker_failures = 2

        snapshot = create_test_snapshot()
        await publisher._publish_snapshot_model(snapshot)

        # Circuit breaker should be reset
        assert publisher._circuit_breaker_failures == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestPublishBatch:
    """Test publish_batch method functionality."""

    async def test_empty_batch_returns_zero(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test empty batch returns 0 without any calls."""
        result = await publisher.publish_batch([])

        assert result == 0
        mock_producer.send_and_wait.assert_not_called()

    async def test_batch_all_success(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test batch with all successful publishes."""
        projections = [create_test_projection() for _ in range(3)]

        result = await publisher.publish_batch(projections)

        assert result == 3
        assert mock_producer.send_and_wait.call_count == 3

    async def test_batch_with_partial_failures(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test batch continues after individual failures, returns partial count."""
        projections = [create_test_projection() for _ in range(3)]

        # Second call fails
        mock_producer.send_and_wait.side_effect = [
            None,  # Success
            Exception("Kafka error"),  # Failure
            None,  # Success
        ]

        result = await publisher.publish_batch(projections)

        # Should return count of successful publishes
        assert result == 2
        assert mock_producer.send_and_wait.call_count == 3

    async def test_batch_all_failures_returns_zero(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test batch with all failures returns 0."""
        projections = [create_test_projection() for _ in range(3)]
        mock_producer.send_and_wait.side_effect = Exception("Kafka unavailable")

        result = await publisher.publish_batch(projections)

        assert result == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestPublishSnapshotBatch:
    """Test publish_snapshot_batch method for pre-built snapshots."""

    async def test_empty_snapshot_batch_returns_zero(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test empty snapshot batch returns 0."""
        result = await publisher.publish_snapshot_batch([])

        assert result == 0
        mock_producer.send_and_wait.assert_not_called()

    async def test_snapshot_batch_all_success(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test snapshot batch with all successful publishes."""
        snapshots = [create_test_snapshot(version=i) for i in range(1, 4)]

        result = await publisher.publish_snapshot_batch(snapshots)

        assert result == 3

    async def test_snapshot_batch_partial_failures(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test snapshot batch continues after failures."""
        snapshots = [create_test_snapshot(version=i) for i in range(1, 4)]

        mock_producer.send_and_wait.side_effect = [
            None,
            TimeoutError("Timeout"),
            None,
        ]

        result = await publisher.publish_snapshot_batch(snapshots)

        assert result == 2


@pytest.mark.unit
@pytest.mark.asyncio
class TestDeleteSnapshot:
    """Test delete_snapshot tombstone publishing."""

    async def test_successful_tombstone_publish(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test successful tombstone publish."""
        result = await publisher.delete_snapshot("entity-123", "registration")

        assert result is True
        mock_producer.send_and_wait.assert_called_once()

        # Verify tombstone has null value
        call_args = mock_producer.send_and_wait.call_args
        assert call_args[1]["value"] is None
        assert call_args[1]["key"] == b"registration:entity-123"

    async def test_version_tracker_cleared_after_delete(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test version tracker is cleared after delete."""
        # Pre-populate version tracker
        publisher._version_tracker["registration:entity-123"] = 5

        result = await publisher.delete_snapshot("entity-123", "registration")

        assert result is True
        assert "registration:entity-123" not in publisher._version_tracker

    async def test_delete_with_kafka_error_returns_false(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test delete with Kafka error returns False."""
        mock_producer.send_and_wait.side_effect = Exception("Kafka unavailable")

        result = await publisher.delete_snapshot("entity-123", "registration")

        assert result is False

    async def test_delete_records_circuit_failure_on_error(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test delete records circuit breaker failure on error."""
        mock_producer.send_and_wait.side_effect = Exception("Kafka unavailable")

        await publisher.delete_snapshot("entity-123", "registration")

        assert publisher._circuit_breaker_failures == 1


@pytest.mark.unit
@pytest.mark.asyncio
class TestPublishFromProjection:
    """Test publish_from_projection with version tracking."""

    async def test_creates_snapshot_with_version_one(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test first publish creates snapshot with version 1."""
        projection = create_test_projection()

        snapshot = await publisher.publish_from_projection(projection)

        assert snapshot.snapshot_version == 1
        assert snapshot.entity_id == projection.entity_id
        assert snapshot.domain == projection.domain
        assert snapshot.current_state == projection.current_state

    async def test_version_increments_on_successive_calls(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test version increments across multiple calls for same entity."""
        projection = create_test_projection()

        snapshot1 = await publisher.publish_from_projection(projection)
        snapshot2 = await publisher.publish_from_projection(projection)
        snapshot3 = await publisher.publish_from_projection(projection)

        assert snapshot1.snapshot_version == 1
        assert snapshot2.snapshot_version == 2
        assert snapshot3.snapshot_version == 3

    async def test_snapshot_has_correct_source_projection_sequence(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test snapshot has correct source_projection_sequence from offset."""
        projection = create_test_projection(offset=500)

        snapshot = await publisher.publish_from_projection(projection)

        assert snapshot.source_projection_sequence == 500

    async def test_snapshot_includes_node_name_when_provided(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test node_name is included when provided."""
        projection = create_test_projection()

        snapshot = await publisher.publish_from_projection(
            projection, node_name="PostgresAdapter"
        )

        assert snapshot.node_name == "PostgresAdapter"

    async def test_snapshot_has_correct_node_type(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test snapshot preserves node_type from projection."""
        projection = create_test_projection()

        snapshot = await publisher.publish_from_projection(projection)

        assert snapshot.node_type == projection.node_type

    async def test_snapshot_has_snapshot_created_at(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test snapshot has snapshot_created_at timestamp."""
        projection = create_test_projection()
        before = datetime.now(UTC)

        snapshot = await publisher.publish_from_projection(projection)

        after = datetime.now(UTC)
        assert before <= snapshot.snapshot_created_at <= after


@pytest.mark.unit
@pytest.mark.asyncio
class TestVersionTracking:
    """Test version tracking mechanics."""

    async def test_versions_increment_per_entity(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test versions increment for each entity independently."""
        entity_id = str(uuid4())

        # Call _get_next_version multiple times (now async)
        v1 = await publisher._get_next_version(entity_id, "registration")
        v2 = await publisher._get_next_version(entity_id, "registration")
        v3 = await publisher._get_next_version(entity_id, "registration")

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    async def test_different_entities_have_independent_versions(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test different entities track versions independently."""
        entity_a = str(uuid4())
        entity_b = str(uuid4())

        # Get versions for entity A (now async)
        v_a1 = await publisher._get_next_version(entity_a, "registration")
        v_a2 = await publisher._get_next_version(entity_a, "registration")

        # Get versions for entity B
        v_b1 = await publisher._get_next_version(entity_b, "registration")

        # Entity A should be at version 2, entity B should be at version 1
        assert v_a1 == 1
        assert v_a2 == 2
        assert v_b1 == 1

    async def test_different_domains_have_independent_versions(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test same entity in different domains has independent versions."""
        entity_id = str(uuid4())

        v_reg = await publisher._get_next_version(entity_id, "registration")
        v_disc = await publisher._get_next_version(entity_id, "discovery")

        assert v_reg == 1
        assert v_disc == 1

    async def test_version_cleared_after_delete(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test version is cleared after delete, starts fresh."""
        entity_id = str(uuid4())

        # Build up version (now async)
        await publisher._get_next_version(entity_id, "registration")
        await publisher._get_next_version(entity_id, "registration")
        assert publisher._version_tracker[f"registration:{entity_id}"] == 2

        # Delete clears the version
        await publisher.delete_snapshot(entity_id, "registration")

        # Next version should be 1 again
        v_new = await publisher._get_next_version(entity_id, "registration")
        assert v_new == 1


@pytest.mark.unit
@pytest.mark.asyncio
class TestCircuitBreakerIntegration:
    """Test circuit breaker behavior."""

    async def test_threshold_triggers_open_state(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test circuit breaker opens after threshold failures."""
        publisher = SnapshotPublisherRegistration(mock_producer, snapshot_config)
        mock_producer.send_and_wait.side_effect = Exception("Kafka unavailable")

        projection = create_test_projection()

        # Make 5 failed calls (default threshold)
        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await publisher.publish_snapshot(projection)

        # Circuit should now be open
        assert publisher._circuit_breaker_open is True
        assert publisher._circuit_breaker_failures >= 5

    async def test_infra_unavailable_error_when_circuit_open(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test InfraUnavailableError raised when circuit is open."""
        publisher = SnapshotPublisherRegistration(mock_producer, snapshot_config)
        mock_producer.send_and_wait.side_effect = Exception("Kafka unavailable")

        projection = create_test_projection()

        # Exhaust threshold
        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await publisher.publish_snapshot(projection)

        # Next call should be blocked by circuit breaker
        with pytest.raises(InfraUnavailableError) as exc_info:
            await publisher.publish_snapshot(projection)

        assert "Circuit breaker is open" in str(exc_info.value)

    async def test_reset_after_timeout(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test circuit breaker resets after timeout."""
        publisher = SnapshotPublisherRegistration(mock_producer, snapshot_config)

        # Open the circuit
        mock_producer.send_and_wait.side_effect = Exception("Kafka unavailable")
        projection = create_test_projection()

        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await publisher.publish_snapshot(projection)

        assert publisher._circuit_breaker_open is True

        # Simulate timeout elapsed by patching time
        import time

        with patch.object(time, "time", return_value=time.time() + 120):
            # Reset the producer to work
            mock_producer.send_and_wait.side_effect = None

            # This should succeed because timeout has passed
            # The circuit will transition to half-open and then closed
            await publisher.publish_snapshot(projection)

            assert publisher._circuit_breaker_open is False
            assert publisher._circuit_breaker_failures == 0

    async def test_circuit_breaker_on_delete_raises_unavailable(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test delete_snapshot raises InfraUnavailableError when circuit is open.

        Per ONEX fail-fast principles, circuit breaker errors should propagate
        so callers know the service is unavailable.
        """
        import time

        publisher = SnapshotPublisherRegistration(mock_producer, snapshot_config)

        # Open the circuit manually - must set open_until to a future time
        async with publisher._circuit_breaker_lock:
            publisher._circuit_breaker_open = True
            publisher._circuit_breaker_failures = 5
            publisher._circuit_breaker_open_until = (
                time.time() + 120
            )  # 2 minutes from now

        # Should raise InfraUnavailableError because circuit is open (fail-fast)
        with pytest.raises(InfraUnavailableError) as exc_info:
            await publisher.delete_snapshot("entity-123", "registration")

        assert "Circuit breaker is open" in str(exc_info.value)

    async def test_success_resets_circuit_breaker(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test successful operation resets circuit breaker."""
        # Simulate some failures (but not enough to open)
        async with publisher._circuit_breaker_lock:
            publisher._circuit_breaker_failures = 3

        projection = create_test_projection()
        await publisher.publish_snapshot(projection)

        # Circuit breaker should be reset
        assert publisher._circuit_breaker_failures == 0
        assert publisher._circuit_breaker_open is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetLatestSnapshot:
    """Test get_latest_snapshot behavior."""

    async def test_raises_connection_error_without_bootstrap_servers(
        self,
        publisher: SnapshotPublisherRegistration,
    ) -> None:
        """Test get_latest_snapshot raises InfraConnectionError when bootstrap_servers not configured.

        The publisher needs bootstrap_servers to create a consumer for reading snapshots.
        When not configured, it should raise InfraConnectionError with a helpful message.
        """
        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.get_latest_snapshot("entity-123", "registration")

        assert "bootstrap_servers not configured or empty" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestBootstrapServersValidation:
    """Test bootstrap_servers validation in get_latest_snapshot."""

    async def test_empty_string_bootstrap_servers(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test empty string bootstrap_servers raises InfraConnectionError."""
        publisher = SnapshotPublisherRegistration(
            mock_producer, snapshot_config, bootstrap_servers=""
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.get_latest_snapshot("entity-123", "registration")

        assert "bootstrap_servers not configured or empty" in str(exc_info.value)

    async def test_whitespace_only_bootstrap_servers(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test whitespace-only bootstrap_servers raises InfraConnectionError."""
        publisher = SnapshotPublisherRegistration(
            mock_producer, snapshot_config, bootstrap_servers="   "
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.get_latest_snapshot("entity-123", "registration")

        assert "bootstrap_servers not configured or empty" in str(exc_info.value)

    async def test_missing_port_in_bootstrap_servers(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test bootstrap_servers without port raises InfraConnectionError."""
        publisher = SnapshotPublisherRegistration(
            mock_producer, snapshot_config, bootstrap_servers="localhost"
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.get_latest_snapshot("entity-123", "registration")

        assert "Invalid bootstrap server format" in str(exc_info.value)
        assert "Expected 'host:port'" in str(exc_info.value)

    async def test_invalid_port_in_bootstrap_servers(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test bootstrap_servers with invalid port raises InfraConnectionError."""
        publisher = SnapshotPublisherRegistration(
            mock_producer, snapshot_config, bootstrap_servers="localhost:notaport"
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.get_latest_snapshot("entity-123", "registration")

        assert "Invalid port" in str(exc_info.value)
        assert "Port must be a valid integer" in str(exc_info.value)

    async def test_port_out_of_range_bootstrap_servers(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test bootstrap_servers with out-of-range port raises InfraConnectionError."""
        publisher = SnapshotPublisherRegistration(
            mock_producer, snapshot_config, bootstrap_servers="localhost:99999"
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.get_latest_snapshot("entity-123", "registration")

        assert "Invalid port 99999" in str(exc_info.value)
        assert "Port must be between 1 and 65535" in str(exc_info.value)

    async def test_empty_host_in_bootstrap_servers(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test bootstrap_servers with empty host raises InfraConnectionError."""
        publisher = SnapshotPublisherRegistration(
            mock_producer, snapshot_config, bootstrap_servers=":9092"
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.get_latest_snapshot("entity-123", "registration")

        assert "Host cannot be empty" in str(exc_info.value)

    async def test_empty_entry_in_comma_separated_bootstrap_servers(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test bootstrap_servers with empty entry in list raises InfraConnectionError."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092,,broker2:9092",
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.get_latest_snapshot("entity-123", "registration")

        assert "contains empty entries" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_publish_with_all_registration_states(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test publish works with all registration states."""
        for state in EnumRegistrationState:
            projection = create_test_projection(state=state)

            snapshot = await publisher.publish_from_projection(projection)

            assert snapshot.current_state == state

    async def test_publish_with_custom_domain(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test publish with custom domain namespace."""
        projection = create_test_projection(domain="custom_domain")

        snapshot = await publisher.publish_from_projection(projection)

        assert snapshot.domain == "custom_domain"
        # Verify key includes domain
        call_args = mock_producer.send_and_wait.call_args
        key = call_args[1]["key"].decode("utf-8")
        assert key.startswith("custom_domain:")

    async def test_publish_with_complex_capabilities(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test publish with complex capabilities object."""
        now = datetime.now(UTC)
        capabilities = ModelNodeCapabilities(
            postgres=True,
            read=True,
            write=True,
            database=True,
            transactions=True,
            batch_size=100,
            max_batch=1000,
            supported_types=["json", "csv", "xml"],
            config={"timeout": 30, "retry": 3},
        )

        projection = ModelRegistrationProjection(
            entity_id=uuid4(),
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            node_version=ModelSemVer.parse("1.0.0"),
            capabilities=capabilities,
            ack_deadline=now,
            liveness_deadline=now,
            last_applied_event_id=uuid4(),
            last_applied_offset=100,
            registered_at=now,
            updated_at=now,
        )

        snapshot = await publisher.publish_from_projection(projection)

        assert snapshot.capabilities is not None
        assert snapshot.capabilities.postgres is True
        assert snapshot.capabilities.batch_size == 100

    async def test_publish_with_none_node_name(
        self,
        publisher: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
    ) -> None:
        """Test publish with None node_name (default)."""
        projection = create_test_projection()

        snapshot = await publisher.publish_from_projection(projection, node_name=None)

        assert snapshot.node_name is None


@pytest.fixture
def mock_consumer() -> AsyncMock:
    """Create a mock AIOKafkaConsumer for cache tests."""
    consumer = AsyncMock()
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    consumer.seek_to_beginning = AsyncMock()
    consumer.getmany = AsyncMock(return_value={})
    return consumer


def create_mock_kafka_message(
    key: str,
    value: bytes | None,
    offset: int = 0,
) -> AsyncMock:
    """Create a mock Kafka message for testing cache loading."""
    message = AsyncMock()
    message.key = key.encode("utf-8")
    message.value = value
    message.offset = offset
    return message


@pytest.mark.unit
@pytest.mark.asyncio
class TestSnapshotCacheOperations:
    """Test snapshot cache operations with mocked Kafka consumer.

    These tests validate:
    - Cache miss (snapshot not found) returns None
    - Cache hit returns correct snapshot
    - Cache loading from Kafka topic
    - Cache loading handles tombstones correctly
    - Cache refresh preserves old cache on failure
    - Cache refresh clears and reloads on success
    - Consumer cleanup is idempotent
    - Cache warming doesn't fail startup
    - Cache size property returns correct count
    - is_cache_loaded property works correctly

    Related Tickets:
        - OMN-1059: Implement snapshot read functionality
        - PR #107: Add unit tests for snapshot cache operations
    """

    async def test_cache_miss_returns_none(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test get_latest_snapshot returns None when snapshot not found."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Mock consumer to return empty messages (no snapshots)
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            result = await publisher.get_latest_snapshot(
                "nonexistent-id", "registration"
            )

        assert result is None
        assert publisher.is_cache_loaded is True

    async def test_cache_hit_returns_correct_snapshot(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test get_latest_snapshot returns correct snapshot from cache."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        entity_id = uuid4()
        snapshot = create_test_snapshot(entity_id=entity_id, version=5)
        cache_key = f"registration:{entity_id}"

        # Create mock message with snapshot data
        message = create_mock_kafka_message(
            key=cache_key,
            value=snapshot.model_dump_json().encode("utf-8"),
        )

        # Mock consumer to return one message then empty
        mock_topic_partition = AsyncMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[
                {mock_topic_partition: [message]},  # First call returns message
                {},  # Second call returns empty (end of topic)
            ]
        )

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            result = await publisher.get_latest_snapshot(str(entity_id), "registration")

        assert result is not None
        assert result.entity_id == entity_id
        assert result.snapshot_version == 5
        assert publisher.cache_size == 1

    async def test_cache_loading_handles_tombstones_correctly(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test cache loading removes entries when tombstone (null value) encountered."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        entity_id = uuid4()
        cache_key = f"registration:{entity_id}"
        snapshot = create_test_snapshot(entity_id=entity_id, version=1)

        # Create messages: first a snapshot, then a tombstone
        snapshot_message = create_mock_kafka_message(
            key=cache_key,
            value=snapshot.model_dump_json().encode("utf-8"),
            offset=0,
        )
        tombstone_message = create_mock_kafka_message(
            key=cache_key,
            value=None,  # Tombstone
            offset=1,
        )

        mock_topic_partition = AsyncMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[
                {mock_topic_partition: [snapshot_message, tombstone_message]},
                {},  # End of topic
            ]
        )

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            result = await publisher.get_latest_snapshot(str(entity_id), "registration")

        # Tombstone should have removed the snapshot from cache
        assert result is None
        assert publisher.cache_size == 0

    async def test_cache_loading_keeps_latest_snapshot_after_tombstone(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test cache loading: snapshot after tombstone is retained."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        entity_id = uuid4()
        cache_key = f"registration:{entity_id}"
        snapshot_v1 = create_test_snapshot(entity_id=entity_id, version=1)
        snapshot_v2 = create_test_snapshot(entity_id=entity_id, version=2)

        # Create messages: snapshot v1, tombstone, snapshot v2
        messages = [
            create_mock_kafka_message(
                key=cache_key,
                value=snapshot_v1.model_dump_json().encode("utf-8"),
                offset=0,
            ),
            create_mock_kafka_message(
                key=cache_key,
                value=None,  # Tombstone
                offset=1,
            ),
            create_mock_kafka_message(
                key=cache_key,
                value=snapshot_v2.model_dump_json().encode("utf-8"),
                offset=2,
            ),
        ]

        mock_topic_partition = AsyncMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[
                {mock_topic_partition: messages},
                {},  # End of topic
            ]
        )

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            result = await publisher.get_latest_snapshot(str(entity_id), "registration")

        # Latest snapshot (v2) should be in cache
        assert result is not None
        assert result.snapshot_version == 2
        assert publisher.cache_size == 1

    async def test_cache_refresh_clears_and_reloads_on_success(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test refresh_cache clears old cache and reloads from topic."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Pre-populate cache with old data
        old_entity_id = uuid4()
        old_snapshot = create_test_snapshot(entity_id=old_entity_id, version=1)
        publisher._snapshot_cache[f"registration:{old_entity_id}"] = old_snapshot
        publisher._cache_loaded = True

        # New snapshot from topic
        new_entity_id = uuid4()
        new_snapshot = create_test_snapshot(entity_id=new_entity_id, version=1)
        new_message = create_mock_kafka_message(
            key=f"registration:{new_entity_id}",
            value=new_snapshot.model_dump_json().encode("utf-8"),
        )

        mock_topic_partition = AsyncMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[
                {mock_topic_partition: [new_message]},
                {},
            ]
        )

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            count = await publisher.refresh_cache()

        assert count == 1
        # Old entry should be gone
        assert f"registration:{old_entity_id}" not in publisher._snapshot_cache
        # New entry should be present
        assert f"registration:{new_entity_id}" in publisher._snapshot_cache

    async def test_cache_refresh_preserves_old_cache_on_failure(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test refresh_cache preserves existing cache when reload fails."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Pre-populate cache
        entity_id = uuid4()
        old_snapshot = create_test_snapshot(entity_id=entity_id, version=3)
        cache_key = f"registration:{entity_id}"
        publisher._snapshot_cache[cache_key] = old_snapshot
        publisher._cache_loaded = True

        # Consumer start fails
        mock_consumer.start.side_effect = Exception("Kafka connection failed")

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            with pytest.raises(InfraConnectionError):
                await publisher.refresh_cache()

        # Old cache should be preserved (graceful degradation)
        assert publisher._cache_loaded is True
        assert cache_key in publisher._snapshot_cache
        assert publisher._snapshot_cache[cache_key].snapshot_version == 3

    async def test_consumer_cleanup_is_idempotent(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test _cleanup_consumer is safe to call multiple times."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Cleanup when no consumer exists
        await publisher._cleanup_consumer()
        assert publisher._consumer is None
        assert publisher._consumer_started is False

        # Cleanup again - should be idempotent
        await publisher._cleanup_consumer()
        assert publisher._consumer is None
        assert publisher._consumer_started is False

    async def test_consumer_cleanup_handles_stop_error(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test _cleanup_consumer handles errors during consumer stop."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Set up consumer that fails on stop
        publisher._consumer = mock_consumer
        publisher._consumer_started = True
        mock_consumer.stop.side_effect = Exception("Stop failed")

        # Should not raise
        await publisher._cleanup_consumer()

        # State should be reset despite error
        assert publisher._consumer is None
        assert publisher._consumer_started is False

    async def test_cache_warming_does_not_fail_startup(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test start(warm_cache=True) does not fail when cache warming fails."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Consumer fails to start during cache warming
        mock_consumer.start.side_effect = Exception("Kafka connection failed")

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            # Should not raise despite cache warming failure
            await publisher.start(warm_cache=True)

        # Publisher should still be started
        assert publisher.is_started is True
        # Cache should not be loaded
        assert publisher.is_cache_loaded is False

    async def test_cache_warming_loads_cache_on_success(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test start(warm_cache=True) loads cache when successful."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        entity_id = uuid4()
        snapshot = create_test_snapshot(entity_id=entity_id)
        message = create_mock_kafka_message(
            key=f"registration:{entity_id}",
            value=snapshot.model_dump_json().encode("utf-8"),
        )

        mock_topic_partition = AsyncMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[
                {mock_topic_partition: [message]},
                {},
            ]
        )

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            await publisher.start(warm_cache=True)

        assert publisher.is_started is True
        assert publisher.is_cache_loaded is True
        assert publisher.cache_size == 1

    async def test_cache_size_property_returns_correct_count(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test cache_size property returns correct number of cached snapshots."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Initially empty
        assert publisher.cache_size == 0

        # Add some snapshots
        for i in range(1, 6):  # version must be >= 1
            entity_id = uuid4()
            snapshot = create_test_snapshot(entity_id=entity_id, version=i)
            publisher._snapshot_cache[f"registration:{entity_id}"] = snapshot

        assert publisher.cache_size == 5

    async def test_is_cache_loaded_property_initially_false(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test is_cache_loaded is False before any cache operations."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        assert publisher.is_cache_loaded is False

    async def test_is_cache_loaded_property_true_after_load(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test is_cache_loaded is True after successful cache load."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        mock_consumer.getmany = AsyncMock(return_value={})

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            await publisher.get_latest_snapshot("any-id", "registration")

        assert publisher.is_cache_loaded is True

    async def test_stop_clears_cache(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test stop() clears the snapshot cache."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Pre-populate cache
        entity_id = uuid4()
        snapshot = create_test_snapshot(entity_id=entity_id)
        publisher._snapshot_cache[f"registration:{entity_id}"] = snapshot
        publisher._cache_loaded = True
        publisher._started = True

        await publisher.stop()

        assert publisher.cache_size == 0
        assert publisher.is_cache_loaded is False

    async def test_cache_loading_skips_messages_without_keys(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test cache loading ignores messages with None keys."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Create message without key
        message_no_key = AsyncMock()
        message_no_key.key = None
        message_no_key.value = b'{"some": "data"}'

        # Create valid message
        entity_id = uuid4()
        snapshot = create_test_snapshot(entity_id=entity_id)
        valid_message = create_mock_kafka_message(
            key=f"registration:{entity_id}",
            value=snapshot.model_dump_json().encode("utf-8"),
        )

        mock_topic_partition = AsyncMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[
                {mock_topic_partition: [message_no_key, valid_message]},
                {},
            ]
        )

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            result = await publisher.get_latest_snapshot(str(entity_id), "registration")

        # Only valid message should be cached
        assert result is not None
        assert publisher.cache_size == 1

    async def test_cache_loading_handles_parse_errors_gracefully(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test cache loading continues when encountering malformed JSON."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Create malformed message
        bad_message = create_mock_kafka_message(
            key="registration:bad-entity",
            value=b"not valid json",
        )

        # Create valid message
        entity_id = uuid4()
        snapshot = create_test_snapshot(entity_id=entity_id)
        good_message = create_mock_kafka_message(
            key=f"registration:{entity_id}",
            value=snapshot.model_dump_json().encode("utf-8"),
        )

        mock_topic_partition = AsyncMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[
                {mock_topic_partition: [bad_message, good_message]},
                {},
            ]
        )

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            result = await publisher.get_latest_snapshot(str(entity_id), "registration")

        # Valid message should still be cached despite earlier parse error
        assert result is not None
        assert publisher.cache_size == 1

    async def test_cache_loading_timeout_raises_infra_timeout_error(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test cache loading timeout raises InfraTimeoutError."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        mock_consumer.start.side_effect = TimeoutError("Connection timed out")

        with patch("aiokafka.AIOKafkaConsumer", return_value=mock_consumer):
            with pytest.raises(InfraTimeoutError) as exc_info:
                await publisher.get_latest_snapshot("any-id", "registration")

        assert "Timeout loading snapshot cache" in str(exc_info.value)

    async def test_cache_is_not_reloaded_if_already_loaded(
        self,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test get_latest_snapshot uses existing cache without reloading."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Pre-populate cache and mark as loaded
        entity_id = uuid4()
        snapshot = create_test_snapshot(entity_id=entity_id, version=42)
        publisher._snapshot_cache[f"registration:{entity_id}"] = snapshot
        publisher._cache_loaded = True

        # Consumer should not be called since cache is already loaded
        with patch(
            "aiokafka.AIOKafkaConsumer", return_value=mock_consumer
        ) as mock_class:
            result = await publisher.get_latest_snapshot(str(entity_id), "registration")

        # Consumer should not have been created
        mock_class.assert_not_called()
        # Should return cached snapshot
        assert result is not None
        assert result.snapshot_version == 42

    async def test_publish_updates_cache_if_loaded(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test publish_from_projection updates cache for read-after-write consistency."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Mark cache as loaded (simulating previous read)
        publisher._cache_loaded = True

        projection = create_test_projection()
        await publisher.publish_from_projection(projection)

        # Cache should now contain the published snapshot
        cache_key = f"{projection.domain}:{projection.entity_id}"
        assert cache_key in publisher._snapshot_cache
        assert publisher._snapshot_cache[cache_key].snapshot_version == 1

    async def test_delete_removes_from_cache_if_loaded(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test delete_snapshot removes entry from cache."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        entity_id = uuid4()
        cache_key = f"registration:{entity_id}"

        # Pre-populate cache
        snapshot = create_test_snapshot(entity_id=entity_id)
        publisher._snapshot_cache[cache_key] = snapshot
        publisher._cache_loaded = True

        await publisher.delete_snapshot(str(entity_id), "registration")

        # Entry should be removed from cache
        assert cache_key not in publisher._snapshot_cache

    async def test_cache_loading_with_circuit_breaker_open(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test cache loading fails fast when circuit breaker is open."""
        import time

        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            bootstrap_servers="localhost:9092",
        )

        # Open the circuit breaker manually
        async with publisher._circuit_breaker_lock:
            publisher._circuit_breaker_open = True
            publisher._circuit_breaker_failures = 5
            publisher._circuit_breaker_open_until = time.time() + 120

        with pytest.raises(InfraUnavailableError) as exc_info:
            await publisher.get_latest_snapshot("any-id", "registration")

        assert "Circuit breaker is open" in str(exc_info.value)
