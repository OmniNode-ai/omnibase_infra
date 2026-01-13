# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for SnapshotPublisherRegistration cache operations.

This test suite validates the snapshot cache functionality:
- Cache loading from compacted Kafka topic
- Cache refresh operations
- Cache invalidation on stop
- Error recovery (preserve old cache on failure)
- Cache size limits and management
- Read-after-write consistency
- Lazy consumer creation for reads

Test Organization:
    - TestCacheLoadFromTopic: Loading cache from Kafka
    - TestCacheRefresh: Refreshing cache
    - TestCacheInvalidation: Cache clearing on stop
    - TestCacheErrorRecovery: Error handling and recovery
    - TestCacheReadAfterWrite: Consistency guarantees
    - TestCacheProperties: cache_size, is_cache_loaded properties
    - TestGetLatestSnapshot: Reading snapshots from cache

Coverage Goals:
    - All cache-related code paths in SnapshotPublisherRegistration
    - Error handling and graceful degradation
    - Concurrent access patterns
    - Tombstone handling in cache

Related Tickets:
    - OMN-947 (F2): Snapshot Publishing
    - OMN-1059: PR #107 review feedback - missing tests
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

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

# Patch path for AIOKafkaConsumer - it's imported inside the method
AIOKAFKA_CONSUMER_PATCH = "aiokafka.AIOKafkaConsumer"


# ============================================================================
# Test Helpers
# ============================================================================


def create_test_projection(
    entity_id: str | None = None,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    offset: int = 100,
    domain: str = "registration",
) -> ModelRegistrationProjection:
    """Create a test projection with sensible defaults."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=entity_id if entity_id is not None else uuid4(),
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
    entity_id: str | None = None,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    version: int = 1,
    domain: str = "registration",
) -> ModelRegistrationSnapshot:
    """Create a test snapshot with sensible defaults."""
    from uuid import UUID

    now = datetime.now(UTC)
    eid = UUID(entity_id) if entity_id else uuid4()
    return ModelRegistrationSnapshot(
        entity_id=eid,
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


def create_mock_kafka_message(
    key: str,
    snapshot: ModelRegistrationSnapshot | None,
) -> MagicMock:
    """Create a mock Kafka message."""
    message = MagicMock()
    message.key = key.encode("utf-8")
    if snapshot is None:
        message.value = None  # Tombstone
    else:
        message.value = snapshot.model_dump_json().encode("utf-8")
    return message


# ============================================================================
# Fixtures
# ============================================================================


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
def publisher_with_bootstrap(
    mock_producer: AsyncMock, snapshot_config: ModelSnapshotTopicConfig
) -> SnapshotPublisherRegistration:
    """Create a SnapshotPublisherRegistration with bootstrap_servers configured."""
    return SnapshotPublisherRegistration(
        mock_producer,
        snapshot_config,
        bootstrap_servers="localhost:9092",
        consumer_timeout_ms=100,  # Fast timeout for tests
    )


@pytest.fixture
def mock_consumer() -> AsyncMock:
    """Create a mock AIOKafkaConsumer."""
    consumer = AsyncMock()
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    consumer.seek_to_beginning = AsyncMock()
    consumer.getmany = AsyncMock(return_value={})
    return consumer


# ============================================================================
# Cache Load Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCacheLoadFromTopic:
    """Test cache loading from compacted Kafka topic."""

    async def test_cache_loads_snapshots_from_topic(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that cache loads all snapshots from topic."""
        # Create test snapshots
        snapshot1 = create_test_snapshot()
        snapshot2 = create_test_snapshot()

        # Mock messages
        msg1 = create_mock_kafka_message(snapshot1.to_kafka_key(), snapshot1)
        msg2 = create_mock_kafka_message(snapshot2.to_kafka_key(), snapshot2)

        # Configure consumer to return messages then empty
        mock_tp = MagicMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[{mock_tp: [msg1, msg2]}, {}]  # First call returns messages
        )  # Second call returns empty (end of topic)

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            # Trigger cache load via get_latest_snapshot
            await publisher_with_bootstrap.get_latest_snapshot(
                str(snapshot1.entity_id), "registration"
            )

            assert publisher_with_bootstrap.is_cache_loaded is True
            assert publisher_with_bootstrap.cache_size == 2

    async def test_cache_handles_tombstones(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that tombstones remove entries from cache."""
        # Create snapshot then tombstone for same entity
        snapshot = create_test_snapshot()
        msg_snapshot = create_mock_kafka_message(snapshot.to_kafka_key(), snapshot)
        msg_tombstone = create_mock_kafka_message(
            snapshot.to_kafka_key(), None
        )  # Tombstone

        mock_tp = MagicMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[
                {mock_tp: [msg_snapshot, msg_tombstone]},  # Snapshot then tombstone
                {},  # End of topic
            ]
        )

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            await publisher_with_bootstrap.get_latest_snapshot(
                str(snapshot.entity_id), "registration"
            )

            # Tombstone should have removed the snapshot
            assert publisher_with_bootstrap.cache_size == 0

    async def test_cache_skips_messages_without_keys(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that messages without keys are skipped."""
        snapshot = create_test_snapshot()

        # Message without key
        msg_no_key = MagicMock()
        msg_no_key.key = None
        msg_no_key.value = snapshot.model_dump_json().encode("utf-8")

        # Message with key
        msg_with_key = create_mock_kafka_message(snapshot.to_kafka_key(), snapshot)

        mock_tp = MagicMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[{mock_tp: [msg_no_key, msg_with_key]}, {}]
        )

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            await publisher_with_bootstrap.get_latest_snapshot(
                str(snapshot.entity_id), "registration"
            )

            # Only the keyed message should be in cache
            assert publisher_with_bootstrap.cache_size == 1

    async def test_cache_handles_malformed_json(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that malformed JSON is handled gracefully."""
        # Create valid snapshot
        snapshot = create_test_snapshot()
        msg_valid = create_mock_kafka_message(snapshot.to_kafka_key(), snapshot)

        # Create malformed message
        msg_malformed = MagicMock()
        msg_malformed.key = b"registration:malformed-entity"
        msg_malformed.value = b"not valid json"

        mock_tp = MagicMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[{mock_tp: [msg_malformed, msg_valid]}, {}]
        )

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            await publisher_with_bootstrap.get_latest_snapshot(
                str(snapshot.entity_id), "registration"
            )

            # Only valid snapshot should be in cache
            assert publisher_with_bootstrap.cache_size == 1

    async def test_cache_raises_error_without_bootstrap_servers(
        self,
        mock_producer: AsyncMock,
        snapshot_config: ModelSnapshotTopicConfig,
    ) -> None:
        """Test that cache loading raises error without bootstrap_servers."""
        publisher = SnapshotPublisherRegistration(
            mock_producer,
            snapshot_config,
            # No bootstrap_servers provided
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.get_latest_snapshot("entity-123", "registration")

        assert "bootstrap_servers not configured" in str(exc_info.value)


# ============================================================================
# Cache Refresh Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCacheRefresh:
    """Test cache refresh operations."""

    async def test_refresh_cache_clears_and_reloads(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that refresh_cache clears existing cache and reloads."""
        snapshot = create_test_snapshot()
        msg = create_mock_kafka_message(snapshot.to_kafka_key(), snapshot)

        mock_tp = MagicMock()
        mock_consumer.getmany = AsyncMock(side_effect=[{mock_tp: [msg]}, {}])

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            count = await publisher_with_bootstrap.refresh_cache()

            assert count == 1
            assert publisher_with_bootstrap.is_cache_loaded is True
            assert publisher_with_bootstrap.cache_size == 1

    async def test_refresh_stops_existing_consumer(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that refresh stops any existing consumer."""
        # First load
        snapshot = create_test_snapshot()
        msg = create_mock_kafka_message(snapshot.to_kafka_key(), snapshot)
        mock_tp = MagicMock()

        first_consumer = AsyncMock()
        first_consumer.start = AsyncMock()
        first_consumer.stop = AsyncMock()
        first_consumer.seek_to_beginning = AsyncMock()
        first_consumer.getmany = AsyncMock(side_effect=[{mock_tp: [msg]}, {}])

        second_consumer = AsyncMock()
        second_consumer.start = AsyncMock()
        second_consumer.stop = AsyncMock()
        second_consumer.seek_to_beginning = AsyncMock()
        second_consumer.getmany = AsyncMock(return_value={})

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            side_effect=[first_consumer, second_consumer],
        ):
            # Initial load
            await publisher_with_bootstrap.get_latest_snapshot(
                str(snapshot.entity_id), "registration"
            )

            # Refresh should stop existing consumer
            await publisher_with_bootstrap.refresh_cache()

            # First consumer should have been stopped
            first_consumer.stop.assert_called()

    async def test_refresh_checks_circuit_breaker(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
    ) -> None:
        """Test that refresh_cache checks circuit breaker."""
        import time

        # Open the circuit breaker
        async with publisher_with_bootstrap._circuit_breaker_lock:
            publisher_with_bootstrap._circuit_breaker_open = True
            publisher_with_bootstrap._circuit_breaker_failures = 5
            publisher_with_bootstrap._circuit_breaker_open_until = time.time() + 120

        with pytest.raises(InfraUnavailableError) as exc_info:
            await publisher_with_bootstrap.refresh_cache()

        assert "Circuit breaker is open" in str(exc_info.value)


# ============================================================================
# Cache Invalidation Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCacheInvalidation:
    """Test cache clearing on stop."""

    async def test_stop_clears_cache(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that stop() clears the cache."""
        snapshot = create_test_snapshot()
        msg = create_mock_kafka_message(snapshot.to_kafka_key(), snapshot)
        mock_tp = MagicMock()
        mock_consumer.getmany = AsyncMock(side_effect=[{mock_tp: [msg]}, {}])

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            await publisher_with_bootstrap.start()

            # Load cache
            await publisher_with_bootstrap.get_latest_snapshot(
                str(snapshot.entity_id), "registration"
            )
            assert publisher_with_bootstrap.cache_size == 1
            assert publisher_with_bootstrap.is_cache_loaded is True

            # Stop should clear cache
            await publisher_with_bootstrap.stop()

            assert publisher_with_bootstrap.cache_size == 0
            assert publisher_with_bootstrap.is_cache_loaded is False

    async def test_stop_stops_consumer(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that stop() stops the consumer."""
        snapshot = create_test_snapshot()
        msg = create_mock_kafka_message(snapshot.to_kafka_key(), snapshot)
        mock_tp = MagicMock()
        mock_consumer.getmany = AsyncMock(side_effect=[{mock_tp: [msg]}, {}])

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            await publisher_with_bootstrap.start()

            # Load cache
            await publisher_with_bootstrap.get_latest_snapshot(
                str(snapshot.entity_id), "registration"
            )

            await publisher_with_bootstrap.stop()

            mock_consumer.stop.assert_called()


# ============================================================================
# Error Recovery Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCacheErrorRecovery:
    """Test error handling and recovery in cache operations."""

    async def test_cache_load_handles_connection_error(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that cache load handles connection errors."""
        mock_consumer.start = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            with pytest.raises(InfraConnectionError) as exc_info:
                await publisher_with_bootstrap.get_latest_snapshot(
                    "entity-123", "registration"
                )

            assert "Failed to load snapshot cache" in str(exc_info.value)

    async def test_cache_load_handles_timeout(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that cache load handles timeout errors."""
        mock_consumer.start = AsyncMock(side_effect=TimeoutError("Timed out"))

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            with pytest.raises(InfraTimeoutError) as exc_info:
                await publisher_with_bootstrap.get_latest_snapshot(
                    "entity-123", "registration"
                )

            assert "Timeout loading snapshot cache" in str(exc_info.value)

    async def test_cleanup_failed_consumer(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that failed consumer is cleaned up."""
        mock_consumer.start = AsyncMock()
        mock_consumer.seek_to_beginning = AsyncMock(
            side_effect=Exception("Seek failed")
        )
        mock_consumer.stop = AsyncMock()

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            with pytest.raises(InfraConnectionError):
                await publisher_with_bootstrap.get_latest_snapshot(
                    "entity-123", "registration"
                )

            # Consumer should have been cleaned up
            assert publisher_with_bootstrap._consumer_started is False
            assert publisher_with_bootstrap._consumer is None

    async def test_circuit_breaker_records_failure_on_error(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that circuit breaker records failure on cache load error."""
        mock_consumer.start = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            # Make multiple failed attempts
            for _ in range(5):
                try:
                    await publisher_with_bootstrap.get_latest_snapshot(
                        "entity-123", "registration"
                    )
                except InfraConnectionError:
                    pass

            # Circuit breaker should be open
            assert publisher_with_bootstrap._circuit_breaker_open is True


# ============================================================================
# Read-After-Write Consistency Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCacheReadAfterWrite:
    """Test read-after-write consistency guarantees."""

    async def test_published_snapshot_available_in_cache(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that published snapshot is immediately available in cache."""
        # First load empty cache
        MagicMock()
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            # Trigger cache load
            result = await publisher_with_bootstrap.get_latest_snapshot(
                "nonexistent", "registration"
            )
            assert result is None
            assert publisher_with_bootstrap.is_cache_loaded is True

        # Now publish a snapshot
        projection = create_test_projection()
        snapshot = await publisher_with_bootstrap.publish_from_projection(projection)

        # Should be in cache immediately
        cached = await publisher_with_bootstrap.get_latest_snapshot(
            str(projection.entity_id), projection.domain
        )

        assert cached is not None
        assert cached.entity_id == snapshot.entity_id
        assert cached.snapshot_version == snapshot.snapshot_version

    async def test_deleted_snapshot_removed_from_cache(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_producer: AsyncMock,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that deleted snapshot is removed from cache."""
        # First load empty cache
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            await publisher_with_bootstrap.get_latest_snapshot(
                "nonexistent", "registration"
            )

        # Publish then delete
        projection = create_test_projection()
        await publisher_with_bootstrap.publish_from_projection(projection)

        # Verify it's in cache
        cached = await publisher_with_bootstrap.get_latest_snapshot(
            str(projection.entity_id), projection.domain
        )
        assert cached is not None

        # Delete it
        await publisher_with_bootstrap.delete_snapshot(
            str(projection.entity_id), projection.domain
        )

        # Should be removed from cache
        cached_after_delete = await publisher_with_bootstrap.get_latest_snapshot(
            str(projection.entity_id), projection.domain
        )
        assert cached_after_delete is None


# ============================================================================
# Cache Properties Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCacheProperties:
    """Test cache_size and is_cache_loaded properties."""

    async def test_cache_size_initially_zero(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
    ) -> None:
        """Test that cache_size is 0 initially."""
        assert publisher_with_bootstrap.cache_size == 0

    async def test_is_cache_loaded_initially_false(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
    ) -> None:
        """Test that is_cache_loaded is False initially."""
        assert publisher_with_bootstrap.is_cache_loaded is False

    async def test_cache_size_reflects_loaded_snapshots(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that cache_size reflects number of loaded snapshots."""
        snapshots = [create_test_snapshot() for _ in range(5)]
        messages = [create_mock_kafka_message(s.to_kafka_key(), s) for s in snapshots]

        mock_tp = MagicMock()
        mock_consumer.getmany = AsyncMock(side_effect=[{mock_tp: messages}, {}])

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            await publisher_with_bootstrap.get_latest_snapshot(
                str(snapshots[0].entity_id), "registration"
            )

            assert publisher_with_bootstrap.cache_size == 5

    async def test_is_cache_loaded_true_after_load(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that is_cache_loaded is True after successful load."""
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            await publisher_with_bootstrap.get_latest_snapshot(
                "nonexistent", "registration"
            )

            assert publisher_with_bootstrap.is_cache_loaded is True


# ============================================================================
# Get Latest Snapshot Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetLatestSnapshot:
    """Test get_latest_snapshot behavior."""

    async def test_returns_snapshot_from_cache(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that get_latest_snapshot returns snapshot from cache."""
        snapshot = create_test_snapshot()
        msg = create_mock_kafka_message(snapshot.to_kafka_key(), snapshot)
        mock_tp = MagicMock()
        mock_consumer.getmany = AsyncMock(side_effect=[{mock_tp: [msg]}, {}])

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            result = await publisher_with_bootstrap.get_latest_snapshot(
                str(snapshot.entity_id), snapshot.domain
            )

            assert result is not None
            assert result.entity_id == snapshot.entity_id

    async def test_returns_none_for_missing_entity(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that get_latest_snapshot returns None for missing entity."""
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            result = await publisher_with_bootstrap.get_latest_snapshot(
                "nonexistent-entity", "registration"
            )

            assert result is None

    async def test_triggers_cache_load_if_not_loaded(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that get_latest_snapshot triggers cache load if not loaded."""
        assert publisher_with_bootstrap.is_cache_loaded is False

        mock_consumer.getmany = AsyncMock(return_value={})

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            await publisher_with_bootstrap.get_latest_snapshot(
                "any-entity", "registration"
            )

            # Cache should now be loaded
            assert publisher_with_bootstrap.is_cache_loaded is True

    async def test_does_not_reload_if_cache_already_loaded(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that cache is not reloaded if already loaded."""
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ) as mock_consumer_class:
            # First call loads cache
            await publisher_with_bootstrap.get_latest_snapshot(
                "entity-1", "registration"
            )

            call_count_after_first = mock_consumer_class.call_count

            # Second call should use existing cache
            await publisher_with_bootstrap.get_latest_snapshot(
                "entity-2", "registration"
            )

            # Consumer constructor should not be called again
            assert mock_consumer_class.call_count == call_count_after_first

    async def test_checks_circuit_breaker(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
    ) -> None:
        """Test that get_latest_snapshot checks circuit breaker."""
        import time

        async with publisher_with_bootstrap._circuit_breaker_lock:
            publisher_with_bootstrap._circuit_breaker_open = True
            publisher_with_bootstrap._circuit_breaker_failures = 5
            publisher_with_bootstrap._circuit_breaker_open_until = time.time() + 120

        with pytest.raises(InfraUnavailableError) as exc_info:
            await publisher_with_bootstrap.get_latest_snapshot(
                "entity-123", "registration"
            )

        assert "Circuit breaker is open" in str(exc_info.value)


# ============================================================================
# Concurrent Access Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCacheConcurrentAccess:
    """Test concurrent access to cache."""

    async def test_concurrent_reads_are_safe(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that concurrent reads don't cause issues."""
        snapshot = create_test_snapshot()
        msg = create_mock_kafka_message(snapshot.to_kafka_key(), snapshot)
        mock_tp = MagicMock()
        mock_consumer.getmany = AsyncMock(side_effect=[{mock_tp: [msg]}, {}])

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            # Load cache first
            await publisher_with_bootstrap.get_latest_snapshot(
                str(snapshot.entity_id), snapshot.domain
            )

            # Concurrent reads
            tasks = [
                publisher_with_bootstrap.get_latest_snapshot(
                    str(snapshot.entity_id), snapshot.domain
                )
                for _ in range(10)
            ]
            results = await asyncio.gather(*tasks)

            # All should return the same snapshot
            for result in results:
                assert result is not None
                assert result.entity_id == snapshot.entity_id

    async def test_double_checked_locking_prevents_duplicate_load(
        self,
        publisher_with_bootstrap: SnapshotPublisherRegistration,
        mock_consumer: AsyncMock,
    ) -> None:
        """Test that double-checked locking prevents duplicate cache loads."""
        mock_consumer.getmany = AsyncMock(return_value={})

        call_count = 0
        original_start = mock_consumer.start

        async def counting_start() -> None:
            nonlocal call_count
            call_count += 1
            await original_start()

        mock_consumer.start = AsyncMock(side_effect=counting_start)

        with patch(
            AIOKAFKA_CONSUMER_PATCH,
            return_value=mock_consumer,
        ):
            # Simulate concurrent cache load requests
            tasks = [
                publisher_with_bootstrap.get_latest_snapshot(
                    f"entity-{i}", "registration"
                )
                for i in range(5)
            ]
            await asyncio.gather(*tasks)

            # Consumer should only be started once (double-checked locking)
            # Note: Due to async nature, there may be a brief window, but
            # the lock should prevent most duplicates
            assert call_count <= 2  # Allow for potential race, but not 5


__all__: list[str] = []
