# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for DLQ PostgreSQL tracking service.

These tests validate DLQTrackingService behavior against actual PostgreSQL
infrastructure running on the remote infrastructure server. They require
proper database credentials and will be skipped gracefully if the database
is not available.

CI/CD Graceful Skip Behavior
============================

These tests skip gracefully in CI/CD environments without database access:

Skip Conditions:
    - Skips if POSTGRES_HOST not set
    - Skips if POSTGRES_PASSWORD not set
    - Module-level ``pytestmark`` with ``pytest.mark.skipif`` used

Example CI/CD Output::

    $ pytest tests/integration/dlq/test_dlq_tracking_integration.py -v
    test_initialize_creates_table SKIPPED (PostgreSQL not available)
    test_record_replay_attempt_success SKIPPED (PostgreSQL not available)

Test Categories
===============

- Initialization Tests: Validate service startup and table creation
- Record Tests: Verify replay attempt recording
- Query Tests: Validate history retrieval and ordering
- Health Check Tests: Verify service health monitoring

Environment Variables
=====================

    POSTGRES_HOST: PostgreSQL server hostname (required)
    POSTGRES_PORT: PostgreSQL server port (default: 5436)
    POSTGRES_DATABASE: Database name (default: omninode_bridge)
    POSTGRES_USER: Database username (default: postgres)
    POSTGRES_PASSWORD: Database password (required)

Related Ticket: OMN-1032 - Complete DLQ Replay PostgreSQL Tracking Integration
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from omnibase_infra.dlq import (
    DLQTrackingService,
    EnumReplayStatus,
    ModelDlqReplayRecord,
    ModelDlqTrackingConfig,
)

from .conftest import POSTGRES_AVAILABLE

# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Module-level markers - skip all tests if PostgreSQL is not available
pytestmark = [
    pytest.mark.skipif(
        not POSTGRES_AVAILABLE,
        reason="PostgreSQL not available (POSTGRES_PASSWORD not set)",
    ),
]


# =============================================================================
# Initialization Tests - Validate service startup and table creation
# =============================================================================


class TestDLQTrackingServiceInitialization:
    """Tests for DLQTrackingService initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_creates_table(
        self,
        dlq_tracking_service: DLQTrackingService,
        dlq_tracking_config: ModelDlqTrackingConfig,
    ) -> None:
        """Test that initialize creates the tracking table.

        Verifies that:
        1. Connection pool is created successfully
        2. Table is created with correct name
        3. Table exists in information_schema
        """
        # Table should exist after initialization
        assert dlq_tracking_service._pool is not None
        assert dlq_tracking_service.is_initialized is True

        async with dlq_tracking_service._pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT 1 FROM information_schema.tables WHERE table_name = $1",
                dlq_tracking_config.storage_table,
            )
            assert result == 1, (
                f"Table {dlq_tracking_config.storage_table} should exist after init"
            )

    @pytest.mark.asyncio
    async def test_initialize_creates_indexes(
        self,
        dlq_tracking_service: DLQTrackingService,
        dlq_tracking_config: ModelDlqTrackingConfig,
    ) -> None:
        """Test that initialize creates required indexes.

        Verifies that:
        1. Index on original_message_id exists
        2. Index on replay_timestamp exists
        """
        assert dlq_tracking_service._pool is not None

        async with dlq_tracking_service._pool.acquire() as conn:
            # Check for message_id index
            message_id_index = await conn.fetchval(
                """
                SELECT 1 FROM pg_indexes
                WHERE indexname = $1
                """,
                f"idx_{dlq_tracking_config.storage_table}_message_id",
            )
            assert message_id_index == 1, "Index on original_message_id should exist"

            # Check for timestamp index
            timestamp_index = await conn.fetchval(
                """
                SELECT 1 FROM pg_indexes
                WHERE indexname = $1
                """,
                f"idx_{dlq_tracking_config.storage_table}_timestamp",
            )
            assert timestamp_index == 1, "Index on replay_timestamp should exist"

    @pytest.mark.asyncio
    async def test_initialize_idempotent(
        self,
        dlq_tracking_service: DLQTrackingService,
    ) -> None:
        """Test that calling initialize multiple times is safe.

        The service should handle repeated initialization calls gracefully
        without raising errors or creating duplicate resources.
        """
        # Already initialized by fixture
        assert dlq_tracking_service.is_initialized is True

        # Call initialize again - should be a no-op
        await dlq_tracking_service.initialize()

        # Still should be initialized and functional
        assert dlq_tracking_service.is_initialized is True
        health = await dlq_tracking_service.health_check()
        assert health is True


# =============================================================================
# Record Tests - Verify replay attempt recording
# =============================================================================


class TestDLQTrackingServiceRecord:
    """Tests for recording replay attempts."""

    @pytest.mark.asyncio
    async def test_record_replay_attempt_success(
        self,
        dlq_tracking_service: DLQTrackingService,
    ) -> None:
        """Test recording a successful replay attempt.

        Verifies that:
        1. Record is inserted into database
        2. All fields are correctly persisted
        3. Record can be retrieved via get_replay_history
        """
        record = ModelDlqReplayRecord(
            id=uuid4(),
            original_message_id=uuid4(),
            replay_correlation_id=uuid4(),
            original_topic="test.events.v1",
            target_topic="test.events.v1",
            replay_status=EnumReplayStatus.COMPLETED,
            replay_timestamp=datetime.now(UTC),
            success=True,
            error_message=None,
            dlq_offset=100,
            dlq_partition=0,
            retry_count=1,
        )

        await dlq_tracking_service.record_replay_attempt(record)

        # Verify record was inserted
        history = await dlq_tracking_service.get_replay_history(
            record.original_message_id
        )
        assert len(history) == 1
        assert history[0].id == record.id
        assert history[0].replay_status == EnumReplayStatus.COMPLETED
        assert history[0].success is True
        assert history[0].error_message is None
        assert history[0].dlq_offset == 100
        assert history[0].dlq_partition == 0
        assert history[0].retry_count == 1

    @pytest.mark.asyncio
    async def test_record_replay_attempt_failure(
        self,
        dlq_tracking_service: DLQTrackingService,
    ) -> None:
        """Test recording a failed replay attempt.

        Verifies that:
        1. Failed status is correctly persisted
        2. Error message is stored
        3. success=False is recorded
        """
        record = ModelDlqReplayRecord(
            id=uuid4(),
            original_message_id=uuid4(),
            replay_correlation_id=uuid4(),
            original_topic="test.events.v1",
            target_topic="test.events.v1",
            replay_status=EnumReplayStatus.FAILED,
            replay_timestamp=datetime.now(UTC),
            success=False,
            error_message="Connection timeout after 30s",
            dlq_offset=101,
            dlq_partition=0,
            retry_count=2,
        )

        await dlq_tracking_service.record_replay_attempt(record)

        history = await dlq_tracking_service.get_replay_history(
            record.original_message_id
        )
        assert len(history) == 1
        assert history[0].replay_status == EnumReplayStatus.FAILED
        assert history[0].success is False
        assert history[0].error_message == "Connection timeout after 30s"
        assert history[0].retry_count == 2

    @pytest.mark.asyncio
    async def test_record_replay_attempt_skipped(
        self,
        dlq_tracking_service: DLQTrackingService,
    ) -> None:
        """Test recording a skipped replay attempt.

        Skipped status is used for non-retryable messages that should
        not be replayed (e.g., malformed messages, business rule violations).
        """
        record = ModelDlqReplayRecord(
            id=uuid4(),
            original_message_id=uuid4(),
            replay_correlation_id=uuid4(),
            original_topic="test.commands.v1",
            target_topic="test.commands.v1",
            replay_status=EnumReplayStatus.SKIPPED,
            replay_timestamp=datetime.now(UTC),
            success=False,
            error_message="Non-retryable: message payload corrupted",
            dlq_offset=102,
            dlq_partition=1,
            retry_count=0,
        )

        await dlq_tracking_service.record_replay_attempt(record)

        history = await dlq_tracking_service.get_replay_history(
            record.original_message_id
        )
        assert len(history) == 1
        assert history[0].replay_status == EnumReplayStatus.SKIPPED
        assert history[0].success is False
        assert "Non-retryable" in str(history[0].error_message)

    @pytest.mark.asyncio
    async def test_record_replay_attempt_pending(
        self,
        dlq_tracking_service: DLQTrackingService,
    ) -> None:
        """Test recording a pending replay attempt.

        Pending status represents a replay that has been initiated
        but not yet completed.
        """
        record = ModelDlqReplayRecord(
            id=uuid4(),
            original_message_id=uuid4(),
            replay_correlation_id=uuid4(),
            original_topic="test.intents.v1",
            target_topic="test.intents.v1",
            replay_status=EnumReplayStatus.PENDING,
            replay_timestamp=datetime.now(UTC),
            success=False,
            error_message=None,
            dlq_offset=103,
            dlq_partition=2,
            retry_count=1,
        )

        await dlq_tracking_service.record_replay_attempt(record)

        history = await dlq_tracking_service.get_replay_history(
            record.original_message_id
        )
        assert len(history) == 1
        assert history[0].replay_status == EnumReplayStatus.PENDING

    @pytest.mark.asyncio
    async def test_record_replay_different_topics(
        self,
        dlq_tracking_service: DLQTrackingService,
    ) -> None:
        """Test recording a replay with different original and target topics.

        This validates the rerouting use case where a message is replayed
        to a different topic than where it originally failed.
        """
        record = ModelDlqReplayRecord(
            id=uuid4(),
            original_message_id=uuid4(),
            replay_correlation_id=uuid4(),
            original_topic="dev.orders.failed.v1",
            target_topic="dev.orders.retry.v1",  # Different target
            replay_status=EnumReplayStatus.COMPLETED,
            replay_timestamp=datetime.now(UTC),
            success=True,
            error_message=None,
            dlq_offset=104,
            dlq_partition=0,
            retry_count=1,
        )

        await dlq_tracking_service.record_replay_attempt(record)

        history = await dlq_tracking_service.get_replay_history(
            record.original_message_id
        )
        assert len(history) == 1
        assert history[0].original_topic == "dev.orders.failed.v1"
        assert history[0].target_topic == "dev.orders.retry.v1"


# =============================================================================
# Query Tests - Validate history retrieval and ordering
# =============================================================================


class TestDLQTrackingServiceQuery:
    """Tests for querying replay history."""

    @pytest.mark.asyncio
    async def test_get_replay_history_multiple_attempts(
        self,
        dlq_tracking_service: DLQTrackingService,
    ) -> None:
        """Test getting history with multiple replay attempts.

        Verifies that:
        1. Multiple attempts for same message are stored
        2. Results are ordered by timestamp (most recent first)
        3. All attempts are returned correctly
        """
        message_id = uuid4()

        # Record multiple attempts with explicit timestamp offsets for deterministic ordering
        base_time = datetime.now(UTC)
        for i in range(3):
            record = ModelDlqReplayRecord(
                id=uuid4(),
                original_message_id=message_id,
                replay_correlation_id=uuid4(),
                original_topic="test.events.v1",
                target_topic="test.events.v1",
                replay_status=(
                    EnumReplayStatus.FAILED if i < 2 else EnumReplayStatus.COMPLETED
                ),
                replay_timestamp=base_time + timedelta(seconds=i),
                success=i == 2,
                error_message="Retry needed" if i < 2 else None,
                dlq_offset=100 + i,
                dlq_partition=0,
                retry_count=i + 1,
            )
            await dlq_tracking_service.record_replay_attempt(record)

        history = await dlq_tracking_service.get_replay_history(message_id)
        assert len(history) == 3

        # Most recent should be first (ordered by replay_timestamp DESC)
        assert history[0].success is True
        assert history[0].replay_status == EnumReplayStatus.COMPLETED
        assert history[0].retry_count == 3

        # Earlier attempts should follow
        assert history[1].success is False
        assert history[1].retry_count == 2
        assert history[2].retry_count == 1

    @pytest.mark.asyncio
    async def test_get_replay_history_empty(
        self,
        dlq_tracking_service: DLQTrackingService,
    ) -> None:
        """Test getting history for non-existent message returns empty list.

        This validates the expected behavior when querying for a message
        that has never been replayed.
        """
        history = await dlq_tracking_service.get_replay_history(uuid4())
        assert history == []
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_get_replay_history_isolation(
        self,
        dlq_tracking_service: DLQTrackingService,
    ) -> None:
        """Test that history queries are properly isolated by message ID.

        Verifies that querying for one message_id does not return
        records from other messages.
        """
        message_id_1 = uuid4()
        message_id_2 = uuid4()

        # Create records for two different messages
        for message_id, topic_suffix in [
            (message_id_1, "orders"),
            (message_id_2, "payments"),
        ]:
            record = ModelDlqReplayRecord(
                id=uuid4(),
                original_message_id=message_id,
                replay_correlation_id=uuid4(),
                original_topic=f"test.{topic_suffix}.v1",
                target_topic=f"test.{topic_suffix}.v1",
                replay_status=EnumReplayStatus.COMPLETED,
                replay_timestamp=datetime.now(UTC),
                success=True,
                error_message=None,
                dlq_offset=100,
                dlq_partition=0,
                retry_count=1,
            )
            await dlq_tracking_service.record_replay_attempt(record)

        # Query for message_id_1 should only return its records
        history_1 = await dlq_tracking_service.get_replay_history(message_id_1)
        assert len(history_1) == 1
        assert history_1[0].original_topic == "test.orders.v1"

        # Query for message_id_2 should only return its records
        history_2 = await dlq_tracking_service.get_replay_history(message_id_2)
        assert len(history_2) == 1
        assert history_2[0].original_topic == "test.payments.v1"

    @pytest.mark.asyncio
    async def test_get_replay_history_preserves_uuids(
        self,
        dlq_tracking_service: DLQTrackingService,
    ) -> None:
        """Test that UUID fields are correctly preserved through storage.

        Verifies that all UUID fields (id, original_message_id,
        replay_correlation_id) maintain their values through the
        insert/query cycle.
        """
        record_id = uuid4()
        message_id = uuid4()
        correlation_id = uuid4()

        record = ModelDlqReplayRecord(
            id=record_id,
            original_message_id=message_id,
            replay_correlation_id=correlation_id,
            original_topic="test.events.v1",
            target_topic="test.events.v1",
            replay_status=EnumReplayStatus.COMPLETED,
            replay_timestamp=datetime.now(UTC),
            success=True,
            error_message=None,
            dlq_offset=100,
            dlq_partition=0,
            retry_count=1,
        )

        await dlq_tracking_service.record_replay_attempt(record)

        history = await dlq_tracking_service.get_replay_history(message_id)
        assert len(history) == 1

        # Verify all UUIDs are preserved
        assert history[0].id == record_id
        assert isinstance(history[0].id, UUID)
        assert history[0].original_message_id == message_id
        assert isinstance(history[0].original_message_id, UUID)
        assert history[0].replay_correlation_id == correlation_id
        assert isinstance(history[0].replay_correlation_id, UUID)


# =============================================================================
# Health Check Tests - Verify service health monitoring
# =============================================================================


class TestDLQTrackingServiceHealth:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(
        self,
        dlq_tracking_service: DLQTrackingService,
    ) -> None:
        """Test health check returns True when service is healthy.

        Verifies that the health check:
        1. Returns True for initialized service
        2. Can query the database successfully
        3. Verifies table exists
        """
        result = await dlq_tracking_service.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(
        self,
        dlq_tracking_config: ModelDlqTrackingConfig,
    ) -> None:
        """Test health check returns False when service not initialized.

        A service that has not been initialized should report unhealthy
        since it cannot perform any operations.
        """
        service = DLQTrackingService(dlq_tracking_config)
        # Not initialized

        result = await service.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_shutdown(
        self,
        dlq_tracking_config: ModelDlqTrackingConfig,
    ) -> None:
        """Test health check returns False after shutdown.

        After shutdown, the service should report unhealthy since
        the connection pool is closed.
        """
        service = DLQTrackingService(dlq_tracking_config)
        await service.initialize()

        # Verify healthy before shutdown
        assert await service.health_check() is True

        # Shutdown
        await service.shutdown()

        # Should be unhealthy after shutdown
        assert await service.health_check() is False
        assert service.is_initialized is False


# =============================================================================
# Lifecycle Tests - Verify service lifecycle management
# =============================================================================


class TestDLQTrackingServiceLifecycle:
    """Tests for service lifecycle management."""

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(
        self,
        dlq_tracking_config: ModelDlqTrackingConfig,
    ) -> None:
        """Test that calling shutdown multiple times is safe.

        The service should handle repeated shutdown calls gracefully
        without raising errors.
        """
        service = DLQTrackingService(dlq_tracking_config)
        await service.initialize()

        # First shutdown
        await service.shutdown()
        assert service.is_initialized is False

        # Second shutdown should be a no-op
        await service.shutdown()
        assert service.is_initialized is False

        # Third shutdown also safe
        await service.shutdown()
        assert service.is_initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_closes_pool(
        self,
        dlq_tracking_config: ModelDlqTrackingConfig,
    ) -> None:
        """Test that shutdown properly closes the connection pool."""
        service = DLQTrackingService(dlq_tracking_config)
        await service.initialize()

        # Verify pool exists
        assert service._pool is not None

        # Shutdown
        await service.shutdown()

        # Pool should be None after shutdown
        assert service._pool is None
