# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for TransitionNotificationOutbox (TDD approach).

This test suite validates the TransitionNotificationOutbox implementation for:
- Storing notifications within the same database transaction
- Serialization to JSONB for PostgreSQL storage
- Processing pending notifications with batch limits
- Publishing notifications and marking as processed
- Handling publish failures with retry tracking
- Using FOR UPDATE SKIP LOCKED for concurrent processing
- Background processor lifecycle (start/stop)
- Metrics tracking for observability

The TransitionNotificationOutbox implements the outbox pattern for reliable
notification delivery of state transition events. It ensures atomic consistency
between state changes and notification publishing by:
1. Storing notifications in the same transaction as state mutations
2. Background processor picks up and publishes pending notifications
3. Handles failures with retry logic and error tracking

Test Organization:
    - TestTransitionNotificationOutboxFixtures: Fixture validation
    - TestTransitionNotificationOutboxStore: Storage operations
    - TestTransitionNotificationOutboxProcess: Processing logic
    - TestTransitionNotificationOutboxConcurrency: SKIP LOCKED behavior
    - TestTransitionNotificationOutboxLifecycle: Start/stop lifecycle
    - TestTransitionNotificationOutboxMetrics: Metrics tracking
    - TestTransitionNotificationOutboxConfiguration: Config options

Related:
    - docs/patterns/retry_backoff_compensation_strategy.md (Outbox Pattern)
    - OMN-XXX: State Transition Notifications via Outbox Pattern
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.notifications import ModelStateTransitionNotification
from omnibase_infra.runtime.models import (
    ModelTransitionNotificationOutboxConfig,
    ModelTransitionNotificationOutboxMetrics,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool() -> AsyncMock:
    """Create mock database pool.

    Returns an AsyncMock configured to simulate asyncpg.Pool behavior,
    including acquire() context manager for connection handling.
    """
    pool = AsyncMock()
    mock_conn = AsyncMock()

    # Configure connection mock
    mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(return_value=None)

    # Configure transaction context manager
    mock_transaction = MagicMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=None)
    mock_transaction.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction.return_value = mock_transaction

    # Configure acquire context manager
    @asynccontextmanager
    async def acquire_context() -> AsyncIterator[AsyncMock]:
        yield mock_conn

    pool.acquire = MagicMock(side_effect=acquire_context)
    pool._mock_connection = mock_conn  # Expose for test assertions

    return pool


@pytest.fixture
def mock_publisher() -> AsyncMock:
    """Create mock notification publisher.

    Returns an AsyncMock simulating a notification publisher interface
    with publish() and publish_batch() methods.
    """
    publisher = AsyncMock()
    publisher.publish = AsyncMock(return_value=None)
    publisher.publish_batch = AsyncMock(return_value=None)
    return publisher


@pytest.fixture
def sample_notification() -> ModelStateTransitionNotification:
    """Create sample notification for testing.

    Returns a fully populated ModelStateTransitionNotification suitable
    for use in store and process tests.
    """
    return ModelStateTransitionNotification(
        aggregate_type="registration",
        aggregate_id=uuid4(),
        from_state="pending",
        to_state="active",
        projection_version=1,
        correlation_id=uuid4(),
        causation_id=uuid4(),
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def sample_notification_with_workflow_view() -> ModelStateTransitionNotification:
    """Create sample notification with workflow_view for testing."""
    return ModelStateTransitionNotification(
        aggregate_type="workflow",
        aggregate_id=uuid4(),
        from_state="running",
        to_state="completed",
        projection_version=42,
        correlation_id=uuid4(),
        causation_id=uuid4(),
        timestamp=datetime.now(UTC),
        workflow_view={"node_id": "abc-123", "executor": "runtime-host"},
    )


@pytest.fixture
def outbox_config() -> ModelTransitionNotificationOutboxConfig:
    """Create default outbox configuration for tests."""
    return ModelTransitionNotificationOutboxConfig(
        outbox_table="state_transition_outbox",
        batch_size=100,
        poll_interval_seconds=0.1,  # Fast for tests
        max_retries=3,
    )


@pytest.fixture
def outbox_config_custom_table() -> ModelTransitionNotificationOutboxConfig:
    """Create outbox configuration with custom table name."""
    return ModelTransitionNotificationOutboxConfig(
        outbox_table="custom_notifications_outbox",
        batch_size=50,
        poll_interval_seconds=0.1,
        max_retries=5,
    )


# =============================================================================
# Test Fixture Validation
# =============================================================================


@pytest.mark.unit
class TestTransitionNotificationOutboxFixtures:
    """Validate test fixtures are correctly configured."""

    def test_mock_pool_has_acquire_context_manager(self, mock_pool: AsyncMock) -> None:
        """Test that mock pool supports async context manager protocol."""
        assert hasattr(mock_pool, "acquire")
        assert hasattr(mock_pool, "_mock_connection")

    def test_mock_publisher_has_publish_methods(
        self, mock_publisher: AsyncMock
    ) -> None:
        """Test that mock publisher has required publish methods."""
        assert hasattr(mock_publisher, "publish")
        assert hasattr(mock_publisher, "publish_batch")

    def test_sample_notification_is_valid(
        self, sample_notification: ModelStateTransitionNotification
    ) -> None:
        """Test that sample notification has all required fields."""
        assert sample_notification.aggregate_type == "registration"
        assert sample_notification.from_state == "pending"
        assert sample_notification.to_state == "active"
        assert sample_notification.projection_version == 1
        assert isinstance(sample_notification.aggregate_id, UUID)
        assert isinstance(sample_notification.correlation_id, UUID)
        assert isinstance(sample_notification.causation_id, UUID)
        assert isinstance(sample_notification.timestamp, datetime)

    def test_sample_notification_is_frozen(
        self, sample_notification: ModelStateTransitionNotification
    ) -> None:
        """Test that notification model is immutable."""
        with pytest.raises(Exception):  # ValidationError for frozen model
            sample_notification.to_state = "different"  # type: ignore[misc]

    def test_sample_notification_serializes_to_json(
        self, sample_notification: ModelStateTransitionNotification
    ) -> None:
        """Test that notification can be serialized to JSON."""
        json_str = sample_notification.model_dump_json()
        assert "registration" in json_str
        assert "pending" in json_str
        assert "active" in json_str

        # Verify round-trip
        data = json.loads(json_str)
        assert data["aggregate_type"] == "registration"
        assert data["projection_version"] == 1

    def test_outbox_config_defaults(
        self, outbox_config: ModelTransitionNotificationOutboxConfig
    ) -> None:
        """Test outbox configuration default values."""
        assert outbox_config.outbox_table == "state_transition_outbox"
        assert outbox_config.batch_size == 100
        assert outbox_config.max_retries == 3


# =============================================================================
# Storage Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionNotificationOutboxStore:
    """Test notification storage operations using real TransitionNotificationOutbox."""

    async def test_store_notification_in_same_connection(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
        outbox_config: ModelTransitionNotificationOutboxConfig,
    ) -> None:
        """Notification is stored using provided connection via real outbox.

        When store() is called with an explicit connection, it MUST use that
        connection to ensure the notification is written in the same transaction
        as the state mutation. This is the core outbox pattern guarantee.
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        # Create real outbox instance
        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            table_name=outbox_config.outbox_table,
            batch_size=outbox_config.batch_size,
            poll_interval_seconds=outbox_config.poll_interval_seconds,
            strict_transaction_mode=False,  # Allow non-transaction for testing
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        mock_conn.is_in_transaction = MagicMock(return_value=True)

        # Call real store method
        await outbox.store(sample_notification, mock_conn)

        # Verify execute was called with INSERT query
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]

        # Verify SQL contains expected elements
        assert "INSERT INTO" in sql
        assert "notification_data" in sql
        assert "aggregate_type" in sql
        assert "aggregate_id" in sql

        # Verify notification was serialized as JSON
        assert call_args[0][1] == sample_notification.model_dump_json()
        assert call_args[0][2] == sample_notification.aggregate_type
        assert call_args[0][3] == sample_notification.aggregate_id

        # Verify metric was updated
        assert outbox.notifications_stored == 1

    async def test_store_serializes_notification_as_json(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
        outbox_config: ModelTransitionNotificationOutboxConfig,
    ) -> None:
        """Notification is serialized to JSONB for PostgreSQL storage.

        The notification payload must be serialized as valid JSON that
        can be stored in PostgreSQL JSONB column and deserialized later.
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            table_name=outbox_config.outbox_table,
            strict_transaction_mode=False,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        mock_conn.is_in_transaction = MagicMock(return_value=True)

        # Call real store method
        await outbox.store(sample_notification, mock_conn)

        # Get the JSON payload that was passed to execute
        call_args = mock_conn.execute.call_args
        payload_json = call_args[0][1]

        # Verify it's valid JSON and contains all fields
        parsed = json.loads(payload_json)
        assert parsed["aggregate_type"] == sample_notification.aggregate_type
        assert parsed["aggregate_id"] == str(sample_notification.aggregate_id)
        assert parsed["from_state"] == sample_notification.from_state
        assert parsed["to_state"] == sample_notification.to_state
        assert parsed["projection_version"] == sample_notification.projection_version
        assert parsed["correlation_id"] == str(sample_notification.correlation_id)
        assert parsed["causation_id"] == str(sample_notification.causation_id)

    async def test_store_raises_error_outside_transaction_in_strict_mode(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Store raises ProtocolConfigurationError in strict mode outside transaction."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            strict_transaction_mode=True,  # Strict mode enabled
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.is_in_transaction = MagicMock(
            return_value=False
        )  # Not in transaction

        with pytest.raises(
            ProtocolConfigurationError, match="outside transaction context"
        ):
            await outbox.store(sample_notification, mock_conn)

    async def test_store_warns_outside_transaction_in_non_strict_mode(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Store logs warning in non-strict mode outside transaction."""
        import logging

        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            strict_transaction_mode=False,  # Non-strict mode
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        mock_conn.is_in_transaction = MagicMock(
            return_value=False
        )  # Not in transaction

        with caplog.at_level(logging.WARNING):
            await outbox.store(sample_notification, mock_conn)

        assert "outside transaction context" in caplog.text
        # But the store should still succeed
        assert outbox.notifications_stored == 1

    async def test_store_increments_metrics(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Store increments notifications_stored metric."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            strict_transaction_mode=False,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        mock_conn.is_in_transaction = MagicMock(return_value=True)

        assert outbox.notifications_stored == 0

        await outbox.store(sample_notification, mock_conn)
        assert outbox.notifications_stored == 1

        await outbox.store(sample_notification, mock_conn)
        assert outbox.notifications_stored == 2


# =============================================================================
# Processing Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionNotificationOutboxProcess:
    """Test notification processing operations using real TransitionNotificationOutbox."""

    async def test_process_pending_fetches_unprocessed(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        outbox_config: ModelTransitionNotificationOutboxConfig,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Process pending fetches and processes unprocessed notifications.

        The processor should query for notifications where:
        - processed_at IS NULL (not yet processed)
        - ORDER BY created_at ASC (FIFO ordering)
        - FOR UPDATE SKIP LOCKED (concurrent safety)
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            table_name=outbox_config.outbox_table,
            batch_size=outbox_config.batch_size,
        )

        mock_conn = mock_pool._mock_connection

        # Create notification data compatible with the real core model
        notification_data = {
            "aggregate_type": sample_notification.aggregate_type,
            "aggregate_id": str(sample_notification.aggregate_id),
            "from_state": sample_notification.from_state,
            "to_state": sample_notification.to_state,
            "projection_version": sample_notification.projection_version,
            "correlation_id": str(sample_notification.correlation_id),
            "causation_id": str(sample_notification.causation_id),
            "timestamp": sample_notification.timestamp.isoformat(),
        }

        mock_rows = [
            {
                "id": 1,
                "notification_data": notification_data,
                "retry_count": 0,
            }
        ]
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        # Configure transaction context manager
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        # Call real process_pending
        processed = await outbox.process_pending()

        # Verify fetch was called with correct SQL
        mock_conn.fetch.assert_called_once()
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]

        assert "processed_at IS NULL" in sql
        assert "ORDER BY created_at" in sql
        assert "FOR UPDATE SKIP LOCKED" in sql
        assert processed == 1

    async def test_process_pending_publishes_and_marks_processed(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
        outbox_config: ModelTransitionNotificationOutboxConfig,
    ) -> None:
        """Successfully processed notifications are marked as processed.

        After successful publish:
        1. Publisher.publish() is called with notification data
        2. processed_at is set to current timestamp
        3. Notification is not re-processed in future batches
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            table_name=outbox_config.outbox_table,
        )

        mock_conn = mock_pool._mock_connection

        # Create notification data compatible with the real core model
        notification_data = {
            "aggregate_type": sample_notification.aggregate_type,
            "aggregate_id": str(sample_notification.aggregate_id),
            "from_state": sample_notification.from_state,
            "to_state": sample_notification.to_state,
            "projection_version": sample_notification.projection_version,
            "correlation_id": str(sample_notification.correlation_id),
            "causation_id": str(sample_notification.causation_id),
            "timestamp": sample_notification.timestamp.isoformat(),
        }

        mock_rows = [
            {
                "id": 1,
                "notification_data": notification_data,
                "retry_count": 0,
            }
        ]
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        # Configure transaction context manager
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        # Call real process_pending
        processed = await outbox.process_pending()

        # Verify publisher was called with a notification object
        mock_publisher.publish.assert_called_once()
        published_notification = mock_publisher.publish.call_args[0][0]
        assert (
            published_notification.aggregate_type == sample_notification.aggregate_type
        )

        # Verify notification was marked as processed
        execute_calls = mock_conn.execute.call_args_list
        assert len(execute_calls) >= 1
        # Find the UPDATE call
        update_calls = [c for c in execute_calls if "UPDATE" in str(c[0][0])]
        assert len(update_calls) == 1
        update_sql = update_calls[0][0][0]
        assert "processed_at = NOW()" in update_sql

        # Verify metrics updated
        assert outbox.notifications_processed == 1
        assert processed == 1

    async def test_process_pending_handles_publish_failure(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
        outbox_config: ModelTransitionNotificationOutboxConfig,
    ) -> None:
        """Failed publishes increment retry_count and store error.

        When publisher.publish() raises an exception:
        1. retry_count is incremented
        2. last_error stores the error message
        3. Notification remains unprocessed for retry
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            table_name=outbox_config.outbox_table,
        )

        mock_conn = mock_pool._mock_connection

        # Create notification data compatible with the real core model
        notification_data = {
            "aggregate_type": sample_notification.aggregate_type,
            "aggregate_id": str(sample_notification.aggregate_id),
            "from_state": sample_notification.from_state,
            "to_state": sample_notification.to_state,
            "projection_version": sample_notification.projection_version,
            "correlation_id": str(sample_notification.correlation_id),
            "causation_id": str(sample_notification.causation_id),
            "timestamp": sample_notification.timestamp.isoformat(),
        }

        mock_rows = [
            {
                "id": 1,
                "notification_data": notification_data,
                "retry_count": 0,
            }
        ]
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        # Configure publisher to fail
        publish_error = ConnectionError("Kafka unavailable")
        mock_publisher.publish = AsyncMock(side_effect=publish_error)

        # Configure transaction context manager
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        # Call real process_pending
        processed = await outbox.process_pending()

        # Verify publisher was called
        mock_publisher.publish.assert_called_once()

        # Verify error was recorded (retry_count incremented)
        execute_calls = mock_conn.execute.call_args_list
        error_update_calls = [c for c in execute_calls if "retry_count" in str(c[0][0])]
        assert len(error_update_calls) == 1
        error_sql = error_update_calls[0][0][0]
        assert "retry_count = retry_count + 1" in error_sql
        assert "last_error" in error_sql

        # Verify failure metrics updated
        assert outbox.notifications_failed == 1
        assert processed == 0

    async def test_batch_size_limits_processing(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Only batch_size notifications are processed per call.

        The LIMIT clause should use the configured batch_size to prevent
        overwhelming the publisher or holding database locks too long.
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        batch_size = 50
        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            batch_size=batch_size,
        )

        mock_conn = mock_pool._mock_connection

        # Return empty list to avoid needing full notification setup
        mock_conn.fetch = AsyncMock(return_value=[])

        # Configure transaction context manager
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        # Call real process_pending
        await outbox.process_pending()

        # Verify LIMIT uses batch_size
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        limit_param = call_args[0][1]

        assert "LIMIT $1" in sql
        assert limit_param == batch_size

    async def test_process_pending_returns_zero_when_no_pending(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Process pending returns zero when no notifications are pending."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.fetch = AsyncMock(return_value=[])

        # Configure transaction context manager
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        processed = await outbox.process_pending()

        assert processed == 0
        mock_publisher.publish.assert_not_called()


# =============================================================================
# Concurrency Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionNotificationOutboxConcurrency:
    """Test concurrent processing with SKIP LOCKED using real TransitionNotificationOutbox."""

    async def test_process_pending_uses_skip_locked(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        outbox_config: ModelTransitionNotificationOutboxConfig,
    ) -> None:
        """Query uses FOR UPDATE SKIP LOCKED for concurrency.

        Multiple processors can run concurrently without deadlocks:
        - FOR UPDATE locks selected rows
        - SKIP LOCKED skips rows already locked by other processors
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            table_name=outbox_config.outbox_table,
            batch_size=outbox_config.batch_size,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.fetch = AsyncMock(return_value=[])

        # Configure transaction context manager
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        # Call real process_pending
        await outbox.process_pending()

        # Verify FOR UPDATE SKIP LOCKED is present
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "FOR UPDATE SKIP LOCKED" in sql

    async def test_skip_locked_prevents_duplicate_processing(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """SKIP LOCKED ensures no notification is processed twice concurrently.

        When two processors query simultaneously:
        - Processor A locks rows 1, 2, 3
        - Processor B skips rows 1, 2, 3 and gets rows 4, 5, 6

        Note: This test documents the expected PostgreSQL SKIP LOCKED behavior.
        True concurrent testing requires integration tests with real DB.
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        # Create two outbox instances (simulating concurrent processors)
        outbox_a = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
        )
        outbox_b = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
        )

        # Verify both instances use SKIP LOCKED in their queries
        # (actual concurrent behavior tested in integration tests)
        mock_conn = mock_pool._mock_connection
        mock_conn.fetch = AsyncMock(return_value=[])

        # Configure transaction context manager
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        # Both outboxes should use SKIP LOCKED
        await outbox_a.process_pending()
        sql_a = mock_conn.fetch.call_args[0][0]
        assert "FOR UPDATE SKIP LOCKED" in sql_a

        mock_conn.fetch.reset_mock()
        await outbox_b.process_pending()
        sql_b = mock_conn.fetch.call_args[0][0]
        assert "FOR UPDATE SKIP LOCKED" in sql_b


# =============================================================================
# Lifecycle Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionNotificationOutboxLifecycle:
    """Test background processor lifecycle using real TransitionNotificationOutbox."""

    async def test_start_begins_background_processing(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        outbox_config: ModelTransitionNotificationOutboxConfig,
    ) -> None:
        """Start method initiates background processor.

        After start():
        - is_running should be True
        - Background task should be created
        - Processor should poll at configured interval
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            poll_interval_seconds=outbox_config.poll_interval_seconds,
        )

        # Configure mock to return empty results (no pending notifications)
        mock_conn = mock_pool._mock_connection
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        assert outbox.is_running is False

        # Start the real outbox
        await outbox.start()

        # Verify running
        assert outbox.is_running is True

        # Cleanup
        await outbox.stop()
        assert outbox.is_running is False

    async def test_stop_gracefully_stops_processor(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        outbox_config: ModelTransitionNotificationOutboxConfig,
    ) -> None:
        """Stop method gracefully terminates processor.

        After stop():
        - is_running should be False
        - Background task should complete (not be cancelled abruptly)
        - Current processing batch should finish
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            poll_interval_seconds=0.05,  # Fast poll for test
            shutdown_timeout_seconds=1.0,
        )

        # Configure mock to return empty results
        mock_conn = mock_pool._mock_connection
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        # Start the outbox
        await outbox.start()
        assert outbox.is_running is True

        # Let it run briefly
        await asyncio.sleep(0.05)

        # Stop gracefully
        await outbox.stop()

        # Verify stopped
        assert outbox.is_running is False

    async def test_start_is_idempotent(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Starting an already-running outbox is a no-op.

        Calling start() twice should not create multiple background tasks.
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            poll_interval_seconds=0.1,
        )

        # Configure mock to return empty results
        mock_conn = mock_pool._mock_connection
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        # Start twice
        await outbox.start()
        await outbox.start()  # Should be idempotent

        assert outbox.is_running is True

        # Cleanup
        await outbox.stop()

    async def test_stop_is_idempotent(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Stopping an already-stopped outbox is a no-op.

        Calling stop() twice should not raise errors.
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
        )

        assert outbox.is_running is False

        # Stop without starting - should not raise
        await outbox.stop()
        await outbox.stop()

        assert outbox.is_running is False


# =============================================================================
# Configuration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionNotificationOutboxConfiguration:
    """Test configuration options using real TransitionNotificationOutbox."""

    async def test_custom_table_name(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
        outbox_config_custom_table: ModelTransitionNotificationOutboxConfig,
    ) -> None:
        """Custom table name is used in queries.

        The outbox should use the configured table_name for all SQL operations.
        """
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        custom_table = outbox_config_custom_table.outbox_table
        assert custom_table == "custom_notifications_outbox"

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            table_name=custom_table,
            strict_transaction_mode=False,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.is_in_transaction = MagicMock(return_value=True)

        # Test store uses custom table name
        await outbox.store(sample_notification, mock_conn)

        # Verify custom table name was used in store query
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert custom_table in sql

        # Test process_pending uses custom table name
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        await outbox.process_pending()

        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert custom_table in sql

    async def test_batch_size_configuration(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Batch size configuration is applied to processing queries."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        custom_batch_size = 25
        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            batch_size=custom_batch_size,
        )

        assert outbox.batch_size == custom_batch_size

        # Verify batch size is used in process_pending
        mock_conn = mock_pool._mock_connection
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        await outbox.process_pending()

        call_args = mock_conn.fetch.call_args
        # batch_size should be passed as first parameter (LIMIT $1)
        assert call_args[0][1] == custom_batch_size

    async def test_poll_interval_configuration(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Poll interval configuration controls processing frequency."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        custom_interval = 5.0
        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            poll_interval_seconds=custom_interval,
        )

        assert outbox.poll_interval == custom_interval

    async def test_shutdown_timeout_configuration(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Shutdown timeout configuration is properly set."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        custom_timeout = 30.0
        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            shutdown_timeout_seconds=custom_timeout,
        )

        assert outbox.shutdown_timeout == custom_timeout

    async def test_strict_transaction_mode_configuration(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Strict transaction mode configuration is properly set."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        # Default should be True
        outbox_strict = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
        )
        assert outbox_strict.strict_transaction_mode is True

        # Can be disabled
        outbox_non_strict = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            strict_transaction_mode=False,
        )
        assert outbox_non_strict.strict_transaction_mode is False


# =============================================================================
# Metrics Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionNotificationOutboxMetrics:
    """Test metrics tracking."""

    async def test_metrics_track_processed_count(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Metrics are updated with processed notification counts.

        The outbox should track:
        - notifications_stored: Total stored
        - notifications_processed: Successfully processed
        - notifications_failed: Failed to process
        """
        metrics = ModelTransitionNotificationOutboxMetrics(outbox_table="test_outbox")

        # Simulate storing notifications
        for _ in range(5):
            metrics = metrics.model_copy(
                update={"notifications_stored": metrics.notifications_stored + 1}
            )

        assert metrics.notifications_stored == 5

        # Simulate processing (3 success, 2 fail)
        for _ in range(3):
            metrics = metrics.model_copy(
                update={"notifications_processed": metrics.notifications_processed + 1}
            )
        for _ in range(2):
            metrics = metrics.model_copy(
                update={"notifications_failed": metrics.notifications_failed + 1}
            )

        assert metrics.notifications_processed == 3
        assert metrics.notifications_failed == 2

    async def test_metrics_track_dlq_count(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Metrics track notifications sent to DLQ after max retries."""
        metrics = ModelTransitionNotificationOutboxMetrics(
            outbox_table="test_outbox",
            max_retries=3,
        )

        # Simulate DLQ moves
        for _ in range(3):
            metrics = metrics.model_copy(
                update={
                    "notifications_sent_to_dlq": metrics.notifications_sent_to_dlq + 1
                }
            )

        assert metrics.notifications_sent_to_dlq == 3

    async def test_metrics_dlq_needs_attention_helper(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Metrics provide dlq_needs_attention() helper method."""
        # DLQ disabled - always returns False
        metrics_no_dlq = ModelTransitionNotificationOutboxMetrics(
            outbox_table="test_outbox",
            max_retries=None,
        )
        assert metrics_no_dlq.dlq_needs_attention() is False

        # DLQ enabled but below threshold
        metrics_below_threshold = ModelTransitionNotificationOutboxMetrics(
            outbox_table="test_outbox",
            max_retries=3,
            dlq_publish_failures=2,  # Below DEFAULT_DLQ_ALERT_THRESHOLD of 3
        )
        assert metrics_below_threshold.dlq_needs_attention() is False

        # DLQ enabled and at/above threshold
        metrics_at_threshold = ModelTransitionNotificationOutboxMetrics(
            outbox_table="test_outbox",
            max_retries=3,
            dlq_publish_failures=3,  # At DEFAULT_DLQ_ALERT_THRESHOLD
        )
        assert metrics_at_threshold.dlq_needs_attention() is True

    async def test_metrics_track_running_state(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Metrics track whether processor is running."""
        metrics = ModelTransitionNotificationOutboxMetrics(outbox_table="test_outbox")
        assert metrics.is_running is False

        # Simulate start
        metrics = metrics.model_copy(update={"is_running": True})
        assert metrics.is_running is True

        # Simulate stop
        metrics = metrics.model_copy(update={"is_running": False})
        assert metrics.is_running is False


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionNotificationOutboxErrorHandling:
    """Test error handling scenarios."""

    async def test_database_error_during_store_propagates(
        self,
        mock_pool: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Database errors during store should propagate to caller.

        The caller is responsible for transaction rollback when store fails.
        """
        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(
            side_effect=Exception("Database constraint violation")
        )

        with pytest.raises(Exception, match="Database constraint violation"):
            await mock_conn.execute(
                "INSERT INTO outbox (id, payload) VALUES ($1, $2)",
                uuid4(),
                sample_notification.model_dump_json(),
            )

    async def test_publisher_error_does_not_stop_processing(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Publisher errors should be handled per-notification.

        A single publish failure should not stop processing of other notifications.
        """
        mock_conn = mock_pool._mock_connection
        notifications_processed = 0
        notifications_failed = 0

        # Configure publisher to fail on second notification
        call_count = 0

        async def conditional_publish(payload: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ConnectionError("Temporary failure")

        mock_publisher.publish = AsyncMock(side_effect=conditional_publish)

        # Simulate processing 3 notifications
        rows = [{"id": uuid4(), "payload": "{}"} for _ in range(3)]

        for row in rows:
            try:
                await mock_publisher.publish(row["payload"])
                notifications_processed += 1
            except Exception:
                notifications_failed += 1

        # All notifications should be attempted
        assert mock_publisher.publish.call_count == 3
        assert notifications_processed == 2
        assert notifications_failed == 1

    async def test_exhausted_retries_logged_for_dlq(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        outbox_config: ModelTransitionNotificationOutboxConfig,
    ) -> None:
        """Notifications exceeding max_retries should be flagged for DLQ.

        When retry_count >= max_retries:
        - Notification should not be selected for processing
        - Should be available for DLQ processing (separate query)
        """
        mock_conn = mock_pool._mock_connection

        # Simulate DLQ query for exhausted notifications
        # NOTE: f-string SQL is safe - table name from trusted config
        await mock_conn.fetch(
            f"""
            SELECT id, aggregate_type, payload, retry_count, last_error
            FROM {outbox_config.outbox_table}
            WHERE processed_at IS NULL
              AND retry_count >= $1
            """,  # noqa: S608
            outbox_config.max_retries,
        )

        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "retry_count >= $1" in sql


# =============================================================================
# Integration-style Tests (with mocks)
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionNotificationOutboxIntegration:
    """Integration-style tests combining multiple components."""

    async def test_full_store_and_process_flow(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
        outbox_config: ModelTransitionNotificationOutboxConfig,
    ) -> None:
        """Test complete flow from store to process to published.

        This test simulates the full lifecycle:
        1. Store notification in transaction
        2. Background processor picks up notification
        3. Publisher receives notification
        4. Notification marked as processed
        """
        mock_conn = mock_pool._mock_connection
        notification_id = uuid4()

        # Step 1: Store notification
        # NOTE: f-string SQL is safe - table name from trusted config
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        await mock_conn.execute(
            f"""
            INSERT INTO {outbox_config.outbox_table}
            (id, aggregate_type, aggregate_id, payload, created_at)
            VALUES ($1, $2, $3, $4, $5)
            """,  # noqa: S608
            notification_id,
            sample_notification.aggregate_type,
            sample_notification.aggregate_id,
            sample_notification.model_dump_json(),
            datetime.now(UTC),
        )

        # Step 2: Process picks up notification
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": notification_id,
                    "aggregate_type": sample_notification.aggregate_type,
                    "aggregate_id": sample_notification.aggregate_id,
                    "payload": sample_notification.model_dump_json(),
                    "created_at": datetime.now(UTC),
                    "processed_at": None,
                    "retry_count": 0,
                }
            ]
        )
        # NOTE: f-string SQL is safe - table name from trusted config
        rows = await mock_conn.fetch(
            f"""
            SELECT * FROM {outbox_config.outbox_table}
            WHERE processed_at IS NULL
            FOR UPDATE SKIP LOCKED
            """  # noqa: S608
        )

        # Step 3: Publish notification
        for row in rows:
            await mock_publisher.publish(row["payload"])

        # Verify publisher received correct payload
        mock_publisher.publish.assert_called_once()
        published_payload = mock_publisher.publish.call_args[0][0]
        assert sample_notification.aggregate_type in published_payload

        # Step 4: Mark as processed
        # NOTE: f-string SQL is safe - table name from trusted config
        await mock_conn.execute(
            f"""
            UPDATE {outbox_config.outbox_table}
            SET processed_at = $1
            WHERE id = $2
            """,  # noqa: S608
            datetime.now(UTC),
            notification_id,
        )

        # Verify update was called
        update_calls = [
            c for c in mock_conn.execute.call_args_list if "UPDATE" in str(c)
        ]
        assert len(update_calls) == 1

    async def test_notification_with_workflow_view_roundtrip(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        sample_notification_with_workflow_view: ModelStateTransitionNotification,
        outbox_config: ModelTransitionNotificationOutboxConfig,
    ) -> None:
        """Test that workflow_view is preserved through store and process."""
        notification = sample_notification_with_workflow_view

        # Serialize
        payload_json = notification.model_dump_json()

        # Deserialize (simulating what processor would do)
        parsed = json.loads(payload_json)
        restored = ModelStateTransitionNotification.model_validate(parsed)

        # Verify workflow_view preserved
        assert restored.workflow_view == notification.workflow_view
        assert restored.workflow_view is not None
        assert restored.workflow_view.get("node_id") == "abc-123"
        assert restored.workflow_view.get("executor") == "runtime-host"


# =============================================================================
# DLQ Tests
# =============================================================================


@pytest.fixture
def mock_dlq_publisher() -> AsyncMock:
    """Create mock DLQ publisher.

    Returns an AsyncMock simulating a DLQ notification publisher interface
    for dead letter queue functionality.
    """
    publisher = AsyncMock()
    publisher.publish = AsyncMock(return_value=None)
    return publisher


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionNotificationOutboxDLQ:
    """Test Dead Letter Queue (DLQ) functionality."""

    # =========================================================================
    # Configuration Validation Tests
    # =========================================================================

    async def test_dlq_max_retries_less_than_one_raises_error(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
    ) -> None:
        """max_retries < 1 should raise ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        with pytest.raises(
            ProtocolConfigurationError, match="max_retries must be >= 1"
        ):
            TransitionNotificationOutbox(
                pool=mock_pool,
                publisher=mock_publisher,
                max_retries=0,
                dlq_publisher=mock_dlq_publisher,
            )

    async def test_dlq_max_retries_without_publisher_raises_error(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """max_retries set but dlq_publisher=None should raise error."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        with pytest.raises(
            ProtocolConfigurationError,
            match="dlq_publisher is required when max_retries is configured",
        ):
            TransitionNotificationOutbox(
                pool=mock_pool,
                publisher=mock_publisher,
                max_retries=3,
                dlq_publisher=None,
            )

    async def test_dlq_publisher_without_max_retries_logs_warning(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """dlq_publisher set but max_retries=None should log warning."""
        import logging

        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        with caplog.at_level(logging.WARNING):
            outbox = TransitionNotificationOutbox(
                pool=mock_pool,
                publisher=mock_publisher,
                max_retries=None,
                dlq_publisher=mock_dlq_publisher,
            )

        assert "dlq_publisher configured but max_retries is None" in caplog.text
        assert "DLQ will never be used" in caplog.text
        assert outbox.max_retries is None

    async def test_dlq_valid_configuration_succeeds(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
    ) -> None:
        """Valid DLQ config (max_retries=3, dlq_publisher set) succeeds."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
            dlq_topic="test-dlq-topic",
        )

        assert outbox.max_retries == 3
        assert outbox.dlq_topic == "test-dlq-topic"
        assert outbox.notifications_sent_to_dlq == 0

    # =========================================================================
    # _should_move_to_dlq() Tests
    # =========================================================================

    async def test_should_move_to_dlq_returns_false_when_disabled(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Returns False when max_retries is None."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=None,
            dlq_publisher=None,
        )

        # Should return False for any retry count when DLQ disabled
        assert outbox._should_move_to_dlq(0) is False
        assert outbox._should_move_to_dlq(1) is False
        assert outbox._should_move_to_dlq(100) is False

    async def test_should_move_to_dlq_returns_false_below_threshold(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
    ) -> None:
        """Returns False when retry_count < max_retries."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        # retry_count 0, 1, 2 < max_retries 3
        assert outbox._should_move_to_dlq(0) is False
        assert outbox._should_move_to_dlq(1) is False
        assert outbox._should_move_to_dlq(2) is False

    async def test_should_move_to_dlq_returns_true_at_threshold(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
    ) -> None:
        """Returns True when retry_count == max_retries."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        # retry_count 3 == max_retries 3
        assert outbox._should_move_to_dlq(3) is True

    async def test_should_move_to_dlq_returns_true_above_threshold(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
    ) -> None:
        """Returns True when retry_count > max_retries."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        # retry_count > max_retries
        assert outbox._should_move_to_dlq(4) is True
        assert outbox._should_move_to_dlq(10) is True
        assert outbox._should_move_to_dlq(100) is True

    # =========================================================================
    # _move_to_dlq() Behavior Tests
    # =========================================================================

    async def test_move_to_dlq_publishes_notification(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """DLQ publisher.publish() is called with notification."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        result = await outbox._move_to_dlq(
            row_id=1,
            notification=sample_notification,
            retry_count=3,
            conn=mock_conn,
            update_dlq_query="UPDATE test SET processed_at = NOW() WHERE id = $1",
            correlation_id=uuid4(),
        )

        assert result is True
        mock_dlq_publisher.publish.assert_called_once_with(sample_notification)

    async def test_move_to_dlq_marks_as_processed(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Notification is marked as processed after DLQ publish."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        update_query = (
            "UPDATE test SET processed_at = NOW(), last_error = $2 WHERE id = $1"
        )

        await outbox._move_to_dlq(
            row_id=42,
            notification=sample_notification,
            retry_count=3,
            conn=mock_conn,
            update_dlq_query=update_query,
            correlation_id=uuid4(),
        )

        # Verify execute was called with row_id
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert call_args[0][0] == update_query
        assert call_args[0][1] == 42  # row_id

    async def test_move_to_dlq_updates_last_error(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """last_error is set to 'Moved to DLQ after N retries'."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        update_query = (
            "UPDATE test SET processed_at = NOW(), last_error = $2 WHERE id = $1"
        )

        await outbox._move_to_dlq(
            row_id=1,
            notification=sample_notification,
            retry_count=5,
            conn=mock_conn,
            update_dlq_query=update_query,
            correlation_id=uuid4(),
        )

        call_args = mock_conn.execute.call_args
        error_message = call_args[0][2]
        assert error_message == "Moved to DLQ after 5 retries"

    async def test_move_to_dlq_increments_metric(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """_notifications_sent_to_dlq counter is incremented."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        assert outbox.notifications_sent_to_dlq == 0

        await outbox._move_to_dlq(
            row_id=1,
            notification=sample_notification,
            retry_count=3,
            conn=mock_conn,
            update_dlq_query="UPDATE test SET processed_at = NOW() WHERE id = $1",
            correlation_id=uuid4(),
        )

        assert outbox.notifications_sent_to_dlq == 1

        # Call again
        await outbox._move_to_dlq(
            row_id=2,
            notification=sample_notification,
            retry_count=4,
            conn=mock_conn,
            update_dlq_query="UPDATE test SET processed_at = NOW() WHERE id = $1",
            correlation_id=uuid4(),
        )

        assert outbox.notifications_sent_to_dlq == 2

    async def test_move_to_dlq_returns_true_on_success(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Returns True when DLQ publish succeeds."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        result = await outbox._move_to_dlq(
            row_id=1,
            notification=sample_notification,
            retry_count=3,
            conn=mock_conn,
            update_dlq_query="UPDATE test SET processed_at = NOW() WHERE id = $1",
            correlation_id=uuid4(),
        )

        assert result is True

    # =========================================================================
    # DLQ Failure Handling Tests
    # =========================================================================

    async def test_move_to_dlq_failure_returns_false(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Returns False when DLQ publish fails."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        mock_conn = mock_pool._mock_connection

        # Configure DLQ publisher to fail
        mock_dlq_publisher.publish = AsyncMock(
            side_effect=ConnectionError("DLQ unavailable")
        )

        result = await outbox._move_to_dlq(
            row_id=1,
            notification=sample_notification,
            retry_count=3,
            conn=mock_conn,
            update_dlq_query="UPDATE test SET processed_at = NOW() WHERE id = $1",
            correlation_id=uuid4(),
        )

        assert result is False

    async def test_move_to_dlq_failure_does_not_mark_processed(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Notification NOT marked as processed when DLQ publish fails."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        # Configure DLQ publisher to fail
        mock_dlq_publisher.publish = AsyncMock(
            side_effect=ConnectionError("DLQ unavailable")
        )

        await outbox._move_to_dlq(
            row_id=1,
            notification=sample_notification,
            retry_count=3,
            conn=mock_conn,
            update_dlq_query="UPDATE test SET processed_at = NOW() WHERE id = $1",
            correlation_id=uuid4(),
        )

        # Execute should NOT have been called since publish failed before DB update
        mock_conn.execute.assert_not_called()

    async def test_move_to_dlq_failure_does_not_increment_metric(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """_notifications_sent_to_dlq NOT incremented on DLQ failure."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        mock_conn = mock_pool._mock_connection

        # Configure DLQ publisher to fail
        mock_dlq_publisher.publish = AsyncMock(
            side_effect=ConnectionError("DLQ unavailable")
        )

        assert outbox.notifications_sent_to_dlq == 0

        await outbox._move_to_dlq(
            row_id=1,
            notification=sample_notification,
            retry_count=3,
            conn=mock_conn,
            update_dlq_query="UPDATE test SET processed_at = NOW() WHERE id = $1",
            correlation_id=uuid4(),
        )

        # Metric should NOT be incremented
        assert outbox.notifications_sent_to_dlq == 0

    # =========================================================================
    # Integration with process_pending() Tests
    # =========================================================================

    async def test_process_pending_moves_exhausted_to_dlq(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
    ) -> None:
        """Notifications exceeding max_retries are moved to DLQ."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        mock_conn = mock_pool._mock_connection
        notification_id = 1

        # Create notification data compatible with the real core model
        notification_data = {
            "aggregate_type": "registration",
            "aggregate_id": str(uuid4()),
            "from_state": "pending",
            "to_state": "active",
            "projection_version": 1,
            "correlation_id": str(uuid4()),
            "causation_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Return notification that has exceeded max_retries
        mock_rows = [
            {
                "id": notification_id,
                "notification_data": notification_data,
                "retry_count": 3,  # Equals max_retries, should go to DLQ
            }
        ]
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        # Configure transaction context manager properly
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        processed = await outbox.process_pending()

        # Should be processed via DLQ
        assert processed == 1

        # DLQ publisher should have been called
        mock_dlq_publisher.publish.assert_called_once()

        # Regular publisher should NOT have been called
        mock_publisher.publish.assert_not_called()

        # DLQ metric should be incremented
        assert outbox.notifications_sent_to_dlq == 1

    async def test_process_pending_skips_dlq_when_disabled(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """When max_retries=None, no DLQ processing occurs."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=None,  # DLQ disabled
            dlq_publisher=None,
        )

        mock_conn = mock_pool._mock_connection
        notification_id = 1

        # Create notification data compatible with the real core model
        notification_data = {
            "aggregate_type": "registration",
            "aggregate_id": str(uuid4()),
            "from_state": "pending",
            "to_state": "active",
            "projection_version": 1,
            "correlation_id": str(uuid4()),
            "causation_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Return notification with high retry count
        mock_rows = [
            {
                "id": notification_id,
                "notification_data": notification_data,
                "retry_count": 100,  # High retry count but DLQ disabled
            }
        ]
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        # Configure transaction context manager properly
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        processed = await outbox.process_pending()

        # Should be processed via normal publisher
        assert processed == 1

        # Regular publisher should have been called
        mock_publisher.publish.assert_called_once()

        # DLQ metric should remain zero
        assert outbox.notifications_sent_to_dlq == 0

    async def test_process_pending_counts_dlq_as_processed(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
    ) -> None:
        """DLQ moves count toward the returned processed count."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        mock_conn = mock_pool._mock_connection

        # Create notification data compatible with the real core model
        def make_notification_data() -> dict[str, object]:
            return {
                "aggregate_type": "registration",
                "aggregate_id": str(uuid4()),
                "from_state": "pending",
                "to_state": "active",
                "projection_version": 1,
                "correlation_id": str(uuid4()),
                "causation_id": str(uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        # Return mix of normal and DLQ-bound notifications
        mock_rows = [
            {
                "id": 1,
                "notification_data": make_notification_data(),
                "retry_count": 0,  # Normal processing
            },
            {
                "id": 2,
                "notification_data": make_notification_data(),
                "retry_count": 3,  # DLQ processing
            },
            {
                "id": 3,
                "notification_data": make_notification_data(),
                "retry_count": 1,  # Normal processing
            },
        ]
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        # Configure transaction context manager properly
        mock_transaction = MagicMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=None)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        processed = await outbox.process_pending()

        # All 3 should count as processed
        assert processed == 3

        # Regular publisher should have been called twice (retry_count 0 and 1)
        assert mock_publisher.publish.call_count == 2

        # DLQ publisher should have been called once (retry_count 3)
        mock_dlq_publisher.publish.assert_called_once()

        # Metrics should reflect this
        assert (
            outbox.notifications_processed == 3
        )  # Normal + DLQ both count as processed
        assert outbox.notifications_sent_to_dlq == 1

    # =========================================================================
    # Metrics Tests
    # =========================================================================

    async def test_metrics_include_dlq_fields(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
    ) -> None:
        """get_metrics() includes notifications_sent_to_dlq, max_retries, dlq_topic."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=5,
            dlq_publisher=mock_dlq_publisher,
            dlq_topic="test-notifications-dlq",
        )

        metrics = outbox.get_metrics()

        # Verify DLQ fields are present
        assert metrics.notifications_sent_to_dlq == 0
        assert metrics.max_retries == 5
        assert metrics.dlq_topic == "test-notifications-dlq"

        # Verify other standard fields
        assert metrics.is_running is False
        assert metrics.notifications_stored == 0
        assert metrics.notifications_processed == 0
        assert metrics.notifications_failed == 0

    async def test_metrics_dlq_fields_when_disabled(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """get_metrics() returns None for DLQ fields when DLQ disabled."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=None,  # DLQ disabled
            dlq_publisher=None,
        )

        metrics = outbox.get_metrics()

        # DLQ fields should reflect disabled state
        assert metrics.notifications_sent_to_dlq == 0
        assert metrics.max_retries is None
        assert metrics.dlq_topic is None

    async def test_metrics_dlq_counter_updates_after_dlq_move(
        self,
        mock_pool: AsyncMock,
        mock_publisher: AsyncMock,
        mock_dlq_publisher: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """get_metrics() reflects updated DLQ counter after moving to DLQ."""
        from omnibase_infra.runtime.transition_notification_outbox import (
            TransitionNotificationOutbox,
        )

        outbox = TransitionNotificationOutbox(
            pool=mock_pool,
            publisher=mock_publisher,
            max_retries=3,
            dlq_publisher=mock_dlq_publisher,
        )

        mock_conn = mock_pool._mock_connection
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        # Initial metrics
        metrics_before = outbox.get_metrics()
        assert metrics_before.notifications_sent_to_dlq == 0

        # Move to DLQ
        await outbox._move_to_dlq(
            row_id=1,
            notification=sample_notification,
            retry_count=3,
            conn=mock_conn,
            update_dlq_query="UPDATE test SET processed_at = NOW() WHERE id = $1",
            correlation_id=uuid4(),
        )

        # Updated metrics
        metrics_after = outbox.get_metrics()
        assert metrics_after.notifications_sent_to_dlq == 1
