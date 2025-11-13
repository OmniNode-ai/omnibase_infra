#!/usr/bin/env python3
"""
Unit tests for ProjectionMaterializerService.

Tests event processing pipeline, watermark atomicity, idempotence,
and lag calculation accuracy.

Pure Reducer Refactor - Wave 2, Workstream 2C
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omninode_bridge.infrastructure.entities import ModelStateCommittedEvent
from omninode_bridge.services.projection_materializer import (
    ProjectionMaterializerMetrics,
    ProjectionMaterializerService,
)


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables for all tests."""
    monkeypatch.setenv("POSTGRES_DATABASE", "test_db")
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_password")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer wrapper."""
    consumer = MagicMock()
    consumer.subscribe_to_topics = AsyncMock()
    consumer.consume_messages_stream = AsyncMock()
    consumer.commit_offsets = AsyncMock()
    consumer.close_consumer = AsyncMock()
    consumer.subscribed_topics = ["dev.omninode_bridge.onex.evt.state-committed.v1"]
    consumer.is_subscribed = True
    return consumer


@pytest.fixture
def mock_db_connection():
    """Mock PostgreSQL connection manager."""
    db = MagicMock()
    db.initialize = AsyncMock()
    db.close = AsyncMock()

    # Mock transaction context manager
    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=None)

    @asynccontextmanager
    async def mock_transaction():
        yield mock_conn

    db.transaction = mock_transaction

    return db, mock_conn


@pytest.fixture
def sample_state_committed_event():
    """Sample StateCommitted event for testing."""
    return {
        "event_id": str(uuid4()),
        "workflow_key": "workflow-test-123",
        "version": 2,
        "state": {"items": ["a", "b"], "count": 2},
        "tag": "PROCESSING",
        "last_action": "AddItem",
        "namespace": "test-namespace",
        "provenance": {
            "effect_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "action_id": str(uuid4()),
        },
        "committed_at": datetime.now(UTC).isoformat(),
        "partition_id": "kafka-partition-0",
        "offset": 12345,
        "indices": {"priority": "high"},
        "extras": {},
    }


@pytest.fixture
def sample_kafka_message(sample_state_committed_event):
    """Sample Kafka message with StateCommitted event."""
    return {
        "key": "workflow-test-123",
        "value": sample_state_committed_event,
        "topic": "dev.omninode_bridge.onex.evt.state-committed.v1",
        "partition": 0,
        "offset": 12345,
        "timestamp": int(datetime.now(UTC).timestamp() * 1000),
        "headers": {},
    }


class TestProjectionMaterializerService:
    """Test suite for ProjectionMaterializerService."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test service initialization."""
        service = ProjectionMaterializerService(
            bootstrap_servers="localhost:29092",
            consumer_group="test-group",
            enable_idempotence=True,
        )

        assert service._bootstrap_servers == "localhost:29092"
        assert service._consumer_group == "test-group"
        assert service._enable_idempotence is True
        assert service._is_running is False
        assert service._consumer is not None
        assert service._db is not None

    @pytest.mark.asyncio
    async def test_start_service(self, mock_kafka_consumer, mock_db_connection):
        """Test service startup."""
        db, mock_conn = mock_db_connection

        with (
            patch(
                "omninode_bridge.services.projection_materializer.KafkaConsumerWrapper",
                return_value=mock_kafka_consumer,
            ),
            patch(
                "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
                return_value=db,
            ),
        ):
            service = ProjectionMaterializerService()

            await service.start()

            # Verify database initialized
            db.initialize.assert_awaited_once()

            # Verify Kafka subscription
            mock_kafka_consumer.subscribe_to_topics.assert_awaited_once_with(
                topics=["state-committed"],
                group_id="projection-materializer",
                topic_class="evt",
            )

            # Verify service is running
            assert service.is_running is True
            assert service._consumer_task is not None

            # Cleanup
            await service.stop()

    @pytest.mark.asyncio
    async def test_stop_service(self, mock_kafka_consumer, mock_db_connection):
        """Test service shutdown."""
        db, mock_conn = mock_db_connection

        with (
            patch(
                "omninode_bridge.services.projection_materializer.KafkaConsumerWrapper",
                return_value=mock_kafka_consumer,
            ),
            patch(
                "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
                return_value=db,
            ),
        ):
            service = ProjectionMaterializerService()

            await service.start()
            await service.stop()

            # Verify Kafka consumer closed
            mock_kafka_consumer.close_consumer.assert_awaited_once()

            # Verify database closed
            db.close.assert_awaited_once()

            # Verify service is stopped
            assert service.is_running is False

    @pytest.mark.asyncio
    async def test_process_state_committed_event(
        self, sample_kafka_message, mock_db_connection
    ):
        """Test processing of StateCommitted event."""
        db, mock_conn = mock_db_connection

        with patch(
            "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
            return_value=db,
        ):
            service = ProjectionMaterializerService()

            # Process event
            await service._process_state_committed_event(sample_kafka_message)

            # Verify projection upsert called
            calls = mock_conn.execute.await_args_list
            assert len(calls) >= 1

            # Verify metrics updated
            assert service._metrics.projections_materialized_total == 1
            assert service._metrics.watermark_updates_total == 1

    @pytest.mark.asyncio
    async def test_upsert_projection(
        self, sample_state_committed_event, mock_db_connection
    ):
        """Test projection upsert logic."""
        db, mock_conn = mock_db_connection
        event = ModelStateCommittedEvent(**sample_state_committed_event)

        with patch(
            "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
            return_value=db,
        ):
            service = ProjectionMaterializerService()

            await service._upsert_projection(mock_conn, event)

            # Verify INSERT ... ON CONFLICT query executed
            mock_conn.execute.assert_awaited_once()
            call_args = mock_conn.execute.await_args

            # Verify query contains required fields
            query = call_args[0][0]
            assert "INSERT INTO workflow_projection" in query
            assert "ON CONFLICT (workflow_key) DO UPDATE" in query
            assert "WHERE workflow_projection.version < EXCLUDED.version" in query

            # Verify parameters
            params = call_args[0][1:]
            assert params[0] == event.workflow_key
            assert params[1] == event.version
            assert params[2] == event.tag

    @pytest.mark.asyncio
    async def test_advance_watermark(self, mock_db_connection):
        """Test watermark advancement logic."""
        db, mock_conn = mock_db_connection
        partition_id = "kafka-partition-0"
        offset = 12345

        # Mock current watermark
        mock_conn.fetchval = AsyncMock(return_value=12000)

        with patch(
            "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
            return_value=db,
        ):
            service = ProjectionMaterializerService()

            await service._advance_watermark(mock_conn, partition_id, offset)

            # Verify watermark update query executed
            assert mock_conn.execute.await_count >= 1

            # Verify GREATEST used to prevent regressions
            call_args = mock_conn.execute.await_args
            query = call_args[0][0]
            assert "GREATEST" in query
            assert "projection_watermarks" in query

    @pytest.mark.asyncio
    async def test_watermark_regression_detection(self, mock_db_connection):
        """Test watermark regression detection."""
        db, mock_conn = mock_db_connection
        partition_id = "kafka-partition-0"

        # Current watermark is higher than new offset (regression)
        current_offset = 12500
        new_offset = 12345

        mock_conn.fetchval = AsyncMock(return_value=current_offset)

        with patch(
            "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
            return_value=db,
        ):
            service = ProjectionMaterializerService()

            await service._advance_watermark(mock_conn, partition_id, new_offset)

            # Verify regression detected
            assert service._metrics.wm_regressions_total == 1

            # Verify no watermark update (early return)
            assert mock_conn.execute.await_count == 0

    @pytest.mark.asyncio
    async def test_idempotence_duplicate_detection(self, mock_db_connection):
        """Test duplicate event detection (idempotence)."""
        db, mock_conn = mock_db_connection
        partition_id = "kafka-partition-0"
        offset = 12345

        # Mock watermark shows event already processed
        mock_conn.fetchval = AsyncMock(return_value=12500)

        with patch(
            "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
            return_value=db,
        ):
            service = ProjectionMaterializerService(enable_idempotence=True)

            is_duplicate = await service._check_duplicate(
                mock_conn, partition_id, offset
            )

            assert is_duplicate is True

    @pytest.mark.asyncio
    async def test_idempotence_new_event(self, mock_db_connection):
        """Test new event detection (not duplicate)."""
        db, mock_conn = mock_db_connection
        partition_id = "kafka-partition-0"
        offset = 12500

        # Mock watermark shows event not yet processed
        mock_conn.fetchval = AsyncMock(return_value=12345)

        with patch(
            "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
            return_value=db,
        ):
            service = ProjectionMaterializerService(enable_idempotence=True)

            is_duplicate = await service._check_duplicate(
                mock_conn, partition_id, offset
            )

            assert is_duplicate is False

    @pytest.mark.asyncio
    async def test_lag_calculation(self, sample_kafka_message, mock_db_connection):
        """Test watermark lag calculation."""
        db, mock_conn = mock_db_connection

        # Set timestamp to 5 seconds ago
        past_timestamp = int((datetime.now(UTC).timestamp() - 5) * 1000)
        sample_kafka_message["timestamp"] = past_timestamp

        with patch(
            "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
            return_value=db,
        ):
            service = ProjectionMaterializerService()

            await service._process_state_committed_event(sample_kafka_message)

            # Verify lag calculated (should be around 5000ms)
            assert service._metrics.projection_wm_lag_ms >= 4500
            assert service._metrics.projection_wm_lag_ms <= 5500
            assert service._metrics.max_lag_ms >= service._metrics.projection_wm_lag_ms

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, sample_kafka_message, mock_db_connection):
        """Test metrics collection."""
        db, mock_conn = mock_db_connection

        with patch(
            "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
            return_value=db,
        ):
            service = ProjectionMaterializerService()

            # Process multiple events
            for i in range(5):
                msg = sample_kafka_message.copy()
                msg["offset"] = 12345 + i
                await service._process_state_committed_event(msg)

            # Verify metrics
            assert service._metrics.projections_materialized_total == 5
            assert service._metrics.watermark_updates_total == 5
            assert service._metrics.projections_failed_total == 0

            # Get metrics snapshot
            metrics = service.metrics
            assert isinstance(metrics, ProjectionMaterializerMetrics)
            assert metrics.projections_materialized_total == 5

    @pytest.mark.asyncio
    async def test_error_handling(self, sample_kafka_message):
        """Test error handling during event processing."""
        # Create mock with side effect
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("Database error"))
        mock_conn.fetchval = AsyncMock(return_value=None)

        @asynccontextmanager
        async def mock_transaction():
            yield mock_conn

        db = MagicMock()
        db.initialize = AsyncMock()
        db.close = AsyncMock()
        db.transaction = mock_transaction

        with patch(
            "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
            return_value=db,
        ):
            service = ProjectionMaterializerService()

            # Process event should not raise (error handled internally)
            # Note: The error occurs during consumption, not during _process_state_committed_event
            # The service catches errors in _consume_events
            # For unit test, we expect the exception to be raised since we're calling _process directly
            try:
                await service._process_state_committed_event(sample_kafka_message)
                raise AssertionError("Expected exception to be raised")
            except Exception as e:
                assert str(e) == "Database error"

    @pytest.mark.asyncio
    async def test_atomic_transaction(self, sample_kafka_message):
        """Test atomic projection + watermark update."""
        # Track transaction usage
        transaction_entered = False
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)

        @asynccontextmanager
        async def mock_transaction_context():
            nonlocal transaction_entered
            transaction_entered = True
            yield mock_conn

        db = MagicMock()
        db.initialize = AsyncMock()
        db.close = AsyncMock()
        db.transaction = mock_transaction_context

        with patch(
            "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
            return_value=db,
        ):
            service = ProjectionMaterializerService()

            await service._process_state_committed_event(sample_kafka_message)

            # Verify transaction was used
            assert transaction_entered is True

            # Verify both projection and watermark updated
            assert mock_conn.execute.await_count >= 2

    @pytest.mark.asyncio
    async def test_disabled_idempotence(self, sample_kafka_message, mock_db_connection):
        """Test service with idempotence disabled."""
        db, mock_conn = mock_db_connection

        with patch(
            "omninode_bridge.services.projection_materializer.PostgresConnectionManager",
            return_value=db,
        ):
            service = ProjectionMaterializerService(enable_idempotence=False)

            # Process event twice (should process both without skipping)
            await service._process_state_committed_event(sample_kafka_message)
            await service._process_state_committed_event(sample_kafka_message)

            # Verify both processed
            assert service._metrics.projections_materialized_total == 2
            assert service._metrics.duplicate_events_skipped == 0


class TestProjectionMaterializerMetrics:
    """Test suite for ProjectionMaterializerMetrics model."""

    def test_metrics_initialization(self):
        """Test metrics model initialization."""
        metrics = ProjectionMaterializerMetrics()

        assert metrics.projections_materialized_total == 0
        assert metrics.projections_failed_total == 0
        assert metrics.watermark_updates_total == 0
        assert metrics.wm_regressions_total == 0
        assert metrics.projection_wm_lag_ms == 0.0
        assert metrics.max_lag_ms == 0.0
        assert metrics.events_processed_per_second == 0.0
        assert metrics.duplicate_events_skipped == 0

    def test_metrics_update(self):
        """Test metrics model updates."""
        metrics = ProjectionMaterializerMetrics()

        metrics.projections_materialized_total = 100
        metrics.projection_wm_lag_ms = 50.5
        metrics.events_processed_per_second = 1500.75

        assert metrics.projections_materialized_total == 100
        assert metrics.projection_wm_lag_ms == 50.5
        assert metrics.events_processed_per_second == 1500.75

    def test_metrics_copy(self):
        """Test metrics snapshot (copy)."""
        metrics = ProjectionMaterializerMetrics()
        metrics.projections_materialized_total = 50

        # Get snapshot
        snapshot = metrics.model_copy()

        # Modify original
        metrics.projections_materialized_total = 100

        # Verify snapshot unchanged
        assert snapshot.projections_materialized_total == 50
        assert metrics.projections_materialized_total == 100
