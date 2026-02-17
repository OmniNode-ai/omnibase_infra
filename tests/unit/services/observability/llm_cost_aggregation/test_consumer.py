# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ServiceLlmCostAggregator consumer.

Tests:
    - ConsumerMetrics: Counter tracking and snapshot reporting
    - mask_dsn_password: PostgreSQL DSN password masking for safe logging
    - Consumer lifecycle: Start/stop/context-manager behavior
    - Batch processing: Message parsing, skipping, offset tracking

All tests mock Kafka consumer, asyncpg pool, and aiohttp health server.
No real infrastructure connections required.

Related Tickets:
    - OMN-2240: E1-T4 LLM cost aggregation service
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka import TopicPartition

from omnibase_infra.services.observability.llm_cost_aggregation.config import (
    ConfigLlmCostAggregation,
)
from omnibase_infra.services.observability.llm_cost_aggregation.consumer import (
    ConsumerMetrics,
    ServiceLlmCostAggregator,
    mask_dsn_password,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> ConfigLlmCostAggregation:
    """Create a test configuration with minimal required fields."""
    return ConfigLlmCostAggregation(
        kafka_bootstrap_servers="localhost:9092",
        postgres_dsn="postgresql://testuser:testpass@localhost:5432/testdb",
        batch_size=10,
        batch_timeout_ms=500,
        _env_file=None,
    )


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    pool.close = AsyncMock()
    return pool


@pytest.fixture
def mock_writer() -> MagicMock:
    """Create a mock WriterLlmCostAggregationPostgres."""
    writer = MagicMock()
    writer.write_call_metrics = AsyncMock(return_value=3)
    writer.write_cost_aggregates = AsyncMock(return_value=9)
    writer.get_circuit_breaker_state = MagicMock(
        return_value={"state": "closed", "failure_count": 0}
    )
    return writer


@pytest.fixture
def mock_kafka_consumer() -> MagicMock:
    """Create a mock AIOKafkaConsumer."""
    consumer = MagicMock()
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    consumer.getmany = AsyncMock(return_value={})
    consumer.commit = AsyncMock()
    return consumer


def _make_consumer_record(
    topic: str = "onex.evt.omniintelligence.llm-call-completed.v1",
    partition: int = 0,
    offset: int = 0,
    value: bytes | None = None,
) -> SimpleNamespace:
    """Create a fake ConsumerRecord-like object.

    Uses SimpleNamespace to avoid importing aiokafka internals in tests.
    """
    return SimpleNamespace(
        topic=topic,
        partition=partition,
        offset=offset,
        value=value,
    )


# =============================================================================
# Tests: ConsumerMetrics
# =============================================================================


@pytest.mark.unit
class TestConsumerMetrics:
    """Tests for the ConsumerMetrics counter class."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        """All counters start at zero."""
        metrics = ConsumerMetrics()
        assert metrics.messages_received == 0
        assert metrics.messages_processed == 0
        assert metrics.messages_failed == 0
        assert metrics.messages_skipped == 0
        assert metrics.batches_processed == 0
        assert metrics.aggregations_written == 0
        assert metrics.consecutive_commit_failures == 0
        assert metrics.last_poll_at is None
        assert metrics.last_successful_write_at is None
        assert metrics.last_commit_failure_at is None
        assert metrics.started_at is not None

    @pytest.mark.asyncio
    async def test_record_received(self) -> None:
        """record_received increments messages_received and updates last_poll_at."""
        metrics = ConsumerMetrics()
        await metrics.record_received(5)
        assert metrics.messages_received == 5
        assert metrics.last_poll_at is not None

    @pytest.mark.asyncio
    async def test_record_processed(self) -> None:
        """record_processed increments messages_processed and updates last_successful_write_at."""
        metrics = ConsumerMetrics()
        await metrics.record_processed(3)
        assert metrics.messages_processed == 3
        assert metrics.last_successful_write_at is not None

    @pytest.mark.asyncio
    async def test_record_failed(self) -> None:
        """record_failed increments messages_failed."""
        metrics = ConsumerMetrics()
        await metrics.record_failed(2)
        assert metrics.messages_failed == 2

    @pytest.mark.asyncio
    async def test_record_skipped(self) -> None:
        """record_skipped increments messages_skipped."""
        metrics = ConsumerMetrics()
        await metrics.record_skipped(4)
        assert metrics.messages_skipped == 4

    @pytest.mark.asyncio
    async def test_snapshot(self) -> None:
        """snapshot returns a dict with all counter values."""
        metrics = ConsumerMetrics()
        await metrics.record_received(10)
        await metrics.record_processed(8)
        await metrics.record_failed(1)
        await metrics.record_skipped(1)

        snap = await metrics.snapshot()
        assert snap["messages_received"] == 10
        assert snap["messages_processed"] == 8
        assert snap["messages_failed"] == 1
        assert snap["messages_skipped"] == 1
        assert snap["batches_processed"] == 0
        assert snap["aggregations_written"] == 0
        assert snap["consecutive_commit_failures"] == 0
        assert snap["started_at"] is not None
        assert snap["last_poll_at"] is not None
        assert snap["last_successful_write_at"] is not None
        assert snap["last_commit_failure_at"] is None

    @pytest.mark.asyncio
    async def test_multiple_operations(self) -> None:
        """Multiple operations accumulate correctly in sequence."""
        metrics = ConsumerMetrics()
        await metrics.record_received(10)
        await metrics.record_processed(7)
        await metrics.record_failed(2)
        await metrics.record_skipped(1)
        await metrics.record_batch_processed()

        assert metrics.messages_received == 10
        assert metrics.messages_processed == 7
        assert metrics.messages_failed == 2
        assert metrics.messages_skipped == 1
        assert metrics.batches_processed == 1

    @pytest.mark.asyncio
    async def test_record_commit_failure_and_reset(self) -> None:
        """record_commit_failure increments and reset clears failures."""
        metrics = ConsumerMetrics()
        await metrics.record_commit_failure()
        await metrics.record_commit_failure()
        assert metrics.consecutive_commit_failures == 2
        assert metrics.last_commit_failure_at is not None

        await metrics.reset_consecutive_commit_failures()
        assert metrics.consecutive_commit_failures == 0

    @pytest.mark.asyncio
    async def test_record_aggregations(self) -> None:
        """record_aggregations increments aggregations_written."""
        metrics = ConsumerMetrics()
        await metrics.record_aggregations(12)
        assert metrics.aggregations_written == 12


# =============================================================================
# Tests: mask_dsn_password
# =============================================================================


@pytest.mark.unit
class TestMaskDsnPassword:
    """Tests for the mask_dsn_password utility function."""

    def test_mask_password(self) -> None:
        """Password in DSN is replaced with '***'."""
        dsn = "postgresql://user:secret@localhost:5432/db"
        result = mask_dsn_password(dsn)
        assert "secret" not in result
        assert "***" in result
        assert "user" in result
        assert "localhost" in result
        assert "5432" in result
        assert "/db" in result

    def test_no_password(self) -> None:
        """DSN without password is returned unchanged."""
        dsn = "postgresql://user@localhost:5432/db"
        result = mask_dsn_password(dsn)
        assert result == dsn

    def test_malformed_dsn(self) -> None:
        """Non-URL string is returned unchanged."""
        dsn = "not-a-url"
        result = mask_dsn_password(dsn)
        assert result == dsn

    def test_empty_password(self) -> None:
        """DSN with empty password (user:@host) is returned unchanged.

        urlparse treats an empty password field (the part after ':' and
        before '@') as an empty string, so parsed.password is '' which is
        falsy.  mask_dsn_password therefore returns the DSN as-is, which is
        the correct behavior -- there is nothing sensitive to mask.
        """
        dsn = "postgresql://user:@localhost:5432/db"
        result = mask_dsn_password(dsn)
        # Empty password means parsed.password is '' (falsy), so no masking
        assert result == dsn

    def test_mask_preserves_scheme_and_path(self) -> None:
        """Scheme, path, query, and fragment are preserved after masking."""
        dsn = "postgresql://admin:s3cret@db.host:5436/mydb?sslmode=require"
        result = mask_dsn_password(dsn)
        assert result.startswith("postgresql://")
        assert "admin:***@db.host:5436" in result
        assert "/mydb" in result
        assert "sslmode=require" in result
        assert "s3cret" not in result

    def test_no_port(self) -> None:
        """DSN without port masks correctly."""
        dsn = "postgresql://user:pass@host/db"
        result = mask_dsn_password(dsn)
        assert "pass" not in result
        assert "***" in result
        assert "user:***@host" in result

    def test_password_in_query_param(self) -> None:
        """Password in query string parameter is masked by regex fallback."""
        dsn = "postgresql://user@host:5432/db?password=s3cret&sslmode=require"
        result = mask_dsn_password(dsn)
        assert "s3cret" not in result
        assert "password=***" in result
        assert "sslmode=require" in result

    def test_pwd_in_query_param(self) -> None:
        """pwd= in query string parameter is masked by regex fallback."""
        dsn = "postgresql://user@host:5432/db?pwd=s3cret"
        result = mask_dsn_password(dsn)
        assert "s3cret" not in result
        assert "pwd=***" in result

    def test_passwd_in_query_param(self) -> None:
        """passwd= in query string parameter is masked by regex fallback."""
        dsn = "postgresql://user@host:5432/db?passwd=s3cret"
        result = mask_dsn_password(dsn)
        assert "s3cret" not in result
        assert "passwd=***" in result


# =============================================================================
# Tests: Consumer Lifecycle
# =============================================================================


@pytest.mark.unit
class TestConsumerLifecycle:
    """Tests for ServiceLlmCostAggregator start/stop/context-manager."""

    @pytest.mark.asyncio
    async def test_start_when_already_running(
        self, config: ConfigLlmCostAggregation
    ) -> None:
        """Starting an already-running consumer logs warning and returns early."""
        service = ServiceLlmCostAggregator(config)
        service._running = True

        with patch(
            "omnibase_infra.services.observability.llm_cost_aggregation.consumer.logger"
        ) as mock_logger:
            await service.start()
            mock_logger.warning.assert_called_once()
            assert "already running" in mock_logger.warning.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(
        self, config: ConfigLlmCostAggregation
    ) -> None:
        """Stopping a non-running consumer is a no-op."""
        service = ServiceLlmCostAggregator(config)
        assert service._running is False

        # Should not raise or do anything
        await service.stop()
        assert service._running is False

    @pytest.mark.asyncio
    async def test_context_manager(
        self,
        config: ConfigLlmCostAggregation,
        mock_pool: MagicMock,
        mock_kafka_consumer: MagicMock,
    ) -> None:
        """Async context manager calls start() on enter and stop() on exit."""
        with (
            patch(
                "asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool
            ),
            patch(
                "omnibase_infra.services.observability.llm_cost_aggregation.consumer.AIOKafkaConsumer",
                return_value=mock_kafka_consumer,
            ),
            patch.object(
                ServiceLlmCostAggregator,
                "_start_health_server",
                new_callable=AsyncMock,
            ),
        ):
            async with ServiceLlmCostAggregator(config) as svc:
                assert svc._running is True
                assert svc._consumer is not None
                assert svc._pool is not None

            # After exit, stop() should have been called
            assert svc._running is False

    @pytest.mark.asyncio
    async def test_start_creates_resources(
        self,
        config: ConfigLlmCostAggregation,
        mock_pool: MagicMock,
        mock_kafka_consumer: MagicMock,
    ) -> None:
        """start() creates pool, writer, consumer, and health server."""
        with (
            patch(
                "asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool
            ),
            patch(
                "omnibase_infra.services.observability.llm_cost_aggregation.consumer.AIOKafkaConsumer",
                return_value=mock_kafka_consumer,
            ),
            patch.object(
                ServiceLlmCostAggregator,
                "_start_health_server",
                new_callable=AsyncMock,
            ) as mock_health,
        ):
            service = ServiceLlmCostAggregator(config)
            await service.start()

            assert service._running is True
            assert service._pool is mock_pool
            assert service._consumer is mock_kafka_consumer
            assert service._writer is not None
            mock_kafka_consumer.start.assert_awaited_once()
            mock_health.assert_awaited_once()

            # Cleanup
            await service.stop()

    @pytest.mark.asyncio
    async def test_start_failure_cleans_up(
        self,
        config: ConfigLlmCostAggregation,
        mock_pool: MagicMock,
    ) -> None:
        """If Kafka consumer fails to start, resources are cleaned up."""
        failing_consumer = MagicMock()
        failing_consumer.start = AsyncMock(side_effect=RuntimeError("Kafka down"))
        failing_consumer.stop = AsyncMock()

        with (
            patch(
                "asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool
            ),
            patch(
                "omnibase_infra.services.observability.llm_cost_aggregation.consumer.AIOKafkaConsumer",
                return_value=failing_consumer,
            ),
        ):
            service = ServiceLlmCostAggregator(config)

            with pytest.raises(RuntimeError, match="Kafka down"):
                await service.start()

            assert service._running is False
            # Pool should be cleaned up
            mock_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_properties(self, config: ConfigLlmCostAggregation) -> None:
        """is_running and consumer_id properties work correctly."""
        service = ServiceLlmCostAggregator(config)
        assert service.is_running is False
        assert service.consumer_id.startswith("llm-cost-aggregation-")


# =============================================================================
# Tests: _process_batch
# =============================================================================


@pytest.mark.unit
class TestProcessBatch:
    """Tests for ServiceLlmCostAggregator._process_batch."""

    @pytest.fixture
    def service_with_writer(
        self, config: ConfigLlmCostAggregation, mock_writer: MagicMock
    ) -> ServiceLlmCostAggregator:
        """Create a ServiceLlmCostAggregator with a mock writer injected."""
        service = ServiceLlmCostAggregator(config)
        service._writer = mock_writer
        return service

    @pytest.mark.asyncio
    async def test_null_messages_skipped(
        self, service_with_writer: ServiceLlmCostAggregator
    ) -> None:
        """Messages with None value are skipped and counted."""
        messages = [
            _make_consumer_record(partition=0, offset=0, value=None),
            _make_consumer_record(partition=0, offset=1, value=None),
        ]
        correlation_id = uuid4()
        offsets = await service_with_writer._process_batch(messages, correlation_id)

        # Skipped messages should still track offsets for commit
        snap = await service_with_writer.metrics.snapshot()
        assert snap["messages_skipped"] == 2

        # Offsets should be tracked even for null messages
        assert len(offsets) > 0

    @pytest.mark.asyncio
    async def test_json_decode_error(
        self, service_with_writer: ServiceLlmCostAggregator
    ) -> None:
        """Malformed JSON messages are logged and skipped."""
        messages = [
            _make_consumer_record(partition=0, offset=0, value=b"not valid json{{"),
        ]
        correlation_id = uuid4()
        offsets = await service_with_writer._process_batch(messages, correlation_id)

        snap = await service_with_writer.metrics.snapshot()
        assert snap["messages_skipped"] == 1

        # Writer should not have been called
        service_with_writer._writer.write_call_metrics.assert_not_awaited()

        # Offset for the skipped message should still be tracked
        assert len(offsets) > 0

    @pytest.mark.asyncio
    async def test_successful_batch(
        self, service_with_writer: ServiceLlmCostAggregator, mock_writer: MagicMock
    ) -> None:
        """Valid messages are written to metrics and aggregates."""
        event_payload = {
            "model_id": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "estimated_cost_usd": 0.005,
            "session_id": "session-123",
        }
        messages = [
            _make_consumer_record(
                partition=0,
                offset=5,
                value=json.dumps(event_payload).encode("utf-8"),
            ),
            _make_consumer_record(
                partition=0,
                offset=6,
                value=json.dumps(event_payload).encode("utf-8"),
            ),
            _make_consumer_record(
                partition=1,
                offset=0,
                value=json.dumps(event_payload).encode("utf-8"),
            ),
        ]
        correlation_id = uuid4()
        offsets = await service_with_writer._process_batch(messages, correlation_id)

        # Writer should have been called
        mock_writer.write_call_metrics.assert_awaited_once()
        mock_writer.write_cost_aggregates.assert_awaited_once()

        # Offsets should track highest per partition

        tp0 = TopicPartition("onex.evt.omniintelligence.llm-call-completed.v1", 0)
        tp1 = TopicPartition("onex.evt.omniintelligence.llm-call-completed.v1", 1)
        assert offsets[tp0] == 6  # Highest offset for partition 0
        assert offsets[tp1] == 0  # Only offset for partition 1

    @pytest.mark.asyncio
    async def test_aggregation_failure_nonfatal(
        self, service_with_writer: ServiceLlmCostAggregator, mock_writer: MagicMock
    ) -> None:
        """write_cost_aggregates failure is non-fatal; raw metrics are still committed."""
        mock_writer.write_cost_aggregates = AsyncMock(
            side_effect=RuntimeError("aggregation error")
        )

        event_payload = {"model_id": "gpt-4o", "total_tokens": 100}
        messages = [
            _make_consumer_record(
                partition=0,
                offset=10,
                value=json.dumps(event_payload).encode("utf-8"),
            ),
        ]
        correlation_id = uuid4()
        offsets = await service_with_writer._process_batch(messages, correlation_id)

        # Raw metrics write should still have been called
        mock_writer.write_call_metrics.assert_awaited_once()

        # Offsets should still be tracked (raw metrics succeeded)

        tp = TopicPartition("onex.evt.omniintelligence.llm-call-completed.v1", 0)
        assert offsets[tp] == 10

    @pytest.mark.asyncio
    async def test_offset_tracking_highest_per_partition(
        self, service_with_writer: ServiceLlmCostAggregator
    ) -> None:
        """Offset tracking records the highest offset per partition."""
        event_payload = {"model_id": "test-model"}
        messages = [
            _make_consumer_record(
                partition=0,
                offset=3,
                value=json.dumps(event_payload).encode("utf-8"),
            ),
            _make_consumer_record(
                partition=0,
                offset=7,
                value=json.dumps(event_payload).encode("utf-8"),
            ),
            _make_consumer_record(
                partition=0,
                offset=5,
                value=json.dumps(event_payload).encode("utf-8"),
            ),
        ]
        correlation_id = uuid4()
        offsets = await service_with_writer._process_batch(messages, correlation_id)

        tp = TopicPartition("onex.evt.omniintelligence.llm-call-completed.v1", 0)
        assert offsets[tp] == 7  # Highest offset

    @pytest.mark.asyncio
    async def test_metrics_write_failure_skips_aggregation(
        self, service_with_writer: ServiceLlmCostAggregator, mock_writer: MagicMock
    ) -> None:
        """If write_call_metrics fails, aggregation is skipped entirely."""
        mock_writer.write_call_metrics = AsyncMock(side_effect=RuntimeError("db error"))

        event_payload = {"model_id": "gpt-4o"}
        messages = [
            _make_consumer_record(
                partition=0,
                offset=0,
                value=json.dumps(event_payload).encode("utf-8"),
            ),
        ]
        correlation_id = uuid4()
        offsets = await service_with_writer._process_batch(messages, correlation_id)

        # Aggregation should NOT have been called
        mock_writer.write_cost_aggregates.assert_not_awaited()

        # Failed count should be recorded
        snap = await service_with_writer.metrics.snapshot()
        assert snap["messages_failed"] == 1

    @pytest.mark.asyncio
    async def test_mixed_null_and_valid_messages(
        self, service_with_writer: ServiceLlmCostAggregator, mock_writer: MagicMock
    ) -> None:
        """Batch with both null and valid messages processes correctly."""
        event_payload = {"model_id": "test-model"}
        messages = [
            _make_consumer_record(partition=0, offset=0, value=None),
            _make_consumer_record(
                partition=0,
                offset=1,
                value=json.dumps(event_payload).encode("utf-8"),
            ),
            _make_consumer_record(partition=0, offset=2, value=None),
        ]
        correlation_id = uuid4()
        offsets = await service_with_writer._process_batch(messages, correlation_id)

        # 2 null messages skipped
        snap = await service_with_writer.metrics.snapshot()
        assert snap["messages_skipped"] == 2

        # Writer should have been called with the 1 valid event
        mock_writer.write_call_metrics.assert_awaited_once()
        events_arg = mock_writer.write_call_metrics.call_args[0][0]
        assert len(events_arg) == 1

        # Highest offset should be tracked across both null and valid

        tp = TopicPartition("onex.evt.omniintelligence.llm-call-completed.v1", 0)
        assert offsets[tp] == 2  # Highest across skipped + successful

    @pytest.mark.asyncio
    async def test_non_dict_json_skipped(
        self, service_with_writer: ServiceLlmCostAggregator
    ) -> None:
        """JSON values that parse to non-dict types are skipped."""
        messages = [
            _make_consumer_record(
                partition=0,
                offset=0,
                value=json.dumps([1, 2, 3]).encode("utf-8"),
            ),
            _make_consumer_record(
                partition=0,
                offset=1,
                value=json.dumps("just a string").encode("utf-8"),
            ),
        ]
        correlation_id = uuid4()
        await service_with_writer._process_batch(messages, correlation_id)

        snap = await service_with_writer.metrics.snapshot()
        assert snap["messages_skipped"] == 2

    @pytest.mark.asyncio
    async def test_empty_batch_returns_empty_offsets(
        self, service_with_writer: ServiceLlmCostAggregator
    ) -> None:
        """An empty message list returns empty offsets."""
        correlation_id = uuid4()
        offsets = await service_with_writer._process_batch([], correlation_id)
        assert offsets == {}

    @pytest.mark.asyncio
    async def test_no_writer_returns_empty_offsets(
        self, config: ConfigLlmCostAggregation
    ) -> None:
        """If writer is None, _process_batch returns empty offsets."""
        service = ServiceLlmCostAggregator(config)
        service._writer = None

        event_payload = {"model_id": "test"}
        messages = [
            _make_consumer_record(
                partition=0,
                offset=0,
                value=json.dumps(event_payload).encode("utf-8"),
            ),
        ]
        correlation_id = uuid4()
        offsets = await service._process_batch(messages, correlation_id)
        assert offsets == {}
