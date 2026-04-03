# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ContextAuditConsumer (OMN-5240).

This module tests:
    - Config: defaults, env prefix, topic validation
    - Consumer: message parsing, batch processing, offset tracking
    - Writer: schema pre-filter, batch write, circuit breaker state
    - Health check: status transitions (HEALTHY, DEGRADED, UNHEALTHY)
    - mask_dsn_password: password masking utility

All tests mock aiokafka and asyncpg — no real Kafka/PostgreSQL required.

Related Tickets:
    - OMN-5240: Create audit event Kafka consumer (current)
    - OMN-5234: Create audit event schemas and topics in omniclaude (producer)
    - OMN-5239: Add context_audit_events DB migration (table schema)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.services.observability.context_audit.config import (
    ConfigContextAuditConsumer,
)
from omnibase_infra.services.observability.context_audit.consumer import (
    ContextAuditConsumer,
    EnumHealthStatus,
    mask_dsn_password,
)
from omnibase_infra.services.observability.context_audit.writer_postgres import (
    WriterContextAuditPostgres,
)

# =============================================================================
# Helpers
# =============================================================================


def make_mock_consumer_record(
    topic: str,
    partition: int,
    offset: int,
    value: dict[str, object],
) -> MagicMock:
    """Create a mock Kafka ConsumerRecord."""
    record = MagicMock()
    record.topic = topic
    record.partition = partition
    record.offset = offset
    record.value = json.dumps(value).encode("utf-8")
    return record


def make_audit_event(
    event_type: str = "audit-dispatch-validated",
    enforcement_level: str = "STRICT",
    task_id: str | None = None,
) -> dict[str, object]:
    """Create a minimal valid audit event dict."""
    return {
        "task_id": task_id or str(uuid4()),
        "parent_task_id": None,
        "correlation_id": str(uuid4()),
        "contract_id": "test-contract",
        "event_type": event_type,
        "enforcement_level": enforcement_level,
        "enforcement_action": None,
        "violation_details": None,
        "context_tokens_used": 1000,
        "context_budget_tokens": 2000,
        "return_tokens": None,
        "return_max_tokens": None,
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> ConfigContextAuditConsumer:
    """Create a test configuration with non-conflicting port."""
    return ConfigContextAuditConsumer(
        kafka_bootstrap_servers="localhost:19092",
        postgres_dsn="postgresql://test:test@localhost:5432/test",
        batch_size=10,
        batch_timeout_ms=500,
        health_check_port=18093,  # Non-standard port to avoid conflicts in tests
    )


@pytest.fixture
def consumer(mock_config: ConfigContextAuditConsumer) -> ContextAuditConsumer:
    """Create a consumer instance (not started)."""
    return ContextAuditConsumer(mock_config)


# =============================================================================
# Config Tests
# =============================================================================


@pytest.mark.unit
class TestConfigContextAuditConsumer:
    """Tests for ConfigContextAuditConsumer."""

    def test_default_topics(self) -> None:
        """Default topic list contains all 5 audit topics."""
        config = ConfigContextAuditConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:test@localhost:5432/test",
        )
        assert "onex.evt.omniclaude.audit-dispatch-validated.v1" in config.topics
        assert "onex.evt.omniclaude.audit-scope-violation.v1" in config.topics
        assert "onex.evt.omniclaude.audit-context-budget-exceeded.v1" in config.topics
        assert "onex.evt.omniclaude.audit-return-bounded.v1" in config.topics
        assert "onex.evt.omniclaude.audit-compression-triggered.v1" in config.topics
        assert len(config.topics) == 5

    def test_default_health_check_port(self) -> None:
        """Default health check port is 8093."""
        config = ConfigContextAuditConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:test@localhost:5432/test",
        )
        assert config.health_check_port == 8093

    def test_default_group_id(self) -> None:
        """Default consumer group ID is context-audit-postgres."""
        config = ConfigContextAuditConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:test@localhost:5432/test",
        )
        assert config.kafka_group_id == "context-audit-postgres"

    def test_empty_topics_raises(self) -> None:
        """Empty topics list raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError, match="No topics configured"):
            ConfigContextAuditConsumer(
                kafka_bootstrap_servers="localhost:19092",
                postgres_dsn="postgresql://test:test@localhost:5432/test",
                topics=[],
            )

    def test_custom_topics(self) -> None:
        """Custom topics list is accepted."""
        config = ConfigContextAuditConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:test@localhost:5432/test",
            topics=["onex.evt.omniclaude.audit-dispatch-validated.v1"],
        )
        assert config.topics == ["onex.evt.omniclaude.audit-dispatch-validated.v1"]


# =============================================================================
# mask_dsn_password Tests
# =============================================================================


@pytest.mark.unit
class TestMaskDsnPassword:
    """Tests for mask_dsn_password utility."""

    def test_masks_password(self) -> None:
        """Password is replaced with ***."""
        dsn = "postgresql://user:secret@localhost:5432/db"
        result = mask_dsn_password(dsn)
        assert "secret" not in result
        assert "***" in result

    def test_no_password_unchanged(self) -> None:
        """DSN without password is returned unchanged."""
        dsn = "postgresql://localhost:5432/db"
        assert mask_dsn_password(dsn) == dsn

    def test_invalid_dsn_returns_as_is(self) -> None:
        """Unparseable DSN is returned as-is without raising."""
        assert mask_dsn_password("not-a-url") == "not-a-url"


# =============================================================================
# Consumer Message Parsing Tests
# =============================================================================


@pytest.mark.unit
class TestContextAuditConsumerParsing:
    """Tests for _parse_message."""

    def test_parse_valid_dict_message(self, consumer: ContextAuditConsumer) -> None:
        """Valid JSON dict message parses correctly."""
        payload = make_audit_event()
        record = make_mock_consumer_record(
            "onex.evt.omniclaude.audit-dispatch-validated.v1", 0, 0, payload
        )
        result = consumer._parse_message(record)
        assert result is not None
        assert result["task_id"] == payload["task_id"]

    def test_parse_invalid_json_returns_none(
        self, consumer: ContextAuditConsumer
    ) -> None:
        """Invalid JSON returns None."""
        record = MagicMock()
        record.topic = "onex.evt.omniclaude.audit-dispatch-validated.v1"
        record.partition = 0
        record.offset = 0
        record.value = b"not-json"
        result = consumer._parse_message(record)
        assert result is None

    def test_parse_array_wrapped_legacy(self, consumer: ContextAuditConsumer) -> None:
        """Array-wrapped single event (legacy format) is unwrapped."""
        payload = make_audit_event()
        record = MagicMock()
        record.topic = "onex.evt.omniclaude.audit-dispatch-validated.v1"
        record.partition = 0
        record.offset = 0
        record.value = json.dumps([payload]).encode("utf-8")
        result = consumer._parse_message(record)
        assert result is not None
        assert result["task_id"] == payload["task_id"]

    def test_parse_list_with_multiple_items_returns_none(
        self, consumer: ContextAuditConsumer
    ) -> None:
        """Array with multiple items is rejected."""
        payload1 = make_audit_event()
        payload2 = make_audit_event()
        record = MagicMock()
        record.topic = "onex.evt.omniclaude.audit-dispatch-validated.v1"
        record.partition = 0
        record.offset = 0
        record.value = json.dumps([payload1, payload2]).encode("utf-8")
        result = consumer._parse_message(record)
        assert result is None


# =============================================================================
# Writer Schema Pre-Filter Tests
# =============================================================================


@pytest.mark.unit
class TestWriterContextAuditPostgres:
    """Tests for WriterContextAuditPostgres."""

    def test_empty_batch_returns_zero(self) -> None:
        """Empty event list returns 0 without calling DB."""
        pool = MagicMock()
        writer = WriterContextAuditPostgres(pool)

        import asyncio

        result = asyncio.run(writer.write_audit_events([]))
        assert result == 0

    @pytest.mark.asyncio
    async def test_missing_required_fields_skipped(self) -> None:
        """Events missing required fields are skipped with WARNING."""
        pool = MagicMock()
        writer = WriterContextAuditPostgres(pool)

        # Missing task_id
        invalid_event: dict[str, object] = {
            "correlation_id": str(uuid4()),
            "event_type": "audit-dispatch-validated",
            "enforcement_level": "STRICT",
        }

        # write_audit_events should skip the invalid event and return 0
        # (pool.acquire() should never be called)
        result = await writer.write_audit_events([invalid_event])
        assert result == 0
        pool.acquire.assert_not_called()

    def test_get_circuit_breaker_state_returns_dict(self) -> None:
        """get_circuit_breaker_state returns a dict."""
        pool = MagicMock()
        writer = WriterContextAuditPostgres(pool)
        state = writer.get_circuit_breaker_state()
        assert isinstance(state, dict)


# =============================================================================
# Batch Processing Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestContextAuditConsumerBatchProcessing:
    """Tests for _process_batch and offset tracking."""

    async def test_process_batch_records_offsets(
        self, consumer: ContextAuditConsumer
    ) -> None:
        """Batch processing returns correct per-partition offsets on success."""
        topic = "onex.evt.omniclaude.audit-dispatch-validated.v1"
        records = [
            make_mock_consumer_record(topic, 0, 0, make_audit_event()),
            make_mock_consumer_record(topic, 0, 1, make_audit_event()),
            make_mock_consumer_record(topic, 1, 5, make_audit_event()),
        ]

        # Mock the writer
        mock_writer = AsyncMock()
        mock_writer.write_audit_events = AsyncMock(return_value=3)
        consumer._writer = mock_writer

        from aiokafka import TopicPartition

        committed = await consumer._process_batch(records)

        # Max offset per partition
        assert committed[TopicPartition(topic, 0)] == 1
        assert committed[TopicPartition(topic, 1)] == 5

    async def test_process_batch_excludes_failed_partitions(
        self, consumer: ContextAuditConsumer
    ) -> None:
        """Partitions with write failures are excluded from commit map."""
        topic = "onex.evt.omniclaude.audit-dispatch-validated.v1"
        records = [
            make_mock_consumer_record(topic, 0, 0, make_audit_event()),
        ]

        # Mock writer that raises
        mock_writer = AsyncMock()
        mock_writer.write_audit_events = AsyncMock(side_effect=RuntimeError("DB error"))
        consumer._writer = mock_writer

        committed = await consumer._process_batch(records)
        assert len(committed) == 0

    async def test_process_batch_skips_unparseable_messages(
        self, consumer: ContextAuditConsumer
    ) -> None:
        """Unparseable messages are sent to DLQ, not included in write batch."""
        topic = "onex.evt.omniclaude.audit-dispatch-validated.v1"

        bad_record = MagicMock()
        bad_record.topic = topic
        bad_record.partition = 0
        bad_record.offset = 0
        bad_record.value = b"invalid-json"

        mock_writer = AsyncMock()
        mock_writer.write_audit_events = AsyncMock(return_value=0)
        consumer._writer = mock_writer

        # DLQ disabled for this test
        consumer.config = ConfigContextAuditConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:test@localhost:5432/test",
            dlq_enabled=False,
        )

        committed = await consumer._process_batch([bad_record])
        # Bad record offset is not committed
        assert len(committed) == 0


# =============================================================================
# Health Check Tests
# =============================================================================


@pytest.mark.unit
class TestContextAuditConsumerHealthCheck:
    """Tests for _build_health_response."""

    def test_unhealthy_when_not_running(self, consumer: ContextAuditConsumer) -> None:
        """Status is UNHEALTHY when consumer is not running."""
        consumer._running = False
        response, http_code = consumer._build_health_response()
        assert response["status"] == str(EnumHealthStatus.UNHEALTHY)
        assert http_code == 503

    def test_healthy_when_running_and_no_messages(
        self, consumer: ContextAuditConsumer
    ) -> None:
        """Status is HEALTHY when running and no messages received (idle)."""
        consumer._running = True
        consumer.metrics.last_poll_at = datetime.now(UTC)
        consumer.metrics.messages_received = 0

        response, http_code = consumer._build_health_response()
        assert response["status"] == str(EnumHealthStatus.HEALTHY)
        assert http_code == 200
        assert response["idle"] is True

    def test_degraded_when_no_polls(self, consumer: ContextAuditConsumer) -> None:
        """Status is DEGRADED when consumer is running but has never polled."""
        consumer._running = True
        consumer.metrics.last_poll_at = None

        response, http_code = consumer._build_health_response()
        assert response["status"] == str(EnumHealthStatus.DEGRADED)
        assert http_code == 503
