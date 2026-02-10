# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for AgentActionsConsumer.

This module tests:
    - Batch processing (message parsing, model validation, routing)
    - Offset tracking (per-partition, only on success)
    - Health check endpoint (status transitions)
    - Consumer lifecycle (start, stop, context manager)

All tests mock aiokafka and asyncpg - no real Kafka/PostgreSQL required.

Related Tickets:
    - OMN-1743: Migrate agent_actions_consumer to omnibase_infra
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiohttp.test_utils import TestClient

from omnibase_core.errors import OnexError
from omnibase_infra.services.observability.agent_actions.config import (
    ConfigAgentActionsConsumer,
)
from omnibase_infra.services.observability.agent_actions.consumer import (
    TOPIC_TO_MODEL,
    TOPIC_TO_WRITER_METHOD,
    AgentActionsConsumer,
    ConsumerMetrics,
    EnumHealthStatus,
    mask_dsn_password,
)

if TYPE_CHECKING:
    from aiokafka import TopicPartition


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> ConfigAgentActionsConsumer:
    """Create a test configuration."""
    return ConfigAgentActionsConsumer(
        kafka_bootstrap_servers="localhost:9092",
        postgres_dsn="postgresql://test:test@localhost:5432/test",
        batch_size=10,
        batch_timeout_ms=500,
        health_check_port=18087,  # Non-standard port to avoid conflicts
    )


@pytest.fixture
def consumer(mock_config: ConfigAgentActionsConsumer) -> AgentActionsConsumer:
    """Create a consumer instance (not started)."""
    return AgentActionsConsumer(mock_config)


@pytest.fixture
def sample_agent_action_payload() -> dict[str, object]:
    """Create a sample agent action JSON payload."""
    return {
        "id": str(uuid4()),
        "correlation_id": str(uuid4()),
        "agent_name": "test-agent",
        "action_type": "tool_call",
        "action_name": "Read",
        "created_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def sample_routing_decision_payload() -> dict[str, object]:
    """Create a sample routing decision JSON payload."""
    return {
        "id": str(uuid4()),
        "correlation_id": str(uuid4()),
        "selected_agent": "api-architect",
        "confidence_score": 0.95,
        "created_at": datetime.now(UTC).isoformat(),
    }


def make_mock_consumer_record(
    topic: str,
    partition: int,
    offset: int,
    value: dict[str, object],
) -> MagicMock:
    """Create a mock ConsumerRecord."""
    record = MagicMock()
    record.topic = topic
    record.partition = partition
    record.offset = offset
    record.value = json.dumps(value).encode("utf-8")
    return record


# =============================================================================
# Topic/Model Mapping Tests
# =============================================================================


class TestTopicModelMapping:
    """Test topic to model and writer method mappings."""

    def test_all_topics_have_models(self) -> None:
        """All configured topics should have corresponding models."""
        expected_topics = {
            "agent-actions",
            "agent-routing-decisions",
            "agent-transformation-events",
            "router-performance-metrics",
            "agent-detection-failures",
            "agent-execution-logs",
            "onex.evt.agent.status.v1",
        }
        assert set(TOPIC_TO_MODEL.keys()) == expected_topics

    def test_all_topics_have_writer_methods(self) -> None:
        """All configured topics should have corresponding writer methods."""
        expected_topics = {
            "agent-actions",
            "agent-routing-decisions",
            "agent-transformation-events",
            "router-performance-metrics",
            "agent-detection-failures",
            "agent-execution-logs",
            "onex.evt.agent.status.v1",
        }
        assert set(TOPIC_TO_WRITER_METHOD.keys()) == expected_topics

    def test_topic_to_model_mapping_correct(self) -> None:
        """Topic to model mapping should be correct."""
        from omnibase_infra.services.observability.agent_actions.models import (
            ModelAgentAction,
            ModelAgentStatusEvent,
            ModelDetectionFailure,
            ModelExecutionLog,
            ModelPerformanceMetric,
            ModelRoutingDecision,
            ModelTransformationEvent,
        )

        assert TOPIC_TO_MODEL["agent-actions"] is ModelAgentAction
        assert TOPIC_TO_MODEL["agent-routing-decisions"] is ModelRoutingDecision
        assert TOPIC_TO_MODEL["agent-transformation-events"] is ModelTransformationEvent
        assert TOPIC_TO_MODEL["router-performance-metrics"] is ModelPerformanceMetric
        assert TOPIC_TO_MODEL["agent-detection-failures"] is ModelDetectionFailure
        assert TOPIC_TO_MODEL["agent-execution-logs"] is ModelExecutionLog
        assert TOPIC_TO_MODEL["onex.evt.agent.status.v1"] is ModelAgentStatusEvent


# =============================================================================
# DSN Password Masking Tests
# =============================================================================


class TestMaskDsnPassword:
    """Test DSN password masking utility function."""

    def test_mask_standard_dsn_with_password(self) -> None:
        """Standard DSN with password should have password masked."""
        dsn = "postgresql://user:secret@localhost:5432/db"
        result = mask_dsn_password(dsn)
        assert result == "postgresql://user:***@localhost:5432/db"

    def test_mask_dsn_without_port(self) -> None:
        """DSN without explicit port should be handled correctly."""
        dsn = "postgresql://user:password@localhost/db"
        result = mask_dsn_password(dsn)
        assert result == "postgresql://user:***@localhost/db"

    def test_mask_dsn_without_password(self) -> None:
        """DSN without password should be returned unchanged."""
        dsn = "postgresql://user@localhost:5432/db"
        result = mask_dsn_password(dsn)
        assert result == dsn

    def test_mask_dsn_with_complex_password(self) -> None:
        """DSN with special characters in password should be masked."""
        dsn = "postgresql://user:p%40ss%2Fword@localhost:5432/db"
        result = mask_dsn_password(dsn)
        assert result == "postgresql://user:***@localhost:5432/db"

    def test_mask_dsn_with_query_params(self) -> None:
        """DSN with query parameters should preserve them."""
        dsn = "postgresql://user:secret@localhost:5432/db?sslmode=require"
        result = mask_dsn_password(dsn)
        assert result == "postgresql://user:***@localhost:5432/db?sslmode=require"

    def test_mask_invalid_dsn_returns_original(self) -> None:
        """Invalid DSN should be returned unchanged."""
        dsn = "not-a-valid-dsn"
        result = mask_dsn_password(dsn)
        assert result == dsn

    def test_mask_empty_string(self) -> None:
        """Empty string should be returned unchanged."""
        result = mask_dsn_password("")
        assert result == ""

    def test_mask_dsn_ipv4_host(self) -> None:
        """DSN with IPv4 host should be handled correctly."""
        dsn = "postgresql://postgres:mysecret@192.168.1.100:5436/omnibase_infra"
        result = mask_dsn_password(dsn)
        assert result == "postgresql://postgres:***@192.168.1.100:5436/omnibase_infra"


# =============================================================================
# Consumer Metrics Tests
# =============================================================================


class TestConsumerMetrics:
    """Test ConsumerMetrics tracking."""

    @pytest.mark.asyncio
    async def test_initial_metrics_zero(self) -> None:
        """Metrics should start at zero."""
        metrics = ConsumerMetrics()

        assert metrics.messages_received == 0
        assert metrics.messages_processed == 0
        assert metrics.messages_failed == 0
        assert metrics.messages_skipped == 0
        assert metrics.batches_processed == 0
        assert metrics.last_poll_at is None
        assert metrics.last_successful_write_at is None

    @pytest.mark.asyncio
    async def test_record_received_increments(self) -> None:
        """record_received should increment counter and update timestamp."""
        metrics = ConsumerMetrics()

        await metrics.record_received(5)

        assert metrics.messages_received == 5
        assert metrics.last_poll_at is not None

    @pytest.mark.asyncio
    async def test_record_processed_increments(self) -> None:
        """record_processed should increment counter and update timestamp."""
        metrics = ConsumerMetrics()

        await metrics.record_processed(10)

        assert metrics.messages_processed == 10
        assert metrics.last_successful_write_at is not None

    @pytest.mark.asyncio
    async def test_record_failed_increments(self) -> None:
        """record_failed should increment counter."""
        metrics = ConsumerMetrics()

        await metrics.record_failed(3)

        assert metrics.messages_failed == 3

    @pytest.mark.asyncio
    async def test_record_skipped_increments(self) -> None:
        """record_skipped should increment counter."""
        metrics = ConsumerMetrics()

        await metrics.record_skipped(2)

        assert metrics.messages_skipped == 2

    @pytest.mark.asyncio
    async def test_record_batch_processed_increments(self) -> None:
        """record_batch_processed should increment counter."""
        metrics = ConsumerMetrics()

        await metrics.record_batch_processed()
        await metrics.record_batch_processed()

        assert metrics.batches_processed == 2

    @pytest.mark.asyncio
    async def test_record_polled_updates_last_poll_at(self) -> None:
        """record_polled should update last_poll_at timestamp.

        This ensures empty polls still update the timestamp, preventing
        false DEGRADED health status on low-traffic topics.
        """
        metrics = ConsumerMetrics()

        assert metrics.last_poll_at is None

        await metrics.record_polled()

        assert metrics.last_poll_at is not None

    @pytest.mark.asyncio
    async def test_record_polled_does_not_increment_received(self) -> None:
        """record_polled should NOT increment messages_received counter.

        This distinguishes it from record_received() which increments
        the counter AND updates the timestamp.
        """
        metrics = ConsumerMetrics()

        await metrics.record_polled()

        assert metrics.messages_received == 0
        assert metrics.last_poll_at is not None

    @pytest.mark.asyncio
    async def test_snapshot_returns_dict(self) -> None:
        """snapshot should return dictionary with all metrics."""
        metrics = ConsumerMetrics()
        await metrics.record_received(10)
        await metrics.record_processed(8)
        await metrics.record_failed(1)
        await metrics.record_skipped(1)
        await metrics.record_batch_processed()

        snapshot = await metrics.snapshot()

        assert snapshot["messages_received"] == 10
        assert snapshot["messages_processed"] == 8
        assert snapshot["messages_failed"] == 1
        assert snapshot["messages_skipped"] == 1
        assert snapshot["batches_processed"] == 1
        assert snapshot["last_poll_at"] is not None
        assert snapshot["last_successful_write_at"] is not None


# =============================================================================
# Health Status Enum Tests
# =============================================================================


class TestEnumHealthStatus:
    """Test health status enum values."""

    def test_health_status_values(self) -> None:
        """Health status should have expected values."""
        assert EnumHealthStatus.HEALTHY.value == "healthy"
        assert EnumHealthStatus.DEGRADED.value == "degraded"
        assert EnumHealthStatus.UNHEALTHY.value == "unhealthy"


# =============================================================================
# Consumer Initialization Tests
# =============================================================================


class TestConsumerInitialization:
    """Test consumer initialization."""

    def test_consumer_not_running_initially(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Consumer should not be running after initialization."""
        assert consumer.is_running is False

    def test_consumer_has_unique_id(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Consumer should have a unique consumer_id."""
        assert consumer.consumer_id.startswith("agent-actions-consumer-")
        assert len(consumer.consumer_id) > len("agent-actions-consumer-")

    def test_consumer_metrics_initialized(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Consumer should have initialized metrics."""
        assert consumer.metrics is not None
        assert consumer.metrics.messages_received == 0


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    """Test batch processing logic."""

    @pytest.mark.asyncio
    async def test_process_batch_parses_json_messages(
        self,
        consumer: AgentActionsConsumer,
        sample_agent_action_payload: dict[str, object],
    ) -> None:
        """Batch processing should parse JSON messages correctly."""
        # Setup mocks
        mock_writer = AsyncMock()
        mock_writer.write_agent_actions = AsyncMock(return_value=1)
        consumer._writer = mock_writer
        consumer._running = True

        # Create mock message
        message = make_mock_consumer_record(
            topic="agent-actions",
            partition=0,
            offset=100,
            value=sample_agent_action_payload,
        )

        # Process batch
        result = await consumer._process_batch([message], uuid4())

        # Verify writer was called
        mock_writer.write_agent_actions.assert_called_once()

        # Verify offset tracking
        from aiokafka import TopicPartition

        tp = TopicPartition("agent-actions", 0)
        assert tp in result
        assert result[tp] == 100

    @pytest.mark.asyncio
    async def test_process_batch_routes_to_correct_writer(
        self,
        consumer: AgentActionsConsumer,
        sample_routing_decision_payload: dict[str, object],
    ) -> None:
        """Batch processing should route messages to correct writer method."""
        mock_writer = AsyncMock()
        mock_writer.write_routing_decisions = AsyncMock(return_value=1)
        consumer._writer = mock_writer
        consumer._running = True

        message = make_mock_consumer_record(
            topic="agent-routing-decisions",
            partition=0,
            offset=50,
            value=sample_routing_decision_payload,
        )

        await consumer._process_batch([message], uuid4())

        # Should call routing decisions writer, not agent actions
        mock_writer.write_routing_decisions.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_batch_skips_invalid_json(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Invalid JSON messages should be skipped."""
        mock_writer = AsyncMock()
        consumer._writer = mock_writer
        consumer._running = True

        # Create message with invalid JSON
        message = MagicMock()
        message.topic = "agent-actions"
        message.partition = 0
        message.offset = 100
        message.value = b"not valid json"

        result = await consumer._process_batch([message], uuid4())

        # Should still track offset (skip and continue)
        from aiokafka import TopicPartition

        tp = TopicPartition("agent-actions", 0)
        assert tp in result
        assert result[tp] == 100

    @pytest.mark.asyncio
    async def test_process_batch_skips_validation_failures(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Messages failing validation should be skipped."""
        mock_writer = AsyncMock()
        consumer._writer = mock_writer
        consumer._running = True

        # Create message with valid JSON but invalid model (missing required fields)
        message = make_mock_consumer_record(
            topic="agent-actions",
            partition=0,
            offset=100,
            value={"invalid": "payload"},  # Missing required fields
        )

        result = await consumer._process_batch([message], uuid4())

        # Should track offset for skipped message
        from aiokafka import TopicPartition

        tp = TopicPartition("agent-actions", 0)
        assert tp in result
        assert result[tp] == 100

    @pytest.mark.asyncio
    async def test_process_batch_handles_unknown_topic(
        self,
        consumer: AgentActionsConsumer,
        sample_agent_action_payload: dict[str, object],
    ) -> None:
        """Unknown topics should be skipped but offset tracked."""
        mock_writer = AsyncMock()
        consumer._writer = mock_writer
        consumer._running = True

        message = make_mock_consumer_record(
            topic="unknown-topic",
            partition=0,
            offset=100,
            value=sample_agent_action_payload,
        )

        result = await consumer._process_batch([message], uuid4())

        from aiokafka import TopicPartition

        tp = TopicPartition("unknown-topic", 0)
        assert tp in result
        assert result[tp] == 100

    @pytest.mark.asyncio
    async def test_process_batch_skips_tombstone_messages(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Tombstone messages (value=None) should be skipped but offset tracked."""
        mock_writer = AsyncMock()
        consumer._writer = mock_writer
        consumer._running = True

        # Create tombstone message (value is None)
        message = MagicMock()
        message.topic = "agent-actions"
        message.partition = 0
        message.offset = 100
        message.value = None  # Tombstone

        result = await consumer._process_batch([message], uuid4())

        from aiokafka import TopicPartition

        tp = TopicPartition("agent-actions", 0)
        assert tp in result
        assert result[tp] == 100

    @pytest.mark.asyncio
    async def test_process_batch_skips_invalid_utf8_messages(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Messages with invalid UTF-8 encoding should be skipped but offset tracked."""
        mock_writer = AsyncMock()
        consumer._writer = mock_writer
        consumer._running = True

        # Create message with invalid UTF-8 bytes
        message = MagicMock()
        message.topic = "agent-actions"
        message.partition = 0
        message.offset = 100
        message.value = b"\xff\xfe invalid utf-8"  # Invalid UTF-8 sequence

        result = await consumer._process_batch([message], uuid4())

        from aiokafka import TopicPartition

        tp = TopicPartition("agent-actions", 0)
        assert tp in result
        assert result[tp] == 100

    @pytest.mark.asyncio
    async def test_skipped_offsets_preserved_on_write_failure(
        self,
        consumer: AgentActionsConsumer,
        sample_agent_action_payload: dict[str, object],
    ) -> None:
        """Skipped message offsets should be preserved even when write fails.

        Scenario:
        - Message A (offset 100): Invalid JSON -> skipped, offset tracked
        - Message B (offset 101): Valid -> write attempt fails
        - Expected: offset 100 should still be in result (not lost)
        """
        mock_writer = AsyncMock()
        mock_writer.write_agent_actions = AsyncMock(
            side_effect=Exception("Database error")
        )
        consumer._writer = mock_writer
        consumer._running = True

        # Skipped message (invalid JSON) at offset 100
        skipped_message = MagicMock()
        skipped_message.topic = "agent-actions"
        skipped_message.partition = 0
        skipped_message.offset = 100
        skipped_message.value = b"not valid json"

        # Valid message at offset 101 (will fail write)
        valid_message = make_mock_consumer_record(
            topic="agent-actions",
            partition=0,
            offset=101,
            value=sample_agent_action_payload,
        )

        result = await consumer._process_batch(
            [skipped_message, valid_message], uuid4()
        )

        from aiokafka import TopicPartition

        tp = TopicPartition("agent-actions", 0)
        # Skipped offset should be preserved even though write failed
        assert tp in result
        # The highest skipped offset (100) should be there, not 101 (which failed)
        assert result[tp] == 100


# =============================================================================
# Offset Tracking Tests
# =============================================================================


class TestOffsetTracking:
    """Test per-partition offset tracking."""

    @pytest.mark.asyncio
    async def test_successful_write_updates_offsets(
        self,
        consumer: AgentActionsConsumer,
        sample_agent_action_payload: dict[str, object],
    ) -> None:
        """Successful writes should update offsets for that partition."""
        mock_writer = AsyncMock()
        mock_writer.write_agent_actions = AsyncMock(return_value=1)
        consumer._writer = mock_writer
        consumer._running = True

        message = make_mock_consumer_record(
            topic="agent-actions",
            partition=2,
            offset=500,
            value=sample_agent_action_payload,
        )

        result = await consumer._process_batch([message], uuid4())

        from aiokafka import TopicPartition

        tp = TopicPartition("agent-actions", 2)
        assert tp in result
        assert result[tp] == 500

    @pytest.mark.asyncio
    async def test_failed_write_does_not_update_offsets(
        self,
        consumer: AgentActionsConsumer,
        sample_agent_action_payload: dict[str, object],
    ) -> None:
        """Failed writes should not update offsets for that partition."""
        mock_writer = AsyncMock()
        mock_writer.write_agent_actions = AsyncMock(
            side_effect=Exception("Database error")
        )
        consumer._writer = mock_writer
        consumer._running = True

        message = make_mock_consumer_record(
            topic="agent-actions",
            partition=0,
            offset=100,
            value=sample_agent_action_payload,
        )

        result = await consumer._process_batch([message], uuid4())

        from aiokafka import TopicPartition

        tp = TopicPartition("agent-actions", 0)
        # Failed partition should NOT be in result
        assert tp not in result

    @pytest.mark.asyncio
    async def test_multiple_partitions_tracked_independently(
        self,
        consumer: AgentActionsConsumer,
        sample_agent_action_payload: dict[str, object],
    ) -> None:
        """Each partition should have independent offset tracking."""
        mock_writer = AsyncMock()
        mock_writer.write_agent_actions = AsyncMock(return_value=2)
        consumer._writer = mock_writer
        consumer._running = True

        # Messages from different partitions
        payload1 = sample_agent_action_payload.copy()
        payload1["id"] = str(uuid4())
        payload2 = sample_agent_action_payload.copy()
        payload2["id"] = str(uuid4())

        messages = [
            make_mock_consumer_record(
                topic="agent-actions",
                partition=0,
                offset=100,
                value=payload1,
            ),
            make_mock_consumer_record(
                topic="agent-actions",
                partition=1,
                offset=200,
                value=payload2,
            ),
        ]

        result = await consumer._process_batch(messages, uuid4())

        from aiokafka import TopicPartition

        tp0 = TopicPartition("agent-actions", 0)
        tp1 = TopicPartition("agent-actions", 1)

        assert tp0 in result
        assert tp1 in result
        assert result[tp0] == 100
        assert result[tp1] == 200

    @pytest.mark.asyncio
    async def test_highest_offset_per_partition_tracked(
        self,
        consumer: AgentActionsConsumer,
        sample_agent_action_payload: dict[str, object],
    ) -> None:
        """Highest offset should be tracked for each partition."""
        mock_writer = AsyncMock()
        mock_writer.write_agent_actions = AsyncMock(return_value=3)
        consumer._writer = mock_writer
        consumer._running = True

        # Multiple messages from same partition with different offsets
        messages = []
        for offset in [100, 150, 125]:  # Out of order
            payload = sample_agent_action_payload.copy()
            payload["id"] = str(uuid4())
            messages.append(
                make_mock_consumer_record(
                    topic="agent-actions",
                    partition=0,
                    offset=offset,
                    value=payload,
                )
            )

        result = await consumer._process_batch(messages, uuid4())

        from aiokafka import TopicPartition

        tp = TopicPartition("agent-actions", 0)
        # Should track highest offset (150)
        assert result[tp] == 150


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_not_running(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check should return UNHEALTHY when consumer not running."""
        consumer._running = False

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.UNHEALTHY.value
        assert health["consumer_running"] is False

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_running(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check should return HEALTHY when running with closed circuit."""
        consumer._running = True

        # Mock writer with closed circuit
        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.HEALTHY.value
        assert health["consumer_running"] is True

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_circuit_open(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check should return DEGRADED when circuit breaker is open."""
        consumer._running = True

        # Mock writer with open circuit
        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "open", "failure_count": 5}
        )
        consumer._writer = mock_writer

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.DEGRADED.value

    @pytest.mark.asyncio
    async def test_health_check_returns_expected_fields(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check should return all expected fields."""
        consumer._running = True

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed"}
        )
        consumer._writer = mock_writer

        health = await consumer.health_check()

        assert "status" in health
        assert "consumer_running" in health
        assert "consumer_id" in health
        assert "group_id" in health
        assert "topics" in health
        assert "circuit_breaker_state" in health
        assert "metrics" in health

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_circuit_half_open(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check should return DEGRADED when circuit breaker is half-open."""
        consumer._running = True

        # Mock writer with half-open circuit
        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "half_open", "failure_count": 3}
        )
        consumer._writer = mock_writer

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.DEGRADED.value

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_poll_stale(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check should return DEGRADED when last poll exceeds threshold."""
        from datetime import timedelta

        consumer._running = True

        # Mock writer with closed circuit
        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Set last poll to be stale (older than poll_staleness_seconds threshold)
        stale_time = datetime.now(UTC) - timedelta(
            seconds=consumer._config.health_check_poll_staleness_seconds + 10
        )
        consumer.metrics.last_poll_at = stale_time

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.DEGRADED.value

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_poll_recent(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check should return HEALTHY when last poll is recent."""
        from datetime import timedelta

        consumer._running = True

        # Mock writer with closed circuit
        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Set last poll to be recent (within poll_staleness_seconds threshold)
        recent_time = datetime.now(UTC) - timedelta(seconds=5)
        consumer.metrics.last_poll_at = recent_time
        consumer.metrics.last_successful_write_at = recent_time

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.HEALTHY.value

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_write_stale_with_traffic(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check should return DEGRADED when write is stale AND messages received."""
        from datetime import timedelta

        consumer._running = True

        # Mock writer with closed circuit
        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Set last poll to be recent
        recent_time = datetime.now(UTC) - timedelta(seconds=5)
        consumer.metrics.last_poll_at = recent_time

        # Set last write to be stale (older than staleness_seconds threshold)
        stale_time = datetime.now(UTC) - timedelta(
            seconds=consumer._config.health_check_staleness_seconds + 10
        )
        consumer.metrics.last_successful_write_at = stale_time

        # Set messages received > 0 (traffic has been received)
        consumer.metrics.messages_received = 100

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.DEGRADED.value

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_write_stale_but_no_traffic(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check should return HEALTHY when write stale but no messages received."""
        from datetime import timedelta

        consumer._running = True

        # Mock writer with closed circuit
        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Set last poll to be recent
        recent_time = datetime.now(UTC) - timedelta(seconds=5)
        consumer.metrics.last_poll_at = recent_time

        # Set last write to be stale (older than staleness_seconds threshold)
        stale_time = datetime.now(UTC) - timedelta(
            seconds=consumer._config.health_check_staleness_seconds + 10
        )
        consumer.metrics.last_successful_write_at = stale_time

        # No messages received (no traffic)
        consumer.metrics.messages_received = 0

        health = await consumer.health_check()

        # Should be HEALTHY because no traffic means stale write is expected
        assert health["status"] == EnumHealthStatus.HEALTHY.value


# =============================================================================
# Consumer Lifecycle Tests
# =============================================================================


class TestConsumerLifecycle:
    """Test consumer lifecycle methods."""

    @pytest.mark.asyncio
    async def test_consumer_context_manager(
        self,
        mock_config: ConfigAgentActionsConsumer,
    ) -> None:
        """Consumer should work as async context manager."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_pool:
            with patch(
                "omnibase_infra.services.observability.agent_actions.consumer.AIOKafkaConsumer"
            ) as mock_kafka:
                mock_pool.return_value = AsyncMock()
                mock_pool.return_value.close = AsyncMock()

                mock_kafka_instance = AsyncMock()
                mock_kafka.return_value = mock_kafka_instance

                consumer = AgentActionsConsumer(mock_config)

                # Patch health server to avoid binding
                object.__setattr__(consumer, "_start_health_server", AsyncMock())

                async with consumer as ctx:
                    assert ctx is consumer
                    assert ctx.is_running is True

                # After exit, should be stopped
                assert consumer.is_running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_safe(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Calling stop() when not running should be safe (no-op)."""
        consumer._running = False

        # Should not raise
        await consumer.stop()

        assert consumer.is_running is False

    @pytest.mark.asyncio
    async def test_start_when_already_running_logs_warning(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Calling start() when already running should log warning and return."""
        consumer._running = True

        # Should not raise, should return early
        await consumer.start()

        # Still running
        assert consumer.is_running is True


# =============================================================================
# Commit Offsets Tests
# =============================================================================


class TestCommitOffsets:
    """Test offset commit logic."""

    @pytest.mark.asyncio
    async def test_commit_offsets_increments_offset(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Commit should use offset + 1 (next offset to consume)."""
        mock_kafka = AsyncMock()
        consumer._consumer = mock_kafka

        from aiokafka import TopicPartition

        offsets = {
            TopicPartition("agent-actions", 0): 100,
            TopicPartition("agent-actions", 1): 200,
        }

        await consumer._commit_offsets(offsets, uuid4())

        mock_kafka.commit.assert_called_once()
        call_args = mock_kafka.commit.call_args[0][0]

        # Should commit offset + 1
        assert call_args[TopicPartition("agent-actions", 0)] == 101
        assert call_args[TopicPartition("agent-actions", 1)] == 201

    @pytest.mark.asyncio
    async def test_commit_offsets_empty_dict_skips_commit(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Empty offsets dict should skip commit call."""
        mock_kafka = AsyncMock()
        consumer._consumer = mock_kafka

        await consumer._commit_offsets({}, uuid4())

        mock_kafka.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_offsets_handles_kafka_error(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Kafka commit errors should be logged but not raised."""
        from aiokafka.errors import KafkaError

        mock_kafka = AsyncMock()
        mock_kafka.commit = AsyncMock(side_effect=KafkaError())
        consumer._consumer = mock_kafka

        from aiokafka import TopicPartition

        offsets = {TopicPartition("agent-actions", 0): 100}

        # Should not raise
        await consumer._commit_offsets(offsets, uuid4())


# =============================================================================
# Run Method Tests
# =============================================================================


class TestRunMethod:
    """Test run() method behavior."""

    @pytest.mark.asyncio
    async def test_run_raises_when_not_started(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """run() should raise OnexError if not started."""
        with pytest.raises(OnexError, match="Consumer not started"):
            await consumer.run()


__all__ = [
    "TestMaskDsnPassword",
    "TestTopicModelMapping",
    "TestConsumerMetrics",
    "TestEnumHealthStatus",
    "TestConsumerInitialization",
    "TestBatchProcessing",
    "TestOffsetTracking",
    "TestHealthCheck",
    "TestConsumerLifecycle",
    "TestCommitOffsets",
    "TestRunMethod",
]
