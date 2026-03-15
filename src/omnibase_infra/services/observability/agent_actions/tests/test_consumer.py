# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
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

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiohttp.test_utils import TestClient

from omnibase_core.errors import OnexError
from omnibase_core.types import JsonType
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
        """All configured topics should have corresponding models.

        OMN-2621: 5 legacy bare topic names replaced with ONEX canonical names.
        """
        expected_topics = {
            "onex.evt.omniclaude.agent-actions.v1",
            "onex.evt.omniclaude.routing-decision.v1",
            "onex.evt.omniclaude.agent-transformation.v1",
            "onex.evt.omniclaude.performance-metrics.v1",
            "onex.evt.omniclaude.detection-failure.v1",
            "onex.evt.omniclaude.agent-execution-logs.v1",  # OMN-2902: renamed from agent-execution-logs
            "onex.evt.omniclaude.agent-status.v1",
        }
        assert set(TOPIC_TO_MODEL.keys()) == expected_topics

    def test_all_topics_have_writer_methods(self) -> None:
        """All configured topics should have corresponding writer methods.

        OMN-2621: 5 legacy bare topic names replaced with ONEX canonical names.
        OMN-2902: agent-execution-logs → onex.evt.omniclaude.agent-execution-logs.v1.
        """
        expected_topics = {
            "onex.evt.omniclaude.agent-actions.v1",
            "onex.evt.omniclaude.routing-decision.v1",
            "onex.evt.omniclaude.agent-transformation.v1",
            "onex.evt.omniclaude.performance-metrics.v1",
            "onex.evt.omniclaude.detection-failure.v1",
            "onex.evt.omniclaude.agent-execution-logs.v1",  # OMN-2902: renamed from agent-execution-logs
            "onex.evt.omniclaude.agent-status.v1",
        }
        assert set(TOPIC_TO_WRITER_METHOD.keys()) == expected_topics

    def test_topic_to_model_mapping_correct(self) -> None:
        """Topic to model mapping should be correct.

        OMN-2621: Asserts ONEX canonical topic names, not legacy bare names.
        OMN-3422: routing-decision.v1 now uses ModelRoutingDecisionIngest (permissive
            ingest model at Kafka boundary) instead of ModelRoutingDecision (strict).
        """
        from omnibase_infra.services.observability.agent_actions.models import (
            ModelAgentAction,
            ModelAgentStatusEvent,
            ModelDetectionFailure,
            ModelExecutionLog,
            ModelPerformanceMetric,
            ModelTransformationEvent,
        )
        from omnibase_infra.services.observability.agent_actions.models.model_routing_decision_ingest import (
            ModelRoutingDecisionIngest,
        )

        assert (
            TOPIC_TO_MODEL["onex.evt.omniclaude.agent-actions.v1"] is ModelAgentAction
        )
        # OMN-3422: uses ingest model at Kafka boundary to tolerate producer field names
        assert (
            TOPIC_TO_MODEL["onex.evt.omniclaude.routing-decision.v1"]
            is ModelRoutingDecisionIngest
        )
        assert (
            TOPIC_TO_MODEL["onex.evt.omniclaude.agent-transformation.v1"]
            is ModelTransformationEvent
        )
        assert (
            TOPIC_TO_MODEL["onex.evt.omniclaude.performance-metrics.v1"]
            is ModelPerformanceMetric
        )
        assert (
            TOPIC_TO_MODEL["onex.evt.omniclaude.detection-failure.v1"]
            is ModelDetectionFailure
        )
        assert (
            TOPIC_TO_MODEL["onex.evt.omniclaude.agent-execution-logs.v1"]
            is ModelExecutionLog
        )  # OMN-2902
        assert (
            TOPIC_TO_MODEL["onex.evt.omniclaude.agent-status.v1"]
            is ModelAgentStatusEvent
        )

    def test_all_subscribed_topics_have_model_and_writer_mapping(self) -> None:
        """Every topic in config default list must appear in both dispatch dicts.

        Regression test for OMN-2621 (CONTRACT_DRIFT: orphaned subscriptions).
        Prevents topic name mismatch from returning silently — if a topic is
        subscribed but has no model/writer mapping, events are silently dropped.
        """
        config = ConfigAgentActionsConsumer(
            kafka_bootstrap_servers="localhost:9092",
            postgres_dsn="postgresql://test:test@localhost:5432/test",
        )
        for topic in config.topics:
            assert topic in TOPIC_TO_MODEL, (
                f"Topic '{topic}' is subscribed but has no model mapping. "
                "Either add it to TOPIC_TO_MODEL or remove it from config."
            )
            assert topic in TOPIC_TO_WRITER_METHOD, (
                f"Topic '{topic}' is subscribed but has no writer mapping. "
                "Either add it to TOPIC_TO_WRITER_METHOD or remove it from config."
            )


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
            topic="onex.evt.omniclaude.agent-actions.v1",
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

        tp = TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0)
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
            topic="onex.evt.omniclaude.routing-decision.v1",
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
        message.topic = "onex.evt.omniclaude.agent-actions.v1"
        message.partition = 0
        message.offset = 100
        message.value = b"not valid json"

        result = await consumer._process_batch([message], uuid4())

        # Should still track offset (skip and continue)
        from aiokafka import TopicPartition

        tp = TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0)
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
            topic="onex.evt.omniclaude.agent-actions.v1",
            partition=0,
            offset=100,
            value={"invalid": "payload"},  # Missing required fields
        )

        result = await consumer._process_batch([message], uuid4())

        # Should track offset for skipped message
        from aiokafka import TopicPartition

        tp = TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0)
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
        message.topic = "onex.evt.omniclaude.agent-actions.v1"
        message.partition = 0
        message.offset = 100
        message.value = None  # Tombstone

        result = await consumer._process_batch([message], uuid4())

        from aiokafka import TopicPartition

        tp = TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0)
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
        message.topic = "onex.evt.omniclaude.agent-actions.v1"
        message.partition = 0
        message.offset = 100
        message.value = b"\xff\xfe invalid utf-8"  # Invalid UTF-8 sequence

        result = await consumer._process_batch([message], uuid4())

        from aiokafka import TopicPartition

        tp = TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0)
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
        skipped_message.topic = "onex.evt.omniclaude.agent-actions.v1"
        skipped_message.partition = 0
        skipped_message.offset = 100
        skipped_message.value = b"not valid json"

        # Valid message at offset 101 (will fail write)
        valid_message = make_mock_consumer_record(
            topic="onex.evt.omniclaude.agent-actions.v1",
            partition=0,
            offset=101,
            value=sample_agent_action_payload,
        )

        result = await consumer._process_batch(
            [skipped_message, valid_message], uuid4()
        )

        from aiokafka import TopicPartition

        tp = TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0)
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
            topic="onex.evt.omniclaude.agent-actions.v1",
            partition=2,
            offset=500,
            value=sample_agent_action_payload,
        )

        result = await consumer._process_batch([message], uuid4())

        from aiokafka import TopicPartition

        tp = TopicPartition("onex.evt.omniclaude.agent-actions.v1", 2)
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
            topic="onex.evt.omniclaude.agent-actions.v1",
            partition=0,
            offset=100,
            value=sample_agent_action_payload,
        )

        result = await consumer._process_batch([message], uuid4())

        from aiokafka import TopicPartition

        tp = TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0)
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
                topic="onex.evt.omniclaude.agent-actions.v1",
                partition=0,
                offset=100,
                value=payload1,
            ),
            make_mock_consumer_record(
                topic="onex.evt.omniclaude.agent-actions.v1",
                partition=1,
                offset=200,
                value=payload2,
            ),
        ]

        result = await consumer._process_batch(messages, uuid4())

        from aiokafka import TopicPartition

        tp0 = TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0)
        tp1 = TopicPartition("onex.evt.omniclaude.agent-actions.v1", 1)

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
                    topic="onex.evt.omniclaude.agent-actions.v1",
                    partition=0,
                    offset=offset,
                    value=payload,
                )
            )

        result = await consumer._process_batch(messages, uuid4())

        from aiokafka import TopicPartition

        tp = TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0)
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

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_no_writes_and_no_messages_beyond_grace(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check should return HEALTHY when no writes and no messages, even past grace period.

        This is the core bug fix (OMN-2781): an idle consumer on an empty topic must
        not be treated as DEGRADED simply because it has been running for more than
        60 seconds without writing.  Zero messages received means the topic is empty —
        that is a normal operating state, not a failure.
        """
        from datetime import timedelta

        consumer._running = True

        # Mock writer with closed circuit
        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Simulate consumer started well beyond the 60-second grace period
        consumer.metrics.started_at = datetime.now(UTC) - timedelta(seconds=300)

        # No messages received, no writes
        consumer.metrics.messages_received = 0
        consumer.metrics.last_successful_write_at = None

        health = await consumer.health_check()

        # Must be HEALTHY — an idle consumer is not degraded
        assert health["status"] == EnumHealthStatus.HEALTHY.value

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_messages_received_but_no_writes_beyond_grace(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check returns DEGRADED when messages received but nothing written past grace period.

        If the consumer has consumed messages (messages_received > 0) but
        last_successful_write_at is still None after the 60-second grace period,
        the write pipeline is failing and DEGRADED is the correct status.
        """
        from datetime import timedelta

        consumer._running = True

        # Mock writer with closed circuit (circuit itself is fine)
        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Consumer started well beyond the 60-second grace period
        consumer.metrics.started_at = datetime.now(UTC) - timedelta(seconds=300)

        # Messages were received but nothing written
        consumer.metrics.messages_received = 50
        consumer.metrics.last_successful_write_at = None

        # Ensure poll is recent so poll-staleness check does not fire first
        consumer.metrics.last_poll_at = datetime.now(UTC) - timedelta(seconds=1)

        health = await consumer.health_check()

        # Write pipeline is failing — DEGRADED is correct
        assert health["status"] == EnumHealthStatus.DEGRADED.value

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_messages_received_but_no_writes_within_grace(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check returns HEALTHY when messages received but still within grace period."""
        from datetime import timedelta

        consumer._running = True

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Consumer started only 10 seconds ago — within 60-second grace period
        consumer.metrics.started_at = datetime.now(UTC) - timedelta(seconds=10)

        # Messages were received but nothing written yet (normal during startup)
        consumer.metrics.messages_received = 5
        consumer.metrics.last_successful_write_at = None

        consumer.metrics.last_poll_at = datetime.now(UTC) - timedelta(seconds=1)

        health = await consumer.health_check()

        # Within grace period — HEALTHY
        assert health["status"] == EnumHealthStatus.HEALTHY.value

    # OMN-3426: DLQ-aware health check redesign tests

    @pytest.mark.asyncio
    async def test_health_check_degraded_reason_circuit_open(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check returns degraded_reason='circuit_open' for open circuit."""
        consumer._running = True

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "open", "failure_count": 5}
        )
        consumer._writer = mock_writer

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.DEGRADED.value
        assert health["degraded_reason"] == "circuit_open"

    @pytest.mark.asyncio
    async def test_health_check_degraded_reason_poll_stale(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check returns degraded_reason='poll_stale' when poll is stale."""
        from datetime import timedelta

        consumer._running = True

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        stale_time = datetime.now(UTC) - timedelta(
            seconds=consumer._config.health_check_poll_staleness_seconds + 10
        )
        consumer.metrics.last_poll_at = stale_time

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.DEGRADED.value
        assert health["degraded_reason"] == "poll_stale"

    @pytest.mark.asyncio
    async def test_health_check_degraded_reason_dlq_rate_exceeded(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check returns DEGRADED with reason 'dlq_rate_exceeded' when DLQ ratio is high.

        OMN-3426: When the consumer is receiving events but most are going to the DLQ
        (validation failures), this is DEGRADED but NOT a write-pipeline failure.
        The reason field distinguishes the cause.
        """
        from datetime import timedelta

        consumer._running = True

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Recent poll
        consumer.metrics.last_poll_at = datetime.now(UTC) - timedelta(seconds=5)

        # 20 messages received, 18 went to DLQ → ratio = 0.9 > default threshold 0.5
        consumer.metrics.messages_received = 20
        consumer.metrics.messages_sent_to_dlq = 18

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.DEGRADED.value
        assert health["degraded_reason"] == "dlq_rate_exceeded"
        assert health["dlq_ratio"] is not None
        assert health["dlq_ratio"] > 0.5  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_all_messages_dlq_but_below_threshold(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check returns HEALTHY when DLQ ratio is below threshold.

        OMN-3426: A DLQ ratio below the configured threshold does not trigger DEGRADED.
        Even if writes are stale, if DLQ explains the gap the write pipeline is not broken.
        """
        from datetime import timedelta

        consumer._running = True

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Recent poll
        consumer.metrics.last_poll_at = datetime.now(UTC) - timedelta(seconds=5)

        # Only 4 messages received — below dlq_min_messages (10), so DLQ ratio not evaluated
        consumer.metrics.messages_received = 4
        consumer.metrics.messages_sent_to_dlq = 4
        consumer.metrics.last_successful_write_at = None
        consumer.metrics.started_at = datetime.now(UTC) - timedelta(seconds=300)

        health = await consumer.health_check()

        # dlq_ratio is None because below min_messages threshold
        assert health["dlq_ratio"] is None
        # Consumer is HEALTHY: no write gap (no writes yet), below DLQ threshold
        assert health["status"] == EnumHealthStatus.HEALTHY.value

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_writes_stale_but_dlq_explains_gap(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check returns HEALTHY when write staleness is fully explained by DLQ.

        OMN-3426 core case: consumer receives events, all go to DLQ (schema failures),
        last_write becomes stale. OLD Rule 5 would fire DEGRADED here. The new rule
        should return HEALTHY because the DLQ explains the write gap — the write
        pipeline itself is not broken. (DLQ rate gate in Rule 4 will catch the high
        DLQ rate separately if the threshold is breached.)
        """
        from datetime import timedelta

        consumer._running = True

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Recent poll
        consumer.metrics.last_poll_at = datetime.now(UTC) - timedelta(seconds=5)

        # 8 messages received (below min_messages=10), all went to DLQ → ratio not evaluated
        # Set last write stale to test DLQ-explained staleness path
        consumer.metrics.messages_received = 8
        consumer.metrics.messages_sent_to_dlq = 8
        stale_write = datetime.now(UTC) - timedelta(
            seconds=consumer._config.health_check_staleness_seconds + 60
        )
        consumer.metrics.last_successful_write_at = stale_write

        health = await consumer.health_check()

        # Write is stale but all messages went to DLQ — write pipeline not broken
        # DLQ ratio not evaluated (below min_messages), so HEALTHY via dlq_explained path
        assert health["status"] == EnumHealthStatus.HEALTHY.value

    @pytest.mark.asyncio
    async def test_health_check_degraded_write_pipeline_stale_not_dlq_explained(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check returns DEGRADED when write is stale and DLQ does NOT explain gap.

        OMN-3426: If messages are received but only a fraction goes to DLQ while writes
        are stale, the write pipeline itself is failing. DEGRADED with reason
        'write_pipeline_stale' is correct.
        """
        from datetime import timedelta

        consumer._running = True

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Recent poll
        consumer.metrics.last_poll_at = datetime.now(UTC) - timedelta(seconds=5)

        # 100 messages received, only 5 went to DLQ → DLQ does not explain the gap
        consumer.metrics.messages_received = 100
        consumer.metrics.messages_sent_to_dlq = 5  # Only 5% DLQ rate — below threshold

        stale_write = datetime.now(UTC) - timedelta(
            seconds=consumer._config.health_check_staleness_seconds + 60
        )
        consumer.metrics.last_successful_write_at = stale_write

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.DEGRADED.value
        assert health["degraded_reason"] == "write_pipeline_stale"

    @pytest.mark.asyncio
    async def test_health_check_degraded_write_pipeline_failing_no_writes_no_dlq(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check returns DEGRADED when messages received, no writes, no DLQ to explain.

        OMN-3426: If messages are received but nothing was ever written AND the messages
        did not go to DLQ, the write pipeline is failing from the start.
        """
        from datetime import timedelta

        consumer._running = True

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        # Consumer started 5 minutes ago — past grace period
        consumer.metrics.started_at = datetime.now(UTC) - timedelta(seconds=300)

        # 50 messages received, 0 to DLQ, nothing written
        consumer.metrics.messages_received = 50
        consumer.metrics.messages_sent_to_dlq = 0
        consumer.metrics.last_successful_write_at = None

        consumer.metrics.last_poll_at = datetime.now(UTC) - timedelta(seconds=1)

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.DEGRADED.value
        assert health["degraded_reason"] == "write_pipeline_failing"

    @pytest.mark.asyncio
    async def test_health_check_healthy_returns_none_degraded_reason(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Healthy consumer returns degraded_reason=None."""
        from datetime import timedelta

        consumer._running = True

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed", "failure_count": 0}
        )
        consumer._writer = mock_writer

        consumer.metrics.last_poll_at = datetime.now(UTC) - timedelta(seconds=5)
        consumer.metrics.last_successful_write_at = datetime.now(UTC) - timedelta(
            seconds=10
        )
        consumer.metrics.messages_received = 5

        health = await consumer.health_check()

        assert health["status"] == EnumHealthStatus.HEALTHY.value
        assert health["degraded_reason"] is None

    @pytest.mark.asyncio
    async def test_health_check_returns_dlq_ratio_field(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """Health check response always includes dlq_ratio field (OMN-3426)."""
        consumer._running = True

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed"}
        )
        consumer._writer = mock_writer

        health = await consumer.health_check()

        assert "dlq_ratio" in health
        assert "degraded_reason" in health

    @pytest.mark.asyncio
    async def test_health_check_http_unhealthy_returns_503(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """HTTP health endpoint returns 503 only for UNHEALTHY (OMN-3426).

        DEGRADED returns 200. Only UNHEALTHY (consumer not running) returns 503.
        """
        from aiohttp.test_utils import make_mocked_request

        # Consumer not running → UNHEALTHY
        consumer._running = False

        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "closed"}
        )
        consumer._writer = mock_writer

        request = make_mocked_request("GET", "/health")
        response = await consumer._health_handler(request)

        assert response.status == 503

    @pytest.mark.asyncio
    async def test_health_check_http_degraded_returns_200(
        self,
        consumer: AgentActionsConsumer,
    ) -> None:
        """HTTP health endpoint returns 200 for DEGRADED status (OMN-3426).

        DEGRADED means consumer is alive but impaired. It should NOT return 503,
        which would cause K8s to restart a running pod unnecessarily.
        """
        from aiohttp.test_utils import make_mocked_request

        consumer._running = True

        # Open circuit → DEGRADED
        mock_writer = MagicMock()
        mock_writer.get_circuit_breaker_state = MagicMock(
            return_value={"state": "open", "failure_count": 5}
        )
        consumer._writer = mock_writer

        request = make_mocked_request("GET", "/health")
        response = await consumer._health_handler(request)

        assert response.status == 200


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
                with patch(
                    "omnibase_infra.services.observability.agent_actions.consumer.AIOKafkaProducer"
                ) as mock_producer:
                    mock_pool.return_value = AsyncMock()
                    mock_pool.return_value.close = AsyncMock()

                    mock_kafka_instance = AsyncMock()
                    mock_kafka.return_value = mock_kafka_instance

                    mock_producer_instance = AsyncMock()
                    mock_producer.return_value = mock_producer_instance

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
            TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0): 100,
            TopicPartition("onex.evt.omniclaude.agent-actions.v1", 1): 200,
        }

        await consumer._commit_offsets(offsets, uuid4())

        mock_kafka.commit.assert_called_once()
        call_args = mock_kafka.commit.call_args[0][0]

        # Should commit offset + 1
        assert (
            call_args[TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0)] == 101
        )
        assert (
            call_args[TopicPartition("onex.evt.omniclaude.agent-actions.v1", 1)] == 201
        )

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

        offsets = {TopicPartition("onex.evt.omniclaude.agent-actions.v1", 0): 100}

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


# =============================================================================
# Kafka Reconnect Health Regression Tests (OMN-3430)
# =============================================================================


class TestKafkaReconnectHealth:
    """Regression tests for OMN-3430.

    When aiokafka is in its internal reconnect/retry loop, every getmany()
    call blocks until asyncio.wait_for fires a TimeoutError.  Before the fix,
    the TimeoutError handler called ``continue`` without updating
    ``last_poll_at``, causing the poll-staleness check to flip health to
    DEGRADED after 60 s even though the consumer loop was still running.
    """

    @pytest.mark.asyncio
    async def test_record_polled_called_on_timeout(
        self, consumer: AgentActionsConsumer
    ) -> None:
        """TimeoutError from getmany must still update last_poll_at (OMN-3430).

        Simulates the Kafka-coordinator-dead scenario where every getmany()
        call times out.  Verifies that after a single poll-loop iteration that
        hits TimeoutError, last_poll_at is not None (i.e. record_polled was
        called inside the TimeoutError handler).

        Implementation: patch asyncio.wait_for so the first call raises
        TimeoutError (and sets _running=False to stop the loop), then confirm
        the metrics were updated.
        """
        # Consumer must be in running state for the loop to execute.
        consumer._running = True

        # last_poll_at should be None before any iteration.
        assert consumer.metrics.last_poll_at is None

        call_count = 0

        async def _wait_for_side_effect(coro: object, timeout: float) -> object:
            nonlocal call_count
            call_count += 1
            # Stop the loop so _consume_loop exits after this iteration.
            consumer._running = False
            raise TimeoutError

        mock_kafka = AsyncMock()
        consumer._consumer = mock_kafka

        with patch(
            "omnibase_infra.services.observability.agent_actions.consumer.asyncio.wait_for",
            side_effect=_wait_for_side_effect,
        ):
            await consumer._consume_loop(uuid4())

        # After the TimeoutError iteration, last_poll_at must have been updated.
        assert consumer.metrics.last_poll_at is not None, (
            "record_polled() was not called during TimeoutError — "
            "health check will report DEGRADED during Kafka reconnect (OMN-3430)"
        )

    @pytest.mark.asyncio
    async def test_poll_staleness_not_triggered_during_kafka_timeout(
        self, consumer: AgentActionsConsumer
    ) -> None:
        """Health stays non-DEGRADED even after repeated poll timeouts (OMN-3430).

        Verifies end-to-end: after a sequence of getmany() TimeoutErrors,
        _determine_health_status does NOT return DEGRADED due to poll staleness
        because record_polled() keeps last_poll_at fresh.

        This tests the full chain: TimeoutError → record_polled() → fresh
        last_poll_at → health check passes staleness check → not DEGRADED.
        """
        consumer._running = True

        iteration = 0

        async def _wait_for_side_effect(coro: object, timeout: float) -> object:
            nonlocal iteration
            iteration += 1
            # Stop the loop after 3 iterations to keep the test fast.
            if iteration >= 3:
                consumer._running = False
            raise TimeoutError

        mock_kafka = AsyncMock()
        consumer._consumer = mock_kafka

        with patch(
            "omnibase_infra.services.observability.agent_actions.consumer.asyncio.wait_for",
            side_effect=_wait_for_side_effect,
        ):
            await consumer._consume_loop(uuid4())

        # Confirm last_poll_at is fresh (within poll_staleness threshold).
        assert consumer.metrics.last_poll_at is not None
        now = datetime.now(UTC)
        poll_age = (now - consumer.metrics.last_poll_at).total_seconds()
        assert poll_age < consumer._config.health_check_poll_staleness_seconds, (
            f"last_poll_at is {poll_age:.1f}s old, exceeds staleness threshold "
            f"{consumer._config.health_check_poll_staleness_seconds}s"
        )

        # Health status should NOT be DEGRADED due to poll staleness.
        metrics_snapshot = await consumer.metrics.snapshot()
        circuit_state: dict[str, JsonType] = {"state": "closed"}
        status = consumer._determine_health_status(metrics_snapshot, circuit_state)
        assert status != EnumHealthStatus.DEGRADED


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
    "TestKafkaReconnectHealth",
]
