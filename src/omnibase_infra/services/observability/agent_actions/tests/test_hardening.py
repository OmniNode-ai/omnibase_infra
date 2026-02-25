# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for Phase 2 hardening features (OMN-1768).

This module tests:
    - Payload size limits (metadata and raw_payload validators)
    - Health status edge cases (_determine_health_status boundaries)
    - ConsumerMetrics enhancements (per-topic, DLQ, latency, export hooks)
    - DLQ configuration

Related Tickets:
    - OMN-1768: Phase 2: Agent Actions Consumer Hardening
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.services.observability.agent_actions.config import (
    ConfigAgentActionsConsumer,
)
from omnibase_infra.services.observability.agent_actions.consumer import (
    AgentActionsConsumer,
    ConsumerMetrics,
    EnumHealthStatus,
)
from omnibase_infra.services.observability.agent_actions.models.model_agent_action import (
    MAX_METADATA_SIZE_BYTES,
    MAX_RAW_PAYLOAD_SIZE_BYTES,
    ModelAgentAction,
)

# =============================================================================
# Payload Size Limit Tests (Task 1)
# =============================================================================


class TestPayloadSizeLimits:
    """Test Pydantic field validators for metadata and raw_payload size limits."""

    def test_metadata_within_limit_passes(self) -> None:
        """Metadata within size limit should be accepted unchanged."""
        metadata = {"key": "value", "nested": {"a": 1}}
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            metadata=metadata,
        )
        assert action.metadata == metadata

    def test_metadata_exceeding_limit_truncated(self) -> None:
        """Metadata exceeding size limit should be truncated with marker."""
        # Create metadata larger than 64KB
        large_metadata = {"data": "x" * (MAX_METADATA_SIZE_BYTES + 1000)}
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            metadata=large_metadata,
        )
        assert action.metadata is not None
        assert action.metadata.get("_truncated") is True
        assert "_original_size_bytes" in action.metadata

    def test_metadata_none_passes(self) -> None:
        """None metadata should pass validation."""
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            metadata=None,
        )
        assert action.metadata is None

    def test_raw_payload_within_limit_passes(self) -> None:
        """Raw payload within size limit should be accepted unchanged."""
        payload = {"full": "payload", "data": [1, 2, 3]}
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            raw_payload=payload,
        )
        assert action.raw_payload == payload

    def test_raw_payload_exceeding_limit_truncated(self) -> None:
        """Raw payload exceeding 1MB limit should be truncated with marker."""
        large_payload = {"data": "x" * (MAX_RAW_PAYLOAD_SIZE_BYTES + 1000)}
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            raw_payload=large_payload,
        )
        assert action.raw_payload is not None
        assert action.raw_payload.get("_truncated") is True
        assert "_original_size_bytes" in action.raw_payload

    def test_raw_payload_none_passes(self) -> None:
        """None raw_payload should pass validation."""
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            raw_payload=None,
        )
        assert action.raw_payload is None

    def test_metadata_exactly_at_limit_passes(self) -> None:
        """Metadata exactly at the size limit boundary should pass."""
        # Create metadata that is just under the limit
        # JSON overhead for {"k": "..."} is about 7 bytes + key length
        target_size = MAX_METADATA_SIZE_BYTES - 20
        metadata = {"k": "a" * target_size}
        serialized_size = len(json.dumps(metadata).encode("utf-8"))
        # Verify we're under the limit
        assert serialized_size <= MAX_METADATA_SIZE_BYTES

        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            metadata=metadata,
        )
        assert action.metadata == metadata


# =============================================================================
# Health Status Edge Case Tests (Task 6)
# =============================================================================


def _make_health_consumer() -> AgentActionsConsumer:
    """Create a partially-initialized consumer for health status testing.

    Uses ``__new__`` to bypass ``__init__`` (avoids Kafka/DB connections)
    and sets only the attributes needed by ``_determine_health_status``.
    """
    config = ConfigAgentActionsConsumer(
        kafka_bootstrap_servers="localhost:9092",
        postgres_dsn="postgresql://test:test@localhost:5432/test",
        health_check_staleness_seconds=300,
        health_check_poll_staleness_seconds=60,
    )
    consumer = AgentActionsConsumer.__new__(AgentActionsConsumer)
    consumer._config = config
    consumer._running = True
    consumer._consumer_id = "test-consumer"
    return consumer


class TestHealthStatusStartupGrace:
    """Test startup grace period edge cases in _determine_health_status."""

    @pytest.fixture
    def mock_consumer(self) -> AgentActionsConsumer:
        return _make_health_consumer()

    def test_within_startup_grace_period_is_healthy(
        self, mock_consumer: AgentActionsConsumer
    ) -> None:
        """Consumer well within 60s grace period with messages but no writes should be HEALTHY."""
        now = datetime.now(UTC)
        started = (now - timedelta(seconds=30)).isoformat()
        metrics = {
            "messages_received": 5,
            "last_poll_at": now.isoformat(),
            "last_successful_write_at": None,
            "started_at": started,
        }
        circuit = {"state": "closed"}
        status = mock_consumer._determine_health_status(metrics, circuit)
        assert status == EnumHealthStatus.HEALTHY

    def test_past_startup_grace_period_is_degraded(
        self, mock_consumer: AgentActionsConsumer
    ) -> None:
        """Consumer past 60s grace with messages but no writes should be DEGRADED."""
        now = datetime.now(UTC)
        started = (now - timedelta(seconds=90)).isoformat()
        metrics = {
            "messages_received": 5,
            "last_poll_at": now.isoformat(),
            "last_successful_write_at": None,
            "started_at": started,
        }
        circuit = {"state": "closed"}
        status = mock_consumer._determine_health_status(metrics, circuit)
        assert status == EnumHealthStatus.DEGRADED

    def test_idle_consumer_always_healthy(
        self, mock_consumer: AgentActionsConsumer
    ) -> None:
        """Idle consumer (zero messages) should always be HEALTHY regardless of uptime."""
        now = datetime.now(UTC)
        metrics = {
            "messages_received": 0,
            "last_poll_at": now.isoformat(),
            "last_successful_write_at": None,
            "started_at": (now - timedelta(hours=24)).isoformat(),
        }
        circuit = {"state": "closed"}
        status = mock_consumer._determine_health_status(metrics, circuit)
        assert status == EnumHealthStatus.HEALTHY


class TestHealthStatusStaleness:
    """Test poll and write staleness thresholds in _determine_health_status."""

    @pytest.fixture
    def mock_consumer(self) -> AgentActionsConsumer:
        return _make_health_consumer()

    def test_poll_within_staleness_threshold_is_healthy(
        self, mock_consumer: AgentActionsConsumer
    ) -> None:
        """Poll age within threshold should be HEALTHY."""
        now = datetime.now(UTC)
        last_poll = (now - timedelta(seconds=30)).isoformat()
        metrics = {
            "messages_received": 10,
            "last_poll_at": last_poll,
            "last_successful_write_at": now.isoformat(),
            "started_at": (now - timedelta(hours=1)).isoformat(),
        }
        circuit = {"state": "closed"}
        status = mock_consumer._determine_health_status(metrics, circuit)
        assert status == EnumHealthStatus.HEALTHY

    def test_poll_exceeding_staleness_threshold_is_degraded(
        self, mock_consumer: AgentActionsConsumer
    ) -> None:
        """Poll age clearly exceeding threshold should be DEGRADED."""
        now = datetime.now(UTC)
        last_poll = (now - timedelta(seconds=120)).isoformat()
        metrics = {
            "messages_received": 10,
            "last_poll_at": last_poll,
            "last_successful_write_at": now.isoformat(),
            "started_at": (now - timedelta(hours=1)).isoformat(),
        }
        circuit = {"state": "closed"}
        status = mock_consumer._determine_health_status(metrics, circuit)
        assert status == EnumHealthStatus.DEGRADED

    def test_write_staleness_within_threshold_is_healthy(
        self, mock_consumer: AgentActionsConsumer
    ) -> None:
        """Write age within threshold (200s < 300s) should be HEALTHY."""
        now = datetime.now(UTC)
        last_write = (now - timedelta(seconds=200)).isoformat()
        metrics = {
            "messages_received": 100,
            "last_poll_at": now.isoformat(),
            "last_successful_write_at": last_write,
            "started_at": (now - timedelta(hours=1)).isoformat(),
        }
        circuit = {"state": "closed"}
        status = mock_consumer._determine_health_status(metrics, circuit)
        assert status == EnumHealthStatus.HEALTHY

    def test_write_staleness_exceeding_threshold_is_degraded(
        self, mock_consumer: AgentActionsConsumer
    ) -> None:
        """Write age clearly exceeding threshold (600s > 300s) should be DEGRADED."""
        now = datetime.now(UTC)
        last_write = (now - timedelta(seconds=600)).isoformat()
        metrics = {
            "messages_received": 100,
            "last_poll_at": now.isoformat(),
            "last_successful_write_at": last_write,
            "started_at": (now - timedelta(hours=1)).isoformat(),
        }
        circuit = {"state": "closed"}
        status = mock_consumer._determine_health_status(metrics, circuit)
        assert status == EnumHealthStatus.DEGRADED


class TestHealthStatusCircuitBreaker:
    """Test circuit breaker state effects on _determine_health_status."""

    @pytest.fixture
    def mock_consumer(self) -> AgentActionsConsumer:
        return _make_health_consumer()

    def test_not_running_is_unhealthy(
        self, mock_consumer: AgentActionsConsumer
    ) -> None:
        """Consumer not running should always be UNHEALTHY."""
        mock_consumer._running = False
        metrics = {
            "messages_received": 0,
            "last_poll_at": None,
            "last_successful_write_at": None,
            "started_at": datetime.now(UTC).isoformat(),
        }
        circuit = {"state": "closed"}
        status = mock_consumer._determine_health_status(metrics, circuit)
        assert status == EnumHealthStatus.UNHEALTHY

    def test_circuit_open_is_degraded(
        self, mock_consumer: AgentActionsConsumer
    ) -> None:
        """Open circuit breaker should be DEGRADED."""
        metrics = {
            "messages_received": 10,
            "last_poll_at": datetime.now(UTC).isoformat(),
            "last_successful_write_at": datetime.now(UTC).isoformat(),
            "started_at": datetime.now(UTC).isoformat(),
        }
        circuit = {"state": "open"}
        status = mock_consumer._determine_health_status(metrics, circuit)
        assert status == EnumHealthStatus.DEGRADED

    def test_circuit_half_open_is_degraded(
        self, mock_consumer: AgentActionsConsumer
    ) -> None:
        """Half-open circuit breaker should be DEGRADED."""
        metrics = {
            "messages_received": 10,
            "last_poll_at": datetime.now(UTC).isoformat(),
            "last_successful_write_at": datetime.now(UTC).isoformat(),
            "started_at": datetime.now(UTC).isoformat(),
        }
        circuit = {"state": "half_open"}
        status = mock_consumer._determine_health_status(metrics, circuit)
        assert status == EnumHealthStatus.DEGRADED


# =============================================================================
# ConsumerMetrics Enhancement Tests (Task 5)
# =============================================================================


class TestConsumerMetricsEnhancements:
    """Test Phase 2 enhancements to ConsumerMetrics."""

    @pytest.mark.asyncio
    async def test_per_topic_received_tracking(self) -> None:
        """Per-topic received counters should increment correctly."""
        metrics = ConsumerMetrics()
        await metrics.record_received(5, topic="onex.evt.omniclaude.agent-actions.v1")
        await metrics.record_received(
            3, topic="onex.evt.omniclaude.routing-decision.v1"
        )
        await metrics.record_received(2, topic="onex.evt.omniclaude.agent-actions.v1")

        snapshot = await metrics.snapshot()
        per_topic = snapshot["per_topic_received"]
        assert isinstance(per_topic, dict)
        assert per_topic["onex.evt.omniclaude.agent-actions.v1"] == 7
        assert per_topic["onex.evt.omniclaude.routing-decision.v1"] == 3

    @pytest.mark.asyncio
    async def test_per_topic_processed_tracking(self) -> None:
        """Per-topic processed counters should increment correctly."""
        metrics = ConsumerMetrics()
        await metrics.record_processed(10, topic="test-topic")

        snapshot = await metrics.snapshot()
        per_topic = snapshot["per_topic_processed"]
        assert isinstance(per_topic, dict)
        assert per_topic["test-topic"] == 10

    @pytest.mark.asyncio
    async def test_per_topic_failed_tracking(self) -> None:
        """Per-topic failed counters should increment correctly."""
        metrics = ConsumerMetrics()
        await metrics.record_failed(2, topic="test-topic")

        snapshot = await metrics.snapshot()
        per_topic = snapshot["per_topic_failed"]
        assert isinstance(per_topic, dict)
        assert per_topic["test-topic"] == 2

    @pytest.mark.asyncio
    async def test_dlq_counter(self) -> None:
        """DLQ message counter should increment."""
        metrics = ConsumerMetrics()
        await metrics.record_sent_to_dlq(3)

        snapshot = await metrics.snapshot()
        assert snapshot["messages_sent_to_dlq"] == 3

    @pytest.mark.asyncio
    async def test_batch_latency_tracking(self) -> None:
        """Batch latency samples should be tracked and stats computed."""
        metrics = ConsumerMetrics()
        await metrics.record_batch_processed(latency_ms=10.0)
        await metrics.record_batch_processed(latency_ms=20.0)
        await metrics.record_batch_processed(latency_ms=30.0)

        snapshot = await metrics.snapshot()
        latency = snapshot["batch_latency"]
        assert latency is not None
        assert isinstance(latency, dict)
        assert latency["count"] == 3
        assert latency["min_ms"] == 10.0
        assert latency["max_ms"] == 30.0
        assert latency["avg_ms"] == 20.0

    @pytest.mark.asyncio
    async def test_batch_latency_ring_buffer_cap(self) -> None:
        """Batch latency ring buffer should cap at MAX_LATENCY_SAMPLES."""
        metrics = ConsumerMetrics()
        for i in range(150):
            await metrics.record_batch_processed(latency_ms=float(i))

        snapshot = await metrics.snapshot()
        latency = snapshot["batch_latency"]
        assert latency is not None
        assert isinstance(latency, dict)
        assert latency["count"] == ConsumerMetrics.MAX_LATENCY_SAMPLES

    @pytest.mark.asyncio
    async def test_export_hook_called(self) -> None:
        """Export hooks should be called on metric updates."""
        metrics = ConsumerMetrics()
        hook_calls: list[tuple[str, float, dict[str, str]]] = []

        def test_hook(name: str, value: float, labels: dict[str, str]) -> None:
            hook_calls.append((name, value, labels))

        metrics.register_export_hook(test_hook)
        await metrics.record_received(5, topic="test-topic")

        assert len(hook_calls) == 1
        assert hook_calls[0][0] == "consumer_messages_received_total"
        assert hook_calls[0][1] == 5.0
        assert hook_calls[0][2] == {"topic": "test-topic"}

    @pytest.mark.asyncio
    async def test_failing_export_hook_does_not_crash(self) -> None:
        """A failing export hook should not crash the metrics system."""
        metrics = ConsumerMetrics()

        def failing_hook(name: str, value: float, labels: dict[str, str]) -> None:
            raise RuntimeError("Hook failed")

        metrics.register_export_hook(failing_hook)
        # Should not raise
        await metrics.record_received(1, topic="test-topic")
        assert metrics.messages_received == 1

    @pytest.mark.asyncio
    async def test_snapshot_with_no_latency(self) -> None:
        """Snapshot with no batch latency should return None for latency."""
        metrics = ConsumerMetrics()
        snapshot = await metrics.snapshot()
        assert snapshot["batch_latency"] is None


# =============================================================================
# DLQ Configuration Tests (Task 2)
# =============================================================================


class TestDLQConfiguration:
    """Test DLQ configuration defaults and overrides."""

    def test_default_dlq_config(self) -> None:
        """Default DLQ configuration should have sensible defaults."""
        config = ConfigAgentActionsConsumer(
            postgres_dsn="postgresql://test:test@localhost:5432/test",
        )
        assert config.dlq_enabled is True
        assert config.dlq_topic == "onex.evt.omniclaude.agent-actions-dlq.v1"
        assert config.max_retry_count == 3

    def test_dlq_disabled(self) -> None:
        """DLQ can be disabled via config."""
        config = ConfigAgentActionsConsumer(
            postgres_dsn="postgresql://test:test@localhost:5432/test",
            dlq_enabled=False,
        )
        assert config.dlq_enabled is False


# =============================================================================
# Health Check Host Default Tests (Task 4)
# =============================================================================


class TestHealthCheckHostDefault:
    """Test health check host default changed from 0.0.0.0 to 127.0.0.1."""

    def test_default_health_check_host_is_localhost(self) -> None:
        """Default health check host should be 127.0.0.1 for security."""
        config = ConfigAgentActionsConsumer(
            postgres_dsn="postgresql://test:test@localhost:5432/test",
        )
        assert config.health_check_host == "127.0.0.1"

    def test_health_check_host_override_for_containers(self) -> None:
        """Health check host can be overridden to 0.0.0.0 for containers."""
        all_interfaces = "0.0.0.0"  # noqa: S104 - test verifying container override
        config = ConfigAgentActionsConsumer(
            postgres_dsn="postgresql://test:test@localhost:5432/test",
            health_check_host=all_interfaces,
        )
        assert config.health_check_host == all_interfaces


__all__ = [
    "TestPayloadSizeLimits",
    "TestHealthStatusStartupGrace",
    "TestHealthStatusStaleness",
    "TestHealthStatusCircuitBreaker",
    "TestConsumerMetricsEnhancements",
    "TestDLQConfiguration",
    "TestHealthCheckHostDefault",
]
