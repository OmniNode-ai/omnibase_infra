# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for SkillLifecycleConsumer and helpers (OMN-2934).

Tests:
    - mask_dsn_password: password masking for safe logging
    - ConsumerMetrics: metric tracking and snapshot
    - SkillLifecycleConsumer._parse_message: JSON parsing edge cases
    - SkillLifecycleConsumer._build_health_response: health status logic

All tests are unit-level — no real Kafka or PostgreSQL required.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from omnibase_infra.services.observability.skill_lifecycle.config import (
    ConfigSkillLifecycleConsumer,
)
from omnibase_infra.services.observability.skill_lifecycle.consumer import (
    ConsumerMetrics,
    EnumHealthStatus,
    SkillLifecycleConsumer,
    mask_dsn_password,
)

_REQUIRED_DSN = "postgresql://postgres:secret@localhost:5432/testdb"


def _make_config(**overrides: object) -> ConfigSkillLifecycleConsumer:
    defaults: dict[str, object] = {
        "kafka_bootstrap_servers": "localhost:19092",
        "postgres_dsn": _REQUIRED_DSN,
        "_env_file": None,
    }
    defaults.update(overrides)
    return ConfigSkillLifecycleConsumer(**defaults)  # type: ignore[arg-type]


def _make_consumer(**config_overrides: object) -> SkillLifecycleConsumer:
    return SkillLifecycleConsumer(_make_config(**config_overrides))


def _make_mock_record(
    topic: str = "onex.evt.omniclaude.skill-started.v1",
    value: bytes = b'{"event_id": "abc", "run_id": "xyz"}',
    partition: int = 0,
    offset: int = 0,
) -> MagicMock:
    record = MagicMock()
    record.topic = topic
    record.value = value
    record.partition = partition
    record.offset = offset
    return record


# =============================================================================
# Tests: mask_dsn_password
# =============================================================================


class TestMaskDsnPassword:
    """Test DSN password masking for safe logging."""

    @pytest.mark.unit
    def test_masks_password(self) -> None:
        """Password component is replaced with ***."""
        dsn = "postgresql://user:secret@localhost:5432/db"
        result = mask_dsn_password(dsn)

        assert "secret" not in result
        assert "***" in result
        assert "user" in result
        assert "localhost" in result
        assert "5432" in result

    @pytest.mark.unit
    def test_no_password_unchanged(self) -> None:
        """DSN without password is returned unchanged."""
        dsn = "postgresql://user@localhost/db"
        result = mask_dsn_password(dsn)

        assert result == dsn

    @pytest.mark.unit
    def test_invalid_dsn_returned_as_is(self) -> None:
        """Unparseable DSN is returned as-is."""
        dsn = "not-a-valid-dsn"
        result = mask_dsn_password(dsn)

        assert result == dsn

    @pytest.mark.unit
    def test_dsn_without_port(self) -> None:
        """DSN without port is handled."""
        dsn = "postgresql://user:pass@host/db"
        result = mask_dsn_password(dsn)

        assert "pass" not in result
        assert "***" in result


# =============================================================================
# Tests: ConsumerMetrics
# =============================================================================


class TestConsumerMetrics:
    """Test ConsumerMetrics tracking and snapshot."""

    @pytest.mark.unit
    def test_initial_values(self) -> None:
        """All counters start at zero."""
        metrics = ConsumerMetrics()

        assert metrics.messages_received == 0
        assert metrics.messages_processed == 0
        assert metrics.messages_failed == 0
        assert metrics.messages_skipped == 0
        assert metrics.messages_sent_to_dlq == 0
        assert metrics.batches_processed == 0
        assert metrics.last_poll_at is None
        assert metrics.last_successful_write_at is None

    @pytest.mark.unit
    def test_record_received_increments(self) -> None:
        """record_received increments messages_received and sets last_poll_at."""
        metrics = ConsumerMetrics()
        asyncio.get_event_loop().run_until_complete(
            metrics.record_received(count=3, topic="topic-a")
        )

        assert metrics.messages_received == 3
        assert metrics.last_poll_at is not None
        assert metrics.per_topic_received.get("topic-a") == 3

    @pytest.mark.unit
    def test_record_processed_increments(self) -> None:
        """record_processed increments messages_processed and sets last_successful_write_at."""
        metrics = ConsumerMetrics()
        asyncio.get_event_loop().run_until_complete(
            metrics.record_processed(count=2, topic="topic-b")
        )

        assert metrics.messages_processed == 2
        assert metrics.last_successful_write_at is not None
        assert metrics.per_topic_processed.get("topic-b") == 2

    @pytest.mark.unit
    def test_record_failed_increments(self) -> None:
        """record_failed increments messages_failed."""
        metrics = ConsumerMetrics()
        asyncio.get_event_loop().run_until_complete(
            metrics.record_failed(count=1, topic="topic-c")
        )

        assert metrics.messages_failed == 1
        assert metrics.per_topic_failed.get("topic-c") == 1

    @pytest.mark.unit
    def test_record_batch_processed_increments(self) -> None:
        """record_batch_processed increments batches_processed."""
        metrics = ConsumerMetrics()
        asyncio.get_event_loop().run_until_complete(
            metrics.record_batch_processed(latency_ms=42.0)
        )

        assert metrics.batches_processed == 1
        assert metrics.batch_latency_ms == [42.0]

    @pytest.mark.unit
    def test_snapshot_contains_expected_keys(self) -> None:
        """Snapshot dict contains all expected metric keys."""
        metrics = ConsumerMetrics()
        snapshot = asyncio.get_event_loop().run_until_complete(metrics.snapshot())

        expected_keys = {
            "messages_received",
            "messages_processed",
            "messages_failed",
            "messages_skipped",
            "messages_sent_to_dlq",
            "batches_processed",
            "last_poll_at",
            "last_successful_write_at",
            "started_at",
            "uptime_seconds",
            "per_topic_received",
            "per_topic_processed",
            "per_topic_failed",
            "batch_latency_stats",
        }
        assert expected_keys.issubset(snapshot.keys())

    @pytest.mark.unit
    def test_latency_ring_buffer(self) -> None:
        """Latency ring buffer is capped at MAX_LATENCY_SAMPLES."""
        metrics = ConsumerMetrics()

        loop = asyncio.get_event_loop()
        for i in range(ConsumerMetrics.MAX_LATENCY_SAMPLES + 10):
            loop.run_until_complete(metrics.record_batch_processed(latency_ms=float(i)))

        assert len(metrics.batch_latency_ms) == ConsumerMetrics.MAX_LATENCY_SAMPLES


# =============================================================================
# Tests: SkillLifecycleConsumer._parse_message
# =============================================================================


class TestParseMessage:
    """Test _parse_message for JSON edge cases."""

    @pytest.mark.unit
    def test_valid_json_parsed(self) -> None:
        """Valid JSON bytes are decoded to a dict."""
        consumer = _make_consumer()
        record = _make_mock_record(
            value=b'{"event_id": "abc", "run_id": "xyz", "skill_name": "pr-review"}'
        )

        result = consumer._parse_message(record)

        assert result is not None
        assert result["event_id"] == "abc"
        assert result["skill_name"] == "pr-review"

    @pytest.mark.unit
    def test_invalid_json_returns_none(self) -> None:
        """Invalid JSON returns None."""
        consumer = _make_consumer()
        record = _make_mock_record(value=b"not-valid-json")

        result = consumer._parse_message(record)

        assert result is None

    @pytest.mark.unit
    def test_empty_bytes_returns_none(self) -> None:
        """Empty bytes returns None."""
        consumer = _make_consumer()
        record = _make_mock_record(value=b"")

        result = consumer._parse_message(record)

        assert result is None

    @pytest.mark.unit
    def test_json_array_returns_none(self) -> None:
        """JSON array of non-dicts (e.g. strings) returns None."""
        consumer = _make_consumer()
        record = _make_mock_record(value=b'["a", "b"]')

        result = consumer._parse_message(record)

        assert result is None

    @pytest.mark.unit
    def test_array_wrapped_single_dict_unwrapped(self) -> None:
        """Legacy array-wrapped single-event format is unwrapped and returned as dict."""
        consumer = _make_consumer()
        payload = {"event_id": "abc", "run_id": "xyz", "skill_name": "pr-review"}
        record = _make_mock_record(value=json.dumps([payload]).encode())

        result = consumer._parse_message(record)

        assert result is not None
        assert result["event_id"] == "abc"
        assert result["skill_name"] == "pr-review"

    @pytest.mark.unit
    def test_array_wrapped_multi_element_returns_none(self) -> None:
        """Array with more than one element is not a known legacy format and returns None."""
        consumer = _make_consumer()
        payload1 = {"event_id": "abc"}
        payload2 = {"event_id": "def"}
        record = _make_mock_record(value=json.dumps([payload1, payload2]).encode())

        result = consumer._parse_message(record)

        assert result is None

    @pytest.mark.unit
    def test_deprecated_skill_event_type_migrated(self) -> None:
        """skill_event_type is renamed to event_type when event_type is absent."""
        consumer = _make_consumer()
        record = _make_mock_record(
            value=b'{"skill_event_type": "started", "run_id": "xyz"}'
        )

        result = consumer._parse_message(record)

        assert result is not None
        assert result["event_type"] == "started"
        assert "skill_event_type" not in result
        assert result["run_id"] == "xyz"

    @pytest.mark.unit
    def test_skill_event_type_not_migrated_when_event_type_present(self) -> None:
        """skill_event_type is left alone when event_type is already present."""
        consumer = _make_consumer()
        record = _make_mock_record(
            value=b'{"skill_event_type": "old_value", "event_type": "started", "run_id": "xyz"}'
        )

        result = consumer._parse_message(record)

        assert result is not None
        assert result["event_type"] == "started"
        # skill_event_type preserved since event_type already present
        assert result["skill_event_type"] == "old_value"

    @pytest.mark.unit
    def test_array_wrapped_with_skill_event_type_migration(self) -> None:
        """Array-wrapped legacy message with deprecated field is unwrapped and migrated."""
        consumer = _make_consumer()
        payload = {"skill_event_type": "completed", "run_id": "xyz"}
        record = _make_mock_record(value=json.dumps([payload]).encode())

        result = consumer._parse_message(record)

        assert result is not None
        assert result["event_type"] == "completed"
        assert "skill_event_type" not in result

    @pytest.mark.unit
    def test_non_utf8_bytes_returns_none(self) -> None:
        """Non-UTF-8 bytes returns None."""
        consumer = _make_consumer()
        record = _make_mock_record(value=b"\xff\xfe")

        result = consumer._parse_message(record)

        assert result is None


# =============================================================================
# Tests: SkillLifecycleConsumer._build_health_response
# =============================================================================


class TestHealthResponse:
    """Test _build_health_response health status logic.

    OMN-3784: Idle-aware health check — consumer reports HEALTHY when idle
    (connected to Kafka, polling, but no events received yet), and only
    reports DEGRADED for actual failures.

    OMN-4568: Caught-up health check — consumer reports HEALTHY when it has
    written before but Kafka lag is 0 (messages_received == 0 since startup),
    even if write_age exceeds the staleness threshold.
    """

    @pytest.mark.unit
    def test_healthy_when_running_and_recent_writes(self) -> None:
        """Returns HEALTHY when consumer is running with recent writes and polls."""
        consumer = _make_consumer()
        consumer._running = True
        now = datetime.now(UTC)
        consumer.metrics.last_successful_write_at = now
        consumer.metrics.messages_received = 5
        consumer.metrics.last_poll_at = now

        response, status_code = consumer._build_health_response()

        assert response["status"] == str(EnumHealthStatus.HEALTHY)
        assert status_code == 200
        assert response["idle"] is False

    @pytest.mark.unit
    def test_healthy_when_idle_no_events(self) -> None:
        """Returns HEALTHY (200) when consumer is idle — polling but no events received.

        OMN-3784: This is the core fix. Previously returned DEGRADED (503) because
        last_successful_write_at was None, conflating idle with unhealthy.
        """
        consumer = _make_consumer()
        consumer._running = True
        consumer.metrics.last_successful_write_at = None  # No events ever received
        consumer.metrics.last_poll_at = datetime.now(UTC)  # Polling works fine

        response, status_code = consumer._build_health_response()

        assert response["status"] == str(EnumHealthStatus.HEALTHY)
        assert status_code == 200
        assert response["idle"] is True

    @pytest.mark.unit
    def test_degraded_when_stale_writes(self) -> None:
        """Returns DEGRADED when last write exceeds staleness threshold AND messages received.

        OMN-4568: The stale-write DEGRADED path requires messages_received > 0 to
        distinguish an active-but-broken pipeline from an idle caught-up consumer.
        """
        consumer = _make_consumer()
        consumer._running = True
        # Set last write to far in the past
        stale_time = datetime(2020, 1, 1, tzinfo=UTC)
        consumer.metrics.last_successful_write_at = stale_time
        consumer.metrics.last_poll_at = datetime.now(UTC)
        # Messages were received but not written — write pipeline is broken
        consumer.metrics.messages_received = 10

        response, status_code = consumer._build_health_response()

        assert response["status"] == str(EnumHealthStatus.DEGRADED)
        assert status_code == 503

    @pytest.mark.unit
    def test_healthy_when_caught_up_no_new_messages(self) -> None:
        """Returns HEALTHY when last write is stale but no messages received (lag=0).

        OMN-4568: Core fix. When the consumer has fully caught up (Kafka lag = 0,
        messages_received == 0 since startup) and is actively polling, write staleness
        does NOT indicate a problem — there is simply no traffic to write.
        Previously this returned DEGRADED (503).
        """
        consumer = _make_consumer()
        consumer._running = True
        # Last write was long ago (consumer processed events then went idle)
        stale_time = datetime(2020, 1, 1, tzinfo=UTC)
        consumer.metrics.last_successful_write_at = stale_time
        consumer.metrics.last_poll_at = datetime.now(UTC)
        # No messages received since startup — consumer is caught up (lag=0)
        consumer.metrics.messages_received = 0

        response, status_code = consumer._build_health_response()

        assert response["status"] == str(EnumHealthStatus.HEALTHY)
        assert status_code == 200
        assert response["idle"] is True

    @pytest.mark.unit
    def test_unhealthy_when_not_running(self) -> None:
        """Returns UNHEALTHY when consumer is stopped."""
        consumer = _make_consumer()
        consumer._running = False

        response, status_code = consumer._build_health_response()

        assert response["status"] == str(EnumHealthStatus.UNHEALTHY)
        assert status_code == 503

    @pytest.mark.unit
    def test_degraded_when_stale_polls(self) -> None:
        """Returns DEGRADED when last poll exceeds poll staleness threshold."""
        consumer = _make_consumer()
        consumer._running = True
        now = datetime.now(UTC)
        consumer.metrics.last_successful_write_at = now
        # Stale poll
        consumer.metrics.last_poll_at = datetime(2020, 1, 1, tzinfo=UTC)

        response, status_code = consumer._build_health_response()

        assert response["status"] == str(EnumHealthStatus.DEGRADED)
        assert status_code == 503

    @pytest.mark.unit
    def test_degraded_when_no_polls_at_all(self) -> None:
        """Returns DEGRADED when consumer has never polled (Kafka not connected)."""
        consumer = _make_consumer()
        consumer._running = True
        consumer.metrics.last_poll_at = None
        consumer.metrics.last_successful_write_at = None

        response, status_code = consumer._build_health_response()

        assert response["status"] == str(EnumHealthStatus.DEGRADED)
        assert status_code == 503

    @pytest.mark.unit
    def test_response_includes_running_field(self) -> None:
        """Health response includes 'running' field."""
        consumer = _make_consumer()
        consumer._running = True
        now = datetime.now(UTC)
        consumer.metrics.last_successful_write_at = now
        consumer.metrics.last_poll_at = now

        response, _ = consumer._build_health_response()

        assert "running" in response
        assert response["running"] is True

    @pytest.mark.unit
    def test_response_includes_idle_field(self) -> None:
        """Health response includes 'idle' field (OMN-3784)."""
        consumer = _make_consumer()
        consumer._running = True
        consumer.metrics.last_successful_write_at = None
        consumer.metrics.last_poll_at = datetime.now(UTC)

        response, _ = consumer._build_health_response()

        assert "idle" in response
        assert response["idle"] is True

    @pytest.mark.unit
    def test_not_idle_after_first_write(self) -> None:
        """Consumer is not idle once it has processed at least one event (OMN-4568).

        idle is True only when messages_received == 0. Once messages have been
        received and processed, the consumer is active — not idle.
        """
        consumer = _make_consumer()
        consumer._running = True
        now = datetime.now(UTC)
        consumer.metrics.last_successful_write_at = now
        consumer.metrics.messages_received = 3
        consumer.metrics.last_poll_at = now

        response, _ = consumer._build_health_response()

        assert response["idle"] is False
