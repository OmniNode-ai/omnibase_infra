# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for RuntimeLogEventBridge.

Ticket: OMN-5521
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.enum_runtime_error_severity import (
    EnumRuntimeErrorSeverity,
)
from omnibase_infra.observability.runtime_log_event_bridge import (
    RuntimeLogEventBridge,
    _categorize_logger,
    _log_level_to_severity,
    _templatize_message,
)
from omnibase_infra.runtime.observability import active_flow_key


@pytest.mark.unit
class TestTemplatizeMessage:
    """Tests for _templatize_message."""

    def test_replaces_uuid(self) -> None:
        msg = "Failed for id 123e4567-e89b-12d3-a456-426614174000"
        result = _templatize_message(msg)
        assert "123e4567" not in result
        assert "{}" in result

    def test_replaces_ip(self) -> None:
        msg = "Connection refused to 192.168.1.100:5432"
        result = _templatize_message(msg)
        assert "192.168.1.100" not in result

    def test_replaces_standalone_numbers(self) -> None:
        msg = "Timeout after 30000 ms for partition 5"
        result = _templatize_message(msg)
        assert "30000" not in result
        assert "partition {}" in result


@pytest.mark.unit
class TestCategorizeLogger:
    """Tests for _categorize_logger."""

    def test_kafka_consumer(self) -> None:
        assert (
            _categorize_logger("aiokafka.consumer.group")
            == EnumRuntimeErrorCategory.KAFKA_CONSUMER
        )

    def test_kafka_producer(self) -> None:
        assert (
            _categorize_logger("aiokafka.producer")
            == EnumRuntimeErrorCategory.KAFKA_PRODUCER
        )

    def test_database(self) -> None:
        assert _categorize_logger("asyncpg.pool") == EnumRuntimeErrorCategory.DATABASE

    def test_http_client(self) -> None:
        assert (
            _categorize_logger("aiohttp.client") == EnumRuntimeErrorCategory.HTTP_CLIENT
        )

    def test_unknown(self) -> None:
        assert (
            _categorize_logger("my.custom.logger") == EnumRuntimeErrorCategory.UNKNOWN
        )


@pytest.mark.unit
class TestLogLevelToSeverity:
    """Tests for _log_level_to_severity."""

    def test_critical(self) -> None:
        assert (
            _log_level_to_severity(logging.CRITICAL)
            == EnumRuntimeErrorSeverity.CRITICAL
        )

    def test_error(self) -> None:
        assert _log_level_to_severity(logging.ERROR) == EnumRuntimeErrorSeverity.ERROR

    def test_warning(self) -> None:
        assert (
            _log_level_to_severity(logging.WARNING) == EnumRuntimeErrorSeverity.WARNING
        )


@pytest.mark.unit
class TestRuntimeLogEventBridge:
    """Tests for RuntimeLogEventBridge."""

    @patch.dict("os.environ", {"ENABLE_RUNTIME_LOG_BRIDGE": "true"})
    def test_emit_enqueues_event(self) -> None:
        producer = AsyncMock()
        bridge = RuntimeLogEventBridge(producer, hostname="test-host")

        record = logging.LogRecord(
            name="aiokafka.consumer",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Heartbeat failed for group test-group",
            args=None,
            exc_info=None,
        )
        bridge.emit(record)
        assert bridge._queue.qsize() == 1

    @patch.dict("os.environ", {"ENABLE_RUNTIME_LOG_BRIDGE": ""})
    def test_emit_skipped_when_disabled(self) -> None:
        producer = AsyncMock()
        bridge = RuntimeLogEventBridge(producer)

        record = logging.LogRecord(
            name="aiokafka.consumer",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="test",
            args=None,
            exc_info=None,
        )
        bridge.emit(record)
        assert bridge._queue.qsize() == 0

    @patch.dict("os.environ", {"ENABLE_RUNTIME_LOG_BRIDGE": "true"})
    def test_rate_limiting(self) -> None:
        producer = AsyncMock()
        bridge = RuntimeLogEventBridge(producer)

        # Emit 5 events (max per window), then 6th should be rate-limited
        for i in range(6):
            record = logging.LogRecord(
                name="aiokafka.consumer",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="same error message",
                args=None,
                exc_info=None,
            )
            bridge.emit(record)

        assert bridge._queue.qsize() == 5
        assert bridge.events_rate_limited == 1

    @patch.dict("os.environ", {"ENABLE_RUNTIME_LOG_BRIDGE": "true"})
    async def test_drain_loop_emits_to_kafka(self) -> None:
        producer = AsyncMock()
        producer.send = AsyncMock()
        bridge = RuntimeLogEventBridge(producer, hostname="test-host")

        record = logging.LogRecord(
            name="asyncpg",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="connection refused",
            args=None,
            exc_info=None,
        )
        bridge.emit(record)

        await bridge.start()
        # Give drain loop a chance to process
        import asyncio

        await asyncio.sleep(0.1)
        await bridge.stop()

        assert bridge.events_emitted == 1
        producer.send.assert_called_once()

    @patch.dict("os.environ", {"ENABLE_RUNTIME_LOG_BRIDGE": "true"})
    async def test_runtime_error_dispatch_context_suppresses_only_recursive_logs(
        self,
    ) -> None:
        """An ERROR from this topic's active consumer cannot republish itself."""
        import asyncio

        producer = AsyncMock()
        producer.send = AsyncMock()
        topic = "onex.evt.omnibase-infra.runtime-error.v1"
        bridge = RuntimeLogEventBridge(producer, topic=topic)
        logger_name = "omnibase_infra.runtime.auto_wiring.recursive-guard-test"
        target_logger = logging.getLogger(logger_name)
        previous_level = target_logger.level
        previous_propagate = target_logger.propagate
        target_logger.setLevel(logging.ERROR)
        target_logger.propagate = False
        bridge.attach_to_loggers([logger_name])
        await bridge.start()
        try:
            with active_flow_key("test-runtime-error", topic):
                target_logger.error("boundary failure must not re-enter runtime-error")

            await asyncio.sleep(0)
            producer.send.assert_not_awaited()
            assert bridge.events_suppressed_in_flight == 1

            target_logger.error("unrelated error still reaches the runtime-error topic")
            await asyncio.sleep(0.05)

            producer.send.assert_awaited_once()
            assert bridge.events_emitted == 1
        finally:
            bridge.detach_from_loggers([logger_name])
            target_logger.setLevel(previous_level)
            target_logger.propagate = previous_propagate
            await bridge.stop()

    def test_is_enabled_default_false(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert not RuntimeLogEventBridge.is_enabled()

    def test_attach_and_detach(self) -> None:
        producer = AsyncMock()
        bridge = RuntimeLogEventBridge(producer)
        test_logger = logging.getLogger("test.attach.target")

        bridge.attach_to_loggers(["test.attach.target"])
        assert bridge in test_logger.handlers

        bridge.detach_from_loggers(["test.attach.target"])
        assert bridge not in test_logger.handlers

    @patch.dict("os.environ", {"ENABLE_RUNTIME_LOG_BRIDGE": "true"})
    def test_circular_prevention(self) -> None:
        """Bridge should NOT capture its own log records."""
        producer = AsyncMock()
        bridge = RuntimeLogEventBridge(producer)

        # Create a record from the bridge's own module
        record = logging.LogRecord(
            name="omnibase_infra.observability.runtime_log_event_bridge._bridge",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="internal error",
            args=None,
            exc_info=None,
        )
        bridge.emit(record)
        # Should be filtered out
        assert bridge._queue.qsize() == 0
