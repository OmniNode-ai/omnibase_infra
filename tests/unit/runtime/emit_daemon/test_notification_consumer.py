# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for NotificationConsumer.

Tests the notification consumer that routes notification events to Slack.

Related Tickets:
    - OMN-1831: Implement event-driven Slack notifications via runtime
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from omnibase_infra.handlers.models.model_slack_alert import EnumAlertSeverity
from omnibase_infra.runtime.emit_daemon.notification_consumer import (
    NotificationConsumer,
)
from omnibase_infra.runtime.emit_daemon.topics import (
    TOPIC_NOTIFICATION_BLOCKED,
    TOPIC_NOTIFICATION_COMPLETED,
)

FIXED_UUID = UUID("12345678-1234-5678-1234-567812345678")
FIXED_SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")


@pytest.fixture
def mock_event_bus() -> MagicMock:
    """Create a mock event bus."""
    return MagicMock()


@pytest.fixture
def mock_handler() -> AsyncMock:
    """Create a mock handler that returns a successful result."""
    handler = AsyncMock()
    handler.handle.return_value = MagicMock(
        success=True,
        duration_ms=50.0,
        error=None,
        error_code=None,
    )
    return handler


class TestNotificationConsumerConstants:
    """Tests for consumer topic constants."""

    def test_blocked_topic_constant(self) -> None:
        """Should have correct blocked topic constant."""
        assert (
            TOPIC_NOTIFICATION_BLOCKED == "onex.evt.omniclaude.notification-blocked.v1"
        )

    def test_completed_topic_constant(self) -> None:
        """Should have correct completed topic constant."""
        assert (
            TOPIC_NOTIFICATION_COMPLETED
            == "onex.evt.omniclaude.notification-completed.v1"
        )


class TestNotificationConsumerInit:
    """Tests for NotificationConsumer initialization."""

    def test_init_with_event_bus(self, mock_event_bus: MagicMock) -> None:
        """Should initialize with event bus."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)
        assert consumer._event_bus is mock_event_bus
        assert consumer._running is False
        assert consumer._consumer_tasks == []

    def test_init_with_webhook_url(self, mock_event_bus: MagicMock) -> None:
        """Should initialize with optional webhook URL."""
        consumer = NotificationConsumer(
            event_bus=mock_event_bus,
            webhook_url="https://hooks.slack.com/test",
        )
        assert consumer._handler._webhook_url == "https://hooks.slack.com/test"


class TestNotificationConsumerStartStop:
    """Tests for consumer start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, mock_event_bus: MagicMock) -> None:
        """Should handle stop when not running."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)
        await consumer.stop()  # Should not raise
        assert consumer._running is False

    def test_initial_state(self, mock_event_bus: MagicMock) -> None:
        """Should have correct initial state."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)
        assert consumer._running is False
        assert consumer._consumer_tasks == []


class TestNotificationConsumerBlockedHandler:
    """Tests for _handle_blocked_event method."""

    @pytest.mark.asyncio
    async def test_transforms_blocked_event_to_alert(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Should transform blocked payload to Slack alert."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)

        # Mock the handler
        consumer._handler = MagicMock()
        consumer._handler.handle = AsyncMock(
            return_value=MagicMock(
                success=True, duration_ms=50.0, error=None, error_code=None
            )
        )

        payload = {
            "ticket_identifier": "OMN-1234",
            "reason": "Waiting for approval",
            "details": ["Phase: spec", "Gate: approve"],
            "repo": "omniclaude",
            "session_id": str(FIXED_SESSION_UUID),
            "correlation_id": str(FIXED_UUID),
        }

        await consumer._handle_blocked_event(payload)

        # Verify handler was called
        consumer._handler.handle.assert_called_once()
        call_args = consumer._handler.handle.call_args[0][0]

        assert call_args.severity == EnumAlertSeverity.WARNING
        assert call_args.title == ":ticket: OMN-1234 needs input"
        assert "*Waiting for approval*" in call_args.message
        assert "- Phase: spec" in call_args.message
        assert call_args.details["Ticket"] == "OMN-1234"
        assert call_args.details["Repo"] == "omniclaude"

    @pytest.mark.asyncio
    async def test_handles_missing_details(self, mock_event_bus: MagicMock) -> None:
        """Should handle payload without details field."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)
        consumer._handler = MagicMock()
        consumer._handler.handle = AsyncMock(
            return_value=MagicMock(success=True, duration_ms=50.0)
        )

        payload = {
            "ticket_identifier": "OMN-1234",
            "reason": "Waiting",
            "repo": "test",
            "session_id": str(FIXED_SESSION_UUID),
        }

        await consumer._handle_blocked_event(payload)

        call_args = consumer._handler.handle.call_args[0][0]
        assert "*Waiting*" in call_args.message
        # No bullet points for empty details
        assert (
            "- " not in call_args.message
            or "- " in call_args.message.split("*Waiting*")[0]
        )


class TestNotificationConsumerCompletedHandler:
    """Tests for _handle_completed_event method."""

    @pytest.mark.asyncio
    async def test_transforms_completed_event_to_alert(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Should transform completed payload to Slack alert."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)
        consumer._handler = MagicMock()
        consumer._handler.handle = AsyncMock(
            return_value=MagicMock(success=True, duration_ms=50.0)
        )

        payload = {
            "ticket_identifier": "OMN-1234",
            "summary": "Feature implemented",
            "repo": "omniclaude",
            "pr_url": "https://github.com/org/repo/pull/123",
            "session_id": str(FIXED_SESSION_UUID),
            "correlation_id": str(FIXED_UUID),
        }

        await consumer._handle_completed_event(payload)

        consumer._handler.handle.assert_called_once()
        call_args = consumer._handler.handle.call_args[0][0]

        assert call_args.severity == EnumAlertSeverity.INFO
        assert call_args.title == ":white_check_mark: OMN-1234 completed"
        assert call_args.message == "Feature implemented"
        assert call_args.details["Ticket"] == "OMN-1234"
        assert call_args.details["Repo"] == "omniclaude"
        assert call_args.details["PR"] == "https://github.com/org/repo/pull/123"

    @pytest.mark.asyncio
    async def test_handles_missing_pr_url(self, mock_event_bus: MagicMock) -> None:
        """Should handle payload without pr_url field."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)
        consumer._handler = MagicMock()
        consumer._handler.handle = AsyncMock(
            return_value=MagicMock(success=True, duration_ms=50.0)
        )

        payload = {
            "ticket_identifier": "OMN-1234",
            "summary": "Done",
            "repo": "test",
            "session_id": str(FIXED_SESSION_UUID),
        }

        await consumer._handle_completed_event(payload)

        call_args = consumer._handler.handle.call_args[0][0]
        assert "PR" not in call_args.details


class TestNotificationConsumerCorrelationId:
    """Tests for correlation_id extraction."""

    @pytest.mark.asyncio
    async def test_extracts_valid_correlation_id(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Should extract valid UUID from payload."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)
        consumer._handler = MagicMock()
        consumer._handler.handle = AsyncMock(
            return_value=MagicMock(success=True, duration_ms=50.0)
        )

        payload = {
            "ticket_identifier": "OMN-1234",
            "reason": "Waiting",
            "repo": "test",
            "session_id": str(FIXED_SESSION_UUID),
            "correlation_id": str(FIXED_UUID),
        }

        await consumer._handle_blocked_event(payload)

        call_args = consumer._handler.handle.call_args[0][0]
        assert call_args.correlation_id == FIXED_UUID

    @pytest.mark.asyncio
    async def test_generates_correlation_id_if_missing(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Should generate new UUID if correlation_id is missing."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)
        consumer._handler = MagicMock()
        consumer._handler.handle = AsyncMock(
            return_value=MagicMock(success=True, duration_ms=50.0)
        )

        payload = {
            "ticket_identifier": "OMN-1234",
            "reason": "Waiting",
            "repo": "test",
            "session_id": str(FIXED_SESSION_UUID),
        }

        await consumer._handle_blocked_event(payload)

        call_args = consumer._handler.handle.call_args[0][0]
        assert isinstance(call_args.correlation_id, UUID)

    @pytest.mark.asyncio
    async def test_generates_correlation_id_if_invalid(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Should generate new UUID if correlation_id is invalid."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)
        consumer._handler = MagicMock()
        consumer._handler.handle = AsyncMock(
            return_value=MagicMock(success=True, duration_ms=50.0)
        )

        payload = {
            "ticket_identifier": "OMN-1234",
            "reason": "Waiting",
            "repo": "test",
            "session_id": str(FIXED_SESSION_UUID),
            "correlation_id": "not-a-valid-uuid",
        }

        await consumer._handle_blocked_event(payload)

        call_args = consumer._handler.handle.call_args[0][0]
        assert isinstance(call_args.correlation_id, UUID)
        # Should be a new UUID, not the invalid string
        assert str(call_args.correlation_id) != "not-a-valid-uuid"


class TestNotificationConsumerMessageProcessing:
    """Tests for message processing."""

    @pytest.mark.asyncio
    async def test_process_message_parses_json(self, mock_event_bus: MagicMock) -> None:
        """Should parse JSON message and invoke handler."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)

        handler_called = False
        received_payload = None

        async def mock_handler(payload: dict[str, object]) -> None:
            nonlocal handler_called, received_payload
            handler_called = True
            received_payload = payload

        payload = {"test": "data"}
        message = json.dumps(payload).encode("utf-8")

        await consumer._process_message(message, mock_handler)

        assert handler_called is True
        assert received_payload == payload

    @pytest.mark.asyncio
    async def test_process_message_handles_invalid_json(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Should handle invalid JSON gracefully."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)

        handler_called = False

        async def mock_handler(payload: dict[str, object]) -> None:
            nonlocal handler_called
            handler_called = True

        message = b"not valid json"

        await consumer._process_message(message, mock_handler)

        assert handler_called is False  # Handler should not be called

    @pytest.mark.asyncio
    async def test_process_message_handles_non_dict_payload(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Should handle non-dict payload gracefully."""
        consumer = NotificationConsumer(event_bus=mock_event_bus)

        handler_called = False

        async def mock_handler(payload: dict[str, object]) -> None:
            nonlocal handler_called
            handler_called = True

        message = json.dumps(["not", "a", "dict"]).encode("utf-8")

        await consumer._process_message(message, mock_handler)

        assert handler_called is False  # Handler should not be called
