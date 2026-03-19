# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerConsumerHealthTriage.

Tests graduated response logic: Slack warning -> repeated -> restart -> Linear.

Related Tickets:
    - OMN-5520: Create NodeConsumerHealthTriageEffect
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.models.health.enum_consumer_health_event_type import (
    EnumConsumerHealthEventType,
)
from omnibase_infra.models.health.enum_consumer_health_severity import (
    EnumConsumerHealthSeverity,
)
from omnibase_infra.models.health.model_consumer_health_event import (
    ModelConsumerHealthEvent,
)
from omnibase_infra.nodes.node_consumer_health_triage_effect.handlers.handler_consumer_health_triage import (
    HandlerConsumerHealthTriage,
)


def _make_event(
    *,
    fingerprint: str = "abc123",
    event_type: EnumConsumerHealthEventType = EnumConsumerHealthEventType.HEARTBEAT_FAILURE,
    severity: EnumConsumerHealthSeverity = EnumConsumerHealthSeverity.ERROR,
) -> ModelConsumerHealthEvent:
    """Create a test health event."""
    return ModelConsumerHealthEvent.create(
        consumer_identity="test-consumer",
        consumer_group="test-group",
        topic="test-topic",
        event_type=event_type,
        severity=severity,
    )


@pytest.mark.unit
class TestHandlerConsumerHealthTriage:
    """Tests for graduated triage response."""

    @patch.dict("os.environ", {"ENABLE_CONSUMER_HEALTH_TRIAGE": ""})
    async def test_suppressed_when_disabled(self) -> None:
        """Handler returns suppressed when feature flag is off."""
        db_pool = MagicMock()
        handler = HandlerConsumerHealthTriage(db_pool=db_pool)
        event = _make_event()

        result = await handler.handle(event)

        assert result.action == "suppressed"
        assert result.occurrence_count == 0

    @patch.dict("os.environ", {"ENABLE_CONSUMER_HEALTH_TRIAGE": "true"})
    async def test_first_occurrence_slack_warning(self) -> None:
        """First occurrence triggers Slack WARNING."""
        db_pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={"occurrence_count": 1, "incident_state": "open"}
        )
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        slack_handler = AsyncMock()
        handler = HandlerConsumerHealthTriage(
            db_pool=db_pool, slack_handler=slack_handler
        )
        event = _make_event()

        result = await handler.handle(event)

        assert result.action == "slack_warning"
        assert result.occurrence_count == 1
        slack_handler.assert_awaited_once()

    @patch.dict("os.environ", {"ENABLE_CONSUMER_HEALTH_TRIAGE": "true"})
    async def test_second_occurrence_slack_repeated(self) -> None:
        """Second occurrence triggers Slack REPEATED."""
        db_pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={"occurrence_count": 2, "incident_state": "open"}
        )
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        slack_handler = AsyncMock()
        handler = HandlerConsumerHealthTriage(
            db_pool=db_pool, slack_handler=slack_handler
        )
        event = _make_event()

        result = await handler.handle(event)

        assert result.action == "slack_repeated"
        assert result.occurrence_count == 2

    @patch.dict(
        "os.environ",
        {
            "ENABLE_CONSUMER_HEALTH_TRIAGE": "true",
            "ENABLE_CONSUMER_AUTO_RESTART": "true",
        },
    )
    async def test_third_occurrence_restart_command(self) -> None:
        """Third occurrence triggers restart command when auto-restart enabled."""
        db_pool = MagicMock()
        conn = AsyncMock()
        # First call: upsert returns count=3
        # Second call: rate limit check returns no prior restarts
        conn.fetchrow = AsyncMock(
            side_effect=[
                {"occurrence_count": 3, "incident_state": "open"},
                None,  # No prior restarts
            ]
        )
        conn.execute = AsyncMock()
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        producer = AsyncMock()
        producer.send = AsyncMock()
        handler = HandlerConsumerHealthTriage(db_pool=db_pool, producer=producer)
        event = _make_event()

        result = await handler.handle(event)

        assert result.action == "restart_command"
        assert result.incident_state == "restart_pending"
        producer.send.assert_awaited_once()

    @patch.dict(
        "os.environ",
        {
            "ENABLE_CONSUMER_HEALTH_TRIAGE": "true",
            "ENABLE_CONSUMER_AUTO_RESTART": "",
        },
    )
    async def test_third_occurrence_no_restart_when_disabled(self) -> None:
        """Third occurrence sends Slack repeated when auto-restart is disabled."""
        db_pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={"occurrence_count": 3, "incident_state": "open"}
        )
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        slack_handler = AsyncMock()
        handler = HandlerConsumerHealthTriage(
            db_pool=db_pool, slack_handler=slack_handler
        )
        event = _make_event()

        result = await handler.handle(event)

        assert result.action == "slack_repeated"

    def test_is_enabled_default_off(self) -> None:
        """Feature flag defaults to off."""
        with patch.dict("os.environ", {}, clear=True):
            assert not HandlerConsumerHealthTriage.is_enabled()

    def test_is_enabled_on(self) -> None:
        """Feature flag responds to truthy values."""
        for val in ("1", "true", "yes", "on"):
            with patch.dict("os.environ", {"ENABLE_CONSUMER_HEALTH_TRIAGE": val}):
                assert HandlerConsumerHealthTriage.is_enabled()
