# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerRuntimeErrorTriage.

Tests first-match-wins triage rule engine, cross-layer correlation,
and default rules for aiokafka/asyncpg/aiohttp.

Related Tickets:
    - OMN-5522: Create NodeRuntimeErrorTriageEffect
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.enum_runtime_error_severity import (
    EnumRuntimeErrorSeverity,
)
from omnibase_infra.models.health.model_runtime_error_event import (
    ModelRuntimeErrorEvent,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.handlers.handler_runtime_error_triage import (
    HandlerRuntimeErrorTriage,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.model_triage_rule import (
    DEFAULT_TRIAGE_RULES,
    ModelTriageRule,
)


def _make_error_event(
    *,
    logger_family: str = "aiokafka.consumer",
    error_category: EnumRuntimeErrorCategory = EnumRuntimeErrorCategory.KAFKA_CONSUMER,
    raw_message: str = "Heartbeat session expired",
    message_template: str = "Heartbeat session expired",
) -> ModelRuntimeErrorEvent:
    """Create a test runtime error event."""
    return ModelRuntimeErrorEvent.create(
        logger_family=logger_family,
        log_level="ERROR",
        message_template=message_template,
        raw_message=raw_message,
        error_category=error_category,
        severity=EnumRuntimeErrorSeverity.ERROR,
    )


@pytest.mark.unit
class TestModelTriageRule:
    """Tests for the triage rule matching logic."""

    def test_matches_logger_prefix(self) -> None:
        """Rule matches on logger prefix."""
        rule = ModelTriageRule(name="test", logger_prefix="aiokafka", action="alert")
        assert rule.matches(
            "aiokafka.consumer", EnumRuntimeErrorCategory.KAFKA_CONSUMER, "msg"
        )
        assert not rule.matches("asyncpg", EnumRuntimeErrorCategory.DATABASE, "msg")

    def test_matches_error_category(self) -> None:
        """Rule matches on error category."""
        rule = ModelTriageRule(
            name="test",
            error_category=EnumRuntimeErrorCategory.DATABASE,
            action="alert",
        )
        assert rule.matches("asyncpg", EnumRuntimeErrorCategory.DATABASE, "msg")
        assert not rule.matches(
            "asyncpg", EnumRuntimeErrorCategory.KAFKA_CONSUMER, "msg"
        )

    def test_matches_message_pattern(self) -> None:
        """Rule matches on message regex pattern."""
        rule = ModelTriageRule(
            name="test",
            message_pattern=r"heartbeat|session.*timeout",
            action="alert",
        )
        assert rule.matches(
            "logger", EnumRuntimeErrorCategory.UNKNOWN, "heartbeat failed"
        )
        assert rule.matches(
            "logger", EnumRuntimeErrorCategory.UNKNOWN, "session has timeout"
        )
        assert not rule.matches(
            "logger", EnumRuntimeErrorCategory.UNKNOWN, "connection refused"
        )

    def test_matches_all_conditions(self) -> None:
        """Rule requires all conditions to match."""
        rule = ModelTriageRule(
            name="test",
            logger_prefix="aiokafka",
            error_category=EnumRuntimeErrorCategory.KAFKA_CONSUMER,
            message_pattern=r"heartbeat",
            action="alert",
        )
        assert rule.matches(
            "aiokafka.consumer",
            EnumRuntimeErrorCategory.KAFKA_CONSUMER,
            "heartbeat failed",
        )
        assert not rule.matches(
            "aiokafka.consumer",
            EnumRuntimeErrorCategory.KAFKA_CONSUMER,
            "connection refused",
        )

    def test_empty_conditions_match_all(self) -> None:
        """Rule with empty conditions matches everything."""
        rule = ModelTriageRule(name="catch_all", action="alert")
        assert rule.matches(
            "any.logger", EnumRuntimeErrorCategory.UNKNOWN, "any message"
        )

    def test_default_rules_have_catch_all(self) -> None:
        """Default rules include a catch-all at the end."""
        sorted_rules = sorted(DEFAULT_TRIAGE_RULES, key=lambda r: r.priority)
        last_rule = sorted_rules[-1]
        assert last_rule.name == "catch_all"
        assert last_rule.matches(
            "anything", EnumRuntimeErrorCategory.UNKNOWN, "anything"
        )


@pytest.mark.unit
class TestHandlerRuntimeErrorTriage:
    """Tests for the triage handler."""

    async def test_alert_action(self) -> None:
        """Alert action sends Slack notification."""
        db_pool = MagicMock()
        conn = AsyncMock()
        # Order: correlate_with_layer1 (KAFKA_CONSUMER) -> upsert_incident
        conn.fetchrow = AsyncMock(
            side_effect=[
                None,  # correlation query: no active consumer incidents
                {"occurrence_count": 1, "incident_state": "open"},  # upsert
            ]
        )
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        slack_handler = AsyncMock()
        rules = [ModelTriageRule(name="alert_all", priority=1, action="alert")]
        handler = HandlerRuntimeErrorTriage(
            db_pool=db_pool, rules=rules, slack_handler=slack_handler
        )
        event = _make_error_event()

        result = await handler.handle(event)

        assert result.action == "alert"
        assert result.matched_rule == "alert_all"
        slack_handler.assert_awaited_once()

    async def test_suppress_action(self) -> None:
        """Suppress action suppresses the incident."""
        db_pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={"occurrence_count": 1, "incident_state": "open"}
        )
        conn.execute = AsyncMock()
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        slack_handler = AsyncMock()
        rules = [ModelTriageRule(name="suppress_all", priority=1, action="suppress")]
        handler = HandlerRuntimeErrorTriage(
            db_pool=db_pool, rules=rules, slack_handler=slack_handler
        )
        event = _make_error_event()

        result = await handler.handle(event)

        assert result.action == "suppress"
        assert result.incident_state == "suppressed"

    async def test_ticket_action(self) -> None:
        """Ticket action creates Linear ticket."""
        db_pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={"occurrence_count": 1, "incident_state": "open"}
        )
        conn.execute = AsyncMock()
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        linear_handler = AsyncMock()
        rules = [ModelTriageRule(name="ticket_all", priority=1, action="ticket")]
        handler = HandlerRuntimeErrorTriage(
            db_pool=db_pool, rules=rules, linear_handler=linear_handler
        )
        event = _make_error_event()

        result = await handler.handle(event)

        assert result.action == "ticket"
        assert result.incident_state == "ticketed"
        linear_handler.assert_awaited_once()

    async def test_cross_layer_correlation_kafka(self) -> None:
        """Kafka-related errors trigger cross-layer correlation."""
        db_pool = MagicMock()
        conn = AsyncMock()
        # Order: correlate_with_layer1 -> upsert_incident
        conn.fetchrow = AsyncMock(
            side_effect=[
                {"fingerprint": "consumer-fp-123"},  # correlation query
                {"occurrence_count": 1, "incident_state": "open"},  # upsert
            ]
        )
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        rules = [ModelTriageRule(name="alert_all", priority=1, action="alert")]
        handler = HandlerRuntimeErrorTriage(db_pool=db_pool, rules=rules)

        event = _make_error_event(
            error_category=EnumRuntimeErrorCategory.KAFKA_CONSUMER,
        )
        result = await handler.handle(event)

        assert result.correlated_consumer_fingerprint == "consumer-fp-123"

    async def test_no_correlation_for_non_kafka(self) -> None:
        """Non-Kafka errors do not trigger cross-layer correlation."""
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

        rules = [ModelTriageRule(name="alert_all", priority=1, action="alert")]
        handler = HandlerRuntimeErrorTriage(db_pool=db_pool, rules=rules)

        event = _make_error_event(
            logger_family="asyncpg",
            error_category=EnumRuntimeErrorCategory.DATABASE,
            raw_message="connection refused",
        )
        result = await handler.handle(event)

        assert result.correlated_consumer_fingerprint is None

    async def test_first_match_wins_priority(self) -> None:
        """Rules are evaluated in priority order; first match wins."""
        db_pool = MagicMock()
        conn = AsyncMock()
        # Uses DATABASE category to avoid correlation path
        conn.fetchrow = AsyncMock(
            return_value={"occurrence_count": 1, "incident_state": "open"}
        )
        conn.execute = AsyncMock()
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        rules = [
            ModelTriageRule(name="high_priority", priority=1, action="ticket"),
            ModelTriageRule(name="low_priority", priority=100, action="alert"),
        ]
        handler = HandlerRuntimeErrorTriage(db_pool=db_pool, rules=rules)
        event = _make_error_event(
            logger_family="asyncpg",
            error_category=EnumRuntimeErrorCategory.DATABASE,
            raw_message="connection error",
        )

        result = await handler.handle(event)

        assert result.matched_rule == "high_priority"
        assert result.action == "ticket"

    async def test_emit_triage_event_on_alert(self) -> None:
        """Alert action emits error-triaged event to Kafka."""
        db_pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            side_effect=[
                None,  # correlation query
                {"occurrence_count": 1, "incident_state": "open"},  # upsert
            ]
        )
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        rules = [ModelTriageRule(name="alert_all", priority=1, action="alert")]
        handler = HandlerRuntimeErrorTriage(
            db_pool=db_pool, rules=rules, event_bus=event_bus
        )
        event = _make_error_event()

        result = await handler.handle(event)

        assert result.action == "alert"
        event_bus.publish_envelope.assert_awaited_once()
        call_kwargs = event_bus.publish_envelope.call_args
        assert "error-triaged" in call_kwargs.kwargs.get("topic", "")

    async def test_no_emit_without_event_bus(self) -> None:
        """No emit when event_bus is None (DB-only mode)."""
        db_pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            side_effect=[
                None,  # correlation query
                {"occurrence_count": 1, "incident_state": "open"},  # upsert
            ]
        )
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        rules = [ModelTriageRule(name="alert_all", priority=1, action="alert")]
        handler = HandlerRuntimeErrorTriage(
            db_pool=db_pool, rules=rules, event_bus=None
        )
        event = _make_error_event()

        # Should complete without error even without event bus
        result = await handler.handle(event)
        assert result.action == "alert"

    async def test_emit_failure_does_not_block_triage(self) -> None:
        """Emit failure does not prevent triage result from being returned."""
        db_pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            side_effect=[
                None,  # correlation query
                {"occurrence_count": 1, "incident_state": "open"},  # upsert
            ]
        )
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock(
            side_effect=RuntimeError("Kafka unavailable")
        )
        rules = [ModelTriageRule(name="alert_all", priority=1, action="alert")]
        handler = HandlerRuntimeErrorTriage(
            db_pool=db_pool, rules=rules, event_bus=event_bus
        )
        event = _make_error_event()

        # Should still return result despite emit failure
        result = await handler.handle(event)
        assert result.action == "alert"
        assert result.matched_rule == "alert_all"

    async def test_emit_triage_event_on_ticket(self) -> None:
        """Ticket action also emits error-triaged event."""
        db_pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={"occurrence_count": 5, "incident_state": "open"}
        )
        conn.execute = AsyncMock()
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        rules = [ModelTriageRule(name="ticket_all", priority=1, action="ticket")]
        handler = HandlerRuntimeErrorTriage(
            db_pool=db_pool, rules=rules, event_bus=event_bus
        )
        event = _make_error_event(
            logger_family="asyncpg",
            error_category=EnumRuntimeErrorCategory.DATABASE,
            raw_message="connection refused",
        )

        result = await handler.handle(event)

        assert result.action == "ticket"
        event_bus.publish_envelope.assert_awaited_once()

    async def test_emit_triage_event_on_suppress(self) -> None:
        """Suppress action also emits error-triaged event."""
        db_pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={"occurrence_count": 1, "incident_state": "open"}
        )
        conn.execute = AsyncMock()
        db_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
            )
        )

        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        rules = [ModelTriageRule(name="suppress_all", priority=1, action="suppress")]
        handler = HandlerRuntimeErrorTriage(
            db_pool=db_pool, rules=rules, event_bus=event_bus
        )
        event = _make_error_event(
            logger_family="asyncpg",
            error_category=EnumRuntimeErrorCategory.DATABASE,
            raw_message="some error",
        )

        result = await handler.handle(event)

        assert result.action == "suppress"
        event_bus.publish_envelope.assert_awaited_once()
