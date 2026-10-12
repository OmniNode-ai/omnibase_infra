# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Escalation tests for NodeRuntimeErrorTriageEffect (OMN-21029).

The node recorded incidents and never moved one to ``escalated``. These tests
pin the escalation rule declared in the contract, the pure decision, and the
once-per-incident guarantee of the handler's write.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

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
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.enum_escalation_reason import (
    EnumEscalationReason,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.model_escalation_rule import (
    ModelEscalationRule,
    load_escalation_rule,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.model_runtime_error_escalated_event import (
    ModelRuntimeErrorEscalatedEvent,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.model_triage_rule import (
    ModelTriageRule,
)

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/nodes/node_runtime_error_triage_effect/contract.yaml"
)
_ESCALATION_TOPIC = "onex.evt.omnibase-infra.runtime-error-escalated.v1"

_RULE = ModelEscalationRule(
    occurrence_threshold=10,
    new_fingerprint_severity=EnumRuntimeErrorSeverity.CRITICAL,
)


def _event(
    severity: EnumRuntimeErrorSeverity = EnumRuntimeErrorSeverity.ERROR,
) -> ModelRuntimeErrorEvent:
    return ModelRuntimeErrorEvent.create(
        logger_family="asyncpg.pool",
        log_level="ERROR",
        message_template="pool exhausted",
        raw_message="pool exhausted",
        error_category=EnumRuntimeErrorCategory.DATABASE,
        severity=severity,
    )


def _pool(*rows: dict[str, object] | None) -> tuple[MagicMock, AsyncMock]:
    """Pool whose connection answers ``fetchrow`` with ``rows`` in order."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(side_effect=list(rows))
    pool = MagicMock()
    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
        )
    )
    return pool, conn


def _handler(pool: MagicMock) -> HandlerRuntimeErrorTriage:
    return HandlerRuntimeErrorTriage(
        db_pool=pool,
        rules=[ModelTriageRule(name="alert_all", priority=1, action="alert")],
        escalation_rule=_RULE,
    )


@pytest.mark.unit
class TestEscalationRuleContract:
    """The rule and the event topic are declared in contract.yaml."""

    def test_contract_declares_escalation_rule(self) -> None:
        rule = load_escalation_rule(_CONTRACT_PATH)
        contract = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))

        assert contract["escalation"]["occurrence_threshold"] == (
            rule.occurrence_threshold
        )
        assert rule.occurrence_threshold >= 2
        assert rule.new_fingerprint_severity is EnumRuntimeErrorSeverity.CRITICAL

    def test_contract_declares_escalation_topic_and_event(self) -> None:
        contract = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))

        assert contract["event_bus"]["publish_topics"] == [_ESCALATION_TOPIC]
        assert contract["published_events"] == [
            {
                "topic": _ESCALATION_TOPIC,
                "event_type": ModelRuntimeErrorEscalatedEvent.__name__,
                "message_category": "EVENT",
                "description": contract["published_events"][0]["description"],
            }
        ]

    def test_runtime_routes_the_event_to_the_declared_topic(self) -> None:
        """The dispatcher's own map sends the event class to the contract topic."""
        from omnibase_infra.runtime.event_bus_subcontract_wiring import (
            load_published_events_map,
        )

        assert load_published_events_map(_CONTRACT_PATH) == {
            ModelRuntimeErrorEscalatedEvent.__name__: _ESCALATION_TOPIC
        }


@pytest.mark.unit
class TestEscalationDecision:
    """The pure decision: no I/O, no clock."""

    def test_open_incident_below_threshold_does_not_escalate(self) -> None:
        assert (
            _RULE.evaluate(
                incident_state="open",
                occurrence_count=9,
                severity=EnumRuntimeErrorSeverity.ERROR,
                action="alert",
            )
            is None
        )

    def test_open_incident_at_threshold_escalates(self) -> None:
        assert (
            _RULE.evaluate(
                incident_state="open",
                occurrence_count=10,
                severity=EnumRuntimeErrorSeverity.ERROR,
                action="alert",
            )
            is EnumEscalationReason.OCCURRENCE_THRESHOLD
        )

    def test_new_fingerprint_at_severity_escalates_on_first_occurrence(self) -> None:
        assert (
            _RULE.evaluate(
                incident_state="open",
                occurrence_count=1,
                severity=EnumRuntimeErrorSeverity.CRITICAL,
                action="alert",
            )
            is EnumEscalationReason.NEW_FINGERPRINT_SEVERITY
        )

    def test_escalated_incident_does_not_escalate_again(self) -> None:
        assert (
            _RULE.evaluate(
                incident_state="escalated",
                occurrence_count=500,
                severity=EnumRuntimeErrorSeverity.CRITICAL,
                action="alert",
            )
            is None
        )

    @pytest.mark.parametrize("action", ["suppress", "ticket"])
    def test_only_alert_rules_escalate(self, action: str) -> None:
        assert (
            _RULE.evaluate(
                incident_state="open",
                occurrence_count=500,
                severity=EnumRuntimeErrorSeverity.CRITICAL,
                action=action,
            )
            is None
        )


@pytest.mark.unit
class TestEscalation:
    """AC1: crossing the rule escalates and publishes one typed event."""

    async def test_incident_crossing_threshold_is_escalated_and_event_emitted(
        self,
    ) -> None:
        pool, conn = _pool(
            {"occurrence_count": 10, "incident_state": "open"},  # upsert
            {"id": 7, "first_seen_at": None},  # escalation claim
        )

        output = await _handler(pool).handle(_event())

        assert len(output.events) == 1
        escalated = output.events[0]
        assert isinstance(escalated, ModelRuntimeErrorEscalatedEvent)
        assert escalated.occurrence_count == 10
        assert escalated.reason is EnumEscalationReason.OCCURRENCE_THRESHOLD
        claim_sql = conn.fetchrow.await_args_list[1].args[0]
        assert "escalated" in claim_sql
        assert "incident_state = 'open'" in claim_sql

    async def test_new_critical_fingerprint_is_escalated_on_first_occurrence(
        self,
    ) -> None:
        pool, _ = _pool(
            {"occurrence_count": 1, "incident_state": "open"},
            {"id": 8, "first_seen_at": None},
        )

        output = await _handler(pool).handle(_event(EnumRuntimeErrorSeverity.CRITICAL))

        assert len(output.events) == 1
        assert output.events[0].reason is EnumEscalationReason.NEW_FINGERPRINT_SEVERITY

    async def test_incident_below_threshold_publishes_nothing(self) -> None:
        pool, conn = _pool({"occurrence_count": 3, "incident_state": "open"})

        output = await _handler(pool).handle(_event())

        assert output.events == ()
        assert conn.fetchrow.await_count == 1


@pytest.mark.unit
class TestEscalateOnce:
    """AC2: an incident already escalated is never escalated again."""

    async def test_escalate_once_later_occurrences_publish_nothing(self) -> None:
        pool, conn = _pool(
            {"occurrence_count": 11, "incident_state": "escalated"},
            {"occurrence_count": 12, "incident_state": "escalated"},
        )
        handler = _handler(pool)

        first = await handler.handle(_event())
        second = await handler.handle(_event())

        assert first.events == ()
        assert second.events == ()
        # No escalation claim was attempted for an escalated incident.
        assert conn.fetchrow.await_count == 2

    async def test_escalate_once_lost_claim_race_publishes_nothing(self) -> None:
        """A concurrent consumer escalated first: the claim UPDATE matches no row."""
        pool, _ = _pool(
            {"occurrence_count": 10, "incident_state": "open"},
            None,
        )

        output = await _handler(pool).handle(_event())

        assert output.events == ()
