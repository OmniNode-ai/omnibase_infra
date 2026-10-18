# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An escalated runtime error incident reaches the bus [OMN-21029].

The real triage handler (only its database pool is a stand-in), the kernel's
``publish_triage_escalations`` and a real in-memory event bus: an incident that
crosses the contract-declared rule is claimed once and exactly one escalation
envelope arrives on the contract-declared topic; an ordinary occurrence puts
nothing on it.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.models import ModelNodeIdentity
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
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.model_escalation_rule import (
    load_escalation_rule,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.model_triage_rule import (
    ModelTriageRule,
)
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)
from omnibase_infra.runtime.service_kernel import publish_triage_escalations

pytestmark = pytest.mark.integration

_CONTRACT_PATH = (
    Path(inspect.getfile(HandlerRuntimeErrorTriage)).parents[1] / "contract.yaml"
)
_ESCALATION_TOPIC = "onex.evt.omnibase-infra.runtime-error-escalated.v1"


def _event() -> ModelRuntimeErrorEvent:
    return ModelRuntimeErrorEvent.create(
        logger_family="asyncpg.pool",
        log_level="ERROR",
        message_template="pool exhausted",
        raw_message="pool exhausted",
        error_category=EnumRuntimeErrorCategory.DATABASE,
        severity=EnumRuntimeErrorSeverity.ERROR,
    )


def _pool(*rows: dict[str, object] | None) -> MagicMock:
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(side_effect=list(rows))
    pool = MagicMock()
    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()
        )
    )
    return pool


def _handler(pool: MagicMock) -> HandlerRuntimeErrorTriage:
    return HandlerRuntimeErrorTriage(
        db_pool=pool,
        rules=[ModelTriageRule(name="alert_all", priority=1, action="alert")],
        escalation_rule=load_escalation_rule(_CONTRACT_PATH),
    )


async def _collect(bus: EventBusInmemory) -> list[ModelEventMessage]:
    received: list[ModelEventMessage] = []

    async def _on_message(message: ModelEventMessage) -> None:
        received.append(message)

    await bus.subscribe(
        topic=_ESCALATION_TOPIC,
        node_identity=ModelNodeIdentity(
            env="test",
            service="omn21029-probe",
            node_name="escalation-probe",
            version="v1",
        ),
        on_message=_on_message,
        purpose=EnumConsumerGroupPurpose.CONSUME,
    )
    return received


@pytest.mark.asyncio
async def test_escalated_incident_arrives_on_the_contract_topic() -> None:
    bus = EventBusInmemory(environment="test", group="omn21029")
    await bus.start()
    try:
        received = await _collect(bus)
        handler = _handler(
            _pool(
                {"occurrence_count": 10, "incident_state": "open"},  # upsert
                {"id": 7, "first_seen_at": None},  # escalation claim
            )
        )

        output = await handler.handle(_event())
        published = await publish_triage_escalations(
            event_bus=bus,
            output=output,
            topics_by_event=load_published_events_map(_CONTRACT_PATH),
        )

        assert published == 1
        assert len(received) == 1
        body = json.loads(received[0].value)
        assert body["event_type"] == "ModelRuntimeErrorEscalatedEvent"
        assert body["payload"]["occurrence_count"] == 10
        assert received[0].key == body["payload"]["fingerprint"].encode("utf-8")
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_ordinary_occurrence_puts_nothing_on_the_topic() -> None:
    """Negative control: below the threshold the same chain publishes nothing."""
    bus = EventBusInmemory(environment="test", group="omn21029")
    await bus.start()
    try:
        received = await _collect(bus)
        handler = _handler(_pool({"occurrence_count": 3, "incident_state": "open"}))

        output = await handler.handle(_event())
        published = await publish_triage_escalations(
            event_bus=bus,
            output=output,
            topics_by_event=load_published_events_map(_CONTRACT_PATH),
        )

        assert published == 0
        assert received == []
    finally:
        await bus.close()
