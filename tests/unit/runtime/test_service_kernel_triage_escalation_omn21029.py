# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel publishes the escalation events the triage handler returns (OMN-21029).

``HandlerRuntimeErrorTriage.handle`` claims an incident once and returns the one
``ModelRuntimeErrorEscalatedEvent``. The kernel's triage consumer awaited it and
dropped the output, so an incident became ``escalated`` and nothing reached the
bus. ``publish_triage_escalations`` is the step that puts the event on the topic
the node contract declares.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.enum_runtime_error_severity import (
    EnumRuntimeErrorSeverity,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.enum_escalation_reason import (
    EnumEscalationReason,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.model_runtime_error_escalated_event import (
    ModelRuntimeErrorEscalatedEvent,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)
from omnibase_infra.runtime.service_kernel import publish_triage_escalations

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_runtime_error_triage_effect/contract.yaml"
)
_ESCALATION_TOPIC = "onex.evt.omnibase-infra.runtime-error-escalated.v1"


def _escalation(fingerprint: str = "fp-1") -> ModelRuntimeErrorEscalatedEvent:
    return ModelRuntimeErrorEscalatedEvent(
        correlation_id=uuid4(),
        fingerprint=fingerprint,
        logger_family="asyncpg.pool",
        error_category=EnumRuntimeErrorCategory.DATABASE,
        severity=EnumRuntimeErrorSeverity.CRITICAL,
        occurrence_count=10,
        reason=EnumEscalationReason.OCCURRENCE_THRESHOLD,
        matched_rule="alert_all",
    )


def _output(*events: ModelRuntimeErrorEscalatedEvent) -> ModelHandlerOutput[None]:
    return ModelHandlerOutput.for_effect(
        input_envelope_id=uuid4(),
        correlation_id=uuid4(),
        handler_id="HandlerRuntimeErrorTriage",
        events=events,
    )


def _bus() -> MagicMock:
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock()
    return bus


@pytest.mark.unit
@pytest.mark.asyncio
async def test_escalation_is_published_once_to_the_contract_topic() -> None:
    """The returned event reaches the topic the contract declares, keyed by fingerprint."""
    bus = _bus()
    event = _escalation()

    count = await publish_triage_escalations(
        event_bus=bus,
        output=_output(event),
        topics_by_event=load_published_events_map(_CONTRACT_PATH),
    )

    assert count == 1
    bus.publish_envelope.assert_awaited_once()
    envelope = bus.publish_envelope.await_args.args[0]
    assert isinstance(envelope, ModelEventEnvelope)
    assert envelope.event_type == "ModelRuntimeErrorEscalatedEvent"
    assert envelope.payload["fingerprint"] == "fp-1"
    assert envelope.payload["occurrence_count"] == 10
    assert envelope.correlation_id == event.correlation_id
    assert bus.publish_envelope.await_args.kwargs == {
        "topic": _ESCALATION_TOPIC,
        "key": b"fp-1",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_output_without_events_publishes_nothing() -> None:
    """An ordinary occurrence (no escalation) puts nothing on the bus."""
    bus = _bus()

    count = await publish_triage_escalations(
        event_bus=bus,
        output=_output(),
        topics_by_event=load_published_events_map(_CONTRACT_PATH),
    )

    assert count == 0
    bus.publish_envelope.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_every_returned_event_is_published() -> None:
    bus = _bus()

    count = await publish_triage_escalations(
        event_bus=bus,
        output=_output(_escalation("fp-a"), _escalation("fp-b")),
        topics_by_event=load_published_events_map(_CONTRACT_PATH),
    )

    assert count == 2
    keys = [call.kwargs["key"] for call in bus.publish_envelope.await_args_list]
    assert keys == [b"fp-a", b"fp-b"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failed_publish_is_logged_and_the_next_event_still_goes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The incident is already escalated, so a failed publish is loud, not fatal."""
    bus = _bus()
    bus.publish_envelope.side_effect = [InfraConnectionError("broker down"), None]

    with caplog.at_level("ERROR"):
        count = await publish_triage_escalations(
            event_bus=bus,
            output=_output(_escalation("fp-a"), _escalation("fp-b")),
            topics_by_event=load_published_events_map(_CONTRACT_PATH),
        )

    assert count == 1
    assert bus.publish_envelope.await_count == 2
    assert "fingerprint=fp-a" in caplog.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_without_a_declared_topic_is_logged_not_published(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No contract topic for the class: nothing is guessed, the skip is logged."""
    bus = _bus()

    with caplog.at_level("ERROR"):
        count = await publish_triage_escalations(
            event_bus=bus,
            output=_output(_escalation()),
            topics_by_event={},
        )

    assert count == 0
    bus.publish_envelope.assert_not_awaited()
    assert "contract declares no topic for ModelRuntimeErrorEscalatedEvent" in (
        caplog.text
    )
