# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17214 Defect B — attribution survives the whole dispatch stack.

What this adds over the unit tests
----------------------------------
``tests/unit/runtime/auto_wiring/test_omn17214_publish_flow_attribution.py``
binds ``active_flow_key`` by hand and then calls the publish seam. That proves
the seam records, and assumes the binding reaches it.

This test removes the assumption. It wires the REAL subscription callback with a
REAL ``MessageDispatchEngine``, hands it a message on the wire shape the consumer
actually receives, and lets the publish happen from inside the dispatched
handler — several frames below the ``with active_flow_key(...)`` block, across
the engine's own routing and await boundaries. ``active_flow_key`` is a
``contextvar``, so this is the property that actually matters in production and
the one a seam-level test cannot see: a dispatch that hopped to a task created
outside the binding would lose it and silently under-report, which is exactly
the class of bug this ticket exists to close.

The dispatcher below stands in for ``_make_projection_dispatch_callback``, which
publishes its contract's ``terminal_event`` through
``_emit_projection_terminal_event`` at the end of a successful projection.
Driving the real one needs a live database target, so what is substituted here
is the DB write — never the publish seam, never the binding, never the engine.

The failure this reproduces, measured on the ``.201`` dev lane 2026-09-17T09:20Z
with the Defect A fix already deployed: the Phase 1 writer
``local.omnimarket.projection_consumer_flow`` read ``in=76 out=0`` across 19
consecutive windows while its own declared terminal topic advanced
HIGH-WATERMARK 178532 -> 178535 in the same 45 s.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _emit_projection_terminal_event,
    _make_event_bus_callback,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.observability import (
    get_consumer_flow_counters,
    reset_consumer_flow_counters,
)
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

_GROUP = "local.omnimarket.projection_consumer_flow.consume.1.0.1"
_IN_TOPIC = "onex.evt.platform.node-heartbeat.v1"  # onex-topic-allow: real topic from the OMN-17214 incident
_TERMINAL_TOPIC = "onex.evt.omnimarket.projection-consumer-flow-applied.v1"  # onex-topic-allow: real topic from the OMN-17214 incident
_ROUTE_PATTERN = "*.evt.platform.node-heartbeat.*"
_DISPATCHER_ID = "omn17214-projection-dispatcher"


class _ProjectionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rows_upserted: int


@pytest.fixture(autouse=True)
def _clean_counters() -> Any:
    reset_consumer_flow_counters()
    yield
    reset_consumer_flow_counters()


def _bus() -> MagicMock:
    """Spec'd to the real bus so a publish method that stops existing fails here."""
    bus = MagicMock(spec=EventBusKafka)
    bus.publish = AsyncMock()
    bus.publish_envelope = AsyncMock()
    bus._publish_raw_to_dlq = AsyncMock(return_value=True)
    return bus


def _engine_publishing_a_terminal_event(bus: MagicMock) -> MessageDispatchEngine:
    """A frozen engine whose dispatcher publishes a terminal event, as a projection does."""

    async def _dispatcher(envelope: ModelEventEnvelope[object]) -> None:
        # The DB write is what is stood in for. The publish below is the real
        # seam `_make_projection_dispatch_callback` reaches after `projected`.
        await _emit_projection_terminal_event(
            bus, _TERMINAL_TOPIC, envelope, _ProjectionResult(rows_upserted=1)
        )

    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id=_DISPATCHER_ID,
        dispatcher=_dispatcher,
        category=EnumMessageCategory.EVENT,
        message_types=None,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id=f"{_DISPATCHER_ID}-route",
            topic_pattern=_ROUTE_PATTERN,
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=_DISPATCHER_ID,
        )
    )
    engine.freeze()
    return engine


def _wire_message() -> MagicMock:
    msg = MagicMock()
    msg.value = json.dumps({"node_id": str(uuid4())}).encode("utf-8")
    return msg


@pytest.mark.asyncio
async def test_a_publishing_projection_reports_flowing_through_the_real_engine() -> (
    None
):
    """AC3 end to end: consume -> dispatch -> terminal publish -> in=1, out=1.

    ``out == 1`` is the whole assertion. ``in == 1`` was already true before this
    ticket's second fix — a row existed and said the consumer took a message and
    produced nothing, which is what made the verdict STALLED and the signal a
    lie rather than a gap.
    """
    bus = _bus()
    callback = _make_event_bus_callback(
        _IN_TOPIC,
        _engine_publishing_a_terminal_event(bus),
        DispatchResultApplier(event_bus=bus, output_topic=_TERMINAL_TOPIC),
        event_bus=bus,
        allowed_dispatcher_ids={_DISPATCHER_ID},
        consumer_group=_GROUP,
    )

    counters = get_consumer_flow_counters()
    t0 = datetime(2026, 9, 17, 9, 20, 0, tzinfo=UTC)
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)  # priming tick

    await callback(_wire_message())

    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=30))
    assert window is not None
    rows = {(d.consumer_group, d.topic): d for d in window.consumer_deltas}

    bus.publish.assert_awaited_once()
    assert (_GROUP, _IN_TOPIC) in rows, (
        f"no flow row for the subscription at all; rows={sorted(rows)}"
    )
    delta = rows[(_GROUP, _IN_TOPIC)]
    assert delta.messages_in == 1, f"consume was not counted: in={delta.messages_in}"
    assert delta.messages_out == 1, (
        "the terminal event was published from inside the dispatch, and the "
        f"subscription was still credited with {delta.messages_out} outputs — the "
        "task-local flow binding did not reach the publish seam, so this "
        "consumer derives STALLED while producing (OMN-17214 Defect B)"
    )


@pytest.mark.asyncio
async def test_the_terminal_topic_carries_produce_evidence_through_dispatch() -> None:
    """The same run must also record the terminal topic's own production.

    That evidence is what lets a downstream consumer of this topic be scored
    STARVED rather than IDLE. It is taken from our own publish seam and never
    from a broker query, so a publish invisible here leaves every downstream
    verdict underivable.
    """
    bus = _bus()
    callback = _make_event_bus_callback(
        _IN_TOPIC,
        _engine_publishing_a_terminal_event(bus),
        DispatchResultApplier(event_bus=bus, output_topic=_TERMINAL_TOPIC),
        event_bus=bus,
        allowed_dispatcher_ids={_DISPATCHER_ID},
        consumer_group=_GROUP,
    )

    counters = get_consumer_flow_counters()
    t0 = datetime(2026, 9, 17, 9, 20, 0, tzinfo=UTC)
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)

    await callback(_wire_message())

    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=30))
    assert window is not None
    produced = {d.topic: d.messages_produced for d in window.produce_deltas}
    assert produced.get(_TERMINAL_TOPIC) == 1, (
        f"{_TERMINAL_TOPIC} carried no produce evidence; produce_deltas={produced}"
    )
