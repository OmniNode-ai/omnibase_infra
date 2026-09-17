# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17214 Defect B — every auto-wiring publish seam must attribute its output.

The defect
----------
``messages_out`` is attributed by ``record_active_out()``, which reads the
task-local ``active_flow_key`` binding. Since OMN-17214's first fix both
subscription branches bind that key around dispatch-and-apply, so the binding is
live all the way down the stack.

But only ONE publisher ever calls it: the external result applier
(``service_dispatch_result_applier._record_flow_output``). ``handler_wiring``
publishes a handler's own declared output on two further seams of its own, and
neither recorded anything:

* ``_emit_projection_terminal_event`` — the projection/reducer terminal event,
  published straight onto ``event_bus.publish``.
* ``_publish_outbox_batch`` — the state_io in-row outbox, published straight
  onto ``event_bus.publish_envelope``.

A node whose output leaves on either seam therefore reports ``messages_out = 0``
and derives ``STALLED`` while it is demonstrably producing. That is not a
cosmetic miscount: a STALLED verdict on a healthy producer is the alert-storm
failure mode the epic's own cautionary precedent (OMN-14440) was muted for.

Measured on the live ``.201`` dev lane 2026-09-17T09:20Z, with the fix for
Defect A deployed (lane revision ``abca7fb2``) and the projection caught up to
within seconds of live:

    local.omnimarket.projection_consumer_flow.consume.1.0.1
      | onex.evt.platform.node-heartbeat.v1 | in=76 out=0 | 19 windows STALLED

while its own declared terminal topic
``onex.evt.omnimarket.projection-consumer-flow-applied.v1`` advanced
HIGH-WATERMARK 178532 -> 178535 over 45 s in the same window. That is AC3's
falsifier, live: the Phase 1 deliverable reporting its own writer as stalled.

RED proof
---------
Before the fix these tests fail with ``messages_out == 0`` and no produce
evidence. Each half of the fix was then removed on its own to prove it
load-bearing.

The structural gate that stops a THIRD seam drifting the same way is the sibling
``tests/ci/test_omn17214_publish_flow_attribution_gate.py`` (AC6).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _emit_projection_terminal_event,
)
from omnibase_infra.runtime.observability import (
    active_flow_key,
    get_consumer_flow_counters,
    reset_consumer_flow_counters,
)

# The incident's own coordinates (OMN-17214 AC3).
_GROUP = "local.omnimarket.projection_consumer_flow.consume.1.0.1"
_IN_TOPIC = "onex.evt.platform.node-heartbeat.v1"  # onex-topic-allow: real topic from the OMN-17214 incident
_TERMINAL_TOPIC = "onex.evt.omnimarket.projection-consumer-flow-applied.v1"  # onex-topic-allow: real topic from the OMN-17214 incident


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
    return bus


def _source_envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object](
        payload={"node_id": str(uuid4())},
        correlation_id=uuid4(),
        envelope_timestamp=datetime.now(UTC),
        event_type="platform.node-heartbeat",
        source_tool="omn17214-test",
    )


def _deltas_by_key(window: Any) -> dict[tuple[str, str], Any]:
    assert window is not None
    return {(d.consumer_group, d.topic): d for d in window.consumer_deltas}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_projection_terminal_event_is_attributed_to_the_subscription() -> None:
    """AC3: a projection that publishes its terminal event reports out > 0.

    This is the seam AC3's named consumer uses. ``node_projection_consumer_flow``
    declares ``terminal_event: projection-consumer-flow-applied.v1`` and its
    output leaves the runtime here, never through the result applier — so with
    this seam unattributed the Phase 1 writer reports itself STALLED on every
    window in which it publishes.
    """
    bus = _bus()

    async def _emit() -> None:
        with active_flow_key(_GROUP, _IN_TOPIC):
            await _emit_projection_terminal_event(
                bus,
                _TERMINAL_TOPIC,
                _source_envelope(),
                _ProjectionResult(rows_upserted=3),
            )

    counters = get_consumer_flow_counters()
    t0 = datetime(2026, 9, 17, 9, 20, 0, tzinfo=UTC)
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)
    await _emit()
    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=30))

    bus.publish.assert_awaited_once()
    rows = _deltas_by_key(window)
    assert (_GROUP, _IN_TOPIC) in rows, (
        "the projection terminal publish attributed nothing to the in-flight "
        "subscription, so the pair emits no row from this seam at all"
    )
    assert rows[(_GROUP, _IN_TOPIC)].messages_out == 1, (
        "a projection published its declared terminal event and was still "
        f"counted as producing {rows[(_GROUP, _IN_TOPIC)].messages_out} — this is "
        "exactly the healthy-producer-reads-STALLED defect (OMN-17214 Defect B)"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_projection_terminal_event_records_upstream_production() -> None:
    """The terminal topic's own produce evidence must be recorded too.

    ``record_produced`` is what separates ``STARVED`` from ``IDLE`` for whoever
    consumes this terminal topic downstream. It is taken from our own publish
    seam and never from a broker query, so a publish that is invisible here
    leaves every downstream consumer of that topic underivable.
    """
    bus = _bus()
    counters = get_consumer_flow_counters()
    t0 = datetime(2026, 9, 17, 9, 20, 0, tzinfo=UTC)
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)
    with active_flow_key(_GROUP, _IN_TOPIC):
        await _emit_projection_terminal_event(
            bus, _TERMINAL_TOPIC, _source_envelope(), _ProjectionResult(rows_upserted=1)
        )
    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=30))

    assert window is not None
    produced = {d.topic: d.messages_produced for d in window.produce_deltas}
    assert produced.get(_TERMINAL_TOPIC) == 1, (
        f"{_TERMINAL_TOPIC} carried no produce evidence after a successful "
        f"publish; produce_deltas={produced}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_publish_outside_a_subscription_attributes_nothing() -> None:
    """Negative control: no in-flight subscription means no fabricated row.

    A boot-time republish or a sweep runs with no binding. It must record the
    topic's production and attribute ``messages_out`` to nobody — inventing a
    consumer_group here would be worse than the undercount it replaces.
    """
    bus = _bus()
    counters = get_consumer_flow_counters()
    t0 = datetime(2026, 9, 17, 9, 20, 0, tzinfo=UTC)
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)
    await _emit_projection_terminal_event(
        bus, _TERMINAL_TOPIC, _source_envelope(), _ProjectionResult(rows_upserted=1)
    )
    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=30))

    assert window is not None
    assert window.consumer_deltas == (), (
        "a publish with no in-flight subscription invented a consumer row: "
        f"{window.consumer_deltas}"
    )
    produced = {d.topic: d.messages_produced for d in window.produce_deltas}
    assert produced.get(_TERMINAL_TOPIC) == 1
