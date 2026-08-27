# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16777 — the discriminating case, driven through the REAL dispatch seam.

The canonical failure this ticket exists to surface (OMN-16755): on the ``.201``
dev lane ``node_gateway_link_health_projection_compute`` is **Stable, members>0,
LAG 0, current-offset 15,750** — it has read every heartbeat ever produced. Its
declared output topic sits at ``LOG-END-OFFSET 0``. 15,750 messages in, **zero**
out. Every liveness signal the platform had reported it healthy, because every
one of them measured connectedness and none measured throughput.

These tests drive the ACTUAL path — a raw Kafka wire message through
``_make_event_bus_callback`` → a real frozen ``MessageDispatchEngine`` → the real
``_make_dispatch_callback`` dispatcher → a real ``DispatchResultApplier`` — and
assert the four verdict shapes are DISTINGUISHABLE at the counters:

  * consumes and publishes            -> in > 0, out > 0   (FLOWING)
  * consumes and publishes nothing    -> in > 0, out == 0  (STALLED)  <- the bug
  * registered, no traffic at all     -> in == 0, out == 0 (IDLE / STARVED,
                                        separated downstream by whether anything
                                        was produced to the topic)
  * two subscriptions on one process  -> two separate rows, never averaged

RED proof: with the `consumer_group`/`record_in`/`record_out` wiring removed from
``_make_event_bus_callback`` (and the ``_record_flow_output`` call removed from
the applier's publish loop), the stalled and flowing consumers are byte-identical
at every observable surface — which is the production defect. See the PR body for
the recorded RED run.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_dispatch_callback,
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

# The incident's own topics.
_IN_TOPIC = "onex.evt.platform.node-heartbeat.v1"  # onex-topic-allow: real topic from the OMN-16755 incident
_OUT_TOPIC = "onex.cmd.omnibase-infra.gateway-link-health-upsert.v1"  # onex-topic-allow: real topic from the OMN-16755 incident
_GROUP = "onex-dev.omnimarket.gateway-link-health-projection-compute.consume"


class _ModelHeartbeatIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    node_id: UUID


class _ModelUpsertOut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    node_id: UUID


class _HandlerStalled:
    """Consumes every message and publishes nothing — the OMN-16755 shape."""

    def __init__(self) -> None:
        self.seen = 0

    async def handle(self, request: _ModelHeartbeatIn) -> None:
        self.seen += 1


class _HandlerFlowing:
    """Consumes and emits a real output event — the control."""

    def __init__(self) -> None:
        self.seen = 0

    async def handle(self, request: _ModelHeartbeatIn) -> _ModelUpsertOut:
        self.seen += 1
        return _ModelUpsertOut(node_id=request.node_id)


@pytest.fixture(autouse=True)
def _clean_counters() -> object:
    reset_consumer_flow_counters()
    yield
    reset_consumer_flow_counters()


def _bus() -> MagicMock:
    bus = MagicMock(spec=EventBusKafka)
    bus.publish_envelope = AsyncMock()
    bus._publish_raw_to_dlq = AsyncMock(return_value=True)
    return bus


def _frozen_engine_for(handler: object, dispatcher_id: str) -> MessageDispatchEngine:
    engine = MessageDispatchEngine()
    dispatcher = _make_dispatch_callback(handler, None)  # type: ignore[arg-type]
    engine.register_dispatcher(
        dispatcher_id=dispatcher_id,
        dispatcher=dispatcher,
        category=EnumMessageCategory.EVENT,
        message_types=None,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id=f"{dispatcher_id}-route",
            topic_pattern="*.evt.platform.node-heartbeat.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=dispatcher_id,
        )
    )
    engine.freeze()
    return engine


def _kafka_message(payload: dict[str, object]) -> MagicMock:
    msg = MagicMock()
    msg.value = json.dumps(payload).encode("utf-8")
    return msg


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stalled_consumer_is_distinguishable_from_a_flowing_one() -> None:
    """15,750 in / 0 out must not look identical to 15,750 in / 15,750 out.

    This is the whole ticket. Both consumers below are Stable, both are at LAG 0
    when the window closes, both have members>0, and both processed every single
    message handed to them. The ONLY thing that separates them is throughput
    across the seam — which is exactly what nothing in the platform measured.
    """
    from datetime import UTC, datetime, timedelta

    t0 = datetime(2026, 8, 27, 12, 0, 0, tzinfo=UTC)
    counters = get_consumer_flow_counters()
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)  # priming tick

    stalled_group = _GROUP
    flowing_group = "onex-dev.omnimarket.link-health-writer.consume"

    stalled_handler = _HandlerStalled()
    flowing_handler = _HandlerFlowing()

    stalled_cb = _make_event_bus_callback(
        _IN_TOPIC,
        _frozen_engine_for(stalled_handler, "stalled-dispatcher"),
        DispatchResultApplier(event_bus=_bus(), output_topic=_OUT_TOPIC),
        event_bus=_bus(),
        allowed_dispatcher_ids={"stalled-dispatcher"},
        consumer_group=stalled_group,
    )
    flowing_cb = _make_event_bus_callback(
        _IN_TOPIC,
        _frozen_engine_for(flowing_handler, "flowing-dispatcher"),
        DispatchResultApplier(event_bus=_bus(), output_topic=_OUT_TOPIC),
        event_bus=_bus(),
        allowed_dispatcher_ids={"flowing-dispatcher"},
        consumer_group=flowing_group,
    )

    for _ in range(3):
        wire = {"node_id": str(uuid4())}
        await stalled_cb(_kafka_message(wire))
        await flowing_cb(_kafka_message(wire))

    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=60))
    assert window is not None
    rows = {(d.consumer_group, d.topic): d for d in window.consumer_deltas}

    stalled = rows[(stalled_group, _IN_TOPIC)]
    flowing = rows[(flowing_group, _IN_TOPIC)]

    assert stalled_handler.seen == 3 and flowing_handler.seen == 3, (
        "both handlers must actually have run — otherwise this test proves "
        "nothing about throughput"
    )
    assert stalled.messages_in == 3
    assert stalled.messages_out == 0, (
        "a consumer that publishes nothing reported output — the STALLED case "
        "(OMN-16755: 15,750 in, 0 out, all checks green) would stay invisible"
    )
    assert flowing.messages_in == 3
    assert flowing.messages_out == 3
    assert (stalled.messages_in, stalled.messages_out) != (
        flowing.messages_in,
        flowing.messages_out,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_legs_on_one_process_are_separate_rows_never_averaged() -> None:
    """AC3's real content: the OMN-16754 defect was one health verdict covering
    two legs, so a live outbound leg vouched for a dead inbound one. Two
    subscriptions must produce two rows."""
    from datetime import UTC, datetime, timedelta

    t0 = datetime(2026, 8, 27, 12, 0, 0, tzinfo=UTC)
    counters = get_consumer_flow_counters()
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)

    inbound_group = "onex-dev.gateway-forwarder.inbound.consume"
    outbound_group = "onex-dev.gateway-forwarder.outbound.consume"

    outbound_handler = _HandlerFlowing()
    outbound_cb = _make_event_bus_callback(
        _IN_TOPIC,
        _frozen_engine_for(outbound_handler, "outbound-dispatcher"),
        DispatchResultApplier(event_bus=_bus(), output_topic=_OUT_TOPIC),
        event_bus=_bus(),
        allowed_dispatcher_ids={"outbound-dispatcher"},
        consumer_group=outbound_group,
    )
    # The inbound leg is wired and joined but takes NOTHING — it is registered at
    # callback-construction time precisely so its silence is reportable.
    _make_event_bus_callback(
        _IN_TOPIC,
        _frozen_engine_for(_HandlerFlowing(), "inbound-dispatcher"),
        DispatchResultApplier(event_bus=_bus(), output_topic=_OUT_TOPIC),
        event_bus=_bus(),
        allowed_dispatcher_ids={"inbound-dispatcher"},
        consumer_group=inbound_group,
    )

    for _ in range(5):
        await outbound_cb(_kafka_message({"node_id": str(uuid4())}))

    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=60))
    assert window is not None
    rows = {(d.consumer_group, d.topic): d for d in window.consumer_deltas}

    assert (inbound_group, _IN_TOPIC) in rows, (
        "the silent leg produced no row at all — a leg that cannot be seen "
        "cannot be reported dead, which is the OMN-16754 failure exactly"
    )
    assert rows[(inbound_group, _IN_TOPIC)].messages_in == 0
    assert rows[(outbound_group, _IN_TOPIC)].messages_in == 5
    assert rows[(outbound_group, _IN_TOPIC)].messages_out == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_publishing_records_upstream_production_for_the_destination_topic() -> (
    None
):
    """The produce tally is what lets the projection say STARVED without ever
    asking the broker anything."""
    from datetime import UTC, datetime, timedelta

    t0 = datetime(2026, 8, 27, 12, 0, 0, tzinfo=UTC)
    counters = get_consumer_flow_counters()
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)

    cb = _make_event_bus_callback(
        _IN_TOPIC,
        _frozen_engine_for(_HandlerFlowing(), "producer-dispatcher"),
        DispatchResultApplier(event_bus=_bus(), output_topic=_OUT_TOPIC),
        event_bus=_bus(),
        allowed_dispatcher_ids={"producer-dispatcher"},
        consumer_group=_GROUP,
    )
    for _ in range(4):
        await cb(_kafka_message({"node_id": str(uuid4())}))

    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=60))
    assert window is not None
    produced = {p.topic: p.messages_produced for p in window.produce_deltas}
    assert produced.get(_OUT_TOPIC) == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_callback_wired_without_a_group_counts_nothing_rather_than_guessing() -> (
    None
):
    """No fabricated group ids. A row attributed to a consumer that does not
    exist is worse than no row."""
    from datetime import UTC, datetime, timedelta

    t0 = datetime(2026, 8, 27, 12, 0, 0, tzinfo=UTC)
    counters = get_consumer_flow_counters()
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)

    cb = _make_event_bus_callback(
        _IN_TOPIC,
        _frozen_engine_for(_HandlerFlowing(), "ungrouped-dispatcher"),
        DispatchResultApplier(event_bus=_bus(), output_topic=_OUT_TOPIC),
        event_bus=_bus(),
        allowed_dispatcher_ids={"ungrouped-dispatcher"},
    )
    await cb(_kafka_message({"node_id": str(uuid4())}))

    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=60))
    assert window is not None
    assert window.consumer_deltas == ()
