# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17214 — flow instrumentation must cover BOTH wiring branches.

The defect
----------
``_wire_contract_subscriptions`` picks one of two callback factories per
subscription. ``_is_raw_event_projection_contract`` routes every contract
declaring ``consumer_purpose: audit`` or ``consumer_purpose: projection`` to
``_make_raw_event_projection_callback``; everything else goes to
``_make_event_bus_callback``. Only the second branch was ever given a
``consumer_group``, so only the second branch registered a flow counter.

The consequence is worse than a wrong number. ``consumer_flow_counters``'s own
docstring is explicit that a MISSING row means "we do not know" while a zero row
means "alive, took nothing" — two different facts. The audit/projection branch
emitted no row at all, so the projection had nothing to materialize a gap from.
Measured on the live dev lane 2026-08-30: 57 Stable (group, topic) subscriptions
with zero flow rows, 32 of them ``*_projection_compute``, including the epic's
canonical case ``node_gateway_link_health_projection_compute`` at 33,971 in / 0
out with zero rows in ``consumer_flow_windows``.

AC5 is falsified by a test that only covers the branch that already works — the
defect is precisely that one branch was never exercised. So every assertion
below is made against BOTH factories, from the same list, in the same test.

RED proof: before the fix, ``_make_raw_event_projection_callback`` took no
``consumer_group`` parameter at all, so the parametrized cases for that branch
fail at call time (``TypeError: unexpected keyword argument 'consumer_group'``),
and the registration/throughput assertions never even get to run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_dispatch_callback,
    _make_event_bus_callback,
    _make_raw_event_projection_callback,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.observability import (
    get_consumer_flow_counters,
    reset_consumer_flow_counters,
)
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

# The incident's own topics (OMN-16755 / OMN-17214).
_IN_TOPIC = "onex.evt.omnibase-infra.gateway-heartbeat.v1"  # onex-topic-allow: real topic from the OMN-17214 incident
_OUT_TOPIC = "onex.cmd.omnibase-infra.gateway-link-health-upsert.v1"  # onex-topic-allow: real topic from the OMN-17214 incident
_ROUTE_PATTERN = "*.evt.omnibase-infra.gateway-heartbeat.*"


class _ModelIn(BaseModel):
    """Accepts both wire shapes: the enveloped payload and the raw message dump."""

    model_config = ConfigDict(frozen=True, extra="ignore")


class _ModelOut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str


class _HandlerStalled:
    """Consumes every message and publishes nothing — the OMN-16755 shape."""

    def __init__(self) -> None:
        self.seen = 0

    async def handle(self, request: _ModelIn) -> None:
        self.seen += 1


class _HandlerFlowing:
    """Consumes and emits a real output event — the control."""

    def __init__(self) -> None:
        self.seen = 0

    async def handle(self, request: _ModelIn) -> _ModelOut:
        self.seen += 1
        return _ModelOut(node_id=str(uuid4()))


@pytest.fixture(autouse=True)
def _clean_counters() -> Any:
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
            topic_pattern=_ROUTE_PATTERN,
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=dispatcher_id,
        )
    )
    engine.freeze()
    return engine


def _enveloped_message() -> MagicMock:
    """The wire shape ``_make_event_bus_callback`` reads."""
    msg = MagicMock()
    msg.value = json.dumps({"node_id": str(uuid4())}).encode("utf-8")
    return msg


def _raw_message() -> ModelEventMessage:
    """The wire shape ``_make_raw_event_projection_callback`` reads."""
    return ModelEventMessage(
        topic=_IN_TOPIC,
        value=json.dumps({"node_id": str(uuid4())}).encode("utf-8"),
        headers=ModelEventHeaders(
            source="omn17214-test",
            event_type="omnibase-infra.gateway-heartbeat",
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
        ),
    )


def _wire_event_bus_branch(
    handler: object,
    dispatcher_id: str,
    consumer_group: str | None,
) -> Callable[..., Any]:
    return _make_event_bus_callback(
        _IN_TOPIC,
        _frozen_engine_for(handler, dispatcher_id),
        DispatchResultApplier(event_bus=_bus(), output_topic=_OUT_TOPIC),
        event_bus=_bus(),
        allowed_dispatcher_ids={dispatcher_id},
        consumer_group=consumer_group,
    )


def _wire_raw_projection_branch(
    handler: object,
    dispatcher_id: str,
    consumer_group: str | None,
) -> Callable[..., Any]:
    return _make_raw_event_projection_callback(
        _IN_TOPIC,
        _frozen_engine_for(handler, dispatcher_id),
        DispatchResultApplier(event_bus=_bus(), output_topic=_OUT_TOPIC),
        allowed_dispatcher_ids={dispatcher_id},
        consumer_group=consumer_group,
    )


# The two branches the wiring can select, driven from ONE list so a branch
# cannot be silently dropped from coverage. ``_is_raw_event_projection_contract``
# sends ``consumer_purpose: audit`` and ``consumer_purpose: projection`` down the
# second one; every other subscription takes the first.
_BRANCHES: list[tuple[str, Any, Any]] = [
    ("event_bus", _wire_event_bus_branch, _enveloped_message),
    ("raw_event_projection", _wire_raw_projection_branch, _raw_message),
]
_BRANCH_IDS = [name for name, _wire, _msg in _BRANCHES]


def _rows_after_wiring_only(
    wire: Any, name: str, group: str
) -> dict[tuple[str, str], Any]:
    """Wire one subscription, hand it NO traffic, and close a window over it.

    The drained window is the observable form of ``register()``: a key declared
    at wiring time produces a row every window whether or not anything moved.
    Reading it this way rather than through an accessor keeps the assertion on
    the same surface the heartbeat actually carries.
    """
    t0 = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
    counters = get_consumer_flow_counters()
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)  # priming tick
    wire(_HandlerStalled(), f"{name}-dispatcher", group)
    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=30))
    assert window is not None
    return {(d.consumer_group, d.topic): d for d in window.consumer_deltas}


@pytest.mark.unit
@pytest.mark.parametrize(("name", "wire", "_message"), _BRANCHES, ids=_BRANCH_IDS)
def test_every_wiring_branch_registers_a_flow_counter_at_wiring_time(
    name: str,
    wire: Any,
    _message: Any,
) -> None:
    """AC5: a registered counter key must exist for EACH branch, before traffic.

    Registration at wiring time is what makes a zero row possible. Without it a
    subscription that takes nothing emits no row, and "took nothing" becomes
    indistinguishable from "not observed" — which is the exact
    audit/projection-branch failure this ticket exists to close.
    """
    group = f"local.omnibase_infra.omn17214_{name}.consume.1.0.0"

    rows = _rows_after_wiring_only(wire, name, group)

    assert (group, _IN_TOPIC) in rows, (
        f"the {name!r} wiring branch registered no flow counter for "
        f"{(group, _IN_TOPIC)} — this subscription emits NO row at all, which "
        "the projection reads as UNKNOWN rather than as observed-idle"
    )


@pytest.mark.unit
@pytest.mark.parametrize(("name", "wire", "message"), _BRANCHES, ids=_BRANCH_IDS)
def test_every_wiring_branch_emits_a_zero_row_when_it_takes_nothing(
    name: str,
    wire: Any,
    message: Any,
) -> None:
    """AC5 / AC1: an idle subscription on either branch is a ROW, not a silence."""
    t0 = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
    counters = get_consumer_flow_counters()
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)  # priming tick

    group = f"local.omnibase_infra.omn17214_idle_{name}.consume.1.0.0"
    wire(_HandlerStalled(), f"{name}-idle-dispatcher", group)

    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=30))
    assert window is not None
    rows = {(d.consumer_group, d.topic): d for d in window.consumer_deltas}
    assert (group, _IN_TOPIC) in rows, (
        f"the {name!r} branch emitted no row for an idle subscription; a missing "
        "row and a zero row are different facts and must not be conflated"
    )
    row = rows[(group, _IN_TOPIC)]
    assert row.messages_in == 0
    assert row.messages_out == 0


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "wire", "message"), _BRANCHES, ids=_BRANCH_IDS)
async def test_every_wiring_branch_counts_a_stalled_consumer_apart_from_a_flowing_one(
    name: str,
    wire: Any,
    message: Any,
) -> None:
    """AC3 for this seam: a publishing consumer reads FLOWING on EITHER branch.

    Both consumers below are Stable, both are at LAG 0 when the window closes,
    and both processed every message handed to them. The only thing separating
    them is throughput across the seam. On the audit/projection branch neither
    was measured at all before this fix.
    """
    t0 = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
    counters = get_consumer_flow_counters()
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)  # priming tick

    stalled_group = f"local.omnibase_infra.omn17214_stalled_{name}.consume.1.0.0"
    flowing_group = f"local.omnibase_infra.omn17214_flowing_{name}.consume.1.0.0"

    stalled_handler = _HandlerStalled()
    flowing_handler = _HandlerFlowing()
    stalled_cb = wire(stalled_handler, f"{name}-stalled-dispatcher", stalled_group)
    flowing_cb = wire(flowing_handler, f"{name}-flowing-dispatcher", flowing_group)

    for _ in range(3):
        await stalled_cb(message())
        await flowing_cb(message())

    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=60))
    assert window is not None
    rows = {(d.consumer_group, d.topic): d for d in window.consumer_deltas}

    assert stalled_handler.seen == 3 and flowing_handler.seen == 3, (
        f"on the {name!r} branch the handlers did not actually run — this test "
        "would then prove nothing about throughput"
    )

    stalled = rows[(stalled_group, _IN_TOPIC)]
    flowing = rows[(flowing_group, _IN_TOPIC)]
    assert stalled.messages_in == 3
    assert stalled.messages_out == 0
    assert flowing.messages_in == 3, (
        f"the {name!r} branch did not count inbound messages for a live consumer"
    )
    assert flowing.messages_out == 3, (
        f"the {name!r} branch counted a publishing consumer as producing nothing "
        "— a healthy producer reported STALLED is the alert-storm failure mode "
        "arriving from the opposite direction (OMN-14440 precedent)"
    )


@pytest.mark.unit
@pytest.mark.parametrize(("name", "wire", "message"), _BRANCHES, ids=_BRANCH_IDS)
def test_no_branch_fabricates_a_group_when_none_is_wired(
    name: str,
    wire: Any,
    message: Any,
) -> None:
    """Neither branch may invent a group id: that would attribute rows to a
    consumer that does not exist. ``None`` disables counting, it never guesses."""
    t0 = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
    counters = get_consumer_flow_counters()
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)  # priming tick

    wire(_HandlerStalled(), f"{name}-nogroup-dispatcher", None)

    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=30))
    assert window is not None
    assert window.consumer_deltas == (), (
        f"the {name!r} branch emitted a row with no consumer group wired — a "
        f"fabricated group id would attribute flow to a consumer that does not "
        f"exist: {window.consumer_deltas}"
    )
