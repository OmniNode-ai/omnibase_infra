# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18918 — the coordinates reach the boundary the PROJECTION WRITERS use.

Why this module exists, and it is the whole point of it. The first cut of
this change supplied a delivery context at ONE consume boundary,
``EventBusSubcontractWiring``, and the in-process projection writers do not
arrive there. They declare ``db_tables`` and no ``consumer_purpose``, which
routes them to ``handler_wiring``'s own callbacks, and those dispatch through
``dispatch_scoped`` -- a different protocol, which the first cut left
untouched. A live subscription readback on the .201 dev lane measured the
subcontract seam handling 104 calls across four topics, none of them a
projection source.

So the change looked complete, every test passed, and on the lane it would
have injected nothing for exactly the writers the OMN-18905 defect is about,
logged the absence on every message, and left the exposures frozen. It was
caught in second-actor review (``omnibase_infra#3898`` comment 5756939835),
not by a test, because the tests all drove the seam that already worked.

These assertions drive the OTHER path: the real ``MessageDispatchEngine``,
the real ``_dispatch_to_contract_scope`` both handler_wiring callbacks call,
and the real registration lifecycle including ``freeze()``. A dispatcher
nothing routes to is never called at all, which is a way for a coordinate to
"arrive" in a test that proves nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.dispatch.model_message_delivery_context import (
    ModelMessageDeliveryContext,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _dispatch_to_contract_scope,
)
from omnibase_infra.runtime.delivery_context import delivery_context_from_message
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_TOPIC = "onex.evt.omnibase-infra.runner-fleet.v1"
_EVENT_TYPE = "runner-fleet-observation"
_DISPATCHER_ID = "omn18918-scoped-writer"


def _envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object](
        payload={"host": "observer-host.invalid", "runners": []},
        event_type=_EVENT_TYPE,
        correlation_id=uuid4(),
    )


def _engine_with_projection_writer(seen: dict[str, Any]) -> MessageDispatchEngine:
    """A frozen engine routing this topic to a writer that declares delivery."""
    engine = MessageDispatchEngine()

    async def projection_writer(
        envelope: dict[str, object],
        *,
        delivery: ModelMessageDeliveryContext | None = None,
    ) -> None:
        # Exactly what the real projection callback does with it: stamp the
        # two payload keys the writer reads to build its snapshot delta.
        seen["delivery"] = delivery
        if delivery is not None:
            seen["_partition"] = delivery.partition
            seen["_offset"] = delivery.offset

    engine.register_dispatcher(
        dispatcher_id=_DISPATCHER_ID,
        dispatcher=projection_writer,
        category=EnumMessageCategory.EVENT,
        message_types={_EVENT_TYPE},
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="omn18918-scoped-route",
            topic_pattern="onex.evt.*.runner-fleet.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=_DISPATCHER_ID,
        )
    )
    engine.freeze()
    return engine


def _record(partition: int, offset: str) -> ModelEventMessage:
    return ModelEventMessage(
        topic=_TOPIC,
        key=None,
        value=b"{}",
        headers=ModelEventHeaders(
            timestamp=datetime(2026, 9, 21, 7, 57, 8, tzinfo=UTC),
            source="test-omn18918",
            event_type=_EVENT_TYPE,
        ),
        partition=partition,
        offset=offset,
    )


async def test_the_scoped_boundary_delivers_the_coordinates_to_the_writer() -> None:
    """The assertion the first cut of this change would have failed."""
    seen: dict[str, Any] = {}
    engine = _engine_with_projection_writer(seen)

    await _dispatch_to_contract_scope(
        engine,
        _TOPIC,
        _envelope(),
        frozenset({_DISPATCHER_ID}),
        ModelMessageDeliveryContext(topic=_TOPIC, partition=3, offset=16698),
    )

    assert seen.get("_partition") == 3
    assert seen.get("_offset") == 16698


async def test_the_whole_chain_from_a_record_to_the_injected_keys() -> None:
    """Record -> shared builder -> scoped dispatch -> the two payload keys.

    The boundary builds the context from the record it still holds, which is
    the step that was missing. Driving the builder rather than hand-building
    a context is deliberate: a hand-built one would pass even if the callback
    had no record to build from, which is the defect this covers.
    """
    seen: dict[str, Any] = {}
    engine = _engine_with_projection_writer(seen)

    delivery = delivery_context_from_message(
        _record(partition=2, offset="4171"), _TOPIC
    )

    await _dispatch_to_contract_scope(
        engine, _TOPIC, _envelope(), frozenset({_DISPATCHER_ID}), delivery
    )

    assert seen.get("_partition") == 2
    assert seen.get("_offset") == 4171


async def test_two_records_deliver_two_advancing_offsets() -> None:
    """A constant coordinate is the defect; the advance is asserted."""
    seen: dict[str, Any] = {}
    engine = _engine_with_projection_writer(seen)
    offsets: list[int] = []

    for raw_offset in ("4171", "4172"):
        await _dispatch_to_contract_scope(
            engine,
            _TOPIC,
            _envelope(),
            frozenset({_DISPATCHER_ID}),
            delivery_context_from_message(_record(2, raw_offset), _TOPIC),
        )
        offsets.append(int(seen["_offset"]))

    assert offsets == [4171, 4172]


async def test_a_record_with_no_coordinates_injects_nothing_rather_than_zero() -> None:
    """The negative direction: absent must stay absent, never default to 0.

    A defaulted zero is indistinguishable downstream from a measured one, and
    that indistinguishability IS the OMN-18905 defect -- the serving cache
    refuses a delta whose offset does not exceed the cached one, so a
    constant zero froze every key on its first value at zero consumer lag.
    """
    seen: dict[str, Any] = {}
    engine = _engine_with_projection_writer(seen)

    bare = _record(2, "4171").model_copy(update={"partition": None, "offset": None})
    delivery = delivery_context_from_message(bare, _TOPIC)
    assert delivery is None

    await _dispatch_to_contract_scope(
        engine, _TOPIC, _envelope(), frozenset({_DISPATCHER_ID}), delivery
    )

    assert seen.get("delivery") is None
    assert "_partition" not in seen
    assert "_offset" not in seen


async def test_a_scoped_engine_predating_the_parameter_is_called_unchanged() -> None:
    """The consumer-compatibility half, at this protocol too.

    ``delivery`` is optional on ``ProtocolContractScopedDispatchEngine``, so
    an engine written before it is still valid and still has the older
    signature. Passing the keyword unconditionally would raise ``TypeError``
    on its first message. This is the same break that turned the first CI run
    of this change red at the other protocol.
    """
    calls: list[tuple[str, frozenset[str]]] = []

    class LegacyScopedEngine:
        async def dispatch_scoped(
            self,
            topic: str,
            envelope: ModelEventEnvelope[object],
            *,
            allowed_dispatcher_ids: frozenset[str],
        ) -> None:
            calls.append((topic, allowed_dispatcher_ids))

    await _dispatch_to_contract_scope(
        LegacyScopedEngine(),  # type: ignore[arg-type]
        _TOPIC,
        _envelope(),
        frozenset({_DISPATCHER_ID}),
        ModelMessageDeliveryContext(topic=_TOPIC, partition=3, offset=16698),
    )

    assert calls == [(_TOPIC, frozenset({_DISPATCHER_ID}))]
