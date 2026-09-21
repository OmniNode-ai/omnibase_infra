# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18918: the coordinates survive a REAL dispatch, not just a signature.

The unit tests beside this one assert each side of the seam separately: the
consume loop builds a context, the engine inspects a signature, the seam
injects two keys. Every one of them can pass while the coordinate still
fails to travel, because none of them exercises the actual path between a
consumed record and the dispatcher that reads it.

That gap is the defect's own shape. The original bug (OMN-18905) was not a
wrong value anywhere; it was two correct halves that had never been run
together, so every writer defaulted to offset 0, the serving cache refused
every delta whose offset did not exceed the cached one, and each key froze
on its first value -- at zero consumer lag, behind a green readiness
endpoint.

So this drives a real ``MessageDispatchEngine``: register a dispatcher,
dispatch a real envelope through the public entry, and assert what the
dispatcher actually received.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.dispatch.model_message_delivery_context import (
    ModelMessageDeliveryContext,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_TOPIC = "onex.evt.omnibase-infra.runner-fleet.v1"


def _envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object](
        payload={"host": "observer-host.invalid", "runners": []},
        event_type="runner-fleet-observation",
        correlation_id=uuid4(),
    )


async def _dispatch(engine: MessageDispatchEngine, **kwargs: Any) -> None:
    await engine.dispatch(_TOPIC, _envelope(), **kwargs)


async def test_a_declared_dispatcher_receives_the_real_coordinates() -> None:
    """The end-to-end claim: what the consume loop measured is what arrives."""
    engine = MessageDispatchEngine()
    seen: dict[str, Any] = {}

    async def declares_delivery(
        envelope: dict[str, object],
        *,
        delivery: ModelMessageDeliveryContext | None = None,
    ) -> None:
        seen["delivery"] = delivery

    _registered_id = "omn18918-declared"
    engine.register_dispatcher(
        dispatcher_id=_registered_id,
        dispatcher=declares_delivery,
        category=EnumMessageCategory.EVENT,
        message_types={"runner-fleet-observation"},
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="omn18918-route",
            topic_pattern="onex.evt.*.runner-fleet.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=_registered_id,
        )
    )
    # The engine refuses dispatch before freeze(), which is the registration
    # lifecycle a real runtime completes at startup. Exercising the route and
    # the freeze is part of what makes this an integration test rather than a
    # unit one: a dispatcher nothing routes to is never called at all, and
    # that is a way for a coordinate to "arrive" in a test that proves
    # nothing.
    engine.freeze()

    await _dispatch(
        engine,
        delivery=ModelMessageDeliveryContext(topic=_TOPIC, partition=3, offset=16698),
    )

    delivered = seen.get("delivery")
    assert delivered is not None, "the dispatcher was called with no delivery"
    assert (delivered.partition, delivered.offset) == (3, 16698)


async def test_a_dispatcher_that_did_not_declare_it_is_called_unchanged() -> None:
    """The property that makes the whole chain safe to land incrementally.

    Most dispatchers in this runtime never asked for a coordinate. If
    passing one to them were possible, every step of this chain would have
    been a flag day instead of an additive change.
    """
    engine = MessageDispatchEngine()
    calls: list[tuple[Any, ...]] = []

    async def plain(envelope: dict[str, object]) -> None:
        calls.append((envelope,))

    _registered_id = "omn18918-plain"
    engine.register_dispatcher(
        dispatcher_id=_registered_id,
        dispatcher=plain,
        category=EnumMessageCategory.EVENT,
        message_types={"runner-fleet-observation"},
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="omn18918-route",
            topic_pattern="onex.evt.*.runner-fleet.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=_registered_id,
        )
    )
    # The engine refuses dispatch before freeze(), which is the registration
    # lifecycle a real runtime completes at startup. Exercising the route and
    # the freeze is part of what makes this an integration test rather than a
    # unit one: a dispatcher nothing routes to is never called at all, and
    # that is a way for a coordinate to "arrive" in a test that proves
    # nothing.
    engine.freeze()

    # A delivery IS supplied; the dispatcher simply never sees it.
    await _dispatch(
        engine,
        delivery=ModelMessageDeliveryContext(topic=_TOPIC, partition=0, offset=1),
    )

    assert len(calls) == 1, "the undeclared dispatcher was not called exactly once"


async def test_an_absent_delivery_reaches_a_declared_dispatcher_as_none() -> None:
    """Fail closed, end to end: None arrives as None, never as a zero.

    A fabricated 0 here is indistinguishable downstream from a measured 0,
    and that indistinguishability is precisely what froze the board. The
    dispatcher must be able to tell "no coordinate" from "offset zero".
    """
    engine = MessageDispatchEngine()
    seen: dict[str, Any] = {"delivery": "unset"}

    async def declares_delivery(
        envelope: dict[str, object],
        *,
        delivery: ModelMessageDeliveryContext | None = None,
    ) -> None:
        seen["delivery"] = delivery

    _registered_id = "omn18918-absent"
    engine.register_dispatcher(
        dispatcher_id=_registered_id,
        dispatcher=declares_delivery,
        category=EnumMessageCategory.EVENT,
        message_types={"runner-fleet-observation"},
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="omn18918-route",
            topic_pattern="onex.evt.*.runner-fleet.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=_registered_id,
        )
    )
    # The engine refuses dispatch before freeze(), which is the registration
    # lifecycle a real runtime completes at startup. Exercising the route and
    # the freeze is part of what makes this an integration test rather than a
    # unit one: a dispatcher nothing routes to is never called at all, and
    # that is a way for a coordinate to "arrive" in a test that proves
    # nothing.
    engine.freeze()

    await _dispatch(engine)

    assert seen["delivery"] is None


async def test_a_sync_dispatcher_receives_it_too() -> None:
    """The sync path runs through Context.run, which eats keyword arguments.

    ``Context.run`` takes kwargs of its own, so a naive ``delivery=`` there
    is swallowed rather than forwarded. The implementation binds it with
    ``functools.partial`` first, and this is the test that would catch that
    going wrong -- the async paths would stay green while every sync
    dispatcher silently lost its coordinate.
    """
    engine = MessageDispatchEngine()
    seen: dict[str, Any] = {}

    def sync_declares_delivery(
        envelope: dict[str, object],
        *,
        delivery: ModelMessageDeliveryContext | None = None,
    ) -> None:
        seen["delivery"] = delivery

    _registered_id = "omn18918-sync"
    engine.register_dispatcher(
        dispatcher_id=_registered_id,
        dispatcher=sync_declares_delivery,
        category=EnumMessageCategory.EVENT,
        message_types={"runner-fleet-observation"},
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="omn18918-route",
            topic_pattern="onex.evt.*.runner-fleet.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=_registered_id,
        )
    )
    # The engine refuses dispatch before freeze(), which is the registration
    # lifecycle a real runtime completes at startup. Exercising the route and
    # the freeze is part of what makes this an integration test rather than a
    # unit one: a dispatcher nothing routes to is never called at all, and
    # that is a way for a coordinate to "arrive" in a test that proves
    # nothing.
    engine.freeze()

    await _dispatch(
        engine,
        delivery=ModelMessageDeliveryContext(topic=_TOPIC, partition=1, offset=4242),
    )

    delivered = seen.get("delivery")
    assert delivered is not None, "the sync dispatcher was called with no delivery"
    assert (delivered.partition, delivered.offset) == (1, 4242)
