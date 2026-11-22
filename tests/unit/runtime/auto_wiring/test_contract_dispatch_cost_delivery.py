# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The per-contract cost wrapper keeps the dispatcher's own signature (OMN-26441).

The wrapper ``attach_contract_dispatcher`` puts around every attached
contract's dispatcher took only the envelope, so the engine's signature probe
found no keyword-only ``delivery`` parameter and called a projection writer
with no ``_partition`` or ``_offset``. These tests register through the real
engine registration path and dispatch one envelope.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.dispatch.model_message_delivery_context import (
    ModelMessageDeliveryContext,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.handlers.contract_dispatch_cost import attach_contract_dispatcher
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.unit

_TOPIC = "test.projection.events.v1"
_EVENT_TYPE = "ProbeEvent"
_DISPATCHER_ID = "cost-probe"
_OWNER = "contract_dispatch_cost_probe"


def _route_and_freeze(engine: MessageDispatchEngine) -> MessageDispatchEngine:
    engine.register_route(
        ModelDispatchRoute(
            route_id="cost-probe-route",
            topic_pattern=_TOPIC,
            message_category=EnumMessageCategory.EVENT,
            handler_id=_DISPATCHER_ID,
        )
    )
    engine.freeze()
    return engine


def _register(dispatcher: Callable[..., Awaitable[None]]) -> MessageDispatchEngine:
    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id=_DISPATCHER_ID,
        dispatcher=attach_contract_dispatcher(_OWNER, dispatcher),
        category=EnumMessageCategory.EVENT,
        message_types={_EVENT_TYPE},
        owner_contract_name=_OWNER,
    )
    return _route_and_freeze(engine)


def _envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object](payload={"k": "v"}, event_type=_EVENT_TYPE)


async def test_contract_dispatch_cost_delivery_reaches_the_dispatcher() -> None:
    seen: list[ModelMessageDeliveryContext | None] = []

    async def writer(
        envelope: object, *, delivery: ModelMessageDeliveryContext | None = None
    ) -> None:
        seen.append(delivery)

    engine = _register(writer)
    delivery = ModelMessageDeliveryContext(topic=_TOPIC, partition=3, offset=16698)

    result = await engine.dispatch(_TOPIC, _envelope(), delivery=delivery)

    assert result.status is EnumDispatchStatus.SUCCESS, result
    assert engine._dispatchers[_DISPATCHER_ID].accepts_delivery is True
    assert seen == [delivery]
    assert seen[0] is not None
    assert (seen[0].partition, seen[0].offset) == (3, 16698)


async def test_contract_dispatch_cost_signature_without_delivery_gets_no_keyword() -> (
    None
):
    calls: list[tuple[object, ...]] = []

    async def envelope_only(envelope: object) -> None:
        calls.append((envelope,))

    engine = _register(envelope_only)
    delivery = ModelMessageDeliveryContext(topic=_TOPIC, partition=0, offset=1)

    result = await engine.dispatch(_TOPIC, _envelope(), delivery=delivery)

    assert result.status is EnumDispatchStatus.SUCCESS, result
    assert engine._dispatchers[_DISPATCHER_ID].accepts_delivery is False
    assert len(calls) == 1


async def test_contract_dispatch_cost_signature_keeps_context_parameter() -> None:
    received: list[object] = []

    async def effect(envelope: object, context: object) -> None:
        received.append(context)

    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id=_DISPATCHER_ID,
        dispatcher=attach_contract_dispatcher(_OWNER, effect),
        category=EnumMessageCategory.EVENT,
        message_types={_EVENT_TYPE},
        node_kind=EnumNodeKind.EFFECT,
        owner_contract_name=_OWNER,
    )
    _route_and_freeze(engine)

    result = await engine.dispatch(_TOPIC, _envelope())

    assert result.status is EnumDispatchStatus.SUCCESS, result
    assert engine._dispatchers[_DISPATCHER_ID].accepts_context is True
    assert len(received) == 1
    assert received[0] is not None
