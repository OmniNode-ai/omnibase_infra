# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The two publish seams that dropped the delegation chain's causal edge (OMN-18419).

What was measured
-----------------
Read-only on the .201 compose dev lane, ``public.event_ledger`` for the
chain-canary correlation ``41235987-425c-481b-b2e3-8970083ce512`` (run
35037024216)::

    onex.cmd.omnimarket.delegate-skill.v1                  6c9dc22f-…  parent=<null>
    onex.cmd.omnibase-infra.delegation-routing-request.v1  841f9c10-…  parent=<null>
    onex.evt.omnibase-infra.routing-decision.v1            223b9623-…  parent=841f9c10-…
    onex.evt.omnimarket.delegate-skill-completed.v1        3b1b8925-…  parent=6c9dc22f-…

The routing request published NO causal edge at all, and an absent
``parent_message_id`` is the checkable statement "this hop is a chain HEAD",
which it is not. Two runtime seams produced that null, and this file drives
both.

Seam 1 — the pattern-B worker command
-------------------------------------
``RuntimePatternBBroker._publish_worker_command`` builds a fresh envelope for
the route's command topic. It publishes TWO hops of this chain: the head, from
the HTTP ingress, and ``onex.cmd.omnibase-infra.delegation-request.v1``, from
``service_delegation_dispatch_port`` — which runs INSIDE the delegate-skill
handler's dispatch. Both were published as heads. The second is not one, and
that missing edge is why every hop downstream of it pointed at an envelope no
correlation-scoped read could resolve.

Seam 2 — the state_io in-row outbox
-----------------------------------
``MessageDispatchEngine`` hands a dispatcher the JSON-safe MATERIALIZATION of
the envelope and binds the typed envelope beside it on a contextvar, saying so
in its own comment: "for transport identity (for example envelope_id)". The
stateful wrapper read ``envelope_id`` as an ATTRIBUTE off that mapping, which
is always ``None``, so every outbox entry fell back to seeding its causation
with the CORRELATION id. That is provable from the wire: the routing intent's
deterministic id on the run above re-derives exactly as
``uuid5(correlation, f"{correlation}:ModelRoutingIntent:0")`` and NOT from the
``delegation-request`` envelope ``f1cc9a51-ef02-40f4-b2e7-81e5b57942ed`` that
caused it. ``_publish_outbox_batch`` then dropped even that seed from the
published envelope.

Why these tests are shaped this way
-----------------------------------
Each drives the REAL seam — a live broker across a real subscribe/publish
boundary, and the real dispatch engine invoking a real registered dispatcher —
rather than calling the changed function with a hand-built argument. Each
carries a positive control that would catch a broken fixture: the correlation
id still arrives, and the materialized payload really is the attribute-less
mapping that caused the fallback.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker

_PATTERN_B_TOPIC = "onex.cmd.omnibase-infra.pattern-b-dispatch.v1"  # onex-topic-allow: the broker's own client command topic
_RESPONSE_TOPIC = "onex.evt.pattern-b.dispatch-completed.v1"  # onex-topic-allow: the broker's own reply topic
_CONSUMED_TOPIC = "onex.cmd.test-service.consumed-command.v1"  # onex-topic-allow: a local fixture topic for the engine seam
_CONSUMED_EVENT_TYPE = "test-service.consumed-command"


def _route() -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_delegation_orchestrator",
        contract_name="delegation_orchestrator",
        command_topic="onex.cmd.omnibase-infra.delegation-request.v1",
        event_type="omnibase-infra.delegation-request",
        terminal_event="onex.evt.omnibase-infra.delegation-completed.v1",
        contract_path="/tmp/node_delegation_orchestrator/contract.yaml",  # noqa: S108
        package_name="omnimarket",
    )


async def _start_broker_with_worker(
    bus: EventBusInmemory,
    route: ModelRuntimeLocalIngressRoute,
    seen: asyncio.Queue[ModelEventMessage],
) -> RuntimePatternBBroker:
    """Start the broker and a worker that terminalises whatever it consumes."""
    broker = RuntimePatternBBroker(
        bus,
        command_topic=_PATTERN_B_TOPIC,
        routes={"delegation_orchestrator": route},
    )
    await broker.start()

    async def worker(message: ModelEventMessage) -> None:
        if seen.empty():
            await seen.put(message)
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        terminal = ModelEventEnvelope[object](
            payload={"status": "complete"},
            correlation_id=envelope.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=route.terminal_event,
            source_tool="delegation_orchestrator",
        )
        await bus.publish(
            route.terminal_event or "unknown",
            None,
            terminal.model_dump_json().encode("utf-8"),
            None,
        )

    await bus.subscribe(route.command_topic, group_id="worker", on_message=worker)
    return broker


def _command(correlation_id: UUID) -> ModelDispatchBusCommand:
    return ModelDispatchBusCommand(
        command_name="delegation_orchestrator",
        requester="delegate_skill",
        payload={"prompt": "alive"},
        correlation_id=correlation_id,
        response_topic=_RESPONSE_TOPIC,
        timeout_seconds=5,
    )


@pytest.mark.asyncio
async def test_worker_command_records_the_envelope_whose_dispatch_produced_it() -> None:
    """Seam 1, the defect. A nested worker command is not a chain head.

    This is the production shape of ``service_delegation_dispatch_port``: the
    delegate-skill handler is mid-dispatch on the ``delegate-skill`` command
    when it asks the broker for a delegation, so the engine has that envelope
    bound. Pre-fix the header and body parent assertions both fail while the
    correlation assertion beside them passes — the edge was dropped, not the
    identity.
    """
    bus = EventBusInmemory(environment="test", group="pattern-b")
    await bus.start()
    route = _route()
    seen: asyncio.Queue[ModelEventMessage] = asyncio.Queue(maxsize=1)
    broker = await _start_broker_with_worker(bus, route, seen)

    correlation_id = uuid4()
    consumed: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload={"prompt": "alive"},
        correlation_id=correlation_id,
        envelope_timestamp=datetime.now(UTC),
        event_type="omnimarket.delegate-skill",
        source_tool="pattern-b-broker",
    )

    with bind_dispatch_envelope(consumed):
        await broker.dispatch_request(_command(correlation_id))

    message = await asyncio.wait_for(seen.get(), timeout=10)
    published = ModelEventEnvelope[object].model_validate_json(message.value)

    # Positive control: the identity always arrived. A failure here is a
    # broken fixture, not a regression of the causal edge.
    assert published.correlation_id == correlation_id
    assert message.headers.correlation_id == correlation_id

    # The defect. `event_ledger.onex_headers ->> 'parent_message_id'` is
    # populated from the header, which `publish_envelope`'s identity binding
    # derives from the envelope body.
    assert published.parent_envelope_id == consumed.envelope_id
    assert message.headers.parent_message_id == consumed.envelope_id

    await broker.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_worker_command_from_the_ingress_is_still_a_chain_head() -> None:
    """Seam 1, the negative control. Nothing is invented.

    The HTTP ingress consumes no envelope, so the head hop must keep
    publishing with no ``parent_message_id`` — the statement a verifier checks
    the declared head against. A fix that stamped something here would make
    "head" unfalsifiable, which is the same class of defect as the null it
    replaces.
    """
    bus = EventBusInmemory(environment="test", group="pattern-b")
    await bus.start()
    route = _route()
    seen: asyncio.Queue[ModelEventMessage] = asyncio.Queue(maxsize=1)
    broker = await _start_broker_with_worker(bus, route, seen)

    correlation_id = uuid4()
    await broker.dispatch_request(_command(correlation_id))

    message = await asyncio.wait_for(seen.get(), timeout=10)
    published = ModelEventEnvelope[object].model_validate_json(message.value)

    assert published.correlation_id == correlation_id
    assert published.parent_envelope_id is None
    assert message.headers.parent_message_id is None

    await broker.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_causation_survives_the_engines_json_materialization() -> None:
    """Seam 2, the defect, driven through the real dispatch engine.

    The dispatcher is invoked exactly as the auto-wiring boundary's stateful
    wrapper is: with the materialized mapping. The positive control asserts
    that the mapping really is attribute-less — reproducing the fallback's
    cause — and the assertion after it is the fix.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _consumed_envelope_id,
    )

    engine = MessageDispatchEngine()
    observed: dict[str, object] = {}

    async def dispatcher(payload: object) -> None:
        observed["payload"] = payload
        observed["attribute_read"] = getattr(payload, "envelope_id", None)
        observed["resolved"] = _consumed_envelope_id(payload)

    engine.register_dispatcher(
        dispatcher_id="dispatcher-consumed",
        dispatcher=dispatcher,
        category=EnumMessageCategory.COMMAND,
        message_types={_CONSUMED_EVENT_TYPE},
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="route.consumed",
            topic_pattern="*.cmd.test-service.consumed-command.*",
            message_category=EnumMessageCategory.COMMAND,
            dispatcher_id="dispatcher-consumed",
        )
    )
    engine.freeze()

    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload={"prompt": "alive"},
        correlation_id=uuid4(),
        event_type=_CONSUMED_EVENT_TYPE,
    )
    await engine.dispatch(_CONSUMED_TOPIC, envelope)

    # Positive control: the engine really does hand over a mapping with no
    # `envelope_id` attribute. This is the line that made the old read return
    # None on every live dispatch while every unit test that passed a real
    # envelope kept working.
    assert isinstance(observed["payload"], Mapping)
    assert observed["attribute_read"] is None

    # The fix: the identity is read from the typed envelope the engine binds
    # for exactly this purpose.
    assert observed["resolved"] == envelope.envelope_id


@pytest.mark.asyncio
async def test_causation_is_none_when_nothing_was_consumed() -> None:
    """Seam 2, the negative control.

    Outside a dispatch there is no consumed envelope, and the helper must say
    so rather than reach for anything. ``None`` is what makes the outbox's
    self-seeded fallback — used by the completion-bound sweep, which genuinely
    has no causing envelope — still reachable.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _consumed_envelope_id,
    )

    assert _consumed_envelope_id(object()) is None
    assert _consumed_envelope_id({"envelope_id": "not-a-uuid"}) is None

    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload={"prompt": "alive"},
        correlation_id=uuid4(),
        event_type=_CONSUMED_EVENT_TYPE,
    )
    assert _consumed_envelope_id(envelope) == envelope.envelope_id
