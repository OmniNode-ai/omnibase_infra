# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The chain's head hop must publish under the chain's own identity (OMN-18389).

The defect
----------
`RuntimePatternBBroker` publishes hop 0 of the delegation chain — the
`onex.cmd.omnimarket.delegate-skill.v1` command — and it was the ONE hop in the
declared `chain_topology` that did not go through `publish_envelope`. It called
`publish(topic, key, value, None)`, and both concrete buses mint a fresh
`ModelEventHeaders` when handed none, whose `correlation_id` and `message_id`
each default to `uuid4()`.

`event_ledger` — the relation every chain replay reads back — populates both of
those columns from the HEADER and never from the envelope body
(`HandlerLedgerProjection`). So the body was right and the readback was wrong,
which is why this survived: every in-process wait on this path matches on the
decoded body (`service_pattern_b_broker._extract_direct_terminal_correlation_id`,
`port_runtime_delegation_dispatch`), so the delegation terminalised green while
the row recording it carried an unrelated identity.

Measured on the .201 compose dev lane, read from `public.event_ledger`:

    onex.cmd.omnimarket.delegate-skill.v1                  bc0553d0-…
    onex.cmd.omnibase-infra.delegation-routing-request.v1  cb9ac679-…
    onex.evt.omnibase-infra.routing-decision.v1            cb9ac679-…
    onex.evt.omnimarket.delegate-skill-completed.v1        cb9ac679-…

`cb9ac679-…` is the caller-supplied id — every later hop inherits it because
`DispatchResultApplier` derives its envelopes from the consumed one. `bc0553d0-…`
is the bus-minted header on the head. A correlation-scoped read for the
delegation's own identity therefore found four hops of five, and the chain
canary reported `ledger_chain_incomplete` on a chain that had run end to end.
This is the OMN-16931 shape — the identity replaced mid-chain — one seam further
out than OMN-16931 looked.

Why these tests are shaped this way
-----------------------------------
Both drive the REAL broker across the REAL subscribe/publish boundary
(`broker.start()` → `_handle_command_message` → `_publish_worker_command`) over
`EventBusInmemory`, which mints the same default headers Kafka does
(`event_bus_inmemory.py:206`). Nothing about the identity binding is stubbed.

Each carries its own positive control: the assertion that the envelope BODY
already agreed. Pre-fix the body assertion passes and the header assertion
fails, which is the whole defect stated as two lines of one test — a body/header
split, not a lost correlation id. A test that only asserted the header could not
tell a real regression from a broken fixture.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.topic_constants import derive_event_type_alias_for_topic
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker

# The directory placement applies the `integration` marker. It is the right
# home: these drive the REAL subscribe/publish boundary of a live broker
# service rather than calling a function, which is the same shape
# `test_tenant_dimension_seam_omn16831.py` uses for the tenant dimension. No
# external service is required -- `EventBusInmemory` mints the same default
# headers Kafka does (`event_bus_inmemory.py:206`), which is the behaviour
# under test.

_PATTERN_B_TOPIC = "onex.cmd.omnibase-infra.pattern-b-dispatch.v1"  # onex-topic-allow: the topic whose alias is derived below
_PATTERN_B_EVENT_TYPE = derive_event_type_alias_for_topic(_PATTERN_B_TOPIC)
_RESPONSE_TOPIC = "onex.evt.pattern-b.dispatch-completed.v1"  # onex-topic-allow: the broker's own reply topic, asserted on below


def _route() -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_session_orchestrator",
        contract_name="session_orchestrator",
        command_topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
        event_type="omnimarket.session-orchestrator-start",
        terminal_event="onex.evt.omnimarket.session-orchestrator-completed.v1",
        contract_path="/tmp/node_session_orchestrator/contract.yaml",  # noqa: S108
        package_name="omnimarket",
    )


async def _submit(bus: EventBusInmemory, command: ModelDispatchBusCommand) -> None:
    """Publish the pattern-B dispatch command the broker consumes."""
    envelope = ModelEventEnvelope[ModelDispatchBusCommand](
        payload=command,
        correlation_id=command.correlation_id,
        envelope_timestamp=datetime.now(UTC),
        event_type=_PATTERN_B_EVENT_TYPE,
        source_tool="codex",
    )
    await bus.publish(
        _PATTERN_B_TOPIC,
        None,
        envelope.model_dump_json().encode("utf-8"),
        None,
    )


@pytest.mark.asyncio
async def test_head_hop_wire_header_carries_the_chains_own_correlation_id() -> None:
    """Hop 0's HEADER identity must be the delegation's, not a fresh uuid4.

    Pre-fix this fails on the header assertion and passes on the body
    assertion beside it, which is the defect exactly: a body/header split on
    the one hop that did not publish through `publish_envelope`.
    """
    bus = EventBusInmemory(environment="test", group="pattern-b")
    await bus.start()
    route = _route()
    broker = RuntimePatternBBroker(
        bus,
        command_topic=_PATTERN_B_TOPIC,
        routes={"session_orchestrator": route},
    )
    await broker.start()

    seen: asyncio.Queue[ModelEventMessage] = asyncio.Queue(maxsize=1)

    async def worker(message: ModelEventMessage) -> None:
        if seen.empty():
            await seen.put(message)
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        terminal = ModelEventEnvelope[object](
            payload={"status": "complete"},
            correlation_id=envelope.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=route.terminal_event,
            source_tool="session_orchestrator",
        )
        await bus.publish(
            route.terminal_event or "unknown",
            None,
            terminal.model_dump_json().encode("utf-8"),
            None,
        )

    await bus.subscribe(route.command_topic, group_id="worker", on_message=worker)

    command = ModelDispatchBusCommand(
        command_name="session_orchestrator",
        requester="codex",
        payload={"dry_run": True},
        response_topic=_RESPONSE_TOPIC,
        timeout_seconds=1,
    )
    await _submit(bus, command)

    message = await asyncio.wait_for(seen.get(), timeout=5)
    published = ModelEventEnvelope[object].model_validate_json(message.value)

    # Positive control. The body carried the caller's correlation id before
    # this fix and still does; an assertion that fails here means the fixture
    # is broken rather than the header binding.
    assert published.correlation_id == command.correlation_id

    # The defect. `event_ledger.correlation_id` is populated from this field.
    assert message.headers.correlation_id == command.correlation_id

    # The same binding covers per-hop identity, which `event_ledger.envelope_id`
    # is populated from and which the chain replay re-derives every causal edge
    # against (`chain_replay._replay_one_hop`).
    assert message.headers.message_id == published.envelope_id

    # Absence is preserved, never invented: hop 0 is a chain HEAD and must say
    # so rather than pointing at a parent it does not have.
    assert message.headers.parent_message_id is None

    await broker.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_broker_terminal_result_header_carries_the_same_correlation_id() -> None:
    """The broker's own reply publishes under the chain's identity too.

    Same one-line defect, same file, two functions apart
    (`_publish_terminal_result`). Its topic is not in today's `chain_topology`,
    so it is invisible on the ledger surface right now — which is a reason to
    fix it in the same change, not a reason to leave it.
    """
    bus = EventBusInmemory(environment="test", group="pattern-b")
    await bus.start()
    route = _route()
    broker = RuntimePatternBBroker(
        bus,
        command_topic=_PATTERN_B_TOPIC,
        routes={"session_orchestrator": route},
    )
    await broker.start()

    async def worker(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        terminal = ModelEventEnvelope[object](
            payload={"status": "complete"},
            correlation_id=envelope.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=route.terminal_event,
            source_tool="session_orchestrator",
        )
        await bus.publish(
            route.terminal_event or "unknown",
            None,
            terminal.model_dump_json().encode("utf-8"),
            None,
        )

    await bus.subscribe(route.command_topic, group_id="worker", on_message=worker)

    replies: asyncio.Queue[ModelEventMessage] = asyncio.Queue(maxsize=1)

    async def collector(message: ModelEventMessage) -> None:
        if replies.empty():
            await replies.put(message)

    await bus.subscribe(
        _RESPONSE_TOPIC, group_id=f"collector-{uuid4()}", on_message=collector
    )

    command = ModelDispatchBusCommand(
        command_name="session_orchestrator",
        requester="codex",
        payload={"dry_run": True},
        response_topic=_RESPONSE_TOPIC,
        timeout_seconds=2,
    )
    await _submit(bus, command)

    message = await asyncio.wait_for(replies.get(), timeout=5)
    published = ModelEventEnvelope[object].model_validate_json(message.value)

    # Positive control, then the defect.
    assert published.correlation_id == command.correlation_id
    assert message.headers.correlation_id == command.correlation_id
    assert message.headers.message_id == published.envelope_id

    await broker.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_header_identity_is_never_invented_when_the_envelope_has_none() -> None:
    """The negative control: nothing is defaulted in.

    `header_identity_fields_from_envelope` contributes a key only when the
    envelope actually carries that value. An object that is not an envelope
    contributes nothing and keeps `ModelEventHeaders`' own minting behaviour,
    so a non-envelope payload is unaffected by this change.
    """
    from omnibase_infra.event_bus.envelope_header_identity import (
        header_identity_fields_from_envelope,
    )

    assert header_identity_fields_from_envelope(object()) == {}

    envelope = ModelEventEnvelope[object](
        payload={"a": 1},
        correlation_id=uuid4(),
        envelope_timestamp=datetime.now(UTC),
        event_type="omnimarket.session-orchestrator-start",
        source_tool="pattern-b-broker",
    )
    fields = header_identity_fields_from_envelope(envelope)
    assert fields["correlation_id"] == envelope.correlation_id
    assert isinstance(fields["message_id"], UUID)
    assert "parent_message_id" not in fields


# ---------------------------------------------------------------------------
# OMN-18958: the CONSUMER half of the same seam.
#
# OMN-18389 above fixed hop 0's identity on the PUBLISH side. This is the
# other end of the same wire: what the consume boundary does with that
# identity when the body it receives is a bare contract model rather than an
# envelope, which is exactly what `RuntimeLocal` publishes for a chain head.
#
# Pre-fix, the boundary synthesized an envelope and minted a SECOND id. Every
# hop caused by the message then recorded that mint as its parent, so the
# causal edge named an envelope that was never on the wire. Measured on the
# .201 dev lane: head recorded `70982614-...`, both children cited
# `53ffd454-...`, and `event_ledger` contains no such row.
#
# This drives a real publish through `EventBusInmemory` rather than handing
# the boundary a constructed message, so the id being adopted is the id the
# BUS actually put on the wire. A unit test cannot show that the two seams
# agree; only a roundtrip can.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consumer_adopts_the_wire_identity_the_publisher_minted() -> None:
    """The envelope handed to dispatch must carry the wire's own message_id.

    The two assertions are one claim from both ends: whatever id the bus put
    on the wire for a headerless publish is the id the boundary dispatches
    under. A mint on either side breaks every downstream causal edge.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _make_event_bus_callback,
    )

    topic = "onex.cmd.omnimarket.delegate-skill.v1"
    correlation = uuid4()

    bus = EventBusInmemory(environment="test", group="omn18958")
    await bus.start()

    wire: asyncio.Queue[ModelEventMessage] = asyncio.Queue(maxsize=1)
    dispatched: asyncio.Queue[object] = asyncio.Queue(maxsize=1)

    class _StopAfterCaptureError(Exception):
        """Ends the boundary run once the envelope is in hand."""

    async def _dispatch_scoped(*args: object, **kwargs: object) -> object:
        for candidate in args:
            if hasattr(candidate, "envelope_id") and dispatched.empty():
                await dispatched.put(candidate)
        raise _StopAfterCaptureError

    class _Engine:
        async def dispatch_scoped(self, *args: object, **kwargs: object) -> object:
            return await _dispatch_scoped(*args, **kwargs)

    callback = _make_event_bus_callback(
        topic,
        _Engine(),  # type: ignore[arg-type]
        result_applier=None,  # type: ignore[arg-type]
        allowed_dispatcher_ids={"omn18958-test-dispatcher"},
    )

    async def _observe(message: ModelEventMessage) -> None:
        if wire.empty():
            await wire.put(message)
        await callback(message)

    await bus.subscribe(topic, group_id="omn18958", on_message=_observe)

    # A BARE contract input model, headerless — the shape `RuntimeLocal`
    # publishes for a chain head. It states a correlation id and no identity,
    # which is what forces the boundary down the synthesis arm.
    body = {"correlation_id": str(correlation), "prompt_text": "Say ping."}
    await bus.publish(topic, None, json.dumps(body).encode("utf-8"), None)

    observed_message = await asyncio.wait_for(wire.get(), timeout=10)
    observed_envelope = await asyncio.wait_for(dispatched.get(), timeout=10)
    await bus.close()

    wire_message_id = observed_message.headers.message_id
    assert wire_message_id is not None, (
        "the bus put no message_id on the wire, so this test cannot say "
        "whether the boundary adopted one; the fixture is wrong, not the code"
    )
    assert observed_envelope.envelope_id == wire_message_id, (
        f"the boundary dispatched under {observed_envelope.envelope_id!r} for a "
        f"message the bus published as {wire_message_id!r}. Every hop caused by "
        "this one records the dispatched id as its parent, and that id is on no "
        "topic and in no table, so the causal edge cannot close (OMN-18958)"
    )
    # The correlation half, already fixed, must still hold — a change that
    # repaired identity by dropping lineage would be no improvement.
    assert observed_envelope.correlation_id == correlation
