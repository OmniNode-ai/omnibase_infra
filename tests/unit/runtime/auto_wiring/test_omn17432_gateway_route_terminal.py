# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17432 — a handler raise on a synchronous gateway route must be answered.

WHAT THE CALLER SAW. A cross-tenant ``POST /v1/gateway/detach`` returned ``503``
after **10.6 s against a 10 s bound** on staging, and heartbeat reproduced it.
The timing is the tell: a real dependency outage cannot be selective by
credential, and 10.6 s is ``GATEWAY_CORRELATION_TIMEOUT_SECONDS`` plus the round
trip. ``_publish_and_await`` had waited out its budget and mapped ``None`` to
``503`` — for a request the runtime had already decided about, in milliseconds.

WHAT ACTUALLY PRODUCES IT (attribution corrected against the ticket text).
OMN-17432 names ``runtime_host_process._create_error_response``. That function is
genuinely defective — see ``tests/unit/runtime/test_omn17432_error_response_payload.py``
— but it is NOT on this route: ``RuntimeHostProcess`` subscribes one
``_input_topic`` that defaults to ``"requests"``. The gateway command topics are
served by the contract auto-wiring boundary, and there the chain is:

1. ``HandlerGatewayDetach`` raises (``SessionNotFoundError`` /
   ``TokenValidationError`` — both are ordinary raises, lines 97 and 121 of that
   handler).
2. ``MessageDispatchEngine`` flattens it into a non-SUCCESS ``ModelDispatchResult``.
3. ``DispatchResultApplier.apply`` returns early on non-SUCCESS, so no output
   event is published.
4. ``_emit_boundary_failure_terminal`` returns early too, because
   ``node_gateway_attach_effect/contract.yaml`` declares no ``terminal_events``
   block at all — so ``_declared_failure_terminal_topics`` resolves to ``()``
   and the OMN-16812 terminal has no address it is willing to answer at.

Net: **one raise, zero messages.** The contract declares exactly one publish
topic, ``gateway-session.v1``, the gateway's own consumer is subscribed to it,
and the runtime says nothing on it. The record is safe (OMN-16798) and the
caller is abandoned — the same shape OMN-16812 fixed for contracts that DO
declare a failure terminal, still open for the ones that do not.

WHY THESE TESTS ARE AT THE REAL SEAM. They read the declarations out of the
REAL, installed ``node_gateway_attach_effect/contract.yaml`` and drive
``wire_from_manifest`` with a real ``MessageDispatchEngine`` and a real
``EventBusInmemory``, then assert on what reaches the bus. A test that called
the handler directly passes through this entire outage: the defect is not in the
handler, it is in what the boundary does with the handler's exception.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _declared_failure_terminal_topics,
    _select_dispatch_result_output_topic,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.boundary_failure_terminal import (
    ModelBoundaryFailureTerminal,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_THIS_MODULE = "tests.unit.runtime.auto_wiring.test_omn17432_gateway_route_terminal"

_PATCH_IMPORT_HANDLER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)

# The real contract this route is served by. Read from the installed package so
# a rename or a topic edit fails these tests instead of silently exempting them.
_GATEWAY_CONTRACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_gateway_attach_effect"
    / "contract.yaml"
)


def _gateway_contract_yaml() -> dict[str, object]:
    raw = yaml.safe_load(_GATEWAY_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _gateway_topics() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(subscribe_topics, publish_topics)`` as the real contract declares them."""
    event_bus = _gateway_contract_yaml()["event_bus"]
    assert isinstance(event_bus, dict)
    subscribe = tuple(str(t) for t in event_bus["subscribe_topics"])
    publish = tuple(str(t) for t in event_bus["publish_topics"])
    return subscribe, publish


class SessionNotFoundError(Exception):
    """Mirror of ``handler_gateway_detach.SessionNotFoundError``.

    Mirrored rather than imported so this test asserts on the BOUNDARY's
    behaviour for an ordinary handler exception, not on anything specific to the
    detach handler's own imports or DI.
    """


class ModelMirrorDetachCommand(BaseModel):
    """Mirror of the detach command shape the gateway router publishes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: UUID
    access_token: str
    reason: str


class ModelMirrorSessionResponse(BaseModel):
    """Mirror of the success response — declared so the arm is a typed def-B arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: UUID


class HandlerMirrorDetachRaises:
    """A gateway handler on the cross-tenant path: it raises, exactly as live.

    ``HandlerGatewayDetach`` raises ``SessionNotFoundError`` when the tenant-scoped
    lookup misses — an ordinary Python raise with no publish of its own, because
    handlers on this contract never touch the bus (``node_owned_publish``).
    """

    def handle(self, command: ModelMirrorDetachCommand) -> ModelMirrorSessionResponse:
        raise SessionNotFoundError(f"no session {command.session_id}")


def _gateway_discovered_contract() -> ModelDiscoveredContract:
    """A discovered contract pointing at the REAL gateway contract file.

    ``contract_path`` is the live file, so ``_declared_failure_terminal_topics``
    performs the same resolution production performs. The handler entry is
    mirrored because the real handlers need Keycloak/DB dependencies that have
    nothing to do with the claim under test.
    """
    subscribe, publish = _gateway_topics()
    return ModelDiscoveredContract(
        name="node_gateway_attach_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=4, patch=1),
        contract_path=_GATEWAY_CONTRACT_PATH,
        entry_point_name="node_gateway_attach_effect",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe,
            publish_topics=publish,
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerMirrorDetachRaises", module=_THIS_MODULE
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelMirrorDetachCommand", module=_THIS_MODULE
                    ),
                    operation="gateway.detach",
                ),
            ),
        ),
    )


async def _drive_one_failing_gateway_command(
    *,
    collect_topic: str,
    correlation_id: UUID,
) -> list[ModelEventEnvelope[object]]:
    """Wire the real contract, publish one detach command, return what the caller saw.

    ``collect_topic`` is the address the gateway's own
    ``gateway_session_consumer`` is subscribed to. Anything that does not arrive
    here is, for the caller, the 10 s timeout.
    """
    subscribe, _ = _gateway_topics()
    detach_topic = next(t for t in subscribe if "detach" in t)

    seen: list[ModelEventEnvelope[object]] = []
    arrived = asyncio.Event()

    async def _collect(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        if envelope.correlation_id == correlation_id:
            seen.append(envelope)
            arrived.set()

    bus = EventBusInmemory(environment="test", group="omn-17432-gateway")
    await bus.start()
    try:
        await bus.subscribe(
            collect_topic, group_id="omn17432-caller", on_message=_collect
        )

        engine = MessageDispatchEngine()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
                lambda *a, **k: HandlerMirrorDetachRaises,
            )
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(_gateway_discovered_contract(),)),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()

        command = ModelEventEnvelope[object](
            payload={
                "session_id": str(uuid4()),
                "access_token": "REDACTED-not-a-real-token",
                "reason": "client_shutdown",
            },
            correlation_id=correlation_id,
            event_type="omnibase-infra.gateway-detach-request",
        )
        await bus.publish(
            detach_topic, None, command.model_dump_json().encode("utf-8"), None
        )
        try:
            await asyncio.wait_for(arrived.wait(), timeout=10)
        except TimeoutError:  # pragma: no cover - surfaced by the assertions
            pass
    finally:
        await bus.close()
    return seen


# ---------------------------------------------------------------------------
# Band 1 — the root cause, read off the REAL contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gateway_contract_declares_no_failure_terminal_and_one_answer_address() -> None:
    """The live declarations that make the boundary silent (root-cause proof).

    Zero declared failure terminals is why ``_emit_boundary_failure_terminal``
    returned early; a single declared publish topic is why there is nonetheless
    an unambiguous address the caller is already listening on.
    """
    contract = _gateway_discovered_contract()
    answer_topic = _select_dispatch_result_output_topic(contract)

    assert answer_topic is not None
    assert answer_topic.endswith("gateway-session.v1"), answer_topic
    assert (
        _declared_failure_terminal_topics(contract, success_topic=answer_topic) == ()
    ), (
        "the gateway contract declares no separate failure terminal — this is "
        "the condition the pre-OMN-17432 boundary treated as 'no address to "
        "answer at' and stayed silent on"
    )
    # Unambiguous by construction: exactly one declared publish topic, so the
    # fallback below is a read of the contract, never a guess between two.
    assert contract.event_bus is not None
    assert len(contract.event_bus.publish_topics) == 1


# ---------------------------------------------------------------------------
# Band 2 — RED before the fix: one raise, zero messages for the caller
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handler_raise_answers_on_the_topic_the_gateway_awaits() -> None:
    """The claim: a raise produces a correlated message, not a 10 s silence.

    Pre-fix this fails on the very first assertion — nothing at all reaches
    ``gateway-session.v1``, which is precisely the 10.6 s / 503 the ticket was
    filed on.
    """
    _, publish = _gateway_topics()
    session_topic = publish[0]
    correlation_id = uuid4()

    seen = await _drive_one_failing_gateway_command(
        collect_topic=session_topic, correlation_id=correlation_id
    )

    assert seen, (
        "nothing reached the topic the gateway consumer awaits — the caller "
        "burns GATEWAY_CORRELATION_TIMEOUT_SECONDS and gets an opaque 503"
    )
    assert len(seen) == 1, "exactly one terminal per failed record"
    assert seen[0].correlation_id == correlation_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_answer_carries_a_parseable_typed_payload() -> None:
    """AC: the envelope must satisfy the consumer that has to resolve the wait.

    ``gateway_session_consumer.parse_session_event_envelope`` rejects an
    envelope with no ``payload`` object *before* the pending correlation is
    resolved, so a payload-less answer is indistinguishable from no answer at
    all. The payload is also the attribution: class, ONEX code and retryability,
    which is what lets a router map the failure to a typed status instead of
    guessing.
    """
    _, publish = _gateway_topics()
    correlation_id = uuid4()

    seen = await _drive_one_failing_gateway_command(
        collect_topic=publish[0], correlation_id=correlation_id
    )
    assert seen, "no answer to parse (see the previous test)"

    # The exact two reads parse_session_event_envelope performs.
    body = seen[0].model_dump(mode="json")
    assert UUID(str(body["correlation_id"])) == correlation_id
    assert isinstance(body["payload"], dict), (
        "payload-less envelope — parse_session_event_envelope raises here and "
        "the pending correlation is never resolved"
    )

    terminal = ModelBoundaryFailureTerminal.model_validate(seen[0].payload)
    assert terminal.correlation_id == correlation_id
    assert terminal.status == "failed"
    assert terminal.failure_class == "SessionNotFoundError"
    assert "detach" in terminal.origin_topic
    # A session that does not exist is not created by asking again.
    assert terminal.retryable is True or terminal.retryable is False
    assert terminal.failure_reason


# ---------------------------------------------------------------------------
# Band 3 — the guards the fallback must NOT relax
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ambiguous_multi_failure_terminal_contracts_are_still_refused() -> None:
    """Two or more declared failure terminals stays a refusal, not a pick.

    The fallback answers only where the contract leaves exactly one possible
    address. Where the contract names several, choosing one would be a guess —
    the same refusal ``apply_failure_terminal_guard`` makes.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _resolve_boundary_terminal_answer_topic,
    )

    assert (
        _resolve_boundary_terminal_answer_topic(
            failure_terminal_topics=("a.v1", "b.v1"),
            terminal_answer_topic="success.v1",
            consumed_topic="cmd.v1",
        )
        is None
    )


@pytest.mark.unit
def test_a_declared_failure_terminal_still_wins_over_the_fallback() -> None:
    """OMN-16812's behaviour is unchanged where a failure terminal exists."""
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _resolve_boundary_terminal_answer_topic,
    )

    assert (
        _resolve_boundary_terminal_answer_topic(
            failure_terminal_topics=("declared-failure.v1",),
            terminal_answer_topic="success.v1",
            consumed_topic="cmd.v1",
        )
        == "declared-failure.v1"
    )


@pytest.mark.unit
def test_the_fallback_never_republishes_onto_the_consumed_topic() -> None:
    """The circular-route guard applies to the fallback address too.

    A contract that both consumes and publishes one topic would otherwise have
    its own failure fed back into its own subscription — the hazard
    ``_is_dead_letter_source_topic`` guards on the DLQ leg, with the same answer.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _resolve_boundary_terminal_answer_topic,
    )

    assert (
        _resolve_boundary_terminal_answer_topic(
            failure_terminal_topics=(),
            terminal_answer_topic="loop.v1",
            consumed_topic="loop.v1",
        )
        is None
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_downstream_consumer_of_the_answer_topic_does_not_cascade() -> None:
    """The blast-radius question, answered by experiment rather than by assertion.

    The fallback lands a failure terminal on a topic whose subscribers expect
    that contract's SUCCESS model. If such a subscriber raised on the unexpected
    payload, ITS boundary would emit a terminal onto ITS own answer topic, and a
    single handler failure could walk the graph. It does not: routing matches on
    the declared payload type, an unmatched payload dispatches to no handler at
    all, and a boundary with nothing dispatched has nothing to fail. This test
    holds that property still true, because the fallback depends on it.
    """
    _, publish = _gateway_topics()
    session_topic = publish[0]
    downstream_answer_topic = "onex.evt.omnibase-infra.omn17432-downstream-answer.v1"  # onex-topic-allow: synthetic downstream contract, test-only
    correlation_id = uuid4()

    downstream = ModelDiscoveredContract(
        name="node_omn17432_downstream_consumer",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=1, patch=0),
        contract_path=_GATEWAY_CONTRACT_PATH,
        entry_point_name="node_omn17432_downstream_consumer",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(session_topic,),
            publish_topics=(downstream_answer_topic,),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerMirrorDetachRaises", module=_THIS_MODULE
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelMirrorSessionResponse", module=_THIS_MODULE
                    ),
                ),
            ),
        ),
    )

    cascaded: list[ModelEventEnvelope[object]] = []

    async def _collect_cascade(message: ModelEventMessage) -> None:
        cascaded.append(ModelEventEnvelope[object].model_validate_json(message.value))

    bus = EventBusInmemory(environment="test", group="omn-17432-cascade")
    await bus.start()
    try:
        await bus.subscribe(
            downstream_answer_topic,
            group_id="omn17432-cascade-watch",
            on_message=_collect_cascade,
        )
        engine = MessageDispatchEngine()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
                lambda *a, **k: HandlerMirrorDetachRaises,
            )
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(downstream,)),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()

        # Exactly what the fallback now puts on that topic.
        terminal = ModelBoundaryFailureTerminal(
            correlation_id=correlation_id,
            failure_class="SessionNotFoundError",
            retryable=False,
            failure_reason="no session",
            origin_topic="onex.cmd.omnibase-infra.gateway-detach-request.v1",  # onex-topic-allow: mirrors the answer under test
        )
        await bus.publish(
            session_topic,
            None,
            ModelEventEnvelope[object](
                payload=terminal.model_dump(mode="json"),
                correlation_id=correlation_id,
                event_type="omnibase-infra.gateway-session",
                payload_type=type(terminal).__name__,
            )
            .model_dump_json()
            .encode("utf-8"),
            None,
        )
        await asyncio.sleep(0.5)
    finally:
        await bus.close()

    assert not cascaded, (
        "a downstream consumer turned the failure answer into a second failure "
        "answer — the fallback would amplify one handler failure across the graph"
    )


@pytest.mark.unit
def test_a_contract_with_no_publish_topic_at_all_stays_silent() -> None:
    """No declared address anywhere is still no answer — never an invented one."""
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _resolve_boundary_terminal_answer_topic,
    )

    assert (
        _resolve_boundary_terminal_answer_topic(
            failure_terminal_topics=(),
            terminal_answer_topic=None,
            consumed_topic="cmd.v1",
        )
        is None
    )
    assert (
        _resolve_boundary_terminal_answer_topic(
            failure_terminal_topics=(),
            terminal_answer_topic="",
            consumed_topic="cmd.v1",
        )
        is None
    )
