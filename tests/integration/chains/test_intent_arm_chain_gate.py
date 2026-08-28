# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event Chain Gate — the IN-PROCESS INTENT-EXECUTION arm (OMN-16813).

Why this module exists, separately from ``test_event_chain_gate.py``.

OMN-16774's gate drives whole chains through the real dispatch seam and then
asserts exactly two things (``test_event_chain_gate.py:562-595``):

* a terminal event lands on the output topic, and
* nothing lands in the quarantine sink.

Both are statements about :class:`DispatchResultApplier`'s **Phase 2** — output
event publish. Neither one says anything about **Phase 1**, intent execution
(``service_dispatch_result_applier.py``), or about the :class:`IntentExecutor`
behind it.

That hole is not academic. Re-probed live on the ``.201`` dev lane on
2026-08-27, the gateway link-health chain — the surface a High-priority ticket
(OMN-16755), a status document, and the platform testing inventory
(``docs/tracking/2026-08-27-event-chain-testing-inventory.md`` §3.2, verdict
row 7) all recorded as a *dead chain* — turned out to deliver its 17,717 upserts
**entirely in-process**::

    handler -> DispatchResultApplier -> IntentExecutor -> intent effect

never over the bus. The ``onex.cmd.…gateway-link-health-upsert.v1`` topic
sitting at offset 0 was correct by design. The chain was alive; the observer was
blind. Every automated check the platform owns was blind in the same direction,
because the only leg any of them watches is the one this chain does not use.

What this suite adds that the OMN-16774 gate structurally cannot.

A chain whose handler emits intents publishes **no** terminal event and produces
**no** quarantine record. To the two assertions above it is indistinguishable
from a chain that did nothing at all — see
:func:`test_a_dead_intent_leg_is_invisible_on_every_topic_the_gate_watches`,
which proves that claim mechanically rather than asserting it in prose. So the
intent arm cannot be covered by adding a row to ``CHAIN_CASES``; it needs an
assertion aimed at the effect side of the seam, which is what this module makes.

What "real" means here. Every hop is the production object, zero infrastructure:

1. ``EventBusInmemory`` — the default local transport.
2. Raw JSON **bytes** on the entry topic, exactly the shape Kafka delivers.
3. ``EventBusSubcontractWiring.wire_subscriptions`` — the real consumer and the
   real ``_deserialize_to_envelope``.
4. ``_prepare_handler_wiring`` — the real arm selection.
5. ``MessageDispatchEngine.dispatch`` — the real routing/materialization.
6. The real handler, returning real ``ModelIntent`` objects.
7. ``DispatchResultApplier.apply`` — the real Phase-1 intent delegation.
8. ``IntentExecutor.execute_all`` — the real routing-table lookup and the real
   ``ProtocolIntentEffect`` call.

The single patched seam is ``_import_handler_class``, exactly as the OMN-16774
gate patches it, so the resolver returns this module's handler instead of
importing a package path.

Mirrors. ``omnibase_infra`` must not depend on a lane or on omnimarket, so the
link-health payload is mirrored field-shape-only. Everything the mirror touches
— wiring, engine, bus, consumer, applier, executor — is production code.

Non-goals (deliberate, see OMN-16813). This module changes no runtime behavior.
The DLQ/quarantine router is owned by OMN-16798 and the inventory's §7.2
``PermissionError``-swallow finding; it is not touched here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.contracts.subcontracts.model_event_bus_subcontract import (
    ModelEventBusSubcontract,
)
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.topic_constants import get_dlq_topic_for_original
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.handler_wiring import _prepare_handler_wiring
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_dispatch_result_applier import DispatchResultApplier
from omnibase_infra.runtime.service_intent_executor import IntentExecutor
from omnibase_spi.protocols.runtime import ProtocolDispatchEngine
from tests.helpers.application_db_topology import application_topology

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_THIS_MODULE = "tests.integration.chains.test_intent_arm_chain_gate"

# The platform quarantine sink, same constant the OMN-16774 gate watches. Named
# here too so every assertion about "the bus saw nothing" is a real measurement
# of both topics, not an assumption about one.
QUARANTINE_TOPIC = "onex.dlq.omnibase-infra.quarantine.v1"  # onex-topic-allow: asserted-empty sink, never produced to by this suite

# The intent-routing key. On a real intent this lives on the PAYLOAD, not on the
# ``ModelIntent`` envelope — ``IntentExecutor.execute`` reads
# ``payload.intent_type`` and deliberately refuses to fall back to the envelope
# field, so a payload missing it is a routing failure rather than a silent
# mis-route. Mirrored here so this suite exercises that same read.
INTENT_TYPE = "postgres.upsert_link_health"

ENTRY_TOPIC = "onex.evt.omnibase-infra.gateway-heartbeat.v1"  # onex-topic-allow: verbatim from the OMN-16755 live probe
TERMINAL_TOPIC = "onex.evt.omnibase-infra.gateway-link-health-projected.v1"  # onex-topic-allow: this suite's own asserted-empty terminal

# The boundary's OWN dead-letter sink for the entry topic, resolved by the same
# production function the consumer calls rather than hardcoded — a hardcoded
# name would keep passing after a routing change and prove nothing. Distinct
# from QUARANTINE_TOPIC above: the OMN-16774 gate watches only the latter, so
# watching both here is what makes "no sink saw anything" a measurement.
_BOUNDARY_DLQ_TOPIC = get_dlq_topic_for_original(ENTRY_TOPIC)
assert _BOUNDARY_DLQ_TOPIC is not None, (
    "the entry topic no longer resolves to a DLQ; the negative control below "
    "would be asserting against nothing."
)
BOUNDARY_DLQ_TOPIC: str = _BOUNDARY_DLQ_TOPIC

CONTRACT_YAML = f"""
name: "node_intent_arm_chain_gate"
node_type: "ORCHESTRATOR_GENERIC"
event_bus:
  subscribe_topics:
    - "{ENTRY_TOPIC}"
  publish_topics:
    - "{TERMINAL_TOPIC}"
"""


# ---------------------------------------------------------------------------
# Mirrors of the link-health intent shape
# ---------------------------------------------------------------------------


class ModelMirrorLinkHealthUpsert(BaseModel):
    """Mirror of a link-health upsert intent payload.

    ``intent_type`` is a ``Literal`` field, which is what makes this a
    structural :class:`ProtocolIntentPayload`. That protocol conformance is the
    only reason ``IntentExecutor`` can route it at all.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_type: Literal["postgres.upsert_link_health"] = "postgres.upsert_link_health"
    edge_id: str
    health_status: str


class ModelMirrorGatewayHeartbeat(BaseModel):
    """Mirror of the heartbeat event the projection folds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    edge_id: str
    health_status: str


class RecordingIntentEffect:
    """A real :class:`ProtocolIntentEffect` that records what it was handed.

    Recording rather than mocking is load-bearing for AC1: the assertion this
    suite exists to make is that the effect *ran with the right payload and the
    propagated correlation id*, and "no exception was raised" does not prove
    that — ``IntentExecutor`` has a branch that returns without raising.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[object, UUID | None]] = []

    async def execute(
        self, payload: object, *, correlation_id: UUID | None = None
    ) -> None:
        self.calls.append((payload, correlation_id))


class HandlerMirrorLinkHealthProjection:
    """Handler that folds a heartbeat into an intent and emits NOTHING else.

    This return shape — ``ModelHandlerOutput`` carrying ``intents`` and no
    ``events`` — is the only route into ``ModelDispatchResult.output_intents``
    (``handler_wiring._normalize_handler_result``: a bare ``BaseModel`` return
    becomes an output EVENT; only ``ModelHandlerOutput.intents``, or a
    ``ModelIntent`` in ``.result``, reaches the intent arm). That asymmetry is
    precisely why the OMN-16774 gate — whose rows all return bare typed models —
    cannot reach this leg no matter how many rows are appended to it.
    """

    def handle(self, event: ModelMirrorGatewayHeartbeat) -> ModelHandlerOutput[None]:
        return ModelHandlerOutput.for_orchestrator(
            input_envelope_id=uuid4(),
            correlation_id=uuid4(),
            handler_id="handler_mirror_link_health_projection",
            intents=(
                ModelIntent(
                    intent_type=INTENT_TYPE,
                    target=f"postgres://gateway_link_health/{event.edge_id}",
                    payload=ModelMirrorLinkHealthUpsert(
                        edge_id=event.edge_id,
                        health_status=event.health_status,
                    ),
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class IntentChainRun:
    """Everything one chain execution produced, on BOTH sides of the seam.

    ``terminal_messages`` / ``quarantine_messages`` are the two things the
    OMN-16774 gate can see. ``effect_calls`` is the thing it cannot.
    """

    effect_calls: list[tuple[object, UUID | None]]
    terminal_messages: list[bytes] = field(default_factory=list)
    quarantine_messages: list[bytes] = field(default_factory=list)
    boundary_dlq_messages: list[bytes] = field(default_factory=list)
    wire_correlation_id: UUID = field(default_factory=uuid4)


def _contract(contract_path: Path) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_intent_arm_chain_gate",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=1, patch=0),
        contract_path=contract_path,
        entry_point_name="node_intent_arm_chain_gate",
        package_name="omnibase-infra-chain-gate",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(ENTRY_TOPIC,),
            publish_topics=(TERMINAL_TOPIC,),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerMirrorLinkHealthProjection",
                        module=_THIS_MODULE,
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelMirrorGatewayHeartbeat",
                        module=_THIS_MODULE,
                    ),
                    operation="gateway_link_health",
                ),
            ),
        ),
    )


async def _run_intent_chain(
    tmp_path: Path,
    *,
    register_effect: bool = True,
    with_intent_executor: bool = True,
) -> IntentChainRun:
    """Drive one chain from raw wire bytes to the intent effect.

    Args:
        tmp_path: pytest tmp dir for the on-disk contract.
        register_effect: when ``False``, the executor exists but carries no
            handler for ``INTENT_TYPE`` — the "wired but never fires" shape.
        with_intent_executor: when ``False``, the applier is built with no
            executor at all.

    Returns:
        Both sides of the seam. Raises whatever the real chain raises; the
        callers that expect a refusal assert on it.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(CONTRACT_YAML, encoding="utf-8")

    contract = _contract(contract_path)
    assert contract.handler_routing is not None
    entry = contract.handler_routing.handlers[0]

    bus = EventBusInmemory(environment="chain-gate", group="chain-gate")
    await bus.start()

    effect = RecordingIntentEffect()
    run = IntentChainRun(effect_calls=effect.calls)

    async def _collect_terminal(message: object) -> None:
        run.terminal_messages.append(cast("bytes", getattr(message, "value", b"")))

    async def _collect_quarantine(message: object) -> None:
        run.quarantine_messages.append(cast("bytes", getattr(message, "value", b"")))

    async def _collect_boundary_dlq(message: object) -> None:
        run.boundary_dlq_messages.append(cast("bytes", getattr(message, "value", b"")))

    await bus.subscribe(
        TERMINAL_TOPIC,
        on_message=_collect_terminal,
        group_id="chain-gate-terminal-intent-arm",
    )
    await bus.subscribe(
        QUARANTINE_TOPIC,
        on_message=_collect_quarantine,
        group_id="chain-gate-quarantine-intent-arm",
    )
    await bus.subscribe(
        BOUNDARY_DLQ_TOPIC,
        on_message=_collect_boundary_dlq,
        group_id="chain-gate-boundary-dlq-intent-arm",
    )

    engine = MessageDispatchEngine()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerMirrorLinkHealthProjection,
    ):
        prepared = _prepare_handler_wiring(
            contract=contract,
            entry=entry,
            dispatch_engine=engine,
            resolver=ServiceHandlerResolver(),
            ownership_query=ServiceLocalHandlerOwnershipQuery(
                local_node_names=frozenset({contract.name})
            ),
            event_bus=bus,
            container=None,
            topology=application_topology(),
        )

    assert prepared.quarantine_reason is None, (
        "wiring quarantined the handler before a single message was dispatched: "
        f"{prepared.quarantine_reason} ({prepared.quarantine_detail})"
    )

    engine.register_dispatcher(
        dispatcher_id=prepared.dispatcher_id,
        dispatcher=prepared.dispatcher,
        category=prepared.category,
        message_types=prepared.message_types,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="intent-arm-route",
            topic_pattern=ENTRY_TOPIC,
            message_category=prepared.category,
            handler_id=prepared.dispatcher_id,
        )
    )
    engine.freeze()

    intent_executor: IntentExecutor | None = None
    if with_intent_executor:
        intent_executor = IntentExecutor(
            container=ModelONEXContainer(),
            effect_handlers={INTENT_TYPE: effect} if register_effect else {},
        )

    applier = DispatchResultApplier(
        event_bus=cast("ProtocolEventBusLike", bus),
        output_topic=TERMINAL_TOPIC,
        allowed_output_topics=(TERMINAL_TOPIC,),
        intent_executor=intent_executor,
    )

    wiring = EventBusSubcontractWiring(
        event_bus=cast("ProtocolEventBusSubscriber", bus),
        dispatch_engine=cast("ProtocolDispatchEngine", engine),
        environment="chain-gate",
        node_name="intent-arm-chain-gate",
        service="omnibase-infra",
        version="v1",
        result_applier=applier,
    )
    await wiring.wire_subscriptions(
        ModelEventBusSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            subscribe_topics=[ENTRY_TOPIC],
            publish_topics=[TERMINAL_TOPIC],
        ),
        "intent-arm-chain-gate",
    )

    envelope_json = {
        "payload": {"edge_id": "tenant-beta-edge-1", "health_status": "HEALTHY"},
        "event_type": ENTRY_TOPIC,
        "correlation_id": str(run.wire_correlation_id),
        "source_tool": "chain-gate",
    }
    try:
        await bus.publish(
            ENTRY_TOPIC,
            key=None,
            value=json.dumps(envelope_json).encode("utf-8"),
        )
    finally:
        await wiring.cleanup()
        await bus.close()

    return run


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


async def test_intent_arm_reaches_the_real_effect_handler(tmp_path: Path) -> None:
    """AC1 — the whole in-process leg, asserted on the effect side.

    Raw wire bytes in, a real ``ProtocolIntentEffect`` call out, with the
    handler-built payload intact and the correlation id propagated from the
    envelope. This is the assertion the OMN-16755 misdiagnosis needed and that
    nothing in any repo was making.
    """
    run = await _run_intent_chain(tmp_path)

    assert len(run.effect_calls) == 1, (
        "the intent effect was called "
        f"{len(run.effect_calls)} time(s), expected exactly 1. Zero calls is the "
        "OMN-16755 shape: the chain consumed the heartbeat, committed, and "
        "delivered nothing — with no terminal and no quarantine record to show "
        "for it."
    )

    payload, correlation_id = run.effect_calls[0]
    assert isinstance(payload, ModelMirrorLinkHealthUpsert), (
        f"the effect was handed {type(payload).__name__}, not the typed payload "
        "the handler built — the intent arm did not preserve the payload."
    )
    assert payload.edge_id == "tenant-beta-edge-1"
    assert payload.health_status == "HEALTHY"

    assert correlation_id == run.wire_correlation_id, (
        "the wire envelope's correlation id did not reach the effect "
        f"(wire={run.wire_correlation_id}, effect={correlation_id}). A "
        "regenerated correlation id makes the in-process leg untraceable from "
        "the event that caused it, which is what left OMN-16755 undiagnosable "
        "from the bus alone."
    )

    assert not run.quarantine_messages, (
        f"{len(run.quarantine_messages)} message(s) landed in {QUARANTINE_TOPIC}"
    )


async def test_a_dead_intent_leg_is_invisible_on_every_topic_the_gate_watches(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """AC2 negative control — and the blindness itself, proven mechanically.

    Runs the identical chain twice. The only difference is whether the executor
    carries a handler for ``INTENT_TYPE``; everything else — contract, handler,
    wiring, engine, wire bytes — is byte-for-byte the same.

    Live and dead runs are **identical on every topic the OMN-16774 gate
    watches**: no terminal event, nothing in the quarantine sink, and (because
    ``EventBusInmemory`` exposes no ``_publish_raw_to_dlq`` and a
    ``RuntimeHostError`` is classified retryable rather than exhausted on first
    delivery) nothing on the boundary DLQ either. The refusal is real and it is
    raised — ``IntentExecutor`` names the unroutable ``intent_type`` — but the
    in-memory bus logs the subscriber-callback failure and moves on, so it never
    reaches the publisher.

    That is the whole finding: on this leg, *delivered work* and *bus-observable
    state* are not the same measurement, and only the first one moved. The
    ``effect_calls`` assertion in
    :func:`test_intent_arm_reaches_the_real_effect_handler` is therefore the
    ONLY assertion in this repo that can go red when the intent arm breaks —
    which is what makes it non-vacuous, and what this test exists to establish.

    If this ever fails because the two runs diverge on a watched topic, that is
    good news: something now surfaces a dead intent leg on the bus. Re-read the
    gate before deleting the test.
    """
    live = await _run_intent_chain(tmp_path / "live", register_effect=True)

    with caplog.at_level(logging.ERROR):
        dead = await _run_intent_chain(tmp_path / "dead", register_effect=False)

    # 1. The assertion that moves: the effect ran in one case and not the other.
    assert len(live.effect_calls) == 1
    assert dead.effect_calls == [], (
        "the effect ran with no handler registered for its intent_type — the "
        "negative control is not controlling anything."
    )

    # 2. Everything the OMN-16774 gate can see is identical across the two runs.
    assert live.terminal_messages == dead.terminal_messages == [], (
        "an intent-only chain published a terminal event; the premise of this "
        "module (that the intent arm is invisible to a terminal-only assertion) "
        "no longer holds and the OMN-16774 gate may now cover this leg."
    )
    assert live.quarantine_messages == dead.quarantine_messages == [], (
        f"{QUARANTINE_TOPIC} — the only sink the OMN-16774 gate watches — is no "
        "longer silent on this chain. Re-read the gate."
    )
    assert live.boundary_dlq_messages == dead.boundary_dlq_messages == [], (
        f"{BOUNDARY_DLQ_TOPIC} received a record. On this transport it cannot: "
        "EventBusInmemory exposes no _publish_raw_to_dlq and the refusal is "
        "classified retryable, not exhausted. If that changed, the dead leg now "
        "HAS a bus-observable signal and this module should assert on it."
    )

    # 3. The refusal did happen, and named the leg. This is the one place a dead
    #    intent leg surfaces at all on this transport, and it is a log line —
    #    which is exactly why it went unnoticed for a day on the live lane.
    refusals = [
        record.getMessage()
        for record in caplog.records
        if "No effect handler registered" in record.getMessage()
    ]
    assert refusals, (
        "the executor did not refuse an unroutable intent at all — the dead leg "
        "is not merely bus-invisible, it is entirely silent, which is worse "
        "than what this module documents."
    )
    assert any(INTENT_TYPE in message for message in refusals), (
        f"the refusal did not name the unroutable intent_type: {refusals!r}"
    )


async def test_missing_intent_executor_refuses_rather_than_publishing_a_terminal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """AC2 negative control, second direction — no executor wired at all.

    ``DispatchResultApplier`` must refuse in Phase 1 rather than fall through to
    Phase 2, because falling through would publish a terminal event advertising
    a write that never happened. That is strictly worse than a dead leg: it is a
    dead leg that reports success.

    The load-bearing assertion is the empty terminal topic. The log assertion
    only identifies *which* refusal fired.
    """
    with caplog.at_level(logging.ERROR):
        run = await _run_intent_chain(tmp_path, with_intent_executor=False)

    assert run.effect_calls == []
    assert run.terminal_messages == [], (
        "a terminal event was published for a dispatch whose intents were never "
        "executed — the chain advertised a write that did not happen."
    )

    refusals = [
        record.getMessage()
        for record in caplog.records
        if "no IntentExecutor is configured" in record.getMessage()
    ]
    assert refusals, (
        "the applier did not refuse a result carrying intents with no executor "
        f"wired: {[r.getMessage() for r in caplog.records]!r}"
    )


async def test_validation_is_what_keeps_the_executors_silent_drop_unreachable() -> None:
    """AC3 — the one branch of ``IntentExecutor`` that does NOT refuse.

    ``IntentExecutor.execute`` answers a ``None`` payload with a WARNING log and
    a bare ``return`` — no raise, no DLQ, no record. If that branch were
    reachable, ``DispatchResultApplier`` would proceed to Phase 2 and the offset
    would commit over a silently dropped write: the §7.3 "green over an
    unexercised leg" shape at the intent altitude.

    It is not reachable from validated data, and this test is the proof, pinned
    so the guarantee cannot be relaxed by a model edit without going red. The
    branch stays as defence-in-depth; the reason it is *safe* to leave it silent
    is this refusal, and that reason is now asserted rather than assumed.
    """
    with pytest.raises(ValidationError):
        ModelIntent(
            intent_type=INTENT_TYPE,
            target="postgres://gateway_link_health/edge-1",
            payload=None,
        )
