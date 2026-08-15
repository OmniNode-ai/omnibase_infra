# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary seam test for OMN-15474 — one accepted command, one dispatch.

This is the regression test OMN-15474 acceptance criterion 5 requires: it drives
a real ingress-ACCEPTED command through the ACTUAL auto-wiring subscription
boundary (``wire_from_manifest`` -> ``event_bus.subscribe`` -> the wiring
callback -> ``MessageDispatchEngine.dispatch_scoped`` -> the result applier),
not a unit test on one handler. The live defect lives at the wiring/subscription
boundary, so a handler-level test cannot see it.

Observed live shape (OMN-15474, pod ``omninode-runtime-7765f8977f-ggjv7``,
correlation ``a4000001-0000-4000-8000-000000000001``)::

    [WIRING-CALLBACK] Deserialized envelope … topic=…delegation-request.v1
    [WIRING-CALLBACK] Dispatching to engine  … topic=…delegation-request.v1
    [WIRING-CALLBACK] Deserialized envelope … topic=…delegation-request.v1   <-- SECOND
    [WIRING-CALLBACK] Dispatching to engine  … topic=…delegation-request.v1   <-- SECOND

Two subscriptions attached to the SAME command topic inside ONE process, on two
distinct consumer groups, so the broker delivers the accepted command to both
and the whole reducer chain executes twice. Both executions carry the SAME
ingress correlation id, so the duplicate is invisible to any correlation-keyed
dedupe — the only observable is cardinality.

Seam assertions (both must hold):

1. CARDINALITY — exactly ONE command is emitted for one accepted command.
2. CORRELATION AUTHORITY — the emitted command carries the INGRESS-assigned
   correlation id byte-identical. The dispatcher is not an authority on
   correlation identity; it may never mint one. (Deepened in OMN-15546.)
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ENV_SINGLE_OWNER_COMMAND_TOPICS,
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
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.asyncio

SEAM_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "seams"
    / "dispatch_correlation"
    / "accepted_command.json"
)


# ---------------------------------------------------------------------------
# Seam doubles — faithful to the real transport contract, not to a mock's shape
# ---------------------------------------------------------------------------


@dataclass
class _KafkaRecord:
    """Minimal record shape the wiring callback consumes (``.value`` bytes)."""

    value: bytes


@dataclass
class _EmittedCommand:
    """One command the runtime emitted downstream of the accepted command."""

    correlation_id: UUID | None
    status: str
    message_type: str


@dataclass
class _RecordingResultApplier:
    """Stands in for ``DispatchResultApplier`` — records what would be published.

    The applier IS the emit point: every downstream command/event the runtime
    produces for a consumed record goes through ``apply()``. Counting applies is
    therefore counting emitted commands at the seam, without needing a broker.
    """

    emitted: list[_EmittedCommand] = field(default_factory=list)

    async def apply(self, result: object, correlation_id: UUID | None = None) -> None:
        self.emitted.append(
            _EmittedCommand(
                correlation_id=correlation_id,
                status=str(getattr(getattr(result, "status", None), "value", "?")),
                message_type=str(getattr(result, "message_type", "?")),
            )
        )


@dataclass
class _FanOutEventBus:
    """In-process bus that reproduces broker fan-out across consumer groups.

    A real broker delivers a partition record once PER consumer group. Each
    auto-wired contract computes its own group id, so N contracts subscribed to
    the same topic in one process each receive the record. This double models
    exactly that and nothing more.
    """

    subscriptions: list[tuple[str, str, Callable[..., Awaitable[None]]]] = field(
        default_factory=list
    )

    async def subscribe(
        self,
        *,
        topic: str,
        node_identity: Any,
        on_message: Callable[..., Awaitable[None]],
    ) -> Callable[[], Awaitable[None]]:
        self.subscriptions.append(
            (topic, getattr(node_identity, "node_name", "?"), on_message)
        )

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe

    async def publish(self, *args: object, **kwargs: object) -> None:
        return None

    def subscriber_count(self, topic: str) -> int:
        return sum(1 for t, _, _ in self.subscriptions if t == topic)

    async def deliver_once(self, topic: str, record: _KafkaRecord) -> None:
        """Deliver ONE accepted command to every subscribed consumer group."""
        for sub_topic, _, callback in list(self.subscriptions):
            if sub_topic == topic:
                await callback(record)


# ---------------------------------------------------------------------------
# Contract fixtures — two contracts declaring the SAME command subscribe topic
# ---------------------------------------------------------------------------


def _handler_cls(name: str) -> type:
    """A real class: ``ModelHandlerResolverContext.handler_cls`` requires ``type``."""

    ns: dict[str, object] = {}
    exec(  # noqa: S102 — builds a genuinely distinct handler type per contract
        "class _H:\n"
        "    async def handle(self, envelope):\n"
        "        return {'ok': True}\n",
        ns,
    )
    cls = ns["_H"]
    assert isinstance(cls, type)
    cls.__name__ = name
    return cls


def _contract(name: str, topic: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="omnibase-infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(topic,),
            publish_topics=("onex.evt.omnibase-infra.routing-decision.v1",),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=f"Handler{name}",
                        module=f"fake.{name}",
                    ),
                    event_model=None,
                    operation=None,
                    event_type="delegation-request",
                    topic=topic,
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# The seam test
# ---------------------------------------------------------------------------


async def _boot(
    contracts: tuple[ModelDiscoveredContract, ...],
    bus: _FanOutEventBus,
    applier: _RecordingResultApplier,
    *,
    strict: bool = True,
) -> tuple[MessageDispatchEngine, ModelOnexError | None]:
    """Run the real wiring boot; return the engine and any fail-closed refusal."""
    engine = MessageDispatchEngine()
    with (
        patch.dict(
            os.environ, {ENV_SINGLE_OWNER_COMMAND_TOPICS: "1" if strict else "0"}
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            side_effect=lambda ref, *a, **kw: _handler_cls("SeamHandler"),
        ),
    ):
        try:
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=contracts),
                engine,
                event_bus=bus,
                environment="local",
                result_appliers_by_contract={c.name: applier for c in contracts},
            )
        except ModelOnexError as exc:
            return engine, exc
    # The real boot freezes the engine before consumers may drive it; the wiring
    # callback otherwise blocks waiting for freeze.
    if not engine.is_frozen:
        engine.freeze()
    return engine, None


async def test_seam_dispatch_scope_emits_exactly_one_correlated_command() -> None:
    """One ingress-accepted command emits exactly ONE command, ingress-correlated.

    Two halves, both required:

    A. The canonical single-owner wiring emits exactly one command for one
       accepted command, and that command carries the INGRESS-assigned
       correlation id byte-identical — the dispatcher minted nothing.
    B. The live duplicate-owner wiring (OMN-15474's measured shape) is REFUSED
       before any subscription attaches, so the double execution is impossible
       rather than merely unlikely.

    Pre-fix, half B does not hold: the boot completes, two consumers attach, and
    delivering the one accepted command produces TWO SUCCESS dispatches both
    carrying ``a4000001-0000-4000-8000-000000000001`` (see the ticket's PR body
    for the recorded RED). Half A held pre-fix and still holds — it is the
    regression guard for the opposite failure (dropping the command entirely).
    """
    seam = json.loads(SEAM_FIXTURE.read_text(encoding="utf-8"))
    topic: str = seam["topic"]
    ingress_correlation_id: str = seam["ingress_correlation_id"]
    expected: int = seam["expected_emitted_command_count"]
    record = _KafkaRecord(value=json.dumps(seam["envelope"]).encode("utf-8"))

    # --- A. canonical single-owner wiring ---------------------------------
    bus = _FanOutEventBus()
    applier = _RecordingResultApplier()
    _engine, refusal = await _boot((_contract("node_delegation", topic),), bus, applier)
    assert refusal is None, f"single-owner wiring must boot, got refusal: {refusal}"
    assert bus.subscriber_count(topic) == 1

    await bus.deliver_once(topic, record)

    # A.1 CARDINALITY
    assert len(applier.emitted) == expected, (
        f"one ingress-accepted command on {topic} emitted "
        f"{len(applier.emitted)} commands, expected {expected}; emitted "
        f"correlation ids={[str(e.correlation_id) for e in applier.emitted]}"
    )

    # A.2 CORRELATION AUTHORITY IS THE INGRESS, NOT THE DISPATCHER
    emitted = applier.emitted[0]
    assert emitted.correlation_id is not None, (
        "emitted command carries no correlation id — the ingress-assigned id "
        "was dropped at the dispatch seam"
    )
    assert str(emitted.correlation_id) == ingress_correlation_id, (
        "correlation authority violated: the emitted command carries "
        f"{emitted.correlation_id!s}, not the ingress-assigned "
        f"{ingress_correlation_id}. The dispatcher must never mint a "
        "correlation id; it may only propagate the ingress one byte-identical."
    )

    # --- B. the live duplicate-owner wiring must be refused ---------------
    dup_bus = _FanOutEventBus()
    dup_applier = _RecordingResultApplier()
    _dup_engine, dup_refusal = await _boot(
        (
            _contract("node_delegation_primary", topic),
            _contract("node_delegation_effects", topic),
        ),
        dup_bus,
        dup_applier,
    )

    assert dup_refusal is not None, (
        f"OMN-15474: two contracts own command topic {topic} in one process and "
        f"the boot ACCEPTED it — {dup_bus.subscriber_count(topic)} consumers "
        "attached. Each joins its own consumer group, so the accepted command "
        "is delivered to both and the whole reducer chain executes twice under "
        "one correlation id."
    )
    assert topic in str(dup_refusal), (
        f"the refusal must name the offending command topic; got: {dup_refusal}"
    )
    assert dup_bus.subscriber_count(topic) == 0, (
        "fail-closed means refusing BEFORE the side effect: "
        f"{dup_bus.subscriber_count(topic)} subscriptions were already attached "
        "when the boot refused, so the duplicate consumers are live anyway"
    )

    # And with nothing attached, the accepted command cannot be double-executed.
    await dup_bus.deliver_once(topic, record)
    assert dup_applier.emitted == []

    # --- C. flag OFF is the documented default: warn, do not wedge the boot --
    # 8 shipped command topics violate this today (1 infra, 7 omnimarket), so a
    # hard gate would refuse the next real deploy. CLAUDE.md requires the strict
    # invariant to ship default-OFF and be flipped once those are compliant.
    warn_bus = _FanOutEventBus()
    warn_applier = _RecordingResultApplier()
    _warn_engine, warn_refusal = await _boot(
        (
            _contract("node_delegation_primary", topic),
            _contract("node_delegation_effects", topic),
        ),
        warn_bus,
        warn_applier,
        strict=False,
    )
    assert warn_refusal is None, (
        "default (flag OFF) must not refuse the boot — that is the documented "
        f"strict-mode rollout sequencing; got: {warn_refusal}"
    )
    assert warn_bus.subscriber_count(topic) == 2, (
        "flag OFF preserves today's behavior exactly (both consumers attach); "
        "this is the measured duplicate-execution exposure the flip closes"
    )


async def test_seam_command_topic_single_owner_guard_does_not_touch_event_topics() -> (
    None
):
    """Event-topic fan-out stays legal — the guard is command-scoped (OMN-15474).

    An over-broad guard would break every legitimate multi-consumer event topic,
    which is a worse outage than the bug. This pins the boundary.
    """
    event_topic = "onex.evt.omnibase-infra.routing-decision.v1"
    bus = _FanOutEventBus()
    applier = _RecordingResultApplier()

    _engine, refusal = await _boot(
        (
            _contract("node_projection_a", event_topic),
            _contract("node_projection_b", event_topic),
        ),
        bus,
        applier,
    )

    assert refusal is None, (
        "event topics are fan-out by contract; the single-owner command guard "
        f"must not reject them, but it refused: {refusal}"
    )
    assert bus.subscriber_count(event_topic) == 2
