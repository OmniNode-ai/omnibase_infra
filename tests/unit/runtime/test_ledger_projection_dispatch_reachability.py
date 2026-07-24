# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatch-reachability regression for the audit-ledger projection (OMN-14516).

``event_ledger`` held ZERO rows since ``node_ledger_projection_compute`` shipped,
while its 7 subscribed topics carried 1M+ live events and the consumer group
``onex-ledger-projection-compute`` never existed. Nothing was red: the contract
parsed, its routing entry resolved, and CI was green.

Root cause (the REV-2 mechanism, not the ``_missing_handle`` folklore): the
contract declares ``consumer_purpose: "audit"``, so ``handler_wiring`` treats it
as a *raw event projection* that must have a result applier. The kernel resolved
that applier from a **hand-maintained by-NAME allowlist** in ``service_kernel`` —
and nobody remembered to add this contract's name, so auto-wiring returned
``SKIPPED`` (which is not counted by ``total_failed``) and no consumer was ever
created. The runtime booted green over a dead node.

The fix (OMN-14516) removes the by-name allowlist entirely. An audit/projection
consumer now DECLARES ``intent_consumption.intent_routing_table`` in its own
contract, and the kernel DERIVES the applier from it — no name lookup. Two
regression guards live here:

1. A raw audit/projection contract with NO applier now reaches ``FAILED`` (not
   ``SKIPPED``), so ``total_failed`` is accurate and ``ONEX_WIRING_STRICT_MODE``
   crashes boot instead of sailing past.
2. The ledger contract's declared routing table RESOLVES to a real, importable,
   constructable write-effect handler — which is exactly what the kernel
   derivation consumes. This proves the derivation covers the ledger contract
   WITHOUT any allowlist entry.

These tests drive the REAL seam — the actual contract files through the actual
discovery + wiring path — rather than asserting against hand-built fixtures.
"""

from __future__ import annotations

import ast
import base64
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_ledger_projection_compute.handlers.handler_ledger_projection import (
    HandlerLedgerProjection,
)
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts,
    discover_contracts_from_paths,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _import_handler_class,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_intent_routing_loader import (
    load_intent_routing_table,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NODES_ROOT = _REPO_ROOT / "src" / "omnibase_infra" / "nodes"

CONTRACT_PATH = _NODES_ROOT / "node_ledger_projection_compute" / "contract.yaml"
CONTRACT_NAME = "node_ledger_projection_compute"

# EMPTY as of OMN-14524: node_validation_ledger_projection_compute now has a
# write-effect node (node_validation_ledger_write_effect / HandlerValidationLedgerAppend)
# and its handle() emits a ModelIntent, so it declares
# intent_consumption.intent_routing_table like every other derived projection. This
# is a SHRINK-ONLY ratchet — a node stays here only while genuinely
# dead-in-production with a ticket; never add a live node to silence this gate.
KNOWN_UNWIRED: frozenset[str] = frozenset()


class _StubResultApplier:
    """Stand-in for the kernel's DispatchResultApplier.

    Only its PRESENCE in ``result_appliers_by_contract`` is load-bearing for
    wiring — that is precisely the bit that was missing in production.
    """

    async def apply(self, *args: object, **kwargs: object) -> None:
        return None


async def _wire(*, with_applier: bool) -> tuple[EnumWiringOutcome, int, str]:
    """Run the real discovery + wiring path over the real ledger contract file."""
    discovered = discover_contracts_from_paths([CONTRACT_PATH])
    contracts = getattr(discovered, "contracts", discovered)
    manifest = ModelAutoWiringManifest(contracts=tuple(contracts))

    report = await wire_from_manifest(
        manifest,
        MessageDispatchEngine(),
        event_bus=None,
        environment="dev",
        container=ModelONEXContainer(),
        subscribe_immediately=False,
        result_appliers_by_contract=(
            {CONTRACT_NAME: _StubResultApplier()} if with_applier else None
        ),
    )
    result = next(r for r in report.results if r.contract_name == CONTRACT_NAME)
    return result.outcome, len(result.wirings), result.reason or ""


@pytest.mark.asyncio
async def test_ledger_projection_wires_when_result_applier_registered() -> None:
    """The contract must reach WIRED — SKIPPED/FAILED means no consumer, zero rows.

    A gate that only checks "the routing entry parses" or "the handler class
    exists" stays GREEN on the dead node: the entry parses, all 7 topics get
    assigned, and the contract is still never wired.

    OMN-14594: the contract declares 7 topic_match entries (one per topic,
    each type-scoped + correctly categorized — see the contract's own
    comment) instead of 1 operation_match entry spanning all 7. OMN-15006
    widened subscribe_topics by 18 business command/completion/DLQ topics
    (same one-entry-per-topic pattern), so wiring_count is 25, one per
    topic-scoped dispatcher.
    """
    outcome, wiring_count, reason = await _wire(with_applier=True)

    assert outcome is EnumWiringOutcome.WIRED, (
        f"node_ledger_projection_compute did not wire (outcome={outcome}, "
        f"reason={reason!r}). A raw audit/projection contract that does not reach "
        "WIRED never creates a consumer, so event_ledger stays empty."
    )
    assert wiring_count == 25, (
        f"expected 25 wired handlers (1 per topic), got {wiring_count}"
    )


@pytest.mark.asyncio
async def test_ledger_projection_FAILS_without_result_applier() -> None:
    """RED-on-pre-fix proof: a raw projection with no applier is FAILED, not SKIPPED.

    This is the load-bearing SKIPPED→FAILED assertion (OMN-14516/OMN-14530). On the
    pre-fix tree this returned SKIPPED — which ``total_failed`` does not count, so
    the runtime booted green over the dead ledger while 1M+ events flowed. SKIPPED
    would make this assertion RED; FAILED is the only outcome that lets
    ``ONEX_WIRING_STRICT_MODE`` (which reads ``total_failed``) crash the boot.

    If a future change makes a raw projection wire WITHOUT an effect path, that is
    the OTHER half of the bug (consume offsets, drop intents) and this test must be
    revisited deliberately — not deleted.
    """
    outcome, wiring_count, reason = await _wire(with_applier=False)

    assert outcome is EnumWiringOutcome.FAILED, (
        f"expected FAILED (fail-closed), got {outcome}. A SKIPPED raw projection is "
        "invisible to total_failed and to ONEX_WIRING_STRICT_MODE — the exact hole "
        "that kept event_ledger at zero rows."
    )
    assert wiring_count == 0
    assert "no result applier is wired" in reason


def _resolve_routing_to_handler_classes(
    projection_contract_path: Path,
    contracts_by_name: dict[str, object],
) -> dict[str, type]:
    """Mirror the kernel derivation: routing table -> effect handler class.

    Replicates the exact resolution ``service_kernel`` performs at boot so the
    test proves the derivation SUCCEEDS for a contract WITHOUT any allowlist:
    read ``intent_consumption.intent_routing_table``; for each
    ``intent_type -> effect_node``, find the effect node's contract, match the
    handler whose ``operation == intent_type``, and import the class.
    """
    routing = load_intent_routing_table(projection_contract_path)
    resolved: dict[str, type] = {}
    for intent_type, effect_node in routing.items():
        effect_contract = contracts_by_name.get(effect_node)
        assert effect_contract is not None, (
            f"routing table names effect node {effect_node!r} absent from manifest"
        )
        handler_routing = getattr(effect_contract, "handler_routing", None)
        assert handler_routing is not None, (
            f"effect node {effect_node!r} declares no handler_routing"
        )
        handler_ref = next(
            (
                entry.handler
                for entry in handler_routing.handlers
                if entry.operation == intent_type
            ),
            None,
        )
        if handler_ref is None and len(handler_routing.handlers) == 1:
            handler_ref = handler_routing.handlers[0].handler
        assert handler_ref is not None, (
            f"effect node {effect_node!r} has no handler for operation {intent_type!r}"
        )
        resolved[intent_type] = _import_handler_class(
            handler_ref.module, handler_ref.name
        )
    return resolved


def test_ledger_projection_routing_resolves_without_allowlist() -> None:
    """The ledger contract's routing table alone yields a constructable handler.

    This is the derivation proof the adversarial reviewer asked for: with NO
    kernel allowlist entry, the contract's own
    ``intent_consumption.intent_routing_table`` resolves ``ledger.append`` to a
    real, importable, constructable ``HandlerLedgerAppend`` exposing ``handle()``.
    If this resolves, the kernel derivation wires the ledger — by declaration, not
    by name.
    """
    manifest = discover_contracts()
    contracts_by_name = {c.name: c for c in manifest.contracts}

    resolved = _resolve_routing_to_handler_classes(CONTRACT_PATH, contracts_by_name)

    assert "ledger.append" in resolved, (
        "ledger projection contract must declare intent_consumption."
        "intent_routing_table with a 'ledger.append' -> effect-node entry so the "
        "kernel can DERIVE its result applier (no by-name allowlist)."
    )
    handler_cls = resolved["ledger.append"]
    # Constructs with the same (container, dsn) shape the kernel derivation uses.
    instance = handler_cls(ModelONEXContainer(), "")
    assert callable(getattr(instance, "handle", None)), (
        f"{handler_cls.__name__} must expose handle() — the canonical surface the "
        "generic IntentEffectDispatchBridge invokes."
    )


def test_kernel_has_no_byname_projection_allowlist() -> None:
    """The hand-maintained by-NAME result-applier allowlist is DELETED.

    Adversarial guard for the operator ruling ("Why do we have an external
    allowlist? This is bullshit"). Parses the AST of ``service_kernel`` for every
    ``auto_wiring_result_appliers["<literal>"] = ...`` assignment and asserts NONE
    of the derived projection consumers are wired by a string-literal name — they
    must be reached only through the generic derivation loop
    (``auto_wiring_result_appliers[_contract.name] = ...``).
    """
    source = _REPO_ROOT / "src" / "omnibase_infra" / "runtime" / "service_kernel.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    byname: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "auto_wiring_result_appliers"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                byname.add(target.slice.value)

    regressed = byname & {
        "node_ledger_projection_compute",
        "node_build_loop_projection_compute",
        "node_pr_state_projection_compute",
    }
    assert not regressed, (
        "service_kernel re-introduced a by-NAME result-applier allowlist entry for "
        f"{sorted(regressed)}. These audit/projection consumers must be wired by "
        "the generic derivation (from intent_consumption.intent_routing_table), not "
        "a hand-maintained name lookup — OMN-14516."
    )


def test_every_raw_projection_declares_resolvable_routing_or_is_known_unwired() -> None:
    """Generalized gate: every raw audit/projection node is DERIVABLE or ticketed.

    This is the gate that prevents OMN-14516 from recurring for a FUTURE node:
    ship a new ``consumer_purpose: audit|projection`` node with ``subscribe_topics``
    and this goes RED unless the node either (a) declares an
    ``intent_consumption.intent_routing_table`` that resolves to a real, importable
    write-effect handler (so the kernel derives its applier), or (b) is in the
    shrink-only ``KNOWN_UNWIRED`` ratchet with a tracking ticket.
    """
    manifest = discover_contracts()
    contracts_by_name = {c.name: c for c in manifest.contracts}

    dead: list[str] = []
    for contract_path in sorted(_NODES_ROOT.glob("*/contract.yaml")):
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
        event_bus = raw.get("event_bus") or {}
        purpose = str(event_bus.get("consumer_purpose") or "").strip().lower()
        if purpose not in {"audit", "projection"}:
            continue
        if not event_bus.get("subscribe_topics"):
            continue
        name = str(raw.get("name"))
        if name in KNOWN_UNWIRED:
            continue
        try:
            resolved = _resolve_routing_to_handler_classes(
                contract_path, contracts_by_name
            )
        except AssertionError:
            resolved = {}
        if not resolved:
            dead.append(name)

    assert not dead, (
        "Raw audit/projection contracts with NO derivable effect path — these "
        "consume Kafka offsets and drop every intent, so their tables stay empty "
        f"with nothing surfaced: {sorted(dead)}. Declare "
        "intent_consumption.intent_routing_table pointing at a real write-effect "
        "(see node_ledger_projection_compute) or the node is dead on arrival."
    )


def test_handler_exposes_dispatch_entrypoint() -> None:
    """Without handle(), auto-wiring binds _missing_handle and every event raises.

    Asserting the method exists on the MRO is deliberately not enough on its own —
    see the dispatch test below, which drives it. But absence here is an
    unambiguous defect, so it is pinned directly.
    """
    assert callable(getattr(HandlerLedgerProjection, "handle", None)), (
        "HandlerLedgerProjection must expose a handle() dispatch entrypoint; "
        "without it handler_wiring binds the _missing_handle sentinel, which "
        "raises on every dispatched event and is then swallowed by the consume "
        "boundary (events acked, ledger empty, nothing surfaced)."
    )


@pytest.mark.asyncio
async def test_handle_projects_dict_envelope_to_ledger_append_intent() -> None:
    """Drive the REAL engine with the dict-payload envelope the live path delivers.

    The live dispatch path hands the wiring a transport envelope whose ``payload``
    is a dumped ``ModelEventMessage`` dict. Post def-B flip (OMN-14823), the SHARED
    runtime dispatch adapter (``handler_wiring._make_dispatch_callback``) owns the
    dict -> typed materialization: it reads the contract-declared
    ``event_model`` (``ModelEventMessage``, OMN-14594) and validates the dict into
    it BEFORE invoking the canonical def-B ``handle(request: ModelEventMessage)``.
    So the faithful "live dict envelope produces the ledger.append intent"
    assertion drives the engine end-to-end, not a per-handler dict coercion (the
    OMN-14140 trap is now closed at the adapter boundary, not inside the handler).
    """
    engine = await _wire_and_freeze()
    topic = "onex.evt.platform.node-registration.v1"
    correlation_id = uuid4()
    message = ModelEventMessage(
        topic=topic,
        value=b'{"node_id": "abc"}',
        headers=ModelEventHeaders(
            correlation_id=correlation_id,
            timestamp=datetime.now(UTC),
            source="test-producer",
            event_type="NodeRegistrationEvent",
        ),
        partition=3,
        offset="42",
    )
    envelope = ModelEventEnvelope[object](
        payload=message.model_dump(mode="json"),
        correlation_id=correlation_id,
        event_type="platform.node-registration",
    )

    result = await engine.dispatch(topic, envelope)

    assert result.status == EnumDispatchStatus.SUCCESS, result.error_message
    # The ledger.append intent MUST land in output_intents (not output_events) so
    # the intent_routing_table reaches node_ledger_write_effect.
    assert len(result.output_intents) == 1
    intent = result.output_intents[0]
    assert isinstance(intent, ModelIntent)
    # The intent_type is the routing key the kernel's IntentExecutor dispatches on
    # — it must match the "ledger.append" write-effect operation exactly, or the
    # intent is produced and then dropped on the floor.
    assert intent.payload.intent_type == "ledger.append"
    assert intent.payload.topic == "onex.evt.platform.node-registration.v1"
    assert intent.payload.partition == 3
    assert intent.payload.kafka_offset == 42
    assert intent.payload.correlation_id == correlation_id
    # event_value is base64-encoded at this boundary; the Effect layer decodes it.
    assert intent.payload.event_value == base64.b64encode(b'{"node_id": "abc"}').decode(
        "ascii"
    )


# ---------------------------------------------------------------------------
# OMN-14594: cross-contamination on shared topics
# ---------------------------------------------------------------------------
#
# node_ledger_projection_compute's single handler_routing entry (operation
# ledger.project) declares NO event_model, so its dispatcher registers with no
# payload_type_matcher (OMN-12416 type-scoping never applies to it). All 7 of
# its subscribed topics (e.g. onex.evt.platform.node-registration.v1) are ALSO
# subscribed by other nodes (node_registration_orchestrator et al.) whose own
# Kafka consume loop decodes the raw bytes into the DOMAIN event shape
# (ModelNodeRegistrationEvent-like: node_id/node_type/...) before calling
# MessageDispatchEngine.dispatch(topic, envelope) — but dispatch() is a single
# process-wide engine: _find_matching_dispatchers() selects EVERY registered
# route for a topic+category, regardless of which subscriber's callback
# triggered the call. Since this dispatcher has no type-scoping, it gets
# invoked for that domain-shaped envelope too, and HandlerLedgerProjection
# always coerces envelope.payload against the RAW ModelEventMessage wrapper
# shape (topic/value/headers/partition) — which a domain event dict can never
# satisfy, so it fails with a ValidationError on every such cross-dispatch,
# live evidence: "Dispatcher '...HandlerLedgerProjection...' failed:
# ValidationError: N validation errors for ModelEventMessage".
class _StubResultApplierNoOp:
    async def apply(self, *args: object, **kwargs: object) -> None:
        return None


async def _wire_and_freeze() -> MessageDispatchEngine:
    """Wire the real ledger contract into a real, frozen dispatch engine."""
    discovered = discover_contracts_from_paths([CONTRACT_PATH])
    contracts = getattr(discovered, "contracts", discovered)
    manifest = ModelAutoWiringManifest(contracts=tuple(contracts))
    engine = MessageDispatchEngine()
    await wire_from_manifest(
        manifest,
        engine,
        event_bus=None,
        environment="dev",
        container=ModelONEXContainer(),
        subscribe_immediately=False,
        result_appliers_by_contract={CONTRACT_NAME: _StubResultApplierNoOp()},
    )
    engine.freeze()
    return engine


def _raw_wrapper_envelope(topic: str) -> ModelEventEnvelope[object]:
    """Build the envelope shape node_ledger_projection_compute's OWN raw-event-
    projection subscribe callback constructs (envelope.payload = the dumped
    ModelEventMessage — see _make_raw_event_projection_callback)."""
    message = ModelEventMessage(
        topic=topic,
        value=b'{"node_id": "abc"}',
        headers=ModelEventHeaders(
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            source="test-producer",
            event_type="platform.node-registration",
        ),
        partition=0,
        offset="0",
    )
    return ModelEventEnvelope[object](
        payload=message.model_dump(mode="json"),
        correlation_id=uuid4(),
        event_type="platform.node-registration",
    )


def _domain_shaped_envelope(topic: str) -> ModelEventEnvelope[object]:
    """Build the envelope shape a SIBLING node's own consume loop constructs for
    this SAME topic (the standard _make_event_bus_callback path): envelope.payload
    is the decoded domain event dict, never a ModelEventMessage wrapper."""
    return ModelEventEnvelope[object](
        payload={
            "node_id": str(uuid4()),
            "node_type": "effect",
            "default_enabled": False,
        },
        correlation_id=uuid4(),
        event_type="platform.node-registration",
    )


@pytest.mark.asyncio
async def test_ledger_projection_still_processes_its_own_raw_wrapper_shape() -> None:
    """Sanity: the genuinely-shaped envelope this dispatcher was built for must
    keep working before and after any type-scoping fix."""
    engine = await _wire_and_freeze()
    topic = "onex.evt.platform.node-registration.v1"

    result = await engine.dispatch(topic, _raw_wrapper_envelope(topic))

    assert result.status == EnumDispatchStatus.SUCCESS, result.error_message


@pytest.mark.asyncio
async def test_ledger_projection_does_not_error_on_sibling_domain_shaped_envelope() -> (
    None
):
    """OMN-14594: a domain-shaped envelope from a SIBLING subscriber on the same
    topic must not reach (and crash) this dispatcher.

    Pre-fix: no payload_type_matcher exists on this dispatcher, so
    _find_matching_dispatchers() selects it for ANY envelope on this topic —
    the domain-shaped payload gets handed to HandlerLedgerProjection, which
    raises ValidationError trying to coerce it into ModelEventMessage, and
    MessageDispatchEngine reports EnumDispatchStatus.HANDLER_ERROR. Live
    evidence (docker logs omninode-runtime, .201 dev-lane cold boot,
    2026-07-13): 18 such failures in ~6 minutes, recurring in bursts matching
    other nodes' periodic node-heartbeat/node-introspection broadcasts.

    Post-fix: the ledger.project entry declares event_model=ModelEventMessage,
    giving this dispatcher a payload_type_matcher (OMN-12416) that rejects any
    payload failing ModelEventMessage validation BEFORE the handler is ever
    invoked — the engine reports NO_DISPATCHER (cleanly unrouted for this
    contract, not a crash) instead of HANDLER_ERROR.
    """
    engine = await _wire_and_freeze()
    topic = "onex.evt.platform.node-registration.v1"

    result = await engine.dispatch(topic, _domain_shaped_envelope(topic))

    assert result.status == EnumDispatchStatus.NO_DISPATCHER, (
        f"node_ledger_projection_compute's dispatcher must be type-scoped out of "
        f"a domain-shaped envelope from a sibling subscriber (expected "
        f"NO_DISPATCHER, a clean non-match), not invoked-and-crashed: "
        f"status={result.status!r} error={result.error_message!r}"
    )
