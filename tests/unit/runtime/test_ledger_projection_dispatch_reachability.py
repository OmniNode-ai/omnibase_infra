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
from omnibase_core.models.reducer.model_intent import ModelIntent
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
    """
    outcome, wiring_count, reason = await _wire(with_applier=True)

    assert outcome is EnumWiringOutcome.WIRED, (
        f"node_ledger_projection_compute did not wire (outcome={outcome}, "
        f"reason={reason!r}). A raw audit/projection contract that does not reach "
        "WIRED never creates a consumer, so event_ledger stays empty."
    )
    assert wiring_count == 1, f"expected exactly 1 wired handler, got {wiring_count}"


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
    """Drive handle() with the DICT envelope the live dispatch path delivers.

    ``MessageDispatchEngine._materialize_envelope_with_bindings`` hands the
    handler a dict, not an attribute-bearing envelope. A handle() that only did
    ``getattr(envelope, "payload")`` would silently no-op on the real runtime path
    while passing an object-shaped unit test — the OMN-14140 trap. So the envelope
    here is a dict on purpose.
    """
    handler = HandlerLedgerProjection(ModelONEXContainer())
    correlation_id = uuid4()
    message = ModelEventMessage(
        topic="onex.evt.platform.node-registration.v1",
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

    output = await handler.handle({"payload": message.model_dump()})

    intent = output.result
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
