# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatch-reachability regression for the audit-ledger projection (OMN-14516).

``event_ledger`` held ZERO rows since ``node_ledger_projection_compute`` shipped,
while its 7 subscribed topics carried live traffic and the consumer group
``onex-ledger-projection-compute`` never existed. Nothing was red: the contract
parsed, its routing entry resolved, and CI was green.

Two independent defects, either of which alone keeps the table empty:

1. The contract declares ``consumer_purpose: "audit"``, which makes it a *raw
   event projection* contract. ``handler_wiring`` refuses to wire such a contract
   unless the kernel registers a result applier for it BY NAME -- otherwise the
   node would consume Kafka offsets while dropping every intent it emits. No
   applier was registered, so auto-wiring returned ``SKIPPED`` and no consumer
   was ever created.
2. ``HandlerLedgerProjection`` exposed no ``handle()``/``handle_async()``, so
   ``_make_dispatch_callback`` binds the ``_missing_handle`` sentinel, which
   raises on every dispatched event. The auto-wired consume boundary
   log-and-discards that exception (no DLQ, no redelivery), so events are acked
   and silently lost.

These tests drive the REAL seam -- the actual contract file through the actual
discovery + wiring path -- rather than asserting against hand-built fixtures. A
test that passes while the table stays empty is exactly the failure that produced
this bug, so the assertions here are about *dispatch reachability*, not about the
handler's transform in isolation.
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
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_ledger_projection_compute"
    / "contract.yaml"
)

CONTRACT_NAME = "node_ledger_projection_compute"


class _StubResultApplier:
    """Stand-in for the kernel's DispatchResultApplier.

    Only its PRESENCE in ``result_appliers_by_contract`` is load-bearing for
    wiring -- that is precisely the bit that was missing in production.
    """

    async def apply(self, *args: object, **kwargs: object) -> None:
        return None


async def _wire(*, with_applier: bool) -> tuple[EnumWiringOutcome, int, str]:
    """Run the real discovery + wiring path over the real contract file."""
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
    """The contract must reach WIRED -- SKIPPED means no consumer, so zero rows.

    This is the assertion that was missing. A gate that only checks "the routing
    entry parses" or "the handler class exists" stays GREEN on the dead node: the
    entry parses, all 7 topics get assigned, and the contract is still never
    wired.
    """
    outcome, wiring_count, reason = await _wire(with_applier=True)

    assert outcome is EnumWiringOutcome.WIRED, (
        f"node_ledger_projection_compute did not wire (outcome={outcome}, "
        f"reason={reason!r}). A raw audit/projection contract that does not reach "
        "WIRED never creates a consumer, so event_ledger stays empty."
    )
    assert wiring_count == 1, f"expected exactly 1 wired handler, got {wiring_count}"


@pytest.mark.asyncio
async def test_ledger_projection_is_skipped_without_result_applier() -> None:
    """Pin the ACTUAL production defect so the fix cannot silently regress.

    This is the red-proof: it asserts the exact mechanism that kept the ledger at
    zero rows. If a future change makes a raw projection contract wire WITHOUT an
    effect path, that is the other half of the bug (consume offsets, drop intents)
    and this test must be revisited deliberately -- not deleted.
    """
    outcome, wiring_count, reason = await _wire(with_applier=False)

    assert outcome is EnumWiringOutcome.SKIPPED
    assert wiring_count == 0
    assert "raw event projection" in reason


def _kernel_registered_applier_contracts() -> set[str]:
    """Contract names the kernel actually registers result appliers for.

    Parsed structurally from the AST of ``service_kernel`` -- every
    ``auto_wiring_result_appliers["<name>"] = ...`` assignment -- rather than by
    string search, so it reflects real registration sites and cannot be satisfied
    by a comment or a docstring mentioning the name.
    """
    source = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "omnibase_infra"
        / "runtime"
        / "service_kernel.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    registered: set[str] = set()
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
                registered.add(target.slice.value)
    return registered


def test_kernel_registers_applier_for_ledger_projection() -> None:
    """The kernel MUST register the applier -- this is the load-bearing fix.

    Without this assertion the wiring test above is self-fulfilling: it injects
    its own stub applier, so it would stay GREEN even if the kernel registration
    were deleted and event_ledger went back to zero rows. This test fails closed
    on exactly that regression.
    """
    assert CONTRACT_NAME in _kernel_registered_applier_contracts(), (
        f"service_kernel does not register a result applier for {CONTRACT_NAME}. "
        "Auto-wiring SKIPS raw audit/projection contracts that have no effect "
        "path, so the consumer is never created and event_ledger stays empty."
    )


def test_every_raw_projection_contract_has_a_kernel_result_applier() -> None:
    """Generalize the defect: ANY audit/projection node without an applier is dead.

    This is the gate that would have prevented OMN-14516 outright, and it fails
    closed for future nodes too: ship a new ``consumer_purpose: audit`` node
    without registering an applier and this goes RED instead of silently
    consuming offsets into a table that never fills.

    ``KNOWN_UNWIRED`` is a shrink-only ratchet, not an allowlist to grow. Each
    entry is a node that is CURRENTLY DEAD IN PRODUCTION and has a ticket.
    """
    # node_validation_ledger_projection_compute is dead for the identical reason
    # (validation_event_ledger has zero rows). It cannot reuse the ledger.append
    # effect path as-is: its handler returns a plain dict rather than a
    # ModelIntent, and its write side is PostgresValidationLedgerRepository
    # (asyncpg pool) rather than a HandlerDb-composing write-effect node -- so it
    # needs its own intent payload + intent effect. Tracked in OMN-14524.
    known_unwired = {"node_validation_ledger_projection_compute"}

    nodes_root = (
        Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"
    )
    registered = _kernel_registered_applier_contracts()

    dead: list[str] = []
    for contract_path in sorted(nodes_root.glob("*/contract.yaml")):
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
        event_bus = raw.get("event_bus") or {}
        purpose = str(event_bus.get("consumer_purpose") or "").strip().lower()
        if purpose not in {"audit", "projection"}:
            continue
        if not event_bus.get("subscribe_topics"):
            continue
        name = raw.get("name")
        if name not in registered and name not in known_unwired:
            dead.append(str(name))

    assert not dead, (
        "Raw audit/projection contracts with NO kernel result applier -- these "
        "consume Kafka offsets and drop every intent, so their tables stay empty "
        f"with nothing surfaced: {sorted(dead)}. Register an applier in "
        "service_kernel (see node_ledger_projection_compute) or the node is dead "
        "on arrival."
    )


def test_handler_exposes_dispatch_entrypoint() -> None:
    """Without handle(), auto-wiring binds _missing_handle and every event raises.

    Asserting the method exists on the MRO is deliberately not enough on its own
    -- see the dispatch test below, which drives it. But absence here is an
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
    while passing an object-shaped unit test -- the OMN-14140 trap. So the
    envelope here is a dict on purpose.
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
    # The intent_type is the routing key the kernel's IntentExecutor dispatches
    # on -- it must match the "ledger.append" handler registration exactly, or
    # the intent is produced and then dropped on the floor.
    assert intent.payload.intent_type == "ledger.append"
    assert intent.payload.topic == "onex.evt.platform.node-registration.v1"
    assert intent.payload.partition == 3
    assert intent.payload.kafka_offset == 42
    assert intent.payload.correlation_id == correlation_id
    # event_value is base64-encoded at this boundary; the Effect layer decodes it.
    assert intent.payload.event_value == base64.b64encode(b'{"node_id": "abc"}').decode(
        "ascii"
    )
