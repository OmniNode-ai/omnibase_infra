# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-manifest auto-wiring invariant test [OMN-9119].

This is the integration test that would have caught the OMN-8735 regression
(14 real handlers broke in prod) had it existed at the time.  Previous
auto-wiring tests used fake contracts with /fake/ paths and nonexistent modules.
None exercised the actual project tree.

This file does two things:
1. Calls discover_contracts() against the real installed onex.nodes entry points
   and asserts there are no actionable discovery errors.
2. Calls wire_from_manifest() against that manifest with a mock dispatch engine
   (no Kafka, no DB) and asserts total_failed == 0 (wiring phase is clean).

A failure here means a real handler in src/ cannot be imported or instantiated —
the kind of breakage that OMN-8735 introduced and that must never reach prod again.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovery_error import (
    ModelDiscoveryError,
)
from omnibase_infra.runtime.message_dispatch_engine import (
    DispatcherFunc,
    MessageDispatchEngine,
)
from omnibase_infra.runtime.service_intent_routing_loader import (
    load_intent_routing_table,
)
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

_KNOWN_DELETED_OCC_STUBS = {
    "node_contract_dependency_compute",
    "node_contract_dependency_effect",
    "node_contract_dependency_orchestrator",
    "node_contract_dependency_reducer",
}

# SHRINK-ONLY ratchet (OMN-14516). Raw audit/projection contracts that are dead in
# production AND have a tracking ticket. The wiring gate reports these RED (see the
# assertion message) but does not fail on them — every OTHER raw projection with no
# derivable applier IS a hard failure. Removing an entry is part of its ticket's
# DoD; NEVER add a live node here to silence the gate.
#
# EMPTY as of OMN-14524: node_validation_ledger_projection_compute now declares
# intent_consumption.intent_routing_table -> node_validation_ledger_write_effect
# (HandlerValidationLedgerAppend), so the kernel derivation wires it with no
# allowlist entry. Do NOT add a live node here to silence this gate.
_KNOWN_UNWIRED_RAW_PROJECTIONS: set[str] = set()


class _StubResultApplier:
    """Presence-only stand-in for the kernel's derived DispatchResultApplier.

    The kernel derives a real applier for every audit/projection consumer that
    declares an ``intent_consumption.intent_routing_table``. This offline gate
    mirrors that derivation with a no-op stub so the wiring phase proves the same
    set of contracts reaches WIRED — without a DB.
    """

    async def apply(self, *args: object, correlation_id: UUID | None = None) -> None:
        return None


class _PoolSentinel:
    """Offline stand-in for a pool resolved by the runtime dependency map."""


async def _noop_savings_correlation_publisher(
    event_type: str,
    payload: object,
    topic: str | None,
    correlation_id: object,
    **kwargs: object,
) -> bool:
    """Keep the real kernel dependency builder offline for this gate."""
    return True


def _handler_requires_pool(handler_cls: type) -> bool:
    """Return whether a handler has a required concrete ``pool`` parameter."""
    try:
        parameter = inspect.signature(handler_cls).parameters.get("pool")
    except (TypeError, ValueError):  # pragma: no cover - uninspectable handler
        return False
    return (
        parameter is not None
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


def _pool_taking_contracts() -> list[ModelDiscoveredContract]:
    """Discover every real-manifest contract with a required pool handler."""
    selected: list[ModelDiscoveredContract] = []
    for contract in discover_contracts().contracts:
        if contract.handler_routing is None:
            continue
        for entry in contract.handler_routing.handlers:
            module = importlib.import_module(entry.handler.module)
            handler_cls = getattr(module, entry.handler.name)
            if _handler_requires_pool(handler_cls):
                selected.append(contract)
                break
    return selected


def _strict_pool_dependencies(
    *,
    savings_correlation_pool: object | None,
) -> dict[str, dict[str, object]] | None:
    """Build the same explicit dependencies the runtime passes to wiring."""
    return _build_runtime_handler_dependencies(
        _PoolSentinel(),
        savings_correlation_pool=savings_correlation_pool,
        savings_correlation_publisher=_noop_savings_correlation_publisher,
    )


def _actionable_manifest_errors(
    errors: tuple[ModelDiscoveryError, ...],
) -> list[ModelDiscoveryError]:
    """Filter stale dependency entry points that are tracked outside this repo."""
    return [
        error
        for error in errors
        if not (
            error.package_name == "onex-change-control"
            and error.entry_point_name in _KNOWN_DELETED_OCC_STUBS
        )
    ]


@pytest.mark.integration
def test_real_manifest_discovery_has_no_errors() -> None:
    """discover_contracts() against the installed onex.nodes entry points must produce zero errors.

    This is the discovery-phase gate: every entry point must load cleanly and
    every contract.yaml must parse without errors.  A failure here means a node
    was registered in pyproject.toml [project.entry-points."onex.nodes"] but its
    contract.yaml is missing, malformed, or its module cannot be imported.
    """
    manifest = discover_contracts()
    actionable_errors = _actionable_manifest_errors(manifest.errors)

    assert not actionable_errors, (
        f"discover_contracts() reported {len(actionable_errors)} actionable error(s) against the "
        f"real manifest — this is a wiring regression.\n"
        + "\n".join(
            f"  [{e.package_name}] {e.entry_point_name}: {e.error}"
            for e in actionable_errors
        )
    )
    assert manifest.total_discovered > 0, (
        "discover_contracts() found zero contracts — entry points may not be installed"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_manifest_wiring_has_no_failures() -> None:
    """wire_from_manifest() against the real onex.nodes manifest must produce zero failures.

    This is the non-strict wiring-phase gate: every handler module must be
    importable and the real manifest must produce no unexpected failures. A
    separate strict pool-dependency gate below exercises constructor resolution
    with the runtime's actual explicit-dependency map.

    OMN-14516: raw audit/projection consumers are FAILED (not SKIPPED) when they
    have no result applier. The kernel DERIVES an applier for every such consumer
    that declares an ``intent_consumption.intent_routing_table``; this offline gate
    mirrors that derivation by supplying a presence-only stub applier for the same
    set. The remaining failures must be EXACTLY the shrink-only, ticketed
    ``_KNOWN_UNWIRED_RAW_PROJECTIONS`` set — anything else is a real regression.

    A real dispatch engine and container avoid mock auto-attribute resolution;
    event_bus=None skips topic subscriptions so the test runs fully offline.
    """
    manifest = discover_contracts()

    # Mirror the kernel derivation: every contract that declares an intent routing
    # table gets a result applier. Presence in this map is exactly what
    # handler_wiring's _raw_event_projection_enabled checks.
    derived_appliers = {
        contract.name: _StubResultApplier()
        for contract in manifest.contracts
        if load_intent_routing_table(Path(contract.contract_path))
    }

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=MessageDispatchEngine(),
        event_bus=None,
        container=ModelONEXContainer(),
        subscribe_immediately=False,
        result_appliers_by_contract=derived_appliers,
    )

    failed_results = [r for r in report.results if str(r.outcome).endswith("FAILED")]
    failed_names = {r.contract_name for r in failed_results}
    unexpected = failed_names - _KNOWN_UNWIRED_RAW_PROJECTIONS
    assert not unexpected, (
        f"wire_from_manifest() reported {len(unexpected)} unexpected failure(s) "
        f"against the real manifest — this is a wiring regression (OMN-8735 / "
        f"OMN-14516).\n"
        + "\n".join(
            f"  {r.contract_name}: {r.reason}"
            for r in failed_results
            if r.contract_name in unexpected
        )
    )
    # Surface the known-unwired ratchet RED-and-tracked (OMN-14516 must-hold): it is
    # visible in test output, never a silent exclusion. Shrinks to empty when the
    # tracked tickets land.
    tracked_dead = failed_names & _KNOWN_UNWIRED_RAW_PROJECTIONS
    if tracked_dead:
        print(
            "KNOWN-UNWIRED raw projections (dead in prod, ticketed, shrink-only): "
            f"{sorted(tracked_dead)} — see OMN-14524"
        )

    # Confirm no ModelOnexError was raised in the results
    error_results = [
        r for r in report.results if r.reason and "ModelOnexError" in r.reason
    ]
    assert not error_results, "ModelOnexError found in wiring results:\n" + "\n".join(
        f"  {r.contract_name}: {r.reason}" for r in error_results
    )


@pytest.mark.integration
def test_real_manifest_declares_pool_taking_handlers() -> None:
    """The strict constructor-dependency subject set must not be empty."""
    names = {contract.name for contract in _pool_taking_contracts()}
    assert names, (
        "No discovered contract declares a handler requiring `pool`; discovery "
        "is broken or the strict constructor-dependency gate is vacuous."
    )
    assert "node_savings_estimation_compute" in names, (
        "The OMN-17510 regression contract left the pool-taking subject set: "
        f"{sorted(names)}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pool_taking_handlers_resolve_under_strict_real_manifest_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every discovered required-pool handler resolves under strict wiring."""
    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")
    report = await wire_from_manifest(
        manifest=ModelAutoWiringManifest(contracts=_pool_taking_contracts()),
        dispatch_engine=MessageDispatchEngine(),
        event_bus=None,
        container=ModelONEXContainer(),
        subscribe_immediately=False,
        materialized_explicit_dependencies=_strict_pool_dependencies(
            savings_correlation_pool=_PoolSentinel()
        ),
    )

    failed = [
        result for result in report.results if str(result.outcome).endswith("FAILED")
    ]
    assert not failed, "\n".join(
        f"{result.contract_name}: {result.reason}" for result in failed
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_strict_pool_gate_is_red_without_the_application_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting the runtime pool binding reproduces the OMN-17510 TypeError."""
    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")
    with pytest.raises(TypeError, match=r"HandlerSavingsCorrelation.*\['pool'\]"):
        await wire_from_manifest(
            manifest=ModelAutoWiringManifest(contracts=_pool_taking_contracts()),
            dispatch_engine=MessageDispatchEngine(),
            event_bus=None,
            container=ModelONEXContainer(),
            subscribe_immediately=False,
            materialized_explicit_dependencies=_strict_pool_dependencies(
                savings_correlation_pool=None
            ),
        )


# ---------------------------------------------------------------------------
# OMN-16050 — registered-input-model unwrap stop, proven on the REAL manifest
# ---------------------------------------------------------------------------

_OMN16050_TOPIC = "onex.cmd.omnibase-infra.omn16050-unwrap-probe.v1"
_OMN16050_MODULE = "tests.integration.test_auto_wiring_real_manifest"


class ModelOmn16050EmitRequest(BaseModel):
    """Field-for-field mirror of omnimarket's ``ModelEmitRequest`` (OMN-16050).

    Declares a ``payload`` mapping plus FOUR transport marker keys
    (``event_type``, ``correlation_id``, ``partition_key``, ``event_id``), which
    is precisely what made it indistinguishable from a transport envelope to the
    old structural heuristic. ``extra="forbid"`` mirrors the real model, so an
    over-unwrap fails totally — the live DLQ signature.

    Lives here rather than in omnimarket because omnibase_infra is upstream of it
    and cannot import it; the shape, not the identity, is what the defect keys on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: str = Field(..., min_length=1)
    payload: dict[str, object] = Field(default_factory=dict)
    correlation_id: str | None = None
    topic: str | None = None
    partition_key: str | None = None
    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)


class HandlerOmn16050EmitProbe:
    """Canonical def-B handler: ``handle(request: ModelX) -> None``.

    Wired by the real ``wire_from_manifest`` path, so the callback under test is
    the production-built one, not a hand-constructed ``_make_dispatch_callback``.
    """

    received: list[ModelOmn16050EmitRequest] = []

    async def handle(self, request: ModelOmn16050EmitRequest) -> None:
        type(self).received.append(request)


def _omn16050_probe_contract() -> ModelDiscoveredContract:
    """An ``operation_match`` def-B EFFECT contract shaped like node_event_emit_effect."""
    return ModelDiscoveredContract(
        name="node_omn16050_unwrap_probe",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/omn16050/contract.yaml"),
        entry_point_name="node_omn16050_unwrap_probe",
        package_name="omnibase-infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_OMN16050_TOPIC,),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerOmn16050EmitProbe", module=_OMN16050_MODULE
                    ),
                    message_category="command",
                    event_type="omnibase-infra.omn16050-unwrap-probe",
                    operation="omn16050.probe",
                ),
            ),
        ),
    )


def _omn16050_published_bytes() -> dict[str, object]:
    """The live shape: one transport envelope wrapping a ModelEmitRequest.

    Mirrors the in-pod capture on onex-dev (digest sha256:35099472…) verbatim:
    ``RAW KEYS: ['event_type', 'correlation_id', 'source_tool', 'payload']``.
    """
    return {
        "event_type": "session.started",
        "correlation_id": "18a50ff5-c877-481c-b3c1-a183d8069762",
        "source_tool": "defect-ab-probe",
        "payload": {
            "event_type": "session.started",
            "correlation_id": "18a50ff5-c877-481c-b3c1-a183d8069762",
            "partition_key": "session-1",
            "event_id": "evt-defect-ab-probe",
            "payload": {
                "session_id": "18a50ff5-c877-481c-b3c1-a183d8069762",
                "defect_ab_probe": True,
                "emitted_at": "2026-08-13T02:46:13Z",
            },
        },
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_manifest_wiring_preserves_registered_envelope_shaped_input_model() -> (
    None
):
    """Runtime-startup gate for OMN-16050: real manifest + real wiring + real dispatch.

    Satisfies the repo's Runtime Startup CI gate for a PR touching
    ``auto_wiring/``: the manifest is the real one loaded from disk via
    ``discover_contracts()``, ``wire_from_manifest`` runs with the kernel's
    argument shape, and zero unexpected failures are asserted.

    On top of that it proves the defect is closed through the production path:
    one probe contract whose def-B handler declares an envelope-SHAPED input
    model (``payload`` + four transport markers, ``extra="forbid"``) is wired
    alongside the real contracts, and the dispatcher the wiring registered is
    invoked with the exact bytes captured in-pod. Before the fix the callback
    unwrapped through the domain model to the caller's inner payload and raised
    ``ValidationError`` (``event_type`` Field required + 3x extra_forbidden) —
    the live ``boundary_swallow_prevented`` / DLQ signature.
    """
    HandlerOmn16050EmitProbe.received.clear()

    real_manifest = discover_contracts()
    combined = ModelAutoWiringManifest(
        contracts=(*real_manifest.contracts, _omn16050_probe_contract()),
        errors=real_manifest.errors,
    )
    derived_appliers = {
        contract.name: _StubResultApplier()
        for contract in real_manifest.contracts
        if load_intent_routing_table(Path(contract.contract_path))
    }

    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest=combined,
        dispatch_engine=engine,
        event_bus=None,
        subscribe_immediately=False,
        result_appliers_by_contract=derived_appliers,
    )

    failed_names = {
        r.contract_name for r in report.results if str(r.outcome).endswith("FAILED")
    }
    unexpected = failed_names - _KNOWN_UNWIRED_RAW_PROJECTIONS
    assert not unexpected, (
        "wire_from_manifest() reported unexpected failure(s) against the real "
        f"manifest + the OMN-16050 probe contract: {sorted(unexpected)}"
    )

    probe_result = next(
        r for r in report.results if r.contract_name == "node_omn16050_unwrap_probe"
    )
    assert str(probe_result.outcome).endswith("WIRED"), (
        "the OMN-16050 probe contract must WIRE — a skipped/failed probe would "
        f"make this gate vacuous (outcome={probe_result.outcome}, "
        f"reason={probe_result.reason})"
    )
    assert len(probe_result.dispatchers_registered) == 1

    dispatcher_id = probe_result.dispatchers_registered[0]
    dispatcher = cast("DispatcherFunc", engine._dispatchers[dispatcher_id].dispatcher)
    dispatch_result = dispatcher(
        ModelEventEnvelope[object].model_validate(_omn16050_published_bytes())
    )
    assert inspect.isawaitable(dispatch_result)

    await dispatch_result

    assert len(HandlerOmn16050EmitProbe.received) == 1, (
        "the wired dispatcher did not deliver to the handler — pre-fix this "
        "raised ValidationError inside the callback and DLQ'd"
    )
    request = HandlerOmn16050EmitProbe.received[0]
    assert isinstance(request, ModelOmn16050EmitRequest)
    assert request.event_type == "session.started"
    assert request.event_id == "evt-defect-ab-probe"
    assert request.partition_key == "session-1"
    # The load-bearing assertion: the handler owns the CALLER's payload, and the
    # unwrap stopped at the registered model instead of walking through it.
    assert request.payload == {
        "session_id": "18a50ff5-c877-481c-b3c1-a183d8069762",
        "defect_ab_probe": True,
        "emitted_at": "2026-08-13T02:46:13Z",
    }
