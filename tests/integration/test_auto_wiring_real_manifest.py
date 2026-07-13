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

from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models.model_discovery_error import (
    ModelDiscoveryError,
)
from omnibase_infra.runtime.service_intent_routing_loader import (
    load_intent_routing_table,
)

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
#   - node_validation_ledger_projection_compute: no write-effect node exists
#     (PostgresValidationLedgerRepository is never instantiated), so it cannot yet
#     declare an intent_routing_table. Tracked in OMN-14524 (build the write-effect,
#     prove validation_event_ledger 0->N, then delete this entry).
_KNOWN_UNWIRED_RAW_PROJECTIONS = {"node_validation_ledger_projection_compute"}


class _StubResultApplier:
    """Presence-only stand-in for the kernel's derived DispatchResultApplier.

    The kernel derives a real applier for every audit/projection consumer that
    declares an ``intent_consumption.intent_routing_table``. This offline gate
    mirrors that derivation with a no-op stub so the wiring phase proves the same
    set of contracts reaches WIRED — without a DB.
    """

    async def apply(self, *args: object, correlation_id: UUID | None = None) -> None:
        return None


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

    This is the wiring-phase gate: every handler module must be importable and
    every handler class must be instantiable with no constructor arguments.  A
    failure here means a handler that was working before has broken — exactly the
    OMN-8735 regression pattern (14 handlers silently broke without this gate).

    OMN-14516: raw audit/projection consumers are FAILED (not SKIPPED) when they
    have no result applier. The kernel DERIVES an applier for every such consumer
    that declares an ``intent_consumption.intent_routing_table``; this offline gate
    mirrors that derivation by supplying a presence-only stub applier for the same
    set. The remaining failures must be EXACTLY the shrink-only, ticketed
    ``_KNOWN_UNWIRED_RAW_PROJECTIONS`` set — anything else is a real regression.

    The dispatch engine is mocked to avoid Kafka/DB dependencies; event_bus=None
    skips topic subscriptions so the test runs fully offline.
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

    dispatch_engine = MagicMock()
    dispatch_engine.is_frozen = False

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=dispatch_engine,
        event_bus=None,
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
