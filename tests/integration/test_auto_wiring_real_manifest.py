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

from unittest.mock import MagicMock

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models.model_discovery_error import (
    ModelDiscoveryError,
)

_KNOWN_DELETED_OCC_STUBS = {
    "node_contract_dependency_compute",
    "node_contract_dependency_effect",
    "node_contract_dependency_orchestrator",
    "node_contract_dependency_reducer",
}


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

    The dispatch engine is mocked to avoid Kafka/DB dependencies; event_bus=None
    skips topic subscriptions so the test runs fully offline.
    """
    manifest = discover_contracts()

    dispatch_engine = MagicMock()
    dispatch_engine.is_frozen = False

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=dispatch_engine,
        event_bus=None,
        subscribe_immediately=False,
    )

    failed_results = [r for r in report.results if str(r.outcome).endswith("FAILED")]
    assert report.total_failed == 0, (
        f"wire_from_manifest() reported {report.total_failed} failure(s) against the "
        f"real manifest — this is a wiring regression (OMN-8735 pattern).\n"
        + "\n".join(f"  {r.contract_name}: {r.reason}" for r in failed_results)
    )

    # Confirm no ModelOnexError was raised in the results
    error_results = [
        r for r in report.results if r.reason and "ModelOnexError" in r.reason
    ]
    assert not error_results, "ModelOnexError found in wiring results:\n" + "\n".join(
        f"  {r.contract_name}: {r.reason}" for r in error_results
    )
