# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Wiring-report totality at the initial-subscription seam (OMN-15474/OMN-15621).

``subscribe_wired_contract_topics`` requires an exact bijection between the
manifest it is handed and the wiring report it is handed, and it enforces that
BEFORE it does anything else — a mismatch aborts the kernel at boot (ruling 4,
OMN-15474; restored by OMN-15621 after PR #2609 narrowed it to a
report-subset-of-manifest check). That check is only safe if the report is
TOTAL over the manifest: every discovered contract carries an explicit verdict
(wired / failed / skipped-with-a-reason) rather than being silently absent.

These tests drive the real producer (``wire_from_manifest``) and the real
consumer (``subscribe_wired_contract_topics``) against contracts that
legitimately do not wire. No matched-pair fixture builds a report by mirroring
a hand-written name list against a hand-written manifest — the report is either
produced by the engine itself or by the product's own totality constructor
``build_unwired_contract_results``, which derives its rows FROM the manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.runtime.auto_wiring import (
    build_unwired_contract_results,
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine


def _contract(name: str, **overrides: object) -> ModelDiscoveredContract:
    """Build a discovered contract with the minimum required identity fields."""
    return ModelDiscoveredContract(
        name=name,
        node_type="COMPUTE_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake") / f"{name}.yaml",
        entry_point_name=name,
        package_name="omnibase_infra",
        **overrides,  # type: ignore[arg-type]
    )


def _never_wiring_manifest() -> ModelAutoWiringManifest:
    """Three contracts that each legitimately fail to wire, for three reasons.

    None of these is an error state: a contract with no ``handler_routing``
    declares no local handler, a contract with no ``subscribe_topics`` consumes
    nothing, and a ``plugin_managed`` contract hands its subscription to a
    domain plugin (OMN-10864). All three are ordinary steady-state shapes in
    the live manifest, and none of them may vanish from the report.
    """
    return ModelAutoWiringManifest(
        contracts=(
            _contract("node_totality_probe_no_handler_routing"),
            _contract(
                "node_totality_probe_no_subscribe_topics",
                event_bus=ModelEventBusWiring(subscribe_topics=(), publish_topics=()),
            ),
            _contract(
                "node_totality_probe_plugin_managed",
                event_bus=ModelEventBusWiring(
                    subscribe_topics=("onex.evt.platform.totality-probe.v1",),
                    publish_topics=(),
                    plugin_managed=True,
                ),
            ),
        ),
        errors=(),
    )


async def test_wire_from_manifest_report_is_total_over_never_wiring_contracts() -> None:
    """The engine emits a verdict row for every manifest contract, wired or not."""
    manifest = _never_wiring_manifest()

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=MessageDispatchEngine(),
        event_bus=None,
        environment="dev",
        subscribe_immediately=False,
    )

    manifest_names = {contract.name for contract in manifest.contracts}
    report_names = {result.contract_name for result in report.results}
    assert report_names == manifest_names, (
        "wiring report is not total over the manifest; "
        f"missing={sorted(manifest_names - report_names)} "
        f"unexpected={sorted(report_names - manifest_names)}"
    )
    assert all(
        result.outcome is EnumWiringOutcome.SKIPPED and result.reason
        for result in report.results
    ), "every non-wiring row must carry an explicit reason, not an empty string"


async def test_subscribe_accepts_a_total_report_built_by_the_totality_constructor() -> (
    None
):
    """A producer other than the engine can still satisfy the identity check.

    This is the seam the kernel-test fixture broke: a stand-in for the wiring
    engine that returns ``results=()`` claims the runtime discovered N
    contracts and reached no verdict on any of them, which the initial
    subscription check rejects (correctly). Building the rows from the manifest
    via ``build_unwired_contract_results`` is the truthful encoding, and the
    check accepts it without being relaxed.
    """
    manifest = _never_wiring_manifest()
    report = ModelAutoWiringReport(
        results=build_unwired_contract_results(
            manifest,
            reason="wiring engine not run in this configuration",
        ),
        duplicates=(),
    )

    subscribed = await subscribe_wired_contract_topics(
        manifest=manifest,
        report=report,
        dispatch_engine=MessageDispatchEngine(),
        event_bus=object(),
        environment="dev",
    )

    assert subscribed == {}, "no contract wired, so nothing may subscribe"


async def test_subscribe_still_rejects_a_partial_report() -> None:
    """Totality is added at the producer; the identity check is NOT relaxed.

    RED-before/GREEN-after proof for OMN-15621 (ruling 4, OMN-15474): the fix
    must not degrade the bijection to a subset relation — a report that omits
    manifest contracts is still a hard boot-time refusal, because "this
    contract has no verdict" is exactly the state that let a contract's
    events reach a process-global dispatch. At the pre-fix HEAD (PR #2609,
    report-subset-of-manifest only) this test fails because
    ``subscribe_wired_contract_topics`` silently accepts the truncated report
    below instead of raising.
    """
    manifest = _never_wiring_manifest()
    partial = ModelAutoWiringReport(
        results=build_unwired_contract_results(
            manifest,
            reason="wiring engine not run in this configuration",
        )[:1],
        duplicates=(),
    )

    with pytest.raises(ModelOnexError) as excinfo:
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=partial,
            dispatch_engine=MessageDispatchEngine(),
            event_bus=object(),
            environment="dev",
        )

    assert "exact bijection" in str(excinfo.value)


async def test_subscribe_still_rejects_report_rows_with_no_manifest_contract() -> None:
    """The direction no backfill can repair stays fatal."""
    manifest = _never_wiring_manifest()
    foreign = ModelAutoWiringManifest(
        contracts=(
            *manifest.contracts,
            _contract("node_totality_probe_not_discovered"),
        ),
        errors=(),
    )
    report = ModelAutoWiringReport(
        results=build_unwired_contract_results(foreign, reason="foreign producer"),
        duplicates=(),
    )

    with pytest.raises(ModelOnexError) as excinfo:
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=MessageDispatchEngine(),
            event_bus=object(),
            environment="dev",
        )

    assert "node_totality_probe_not_discovered" in str(excinfo.value)


def test_build_unwired_contract_results_covers_only_uncovered_contracts() -> None:
    """The constructor is a completion, not a duplicator."""
    manifest = _never_wiring_manifest()
    covered = manifest.contracts[0].name

    rows = build_unwired_contract_results(
        manifest,
        reason="already reported elsewhere",
        already_reported=(covered,),
    )

    assert [row.contract_name for row in rows] == [
        contract.name for contract in manifest.contracts[1:]
    ]
    assert (
        build_unwired_contract_results(
            manifest,
            reason="fully covered",
            already_reported=tuple(c.name for c in manifest.contracts),
        )
        == ()
    )
