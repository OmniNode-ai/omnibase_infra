# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A zero-route projection entry must not demand a workload DSN [OMN-17519].

The defect, read live off ``onex-dev`` on
``omninode-runtime@sha256:349225be`` (tag ``candidate-5e079d3-20260902060932``),
Deploy onex-staging run 33609666720, job 100181796582, step 37::

    omnibase_core.errors.model_onex_error.ModelOnexError: Auto-wiring failed for
    2 contract(s):
      node_hook_event_capture: ... HandlerHookEventCapture: ValueError:
        Projection handler requires topology bindings with configured DSNs:
        tenant_projection:ONEX_TENANT_DB...
      projection_delegation: ... HandlerProjectionDelegation: ValueError:
        Projection handler requires topology bindings with configured DSNs:
        tenant_projection:ONEX_TENA...

``omninode-runtime`` (``RUNTIME_PROFILE=main``) failed the same way on
``projection_savings``. Under ``ONEX_WIRING_STRICT_MODE=1`` — which onex-dev
sets on both Deployments — the OMN-13203 per-handler quarantine is off, so one
contract's wiring failure takes the whole boot down.

Why ``projection_delegation`` and ``projection_savings`` are the wrong subjects
for the shared runtime at all
------------------------------------------------------------------------------
OMN-15905 introduced dedicated projection-writer Deployments so the shared
runtime and effects processes do not write tenant-domain rows in-process. Each
such contract declares TWO handler entries: the standalone ``*ProjectionRunner``
the writer Deployment runs directly (``command: [python, -m, <runner module>]``)
and an in-process sibling. ``_topics_for_handler_entry`` gives every subscribe
topic to the runner — the sibling gets **zero routes**, so no message can reach
it in this process. Proven live in this file (``test_writer_owned_siblings_are_
route_starved``) rather than asserted: it is the premise the fix rests on.

Wiring a projection database for an entry that cannot be dispatched made the
SHARED runtime demand the tenant-domain credential (``tenant_projection`` ->
``ONEX_TENANT_DB_URL``) for rows the dedicated writer owns — the OMN-15905
separation in reverse, and fatal under strict mode.

Scope: the CLASS, not the two contracts. Subjects are DISCOVERED from the real
manifest; the hermetic half below builds the shape from scratch so the gate is
never vacuous in a CI job that does not install the sibling packages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core.model_deployment_topology import (
    ModelDeploymentTopology,
)
from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDatabaseTarget,
    _make_projection_dispatch_callback,
    _make_undispatched_projection_callback,
    _projection_dispatch_owned_elsewhere,
    _resolve_projection_database_target,
    _topic_owning_handler_names,
    _topics_for_handler_entry,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelDiscoveredContract,
    ModelHandlerRouting,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_infra.topology import load_topology_profile

# The lane whose boot this gate reproduces. Its `tenant_projection` binding
# declares `dsn_env: ONEX_TENANT_DB_URL`, which no onex-dev manifest binds.
_ONEX_DEV_PROFILE = "onex-dev"

# Contracts the OMN-15905 dedicated-writer Deployments own on onex-dev
# (k8s/onex-dev/runtime/deployment-omnimarket-projection-*-writer.yaml). Named
# so a silent removal of the writer-owned shape from either one is a failure
# here rather than a quietly narrowed gate.
_WRITER_OWNED_CONTRACTS: dict[str, str] = {
    "projection_delegation": "HandlerProjectionDelegation",
    "projection_savings": "HandlerProjectionSavings",
}


def _writer_owned_contract() -> ModelDiscoveredContract:
    """The OMN-15905 two-entry shape, built from scratch.

    Mirrors ``node_projection_delegation``: a standalone runner entry that
    declares no ``event_model`` (so it takes every subscribe topic) and an
    in-process sibling that declares one (so, with more than one topic and more
    than one entry, the ambiguity guard leaves it none). Built here rather than
    read from the manifest so this gate is non-vacuous in the CI job that runs
    it — omnimarket is deliberately absent from this repo's canonical venv (the
    OMN-15620 purity gate rejects an undeclared ``onex.nodes`` provider).
    """
    return ModelDiscoveredContract(
        name="omn17519_writer_owned_probe",
        node_type="REDUCER",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("omn17519_writer_owned_probe/contract.yaml"),
        entry_point_name="omn17519_writer_owned_probe",
        package_name="omn17519",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(
                "onex.evt.omn17519.first.v1",
                "onex.evt.omn17519.second.v1",
            ),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="ProbeProjectionRunner", module="omn17519.runner"
                    ),
                    operation="probe_projection_runner",
                ),
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerProbeProjection", module="omn17519.handler"
                    ),
                    operation="probe_projection",
                    event_model=ModelHandlerRef(
                        name="ModelProbeEvent", module="omn17519.models"
                    ),
                ),
            ),
        ),
    )


class _StubHandler:
    """An in-process projection handler: no runner shape, no DB of its own."""

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        return {"rows_upserted": 0}


def _onex_dev_topology() -> ModelDeploymentTopology:
    return load_topology_profile(_ONEX_DEV_PROFILE)


def _tenant_domain_target(
    topology: ModelDeploymentTopology,
) -> ProjectionDatabaseTarget:
    """A TENANT-domain projection target resolved through the real resolver.

    The table declaration mirrors ``node_projection_delegation``'s first
    ``db_io.db_tables`` entry verbatim (``delegation_events``, ``schema:
    tenant``, ``access: read_write``). It is restated here rather than read from
    the manifest because omnimarket is deliberately absent from this repo's
    canonical venv (the OMN-15620 purity gate rejects an undeclared ``onex.nodes``
    provider), and a gate that silently went vacuous whenever the sibling package
    is missing would prove nothing in the CI job that actually runs it.

    The BINDING is not restated: ``_resolve_projection_database_target`` selects
    it from the checked-in onex-dev topology, so this asserts against whatever
    identity that topology really assigns to a tenant-domain write.
    """
    target = _resolve_projection_database_target(
        (
            ModelDbTableDeclaration(
                name="delegation_events",
                database_ref="application",
                schema="tenant",
                migration="0007_delegation_events.sql",
                access="read_write",
                role="events",
            ),
        ),
        topology,
    )
    assert any(
        binding.dsn_env == "ONEX_TENANT_DB_URL" for binding in target.bindings
    ), (
        "The onex-dev topology no longer routes a tenant-domain write to "
        f"ONEX_TENANT_DB_URL (got {[b.dsn_env for b in target.bindings]}); this "
        "gate would no longer reproduce the OMN-17519 boot failure."
    )
    return target


def _projection_contracts_with_db_io() -> list[ModelDiscoveredContract]:
    return [
        contract
        for contract in discover_contracts().contracts
        if contract.db_io is not None
        and contract.db_io.db_tables
        and contract.handler_routing is not None
    ]


# ---------------------------------------------------------------------------
# Hermetic half — always runs, never vacuous.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_zero_route_entry_needs_no_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """The assertion that was RED before the fix, on the real raise site.

    With ``ONEX_TENANT_DB_URL`` unset — the onex-dev pod's actual environment —
    building the projection dispatch callback for an entry this process cannot
    dispatch used to raise ``ValueError: Projection handler requires topology
    bindings with configured DSNs: tenant_projection:ONEX_TENANT_DB_URL``.
    """
    monkeypatch.delenv("ONEX_TENANT_DB_URL", raising=False)
    target = _tenant_domain_target(_onex_dev_topology())

    callback = _make_undispatched_projection_callback(
        "HandlerProjectionDelegation",
        target,
        "projection_delegation",
        ("DelegationProjectionRunner",),
    )
    assert callback is not None


@pytest.mark.integration
def test_dispatchable_entry_still_fails_closed_on_a_missing_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-vacuity: the DSN requirement is untouched for a dispatched entry.

    Without this, ``test_zero_route_entry_needs_no_dsn`` would pass just as well
    against a build that deleted the requirement outright — which is the
    "make the handler tolerate a missing binding" defensive default this fix
    exists to avoid.
    """
    monkeypatch.delenv("ONEX_TENANT_DB_URL", raising=False)
    target = _tenant_domain_target(_onex_dev_topology())

    with pytest.raises(ValueError, match="ONEX_TENANT_DB_URL"):
        _make_projection_dispatch_callback(
            _StubHandler(),
            target,
            (),
            contract_name="omn17519_probe",
        )


@pytest.mark.integration
def test_starved_sibling_is_owned_elsewhere_and_owner_is_named() -> None:
    """The selection rule itself, on the OMN-15905 two-entry shape.

    The runner entry keeps every topic and is dispatched here; the sibling keeps
    none and is the one whose database the runtime stops opening. Both halves
    are asserted so a change that simply stopped wiring the whole contract —
    which would take the runner's routes with it — fails here.
    """
    contract = _writer_owned_contract()
    assert contract.handler_routing is not None
    runner, sibling = contract.handler_routing.handlers

    assert _topics_for_handler_entry(contract, runner) == (
        contract.event_bus.subscribe_topics if contract.event_bus else ()
    )
    assert not _projection_dispatch_owned_elsewhere(contract, runner)

    assert _topics_for_handler_entry(contract, sibling) == ()
    assert _projection_dispatch_owned_elsewhere(contract, sibling)
    assert _topic_owning_handler_names(contract, sibling) == ("ProbeProjectionRunner",)
    assert _topic_owning_handler_names(contract, runner) == ()


# ---------------------------------------------------------------------------
# Real-manifest half — the boot shape, over the contracts onex-dev actually
# wires. Skips only when the sibling packages are absent, and says so.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_writer_owned_siblings_are_route_starved() -> None:
    """The premise: the in-process sibling owns zero topics, the runner owns all.

    This is the fact the fix rests on and the fact that makes it safe — the
    entry whose database the runtime stops opening carries no traffic, before or
    after. If a contract ever gives that sibling a route, this fails and the fix
    must be revisited rather than silently continuing to withhold its DB.
    """
    by_name = {c.name: c for c in _projection_contracts_with_db_io()}
    missing = sorted(set(_WRITER_OWNED_CONTRACTS) - set(by_name))
    if missing:
        pytest.skip(
            f"omnimarket contracts not installed in this environment ({missing}); "
            "the hermetic half of this gate still ran."
        )

    for contract_name, starved_handler in _WRITER_OWNED_CONTRACTS.items():
        contract = by_name[contract_name]
        assert contract.handler_routing is not None
        entries = {e.handler.name: e for e in contract.handler_routing.handlers}
        assert starved_handler in entries, (
            f"{contract_name} no longer declares {starved_handler}; update "
            f"_WRITER_OWNED_CONTRACTS deliberately instead of letting this gate "
            f"stop covering it. Found: {sorted(entries)}"
        )
        entry = entries[starved_handler]

        assert _topics_for_handler_entry(contract, entry) == (), (
            f"{contract_name}.{starved_handler} now owns subscribe topics. The "
            f"shared runtime WOULD dispatch it, so withholding its projection "
            f"database is no longer correct."
        )
        assert _projection_dispatch_owned_elsewhere(contract, entry)

        owners = _topic_owning_handler_names(contract, entry)
        assert owners, (
            f"{contract_name} has no topic-owning entry at all — it is orphaned, "
            f"not writer-owned, and must not be cited as the OMN-15905 shape."
        )
        for owner_name in owners:
            assert _topics_for_handler_entry(contract, entries[owner_name])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_shared_profiles_wire_writer_owned_contracts_without_a_tenant_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The onex-dev boot shape: main + effects, real wiring, no tenant DSN.

    Runs the production ``wire_from_manifest`` path over every db_io contract
    the ``main`` and ``effects`` profiles own, against the checked-in onex-dev
    topology, with the environment those pods actually have — ``OMNINODE_INTERNAL
    _DB_URL`` bound, ``ONEX_TENANT_DB_URL`` not. Before the fix this reported
    ``projection_savings`` (main) and ``projection_delegation`` (effects) among
    the failures, which is what strict mode turned into CrashLoopBackOff.

    Strict mode is deliberately OFF here so every failure is collected instead of
    the first one raising: the assertion is about WHICH contracts fail, and a
    strict run can only ever show one.
    """
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
    from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
    from omnibase_infra.runtime.auto_wiring.profile_ownership import (
        filter_manifest_for_runtime_profile,
    )
    from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

    manifest = discover_contracts()
    present = {c.name for c in manifest.contracts}
    missing = sorted(set(_WRITER_OWNED_CONTRACTS) - present)
    if missing:
        pytest.skip(
            f"omnimarket contracts not installed in this environment ({missing}); "
            "the hermetic half of this gate still ran."
        )

    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "0")
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", "postgresql://probe/omn17519")
    monkeypatch.setenv("OMNIBASE_INFRA_DB_URL", "postgresql://probe/omn17519")
    monkeypatch.delenv("ONEX_TENANT_DB_URL", raising=False)

    topology = _onex_dev_topology()
    failed_by_profile: dict[str, set[str]] = {}
    for profile in ("main", "effects"):
        owned = filter_manifest_for_runtime_profile(manifest, profile).manifest
        subset = tuple(
            c for c in owned.contracts if c.db_io is not None and c.db_io.db_tables
        )
        report = await wire_from_manifest(
            manifest=ModelAutoWiringManifest(contracts=subset),
            dispatch_engine=MessageDispatchEngine(),
            event_bus=None,
            container=ModelONEXContainer(),
            subscribe_immediately=False,
            topology=topology,
        )
        failed_by_profile[profile] = {
            result.contract_name
            for result in report.results
            if str(result.outcome).endswith("FAILED")
        }

    all_failed = set().union(*failed_by_profile.values())
    assert all_failed, (
        "No contract failed with ONEX_TENANT_DB_URL unset. Either the tenant "
        "binding is no longer required anywhere — in which case this gate is "
        "vacuous and must be rewritten — or the manifest lost every "
        "tenant-domain projection."
    )
    still_failing = sorted(set(_WRITER_OWNED_CONTRACTS) & all_failed)
    assert not still_failing, (
        f"Contracts a dedicated OMN-15905 writer owns still fail to wire on the "
        f"shared profiles without the tenant DSN: {still_failing} "
        f"(main={sorted(failed_by_profile['main'])}, "
        f"effects={sorted(failed_by_profile['effects'])}). The shared runtime is "
        f"being made to hold the tenant credential for rows the writer owns — "
        f"the OMN-15905 separation in reverse, and what crash-looped onex-dev."
    )
