# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""End-to-end integration tests for HandlerResolver full auto-wiring (OMN-9204).

Plan: ``docs/plans/2026-04-18-handler-resolver-architecture.md`` §Task 8
(starting line 1342).

This suite exercises ``wire_from_manifest(...)`` end-to-end across every
``ServiceHandlerResolver`` precedence branch with a deterministic in-process
manifest. The fixture contains at least one handler per branch:

    * RESOLVED_VIA_NODE_REGISTRY       — materialized explicit dep map
    * RESOLVED_VIA_CONTAINER           — container.get_service returns instance
    * RESOLVED_VIA_EVENT_BUS           — ``__init__(self, event_bus)``
    * RESOLVED_VIA_ZERO_ARG            — zero-arg constructor
    * RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP — node not hosted here

Because the integration test builds its manifest in-process (rather than
calling ``discover_contracts()`` against the real project tree), it is
hermetic — CI determinism does not depend on which domain plugins happen
to ship at HEAD. This is the structural projection the plan describes as
"full auto-wiring exercises the resolver" without coupling the test to
the transient set of real nodes.

Acceptance (plan §Task 8):
    * Five scenarios — each precedence branch, plus a determinism assertion
      and a negative-case (TypeError) assertion.
    * Each test asserts a specific outcome (no generic "runs without error").
    * Skip-path exercises ``local_node_names`` set-membership — not any SQL.
    * ``wire_from_manifest`` is invoked twice and the structural projection
      of the two reports is byte-identical.
    * ``TypeError`` on an unresolvable handler propagates unchanged and no
      report is returned (OMN-8735 fail-fast preserved via Task 5 cutover).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)
from omnibase_core.errors.error_service_resolution import ServiceResolutionError
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
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
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
)
from omnibase_infra.runtime.service_message_dispatch_engine import (
    MessageDispatchEngine,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixture handlers — one per resolver precedence branch
# ---------------------------------------------------------------------------


class HandlerNodeRegistryOwned:
    """Handler resolved via RESOLVED_VIA_NODE_REGISTRY (materialized deps)."""

    def __init__(self, projection_reader: object) -> None:
        self.projection_reader = projection_reader

    async def handle(self, envelope: object) -> None:
        return None


class HandlerContainerOwned:
    """Handler resolved via RESOLVED_VIA_CONTAINER (DI container supplies it)."""

    def __init__(self, storage: object) -> None:
        self.storage = storage

    async def handle(self, envelope: object) -> None:
        return None


class HandlerEventBusOwned:
    """Handler resolved via RESOLVED_VIA_EVENT_BUS (single event_bus kwarg)."""

    def __init__(self, event_bus: object) -> None:
        self.event_bus = event_bus

    async def handle(self, envelope: object) -> None:
        return None


class HandlerZeroArg:
    """Handler resolved via RESOLVED_VIA_ZERO_ARG (no constructor deps)."""

    async def handle(self, envelope: object) -> None:
        return None


class HandlerForeignSkip:
    """Handler owned by a node that is NOT hosted here — resolver skips."""

    async def handle(self, envelope: object) -> None:
        return None


class HandlerUnresolvable:
    """Handler with an unsatisfiable constructor — resolver raises TypeError."""

    def __init__(self, some_undeclared_service: object) -> None:
        self.some_undeclared_service = some_undeclared_service

    async def handle(self, envelope: object) -> None:
        return None


# ---------------------------------------------------------------------------
# Fake container + event_bus — minimum surface the resolver uses
# ---------------------------------------------------------------------------


class _FakeContainer:
    """Exposes only get_service. Raises ServiceResolutionError for unknowns."""

    def __init__(self, registry: dict[type, object]) -> None:
        # Frozen at construction; tests must not mutate after wiring starts.
        self._registry: dict[type, object] = dict(registry)

    def get_service(self, service_cls: type) -> object:
        if service_cls not in self._registry:
            raise ServiceResolutionError(
                f"Service {service_cls.__name__} is not registered"
            )
        return self._registry[service_cls]


class _FakeEventBus:
    """Event bus stub — no-op subscribe, sentinel for event_bus injection.

    ``handler_wiring._commit_contract_wiring`` calls ``event_bus.subscribe(...)``
    for every wired contract after the resolver step. The resolver itself
    does not introspect event_bus; it only forwards it into the constructor
    when a handler's only required param is named ``event_bus``.
    """

    async def subscribe(
        self, *, topic: str, node_identity: object, on_message: object
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# Manifest fixture — one contract per resolver branch
# ---------------------------------------------------------------------------

_HANDLER_MODULE = "tests.integration.runtime.test_auto_wiring_resolver_end_to_end"


def _make_single_handler_contract(
    *,
    name: str,
    handler_cls: type,
    topic: str,
) -> ModelDiscoveredContract:
    """Build a single-handler contract for one precedence branch.

    All contracts share a stable schema so the structural projection used by
    the determinism assertion is a tuple-of-tuples — no floating pointers.
    """
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(topic,),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=handler_cls.__name__,
                        module=_HANDLER_MODULE,
                    ),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


def _build_full_branch_manifest() -> ModelAutoWiringManifest:
    """Manifest with one contract per resolver precedence branch.

    Contracts are added in a fixed order so ``manifest.contracts`` tuple
    ordering is deterministic — required by the projection determinism
    assertion below.
    """
    contracts = (
        _make_single_handler_contract(
            name="node_registry_owned",
            handler_cls=HandlerNodeRegistryOwned,
            topic="onex.evt.platform.registry-owned.v1",
        ),
        _make_single_handler_contract(
            name="node_container_owned",
            handler_cls=HandlerContainerOwned,
            topic="onex.evt.platform.container-owned.v1",
        ),
        _make_single_handler_contract(
            name="node_event_bus_owned",
            handler_cls=HandlerEventBusOwned,
            topic="onex.evt.platform.event-bus-owned.v1",
        ),
        _make_single_handler_contract(
            name="node_zero_arg_owned",
            handler_cls=HandlerZeroArg,
            topic="onex.evt.platform.zero-arg-owned.v1",
        ),
        _make_single_handler_contract(
            name="node_foreign_skip",
            handler_cls=HandlerForeignSkip,
            topic="onex.evt.platform.foreign-skip.v1",
        ),
    )
    return ModelAutoWiringManifest(contracts=contracts, errors=())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HANDLER_CLS_BY_NAME: dict[str, type] = {
    "HandlerNodeRegistryOwned": HandlerNodeRegistryOwned,
    "HandlerContainerOwned": HandlerContainerOwned,
    "HandlerEventBusOwned": HandlerEventBusOwned,
    "HandlerZeroArg": HandlerZeroArg,
    "HandlerForeignSkip": HandlerForeignSkip,
    "HandlerUnresolvable": HandlerUnresolvable,
}


def _import_handler_stub(module_path: str, class_name: str) -> type:
    """Stub replacement for ``_import_handler_class`` — bypasses importlib.

    The production import path tries ``importlib.import_module(module_path)``
    which fails inside the test module because ``tests.integration.runtime...``
    is not an installable package. Mapping class names to the definitions in
    this file keeps the test hermetic and avoids leaning on test-collection
    side effects.
    """
    if class_name not in _HANDLER_CLS_BY_NAME:
        raise ImportError(
            f"Test fixture has no handler class named {class_name!r}; "
            f"known: {sorted(_HANDLER_CLS_BY_NAME)!r}"
        )
    return _HANDLER_CLS_BY_NAME[class_name]


def _project_report(
    report: ModelAutoWiringReport,
) -> tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...]:
    """Stable structural projection of a wiring report.

    Excludes dispatcher IDs / route IDs (the dispatcher ID already
    contains the contract+handler names, and routes derive from topics —
    both stable), but includes the per-handler resolver outcomes and skip
    reasons which are the load-bearing evidence for determinism.
    """
    return tuple(
        (
            contract_result.contract_name,
            tuple(
                (
                    wiring.handler_name,
                    wiring.resolution_outcome.value,
                    wiring.skipped_reason,
                )
                for wiring in contract_result.wirings
            ),
        )
        for contract_result in report.results
    )


def _find_contract(
    report: ModelAutoWiringReport, name: str
) -> ModelContractWiringResult:
    for contract_result in report.results:
        if contract_result.contract_name == name:
            return contract_result
    raise AssertionError(
        f"Contract {name!r} missing from report — present: "
        f"{[r.contract_name for r in report.results]!r}"
    )


def _local_ownership_excluding(excluded: str) -> ServiceLocalHandlerOwnershipQuery:
    """Ownership query that owns every fixture node except ``excluded``.

    Patched in via ``ServiceLocalHandlerOwnershipQuery`` constructor so the
    production wire_from_manifest path still runs _assert_is_ownership_query.
    """
    return ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset(
            {
                "node_registry_owned",
                "node_container_owned",
                "node_event_bus_owned",
                "node_zero_arg_owned",
                "node_foreign_skip",
            }
            - {excluded}
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullAutoWiringExercisesResolver:
    @pytest.mark.asyncio
    async def test_all_precedence_branches_resolve_with_expected_outcomes(
        self,
    ) -> None:
        """Plan Task 8 centerpiece: every branch is exercised and recorded.

        Builds the full five-contract manifest, calls wire_from_manifest once,
        and asserts that the wiring report has one contract per branch with
        the exact ``EnumHandlerResolutionOutcome`` expected for that branch.
        This is the integration-level evidence that the resolver precedence
        chain is wired end-to-end into the runtime bootstrap surface.
        """
        manifest = _build_full_branch_manifest()
        engine = MessageDispatchEngine()
        container = _FakeContainer(
            registry={HandlerContainerOwned: HandlerContainerOwned(storage=object())}
        )
        event_bus = _FakeEventBus()
        materialized: dict[str, dict[str, object]] = {
            "HandlerNodeRegistryOwned": {"projection_reader": object()}
        }

        # The "foreign" handler's owning node is NOT in local_node_names;
        # every other contract's node IS owned here.
        foreign_ownership = _local_ownership_excluding("node_foreign_skip")

        with (
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "_import_handler_class",
                side_effect=_import_handler_stub,
            ),
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "ServiceLocalHandlerOwnershipQuery",
                return_value=foreign_ownership,
            ),
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "_prepare_contract_wiring",
                side_effect=_make_wrapped_prepare(materialized),
            ),
        ):
            report = await wire_from_manifest(
                manifest=manifest,
                dispatch_engine=engine,
                event_bus=event_bus,
                environment="test",
                container=container,
            )

        assert report.total_failed == 0, "No contract should fail in the fixture"

        # Registry branch
        registry_contract = _find_contract(report, "node_registry_owned")
        assert len(registry_contract.wirings) == 1
        assert registry_contract.wirings[0].handler_name == "HandlerNodeRegistryOwned"
        assert (
            registry_contract.wirings[0].resolution_outcome
            is EnumHandlerResolutionOutcome.RESOLVED_VIA_NODE_REGISTRY
        )
        assert registry_contract.outcome is EnumWiringOutcome.WIRED

        # Container branch
        container_contract = _find_contract(report, "node_container_owned")
        assert len(container_contract.wirings) == 1
        assert (
            container_contract.wirings[0].resolution_outcome
            is EnumHandlerResolutionOutcome.RESOLVED_VIA_CONTAINER
        )

        # Event-bus branch
        event_bus_contract = _find_contract(report, "node_event_bus_owned")
        assert len(event_bus_contract.wirings) == 1
        assert (
            event_bus_contract.wirings[0].resolution_outcome
            is EnumHandlerResolutionOutcome.RESOLVED_VIA_EVENT_BUS
        )

        # Zero-arg branch
        zero_arg_contract = _find_contract(report, "node_zero_arg_owned")
        assert len(zero_arg_contract.wirings) == 1
        assert (
            zero_arg_contract.wirings[0].resolution_outcome
            is EnumHandlerResolutionOutcome.RESOLVED_VIA_ZERO_ARG
        )

        # Skip branch
        foreign_contract = _find_contract(report, "node_foreign_skip")
        assert len(foreign_contract.wirings) == 1
        assert (
            foreign_contract.wirings[0].resolution_outcome
            is EnumHandlerResolutionOutcome.RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP
        )
        # Skip entries surface in skipped_handlers with a name + reason.
        assert len(foreign_contract.skipped_handlers) == 1
        assert foreign_contract.skipped_handlers[0].handler_name == "HandlerForeignSkip"
        assert "node_foreign_skip" in foreign_contract.skipped_handlers[0].reason
        # Skipped handler MUST NOT have produced a dispatcher registration.
        assert foreign_contract.dispatchers_registered == ()

        # Coverage assertion mirroring the plan's
        # test_non_registry_precedence_branches_exercised — the combined
        # outcome set must include at least one non-node_registry path.
        observed_outcomes = {
            wiring.resolution_outcome
            for contract_result in report.results
            for wiring in contract_result.wirings
        }
        non_registry_outcomes = observed_outcomes - {
            EnumHandlerResolutionOutcome.RESOLVED_VIA_NODE_REGISTRY,
        }
        assert non_registry_outcomes, (
            "Expected at least one handler to resolve via a non-node_registry "
            f"precedence path; observed: {observed_outcomes!r}"
        )
        # Stronger: observed set equals expected full-branch coverage.
        expected_outcomes = {
            EnumHandlerResolutionOutcome.RESOLVED_VIA_NODE_REGISTRY,
            EnumHandlerResolutionOutcome.RESOLVED_VIA_CONTAINER,
            EnumHandlerResolutionOutcome.RESOLVED_VIA_EVENT_BUS,
            EnumHandlerResolutionOutcome.RESOLVED_VIA_ZERO_ARG,
            EnumHandlerResolutionOutcome.RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP,
        }
        assert observed_outcomes == expected_outcomes, (
            "Fixture must exercise every precedence branch; missing: "
            f"{expected_outcomes - observed_outcomes!r}, "
            f"unexpected: {observed_outcomes - expected_outcomes!r}"
        )

    @pytest.mark.asyncio
    async def test_wire_from_manifest_report_is_deterministic(self) -> None:
        """Plan Task 8 acceptance: structural projection of two runs is equal.

        The resolver is pure on its context; wire_from_manifest is deterministic
        in handler ordering, skip reasons, and per-handler outcomes. A stale
        import cache, iteration-order drift, or set-in-dict dependency would
        make the two projections diverge.
        """
        manifest = _build_full_branch_manifest()
        foreign_ownership = _local_ownership_excluding("node_foreign_skip")
        materialized: dict[str, dict[str, object]] = {
            "HandlerNodeRegistryOwned": {"projection_reader": object()}
        }

        # Two independent runs on a fresh engine each time.
        report_a = await _run_wiring(manifest, foreign_ownership, materialized)
        report_b = await _run_wiring(manifest, foreign_ownership, materialized)

        projection_a = _project_report(report_a)
        projection_b = _project_report(report_b)
        assert projection_a == projection_b, (
            "Structural projection of two runs diverged — resolver ordering or "
            "reporting is non-deterministic.\n"
            f"A: {projection_a!r}\nB: {projection_b!r}"
        )

        # Aggregate counts must also match.
        assert report_a.total_wired == report_b.total_wired
        assert report_a.total_skipped == report_b.total_skipped
        assert report_a.total_failed == report_b.total_failed

    @pytest.mark.asyncio
    async def test_unresolvable_handler_raises_typeerror_no_report(self) -> None:
        """Plan Task 8 negative case: OMN-8735 fail-fast preserved.

        A single unresolvable handler in the manifest propagates ``TypeError``
        unchanged from the resolver's Step 6; no ``ModelAutoWiringReport``
        is produced. Proves the cutover in Task 5 did not weaken the
        fail-fast invariant that protects the runtime from booting with
        partially-wired handlers.
        """
        unresolvable_contract = _make_single_handler_contract(
            name="node_unresolvable",
            handler_cls=HandlerUnresolvable,
            topic="onex.evt.platform.unresolvable.v1",
        )
        manifest = ModelAutoWiringManifest(
            contracts=(unresolvable_contract,),
            errors=(),
        )
        engine = MessageDispatchEngine()

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            side_effect=_import_handler_stub,
        ):
            with pytest.raises(TypeError) as exc_info:
                await wire_from_manifest(
                    manifest=manifest,
                    dispatch_engine=engine,
                    event_bus=None,
                    environment="test",
                    container=None,
                )

        assert "some_undeclared_service" in str(exc_info.value), (
            "TypeError must name the missing constructor parameter so the "
            "runtime boot log identifies exactly which wiring gap crashed startup."
        )

    @pytest.mark.asyncio
    async def test_skipped_handler_appears_in_report_with_name_and_reason(
        self,
    ) -> None:
        """Plan Task 8 acceptance detail: skipped handlers surface in report.

        Separate from the headline branch-coverage test so failure-isolation
        on the skip path is unambiguous when CI flags a regression.
        """
        manifest = _build_full_branch_manifest()
        engine = MessageDispatchEngine()
        container = _FakeContainer(
            registry={HandlerContainerOwned: HandlerContainerOwned(storage=object())}
        )
        foreign_ownership = _local_ownership_excluding("node_foreign_skip")
        materialized: dict[str, dict[str, object]] = {
            "HandlerNodeRegistryOwned": {"projection_reader": object()}
        }

        with (
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "_import_handler_class",
                side_effect=_import_handler_stub,
            ),
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "ServiceLocalHandlerOwnershipQuery",
                return_value=foreign_ownership,
            ),
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "_prepare_contract_wiring",
                side_effect=_make_wrapped_prepare(materialized),
            ),
        ):
            report = await wire_from_manifest(
                manifest=manifest,
                dispatch_engine=engine,
                event_bus=_FakeEventBus(),
                environment="test",
                container=container,
            )

        foreign_contract = _find_contract(report, "node_foreign_skip")
        assert len(foreign_contract.skipped_handlers) == 1
        skipped = foreign_contract.skipped_handlers[0]
        assert skipped.handler_name == "HandlerForeignSkip"
        # Skip reason must carry the node name so operators can diagnose
        # "why is this handler not running here".
        assert "node_foreign_skip" in skipped.reason
        # The contract-level outcome is still WIRED (not FAILED) — skips
        # are a deliberate non-error state.
        assert foreign_contract.outcome is EnumWiringOutcome.WIRED

    @pytest.mark.asyncio
    async def test_non_registry_precedence_branches_exercised(self) -> None:
        """Plan Task 8 guard: at least ONE handler resolves via container OR zero-arg.

        Mirrors the plan's dedicated coverage test so the suite fails loudly
        if a future refactor accidentally routes every handler through the
        node-registry path. Complements the headline branch-coverage assertion
        with a narrower, single-purpose guard.
        """
        manifest = _build_full_branch_manifest()
        engine = MessageDispatchEngine()
        container = _FakeContainer(
            registry={HandlerContainerOwned: HandlerContainerOwned(storage=object())}
        )
        event_bus = _FakeEventBus()
        foreign_ownership = _local_ownership_excluding("node_foreign_skip")
        materialized: dict[str, dict[str, object]] = {
            "HandlerNodeRegistryOwned": {"projection_reader": object()}
        }

        with (
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "_import_handler_class",
                side_effect=_import_handler_stub,
            ),
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "ServiceLocalHandlerOwnershipQuery",
                return_value=foreign_ownership,
            ),
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "_prepare_contract_wiring",
                side_effect=_make_wrapped_prepare(materialized),
            ),
        ):
            report = await wire_from_manifest(
                manifest=manifest,
                dispatch_engine=engine,
                event_bus=event_bus,
                environment="test",
                container=container,
            )

        outcomes = {
            wiring.resolution_outcome.value
            for contract_result in report.results
            for wiring in contract_result.wirings
        }
        assert (
            "resolved_via_container" in outcomes or "resolved_via_zero_arg" in outcomes
        ), (
            "Integration suite did not exercise container or zero-arg resolver "
            f"paths; observed outcomes: {outcomes!r}"
        )


# ---------------------------------------------------------------------------
# Helpers to thread materialized deps into wire_from_manifest
# ---------------------------------------------------------------------------


def _make_wrapped_prepare(
    materialized: dict[str, dict[str, object]],
) -> Callable[..., object]:
    """Wrap ``_prepare_contract_wiring`` to pass materialized deps.

    ``wire_from_manifest`` does not yet expose a materialized-deps parameter —
    that ships in Task 6 (OMN-9198). Until that lands on main, this wrapper
    injects the map via the already-supported ``_prepare_contract_wiring``
    keyword so the RESOLVED_VIA_NODE_REGISTRY branch is genuinely exercised
    at integration level rather than only in the unit-level resolver tests.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _prepare_contract_wiring as _real_prepare,
    )

    def _wrapped(**kwargs: object) -> object:
        kwargs.setdefault("materialized_explicit_dependencies", materialized)
        return _real_prepare(**kwargs)  # type: ignore[arg-type]

    return _wrapped


async def _run_wiring(
    manifest: ModelAutoWiringManifest,
    ownership: ServiceLocalHandlerOwnershipQuery,
    materialized: dict[str, dict[str, object]],
) -> ModelAutoWiringReport:
    """Single-shot wiring invocation with a fresh engine + container.

    Extracted so the determinism test can call it twice with guaranteed
    fresh side-effect surfaces on each run.
    """
    engine = MessageDispatchEngine()
    container = _FakeContainer(
        registry={HandlerContainerOwned: HandlerContainerOwned(storage=object())}
    )
    event_bus = _FakeEventBus()

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            side_effect=_import_handler_stub,
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring."
            "ServiceLocalHandlerOwnershipQuery",
            return_value=ownership,
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring."
            "_prepare_contract_wiring",
            side_effect=_make_wrapped_prepare(materialized),
        ),
    ):
        return await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=engine,
            event_bus=event_bus,
            environment="test",
            container=container,
        )
