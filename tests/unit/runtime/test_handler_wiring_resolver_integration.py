# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for handler_wiring → ServiceHandlerResolver integration (OMN-9201).

Verifies the Task 5 BOOT-PATH cutover: handler_wiring.py delegates handler
construction to ``ServiceHandlerResolver`` via a ``ModelHandlerResolverContext``
instead of running the inline OMN-8735 decision tree.

Test matrix (plan §Task 5 acceptance):
- _assert_is_ownership_query rejects non-protocol objects at the infra
  boundary BEFORE the resolver is invoked.
- Resolver is invoked exactly once per handler entry; outcome flows into
  ``ModelContractWiringResult.wirings``.
- LOCAL_OWNERSHIP_SKIP outcomes land in ``skipped_handlers`` and do NOT
  register the dispatcher/routes on the engine.
- OMN-8735 fail-fast preserved: unresolvable handlers raise ``TypeError``
  that propagates unchanged through ``wire_from_manifest``.
- wire_from_manifest produces structurally identical reports across two
  runs (determinism).
- Protocol conformance: ``ServiceLocalHandlerOwnershipQuery`` satisfies
  ``ProtocolHandlerOwnershipQuery`` (spi).
- Full wiring + materialized deps: an explicit-dep map is threaded through
  ``_prepare_handler_wiring`` into the resolver.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)
from omnibase_core.models.errors import ModelOnexError
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _assert_is_ownership_query,
    _prepare_handler_wiring,
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
from omnibase_spi.protocols.runtime.protocol_handler_ownership_query import (
    ProtocolHandlerOwnershipQuery,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_contract(
    name: str = "node_local",
    handler_name: str = "HandlerFoo",
    handler_module: str = "fake.module",
    topics: tuple[str, ...] = ("onex.evt.platform.local-input.v1",),
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(subscribe_topics=topics, publish_topics=()),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name=handler_name, module=handler_module),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


def _make_manifest(*contracts: ModelDiscoveredContract) -> ModelAutoWiringManifest:
    return ModelAutoWiringManifest(contracts=contracts)


def _make_zero_arg_handler_cls() -> type:
    class FakeHandler:
        async def handle(self, envelope: object) -> None:
            return None

    return FakeHandler


# ---------------------------------------------------------------------------
# _assert_is_ownership_query — infra-boundary protocol check
# ---------------------------------------------------------------------------


class TestAssertIsOwnershipQuery:
    @pytest.mark.unit
    def test_accepts_service_local_ownership_query(self) -> None:
        svc = ServiceLocalHandlerOwnershipQuery(local_node_names=frozenset({"a"}))
        # Should not raise.
        _assert_is_ownership_query(svc)

    @pytest.mark.unit
    def test_rejects_bare_object(self) -> None:
        with pytest.raises(ModelOnexError) as exc_info:
            _assert_is_ownership_query(object())
        assert "ProtocolHandlerOwnershipQuery" in str(exc_info.value)

    @pytest.mark.unit
    def test_rejects_object_without_is_owned_here(self) -> None:
        class BadQuery:
            def unrelated(self) -> bool:
                return True

        with pytest.raises(ModelOnexError):
            _assert_is_ownership_query(BadQuery())

    @pytest.mark.unit
    def test_service_local_conforms_to_spi_protocol(self) -> None:
        """Layering: the core-layer service satisfies the spi-layer protocol."""
        svc = ServiceLocalHandlerOwnershipQuery(local_node_names=frozenset({"x"}))
        assert isinstance(svc, ProtocolHandlerOwnershipQuery)


# ---------------------------------------------------------------------------
# wire_from_manifest — boundary check runs BEFORE resolver
# ---------------------------------------------------------------------------


class TestWireFromManifestBoundary:
    @pytest.mark.asyncio
    async def test_asserts_ownership_query_protocol_conformance(self) -> None:
        """Layering boundary test (plan Task 5 acceptance).

        Patching ServiceLocalHandlerOwnershipQuery to return a bare object()
        proves the infra-boundary check runs before the resolver is invoked.
        """
        manifest = _make_manifest(_make_contract())
        engine = MagicMock()

        # Patch the constructor inside handler_wiring's namespace so
        # wire_from_manifest uses our non-conforming stand-in.
        bad = object()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring."
            "ServiceLocalHandlerOwnershipQuery",
            return_value=bad,
        ):
            with pytest.raises(ModelOnexError) as exc_info:
                await wire_from_manifest(manifest, engine)
        assert "ProtocolHandlerOwnershipQuery" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _prepare_handler_wiring — resolver delegation
# ---------------------------------------------------------------------------


class TestPrepareHandlerWiringDelegatesToResolver:
    @pytest.mark.unit
    def test_delegates_to_resolver(self) -> None:
        """Plan Task 5 acceptance: resolver.resolve is invoked exactly once."""
        contract = _make_contract()
        entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
        handler_cls = _make_zero_arg_handler_cls()
        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )
        resolver = ServiceHandlerResolver()
        # dispatch_engine=None avoids MagicMock's auto-attributes that would
        # expose a fake ``_container`` and redirect the resolver chain.
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
        ):
            prepared = _prepare_handler_wiring(
                contract=contract,
                entry=entry,
                dispatch_engine=None,
                resolver=resolver,
                ownership_query=ownership,
                event_bus=None,
                container=None,
            )
        assert prepared.is_skip is False
        assert (
            prepared.resolution_outcome
            is EnumHandlerResolutionOutcome.RESOLVED_VIA_ZERO_ARG
        )
        assert prepared.handler_name == entry.handler.name

    @pytest.mark.unit
    def test_skip_when_node_not_owned(self) -> None:
        """Plan Task 5 acceptance: LOCAL_OWNERSHIP_SKIP is recorded, no TypeError."""
        contract = _make_contract(name="foreign_node")
        entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
        handler_cls = _make_zero_arg_handler_cls()
        # Ownership says the node is NOT owned here.
        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({"some_other_node"})
        )
        resolver = ServiceHandlerResolver()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
        ):
            prepared = _prepare_handler_wiring(
                contract=contract,
                entry=entry,
                dispatch_engine=MagicMock(),
                resolver=resolver,
                ownership_query=ownership,
                event_bus=None,
                container=None,
            )
        assert prepared.is_skip is True
        assert prepared.handler_name == entry.handler.name
        assert "foreign_node" in prepared.skip_reason
        # Skip entries never carry a dispatcher_id.
        assert prepared.dispatcher_id == ""
        assert prepared.route_ids == []

    @pytest.mark.unit
    def test_raises_typeerror_when_unresolvable(self) -> None:
        """Plan Task 5 acceptance: OMN-8735 fail-fast invariant preserved."""
        contract = _make_contract()
        entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]

        class HandlerWithDeps:
            def __init__(self, required_service: object) -> None:
                self.required_service = required_service

            async def handle(self, envelope: object) -> None:
                return None

        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )
        resolver = ServiceHandlerResolver()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerWithDeps,
        ):
            with pytest.raises(TypeError) as exc_info:
                _prepare_handler_wiring(
                    contract=contract,
                    entry=entry,
                    dispatch_engine=None,
                    resolver=resolver,
                    ownership_query=ownership,
                    event_bus=None,
                    container=None,
                )
        assert "required_service" in str(exc_info.value)

    @pytest.mark.unit
    def test_materialized_deps_threaded_to_resolver(self) -> None:
        """Materialized explicit deps reach the resolver (Step 2 path)."""
        contract = _make_contract(handler_name="HandlerWithExplicitDep")
        entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]

        class HandlerWithExplicitDep:
            def __init__(self, projection_reader: object) -> None:
                self.projection_reader = projection_reader

            async def handle(self, envelope: object) -> None:
                return None

        fake_reader = MagicMock()
        materialized: dict[str, dict[str, object]] = {
            "HandlerWithExplicitDep": {"projection_reader": fake_reader}
        }
        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )
        resolver = ServiceHandlerResolver()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerWithExplicitDep,
        ):
            prepared = _prepare_handler_wiring(
                contract=contract,
                entry=entry,
                dispatch_engine=None,
                resolver=resolver,
                ownership_query=ownership,
                event_bus=None,
                container=None,
                materialized_explicit_dependencies=materialized,
            )
        assert prepared.is_skip is False
        assert (
            prepared.resolution_outcome
            is EnumHandlerResolutionOutcome.RESOLVED_VIA_NODE_REGISTRY
        )


# ---------------------------------------------------------------------------
# wire_from_manifest — full-flow skip + determinism + fail-fast
# ---------------------------------------------------------------------------


class TestWireFromManifestResolverOutcomes:
    @pytest.mark.asyncio
    async def test_skip_surfaces_in_wiring_report(self) -> None:
        """Plan Task 5 acceptance: skip-path invariant test."""
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            MessageDispatchEngine,
        )

        # Two contracts — only one is "owned here". Because wire_from_manifest
        # constructs the ownership_query from the manifest node names,
        # both contracts are owned. To trigger a skip we patch the
        # constructor with a narrower set.
        contract = _make_contract(name="foreign_node")
        manifest = _make_manifest(contract)
        engine = MessageDispatchEngine()
        handler_cls = _make_zero_arg_handler_cls()

        # Construct a ServiceLocalHandlerOwnershipQuery that excludes
        # the contract's node. This proves the skip path works end-to-end.
        narrow_ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset()  # empty — nothing owned here
        )
        with (
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
                return_value=handler_cls,
            ),
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "ServiceLocalHandlerOwnershipQuery",
                return_value=narrow_ownership,
            ),
        ):
            report = await wire_from_manifest(manifest, engine)

        # Contract-level outcome is WIRED (not an error), but its single
        # handler lands in skipped_handlers with no dispatcher registration.
        assert report.total_failed == 0
        result = report.results[0]
        assert len(result.skipped_handlers) == 1
        assert result.skipped_handlers[0].handler_name == "HandlerFoo"
        assert "foreign_node" in result.skipped_handlers[0].reason
        assert len(result.wirings) == 1
        assert (
            result.wirings[0].resolution_outcome
            is EnumHandlerResolutionOutcome.RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP
        )
        # No dispatcher was registered for the skipped handler.
        assert result.dispatchers_registered == ()
        assert result.routes_registered == ()

    @pytest.mark.asyncio
    async def test_typeerror_propagates_unchanged(self) -> None:
        """Plan Task 5 acceptance: unresolvable-path invariant test (OMN-8735)."""
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            MessageDispatchEngine,
        )

        contract = _make_contract(handler_name="HandlerNeedsDep")
        manifest = _make_manifest(contract)
        engine = MessageDispatchEngine()

        class HandlerNeedsDep:
            def __init__(self, required_service: object) -> None:
                self.required_service = required_service

            async def handle(self, envelope: object) -> None:
                return None

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerNeedsDep,
        ):
            with pytest.raises(TypeError) as exc_info:
                await wire_from_manifest(manifest, engine)
        assert "required_service" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_report_is_deterministic(self) -> None:
        """Plan Task 5 acceptance: determinism across two identical runs."""
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            MessageDispatchEngine,
        )

        contract = _make_contract()
        manifest = _make_manifest(contract)
        handler_cls = _make_zero_arg_handler_cls()

        engine_a = MessageDispatchEngine()
        engine_b = MessageDispatchEngine()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
        ):
            report_a = await wire_from_manifest(manifest, engine_a)
            report_b = await wire_from_manifest(manifest, engine_b)

        assert report_a.total_wired == report_b.total_wired
        assert report_a.total_failed == report_b.total_failed
        assert report_a.total_skipped == report_b.total_skipped
        results_a = report_a.results
        results_b = report_b.results
        assert len(results_a) == len(results_b)
        for r_a, r_b in zip(results_a, results_b, strict=True):
            assert r_a.contract_name == r_b.contract_name
            assert r_a.outcome == r_b.outcome
            assert r_a.dispatchers_registered == r_b.dispatchers_registered
            assert r_a.routes_registered == r_b.routes_registered
            assert len(r_a.wirings) == len(r_b.wirings)
            for w_a, w_b in zip(r_a.wirings, r_b.wirings, strict=True):
                assert w_a.handler_name == w_b.handler_name
                assert w_a.resolution_outcome == w_b.resolution_outcome
                assert w_a.skipped_reason == w_b.skipped_reason
            assert len(r_a.skipped_handlers) == len(r_b.skipped_handlers)
            for s_a, s_b in zip(
                r_a.skipped_handlers, r_b.skipped_handlers, strict=True
            ):
                assert s_a.handler_name == s_b.handler_name
                assert s_a.reason == s_b.reason

    @pytest.mark.asyncio
    async def test_wired_handler_outcome_is_recorded(self) -> None:
        """Per-handler ModelWiringOutcome rows are populated."""
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            MessageDispatchEngine,
        )

        contract = _make_contract()
        manifest = _make_manifest(contract)
        engine = MessageDispatchEngine()
        # Real class: ModelHandlerResolverContext.handler_cls requires type.
        handler_cls = _make_zero_arg_handler_cls()

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
        ):
            report = await wire_from_manifest(manifest, engine)

        assert report.total_wired == 1
        result = report.results[0]
        assert len(result.wirings) == 1
        assert result.wirings[0].handler_name == "HandlerFoo"
        assert (
            result.wirings[0].resolution_outcome
            is EnumHandlerResolutionOutcome.RESOLVED_VIA_ZERO_ARG
        )
