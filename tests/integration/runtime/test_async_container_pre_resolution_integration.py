# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: async container pre-resolution end-to-end [OMN-9410].

Proves that wire_from_manifest completes successfully when container.get_service
(sync) raises RuntimeError (simulating asyncio.run() inside a running event loop)
but container.get_service_async succeeds.

This is the regression test for the omninode-runtime-effects crash:
  HandlerNodeRegistrationAcked and HandlerThreadWatcher raised TypeError on every
  startup because ServiceHandlerResolver.resolve() called get_service_sync() which
  called asyncio.run() inside wire_from_manifest's running event loop.

The integration test builds a minimal in-process manifest (no real contract discovery)
and exercises the full wire_from_manifest path with a container mock that:
  - raises RuntimeError on get_service (sync path)
  - returns a resolved instance on get_service_async (async path)

Acceptance: report.total_wired == 1, report.total_failed == 0.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_core.errors.error_service_resolution import ServiceResolutionError
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


class _HandlerWithDeps:
    def __init__(self, dep_service: object) -> None:
        self.dep_service = dep_service

    async def handle(self, envelope: object) -> None:
        return None


def _make_manifest() -> ModelAutoWiringManifest:
    entry = ModelHandlerRoutingEntry(
        handler=ModelHandlerRef(
            name="_HandlerWithDeps",
            module="tests.integration.runtime.test_async_container_pre_resolution_integration",
        ),
        event_model=ModelHandlerRef(
            name="ModelNodeRegistrationAcked", module="fake.models"
        ),
        operation=None,
    )
    contract = ModelDiscoveredContract(
        name="node_effects_orchestrator",
        node_type="ORCHESTRATOR_GENERIC",
        package_name="omnibase_infra",
        entry_point_name="node_effects_orchestrator",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/effects_contract.yaml"),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(entry,),
        ),
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.runtime.node-registration-acked.v1",),
            publish_topics=(),
        ),
    )
    return ModelAutoWiringManifest(contracts=(contract,), errors=())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_pre_resolution_succeeds_when_sync_raises_runtime_error() -> None:
    """Full wire_from_manifest path: get_service (sync) raises RuntimeError,
    get_service_async succeeds — handler wires without failure [OMN-9410].
    """
    resolved_instance = _HandlerWithDeps(dep_service=object())
    manifest = _make_manifest()

    container = MagicMock()
    container.get_service = MagicMock(
        side_effect=RuntimeError("This event loop is already running")
    )
    container.get_service_async = AsyncMock(return_value=resolved_instance)

    event_bus = MagicMock()
    event_bus.subscribe = AsyncMock()

    dispatch_engine = MagicMock()
    dispatch_engine._routes = {}
    dispatch_engine._container = None
    dispatch_engine.register_dispatcher = MagicMock()
    dispatch_engine.register_route = MagicMock()
    dispatch_engine.freeze = MagicMock()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_HandlerWithDeps,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            container=container,
        )

    assert report.total_failed == 0, f"Wiring failures: {report.results}"
    assert report.total_wired == 1, f"Expected 1 wired, got {report.total_wired}"
    container.get_service_async.assert_called_once()
    container.get_service.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_pre_resolution_miss_falls_through_to_zero_arg() -> None:
    """When get_service_async misses (ServiceResolutionError), a zero-arg handler
    wires via the zero-arg fallback path — no failure [OMN-9410].
    """

    class _HandlerZeroArg:
        async def handle(self, envelope: object) -> None:
            return None

    entry = ModelHandlerRoutingEntry(
        handler=ModelHandlerRef(name="_HandlerZeroArg", module="fake.module"),
        event_model=ModelHandlerRef(name="ModelTestEvent", module="fake.models"),
        operation=None,
    )
    contract = ModelDiscoveredContract(
        name="node_effects_zero_arg",
        node_type="ORCHESTRATOR_GENERIC",
        package_name="omnibase_infra",
        entry_point_name="node_effects_zero_arg",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/zero_arg_contract.yaml"),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(entry,),
        ),
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.test.zero.v1",),
            publish_topics=(),
        ),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    container = MagicMock()
    container.get_service = MagicMock(side_effect=ServiceResolutionError("sync miss"))
    container.get_service_async = AsyncMock(side_effect=ServiceResolutionError("miss"))

    event_bus = MagicMock()
    event_bus.subscribe = AsyncMock()
    dispatch_engine = MagicMock()
    dispatch_engine._routes = {}
    dispatch_engine._container = None

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_HandlerZeroArg,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            container=container,
        )

    assert report.total_failed == 0
    assert report.total_wired == 1
    container.get_service_async.assert_called_once_with(_HandlerZeroArg)
    container.get_service.assert_called_once_with(_HandlerZeroArg)
