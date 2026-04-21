# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for async container pre-resolution in wire_from_manifest [OMN-9410].

Reproduces the EFFECTS profile crash where container.get_service() (sync) calls
asyncio.run() inside a running event loop, raising RuntimeError. The fix adds a
Phase 0 async pre-resolution step that calls get_service_async() before the sync
_prepare_contract_wiring loop, bypassing asyncio.run() entirely.

This test proves:
  1. _async_resolve_from_container returns an instance when get_service_async succeeds.
  2. _async_resolve_from_container returns None on ServiceResolutionError (miss).
  3. wire_from_manifest wires a handler correctly when the container's sync
     get_service raises RuntimeError (simulating asyncio.run() in running loop)
     but get_service_async succeeds.
  4. wire_from_manifest still wires zero-arg handlers (no container resolution needed).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_core.errors.error_service_resolution import ServiceResolutionError
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _async_resolve_from_container,
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
from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler_cls(*, requires_deps: bool = False) -> type:
    """Return a handler class — zero-arg if requires_deps=False, else with a required dep."""
    if requires_deps:

        class HandlerWithDeps:
            def __init__(self, dep_service: object) -> None:
                self.dep_service = dep_service

            async def handle(self, envelope: object) -> None:
                return None

        return HandlerWithDeps
    else:

        class HandlerZeroArg:
            async def handle(self, envelope: object) -> None:
                return None

        return HandlerZeroArg


def _make_contract(
    *,
    node_name: str,
    handler_module: str,
    handler_name: str,
    topic: str = "onex.evt.test.event.v1",
) -> ModelDiscoveredContract:
    from pathlib import Path

    entry = ModelHandlerRoutingEntry(
        handler=ModelHandlerRef(name=handler_name, module=handler_module),
        event_model=ModelHandlerRef(name="ModelTestEvent", module="fake.models"),
        operation=None,
    )
    return ModelDiscoveredContract(
        name=node_name,
        node_type="ORCHESTRATOR_GENERIC",
        package_name="test_pkg",
        entry_point_name=node_name,
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(entry,),
        ),
        event_bus=ModelEventBusWiring(
            subscribe_topics=(topic,),
            publish_topics=(),
        ),
    )


def _make_dispatch_engine() -> MagicMock:
    engine = MagicMock()
    engine._routes = {}
    engine._container = None
    engine.register_dispatcher = MagicMock()
    engine.register_route = MagicMock()
    engine.freeze = MagicMock()
    return engine


def _make_event_bus() -> MagicMock:
    bus = MagicMock()
    bus.subscribe = AsyncMock()
    return bus


# ---------------------------------------------------------------------------
# _async_resolve_from_container tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_resolve_from_container_returns_instance_on_success() -> None:
    """get_service_async succeeding returns the resolved instance."""
    expected = object()
    container = MagicMock()
    container.get_service_async = AsyncMock(return_value=expected)

    result = await _async_resolve_from_container(container, type(expected))

    assert result is expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_resolve_from_container_returns_none_on_service_resolution_error() -> (
    None
):
    """ServiceResolutionError (service not registered) returns None (not an error)."""
    container = MagicMock()
    container.get_service_async = AsyncMock(
        side_effect=ServiceResolutionError("not registered")
    )

    result = await _async_resolve_from_container(container, object)

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_resolve_from_container_returns_none_if_no_get_service_async() -> (
    None
):
    """Containers without get_service_async return None gracefully."""
    container = MagicMock(spec=[])  # no get_service_async attribute

    result = await _async_resolve_from_container(container, object)

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_resolve_from_container_propagates_unexpected_errors() -> None:
    """Non-ServiceResolutionError exceptions propagate (container-internal bugs)."""
    container = MagicMock()
    container.get_service_async = AsyncMock(side_effect=ValueError("unexpected"))

    with pytest.raises(ValueError, match="unexpected"):
        await _async_resolve_from_container(container, object)


# ---------------------------------------------------------------------------
# wire_from_manifest async pre-resolution tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wire_from_manifest_uses_async_pre_resolution_when_sync_would_fail() -> (
    None
):
    """wire_from_manifest wires a handler via get_service_async even when
    get_service (sync) raises RuntimeError (simulating asyncio.run in running loop).

    This is the OMN-9410 regression test: the EFFECTS profile crash.
    """
    HandlerCls = _make_handler_cls(requires_deps=True)
    resolved_instance = HandlerCls(dep_service=object())

    contract = _make_contract(
        node_name="node_test_orchestrator",
        handler_module="omnibase_infra.runtime.auto_wiring.test_handler_wiring_async_container_resolution",
        handler_name="HandlerCls",
        topic="onex.evt.test.orchestrator.v1",
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    container = MagicMock()
    container.get_service = MagicMock(
        side_effect=RuntimeError("This event loop is already running")
    )
    container.get_service_async = AsyncMock(return_value=resolved_instance)

    dispatch_engine = _make_dispatch_engine()
    event_bus = _make_event_bus()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerCls,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            container=container,
        )

    assert report.total_failed == 0, f"Expected no failures, got: {report.results}"
    assert report.total_wired == 1, f"Expected 1 wired, got: {report.total_wired}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wire_from_manifest_zero_arg_handler_wires_without_container() -> None:
    """Zero-arg handlers wire correctly even with no container at all."""
    HandlerCls = _make_handler_cls(requires_deps=False)

    contract = _make_contract(
        node_name="node_zero_arg",
        handler_module="fake.module",
        handler_name="HandlerZeroArg",
        topic="onex.evt.test.zero.v1",
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    dispatch_engine = _make_dispatch_engine()
    event_bus = _make_event_bus()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerCls,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            container=None,
        )

    assert report.total_failed == 0
    assert report.total_wired == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wire_from_manifest_pre_resolution_miss_falls_through_to_resolver() -> (
    None
):
    """When get_service_async raises ServiceResolutionError (miss), the resolver
    falls through to zero-arg construction if the handler has no required deps.
    """
    HandlerCls = _make_handler_cls(requires_deps=False)

    contract = _make_contract(
        node_name="node_miss_then_zero_arg",
        handler_module="fake.module",
        handler_name="HandlerZeroArg",
        topic="onex.evt.test.miss.v1",
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    container = MagicMock()
    container.get_service_async = AsyncMock(side_effect=ServiceResolutionError("miss"))

    dispatch_engine = _make_dispatch_engine()
    event_bus = _make_event_bus()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerCls,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            container=container,
        )

    assert report.total_failed == 0
    assert report.total_wired == 1
