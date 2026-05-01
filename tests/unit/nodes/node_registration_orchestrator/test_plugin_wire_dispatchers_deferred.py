# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ServiceRegistration.wire_dispatchers registration ownership.

Verifies that the registration domain plugin defers dispatcher and route
wiring to generic contract-driven auto-wiring.

Historical context
------------------
Running both the legacy explicit dispatcher wiring path and the generic
contract auto-wiring path against the same registration contract produced
``ONEX_CORE_064_DUPLICATE_REGISTRATION`` errors for dispatcher
``dispatcher.registration.node-introspected`` on fresh runtime-effects boots.

These tests pin the corrected ownership invariant: the plugin's
``wire_dispatchers`` method does not invoke explicit dispatcher wiring or
register plugin-local routes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import omnibase_infra.nodes.node_registration_orchestrator.plugin as plugin_module
from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
    ServiceRegistration,
)
from omnibase_infra.runtime.models.model_domain_plugin_config import (
    ModelDomainPluginConfig,
)

_WIRING_MOD = "omnibase_infra.nodes.node_registration_orchestrator.wiring"


@dataclass
class _DummyNodeIdentity:
    """Minimal node identity stand-in; only attribute access is exercised."""

    env: str = "test"
    service: str = "test-service"
    node_name: str = "test-service"
    version: str = "v1"


def _make_plugin_config() -> ModelDomainPluginConfig:
    """Build a plugin config with the fields required by wire_dispatchers().

    The implementation reads ``container``, ``dispatch_engine``,
    ``event_bus``, and ``correlation_id`` before delegating to the explicit
    wiring helper.
    """
    container = MagicMock()
    container.service_registry = MagicMock()
    event_bus = MagicMock()
    dispatch_engine = MagicMock()
    dispatch_engine.register_dispatcher = MagicMock()
    dispatch_engine.register_route = MagicMock()

    return ModelDomainPluginConfig(
        container=container,
        event_bus=event_bus,
        correlation_id=uuid4(),
        input_topic="onex.evt.platform.node-introspection.v1",
        output_topic="onex.evt.platform.node-registration-accepted.v1",
        consumer_group="onex-runtime-test",
        dispatch_engine=dispatch_engine,
        node_identity=_DummyNodeIdentity(),  # type: ignore[arg-type]
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wire_dispatchers_returns_auto_wiring_skip_result() -> None:
    """wire_dispatchers() returns success with an auto-wiring skip reason."""
    plugin = ServiceRegistration()
    config = _make_plugin_config()

    result = await plugin.wire_dispatchers(config)

    assert result.success is True
    assert result.plugin_id == "registration"
    assert "skipped" in result.message.lower()
    assert "auto-wiring" in result.message.lower()
    assert result.services_registered == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wire_dispatchers_does_not_touch_dispatch_engine() -> None:
    """wire_dispatchers() must not register plugin-local dispatchers or routes."""
    plugin = ServiceRegistration()
    config = _make_plugin_config()

    result = await plugin.wire_dispatchers(config)

    assert result.success is True
    assert config.dispatch_engine is not None
    config.dispatch_engine.register_dispatcher.assert_not_called()
    config.dispatch_engine.register_route.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wire_dispatchers_skips_without_dispatch_engine() -> None:
    """wire_dispatchers() does not require a dispatch engine when deferred."""
    plugin = ServiceRegistration()
    config = _make_plugin_config()
    config.dispatch_engine = None

    result = await plugin.wire_dispatchers(config)

    assert result.success is True
    assert result.error_message is None
    assert "auto-wiring" in result.message.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wire_dispatchers_repeat_calls_remain_idempotent_skips() -> None:
    """Repeat calls return the same deferred ownership result."""
    plugin = ServiceRegistration()
    config = _make_plugin_config()

    first = await plugin.wire_dispatchers(config)
    second = await plugin.wire_dispatchers(config)

    assert first.success is True
    assert second.success is True
    assert first.services_registered == []
    assert second.services_registered == []


@pytest.mark.unit
def test_wire_dispatchers_ast_has_no_explicit_dispatch_registration() -> None:
    """wire_dispatchers() must not call dispatcher/route registration APIs."""
    source = Path(plugin_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    service_cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ServiceRegistration"
    )
    wire_dispatchers = next(
        node
        for node in service_cls.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "wire_dispatchers"
    )

    attrs = {
        node.attr
        for node in ast.walk(wire_dispatchers)
        if isinstance(node, ast.Attribute)
    }
    names = {
        node.id for node in ast.walk(wire_dispatchers) if isinstance(node, ast.Name)
    }

    assert "register_dispatcher" not in attrs
    assert "register_route" not in attrs
    assert "wire_registration_dispatchers" not in names


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_consumers_skips_plugin_local_event_bus_wiring() -> None:
    """start_consumers() must not create a second registration consumer family.

    Generic contract auto-wiring owns registration subscriptions. The plugin
    only prepares the DispatchResultApplier needed to apply outputs from the
    auto-wired callbacks.
    """
    plugin = ServiceRegistration()
    plugin._handler_wiring_succeeded = True
    config = _make_plugin_config()
    assert config.container.service_registry is not None
    config.container.service_registry.register_instance = AsyncMock()

    with (
        patch.object(plugin, "_wire_intent_effects", new=AsyncMock()),
        patch(
            "omnibase_infra.runtime.event_bus_subcontract_wiring.EventBusSubcontractWiring",
        ) as wiring_cls,
    ):
        result = await plugin.start_consumers(config)

    assert result.success is True
    assert "auto-wiring" in result.message.lower()
    assert plugin.result_applier is not None
    wiring_cls.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_on_failure_clears_cached_result_applier() -> None:
    """Failed initialization must not leave stale output wiring cached."""
    plugin = ServiceRegistration()
    plugin._result_applier = MagicMock()  # type: ignore[assignment]

    await plugin._cleanup_on_failure(_make_plugin_config())  # type: ignore[attr-defined]

    assert plugin.result_applier is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_clears_cached_result_applier() -> None:
    """Shutdown must force the next boot to rebuild DI/effect wiring."""
    plugin = ServiceRegistration()
    plugin._result_applier = MagicMock()  # type: ignore[assignment]

    result = await plugin._do_shutdown(_make_plugin_config())  # type: ignore[attr-defined]

    assert result.success is True
    assert plugin.result_applier is None
