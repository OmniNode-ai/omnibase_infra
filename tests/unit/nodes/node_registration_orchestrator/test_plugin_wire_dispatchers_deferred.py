# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ServiceRegistration.wire_dispatchers deferral (OMN-9456).

Verifies that the registration domain plugin defers dispatcher and route
wiring to the generic contract-driven auto-wiring pipeline, rather than
calling the legacy explicit ``wire_registration_dispatchers`` helper.

Historical context
------------------
Running both the legacy explicit dispatcher wiring path and the generic
contract auto-wiring path against the same registration contract produced
``ONEX_CORE_064_DUPLICATE_REGISTRATION`` errors for dispatcher
``dispatcher.registration.node-introspected`` on fresh runtime-effects boots.

These tests pin the single-authority invariant: the plugin's
``wire_dispatchers`` method MUST return a skipped result and MUST NOT
invoke the explicit wiring helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
    ServiceRegistration,
)
from omnibase_infra.runtime.models.model_domain_plugin_config import (
    ModelDomainPluginConfig,
)

_PLUGIN_MOD = "omnibase_infra.nodes.node_registration_orchestrator.plugin"
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

    The legacy implementation read ``container.service_registry``,
    ``dispatch_engine``, ``event_bus``, and ``correlation_id`` before
    delegating to the explicit wiring helper. The deferred implementation
    only needs ``correlation_id`` for logging; the remaining fields are
    populated with plausible doubles so the invariant holds regardless of
    whether the plugin inspects them defensively in the future.
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
async def test_wire_dispatchers_returns_skipped_result() -> None:
    """wire_dispatchers() must return a skipped success result.

    Generic contract auto-wiring is the single authority for registration
    dispatchers and routes after OMN-9456; the plugin's wire_dispatchers()
    is a pass-through.
    """
    plugin = ServiceRegistration()
    config = _make_plugin_config()

    result = await plugin.wire_dispatchers(config)

    # Skipped results carry success=True so the kernel does not treat them
    # as failure (see ModelDomainPluginResult.skipped semantics).
    assert result.success is True
    assert result.plugin_id == "registration"
    assert result.message.lower().startswith("plugin registration skipped")
    # The reason must make the single-authority intent explicit for
    # operators reading kernel logs.
    reason_lower = result.message.lower()
    assert "auto-wiring" in reason_lower or "auto wiring" in reason_lower


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wire_dispatchers_does_not_invoke_legacy_helper() -> None:
    """wire_dispatchers() must NOT call wire_registration_dispatchers().

    The legacy helper is what registered the colliding
    ``dispatcher.registration.node-introspected`` dispatcher ID alongside
    the generic auto-wiring path. Deferring means the helper is not
    invoked on the kernel-native activation path at all.
    """
    plugin = ServiceRegistration()
    config = _make_plugin_config()

    with patch(
        f"{_WIRING_MOD}.wire_registration_dispatchers",
        new=AsyncMock(),
    ) as mock_legacy_wire:
        await plugin.wire_dispatchers(config)

    mock_legacy_wire.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wire_dispatchers_does_not_touch_dispatch_engine() -> None:
    """wire_dispatchers() must register zero dispatchers and zero routes.

    This guards against regressions where the plugin accidentally
    reintroduces direct ``dispatch_engine.register_dispatcher`` /
    ``register_route`` calls — the exact class of bug that produced the
    duplicate-registration error in OMN-9456.
    """
    plugin = ServiceRegistration()
    config = _make_plugin_config()
    engine = config.dispatch_engine
    assert engine is not None  # mypy/runtime guard — see _make_plugin_config

    await plugin.wire_dispatchers(config)

    engine.register_dispatcher.assert_not_called()  # type: ignore[attr-defined]
    engine.register_route.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wire_dispatchers_is_idempotent_across_repeat_calls() -> None:
    """Calling wire_dispatchers() twice is safe — no duplicate registrations.

    The kernel-native activation path invokes registration_service
    .wire_dispatchers() explicitly; a defensive second call (e.g. from a
    retry or a test harness) must not cause the duplicate-registration
    error recorded in the ticket. Because the method is a pure skip, this
    is trivially true — this test pins the invariant.
    """
    plugin = ServiceRegistration()
    config = _make_plugin_config()

    first = await plugin.wire_dispatchers(config)
    second = await plugin.wire_dispatchers(config)

    assert first.success is True
    assert second.success is True
    engine = config.dispatch_engine
    assert engine is not None
    engine.register_dispatcher.assert_not_called()  # type: ignore[attr-defined]
