# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Smoke plugin fixture and seam test for the discovery contract (OMN-2021).

This module provides:

1. **PluginSmoke** -- a minimal plugin that satisfies the full
   ``ProtocolDomainPlugin`` protocol while doing nothing. It exists to prove
   the discovery-to-activation contract is consumable by a kernel-like harness
   without importing or depending on the real kernel.

2. **Seam tests** -- tests that exercise the exact same iteration pattern the
   kernel uses (``registry.get_all()`` -> ``plugin.should_activate(config)`` ->
   ``plugin.initialize(config)`` -> ...) against the smoke plugin. This ensures
   the discovery output is consumable by any kernel-like loop, preventing
   domain-specific assumptions from leaking into the registry.

The smoke plugin intentionally avoids inheriting from any concrete base class
or depending on domain-specific models. It serves as the de facto reference
implementation for external plugin authors.

Design:
    The smoke plugin is NOT ``PluginRegistration`` -- it avoids real-plugin-
    specific assumptions leaking into the test. Only the protocol contract
    matters.

Related:
    - OMN-2017: ``discover_from_entry_points()`` on ``RegistryDomainPlugin``
    - OMN-2000: Contract-driven domain plugin discovery via entry_points
    - OMN-1346: Registration Code Extraction
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.runtime.models import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
)
from omnibase_infra.runtime.protocol_domain_plugin import (
    ProtocolDomainPlugin,
    RegistryDomainPlugin,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Smoke plugin fixture
# ---------------------------------------------------------------------------


class PluginSmoke:
    """Minimal plugin that does nothing -- proves discovery contract.

    This plugin satisfies every method on ``ProtocolDomainPlugin`` using the
    simplest possible implementation. It is the canonical reference for plugin
    authors who want to know the minimum viable implementation.

    The plugin:
    - Implements ``plugin_id`` and ``display_name`` as properties
    - Returns ``True`` from ``should_activate`` unconditionally
    - Returns ``ModelDomainPluginResult.succeeded(...)`` from every async hook
    - Does NOT inherit from ``ProtocolDomainPlugin`` (duck typing is sufficient)
    """

    @property
    def plugin_id(self) -> str:
        """Return unique identifier for this plugin."""
        return "smoke"

    @property
    def display_name(self) -> str:
        """Return human-readable name for this plugin."""
        return "Smoke"

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        """Always activate -- this is a smoke test plugin."""
        return True

    async def initialize(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No-op initialization."""
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def wire_handlers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No-op handler wiring."""
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def wire_dispatchers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No-op dispatcher wiring."""
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def start_consumers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No-op consumer startup."""
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def shutdown(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No-op shutdown."""
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_config() -> ModelDomainPluginConfig:
    """Create a minimal ``ModelDomainPluginConfig`` for seam tests.

    Uses ``MagicMock`` for heavy infrastructure objects (container, event_bus)
    since the smoke plugin ignores them completely. This proves the config
    interface is sufficient without requiring real infrastructure.
    """
    return ModelDomainPluginConfig(
        container=MagicMock(),
        event_bus=MagicMock(),
        correlation_id=uuid4(),
        input_topic="test.input.v1",
        output_topic="test.output.v1",
        consumer_group="test-consumer-group",
    )


def _make_entry_point(
    name: str,
    value: str,
    target_class: type,
) -> MagicMock:
    """Create a mock entry point that loads to ``target_class``."""
    ep = MagicMock()
    ep.name = name
    ep.value = value
    ep.load.return_value = target_class
    return ep


# ---------------------------------------------------------------------------
# Tests: Protocol conformance
# ---------------------------------------------------------------------------


class TestPluginSmokeProtocolConformance:
    """Verify PluginSmoke satisfies ProtocolDomainPlugin at the type level."""

    def test_isinstance_check_passes(self) -> None:
        """PluginSmoke passes isinstance(plugin, ProtocolDomainPlugin).

        This is the critical assertion: ``ProtocolDomainPlugin`` is
        ``@runtime_checkable``, so ``isinstance`` verifies structural
        subtyping at runtime. If the smoke plugin drifts out of conformance
        with the protocol, this test fails.
        """
        plugin = PluginSmoke()
        assert isinstance(plugin, ProtocolDomainPlugin)

    def test_plugin_id_returns_string(self) -> None:
        """plugin_id returns a non-empty string."""
        plugin = PluginSmoke()
        assert isinstance(plugin.plugin_id, str)
        assert len(plugin.plugin_id) > 0

    def test_display_name_returns_string(self) -> None:
        """display_name returns a non-empty string."""
        plugin = PluginSmoke()
        assert isinstance(plugin.display_name, str)
        assert len(plugin.display_name) > 0

    def test_should_activate_returns_bool(self) -> None:
        """should_activate returns a bool."""
        plugin = PluginSmoke()
        config = _make_minimal_config()
        result = plugin.should_activate(config)
        assert isinstance(result, bool)
        assert result is True


# ---------------------------------------------------------------------------
# Tests: Seam -- discovery to activation loop
# ---------------------------------------------------------------------------


class TestDiscoveryToActivationSeam:
    """Seam tests proving discovery output is consumable by a kernel-like loop.

    These tests replicate the exact iteration pattern from ``service_kernel.py``
    without importing the kernel:

        for plugin in plugin_registry.get_all():
            if plugin.should_activate(config):
                await plugin.initialize(config)
                await plugin.wire_handlers(config)
                await plugin.wire_dispatchers(config)
                await plugin.start_consumers(config)

    If this pattern works against the smoke plugin registered via the registry,
    the discovery contract is proven to be consumable by any kernel-like
    harness.
    """

    def test_explicit_register_then_iterate(self) -> None:
        """Explicit register() -> get_all() -> should_activate() works."""
        registry = RegistryDomainPlugin()
        registry.register(PluginSmoke())

        plugins = registry.get_all()
        assert len(plugins) == 1

        config = _make_minimal_config()
        activated = [p for p in plugins if p.should_activate(config)]
        assert len(activated) == 1
        assert activated[0].plugin_id == "smoke"

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_entry_point_discover_then_iterate(
        self,
        mock_entry_points: MagicMock,
    ) -> None:
        """discover_from_entry_points() -> get_all() -> should_activate() works."""
        ep = _make_entry_point(
            "smoke",
            "omnibase_infra.plugins.smoke:PluginSmoke",
            PluginSmoke,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert len(report.accepted) == 1
        assert report.accepted[0] == "smoke"
        assert not report.has_errors

        plugins = registry.get_all()
        assert len(plugins) == 1

        config = _make_minimal_config()
        activated = [p for p in plugins if p.should_activate(config)]
        assert len(activated) == 1
        assert activated[0].plugin_id == "smoke"

    @pytest.mark.asyncio
    async def test_full_lifecycle_from_registry(self) -> None:
        """Full lifecycle: register -> activate -> initialize -> wire -> start.

        This is the complete seam test. It proves that the discovery output
        (a list of ProtocolDomainPlugin instances) is consumable by the
        exact same loop the kernel uses.
        """
        registry = RegistryDomainPlugin()
        registry.register(PluginSmoke())

        config = _make_minimal_config()
        activated_plugins: list[ProtocolDomainPlugin] = []

        for plugin in registry.get_all():
            if not plugin.should_activate(config):
                continue

            # Full lifecycle in kernel order
            init_result = await plugin.initialize(config)
            assert init_result.success
            assert init_result.plugin_id == "smoke"

            handler_result = await plugin.wire_handlers(config)
            assert handler_result.success

            dispatcher_result = await plugin.wire_dispatchers(config)
            assert dispatcher_result.success

            consumer_result = await plugin.start_consumers(config)
            assert consumer_result.success

            activated_plugins.append(plugin)

        assert len(activated_plugins) == 1
        assert activated_plugins[0].plugin_id == "smoke"

    @pytest.mark.asyncio
    async def test_shutdown_after_activation(self) -> None:
        """Shutdown works after full activation (LIFO order for single plugin)."""
        registry = RegistryDomainPlugin()
        registry.register(PluginSmoke())

        config = _make_minimal_config()
        activated: list[ProtocolDomainPlugin] = []

        for plugin in registry.get_all():
            if plugin.should_activate(config):
                await plugin.initialize(config)
                await plugin.wire_handlers(config)
                await plugin.wire_dispatchers(config)
                await plugin.start_consumers(config)
                activated.append(plugin)

        # Shutdown in reverse order (LIFO), matching kernel behavior
        for plugin in reversed(activated):
            result = await plugin.shutdown(config)
            assert result.success
            assert result.plugin_id == "smoke"

    @pytest.mark.asyncio
    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    async def test_discovery_to_full_lifecycle(
        self,
        mock_entry_points: MagicMock,
    ) -> None:
        """End-to-end: entry_point discovery -> full lifecycle loop.

        This is the most complete seam test. It exercises:
        1. Entry-point discovery (discover_from_entry_points)
        2. Plugin iteration (get_all)
        3. Activation check (should_activate)
        4. Full lifecycle (initialize -> wire_handlers -> wire_dispatchers ->
           start_consumers)
        5. Shutdown (reverse order)

        If this test passes, external plugin authors can be confident that any
        class satisfying ProtocolDomainPlugin will integrate with the kernel.
        """
        ep = _make_entry_point(
            "smoke",
            "omnibase_infra.plugins.smoke:PluginSmoke",
            PluginSmoke,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()
        assert not report.has_errors

        config = _make_minimal_config()
        activated: list[ProtocolDomainPlugin] = []

        for plugin in registry.get_all():
            if not plugin.should_activate(config):
                continue

            init_result = await plugin.initialize(config)
            assert init_result.success

            handler_result = await plugin.wire_handlers(config)
            assert handler_result.success

            dispatcher_result = await plugin.wire_dispatchers(config)
            assert dispatcher_result.success

            consumer_result = await plugin.start_consumers(config)
            assert consumer_result.success

            activated.append(plugin)

        assert len(activated) == 1

        # Shutdown
        for plugin in reversed(activated):
            shutdown_result = await plugin.shutdown(config)
            assert shutdown_result.success


# ---------------------------------------------------------------------------
# Tests: Multiple plugins in a single registry
# ---------------------------------------------------------------------------


class PluginSmokeAlpha(PluginSmoke):
    """Smoke variant with a different plugin_id."""

    @property
    def plugin_id(self) -> str:
        return "smoke-alpha"

    @property
    def display_name(self) -> str:
        return "Smoke Alpha"


class PluginSmokeBeta(PluginSmoke):
    """Smoke variant with a different plugin_id."""

    @property
    def plugin_id(self) -> str:
        return "smoke-beta"

    @property
    def display_name(self) -> str:
        return "Smoke Beta"


class TestMultiplePluginSeam:
    """Seam tests with multiple plugins to verify iteration ordering."""

    def test_multiple_explicit_plugins_iterate(self) -> None:
        """Multiple plugins registered explicitly all appear in get_all()."""
        registry = RegistryDomainPlugin()
        registry.register(PluginSmoke())
        registry.register(PluginSmokeAlpha())
        registry.register(PluginSmokeBeta())

        plugins = registry.get_all()
        assert len(plugins) == 3

        config = _make_minimal_config()
        activated = [p for p in plugins if p.should_activate(config)]
        assert len(activated) == 3

        plugin_ids = {p.plugin_id for p in activated}
        assert plugin_ids == {"smoke", "smoke-alpha", "smoke-beta"}

    @pytest.mark.asyncio
    async def test_multiple_plugins_full_lifecycle(self) -> None:
        """Multiple plugins go through full lifecycle successfully."""
        registry = RegistryDomainPlugin()
        registry.register(PluginSmoke())
        registry.register(PluginSmokeAlpha())
        registry.register(PluginSmokeBeta())

        config = _make_minimal_config()
        activated: list[ProtocolDomainPlugin] = []

        for plugin in registry.get_all():
            if not plugin.should_activate(config):
                continue

            init_result = await plugin.initialize(config)
            assert init_result.success

            handler_result = await plugin.wire_handlers(config)
            assert handler_result.success

            dispatcher_result = await plugin.wire_dispatchers(config)
            assert dispatcher_result.success

            consumer_result = await plugin.start_consumers(config)
            assert consumer_result.success

            activated.append(plugin)

        assert len(activated) == 3

        # Shutdown in LIFO order
        for plugin in reversed(activated):
            result = await plugin.shutdown(config)
            assert result.success


# ---------------------------------------------------------------------------
# Tests: Inactive plugin filtering
# ---------------------------------------------------------------------------


class PluginSmokeInactive(PluginSmoke):
    """Smoke plugin that refuses to activate."""

    @property
    def plugin_id(self) -> str:
        return "smoke-inactive"

    @property
    def display_name(self) -> str:
        return "Smoke Inactive"

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        """Never activate -- simulates a plugin whose environment is missing."""
        return False


class TestInactivePluginFiltering:
    """Verify that plugins returning should_activate=False are skipped."""

    def test_inactive_plugin_filtered_out(self) -> None:
        """Inactive plugin is registered but not activated."""
        registry = RegistryDomainPlugin()
        registry.register(PluginSmoke())
        registry.register(PluginSmokeInactive())

        assert len(registry) == 2

        config = _make_minimal_config()
        activated = [p for p in registry.get_all() if p.should_activate(config)]
        assert len(activated) == 1
        assert activated[0].plugin_id == "smoke"

    @pytest.mark.asyncio
    async def test_lifecycle_skips_inactive(self) -> None:
        """Full lifecycle loop correctly skips inactive plugins."""
        registry = RegistryDomainPlugin()
        registry.register(PluginSmoke())
        registry.register(PluginSmokeInactive())

        config = _make_minimal_config()
        activated: list[ProtocolDomainPlugin] = []

        for plugin in registry.get_all():
            if not plugin.should_activate(config):
                continue

            result = await plugin.initialize(config)
            assert result.success
            activated.append(plugin)

        assert len(activated) == 1
        assert activated[0].plugin_id == "smoke"


__all__: list[str] = [
    "PluginSmoke",
    "TestDiscoveryToActivationSeam",
    "TestInactivePluginFiltering",
    "TestMultiplePluginSeam",
    "TestPluginSmokeProtocolConformance",
]
