# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for domain plugin entry_point discovery (OMN-2000).

These tests verify:
1. discover_from_entry_points() discovers and registers valid plugins
2. Namespace rejection blocks untrusted modules BEFORE import
3. Duplicate plugin_id handling (explicit registration wins)
4. Protocol validation rejects non-conforming classes
5. Import error handling for broken entry_points
6. Instantiation error handling for failing constructors
7. ModelSecurityConfig plugin namespace fields
8. Integration with kernel bootstrap flow
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.constants_security import (
    DOMAIN_PLUGIN_ENTRY_POINT_GROUP,
    TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
)
from omnibase_infra.runtime.models import ModelSecurityConfig
from omnibase_infra.runtime.models.model_plugin_discovery_entry import (
    ModelPluginDiscoveryEntry,
)
from omnibase_infra.runtime.models.model_plugin_discovery_report import (
    ModelPluginDiscoveryReport,
)
from omnibase_infra.runtime.protocol_domain_plugin import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
    ProtocolDomainPlugin,
    RegistryDomainPlugin,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class ValidPlugin:
    """A valid plugin that implements ProtocolDomainPlugin."""

    def __init__(self, plugin_id: str = "valid-plugin") -> None:
        self._plugin_id = plugin_id

    @property
    def plugin_id(self) -> str:
        return self._plugin_id

    @property
    def display_name(self) -> str:
        return "Valid Plugin"

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        return True

    async def initialize(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def wire_handlers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def wire_dispatchers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def start_consumers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def shutdown(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)


class InvalidPlugin:
    """A class that does NOT implement ProtocolDomainPlugin (missing methods)."""


class FailingConstructorPlugin:
    """A plugin whose constructor raises an error."""

    def __init__(self) -> None:
        raise RuntimeError("Constructor failed")


def _make_entry_point(
    name: str, value: str, target_class: type | None = None
) -> MagicMock:
    """Create a mock EntryPoint.

    Args:
        name: Entry point name (e.g. "registration").
        value: Module path string (e.g. "omnibase_infra.plugins:Plugin").
        target_class: Class to return from .load(). If None, load() raises ImportError.
    """
    ep = MagicMock()
    ep.name = name
    ep.value = value
    if target_class is not None:
        ep.load.return_value = target_class
    else:
        ep.load.side_effect = ImportError(f"No module named '{value.split(':')[0]}'")
    return ep


# ---------------------------------------------------------------------------
# discover_from_entry_points tests
# ---------------------------------------------------------------------------


class TestDiscoverFromEntryPoints:
    """Tests for RegistryDomainPlugin.discover_from_entry_points()."""

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_discover_valid_plugin(self, mock_entry_points: MagicMock) -> None:
        """A valid plugin from a trusted namespace is discovered and registered."""
        ep = _make_entry_point(
            "test-plugin",
            "omnibase_infra.plugins.test:ValidPlugin",
            ValidPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            allowed_namespaces=TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
        )

        assert len(report.accepted) == 1
        assert report.accepted[0] == "valid-plugin"
        assert len(registry) == 1
        assert registry.get("valid-plugin") is not None
        assert not report.has_errors

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_namespace_rejection(self, mock_entry_points: MagicMock) -> None:
        """Entry points from untrusted namespaces are rejected BEFORE import."""
        ep = _make_entry_point(
            "evil-plugin",
            "evil_corp.plugins:MaliciousPlugin",
            ValidPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            allowed_namespaces=TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
        )

        # Plugin should NOT be loaded (load() should not have been called)
        ep.load.assert_not_called()
        assert len(report.accepted) == 0
        assert len(registry) == 0
        assert len(report.rejected) == 1
        assert report.entries[0].status == "namespace_rejected"
        assert "evil_corp.plugins" in report.entries[0].reason

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_duplicate_plugin_id_skipped(self, mock_entry_points: MagicMock) -> None:
        """If a plugin_id is already registered, entry_point is skipped."""
        ep = _make_entry_point(
            "registration",
            "omnibase_infra.plugins:ValidPlugin",
            ValidPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        # Pre-register a plugin with the same ID
        existing = ValidPlugin(plugin_id="valid-plugin")
        registry.register(existing)

        report = registry.discover_from_entry_points(
            allowed_namespaces=TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
        )

        # Should still have only the original plugin
        assert len(registry) == 1
        assert registry.get("valid-plugin") is existing
        assert len(report.accepted) == 0
        assert report.entries[0].status == "duplicate_skipped"
        assert report.entries[0].plugin_id == "valid-plugin"

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_protocol_validation_failure(self, mock_entry_points: MagicMock) -> None:
        """Classes that don't implement ProtocolDomainPlugin are rejected."""
        ep = _make_entry_point(
            "invalid-plugin",
            "omnibase_infra.plugins:InvalidPlugin",
            InvalidPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            allowed_namespaces=TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
        )

        assert len(report.accepted) == 0
        assert len(registry) == 0
        assert report.entries[0].status == "protocol_invalid"
        assert "ProtocolDomainPlugin" in report.entries[0].reason

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_import_error_handling(self, mock_entry_points: MagicMock) -> None:
        """Entry points that fail to load are reported as import_error."""
        ep = _make_entry_point(
            "broken-plugin",
            "omnibase_infra.plugins.broken:BrokenPlugin",
            None,  # load() will raise ImportError
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            allowed_namespaces=TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
        )

        assert len(report.accepted) == 0
        assert len(registry) == 0
        assert report.entries[0].status == "import_error"
        assert report.has_errors

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_instantiation_error_handling(self, mock_entry_points: MagicMock) -> None:
        """Plugins whose constructors fail are reported as instantiation_error."""
        ep = _make_entry_point(
            "failing-plugin",
            "omnibase_infra.plugins:FailingPlugin",
            FailingConstructorPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            allowed_namespaces=TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
        )

        assert len(report.accepted) == 0
        assert len(registry) == 0
        assert report.entries[0].status == "instantiation_error"
        assert report.has_errors

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_empty_group_returns_empty_report(
        self, mock_entry_points: MagicMock
    ) -> None:
        """When no entry_points exist in the group, report is empty."""
        mock_entry_points.return_value = []

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert report.discovered_count == 0
        assert len(report.accepted) == 0
        assert len(report.entries) == 0
        assert not report.has_errors

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_multiple_plugins_mixed_outcomes(
        self, mock_entry_points: MagicMock
    ) -> None:
        """Multiple entry_points with different outcomes are all tracked."""
        ep_valid = _make_entry_point(
            "valid",
            "omnibase_infra.plugins:ValidPlugin",
            ValidPlugin,
        )
        ep_rejected = _make_entry_point(
            "rejected",
            "evil.namespace:Plugin",
            ValidPlugin,
        )
        ep_broken = _make_entry_point(
            "broken",
            "omnibase_infra.broken:Plugin",
            None,
        )
        mock_entry_points.return_value = [ep_valid, ep_rejected, ep_broken]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            allowed_namespaces=TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
        )

        assert len(report.entries) == 3
        assert len(report.accepted) == 1
        assert len(report.rejected) == 2
        assert report.has_errors  # broken has import_error

        statuses = {e.entry_point_name: e.status for e in report.entries}
        assert statuses["valid"] == "accepted"
        assert statuses["rejected"] == "namespace_rejected"
        assert statuses["broken"] == "import_error"

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_default_group_name(self, mock_entry_points: MagicMock) -> None:
        """Default group name matches DOMAIN_PLUGIN_ENTRY_POINT_GROUP."""
        mock_entry_points.return_value = []

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        mock_entry_points.assert_called_once_with(
            group=DOMAIN_PLUGIN_ENTRY_POINT_GROUP,
        )
        assert report.group == DOMAIN_PLUGIN_ENTRY_POINT_GROUP

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_custom_group_name(self, mock_entry_points: MagicMock) -> None:
        """Custom group name is passed through to entry_points()."""
        mock_entry_points.return_value = []

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(group="custom.plugins")

        mock_entry_points.assert_called_once_with(group="custom.plugins")
        assert report.group == "custom.plugins"


# ---------------------------------------------------------------------------
# ModelSecurityConfig plugin fields tests
# ---------------------------------------------------------------------------


class TestModelSecurityConfigPluginFields:
    """Tests for plugin-related fields on ModelSecurityConfig."""

    def test_default_plugin_discovery_disabled(self) -> None:
        """By default, third-party plugin discovery is disabled."""
        config = ModelSecurityConfig()

        assert config.allow_third_party_plugins is False
        assert config.allowed_plugin_namespaces == TRUSTED_PLUGIN_NAMESPACE_PREFIXES

    def test_effective_plugin_namespaces_when_disabled(self) -> None:
        """When disabled, effective namespaces are the trusted defaults."""
        config = ModelSecurityConfig()
        effective = config.get_effective_plugin_namespaces()

        assert effective == TRUSTED_PLUGIN_NAMESPACE_PREFIXES

    def test_effective_plugin_namespaces_when_enabled(self) -> None:
        """When enabled, custom namespaces are returned."""
        custom = ("mycompany.plugins.", "partner.plugins.")
        config = ModelSecurityConfig(
            allow_third_party_plugins=True,
            allowed_plugin_namespaces=custom,
        )
        effective = config.get_effective_plugin_namespaces()

        assert effective == custom

    def test_enabled_but_default_namespaces(self) -> None:
        """When enabled with defaults, trusted namespaces are returned."""
        config = ModelSecurityConfig(allow_third_party_plugins=True)
        effective = config.get_effective_plugin_namespaces()

        assert effective == TRUSTED_PLUGIN_NAMESPACE_PREFIXES

    def test_handler_and_plugin_namespaces_independent(self) -> None:
        """Handler and plugin namespaces are configured independently."""
        config = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("handler.namespace.",),
            allow_third_party_plugins=True,
            allowed_plugin_namespaces=("plugin.namespace.",),
        )

        assert config.get_effective_namespaces() == ("handler.namespace.",)
        assert config.get_effective_plugin_namespaces() == ("plugin.namespace.",)

    def test_disabled_third_party_ignores_custom_namespaces(self) -> None:
        """When disabled, custom namespaces are ignored."""
        config = ModelSecurityConfig(
            allow_third_party_plugins=False,
            allowed_plugin_namespaces=("malicious.namespace.",),
        )
        effective = config.get_effective_plugin_namespaces()

        assert effective == TRUSTED_PLUGIN_NAMESPACE_PREFIXES
        assert "malicious.namespace." not in effective


# ---------------------------------------------------------------------------
# Integration-style test: discovery report model correctness
# ---------------------------------------------------------------------------


class TestDiscoveryReportIntegration:
    """Tests that the report model works correctly with discovery."""

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_report_structure(self, mock_entry_points: MagicMock) -> None:
        """Report contains correct group, counts, and entries."""
        ep = _make_entry_point(
            "my-plugin",
            "omnibase_infra.plugins:ValidPlugin",
            ValidPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert isinstance(report, ModelPluginDiscoveryReport)
        assert report.group == DOMAIN_PLUGIN_ENTRY_POINT_GROUP
        assert report.discovered_count == 1
        assert len(report.entries) == 1

        entry = report.entries[0]
        assert isinstance(entry, ModelPluginDiscoveryEntry)
        assert entry.entry_point_name == "my-plugin"
        assert entry.status == "accepted"
        assert entry.plugin_id == "valid-plugin"

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_report_rejected_property(self, mock_entry_points: MagicMock) -> None:
        """Report.rejected filters out accepted entries."""
        ep_ok = _make_entry_point("ok", "omnibase_infra.x:ValidPlugin", ValidPlugin)
        ep_bad = _make_entry_point("bad", "evil:Plugin", ValidPlugin)
        mock_entry_points.return_value = [ep_ok, ep_bad]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            allowed_namespaces=TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
        )

        assert len(report.rejected) == 1
        assert report.rejected[0].entry_point_name == "bad"


# ---------------------------------------------------------------------------
# Constants verification
# ---------------------------------------------------------------------------


class TestSecurityConstants:
    """Verify security constants are correctly defined."""

    def test_domain_plugin_entry_point_group(self) -> None:
        """Entry point group name matches expected value."""
        assert DOMAIN_PLUGIN_ENTRY_POINT_GROUP == "onex.domain_plugins"

    def test_trusted_plugin_namespace_prefixes(self) -> None:
        """Trusted plugin namespaces include core and infra."""
        assert "omnibase_core." in TRUSTED_PLUGIN_NAMESPACE_PREFIXES
        assert "omnibase_infra." in TRUSTED_PLUGIN_NAMESPACE_PREFIXES

    def test_trusted_plugin_namespaces_are_immutable(self) -> None:
        """Namespace tuple is a tuple (immutable)."""
        assert isinstance(TRUSTED_PLUGIN_NAMESPACE_PREFIXES, tuple)


__all__: list[str] = [
    "TestDiscoverFromEntryPoints",
    "TestDiscoveryReportIntegration",
    "TestModelSecurityConfigPluginFields",
    "TestSecurityConstants",
]
