# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for RegistryDomainPlugin.discover_from_entry_points().

These tests verify:
1. Happy-path discovery and registration of plugins
2. Security namespace enforcement (pre-import rejection)
3. Deterministic ordering by entry-point name then value
4. Duplicate policy (explicit wins, discovered duplicates skipped)
5. Error handling (import errors, instantiation errors, protocol violations)
6. Strict mode (fail-fast on errors)
7. Default security config (bare call is secure)
8. Private helpers (_validate_plugin_namespace, _parse_module_path)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.runtime.models.model_security_config import ModelSecurityConfig
from omnibase_infra.runtime.protocol_domain_plugin import (
    ProtocolDomainPlugin,
    RegistryDomainPlugin,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakePlugin:
    """Minimal plugin satisfying ProtocolDomainPlugin protocol for tests."""

    def __init__(self, plugin_id: str = "fake") -> None:
        self._plugin_id = plugin_id

    @property
    def plugin_id(self) -> str:
        return self._plugin_id

    @property
    def display_name(self) -> str:
        return self._plugin_id.title()

    def should_activate(self, config: object) -> bool:
        return True

    async def initialize(self, config: object) -> object:
        return MagicMock(success=True)

    async def wire_handlers(self, config: object) -> object:
        return MagicMock(success=True)

    async def wire_dispatchers(self, config: object) -> object:
        return MagicMock(success=True)

    async def start_consumers(self, config: object) -> object:
        return MagicMock(success=True)

    async def shutdown(self, config: object) -> object:
        return MagicMock(success=True)


class _NotAPlugin:
    """Class that does NOT satisfy ProtocolDomainPlugin."""


def _make_entry_point(
    name: str,
    value: str,
    group: str = "onex.domain_plugins",
) -> MagicMock:
    """Create a mock entry-point object."""
    ep = MagicMock()
    ep.name = name
    ep.value = value
    ep.group = group
    return ep


# ---------------------------------------------------------------------------
# Tests: _parse_module_path
# ---------------------------------------------------------------------------


class TestParseModulePath:
    """Tests for RegistryDomainPlugin._parse_module_path."""

    def test_standard_colon_format(self) -> None:
        """Standard 'module:Class' format returns the module portion."""
        result = RegistryDomainPlugin._parse_module_path(
            "omnibase_infra.plugins.foo:PluginFoo"
        )
        assert result == "omnibase_infra.plugins.foo"

    def test_no_colon_returns_full_value(self) -> None:
        """Value with no colon is returned unchanged."""
        result = RegistryDomainPlugin._parse_module_path("omnibase_infra.plugins.foo")
        assert result == "omnibase_infra.plugins.foo"

    def test_multiple_colons_splits_on_first(self) -> None:
        """Multiple colons split only on the first occurrence."""
        result = RegistryDomainPlugin._parse_module_path("pkg.mod:Class:extra")
        assert result == "pkg.mod"


# ---------------------------------------------------------------------------
# Tests: _validate_plugin_namespace
# ---------------------------------------------------------------------------


class TestValidatePluginNamespace:
    """Tests for RegistryDomainPlugin._validate_plugin_namespace."""

    def test_trusted_namespace_with_dot(self) -> None:
        """Module in a trusted dot-suffixed namespace is accepted."""
        assert RegistryDomainPlugin._validate_plugin_namespace(
            "omnibase_infra.plugins.foo",
            ("omnibase_core.", "omnibase_infra."),
        )

    def test_untrusted_namespace_rejected(self) -> None:
        """Module outside all trusted namespaces is rejected."""
        assert not RegistryDomainPlugin._validate_plugin_namespace(
            "malicious.module",
            ("omnibase_core.", "omnibase_infra."),
        )

    def test_boundary_aware_no_trailing_dot(self) -> None:
        """Namespace 'foo' should NOT match 'foobar.module'."""
        assert not RegistryDomainPlugin._validate_plugin_namespace(
            "foobar.module",
            ("foo",),
        )

    def test_boundary_aware_exact_match_no_trailing_dot(self) -> None:
        """Namespace 'foo' should match 'foo' exactly."""
        assert RegistryDomainPlugin._validate_plugin_namespace(
            "foo",
            ("foo",),
        )

    def test_boundary_aware_dot_suffix_no_trailing_dot(self) -> None:
        """Namespace 'foo' should match 'foo.bar'."""
        assert RegistryDomainPlugin._validate_plugin_namespace(
            "foo.bar",
            ("foo",),
        )

    def test_empty_namespaces_blocks_all(self) -> None:
        """Empty namespace tuple blocks all modules."""
        assert not RegistryDomainPlugin._validate_plugin_namespace(
            "omnibase_infra.plugins.foo",
            (),
        )


# ---------------------------------------------------------------------------
# Tests: discover_from_entry_points
# ---------------------------------------------------------------------------


class TestDiscoverFromEntryPoints:
    """Tests for RegistryDomainPlugin.discover_from_entry_points."""

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_happy_path_single_plugin(self, mock_ep: MagicMock) -> None:
        """A single trusted plugin should be discovered and registered."""
        ep = _make_entry_point(
            "registration",
            "omnibase_infra.plugins.reg:PluginReg",
        )
        plugin_instance = _FakePlugin("registration")

        # Patch the type to return our instance
        loaded_cls = MagicMock(return_value=plugin_instance)
        ep.load.return_value = loaded_cls

        mock_ep.return_value = [ep]

        registry = RegistryDomainPlugin()
        config = ModelSecurityConfig()  # default (trusted only)
        report = registry.discover_from_entry_points(security_config=config)

        assert report.group == "onex.domain_plugins"
        assert report.discovered_count == 1
        assert report.accepted == ("registration",)
        assert len(report.entries) == 1
        assert report.entries[0].status == "accepted"
        assert registry.get("registration") is plugin_instance

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_namespace_rejected(self, mock_ep: MagicMock) -> None:
        """An untrusted namespace should be rejected before import."""
        ep = _make_entry_point(
            "evil",
            "malicious_pkg.evil:EvilPlugin",
        )
        mock_ep.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert report.discovered_count == 1
        assert report.accepted == ()
        assert len(report.entries) == 1
        assert report.entries[0].status == "namespace_rejected"
        assert "malicious_pkg.evil" in report.entries[0].reason
        # load() should NOT have been called (pre-import security)
        ep.load.assert_not_called()

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_import_error_recorded(self, mock_ep: MagicMock) -> None:
        """Import errors should be recorded in the report."""
        ep = _make_entry_point(
            "broken",
            "omnibase_infra.broken:BrokenPlugin",
        )
        ep.load.side_effect = ModuleNotFoundError(
            "No module named 'omnibase_infra.broken'"
        )
        mock_ep.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert report.discovered_count == 1
        assert report.accepted == ()
        assert report.entries[0].status == "import_error"
        assert "ModuleNotFoundError" in report.entries[0].reason
        assert report.has_errors

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_instantiation_error_recorded(self, mock_ep: MagicMock) -> None:
        """Instantiation errors should be recorded in the report."""
        ep = _make_entry_point(
            "bad_init",
            "omnibase_infra.plugins.bad:BadPlugin",
        )
        loaded_cls = MagicMock(side_effect=TypeError("missing required arg"))
        ep.load.return_value = loaded_cls
        mock_ep.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert report.discovered_count == 1
        assert report.accepted == ()
        assert report.entries[0].status == "instantiation_error"
        assert "TypeError" in report.entries[0].reason

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_protocol_invalid_recorded(self, mock_ep: MagicMock) -> None:
        """Non-conforming classes should be recorded as protocol_invalid."""
        ep = _make_entry_point(
            "not_a_plugin",
            "omnibase_infra.plugins.nope:NotAPlugin",
        )
        ep.load.return_value = MagicMock(return_value=_NotAPlugin())
        mock_ep.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert report.discovered_count == 1
        assert report.accepted == ()
        assert report.entries[0].status == "protocol_invalid"
        assert "ProtocolDomainPlugin" in report.entries[0].reason

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_duplicate_skipped_explicit_wins(self, mock_ep: MagicMock) -> None:
        """Discovered plugin with same ID as explicit registration is skipped."""
        ep = _make_entry_point(
            "registration",
            "omnibase_infra.plugins.reg:PluginReg",
        )
        discovered_plugin = _FakePlugin("registration")
        ep.load.return_value = MagicMock(return_value=discovered_plugin)
        mock_ep.return_value = [ep]

        # Pre-register with the same plugin_id
        explicit_plugin = _FakePlugin("registration")
        registry = RegistryDomainPlugin()
        registry.register(explicit_plugin)

        report = registry.discover_from_entry_points()

        assert report.discovered_count == 1
        assert report.accepted == ()
        assert report.entries[0].status == "duplicate_skipped"
        assert report.entries[0].plugin_id == "registration"
        # The explicit plugin should still be in the registry
        assert registry.get("registration") is explicit_plugin

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_deterministic_ordering(self, mock_ep: MagicMock) -> None:
        """Entry points should be processed in sorted order (name, value)."""
        ep_b = _make_entry_point("b_plugin", "omnibase_infra.b:B")
        ep_a = _make_entry_point("a_plugin", "omnibase_infra.a:A")
        ep_c = _make_entry_point("c_plugin", "omnibase_infra.c:C")

        plugin_a = _FakePlugin("a")
        plugin_b = _FakePlugin("b")
        plugin_c = _FakePlugin("c")

        ep_a.load.return_value = MagicMock(return_value=plugin_a)
        ep_b.load.return_value = MagicMock(return_value=plugin_b)
        ep_c.load.return_value = MagicMock(return_value=plugin_c)

        # Return in non-sorted order
        mock_ep.return_value = [ep_c, ep_a, ep_b]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        # Should be sorted by name: a_plugin, b_plugin, c_plugin
        assert report.accepted == ("a", "b", "c")
        entry_names = [e.entry_point_name for e in report.entries]
        assert entry_names == ["a_plugin", "b_plugin", "c_plugin"]

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_default_security_config_is_secure(self, mock_ep: MagicMock) -> None:
        """Bare call with no args should use secure defaults."""
        ep = _make_entry_point(
            "third_party",
            "third_party_pkg.plugin:Plugin",
        )
        mock_ep.return_value = [ep]

        registry = RegistryDomainPlugin()
        # No security_config argument -- should default to blocking third-party
        report = registry.discover_from_entry_points()

        assert report.accepted == ()
        assert report.entries[0].status == "namespace_rejected"

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_third_party_allowed_with_config(self, mock_ep: MagicMock) -> None:
        """Third-party plugins should be allowed when explicitly configured."""
        ep = _make_entry_point(
            "custom",
            "mycompany.plugins.custom:CustomPlugin",
        )
        plugin = _FakePlugin("custom")
        ep.load.return_value = MagicMock(return_value=plugin)
        mock_ep.return_value = [ep]

        config = ModelSecurityConfig(
            allow_third_party_plugins=True,
            allowed_plugin_namespaces=(
                "omnibase_core.",
                "omnibase_infra.",
                "mycompany.plugins.",
            ),
        )

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(security_config=config)

        assert report.accepted == ("custom",)
        assert registry.get("custom") is plugin

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_empty_group_returns_empty_report(self, mock_ep: MagicMock) -> None:
        """No entry points should return an empty report."""
        mock_ep.return_value = []

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert report.discovered_count == 0
        assert report.accepted == ()
        assert report.entries == ()
        assert not report.has_errors

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_custom_group(self, mock_ep: MagicMock) -> None:
        """Custom entry-point group should be used in the report."""
        mock_ep.return_value = []

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(group="custom.group")

        mock_ep.assert_called_once_with(group="custom.group")
        assert report.group == "custom.group"


# ---------------------------------------------------------------------------
# Tests: strict mode
# ---------------------------------------------------------------------------


class TestDiscoverFromEntryPointsStrict:
    """Tests for strict=True behavior."""

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_strict_import_error_raises(self, mock_ep: MagicMock) -> None:
        """strict=True should raise ImportError on load failure."""
        ep = _make_entry_point(
            "broken",
            "omnibase_infra.broken:Broken",
        )
        ep.load.side_effect = ModuleNotFoundError("no module")
        mock_ep.return_value = [ep]

        registry = RegistryDomainPlugin()
        with pytest.raises(ImportError, match="broken"):
            registry.discover_from_entry_points(strict=True)

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_strict_instantiation_error_raises(self, mock_ep: MagicMock) -> None:
        """strict=True should raise TypeError on instantiation failure."""
        ep = _make_entry_point(
            "bad",
            "omnibase_infra.bad:Bad",
        )
        ep.load.return_value = MagicMock(side_effect=TypeError("bad init"))
        mock_ep.return_value = [ep]

        registry = RegistryDomainPlugin()
        with pytest.raises(TypeError, match="bad"):
            registry.discover_from_entry_points(strict=True)

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_strict_protocol_invalid_raises(self, mock_ep: MagicMock) -> None:
        """strict=True should raise RuntimeError on protocol failure."""
        ep = _make_entry_point(
            "invalid",
            "omnibase_infra.invalid:Invalid",
        )
        ep.load.return_value = MagicMock(return_value=_NotAPlugin())
        mock_ep.return_value = [ep]

        registry = RegistryDomainPlugin()
        with pytest.raises(RuntimeError, match="ProtocolDomainPlugin"):
            registry.discover_from_entry_points(strict=True)


# ---------------------------------------------------------------------------
# Tests: discovered duplicate among discovered plugins
# ---------------------------------------------------------------------------


class TestDiscoverDuplicateAmongDiscovered:
    """Tests for duplicate handling within a single discovery pass."""

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_first_discovered_wins_second_is_duplicate(
        self, mock_ep: MagicMock
    ) -> None:
        """When two entry points produce the same plugin_id, first wins."""
        ep1 = _make_entry_point("first", "omnibase_infra.first:First")
        ep2 = _make_entry_point("second", "omnibase_infra.second:Second")

        plugin1 = _FakePlugin("same-id")
        plugin2 = _FakePlugin("same-id")

        ep1.load.return_value = MagicMock(return_value=plugin1)
        ep2.load.return_value = MagicMock(return_value=plugin2)

        mock_ep.return_value = [ep1, ep2]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert report.accepted == ("same-id",)
        assert report.entries[0].status == "accepted"
        assert report.entries[1].status == "duplicate_skipped"
        assert registry.get("same-id") is plugin1


__all__: list[str] = [
    "TestDiscoverDuplicateAmongDiscovered",
    "TestDiscoverFromEntryPoints",
    "TestDiscoverFromEntryPointsStrict",
    "TestParseModulePath",
    "TestValidatePluginNamespace",
]
