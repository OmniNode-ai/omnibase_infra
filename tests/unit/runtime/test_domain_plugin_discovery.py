# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for domain plugin discovery (OMN-2020).

Comprehensive unit tests for the plugin discovery mechanism covering:

1. Namespace validation (boundary-aware prefix matching)
2. Entry-point discovery (importlib.metadata integration)
3. Discovery report model (accepted/rejected/errors)
4. Security config plugin namespace fields
5. Security policy enforcement (end-to-end)

Reuses MockPlugin pattern from test_domain_plugin_shutdown.py.

Test classes and counts (22 total):
    - TestPluginNamespaceValidation: 4 tests
    - TestDiscoverFromEntryPoints: 10 tests
    - TestDiscoveryReport: 3 tests
    - TestSecurityConfigPluginNamespaces: 3 tests
    - TestSecurityPolicyEnforcement: 2 tests
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.constants_security import (
    TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
)
from omnibase_infra.runtime.models.model_plugin_discovery_report import (
    ModelPluginDiscoveryReport,
)
from omnibase_infra.runtime.models.model_security_config import ModelSecurityConfig
from omnibase_infra.runtime.protocol_domain_plugin import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
    RegistryDomainPlugin,
)

# ---------------------------------------------------------------------------
# Test helpers (reusing MockPlugin pattern from test_domain_plugin_shutdown.py)
# ---------------------------------------------------------------------------


class MockPlugin:
    """Mock plugin implementing ProtocolDomainPlugin for testing.

    Reuses the pattern from test_domain_plugin_shutdown.py.
    Satisfies ProtocolDomainPlugin via structural subtyping.
    """

    def __init__(
        self, plugin_id: str = "mock-plugin", display_name: str | None = None
    ) -> None:
        """Initialize mock plugin with configurable identity."""
        self._plugin_id = plugin_id
        self._display_name = display_name or plugin_id.title()

    @property
    def plugin_id(self) -> str:
        """Return unique identifier for this plugin."""
        return self._plugin_id

    @property
    def display_name(self) -> str:
        """Return human-readable name for this plugin."""
        return self._display_name

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        """Always activate for testing."""
        return True

    async def initialize(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        """Initialize plugin resources (no-op for testing)."""
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def wire_handlers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        """Wire handlers (no-op for testing)."""
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def wire_dispatchers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        """Wire dispatchers (no-op for testing)."""
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def start_consumers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        """Start consumers (no-op for testing)."""
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def shutdown(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        """Shut down plugin (no-op for testing)."""
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
        name: Entry point name (e.g. ``"registration"``).
        value: Module path string (e.g. ``"omnibase_infra.plugins:Plugin"``).
        target_class: Class to return from ``.load()``. If ``None``,
            ``load()`` raises ``ImportError``.
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
# TestPluginNamespaceValidation (4 tests)
# ---------------------------------------------------------------------------


class TestPluginNamespaceValidation:
    """Tests for namespace validation logic (_validate_plugin_namespace).

    Verifies boundary-aware prefix matching that prevents typosquatting
    attacks and ensures both dotted and non-dotted prefixes work correctly.
    """

    def test_dotted_prefix_matches_correctly(self) -> None:
        """Dotted prefix 'omnibase_infra.' matches 'omnibase_infra.plugin'."""
        assert RegistryDomainPlugin._validate_plugin_namespace(
            "omnibase_infra.plugin",
            ("omnibase_infra.",),
        )

    def test_typosquatting_rejected(self) -> None:
        """Prefix 'omnibase_infra.' does NOT match 'omnibase_infra_evil.plugin'.

        This is the critical typosquatting prevention test. A malicious
        package named ``omnibase_infra_evil`` must not be accepted by the
        prefix ``omnibase_infra.`` because the underscore breaks the boundary.
        """
        assert not RegistryDomainPlugin._validate_plugin_namespace(
            "omnibase_infra_evil.plugin",
            ("omnibase_infra.",),
        )

    def test_empty_allowlist_rejects_everything(self) -> None:
        """Empty allowlist rejects all modules."""
        assert not RegistryDomainPlugin._validate_plugin_namespace(
            "omnibase_infra.plugins.foo",
            (),
        )

    def test_non_dotted_prefix_uses_boundary_check(self) -> None:
        """Non-dotted prefix uses boundary check: matches '.x' but not 'structure.x'.

        The prefix ``"omnibase_infra"`` (no trailing dot) must match
        ``"omnibase_infra.x"`` (next char is dot) but must NOT match
        ``"omnibase_infrastructure.x"`` (no boundary at prefix end).
        """
        # Matches: next char after prefix is "."
        assert RegistryDomainPlugin._validate_plugin_namespace(
            "omnibase_infra.x",
            ("omnibase_infra",),
        )
        # Does NOT match: "omnibase_infrastructure" extends beyond boundary
        assert not RegistryDomainPlugin._validate_plugin_namespace(
            "omnibase_infrastructure.x",
            ("omnibase_infra",),
        )


# ---------------------------------------------------------------------------
# TestDiscoverFromEntryPoints (10 tests)
# ---------------------------------------------------------------------------


class TestDiscoverFromEntryPoints:
    """Tests for RegistryDomainPlugin.discover_from_entry_points().

    All tests mock ``importlib.metadata.entry_points`` to control discovery
    inputs. Tests cover happy path, error handling, strict mode, deduplication,
    and deterministic ordering.
    """

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_valid_plugin_discovered_and_registered(
        self, mock_entry_points: MagicMock
    ) -> None:
        """A valid plugin from a trusted namespace is discovered and registered."""
        ep = _make_entry_point(
            "test-plugin",
            "omnibase_infra.plugins.test:MockPlugin",
            MockPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert len(report.accepted) == 1
        assert report.accepted[0] == "mock-plugin"
        assert len(registry) == 1
        assert registry.get("mock-plugin") is not None
        assert not report.has_errors

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_explicit_registration_wins_on_duplicate(
        self, mock_entry_points: MagicMock
    ) -> None:
        """Explicit registration wins on duplicate plugin_id -- no ValueError.

        When a plugin_id is already registered via explicit ``register()``,
        the discovered plugin is silently skipped and logged as
        ``duplicate_skipped``.
        """
        ep = _make_entry_point(
            "registration",
            "omnibase_infra.plugins:MockPlugin",
            MockPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        existing = MockPlugin(plugin_id="mock-plugin")
        registry.register(existing)

        report = registry.discover_from_entry_points()

        # No ValueError raised; original plugin preserved
        assert len(registry) == 1
        assert registry.get("mock-plugin") is existing
        assert len(report.accepted) == 0
        assert report.entries[0].status == "duplicate_skipped"
        assert report.entries[0].plugin_id == "mock-plugin"

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_namespace_rejected_load_not_called(
        self, mock_entry_points: MagicMock
    ) -> None:
        """Namespace-rejected entry_point is skipped; .load() NOT called.

        This verifies pre-import security enforcement: untrusted modules
        are blocked BEFORE any import/load occurs, preventing side effects
        at import time.
        """
        ep = _make_entry_point(
            "evil-plugin",
            "evil_corp.plugins:MaliciousPlugin",
            MockPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        # Critical: load() must NOT have been called
        ep.load.assert_not_called()
        assert len(report.accepted) == 0
        assert len(registry) == 0
        assert report.entries[0].status == "namespace_rejected"

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_protocol_invalid_rejected(self, mock_entry_points: MagicMock) -> None:
        """Non-protocol-compliant class rejected with protocol_invalid status."""
        ep = _make_entry_point(
            "invalid-plugin",
            "omnibase_infra.plugins:InvalidPlugin",
            InvalidPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert len(report.accepted) == 0
        assert report.entries[0].status == "protocol_invalid"
        assert "ProtocolDomainPlugin" in report.entries[0].reason

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_import_error_graceful_non_strict(
        self, mock_entry_points: MagicMock
    ) -> None:
        """ImportError during .load() handled gracefully in non-strict mode."""
        ep = _make_entry_point(
            "broken-plugin",
            "omnibase_infra.plugins.broken:BrokenPlugin",
            None,  # load() raises ImportError
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert len(report.accepted) == 0
        assert report.entries[0].status == "import_error"
        assert report.has_errors

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_import_error_raises_in_strict_mode(
        self, mock_entry_points: MagicMock
    ) -> None:
        """ImportError during .load() raises in strict mode."""
        ep = _make_entry_point(
            "broken-plugin",
            "omnibase_infra.plugins.broken:BrokenPlugin",
            None,  # load() raises ImportError
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        with pytest.raises(ImportError, match="broken"):
            registry.discover_from_entry_points(strict=True)

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_instantiation_error_graceful(self, mock_entry_points: MagicMock) -> None:
        """Instantiation error handled gracefully (non-strict mode)."""
        ep = _make_entry_point(
            "failing-plugin",
            "omnibase_infra.plugins:FailingPlugin",
            FailingConstructorPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert len(report.accepted) == 0
        assert report.entries[0].status == "instantiation_error"
        assert report.has_errors

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_multiple_valid_all_registered(self, mock_entry_points: MagicMock) -> None:
        """Multiple valid entry_points all registered; returns correct plugin_id list."""

        class PluginAlpha(MockPlugin):
            def __init__(self) -> None:
                super().__init__(plugin_id="alpha")

        class PluginBeta(MockPlugin):
            def __init__(self) -> None:
                super().__init__(plugin_id="beta")

        class PluginGamma(MockPlugin):
            def __init__(self) -> None:
                super().__init__(plugin_id="gamma")

        ep_a = _make_entry_point("alpha", "omnibase_infra.a:PluginAlpha", PluginAlpha)
        ep_b = _make_entry_point("beta", "omnibase_infra.b:PluginBeta", PluginBeta)
        ep_g = _make_entry_point("gamma", "omnibase_infra.g:PluginGamma", PluginGamma)
        mock_entry_points.return_value = [ep_a, ep_b, ep_g]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert len(report.accepted) == 3
        assert set(report.accepted) == {"alpha", "beta", "gamma"}
        assert len(registry) == 3
        for pid in ("alpha", "beta", "gamma"):
            assert registry.get(pid) is not None

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_no_entry_points_empty_report(self, mock_entry_points: MagicMock) -> None:
        """No entry_points returns empty report with zero counts."""
        mock_entry_points.return_value = []

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert report.discovered_count == 0
        assert len(report.accepted) == 0
        assert len(report.entries) == 0
        assert not report.has_errors

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_deterministic_ordering(self, mock_entry_points: MagicMock) -> None:
        """Two plugins always register in same order regardless of iteration order.

        The discover method sorts entry points by ``(name, value)`` before
        processing, ensuring reproducible ordering across runs.
        """

        class PluginFirst(MockPlugin):
            def __init__(self) -> None:
                super().__init__(plugin_id="first")

        class PluginSecond(MockPlugin):
            def __init__(self) -> None:
                super().__init__(plugin_id="second")

        ep_first = _make_entry_point(
            "a_first", "omnibase_infra.first:PluginFirst", PluginFirst
        )
        ep_second = _make_entry_point(
            "b_second", "omnibase_infra.second:PluginSecond", PluginSecond
        )

        # Provide in reverse alphabetical order
        mock_entry_points.return_value = [ep_second, ep_first]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        # Sorted by name: a_first < b_second -> first, second
        assert report.accepted == ("first", "second")

        # Verify same result when input order is already sorted
        mock_entry_points.return_value = [ep_first, ep_second]

        registry2 = RegistryDomainPlugin()
        report2 = registry2.discover_from_entry_points()

        assert report2.accepted == ("first", "second")


# ---------------------------------------------------------------------------
# TestDiscoveryReport (3 tests)
# ---------------------------------------------------------------------------


class TestDiscoveryReport:
    """Tests for ModelPluginDiscoveryReport properties and filtering."""

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_accepted_matches_registered_plugin_ids(
        self, mock_entry_points: MagicMock
    ) -> None:
        """Report accepted list matches registered plugin_ids."""
        ep = _make_entry_point(
            "my-plugin",
            "omnibase_infra.plugins:MockPlugin",
            MockPlugin,
        )
        mock_entry_points.return_value = [ep]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert isinstance(report, ModelPluginDiscoveryReport)
        assert report.accepted == ("mock-plugin",)
        assert report.entries[0].plugin_id == "mock-plugin"
        assert report.entries[0].status == "accepted"

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_rejected_filters_correctly(self, mock_entry_points: MagicMock) -> None:
        """Report rejected property filters out accepted entries."""
        ep_ok = _make_entry_point("ok", "omnibase_infra.x:MockPlugin", MockPlugin)
        ep_bad = _make_entry_point("bad", "evil:Plugin", MockPlugin)
        mock_entry_points.return_value = [ep_ok, ep_bad]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert len(report.rejected) == 1
        assert report.rejected[0].entry_point_name == "bad"
        assert report.rejected[0].status == "namespace_rejected"

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_has_errors_detects_failures(self, mock_entry_points: MagicMock) -> None:
        """Report has_errors detects import/instantiation failures.

        Only ``import_error`` and ``instantiation_error`` count as errors.
        ``namespace_rejected`` is a policy outcome, not an error.
        """
        ep_import = _make_entry_point(
            "import-fail", "omnibase_infra.broken:Plugin", None
        )
        ep_inst = _make_entry_point(
            "inst-fail",
            "omnibase_infra.plugins:FailingPlugin",
            FailingConstructorPlugin,
        )
        ep_ns = _make_entry_point("ns-reject", "evil:Plugin", MockPlugin)
        mock_entry_points.return_value = [ep_import, ep_inst, ep_ns]

        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        # has_errors is True due to import_error and instantiation_error
        assert report.has_errors
        statuses = {e.entry_point_name: e.status for e in report.entries}
        assert statuses["import-fail"] == "import_error"
        assert statuses["inst-fail"] == "instantiation_error"
        assert statuses["ns-reject"] == "namespace_rejected"


# ---------------------------------------------------------------------------
# TestSecurityConfigPluginNamespaces (3 tests)
# ---------------------------------------------------------------------------


class TestSecurityConfigPluginNamespaces:
    """Tests for plugin-related fields on ModelSecurityConfig."""

    def test_default_returns_trusted_prefixes(self) -> None:
        """Default config returns TRUSTED_PLUGIN_NAMESPACE_PREFIXES."""
        config = ModelSecurityConfig()
        effective = config.get_effective_plugin_namespaces()

        assert effective == TRUSTED_PLUGIN_NAMESPACE_PREFIXES

    def test_third_party_disabled_ignores_custom(self) -> None:
        """Third-party disabled ignores custom namespaces."""
        config = ModelSecurityConfig(
            allow_third_party_plugins=False,
            allowed_plugin_namespaces=("malicious.namespace.",),
        )
        effective = config.get_effective_plugin_namespaces()

        assert effective == TRUSTED_PLUGIN_NAMESPACE_PREFIXES
        assert "malicious.namespace." not in effective

    def test_third_party_enabled_returns_custom(self) -> None:
        """Third-party enabled returns custom namespaces."""
        custom = ("mycompany.plugins.", "partner.plugins.")
        config = ModelSecurityConfig(
            allow_third_party_plugins=True,
            allowed_plugin_namespaces=custom,
        )
        effective = config.get_effective_plugin_namespaces()

        assert effective == custom


# ---------------------------------------------------------------------------
# TestSecurityPolicyEnforcement (2 tests)
# ---------------------------------------------------------------------------


class TestSecurityPolicyEnforcement:
    """Tests for end-to-end security policy enforcement in discovery.

    Verifies that the security_config parameter correctly controls which
    namespaces are accepted during entry-point discovery.
    """

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_no_args_default_security_only_trusted(
        self, mock_entry_points: MagicMock
    ) -> None:
        """No args -> default security config -> only trusted namespaces loaded."""
        ep_trusted = _make_entry_point(
            "trusted", "omnibase_infra.plugins:MockPlugin", MockPlugin
        )
        ep_untrusted = _make_entry_point(
            "untrusted", "third_party.plugins:Plugin", MockPlugin
        )
        mock_entry_points.return_value = [ep_trusted, ep_untrusted]

        registry = RegistryDomainPlugin()
        # No security_config argument -- bare call must be secure
        report = registry.discover_from_entry_points()

        assert len(report.accepted) == 1
        assert report.accepted[0] == "mock-plugin"
        rejected_names = [e.entry_point_name for e in report.rejected]
        assert "untrusted" in rejected_names

    @patch("omnibase_infra.runtime.protocol_domain_plugin.entry_points")
    def test_custom_security_config_respects_third_party(
        self, mock_entry_points: MagicMock
    ) -> None:
        """Custom security_config with third-party enabled -> custom namespaces respected."""

        class CustomPlugin(MockPlugin):
            def __init__(self) -> None:
                super().__init__(plugin_id="custom-plugin")

        ep = _make_entry_point(
            "custom",
            "mycompany.plugins.custom:CustomPlugin",
            CustomPlugin,
        )
        mock_entry_points.return_value = [ep]

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

        assert len(report.accepted) == 1
        assert report.accepted[0] == "custom-plugin"
        assert registry.get("custom-plugin") is not None


__all__: list[str] = [
    "TestPluginNamespaceValidation",
    "TestDiscoverFromEntryPoints",
    "TestDiscoveryReport",
    "TestSecurityConfigPluginNamespaces",
    "TestSecurityPolicyEnforcement",
]
