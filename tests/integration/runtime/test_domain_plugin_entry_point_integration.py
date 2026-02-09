# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for real entry_point discovery (OMN-2022).

These tests verify that the ``PluginRegistration`` class is discoverable via
``importlib.metadata.entry_points(group="onex.domain_plugins")`` when the
package is installed, and that the full
``RegistryDomainPlugin.discover_from_entry_points()`` pipeline accepts it.

Unlike the unit tests in ``test_discover_from_entry_points.py`` which mock
``entry_points()``, these tests use the **real** installed entry points to
verify end-to-end plugin discovery works with the actual package metadata.

Prerequisites:
    - ``poetry install`` must have been run so that the entry_point declared
      in ``pyproject.toml`` under ``[tool.poetry.plugins."onex.domain_plugins"]``
      is registered in the installed package metadata.

Dependencies:
    - OMN-2017: Core ``discover_from_entry_points()`` method
    - OMN-2014: ``pyproject.toml`` entry_point declaration

Related:
    - OMN-2022: Write integration test for real entry_point discovery
    - src/omnibase_infra/nodes/node_registration_orchestrator/plugin.py
    - src/omnibase_infra/runtime/protocol_domain_plugin.py
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

import pytest

from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
    PluginRegistration,
)
from omnibase_infra.runtime.constants_security import (
    DOMAIN_PLUGIN_ENTRY_POINT_GROUP,
)
from omnibase_infra.runtime.models.model_security_config import ModelSecurityConfig
from omnibase_infra.runtime.protocol_domain_plugin import (
    ProtocolDomainPlugin,
    RegistryDomainPlugin,
)

pytestmark = pytest.mark.integration


# =============================================================================
# Helpers
# =============================================================================


def _get_registration_entry_point() -> EntryPoint | None:
    """Return the 'registration' entry point from the installed package metadata.

    Returns:
        The entry point object, or None if not found.
    """
    eps = entry_points(group=DOMAIN_PLUGIN_ENTRY_POINT_GROUP)
    for ep in eps:
        if ep.name == "registration":
            return ep
    return None


# =============================================================================
# Test: Raw entry_point discoverability
# =============================================================================


class TestEntryPointDiscoverability:
    """Verify PluginRegistration is discoverable via importlib.metadata."""

    def test_entry_point_exists_in_group(self) -> None:
        """The 'registration' entry point exists in the onex.domain_plugins group."""
        ep = _get_registration_entry_point()
        assert ep is not None, (
            "Entry point 'registration' not found in group "
            f"'{DOMAIN_PLUGIN_ENTRY_POINT_GROUP}'. "
            "Has 'poetry install' been run?"
        )

    def test_entry_point_value_matches_plugin_module(self) -> None:
        """The entry point value points to the correct module and class."""
        ep = _get_registration_entry_point()
        assert ep is not None, "Entry point 'registration' not found"

        expected_value = (
            "omnibase_infra.nodes.node_registration_orchestrator.plugin"
            ":PluginRegistration"
        )
        assert ep.value == expected_value, (
            f"Entry point value mismatch: expected '{expected_value}', got '{ep.value}'"
        )

    def test_entry_point_loads_plugin_class(self) -> None:
        """Loading the entry point returns the PluginRegistration class."""
        ep = _get_registration_entry_point()
        assert ep is not None, "Entry point 'registration' not found"

        loaded_class = ep.load()
        assert loaded_class is PluginRegistration, (
            f"Expected PluginRegistration class, got {loaded_class}"
        )

    def test_loaded_class_instantiates(self) -> None:
        """The loaded class can be instantiated with no arguments."""
        ep = _get_registration_entry_point()
        assert ep is not None, "Entry point 'registration' not found"

        loaded_class = ep.load()
        instance = loaded_class()
        assert instance is not None
        assert instance.plugin_id == "registration"
        assert instance.display_name == "Registration"

    def test_loaded_instance_satisfies_protocol(self) -> None:
        """The instantiated plugin satisfies ProtocolDomainPlugin."""
        ep = _get_registration_entry_point()
        assert ep is not None, "Entry point 'registration' not found"

        loaded_class = ep.load()
        instance = loaded_class()
        assert isinstance(instance, ProtocolDomainPlugin), (
            f"PluginRegistration does not satisfy ProtocolDomainPlugin. "
            f"Type: {type(instance)}"
        )


# =============================================================================
# Test: Full discover_from_entry_points() pipeline
# =============================================================================


class TestDiscoverFromEntryPointsIntegration:
    """Verify full discovery pipeline with real installed entry points."""

    def test_discover_accepts_registration_plugin(self) -> None:
        """discover_from_entry_points() accepts 'registration' plugin."""
        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            security_config=ModelSecurityConfig(),
        )

        assert "registration" in report.accepted, (
            f"Expected 'registration' in accepted plugins. "
            f"Accepted: {report.accepted}. "
            f"Entries: {[(e.entry_point_name, e.status, e.reason) for e in report.entries]}"
        )

    def test_discover_registers_plugin_in_registry(self) -> None:
        """After discovery, the plugin is accessible via registry.get()."""
        registry = RegistryDomainPlugin()
        registry.discover_from_entry_points(
            security_config=ModelSecurityConfig(),
        )

        plugin = registry.get("registration")
        assert plugin is not None, (
            "Plugin 'registration' not found in registry after discovery"
        )
        assert plugin.plugin_id == "registration"
        assert plugin.display_name == "Registration"

    def test_discover_report_has_no_errors(self) -> None:
        """The discovery report should have no import or instantiation errors."""
        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            security_config=ModelSecurityConfig(),
        )

        assert not report.has_errors, (
            f"Discovery report has errors. "
            f"Entries: {[(e.entry_point_name, e.status, e.reason) for e in report.entries]}"
        )

    def test_discover_report_discovered_count(self) -> None:
        """The report shows at least 1 discovered entry point."""
        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            security_config=ModelSecurityConfig(),
        )

        assert report.discovered_count >= 1, (
            f"Expected at least 1 discovered entry point, got {report.discovered_count}"
        )

    def test_discover_report_entry_status(self) -> None:
        """The registration entry has status 'accepted' with correct plugin_id."""
        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            security_config=ModelSecurityConfig(),
        )

        registration_entries = [
            e for e in report.entries if e.entry_point_name == "registration"
        ]
        assert len(registration_entries) == 1, (
            f"Expected exactly 1 'registration' entry, got {len(registration_entries)}"
        )

        entry = registration_entries[0]
        assert entry.status == "accepted"
        assert entry.plugin_id == "registration"
        assert entry.module_path == (
            "omnibase_infra.nodes.node_registration_orchestrator.plugin"
        )

    def test_discover_with_default_security_config(self) -> None:
        """Bare call with no security_config still discovers the plugin.

        This verifies the default security config includes omnibase_infra.
        in its trusted namespaces.
        """
        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points()

        assert "registration" in report.accepted, (
            f"Default security config should accept omnibase_infra plugins. "
            f"Accepted: {report.accepted}"
        )

    def test_discover_with_strict_mode(self) -> None:
        """strict=True succeeds when all entry points are valid."""
        registry = RegistryDomainPlugin()
        # Should not raise -- all installed plugins are valid
        report = registry.discover_from_entry_points(
            security_config=ModelSecurityConfig(),
            strict=True,
        )

        assert "registration" in report.accepted

    def test_explicit_registration_takes_precedence(self) -> None:
        """Explicit registration wins over entry_point discovery."""
        explicit_plugin = PluginRegistration()
        registry = RegistryDomainPlugin()
        registry.register(explicit_plugin)

        report = registry.discover_from_entry_points(
            security_config=ModelSecurityConfig(),
        )

        # The discovered plugin should be skipped as duplicate
        assert "registration" not in report.accepted

        registration_entries = [
            e for e in report.entries if e.entry_point_name == "registration"
        ]
        assert len(registration_entries) == 1
        assert registration_entries[0].status == "duplicate_skipped"

        # The explicit plugin should remain in the registry
        assert registry.get("registration") is explicit_plugin

    def test_discover_report_uses_correct_group(self) -> None:
        """The report records the correct entry-point group name."""
        registry = RegistryDomainPlugin()
        report = registry.discover_from_entry_points(
            security_config=ModelSecurityConfig(),
        )

        assert report.group == DOMAIN_PLUGIN_ENTRY_POINT_GROUP


__all__: list[str] = [
    "TestDiscoverFromEntryPointsIntegration",
    "TestEntryPointDiscoverability",
]
