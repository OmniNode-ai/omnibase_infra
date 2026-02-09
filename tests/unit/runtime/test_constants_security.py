# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for security constants.

Tests validate:
- Plugin namespace prefixes contain expected namespaces
- Plugin namespace prefixes are a superset of handler namespace prefixes
- Domain plugin entry point group has expected PEP 621 value
- Constants are immutable tuples (not lists)

.. versionadded:: 0.3.0
    Test coverage for plugin security constants (OMN-2010).

.. versionchanged:: 0.6.0
    Updated parity tests for omniclaude. plugin namespace (OMN-2047).

Related Tickets:
    - OMN-2010: Add plugin security constants
    - OMN-1519: Security hardening for handler namespace configuration
    - OMN-2047: Add omniclaude to trusted plugin namespace prefixes
"""

from __future__ import annotations

from omnibase_infra.runtime.constants_security import (
    DOMAIN_PLUGIN_ENTRY_POINT_GROUP,
    TRUSTED_HANDLER_NAMESPACE_PREFIXES,
    TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
)


class TestTrustedPluginNamespacePrefixes:
    """Tests for TRUSTED_PLUGIN_NAMESPACE_PREFIXES constant."""

    def test_contains_core_namespace(self) -> None:
        """Test that omnibase_core is a trusted plugin namespace."""
        assert "omnibase_core." in TRUSTED_PLUGIN_NAMESPACE_PREFIXES

    def test_contains_infra_namespace(self) -> None:
        """Test that omnibase_infra is a trusted plugin namespace."""
        assert "omnibase_infra." in TRUSTED_PLUGIN_NAMESPACE_PREFIXES

    def test_contains_omniclaude_namespace(self) -> None:
        """Test that omniclaude is a trusted plugin namespace (OMN-2047)."""
        assert "omniclaude." in TRUSTED_PLUGIN_NAMESPACE_PREFIXES

    def test_is_tuple(self) -> None:
        """Test that the constant is a tuple (immutable), not a list."""
        assert isinstance(TRUSTED_PLUGIN_NAMESPACE_PREFIXES, tuple)

    def test_all_prefixes_end_with_dot(self) -> None:
        """Test that all prefixes end with a dot for package boundary safety."""
        for prefix in TRUSTED_PLUGIN_NAMESPACE_PREFIXES:
            assert prefix.endswith("."), f"Prefix {prefix!r} must end with '.'"

    def test_does_not_include_spi(self) -> None:
        """Test that SPI namespace is excluded (protocols, not implementations)."""
        assert not any(
            p.startswith("omnibase_spi") for p in TRUSTED_PLUGIN_NAMESPACE_PREFIXES
        )


class TestPluginHandlerNamespaceRelationship:
    """Tests that plugin prefixes are a superset of handler prefixes.

    Plugin namespace prefixes include all handler prefixes plus additional
    first-party namespaces that provide domain plugins without handlers
    (e.g., omniclaude).

    .. versionchanged:: 0.6.0
        Renamed from TestPluginHandlerNamespaceParity. Plugin prefixes are
        now a superset of handler prefixes (OMN-2047).
    """

    def test_plugin_prefixes_superset_of_handler_prefixes(self) -> None:
        """Plugin prefixes must contain all handler prefixes."""
        assert set(TRUSTED_HANDLER_NAMESPACE_PREFIXES).issubset(
            set(TRUSTED_PLUGIN_NAMESPACE_PREFIXES)
        )

    def test_plugin_prefixes_include_extra_namespaces(self) -> None:
        """Plugin prefixes include namespaces not in handler prefixes.

        omniclaude. provides domain plugins but not handler implementations.
        """
        extra = set(TRUSTED_PLUGIN_NAMESPACE_PREFIXES) - set(
            TRUSTED_HANDLER_NAMESPACE_PREFIXES
        )
        assert "omniclaude." in extra


class TestDomainPluginEntryPointGroup:
    """Tests for DOMAIN_PLUGIN_ENTRY_POINT_GROUP constant."""

    def test_expected_value(self) -> None:
        """Test the entry point group has the expected PEP 621 value."""
        assert DOMAIN_PLUGIN_ENTRY_POINT_GROUP == "onex.domain_plugins"

    def test_is_string(self) -> None:
        """Test the constant is a string."""
        assert isinstance(DOMAIN_PLUGIN_ENTRY_POINT_GROUP, str)

    def test_no_leading_trailing_whitespace(self) -> None:
        """Test the value has no accidental whitespace."""
        assert (
            DOMAIN_PLUGIN_ENTRY_POINT_GROUP.strip() == DOMAIN_PLUGIN_ENTRY_POINT_GROUP
        )
