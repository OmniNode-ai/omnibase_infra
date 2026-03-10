# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for EffectMockRegistry.

Tests cover registration, resolution, error handling, and
convenience methods of the mock registry.

Related:
    - OMN-1336: Add thread-local utility for EffectMockRegistry
"""

from __future__ import annotations

import pytest

from omnibase_infra.testing.service_effect_mock_registry import (
    EffectMockRegistry,
)


@pytest.mark.unit
class TestEffectMockRegistry:
    """Tests for EffectMockRegistry core operations."""

    def test_register_and_resolve(self) -> None:
        """Register a mock and resolve it by protocol name."""
        registry = EffectMockRegistry()
        mock_bus = object()
        registry.register("ProtocolEventBus", mock_bus)

        resolved = registry.resolve("ProtocolEventBus")
        assert resolved is mock_bus

    def test_resolve_unknown_raises_key_error(self) -> None:
        """Resolving an unregistered protocol raises KeyError."""
        registry = EffectMockRegistry()

        with pytest.raises(KeyError, match="No mock registered for 'ProtocolEventBus'"):
            registry.resolve("ProtocolEventBus")

    def test_resolve_error_lists_registered_protocols(self) -> None:
        """KeyError message includes list of registered protocols."""
        registry = EffectMockRegistry()
        registry.register("ProtocolA", object())
        registry.register("ProtocolB", object())

        with pytest.raises(KeyError, match="ProtocolA, ProtocolB"):
            registry.resolve("ProtocolC")

    def test_resolve_error_shows_none_when_empty(self) -> None:
        """KeyError message shows (none) when registry is empty."""
        registry = EffectMockRegistry()

        with pytest.raises(KeyError, match=r"\(none\)"):
            registry.resolve("ProtocolEventBus")

    def test_register_empty_name_raises_value_error(self) -> None:
        """Registering with an empty protocol name raises ValueError."""
        registry = EffectMockRegistry()

        with pytest.raises(ValueError, match="non-empty string"):
            registry.register("", object())

    def test_has_returns_true_for_registered(self) -> None:
        """has() returns True for registered protocols."""
        registry = EffectMockRegistry()
        registry.register("ProtocolEventBus", object())

        assert registry.has("ProtocolEventBus") is True

    def test_has_returns_false_for_unregistered(self) -> None:
        """has() returns False for unregistered protocols."""
        registry = EffectMockRegistry()

        assert registry.has("ProtocolEventBus") is False

    def test_unregister_removes_mock(self) -> None:
        """unregister() removes a registered mock."""
        registry = EffectMockRegistry()
        registry.register("ProtocolEventBus", object())

        registry.unregister("ProtocolEventBus")
        assert registry.has("ProtocolEventBus") is False

    def test_unregister_unknown_raises_key_error(self) -> None:
        """unregister() raises KeyError for unknown protocols."""
        registry = EffectMockRegistry()

        with pytest.raises(KeyError, match="Cannot unregister"):
            registry.unregister("ProtocolEventBus")

    def test_clear_removes_all(self) -> None:
        """clear() removes all registrations."""
        registry = EffectMockRegistry()
        registry.register("ProtocolA", object())
        registry.register("ProtocolB", object())

        registry.clear()
        assert len(registry) == 0
        assert registry.registered_protocols == []

    def test_registered_protocols_sorted(self) -> None:
        """registered_protocols returns sorted list."""
        registry = EffectMockRegistry()
        registry.register("ProtocolC", object())
        registry.register("ProtocolA", object())
        registry.register("ProtocolB", object())

        assert registry.registered_protocols == [
            "ProtocolA",
            "ProtocolB",
            "ProtocolC",
        ]

    def test_len_reflects_registrations(self) -> None:
        """len() reflects the number of registrations."""
        registry = EffectMockRegistry()
        assert len(registry) == 0

        registry.register("ProtocolA", object())
        assert len(registry) == 1

        registry.register("ProtocolB", object())
        assert len(registry) == 2

    def test_overwrite_registration(self) -> None:
        """Registering the same protocol name overwrites the previous mock."""
        registry = EffectMockRegistry()
        mock_v1 = object()
        mock_v2 = object()

        registry.register("ProtocolEventBus", mock_v1)
        registry.register("ProtocolEventBus", mock_v2)

        assert registry.resolve("ProtocolEventBus") is mock_v2
        assert len(registry) == 1

    def test_repr(self) -> None:
        """repr() includes registered protocol names."""
        registry = EffectMockRegistry()
        registry.register("ProtocolA", object())
        registry.register("ProtocolB", object())

        result = repr(registry)
        assert "EffectMockRegistry" in result
        assert "ProtocolA" in result
        assert "ProtocolB" in result

    def test_repr_empty(self) -> None:
        """repr() works for empty registry."""
        registry = EffectMockRegistry()
        result = repr(registry)
        assert "EffectMockRegistry" in result
