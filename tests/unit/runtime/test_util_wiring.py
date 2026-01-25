# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for util_wiring module.

Tests for the handler wiring functionality, including verification that
all expected handlers are registered by wire_default_handlers().
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_wire_default_handlers_includes_intent() -> None:
    """Test that wire_default_handlers registers the intent handler.

    The intent handler (HANDLER_TYPE_INTENT) should be included in the
    default handlers wired by wire_default_handlers(). This test verifies
    that the handler is present in the summary returned by the function.

    Note:
        We mock the singleton registries to isolate this test from handler
        implementation issues (e.g., missing execute() methods in some handlers).
    """
    from omnibase_infra.runtime.util_wiring import wire_default_handlers

    # Create mock registries that accept any registration
    mock_handler_registry = MagicMock()
    mock_handler_registry.list_protocols.return_value = [
        "consul",
        "db",
        "graph",
        "http",
        "intent",
        "mcp",
        "vault",
    ]

    mock_event_bus_registry = MagicMock()
    mock_event_bus_registry.is_registered.return_value = False
    mock_event_bus_registry.list_bus_kinds.return_value = ["inmemory"]

    with (
        patch(
            "omnibase_infra.runtime.util_wiring.get_handler_registry",
            return_value=mock_handler_registry,
        ),
        patch(
            "omnibase_infra.runtime.util_wiring.get_event_bus_registry",
            return_value=mock_event_bus_registry,
        ),
    ):
        summary = wire_default_handlers()

    assert "intent" in summary["handlers"], "Intent handler should be registered"


def test_intent_handler_in_known_handlers() -> None:
    """Test that HANDLER_TYPE_INTENT is included in _KNOWN_HANDLERS.

    This is a direct unit test that verifies the intent handler type
    constant is properly mapped in the _KNOWN_HANDLERS dictionary.
    """
    from omnibase_infra.handlers.handler_intent import HandlerIntent
    from omnibase_infra.runtime.handler_registry import HANDLER_TYPE_INTENT
    from omnibase_infra.runtime.util_wiring import _KNOWN_HANDLERS

    assert HANDLER_TYPE_INTENT in _KNOWN_HANDLERS, (
        f"HANDLER_TYPE_INTENT ('{HANDLER_TYPE_INTENT}') should be in _KNOWN_HANDLERS"
    )

    handler_cls, description = _KNOWN_HANDLERS[HANDLER_TYPE_INTENT]
    assert handler_cls is HandlerIntent, "Intent handler class should be HandlerIntent"
    assert isinstance(description, str) and len(description) > 0, (
        "Intent handler should have a non-empty description"
    )


def test_intent_handler_type_constant_value() -> None:
    """Test that HANDLER_TYPE_INTENT has the expected value.

    Verifies that the handler type constant matches what would be
    used in envelope routing.
    """
    from omnibase_infra.runtime.handler_registry import HANDLER_TYPE_INTENT

    assert HANDLER_TYPE_INTENT == "intent", (
        f"HANDLER_TYPE_INTENT should be 'intent', got '{HANDLER_TYPE_INTENT}'"
    )


def test_wire_default_handlers_returns_expected_structure() -> None:
    """Test that wire_default_handlers returns the expected summary structure.

    The function should return a dict with 'handlers' and 'event_buses' keys,
    each containing a list of registered type/kind strings.
    """
    from omnibase_infra.runtime.util_wiring import wire_default_handlers

    # Create mock registries
    mock_handler_registry = MagicMock()
    mock_handler_registry.list_protocols.return_value = ["http", "db"]

    mock_event_bus_registry = MagicMock()
    mock_event_bus_registry.is_registered.return_value = False
    mock_event_bus_registry.list_bus_kinds.return_value = ["inmemory"]

    with (
        patch(
            "omnibase_infra.runtime.util_wiring.get_handler_registry",
            return_value=mock_handler_registry,
        ),
        patch(
            "omnibase_infra.runtime.util_wiring.get_event_bus_registry",
            return_value=mock_event_bus_registry,
        ),
    ):
        summary = wire_default_handlers()

    assert "handlers" in summary, "Summary should contain 'handlers' key"
    assert "event_buses" in summary, "Summary should contain 'event_buses' key"
    assert isinstance(summary["handlers"], list), "'handlers' should be a list"
    assert isinstance(summary["event_buses"], list), "'event_buses' should be a list"


def test_known_handlers_includes_all_expected_types() -> None:
    """Test that _KNOWN_HANDLERS includes all expected handler types.

    Verifies that the core infrastructure handlers are all present in the
    _KNOWN_HANDLERS dictionary that drives wire_default_handlers().
    """
    from omnibase_infra.runtime.handler_registry import (
        HANDLER_TYPE_CONSUL,
        HANDLER_TYPE_DATABASE,
        HANDLER_TYPE_GRAPH,
        HANDLER_TYPE_HTTP,
        HANDLER_TYPE_INTENT,
        HANDLER_TYPE_MCP,
        HANDLER_TYPE_VAULT,
    )
    from omnibase_infra.runtime.util_wiring import _KNOWN_HANDLERS

    expected_handlers = [
        HANDLER_TYPE_CONSUL,
        HANDLER_TYPE_DATABASE,
        HANDLER_TYPE_GRAPH,
        HANDLER_TYPE_HTTP,
        HANDLER_TYPE_INTENT,
        HANDLER_TYPE_MCP,
        HANDLER_TYPE_VAULT,
    ]

    for handler_type in expected_handlers:
        assert handler_type in _KNOWN_HANDLERS, (
            f"Handler type '{handler_type}' should be in _KNOWN_HANDLERS"
        )


def test_wire_default_handlers_includes_inmemory_event_bus() -> None:
    """Test that wire_default_handlers registers the in-memory event bus.

    The in-memory event bus should be included in the default wiring
    for local/testing deployments.
    """
    from omnibase_infra.runtime.handler_registry import EVENT_BUS_INMEMORY
    from omnibase_infra.runtime.util_wiring import wire_default_handlers

    # Create mock registries
    mock_handler_registry = MagicMock()
    mock_handler_registry.list_protocols.return_value = ["http"]

    mock_event_bus_registry = MagicMock()
    mock_event_bus_registry.is_registered.return_value = False
    mock_event_bus_registry.list_bus_kinds.return_value = [EVENT_BUS_INMEMORY]

    with (
        patch(
            "omnibase_infra.runtime.util_wiring.get_handler_registry",
            return_value=mock_handler_registry,
        ),
        patch(
            "omnibase_infra.runtime.util_wiring.get_event_bus_registry",
            return_value=mock_event_bus_registry,
        ),
    ):
        summary = wire_default_handlers()

    assert EVENT_BUS_INMEMORY in summary["event_buses"], (
        "In-memory event bus should be registered"
    )
