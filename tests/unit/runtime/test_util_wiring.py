# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for util_wiring module.

Tests for the handler wiring functionality, including verification that
all expected handlers are registered by wire_default_handlers().
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_wire_default_handlers_registers_core_handlers() -> None:
    """Test that wire_default_handlers registers the core handlers (no intent)."""
    from omnibase_infra.runtime.util_wiring import wire_default_handlers

    mock_handler_registry = MagicMock()
    mock_handler_registry.list_protocols.return_value = [
        "db",
        "graph",
        "http",
        "mcp",
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

    # Intent handler was removed (TEMPORARY demo wiring)
    assert "intent" not in summary["handlers"], (
        "Intent handler should not be registered"
    )
    for handler in ("db", "graph", "http", "mcp"):
        assert handler in summary["handlers"], f"{handler} handler should be registered"


def test_wire_default_handlers_returns_expected_structure() -> None:
    """Test that wire_default_handlers returns the expected summary structure."""
    from omnibase_infra.runtime.util_wiring import wire_default_handlers

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
    """Test that _HANDLER_CONTRACT_PATHS includes all expected handler types."""
    from pathlib import Path

    from omnibase_infra.runtime.handler_registry import (
        HANDLER_TYPE_DATABASE,
        HANDLER_TYPE_GRAPH,
        HANDLER_TYPE_HTTP,
        HANDLER_TYPE_MCP,
    )
    from omnibase_infra.runtime.util_wiring import _HANDLER_CONTRACT_PATHS

    expected_handlers = [
        HANDLER_TYPE_DATABASE,
        HANDLER_TYPE_GRAPH,
        HANDLER_TYPE_HTTP,
        HANDLER_TYPE_MCP,
    ]

    for handler_type in expected_handlers:
        assert handler_type in _HANDLER_CONTRACT_PATHS, (
            f"Handler type '{handler_type}' should be in _HANDLER_CONTRACT_PATHS"
        )
        contract_path = _HANDLER_CONTRACT_PATHS[handler_type]
        assert isinstance(contract_path, Path), (
            f"Contract path for '{handler_type}' should be a Path object"
        )


def test_intent_handler_not_in_known_handlers() -> None:
    """Test that HANDLER_TYPE_INTENT is NOT in _HANDLER_CONTRACT_PATHS (removed)."""
    from omnibase_infra.runtime.handler_registry import HANDLER_TYPE_INTENT
    from omnibase_infra.runtime.util_wiring import _HANDLER_CONTRACT_PATHS

    assert HANDLER_TYPE_INTENT not in _HANDLER_CONTRACT_PATHS, (
        "Intent handler (TEMPORARY demo wiring) should have been removed"
    )


def test_all_handler_contracts_use_canonical_name() -> None:
    """Every handler dir must have handler_contract.yaml, not contract.yaml."""
    from pathlib import Path

    HANDLERS_BASE = (
        Path(__file__).parent.parent.parent.parent
        / "src/omnibase_infra/contracts/handlers"
    )

    for handler_dir in HANDLERS_BASE.iterdir():
        if not handler_dir.is_dir() or handler_dir.name.startswith("_"):
            continue
        assert (handler_dir / "handler_contract.yaml").exists(), (
            f"{handler_dir.name}/ missing handler_contract.yaml"
        )
        assert not (handler_dir / "contract.yaml").exists(), (
            f"{handler_dir.name}/ has ambiguous dual contract files"
        )


def test_intent_handler_dir_removed() -> None:
    """Intent handler directory (TEMPORARY demo wiring) must be deleted."""
    from pathlib import Path

    HANDLERS_BASE = (
        Path(__file__).parent.parent.parent.parent
        / "src/omnibase_infra/contracts/handlers"
    )
    assert not (HANDLERS_BASE / "intent").exists()


def test_wire_default_handlers_includes_inmemory_event_bus() -> None:
    """Test that wire_default_handlers registers the in-memory event bus."""
    from omnibase_infra.runtime.handler_registry import EVENT_BUS_INMEMORY
    from omnibase_infra.runtime.util_wiring import wire_default_handlers

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
