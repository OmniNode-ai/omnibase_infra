# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Entry-point discovery finds the three declared-state nodes and their handlers."""

from __future__ import annotations

import importlib

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts

pytestmark = [pytest.mark.integration]

NODES = {
    "node_declared_state_diff_compute": "HandlerDeclaredStateDiff",
    "node_declared_state_apply_effect": "HandlerDeclaredStateApply",
    "node_declared_state_reconcile_orchestrator": "HandlerDeclaredStateReconcile",
}


@pytest.mark.parametrize(("node_name", "handler_name"), sorted(NODES.items()))
def test_node_is_discovered_and_its_handler_imports(
    monkeypatch: pytest.MonkeyPatch, node_name: str, handler_name: str
) -> None:
    """Each contract loads without error and names an importable handler."""
    monkeypatch.setenv("ONEX_ACTIVE_RUNTIME_PACKAGES", "omnibase-infra")
    manifest = discover_contracts()
    assert not [e for e in manifest.errors if e.entry_point_name == node_name]
    discovered = [c for c in manifest.contracts if c.entry_point_name == node_name]
    assert len(discovered) == 1
    assert discovered[0].package_name == "omnibase_infra"
    assert discovered[0].handler_routing is not None
    names = {route.handler.name for route in discovered[0].handler_routing.handlers}
    assert names == {handler_name}
    route = discovered[0].handler_routing.handlers[0]
    assert hasattr(importlib.import_module(route.handler.module), handler_name)
