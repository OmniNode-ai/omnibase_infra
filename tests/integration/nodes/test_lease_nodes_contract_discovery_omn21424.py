# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Entry-point discovery finds the three generic lease nodes and their handlers (OMN-21424)."""

from __future__ import annotations

import importlib

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts

pytestmark = [pytest.mark.integration]

NODES = {
    "node_lease_orchestrator": {
        "HandlerLeaseAcquire",
        "HandlerLeaseRenew",
        "HandlerLeaseRelease",
    },
    "node_lease_reducer": {"HandlerLeaseReducer"},
    "node_projection_lease": {"HandlerLeaseProjection", "HandlerLeaseLedger"},
}


@pytest.mark.parametrize(("node_name", "handler_names"), sorted(NODES.items()))
def test_lease_node_is_discovered_and_its_handlers_import(
    monkeypatch: pytest.MonkeyPatch, node_name: str, handler_names: set[str]
) -> None:
    monkeypatch.setenv("ONEX_ACTIVE_RUNTIME_PACKAGES", "omnibase-infra")
    manifest = discover_contracts()
    assert not [e for e in manifest.errors if e.entry_point_name == node_name]
    discovered = [c for c in manifest.contracts if c.entry_point_name == node_name]
    assert len(discovered) == 1
    assert discovered[0].handler_routing is not None
    routes = discovered[0].handler_routing.handlers
    assert {route.handler.name for route in routes} == handler_names
    for route in routes:
        assert hasattr(
            importlib.import_module(route.handler.module), route.handler.name
        )
