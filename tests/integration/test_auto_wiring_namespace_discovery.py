# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for contract auto-discovery with namespace packages [OMN-8538].

Verifies that discover_contracts() completes without crashing when entry points
include namespace packages (modules without __file__), and that errors are
captured per-entry-point rather than aborting the entire scan.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import (
    _resolve_contract_path,
    discover_contracts,
)

_EP_MODULE = "omnibase_infra.runtime.auto_wiring.discovery.entry_points"


def _make_entry_point(
    name: str, *, load_value: object, dist_name: str = "test-pkg"
) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.value = f"{dist_name}:{name}"
    ep.dist.name = dist_name
    ep.dist.version = "1.0.0"
    ep.load.return_value = load_value
    return ep


@pytest.mark.integration
def test_resolve_contract_path_namespace_package(tmp_path: Path) -> None:
    """_resolve_contract_path finds contract.yaml via __path__ for namespace packages."""
    pkg_dir = tmp_path / "node_ns"
    pkg_dir.mkdir()
    contract = pkg_dir / "contract.yaml"
    contract.write_text("name: test")

    ns_mod = types.ModuleType("node_ns")
    ns_mod.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    # No __file__ attribute — simulates namespace package

    result = _resolve_contract_path(ns_mod)  # type: ignore[arg-type]
    assert result == contract


@pytest.mark.integration
def test_discover_contracts_tolerates_namespace_entry_points(tmp_path: Path) -> None:
    """discover_contracts() records errors for namespace packages but does not abort."""
    # Entry point with a namespace package that has no contract.yaml
    ns_mod = types.ModuleType("node_ns_no_contract")
    ns_mod.__path__ = [str(tmp_path / "does_not_exist")]  # type: ignore[attr-defined]

    ep = _make_entry_point("ns_node", load_value=ns_mod)

    with patch(_EP_MODULE, return_value=[ep]):
        manifest = discover_contracts()

    # Should complete without raising; the missing contract is recorded as an error
    assert manifest.total_discovered == 0
    assert manifest.total_errors == 1


@pytest.mark.integration
def test_discover_contracts_handles_type_error_from_namespace_getfile(
    tmp_path: Path,
) -> None:
    """TypeError from inspect.getfile on a module with no __file__ or __path__ is captured."""
    # Simulate a module where inspect.getfile raises TypeError:
    # no __path__ (skips namespace branch) and no __file__ (causes TypeError in Strategy 3).
    bare_mod = types.ModuleType("node_bare_raises")
    # Deliberately omit __path__ and __file__ so inspect.getfile raises TypeError

    ep = _make_entry_point("raises_node", load_value=bare_mod)

    # This must not raise
    with patch(_EP_MODULE, return_value=[ep]):
        manifest = discover_contracts()

    # TypeError captured, discovery continues
    assert manifest.total_errors >= 1
