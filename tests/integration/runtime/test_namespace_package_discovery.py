# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for namespace package handling in contract discovery [OMN-8538].

Verifies that discover_contracts() gracefully skips entry points that resolve
to namespace packages (no __file__) rather than raising TypeError.
"""

from __future__ import annotations

import sys
import types
from importlib.metadata import EntryPoint
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


def _make_namespace_package(name: str) -> types.ModuleType:
    """Create a minimal namespace package (has __path__ but no __file__)."""
    pkg = types.ModuleType(name)
    pkg.__path__ = []  # type: ignore[assignment]
    # Deliberately no __file__ — this is what makes it a namespace package
    return pkg


def _make_entry_point(name: str, value: str) -> MagicMock:
    ep = MagicMock(spec=EntryPoint)
    ep.name = name
    ep.value = value
    ep.dist = MagicMock()
    ep.dist.name = "test-pkg"
    return ep


@pytest.mark.integration
def test_discover_contracts_skips_namespace_package_entry_point(tmp_path: Path) -> None:
    """An entry point resolving to a namespace package (no __file__) must be skipped.

    Before OMN-8538 this raised TypeError; now it logs a warning and continues.
    The remaining valid entry points must still be discovered.
    """
    from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts

    ns_pkg = _make_namespace_package("fake_namespace_pkg")
    ns_ep = _make_entry_point("ns-node", "fake_namespace_pkg")
    ns_ep.load.return_value = ns_pkg

    with patch(
        "omnibase_infra.runtime.auto_wiring.discovery.entry_points",
        return_value=[ns_ep],
    ):
        # Must not raise; should return an empty (or warning-only) manifest
        manifest = discover_contracts()

    assert manifest is not None
    # Namespace package had no contract.yaml, so it must not appear in contracts
    assert len(manifest.contracts) == 0 or all(
        c.entry_point_name != "ns-node" for c in manifest.contracts
    )


@pytest.mark.integration
def test_resolve_contract_path_namespace_package_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """_resolve_contract_path raises FileNotFoundError for namespace packages without contract.yaml."""
    from omnibase_infra.runtime.auto_wiring.discovery import _resolve_contract_path

    ns_pkg = _make_namespace_package("fake_ns_no_contract")
    ns_pkg.__path__ = [str(tmp_path)]  # type: ignore[assignment]
    # No contract.yaml in tmp_path

    with pytest.raises(FileNotFoundError, match="namespace package"):
        _resolve_contract_path(ns_pkg)  # type: ignore[arg-type]


@pytest.mark.integration
def test_resolve_contract_path_namespace_package_finds_contract(tmp_path: Path) -> None:
    """_resolve_contract_path finds contract.yaml inside a namespace package __path__ entry."""
    from omnibase_infra.runtime.auto_wiring.discovery import _resolve_contract_path

    contract = tmp_path / "contract.yaml"
    contract.write_text("name: test-node\n")

    ns_pkg = _make_namespace_package("fake_ns_with_contract")
    ns_pkg.__path__ = [str(tmp_path)]  # type: ignore[assignment]

    result = _resolve_contract_path(ns_pkg)  # type: ignore[arg-type]
    assert result == contract
