# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for duplicate auto-wiring contract discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts

pytestmark = pytest.mark.integration

_EP_MODULE = "omnibase_infra.runtime.auto_wiring.discovery.entry_points"


@dataclass(frozen=True)
class _FakeDist:
    name: str
    version: str


@dataclass(frozen=True)
class _FakeEntryPoint:
    name: str
    dist: _FakeDist
    node_cls: type

    def load(self) -> type:
        return self.node_cls


def _write_contract(directory: Path, *, name: str) -> Path:
    directory.mkdir()
    contract_path = directory / "contract.yaml"
    contract_path.write_text(
        dedent(f"""\
            name: "{name}"
            node_type: "effect"
            contract_version:
              major: 1
              minor: 0
              patch: 0
            node_version: "1.0.0"
            description: "Duplicate discovery integration fixture"
        """),
        encoding="utf-8",
    )
    return contract_path


def test_duplicate_contract_name_across_entry_point_packages_boots_with_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-package duplicate contract names are skipped before dispatcher wiring.

    This covers the OMN-11958 runtime crash shape: two installed packages expose
    ``onex.nodes`` entry points whose sibling contract files declare the same
    ``name``. Discovery must keep the first contract, report the second as a
    discovery error, and return a manifest that can continue into runtime boot.
    """
    monkeypatch.setenv("ONEX_ACTIVE_RUNTIME_PACKAGES", "omnibase_infra,omnimarket")

    infra_dir = tmp_path / "infra_node"
    market_dir = tmp_path / "market_node"
    duplicate_name = "node_architecture_graph_query_effect"
    _write_contract(infra_dir, name=duplicate_name)
    _write_contract(market_dir, name=duplicate_name)
    infra_module = infra_dir / "node.py"
    market_module = market_dir / "node.py"
    infra_module.write_text("", encoding="utf-8")
    market_module.write_text("", encoding="utf-8")

    infra_cls = type("InfraArchitectureGraphQueryEffect", (), {})
    market_cls = type("MarketArchitectureGraphQueryEffect", (), {})
    entry_points = [
        _FakeEntryPoint(
            name=duplicate_name,
            dist=_FakeDist(name="omnibase_infra", version="0.36.1"),
            node_cls=infra_cls,
        ),
        _FakeEntryPoint(
            name=duplicate_name,
            dist=_FakeDist(name="omnimarket", version="0.4.0"),
            node_cls=market_cls,
        ),
    ]

    def fake_getfile(node_cls: type) -> str:
        if node_cls is infra_cls:
            return str(infra_module)
        if node_cls is market_cls:
            return str(market_module)
        raise AssertionError(f"unexpected node class: {node_cls!r}")

    with (
        patch(_EP_MODULE, return_value=entry_points),
        patch("inspect.getfile", side_effect=fake_getfile),
    ):
        manifest = discover_contracts()

    assert manifest.total_discovered == 1
    assert manifest.contracts[0].name == duplicate_name
    assert manifest.contracts[0].package_name == "omnibase_infra"
    assert manifest.total_errors == 1
    assert manifest.errors[0].package_name == "omnimarket"
    assert duplicate_name in manifest.errors[0].error
    assert "Duplicate contract name" in manifest.errors[0].error
