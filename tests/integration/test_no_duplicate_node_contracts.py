# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression: prevent infra-side node contracts whose `name:` collides with
omnimarket's contracts (the duplicate-local-ingress-route crash that
prevented omninode-runtime from booting before OMN-10865).

The runtime auto-wiring layer aliases every discovered node by its
contract `name:`. If two contracts share a name across packages, wiring
raises `ValueError: Duplicate local ingress route alias '<name>'`.

This test inspects every infra contract.yaml and fails if it declares a
`name:` that lives in the omnimarket-side nodes package — the known
collision shape that crashes boot. It reads the omnimarket sibling clone's
contract files rather than importing omnimarket, because omnibase_infra never
imports omnimarket (OMN-21279).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.integration]

INFRA_NODES_DIR = (
    Path(__file__).parent.parent.parent / "src" / "omnibase_infra" / "nodes"
)


def _read_contract_name(contract_path: Path) -> str | None:
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    return name if isinstance(name, str) else None


def _omnimarket_node_dir() -> Path | None:
    """Locate the omnimarket nodes directory in the sibling clone under $OMNI_HOME.

    Returns None if $OMNI_HOME is unset or holds no omnimarket clone (in which
    case we skip — the runtime wiring check is the source of truth for the
    duplicate-alias error).
    """
    workspace_root = os.environ.get("OMNI_HOME", "").strip()
    if not workspace_root:
        return None
    nodes_dir = Path(workspace_root) / "omnimarket" / "src" / "omnimarket" / "nodes"
    return nodes_dir if nodes_dir.is_dir() else None


def test_no_infra_contract_name_collides_with_omnimarket() -> None:
    market_dir = _omnimarket_node_dir()
    if market_dir is None:
        pytest.skip("no omnimarket sibling clone under $OMNI_HOME in this environment")

    market_names: dict[str, Path] = {}
    for contract in market_dir.rglob("contract.yaml"):
        name = _read_contract_name(contract)
        if name:
            market_names.setdefault(name, contract)

    infra_collisions: list[tuple[str, Path, Path]] = []
    for contract in INFRA_NODES_DIR.rglob("contract.yaml"):
        name = _read_contract_name(contract)
        if name and name in market_names:
            infra_collisions.append((name, contract, market_names[name]))

    assert not infra_collisions, (
        "Duplicate contract `name:` between omnibase_infra and omnimarket "
        "will crash omninode-runtime boot with `ValueError: Duplicate local "
        "ingress route alias '<name>'`. Pick one owner per node.\n"
        "Collisions: "
        + "\n".join(
            f"  {name}: {ic.relative_to(INFRA_NODES_DIR.parent.parent.parent)} vs {mc}"
            for name, ic, mc in infra_collisions
        )
    )
