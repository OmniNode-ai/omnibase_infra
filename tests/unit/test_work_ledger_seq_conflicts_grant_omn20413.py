# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the runtime write grant on work_ledger_seq_conflicts (OMN-20413)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_FILES = sorted(
    [
        *(_ROOT / "src/omnibase_infra/topology/instances").glob("*.yaml"),
        *(_ROOT / "docker/catalog/database-topology").glob("*.yaml"),
    ]
)


@pytest.mark.parametrize(
    "path", _FILES, ids=[f"{p.parent.name}/{p.stem}" for p in _FILES]
)
def test_runtime_has_seq_conflicts_write_grant(path: Path) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    grants = document["databases"]["application"]["principals"]["omninode_runtime"][
        "grants"
    ]
    matching = [
        g
        for g in grants
        if g.get("object_type") == "TABLE"
        and g.get("schema") == "omninode_internal"
        and "work_ledger_seq_conflicts" in g.get("objects", [])
    ]
    assert len(matching) == 1, f"{path}: expected one seq_conflicts grant"
    assert set(matching[0]["privileges"]) == {"INSERT", "SELECT", "UPDATE"}
