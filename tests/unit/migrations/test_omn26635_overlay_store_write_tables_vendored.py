# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26635: the overlay store revision, provenance and audit migrations, vendored and bound.

omnimarket's node_projection_overlay_store adds the revision column and the
overlay_store_revision, _write, _provenance and _audit tables in 0003, and the
runtime grant on them in 0004. These are the vendored copies. Failure modes:

- a vendored file is absent or drifts from the omnimarket source (checksum);
- a file has no ledger row, or a row naming another checksum or domain;
- a file is not declared in config/migration_classes.yaml;
- the runtime principal is not granted the four tables in the database topology,
  so omnimarket's projection-binding check refuses to load its runtime.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_overlay_store"
_TABLES = (
    "overlay_store_audit",
    "overlay_store_provenance",
    "overlay_store_revision",
    "overlay_store_write",
)
# name -> checksum of the omnimarket source the file is copied from.
_SOURCES = {
    "0003_overlay_store_revision_provenance_audit.sql": "eeebfc5609cdfb899c879077705cf5efe62dae64a59fcae857440faac69962e4",
    "0004_grant_omninode_runtime_overlay_store_write.sql": "d8a04194aa51bcd8fff39eee79ff4191b970812183b269df97a07b864018f78c",
}


@pytest.mark.parametrize(("name", "sha256"), sorted(_SOURCES.items()))
def test_bytes_and_manifest_binding_are_exact(name: str, sha256: str) -> None:
    relative = f"nodes/{_NODE}/{name}"
    assert hashlib.sha256((_FORWARD / relative).read_bytes()).hexdigest() == sha256
    rows = [
        row.split("\t")
        for row in (_FORWARD / "_ledger" / "application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if row.split("\t", 1)[0] == relative
    ]
    assert rows == [
        [
            relative,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "omninode_internal",
            f"node:{_NODE}:{name}",
            sha256,
        ]
    ]


@pytest.mark.parametrize("name", sorted(_SOURCES))
def test_is_a_declared_migration_class(name: str) -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{name}"] in {"expand-only", "forward-only"}


def test_0004_grants_the_runtime_the_four_tables_and_no_delete() -> None:
    sql = re.sub(
        r"--[^\n]*",
        "",
        (
            _FORWARD
            / "nodes"
            / _NODE
            / "0004_grant_omninode_runtime_overlay_store_write.sql"
        ).read_text(encoding="utf-8"),
    )
    assert "GRANT SELECT, INSERT, UPDATE" in sql
    assert "DELETE" not in sql
    for table in _TABLES:
        assert f"omninode_internal.{table}" in sql


@pytest.mark.parametrize(
    "path",
    sorted(
        [
            *(_ROOT / "docker" / "catalog" / "database-topology").glob("*.yaml"),
            *(_ROOT / "src" / "omnibase_infra" / "topology" / "instances").glob(
                "*.yaml"
            ),
        ]
    ),
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_topology_lists_the_four_tables_beside_overlay_store(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if not re.search(r"^\s*- overlay_store$", text, re.MULTILINE):
        pytest.skip("topology does not list overlay_store")
    for table in _TABLES:
        assert re.search(rf"^\s*- {table}$", text, re.MULTILINE), table
