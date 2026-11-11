# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""omninode_runtime holds write on omninode_internal.work_ledger_latest (OMN-23705).

The omnimarket node_projection_work_ledger declares ``work_ledger_latest``
in ``db_io.db_tables`` with ``access: write``. The wiring validator requires
SELECT, INSERT and UPDATE for the runtime principal on every shipped topology
instance, and the vendored 0012 migration grants the same set and never DELETE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_INSTANCE_DIR = _ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_NODE_MIGRATION_DIR = (
    _ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_work_ledger"
)
_INSTANCES = tuple(sorted(path.stem for path in _INSTANCE_DIR.glob("*.yaml")))
_TABLE = "work_ledger_latest"
_WRITE = {"INSERT", "SELECT", "UPDATE"}


def _runtime_table_privileges(document: dict[str, Any]) -> set[str]:
    privileges: set[str] = set()
    for database in document["databases"].values():
        principal = database.get("principals", {}).get("omninode_runtime")
        if principal is None:
            continue
        for grant in principal["grants"]:
            if (
                grant["object_type"] == "TABLE"
                and grant.get("schema") == "omninode_internal"
                and _TABLE in grant.get("objects", [])
            ):
                privileges.update(grant["privileges"])
    return privileges


@pytest.mark.unit
def test_instances_are_found() -> None:
    assert _INSTANCES


@pytest.mark.unit
@pytest.mark.parametrize("instance", _INSTANCES)
def test_runtime_can_write_work_ledger_latest(instance: str) -> None:
    document = yaml.safe_load(
        (_INSTANCE_DIR / f"{instance}.yaml").read_text(encoding="utf-8")
    )
    assert _runtime_table_privileges(document) == _WRITE


@pytest.mark.unit
def test_vendored_grant_migration_matches_the_declared_set_and_omits_delete() -> None:
    grant = (
        _NODE_MIGRATION_DIR / "0012_grant_omninode_runtime_work_ledger_latest.sql"
    ).read_text(encoding="utf-8")
    granted = grant.split("GRANT SELECT", 1)[1].split("TO", 1)[0]
    assert "INSERT, UPDATE" in granted
    assert "DELETE" not in granted
    assert "omninode_internal.work_ledger_latest" in granted


@pytest.mark.unit
def test_vendored_create_migration_keys_the_table_by_subject_and_row_type() -> None:
    sql = (_NODE_MIGRATION_DIR / "0011_work_ledger_latest.sql").read_text(
        encoding="utf-8"
    )
    assert "CREATE TABLE IF NOT EXISTS omninode_internal.work_ledger_latest" in sql
    assert "PRIMARY KEY (subject_kind, subject, row_type)" in sql
    assert sql.index("CREATE TABLE") < sql.index("INSERT INTO")
