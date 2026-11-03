# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22320: pin the work-questions migration bytes, ledger binding and table set."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_work_questions"
_VENDOR = _FORWARD / "nodes" / _NODE
_CREATE = "0000_create_work_questions.sql"
_SHA256 = "c821731806d2d088cb9755b38b68cafdf9127ce1902d96b9d77c5aa0317a71db"
_TABLES = (
    "work_outcome_units",
    "orchestration_style_outcomes",
    "lane_drift_and_stalls",
    "brief_quality_vs_outcome",
    "subagent_waste",
)


def _sql() -> str:
    text = (_VENDOR / _CREATE).read_text(encoding="utf-8")
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("--")
    )


def test_create_and_acl_successor_are_vendored() -> None:
    assert sorted(path.name for path in _VENDOR.glob("*")) == [
        _CREATE,
        "0001_revoke_undeclared_runtime_grants.sql",
    ]


def test_vendor_bytes_and_manifest_binding_are_exact() -> None:
    path = _VENDOR / _CREATE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _SHA256
    artifact = f"nodes/{_NODE}/{_CREATE}"
    rows = [
        row.split("\t")
        for row in (_FORWARD / "_ledger" / "application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if row.split("\t", 1)[0] == artifact
    ]
    assert rows == [
        [
            artifact,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "omninode_internal",
            f"node:{_NODE}:{_CREATE}",
            _SHA256,
        ]
    ]


def test_acl_successor_is_checksum_bound_and_scoped_to_the_five_tables() -> None:
    filename = "0001_revoke_undeclared_runtime_grants.sql"
    path = _VENDOR / filename
    sql = path.read_text(encoding="utf-8")
    sql = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    revoke = re.search(r"REVOKE ALL PRIVILEGES ON\s+(.*?)\s+FROM\s+(\w+);", sql, re.S)
    assert revoke is not None
    assert revoke.group(2) == "omninode_runtime"
    assert {table.strip() for table in revoke.group(1).split(",")} == {
        f"omninode_internal.{table}" for table in _TABLES
    }
    assert "aclexplode(c.relacl)" in sql
    assert "1 / (count(*) = 0)::int" in sql
    artifact = f"nodes/{_NODE}/{filename}"
    rows = [
        row.split("\t")
        for row in (_FORWARD / "_ledger/application-migrations.tsv")
        .read_text()
        .splitlines()
        if row.split("\t", 1)[0] == artifact
    ]
    assert rows == [
        [
            artifact,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "omninode_internal",
            f"node:{_NODE}:{filename}",
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ]
    ]
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/{artifact}"] == "forward-only"


def test_migration_class_is_forward_only() -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_CREATE}"] == "forward-only"


def test_creates_exactly_the_five_tables() -> None:
    created = re.findall(r"CREATE TABLE IF NOT EXISTS omninode_internal\.(\w+)", _sql())
    assert created == list(_TABLES)
    assert len(re.findall(r"\bCREATE TABLE\b", _sql())) == len(_TABLES)


def test_touches_only_its_own_tables_and_never_removes_anything() -> None:
    sql = _sql()
    assert not re.search(r"\b(DROP|DELETE|TRUNCATE|GRANT|REVOKE)\b", sql, re.I)
    altered = set(re.findall(r"ALTER TABLE (\S+)", sql))
    assert altered == {f"omninode_internal.{table}" for table in _TABLES}
    assert not re.search(r"ALTER TABLE \S+\s+(ALTER|DROP|RENAME)\b", sql, re.I)


@pytest.mark.parametrize("table", _TABLES)
def test_every_row_is_keyed_by_its_generation(table: str) -> None:
    block = re.search(
        rf"CREATE TABLE IF NOT EXISTS omninode_internal\.{table} \((.*?)\n\);",
        _sql(),
        re.S,
    )
    assert block is not None
    body = block.group(1)
    assert re.search(r"\bgeneration_id\s+TEXT NOT NULL", body)
    assert re.search(r"PRIMARY KEY \(generation_id, \w+\)", body)
    assert re.search(r"generation_source_versions\s+JSONB NOT NULL", body)
