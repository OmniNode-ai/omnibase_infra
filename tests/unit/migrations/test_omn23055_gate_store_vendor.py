# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23055: pin the gate store migration bytes, ledger binding, classes and table grants."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_gate_store_effect"
_VENDOR = _FORWARD / "nodes" / _NODE
_CREATE = "0000_create_gate_store.sql"
_CREATE_SHA256 = "a6d92c816e31b25b6d18956a71c545ffe86748578c1595e78ef515577bb58e3e"
_GRANT = "0001_grant_omninode_runtime_gate_store.sql"
_GRANT_SHA256 = "aa2baabf49d9e11b069aaeaa8169ec992626a1c36070e458b50f752e23dfd276"
_TABLES = (
    "gate_store_reservations",
    "gate_store_safety",
    "gate_store_state",
    "gate_store_steps",
)
_INSTANCES = sorted((_ROOT / "src/omnibase_infra/topology/instances").glob("*.yaml"))
_CATALOGS = sorted((_ROOT / "docker/catalog/database-topology").glob("*.yaml"))


def test_only_the_create_and_grant_files_are_vendored() -> None:
    assert sorted(path.name for path in _VENDOR.glob("*")) == [_CREATE, _GRANT]


@pytest.mark.parametrize(
    ("name", "sha256"), [(_CREATE, _CREATE_SHA256), (_GRANT, _GRANT_SHA256)]
)
def test_vendor_bytes_and_manifest_binding_are_exact(name: str, sha256: str) -> None:
    path = _VENDOR / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256
    artifact = f"nodes/{_NODE}/{name}"
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
            f"node:{_NODE}:{name}",
            sha256,
        ]
    ]


def test_the_create_file_makes_the_four_declared_relations() -> None:
    sql = (_VENDOR / _CREATE).read_text(encoding="utf-8")
    for table in _TABLES:
        assert f"CREATE TABLE IF NOT EXISTS omninode_internal.{table}" in sql


def test_migration_classes_name_both_files() -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_CREATE}"] == "forward-only"
    assert classes[f"forward/nodes/{_NODE}/{_GRANT}"] == "expand-only"


def _runtime_granted_tables(path: Path) -> set[str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    granted: set[str] = set()
    for principal in document["databases"]["application"]["principals"].values():
        for grant in principal.get("grants", []):
            if (
                grant.get("object_type") == "TABLE"
                and grant.get("schema") == "omninode_internal"
                and {"SELECT", "INSERT", "UPDATE"} <= set(grant.get("privileges", []))
            ):
                granted.update(grant.get("objects", []))
    return granted


@pytest.mark.parametrize("path", _INSTANCES, ids=[path.stem for path in _INSTANCES])
def test_gate_store_instance_grants(path: Path) -> None:
    assert set(_TABLES) <= _runtime_granted_tables(path)


@pytest.mark.parametrize("path", _CATALOGS, ids=[path.stem for path in _CATALOGS])
def test_gate_store_catalog_grants(path: Path) -> None:
    assert set(_TABLES) <= _runtime_granted_tables(path)
