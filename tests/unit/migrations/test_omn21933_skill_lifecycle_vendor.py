# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21933 vendor identity and grants for the skill lifecycle projection migrations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_skill_lifecycle"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_INSTANCES = _ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_CATALOGS = _ROOT / "docker" / "catalog" / "database-topology"
_CREATE = "0001_create_skill_executions.sql"
_GRANT = "0002_grant_tenant_projection_writer_skill_executions.sql"
_TABLE = "skill_executions"


def _writer_tables(topology_file: Path) -> set[str]:
    topology = yaml.safe_load(topology_file.read_text(encoding="utf-8"))
    grants = topology["databases"]["application"]["principals"][
        "tenant_projection_writer"
    ]["grants"]
    return {
        table
        for grant in grants
        if grant["object_type"] == "TABLE"
        and set(grant["privileges"]) == {"INSERT", "SELECT", "UPDATE"}
        for table in grant["objects"]
    }


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
def test_vendor_manifest_binding_matches_the_bytes(filename: str) -> None:
    artifact_path = f"nodes/{_NODE}/{filename}"
    digest = hashlib.sha256((_VENDOR / filename).read_bytes()).hexdigest()
    rows = {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }
    assert rows[artifact_path] == [
        artifact_path,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "tenant",
        f"node:{_NODE}:{filename}",
        digest,
    ]


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
def test_migration_classes_are_declared(filename: str) -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{filename}"] == "expand-only"


def test_grant_migration_names_the_role_and_the_table() -> None:
    sql = (_VENDOR / _GRANT).read_text(encoding="utf-8")
    assert "GRANT SELECT, INSERT, UPDATE" in sql
    assert f"public.{_TABLE}" in sql
    assert "TO tenant_projection_writer" in sql


@pytest.mark.parametrize("instance", sorted(p.stem for p in _INSTANCES.glob("*.yaml")))
def test_every_topology_instance_grants_the_writer_the_table(instance: str) -> None:
    assert _TABLE in _writer_tables(_INSTANCES / f"{instance}.yaml")


@pytest.mark.parametrize("catalog", sorted(p.stem for p in _CATALOGS.glob("*.yaml")))
def test_every_rendered_catalog_grants_the_writer_the_table(catalog: str) -> None:
    assert _TABLE in _writer_tables(_CATALOGS / f"{catalog}.yaml")
