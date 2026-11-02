# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21933 vendor identity and grants for the skill lifecycle projection migrations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    derive_table_grants,
)

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
    assert classes[f"forward/nodes/{_NODE}/{filename}"] == "forward-only"


def test_supplemental_declaration_derives_the_writer_privileges() -> None:
    declarations = tuple(
        declaration
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if declaration.table.name == _TABLE
    )
    assert len(declarations) == 1
    for instance in sorted(path.stem for path in _INSTANCES.glob("*.yaml")):
        derived = derive_table_grants(load_topology_profile(instance), declarations)
        assert not derived.unmappable
        assert set(derived.grants) == {"tenant_projection_writer"}
        (grant,) = derived.grants["tenant_projection_writer"]
        assert grant.schema == "public"
        assert grant.objects == (_TABLE,)
        assert {privilege.value for privilege in grant.privileges} == {
            "SELECT",
            "INSERT",
            "UPDATE",
        }


@pytest.mark.parametrize("instance", sorted(p.stem for p in _INSTANCES.glob("*.yaml")))
def test_every_topology_instance_grants_the_writer_the_table(instance: str) -> None:
    assert _TABLE in _writer_tables(_INSTANCES / f"{instance}.yaml")


@pytest.mark.parametrize("catalog", sorted(p.stem for p in _CATALOGS.glob("*.yaml")))
def test_every_rendered_catalog_grants_the_writer_the_table(catalog: str) -> None:
    assert _TABLE in _writer_tables(_CATALOGS / f"{catalog}.yaml")
