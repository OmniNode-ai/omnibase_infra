# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19978 vendor identity for the usage-by-model-day node migrations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_usage_by_model_day"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_usage_by_model_day.sql"
_GRANT = "0001_grant_usage_by_model_day.sql"
_TABLES = ("usage_by_model_day_calls", "usage_by_model_day")
_SHA256 = {
    _CREATE: "0763b51e2d8fc794001d3ab1f9a39618a281fdf55c9acf89677a7e84ef0e9683",
    _GRANT: "2bf2ec13f194a2dfa0738cb57f65a765b730a1739416d40e7126678254fe979f",
}


def _sql(filename: str) -> str:
    return (_VENDOR / filename).read_text(encoding="utf-8")


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in _sql(filename).splitlines()
        if not line.lstrip().startswith("--")
    )


def _manifest_rows() -> dict[str, list[str]]:
    return {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
def test_vendor_bytes_and_manifest_binding_are_exact(filename: str) -> None:
    artifact_path = f"nodes/{_NODE}/{filename}"
    assert (
        hashlib.sha256((_VENDOR / filename).read_bytes()).hexdigest()
        == _SHA256[filename]
    )
    assert _manifest_rows()[artifact_path] == [
        artifact_path,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "tenant",
        f"node:{_NODE}:{filename}",
        _SHA256[filename],
    ]


def test_migration_classes_match_the_classifier() -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    for filename in (_CREATE, _GRANT):
        assert classes[f"forward/nodes/{_NODE}/{filename}"] == "forward-only"


def test_tenant_tables_and_writer_grants_are_exact() -> None:
    create = _statements(_CREATE)
    grant = _statements(_GRANT)
    for table in _TABLES:
        assert re.search(rf"CREATE TABLE IF NOT EXISTS public\.{table}\s*\(", create)
        assert f"public.{table}" in grant
    assert "GRANT USAGE ON SCHEMA public TO tenant_projection_writer;" in grant
    assert re.search(
        r"GRANT SELECT, INSERT, UPDATE\s+ON public\.usage_by_model_day_calls, "
        r"public\.usage_by_model_day\s+TO tenant_projection_writer;",
        grant,
    )
    assert not re.search(
        r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+tenant_projection_writer\b",
        grant,
        re.IGNORECASE,
    )


@pytest.mark.parametrize("profile", ["local", "onex-dev", "onex-prod"])
def test_bridge_derives_the_shipped_tenant_writer_grants(profile: str) -> None:
    from omnibase_core.enums.enum_database_grant_object_type import (
        EnumDatabaseGrantObjectType,
    )
    from omnibase_infra.topology import load_topology_profile
    from omnibase_infra.topology.table_grant_derivation import (
        LEGACY_MIGRATION_TABLE_DECLARATIONS,
        derive_table_grants,
    )

    bridges = tuple(
        entry
        for entry in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if entry.table.name in _TABLES
    )
    assert {entry.table.name for entry in bridges} == set(_TABLES)
    for bridge in bridges:
        assert bridge.node == f"legacy_migration:{bridge.table.name}"
        assert bridge.table.access == "read_write"
        assert bridge.table.schema == "public"
        assert bridge.table.database_ref == "application"
        assert bridge.contract_path == (_VENDOR / _CREATE).relative_to(_ROOT)

    derived = derive_table_grants(load_topology_profile(profile), bridges)
    matching = tuple(
        grant
        for grant in derived.grants["tenant_projection_writer"]
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and grant.schema == "public"
        and set(grant.objects) == set(_TABLES)
    )
    assert len(matching) == 1
    assert {privilege.value for privilege in matching[0].privileges} == {
        "SELECT",
        "INSERT",
        "UPDATE",
    }
