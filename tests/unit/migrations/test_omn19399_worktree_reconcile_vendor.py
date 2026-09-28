# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19399 vendor identity for the worktree reconcile projection migrations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_worktree_reconcile"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_worktree_reconcile_hosts.sql"
_GRANT = "0001_grant_runtime_worktree_reconcile.sql"
_SHA256 = {
    _CREATE: "c2f8e9cc08e41a211d0dd5a6fa3b6bc0f5a598d8acbeea8b1cdadbba275c1e00",
    _GRANT: "15441b82c734228c3f0b2be6935e429c8cfff8fe840993f43534a8914de5e88a",
}
_TABLES = ("worktree_reconcile_hosts",)


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in (_VENDOR / filename).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
def test_vendor_bytes_and_manifest_binding_are_exact(filename: str) -> None:
    artifact_path = f"nodes/{_NODE}/{filename}"
    assert (
        hashlib.sha256((_VENDOR / filename).read_bytes()).hexdigest()
        == _SHA256[filename]
    )
    rows = {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }
    assert rows[artifact_path] == [
        artifact_path,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "omninode_internal",
        f"node:{_NODE}:{filename}",
        _SHA256[filename],
    ]


def test_migration_classes_match_the_classifier() -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_CREATE}"] == "expand-only"
    assert classes[f"forward/nodes/{_NODE}/{_GRANT}"] == "expand-only"


def test_table_and_exact_runtime_grants_are_present() -> None:
    create = _statements(_CREATE)
    grants = _statements(_GRANT)
    for table in _TABLES:
        assert re.search(
            rf"CREATE TABLE IF NOT EXISTS omninode_internal\.{table}\s*\(", create
        )
        assert re.search(
            rf"GRANT\s+SELECT,\s*INSERT,\s*UPDATE\s+"
            rf"ON omninode_internal\.{table}\s+TO omninode_runtime;",
            grants,
        )
    assert "GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;" in grants
    for sql in (create, grants):
        assert not re.search(
            r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+omninode_runtime\b",
            sql,
            re.IGNORECASE,
        )


# The host-keyed projection needs table read/write and schema lookup only.
# No generated cursor or sequence privilege is needed.
_EXPECTED_GRANTS = frozenset(
    {
        ("USAGE", "SCHEMA omninode_internal"),
        *(
            (privilege, f"omninode_internal.{table}")
            for table in _TABLES
            for privilege in ("SELECT", "INSERT", "UPDATE")
        ),
    }
)


def _grants(sql: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for statement in re.findall(
        r"^\s*GRANT\b[^;]*;", sql, re.IGNORECASE | re.MULTILINE
    ):
        match = re.fullmatch(
            r"GRANT\s+(?P<privs>[A-Z,\s]+?)\s+ON\s+(?P<target>.+?)\s+"
            r"TO\s+omninode_runtime\s*;",
            " ".join(statement.split()),
        )
        assert match, f"unparsed or non-omninode_runtime GRANT: {statement!r}"
        for privilege in match["privs"].split(","):
            found.add((privilege.strip(), match["target"]))
    return found


def test_the_grants_are_exactly_the_writer_scope() -> None:
    """Every GRANT in both files, parsed, equals the writer's scope and no more.

    No DELETE, TRUNCATE, REFERENCES, TRIGGER, CREATE or ALL; no schema-wide
    ``ON ALL TABLES IN SCHEMA``; no ``WITH GRANT OPTION``; no other grantee.
    A broader grant added later fails here by name.
    """
    assert _grants(_statements(_CREATE)) == set()
    assert _grants(_statements(_GRANT)) == _EXPECTED_GRANTS
    assert "WITH GRANT OPTION" not in _statements(_GRANT).upper()


def test_runtime_table_grant_assertion_is_present() -> None:
    grants = _statements(_GRANT)
    assert "SELECT 1 / count(*) AS worktree_reconcile_grants_assertion" in grants
    assert "table_schema = 'omninode_internal'" in grants
    assert "table_name = 'worktree_reconcile_hosts'" in grants
    assert "grantee = 'omninode_runtime'" in grants
    assert "privilege_type IN ('SELECT', 'INSERT', 'UPDATE')" in grants
    assert "HAVING count(DISTINCT privilege_type) = 3" in grants


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
@pytest.mark.parametrize(
    "profile", ["local", "onex-dev", "onex-prod", "stability-test"]
)
def test_migrations_pass_the_application_database_sql_gate(
    filename: str, profile: str
) -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / filename).read_text(encoding="utf-8")
    violations = lint_application_database_sql(sql, load_topology_profile(profile))
    assert violations == (), f"{profile}/{filename}: {violations}"


def test_the_sql_gate_is_live_positive_control() -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / _CREATE).read_text(encoding="utf-8")
    broken = sql.replace(
        "omninode_internal.worktree_reconcile_hosts", "tenant.worktree_reconcile_hosts"
    )
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()


@pytest.mark.parametrize("profile", ["local", "onex-dev", "onex-prod"])
def test_bridge_derives_the_shipped_runtime_grant(profile: str) -> None:
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
        if entry.table.name == "worktree_reconcile_hosts"
    )
    assert len(bridges) == 1
    bridge = bridges[0]
    assert bridge.table.access == "read_write"
    assert bridge.table.schema == "omninode_internal"
    assert bridge.table.database_ref == "application"
    assert bridge.contract_path == (_VENDOR / _CREATE).relative_to(_ROOT)
    derived = derive_table_grants(load_topology_profile(profile), bridges)
    matching = tuple(
        grant
        for grant in derived.grants["omninode_runtime"]
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and grant.schema == "omninode_internal"
        and "worktree_reconcile_hosts" in grant.objects
    )
    assert len(matching) == 1
    assert {privilege.value for privilege in matching[0].privileges} == {
        "SELECT",
        "INSERT",
        "UPDATE",
    }
    instance = yaml.safe_load(
        (
            _ROOT / "src/omnibase_infra/topology/instances" / f"{profile}.yaml"
        ).read_text()
    )
    shipped = [
        grant
        for grant in instance["databases"]["application"]["principals"][
            "omninode_runtime"
        ]["grants"]
        if grant["object_type"] == "TABLE"
        and grant["schema"] == "omninode_internal"
        and "worktree_reconcile_hosts" in grant["objects"]
    ]
    assert len(shipped) == 1
    assert set(shipped[0]["privileges"]) == {"SELECT", "INSERT", "UPDATE"}
