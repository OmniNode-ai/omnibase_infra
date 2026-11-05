# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the interim vendored compaction A/B migrations and their topology grants.

omnimarket's node_projection_compaction_ab (OMN-21994) declares two
omninode_internal relations with access read_write, and its Event Chain Gate
loads the contract against this repo's shipped topology. Until the node's
contract is in the pinned omnimarket release, the two SQL files are held here:
the digests pin the byte-for-byte copy of the reviewed bytes, and the
``omninode_runtime`` table grants are derived from the interim declarations, then
proved through the same resolver the auto-wiring gate calls.

The local 0002 repair adds the missing legacy constraints and UPDATE grant;
0000/0001 retain their original digests and migration identities. The live-apply
proof in tests/integration/migrations/test_compaction_ab_vendor_omn21994.py
executes the chain on fresh and legacy tables, including invalid-data refusal.
"""

import hashlib
import re
from pathlib import Path

import pytest
import yaml

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_LEDGER = _FORWARD / "_ledger/application-migrations.tsv"
_NODE = "node_projection_compaction_ab"
_CREATE = "0000_create_compaction_ab.sql"
_GRANT = "0001_grant_omninode_runtime_compaction_ab.sql"
_REPAIR = "0002_repair_compaction_ab_constraints_and_grants.sql"
_VENDORED = [
    (_CREATE, "934d8b2aff324f0be7ab2836fab8547769dd1f3bea451220a27a5746a7b3e97e"),
    (_GRANT, "3bc2e87ba848872098ef2f4b358943a7dd788201482ddc431041ba0cbaeade15"),
]
_MIGRATIONS = [
    *_VENDORED,
    (_REPAIR, "094eef928d8e366b72f9d8bc517136bca737639c9a0df2ef63d25cd1d28a8ec8"),
]
_CLASSES = {_CREATE: "forward-only", _GRANT: "expand-only", _REPAIR: "forward-only"}
_TABLES = ("compaction_ab_rows", "compaction_ab_build_series")


def test_vendor_history_and_local_repair_are_the_complete_chain() -> None:
    names = sorted(path.name for path in (_FORWARD / "nodes" / _NODE).glob("*"))
    assert names == [_CREATE, _GRANT, _REPAIR]


@pytest.mark.parametrize(("name", "sha256"), _MIGRATIONS)
def test_vendor_bytes_and_manifest_binding_are_exact(name: str, sha256: str) -> None:
    artifact = f"nodes/{_NODE}/{name}"
    assert hashlib.sha256((_FORWARD / artifact).read_bytes()).hexdigest() == sha256
    rows = [line.split("\t") for line in _LEDGER.read_text().splitlines()]
    assert [row for row in rows if row[0] == artifact] == [
        [
            artifact,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "omninode_internal",
            f"node:{_NODE}:{name}",
            sha256,
        ]
    ]


@pytest.mark.parametrize(("name", "sha256"), _MIGRATIONS)
def test_the_vendor_is_declared_with_its_class(name: str, sha256: str) -> None:
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/nodes/{_NODE}/{name}"] == _CLASSES[name]


def test_no_migration_in_the_chain_is_fenced() -> None:
    fenced = yaml.safe_load((_FORWARD / "fenced-node-migrations.yaml").read_text())
    ids = {entry["id"] for entry in fenced["fenced_node_migrations"]}
    for name, _ in _MIGRATIONS:
        assert f"node:{_NODE}:{name}" not in ids


@pytest.mark.parametrize("table_name", _TABLES)
def test_the_interim_declaration_names_the_vendored_create(table_name: str) -> None:
    (declaration,) = [
        declaration
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if declaration.table.name == table_name
    ]
    table = declaration.table
    expected = _resolver_input(table_name)
    assert table.model_dump(exclude={"migration"}) == expected.model_dump(
        exclude={"migration"}
    )
    assert table.migration == f"docker/migrations/forward/nodes/{_NODE}/{_CREATE}"


def _resolver_input(table_name: str) -> ModelDbTableDeclaration:
    """Resolver input; migration is node-relative, unlike the vendored bridge.

    This exercises admission privileges, not discovery of an upstream contract
    or resolution of its migration path. The bridge path is checked separately.
    """
    return {
        "compaction_ab_rows": ModelDbTableDeclaration(
            name="compaction_ab_rows",
            database_ref="application",
            schema="omninode_internal",
            migration="0000_create_compaction_ab.sql",
            access="read_write",
            role="compaction_ab_rows",
        ),
        "compaction_ab_build_series": ModelDbTableDeclaration(
            name="compaction_ab_build_series",
            database_ref="application",
            schema="omninode_internal",
            migration="0000_create_compaction_ab.sql",
            access="read_write",
            role="compaction_ab_build_series",
        ),
    }[table_name]


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
@pytest.mark.parametrize("table_name", _TABLES)
def test_runtime_principal_resolves_the_expected_table(
    profile: str, table_name: str
) -> None:
    """The gate's own check, on the shipped topology: read and write succeed."""
    target = _resolve_projection_database_target(
        [_resolver_input(table_name)], load_topology_profile(profile)
    )
    (resolved,) = target.table_targets
    assert resolved.table.name == table_name
    for binding in (resolved.read_binding, resolved.write_binding):
        assert binding is not None
        assert binding.principal == "omninode_runtime"


def _sql_privileges(sql: str, table_name: str) -> set[str]:
    """Read the explicit table grants in this node's small migration chain."""
    statements = re.sub(r"--[^\n]*", "", sql)
    privileges: set[str] = set()
    for match in re.finditer(
        rf"GRANT\s+([A-Z,\s]+)\s+ON\s+(?:TABLE\s+)?"
        rf"omninode_internal\.{table_name}\s+TO\s+omninode_runtime\s*;",
        statements,
        re.IGNORECASE,
    ):
        privileges.update(value.strip().upper() for value in match[1].split(","))
    return privileges


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
@pytest.mark.parametrize("table_name", _TABLES)
def test_sql_grants_deliver_every_topology_privilege(
    profile: str, table_name: str
) -> None:
    topology = load_topology_profile(profile)
    principal = topology.databases["application"].principals["omninode_runtime"]
    expected = {
        privilege.value
        for grant in principal.grants
        if grant.schema == "omninode_internal" and table_name in grant.objects
        for privilege in grant.privileges
    }
    sql = "\n".join(
        path.read_text() for path in sorted((_FORWARD / "nodes" / _NODE).glob("*.sql"))
    )
    assert expected == {"SELECT", "INSERT", "UPDATE"}
    assert _sql_privileges(sql, table_name) == expected


def test_sql_privilege_reader_does_not_invent_missing_update() -> None:
    sql = (_FORWARD / "nodes" / _NODE / _GRANT).read_text()
    assert _sql_privileges(sql, "compaction_ab_rows") == {"SELECT", "INSERT", "UPDATE"}
    assert _sql_privileges(sql, "compaction_ab_build_series") == {"SELECT", "INSERT"}


@pytest.mark.parametrize("table_name", _TABLES)
def test_repair_restores_the_create_time_constraints(table_name: str) -> None:
    create = (_FORWARD / "nodes" / _NODE / _CREATE).read_text()
    repair = (_FORWARD / "nodes" / _NODE / _REPAIR).read_text()
    match = re.search(
        rf"CREATE TABLE IF NOT EXISTS omninode_internal\.{table_name} \((.*?)\n\);",
        create,
        re.DOTALL,
    )
    assert match is not None
    body = match[1]
    not_null = set(re.findall(r"^\s*(\w+)\s+\w+\s+NOT NULL", body, re.MULTILINE))
    assert not_null
    alter = re.search(
        rf"ALTER TABLE omninode_internal\.{table_name}\s+(.*?);", repair, re.DOTALL
    )
    assert alter is not None
    assert set(re.findall(r"ALTER COLUMN (\w+) SET NOT NULL", alter[1])) == not_null
    primary_key = re.search(r"PRIMARY KEY \([^)]+\)", body)
    assert primary_key is not None
    assert f"ADD {primary_key[0]}" in repair
    for constraint in re.findall(r"CHECK \([^)]+\)", body):
        assert constraint in repair
