# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime role may read and write the floor_integration_rows table on every profile (OMN-25103).

omnimarket's node_projection_floor_integration keeps the Running figure's projection (the newest integration pass) in
``omninode_internal.floor_integration_rows`` and reads and writes it as ``omninode_runtime``.
The runtime's auto-wiring refuses to prepare a projection writer whose principal
holds no declared grant for a table its contract names, and the event chain gate
of the omnimarket pull request failed at collection with ``principal
omninode_runtime lacks declared SELECT on omninode_internal.floor_integration_rows``. So
every shipped profile must grant the table, the vendored migrations must be the
node's own, and the control below shows the check reporting a profile that lacks
the grant.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.validation.omninode_internal_live_acl import declared_acl

pytestmark = pytest.mark.unit

_RUNTIME = "omninode_runtime"
_TABLES = ("floor_integration_rows",)
_VENDORED = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_floor_integration"
)
_LEDGER = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
    / "application-migrations.tsv"
)
_BRIDGES = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "topology"
    / "vendored_table_bridges.yaml"
)
_SUPPLEMENTAL_EXPIRY_TEST = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "ci"
    / "test_supplemental_declaration_expiry_omn18863.py"
)
_EXPECTED_SHAS: dict[str, str] = {
    "0000_create_floor_integration_rows.sql": (
        "64a874064994878cb487fe873ebe254eeea1a3228b90715b17f2d6c0e826793c"
    ),
    "0001_grant_omninode_runtime_floor_integration_rows.sql": (
        "47f8f7fef82ac3aaf815ce87f56df7f38c542578c6700ad04c4de0e726e81813"
    ),
}


def _declared(profile: str, table: str) -> set[str]:
    grants = declared_acl(load_topology_profile(profile)).table_grants
    return {
        privilege
        for principal, relation, privilege in grants
        if principal == _RUNTIME and relation == table
    }


def _declared_from(grants: list[tuple[str, str, str]], table: str) -> set[str]:
    return {
        privilege
        for principal, relation, privilege in grants
        if principal == _RUNTIME and relation == table
    }


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_the_runtime_role_may_read_and_write_the_floor_integration_rows_table(
    profile: str,
) -> None:
    """The writer upserts each pass and deletes the rows of an older one, so it needs DELETE."""
    for table in _TABLES:
        assert _declared(profile, table) == {"SELECT", "INSERT", "UPDATE", "DELETE"}, (
            profile,
            table,
        )


def _migration_table_privileges() -> set[str]:
    sql = (
        _VENDORED / "0001_grant_omninode_runtime_floor_integration_rows.sql"
    ).read_text(encoding="utf-8")
    statement = sql[sql.index("GRANT SELECT") :]
    granted = statement[len("GRANT ") : statement.index(" ON ")]
    return {privilege.strip() for privilege in granted.replace("\n", " ").split(",")}


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_declared_privileges_equal_what_the_vendored_migration_grants(
    profile: str,
) -> None:
    """The live ACL gate fails a grant the topology does not declare (OMN-17886)."""
    assert _migration_table_privileges() == {"SELECT", "INSERT", "UPDATE", "DELETE"}
    assert _declared(profile, "floor_integration_rows") == _migration_table_privileges()


def test_control_removing_the_grant_is_reported_as_missing() -> None:
    """Removing the floor_integration_rows grant makes the runtime's read of it empty."""
    for profile in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        grants = declared_acl(load_topology_profile(profile)).table_grants
        assert _declared(profile, "floor_integration_rows"), profile
        broken = [entry for entry in grants if entry[1] not in _TABLES]
        assert _declared_from(broken, "floor_integration_rows") == set(), profile


def test_every_floor_integration_migration_is_vendored() -> None:
    """The rows table (0000-0001) and the held-pass marker (0002-0003, AC7)."""
    assert sorted(path.name for path in _VENDORED.glob("*.sql")) == [
        "0000_create_floor_integration_rows.sql",
        "0001_grant_omninode_runtime_floor_integration_rows.sql",
        "0002_create_floor_integration_pass.sql",
        "0003_grant_omninode_runtime_floor_integration_pass.sql",
    ]


def test_vendored_floor_integration_rows_migrations_match_omnimarket_and_ledger() -> (
    None
):
    ledger_rows: list[list[str]] = [
        line.split("\t")
        for line in _LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for filename, expected_sha in _EXPECTED_SHAS.items():
        path = _VENDORED / filename
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, (filename, actual_sha)

        relative_path = f"nodes/node_projection_floor_integration/{filename}"
        matching_rows = [row for row in ledger_rows if row and row[0] == relative_path]
        assert len(matching_rows) == 1, (filename, matching_rows)
        assert matching_rows[0][5] == expected_sha, (filename, matching_rows[0])


def test_vendored_table_bridge_entry_exists() -> None:
    raw: object = yaml.safe_load(_BRIDGES.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = (
        cast("list[dict[str, Any]]", raw) if isinstance(raw, list) else []
    )

    assert any(
        entry.get("node") == "node_projection_floor_integration"
        and entry.get("create") == "0000_create_floor_integration_rows.sql"
        and entry.get("prefix") == ""
        and entry.get("tables") == ["floor_integration_rows"]
        for entry in entries
    ), entries


def test_supplemental_declaration_expiry_lists_floor_integration_rows() -> None:
    tree = ast.parse(_SUPPLEMENTAL_EXPIRY_TEST.read_text(encoding="utf-8"))
    interim_keys: set[str] = set()

    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value

        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "_INTERIM_ENTRIES"
                and isinstance(value, ast.Dict)
            ):
                for key in value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        interim_keys.add(key.value)

    assert "floor_integration_rows" in interim_keys
