# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime role may read and write the floor_integration_pass marker on every profile (OMN-25103 AC7).

omnimarket's node_projection_floor_integration keeps the newest applied
integration pass in ``omninode_internal.floor_integration_pass`` (one row), so an
empty pass is still remembered and a late older pass changes nothing. The writer
reads and upserts that row as ``omninode_runtime`` in the same transaction as the
rows. The runtime's auto-wiring refuses a projection writer whose principal holds
no declared grant for a table its contract names, and omnimarket#6420's event
chain gate fails with ``principal omninode_runtime lacks declared SELECT on
omninode_internal.floor_integration_pass``. So every shipped profile must grant
the table SELECT, INSERT and UPDATE (never DELETE: the marker is only upserted),
the vendored migrations must be the node's own, and the control shows the check
reporting a profile that lacks the grant.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.validation.omninode_internal_live_acl import declared_acl

pytestmark = pytest.mark.unit

_RUNTIME = "omninode_runtime"
_TABLE = "floor_integration_pass"
_ROOT = Path(__file__).resolve().parents[3]
_VENDORED = (
    _ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_floor_integration"
)
_LEDGER = (
    _ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
    / "application-migrations.tsv"
)
_BRIDGES = _ROOT / "src" / "omnibase_infra" / "topology" / "vendored_table_bridges.yaml"
_SUPPLEMENTAL_EXPIRY_TEST = (
    _ROOT / "tests" / "ci" / "test_supplemental_declaration_expiry_omn18863.py"
)
# sha256 of omnimarket's files at omnimarket#6420 b12c5b44f73327b0d8504e5925016866b7254f32.
_EXPECTED_SHAS: dict[str, str] = {
    "0002_create_floor_integration_pass.sql": (
        "b4433de1286a98682a0e822a2161ec4635f560bda279c1fba1da2252b7b885a4"
    ),
    "0003_grant_omninode_runtime_floor_integration_pass.sql": (
        "cba7e794f565468b29d01ee8d71df57b095ae59f7cc9245ef73a738fc6057dd6"
    ),
}


def _declared_from(grants: Iterable[tuple[str, str, str]]) -> set[str]:
    return {
        privilege
        for principal, relation, privilege in grants
        if principal == _RUNTIME and relation == _TABLE
    }


def _declared(profile: str) -> set[str]:
    return _declared_from(declared_acl(load_topology_profile(profile)).table_grants)


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_the_runtime_role_may_read_and_upsert_the_floor_integration_pass_marker(
    profile: str,
) -> None:
    """The writer reads the held pass and upserts it; it never deletes the marker."""
    assert _declared(profile) == {"SELECT", "INSERT", "UPDATE"}, profile


def _migration_table_privileges() -> set[str]:
    sql = (
        _VENDORED / "0003_grant_omninode_runtime_floor_integration_pass.sql"
    ).read_text(encoding="utf-8")
    statement = sql[sql.index("GRANT SELECT") :]
    granted = statement[len("GRANT ") : statement.index(" ON ")]
    return {privilege.strip() for privilege in granted.replace("\n", " ").split(",")}


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_declared_privileges_equal_what_the_vendored_marker_migration_grants(
    profile: str,
) -> None:
    """The live ACL gate fails a grant the topology does not declare (OMN-17886)."""
    assert _migration_table_privileges() == {"SELECT", "INSERT", "UPDATE"}
    assert _declared(profile) == _migration_table_privileges()


def _without_marker(value: object) -> object:
    """The topology with floor_integration_pass taken out of every grant's objects."""
    if isinstance(value, dict):
        return {key: _without_marker(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_without_marker(item) for item in value if item != _TABLE)
    return value


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_control_a_profile_without_the_marker_grant_is_reported_as_missing(
    profile: str,
) -> None:
    """The real ACL derivation, run on the profile with the grant removed, finds none.

    So the check above reads the shipped instance and would report a profile
    that lacks the grant, rather than passing whatever the profile says.
    """
    topology = load_topology_profile(profile)
    assert _declared(profile), profile
    stripped = type(topology).model_validate(
        _without_marker(topology.model_dump(mode="python"))
    )
    assert _declared_from(declared_acl(stripped).table_grants) == set(), profile


def test_vendored_marker_migrations_match_omnimarket_and_ledger() -> None:
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


def test_vendored_marker_bridge_entry_exists() -> None:
    raw: object = yaml.safe_load(_BRIDGES.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = (
        cast("list[dict[str, Any]]", raw) if isinstance(raw, list) else []
    )
    assert any(
        entry.get("node") == "node_projection_floor_integration"
        and entry.get("create") == "0002_create_floor_integration_pass.sql"
        and entry.get("prefix") == ""
        and entry.get("tables") == [_TABLE]
        for entry in entries
    ), entries


def test_supplemental_declaration_expiry_lists_the_marker() -> None:
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
    assert _TABLE in interim_keys
