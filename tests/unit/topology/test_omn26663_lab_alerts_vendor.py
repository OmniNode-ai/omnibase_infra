# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime role may read and write the lab_alerts table on every profile (OMN-26663).

omnimarket's node_projection_lab_alerts keeps open and cleared lab alerts in
``omninode_internal.lab_alerts`` and reads and writes it as ``omninode_runtime``.
``omninode_internal`` is a schema the runtime already holds USAGE on; the table
is the relation that carries no grant. The runtime's auto-wiring refuses to
prepare a projection writer whose principal holds no declared grant for a table
its contract names, so every shipped profile must grant the table, the vendored
migrations must be the node's own, and the controls below show the check
reporting a profile that lacks the grant.
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
_TABLES = ("lab_alerts",)
_VENDORED = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_lab_alerts"
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
    "0000_create_lab_alerts.sql": (
        "bfe9f7a055846905ef7e0f1d98fc421ff4fd03e1a98772d091bc7329fa4d5afd"
    ),
    "0001_grant_omninode_runtime_lab_alerts.sql": (
        "bf8b57593eeff04d13d1710eba4af8c8baea1ffddf94359ba6e6d1b65c1c7484"
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
def test_the_runtime_role_may_read_and_write_the_lab_alerts_table(
    profile: str,
) -> None:
    for table in _TABLES:
        assert _declared(profile, table) == {"SELECT", "INSERT", "UPDATE"}, (
            profile,
            table,
        )


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_the_runtime_role_cannot_delete_lab_alerts_rows(profile: str) -> None:
    """A cleared alert stays as a cleared row, so DELETE is never granted."""
    assert "DELETE" not in _declared(profile, "lab_alerts"), profile


def test_control_removing_the_grant_is_reported_as_missing() -> None:
    """Removing the lab_alerts grant makes the runtime's read of it empty."""
    for profile in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        grants = declared_acl(load_topology_profile(profile)).table_grants
        assert _declared(profile, "lab_alerts"), profile
        broken = [entry for entry in grants if entry[1] not in _TABLES]
        assert _declared_from(broken, "lab_alerts") == set(), profile


def test_both_lab_alerts_migrations_are_vendored() -> None:
    assert sorted(path.name for path in _VENDORED.glob("*.sql")) == [
        "0000_create_lab_alerts.sql",
        "0001_grant_omninode_runtime_lab_alerts.sql",
    ]


def test_vendored_lab_alerts_migrations_match_omnimarket_and_ledger() -> None:
    ledger_rows: list[list[str]] = [
        line.split("\t")
        for line in _LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for filename, expected_sha in _EXPECTED_SHAS.items():
        path = _VENDORED / filename
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, (filename, actual_sha)

        relative_path = f"nodes/node_projection_lab_alerts/{filename}"
        matching_rows = [row for row in ledger_rows if row and row[0] == relative_path]
        assert len(matching_rows) == 1, (filename, matching_rows)
        assert matching_rows[0][5] == expected_sha, (filename, matching_rows[0])


def test_vendored_table_bridge_entry_exists() -> None:
    raw: object = yaml.safe_load(_BRIDGES.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = (
        cast("list[dict[str, Any]]", raw) if isinstance(raw, list) else []
    )

    assert any(
        entry.get("node") == "node_projection_lab_alerts"
        and entry.get("create") == "0000_create_lab_alerts.sql"
        and entry.get("prefix") == ""
        and entry.get("tables") == ["lab_alerts"]
        for entry in entries
    ), entries


def test_supplemental_declaration_expiry_lists_lab_alerts() -> None:
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

    assert "lab_alerts" in interim_keys
