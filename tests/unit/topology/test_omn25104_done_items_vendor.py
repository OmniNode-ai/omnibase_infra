# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime role may read and write the done_items table on every profile (OMN-25104).

omnimarket's node_projection_done_items keeps the Done figure's projection in
``omninode_internal.done_items`` and reads and writes it as ``omninode_runtime``.
The runtime's auto-wiring refuses to prepare a projection writer whose principal
holds no declared grant for a table its contract names, and the event chain gate
of the omnimarket pull request failed at collection with ``principal
omninode_runtime lacks declared SELECT on omninode_internal.done_items``. So
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
_TABLES = ("done_items",)
_VENDORED = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_done_items"
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
    "0000_create_done_items.sql": (
        "30ad8fa64426d0ddc8c8eb661a9cb82cc18107da0dc173b9e5d86632ce7861f8"
    ),
    "0001_grant_omninode_runtime_done_items.sql": (
        "44282ba796aa43aa9e48389e1138e46bd416783348563c8011cf06663a71b3a7"
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
def test_the_runtime_role_may_read_and_write_the_done_items_table(
    profile: str,
) -> None:
    for table in _TABLES:
        assert _declared(profile, table) == {"SELECT", "INSERT", "UPDATE"}, (
            profile,
            table,
        )


def test_control_removing_the_grant_is_reported_as_missing() -> None:
    """Removing the done_items grant makes the runtime's read of it empty."""
    for profile in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        grants = declared_acl(load_topology_profile(profile)).table_grants
        assert _declared(profile, "done_items"), profile
        broken = [entry for entry in grants if entry[1] not in _TABLES]
        assert _declared_from(broken, "done_items") == set(), profile


def test_both_done_items_migrations_are_vendored() -> None:
    assert sorted(path.name for path in _VENDORED.glob("*.sql")) == [
        "0000_create_done_items.sql",
        "0001_grant_omninode_runtime_done_items.sql",
    ]


def test_vendored_done_items_migrations_match_omnimarket_and_ledger() -> None:
    ledger_rows: list[list[str]] = [
        line.split("\t")
        for line in _LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for filename, expected_sha in _EXPECTED_SHAS.items():
        path = _VENDORED / filename
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, (filename, actual_sha)

        relative_path = f"nodes/node_projection_done_items/{filename}"
        matching_rows = [row for row in ledger_rows if row and row[0] == relative_path]
        assert len(matching_rows) == 1, (filename, matching_rows)
        assert matching_rows[0][5] == expected_sha, (filename, matching_rows[0])
