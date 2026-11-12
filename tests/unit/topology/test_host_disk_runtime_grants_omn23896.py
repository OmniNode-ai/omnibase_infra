# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime role may read and write the two host disk tables on every profile (OMN-23896).

omnimarket's node_projection_host_disk keeps the newest disk reading per host in two tables and reads and
writes them as ``omninode_runtime``. The runtime's auto-wiring refuses to prepare a projection writer whose
principal holds no declared grant for a table its contract names, and the event chain gate of the omnimarket
pull request failed at collection with ``lacks declared read privileges: SELECT on table
omninode_internal.host_disk_class``. So every shipped profile must grant both tables, the vendored migrations
must be the node's own, and the control below shows the check reporting a profile that lacks the grant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.validation.omninode_internal_live_acl import declared_acl

pytestmark = pytest.mark.unit

_RUNTIME = "omninode_runtime"
_TABLES = ("host_disk_class", "host_disk_volume")
_VENDORED = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_host_disk"
)


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
def test_the_runtime_role_may_read_and_write_the_two_host_disk_tables(
    profile: str,
) -> None:
    for table in _TABLES:
        assert _declared(profile, table) == {"SELECT", "INSERT", "UPDATE"}, (
            profile,
            table,
        )


def test_control_removing_the_grant_is_reported_as_missing() -> None:
    """Removing the host_disk_class grant makes the runtime's read of it empty."""
    for profile in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        grants = declared_acl(load_topology_profile(profile)).table_grants
        assert _declared(profile, "host_disk_class"), profile
        broken = [entry for entry in grants if entry[1] not in _TABLES]
        assert _declared_from(broken, "host_disk_class") == set(), profile


def test_both_host_disk_migrations_are_vendored() -> None:
    assert sorted(path.name for path in _VENDORED.glob("*.sql")) == [
        "0000_create_host_disk.sql",
        "0001_grant_omninode_runtime_host_disk.sql",
    ]
