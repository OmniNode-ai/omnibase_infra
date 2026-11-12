# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime role may write the four merge override tables on every profile (OMN-23858).

omnimarket's node_merge_override_alarm_effect store reads and writes its four tables
as ``omninode_runtime``. The runtime's auto-wiring refuses
to prepare a projection writer whose principal holds no declared grant for a
table its contract names, and the dev lane's strict wiring turns that refusal
into a boot crash. So every shipped profile must grant all four tables, and the
vendored migrations must be the node's own bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.validation.omninode_internal_live_acl import declared_acl

pytestmark = pytest.mark.unit

_RUNTIME = "omninode_runtime"
_TABLES = (
    "merge_override_open_findings",
    "merge_override_pending",
    "merge_override_receipts",
    "merge_override_verdicts",
)
_VENDORED = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_merge_override_alarm_effect"
)


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_the_runtime_role_may_read_and_write_the_four_merge_override_tables(
    profile: str,
) -> None:
    grants = declared_acl(load_topology_profile(profile)).table_grants
    for table in _TABLES:
        declared = {
            privilege
            for principal, relation, privilege in grants
            if principal == _RUNTIME and relation == table
        }
        assert declared == {"SELECT", "INSERT", "UPDATE"}, (profile, table)


def test_both_merge_override_migrations_are_vendored() -> None:
    assert sorted(path.name for path in _VENDORED.glob("*.sql")) == [
        "0001_create_merge_override_tables.sql",
        "0002_grant_omninode_runtime_merge_override_tables.sql",
    ]
