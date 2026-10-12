# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime role may write both lab job read models on every profile (OMN-20604).

omnimarket's node_projection_lab_job writer upserts lab_job_state and appends
lab_job_transitions as ``omninode_runtime``. The runtime's auto-wiring refuses
to prepare a projection writer whose principal holds no declared grant for a
table its contract names, and the dev lane's strict wiring turns that refusal
into a boot crash. So every shipped profile must grant both tables, and the
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
_TABLES = ("lab_job_state", "lab_job_transitions")
_VENDORED = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_lab_job"
)


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_the_runtime_role_may_read_and_write_both_lab_job_tables(profile: str) -> None:
    grants = declared_acl(load_topology_profile(profile)).table_grants
    for table in _TABLES:
        declared = {
            privilege
            for principal, relation, privilege in grants
            if principal == _RUNTIME and relation == table
        }
        assert declared == {"SELECT", "INSERT", "UPDATE"}, (profile, table)


def test_both_lab_job_migrations_are_vendored() -> None:
    assert sorted(path.name for path in _VENDORED.glob("*.sql")) == [
        "0000_create_lab_job.sql",
        "0001_grant_omninode_runtime_lab_job.sql",
    ]
