# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime grants and byte-identical delegation execution migration (OMN-17427).

node_llm_delegation_call_effect uses both tables as omninode_runtime. Every
shipped profile must declare the read/write grants before omnimarket#4277
can prepare its projection bindings, and the sibling release must include
exactly the migration bytes from that source pull request.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.validation.omninode_internal_live_acl import declared_acl

pytestmark = pytest.mark.unit

_RUNTIME = "omninode_runtime"
_TABLES = ("delegation_lab_jobs", "delegation_lab_job_outbox")
_VENDORED = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_llm_delegation_call_effect"
)
_SOURCE_SHA256 = "80cd5c3b2f46dedd19542fd9e57dd673d95bff7fc48ecab0bdc2c558a9f3b607"


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_the_runtime_role_may_read_and_write_both_delegation_lab_job_tables(
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


def test_delegation_lab_job_migration_is_vendored_byte_identically() -> None:
    assert sorted(path.name for path in _VENDORED.glob("*.sql")) == [
        "0000_delegation_lab_job_execution.sql",
    ]
    assert (
        hashlib.sha256(
            (_VENDORED / "0000_delegation_lab_job_execution.sql").read_bytes()
        ).hexdigest()
        == _SOURCE_SHA256
    )
