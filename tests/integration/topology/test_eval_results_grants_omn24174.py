# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The tenant projection writer may read and write the eval results tables on every profile (OMN-24174).

omnimarket's node_projection_eval_results keeps four stores (scored items, traces, run closes, run rates) and the
tenant projection writer reads and writes them. The runtime refuses a writer whose principal holds no declared
grant for a table its contract names, and the omnimarket Event Chain Gate must not fail at collection with
``lacks declared read privileges`` on those tables. So every shipped profile
must grant the table, the vendored migrations must be the node's own, and the control below shows the check
reporting a profile that lacks the grant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.validation.omninode_internal_live_acl import declared_acl

pytestmark = pytest.mark.integration

_RUNTIME = "tenant_projection_writer"
_TABLES = ("eval_item_scores", "eval_item_traces", "eval_run_closes", "eval_run_rates")
_VENDORED = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_eval_results"
)


_EXPECTED = {table: {"SELECT", "INSERT", "UPDATE"} for table in _TABLES}


def _declared(profile: str, table: str) -> set[str]:
    grants = declared_acl(load_topology_profile(profile), schema="public").table_grants
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
def test_the_tenant_projection_writer_may_read_and_write_the_eval_results_tables(
    profile: str,
) -> None:
    for table in _TABLES:
        assert _declared(profile, table) == _EXPECTED[table], (
            profile,
            table,
        )


def test_control_removing_the_grant_is_reported_as_missing() -> None:
    """Removing the eval results grants makes the writer's read of it empty."""
    for profile in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        grants = declared_acl(
            load_topology_profile(profile), schema="public"
        ).table_grants
        broken = [entry for entry in grants if entry[1] not in _TABLES]
        for table in _TABLES:
            assert _declared(profile, table), (profile, table)
            assert _declared_from(broken, table) == set(), (profile, table)


def test_the_three_eval_results_migrations_are_vendored() -> None:
    assert sorted(path.name for path in _VENDORED.glob("*.sql")) == [
        "0000_create_eval_results_tables.sql",
        "0001_grant_tenant_projection_writer_eval_results.sql",
        "0002_force_rls_eval_results_tables.sql",
    ]
