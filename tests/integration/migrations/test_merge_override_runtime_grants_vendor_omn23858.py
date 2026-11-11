# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live-apply proofs for the vendored merge override grant chain (OMN-23858).

omnimarket's node_merge_override_alarm_effect store reads and writes its four
tables as ``omninode_runtime``, and deliberately holds no DELETE: receipts are
replaced, never removed, and findings are opened, never deleted. These proofs
drive the real vendored SQL through ``psql`` so the shipped bytes — not a
re-typed summary of them — are what is proven, and a failing control shows the
role is refused without the grant migration (the dev database's state before
this change).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import psycopg2
import psycopg2.errors
import pytest

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.validation.omninode_internal_live_acl import declared_acl
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.ephemeral_pg]

_CHAIN = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/nodes/node_merge_override_alarm_effect"
)
_VENDORED_CHAIN = [
    "0001_create_merge_override_tables.sql",
    "0002_grant_omninode_runtime_merge_override_tables.sql",
]
_TABLES = (
    "merge_override_receipts",
    "merge_override_pending",
    "merge_override_verdicts",
    "merge_override_open_findings",
)
_GRANTED = {"SELECT", "INSERT", "UPDATE"}
_DENIED = ("DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")
_RUNTIME = "omninode_runtime"


def _apply_migrations(pg: EphemeralPostgres, head: int) -> None:
    """Apply the vendored chain's first ``head`` files via the production psql path."""
    migrations = sorted(_CHAIN.glob("*.sql"))
    assert [path.name for path in migrations] == _VENDORED_CHAIN
    for migration in migrations[:head]:
        result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
        assert result.returncode == 0, f"{migration.name}: {result.stderr}"


def _provision(pg: EphemeralPostgres) -> None:
    """Create the schema and the runtime role the vendored grants name."""
    provisioned = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE SCHEMA omninode_internal; CREATE ROLE omninode_runtime NOLOGIN;",
    )
    assert provisioned.returncode == 0, provisioned.stderr


@pytest.fixture
def applied(
    ephemeral_postgres: EphemeralPostgres,
) -> Iterator[psycopg2.extensions.connection]:
    """Provision the schema and runtime role, then apply the real vendored SQL."""
    _provision(ephemeral_postgres)
    _apply_migrations(ephemeral_postgres, head=2)
    conn = ephemeral_postgres.connect()
    try:
        yield conn
    finally:
        conn.close()


def test_runtime_role_holds_exactly_select_insert_update_on_each_of_the_four_tables(
    applied: psycopg2.extensions.connection,
) -> None:
    with applied.cursor() as cur:
        for table in _TABLES:
            cur.execute(
                "SELECT privilege_type FROM information_schema.role_table_grants "
                "WHERE table_schema = 'omninode_internal' AND table_name = %s "
                "AND grantee = %s",
                (table, _RUNTIME),
            )
            assert {row[0] for row in cur.fetchall()} == _GRANTED, table
            for privilege in _DENIED:
                cur.execute(
                    "SELECT has_table_privilege(%s, %s, %s)",
                    (_RUNTIME, f"omninode_internal.{table}", privilege),
                )
                assert cur.fetchone() == (False,), (table, privilege)
        cur.execute(
            "SELECT has_schema_privilege(%s, 'omninode_internal', 'USAGE'), "
            "has_schema_privilege(%s, 'omninode_internal', 'CREATE')",
            (_RUNTIME, _RUNTIME),
        )
        assert cur.fetchone() == (True, False)


def test_runtime_writer_can_insert_and_update_a_receipt(
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            "CREATE ROLE merge_override_test_writer LOGIN NOSUPERUSER NOBYPASSRLS "
            "IN ROLE omninode_runtime"
        )
    applied.commit()
    writer = ephemeral_postgres.connect(user="merge_override_test_writer")
    try:
        writer.autocommit = True
        with writer.cursor() as cur:
            cur.execute("SET ROLE omninode_runtime")
            cur.execute(
                "SELECT current_user, rolsuper, rolbypassrls FROM pg_catalog.pg_roles "
                "WHERE rolname = current_user"
            )
            assert cur.fetchone() == (_RUNTIME, False, False)
            cur.execute(
                "INSERT INTO omninode_internal.merge_override_receipts "
                "(repo_pr, repository, pr_number, head_sha, operation, observed_at, correlation_id) "
                "VALUES ('omnibase/omnibase_infra#5896', 'omnibase_infra', 5896, "
                "'head-5896', 'enforce', NOW(), 'corr-5896')"
            )
            cur.execute(
                "UPDATE omninode_internal.merge_override_receipts SET operation = 'enqueue' "
                "WHERE repo_pr = 'omnibase/omnibase_infra#5896'"
            )
            cur.execute(
                "SELECT repository, pr_number, head_sha, operation, correlation_id "
                "FROM omninode_internal.merge_override_receipts "
                "WHERE repo_pr = 'omnibase/omnibase_infra#5896'"
            )
            assert cur.fetchone() == (
                "omnibase_infra",
                5896,
                "head-5896",
                "enqueue",
                "corr-5896",
            )
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("DELETE FROM omninode_internal.merge_override_receipts")
    finally:
        writer.close()


def test_without_the_grant_migration_the_runtime_role_is_refused(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """Failing control: 0001 alone leaves the role grantless — the pre-change dev state."""
    _provision(ephemeral_postgres)
    _apply_migrations(ephemeral_postgres, head=1)
    conn = ephemeral_postgres.connect()
    try:
        with conn.cursor() as cur:
            for table in _TABLES:
                cur.execute(
                    "SELECT has_table_privilege(%s, %s, %s)",
                    (_RUNTIME, f"omninode_internal.{table}", "SELECT"),
                )
                assert cur.fetchone() == (False,), table
    finally:
        conn.close()


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_the_topology_declares_what_the_database_grants(profile: str) -> None:
    grants = declared_acl(load_topology_profile(profile)).table_grants
    for table in _TABLES:
        declared = {
            privilege
            for principal, relation, privilege in grants
            if principal == _RUNTIME and relation == table
        }
        assert declared == _GRANTED, (profile, table)
