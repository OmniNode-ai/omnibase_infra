# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Prove the work-question ACL successor against inherited PostgreSQL grants."""

from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.ephemeral_pg]

_MIGRATIONS = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/nodes/node_projection_work_questions"
)
_TABLES = (
    "work_outcome_units",
    "orchestration_style_outcomes",
    "lane_drift_and_stalls",
    "brief_quality_vs_outcome",
    "subagent_waste",
)
_SUCCESSOR = _MIGRATIONS / "0001_revoke_undeclared_runtime_grants.sql"


def _prepare(pg: EphemeralPostgres) -> None:
    result = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE ROLE omninode_runtime; "
        "CREATE SCHEMA omninode_internal; "
        "ALTER DEFAULT PRIVILEGES IN SCHEMA omninode_internal "
        "GRANT SELECT, INSERT, UPDATE ON TABLES TO omninode_runtime; "
        "CREATE TABLE omninode_internal.acl_neighbor (id integer);",
    )
    assert result.returncode == 0, result.stderr
    result = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        str(_MIGRATIONS / "0000_create_work_questions.sql"),
    )
    assert result.returncode == 0, result.stderr


def _grants(pg: EphemeralPostgres) -> set[tuple[str, str]]:
    with pg.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT c.relname, a.privilege_type "
            "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "CROSS JOIN LATERAL aclexplode(c.relacl) a "
            "WHERE n.nspname = 'omninode_internal' "
            "AND a.grantee = 'omninode_runtime'::regrole"
        )
        return set(cursor.fetchall())


def test_revoke_is_scoped_and_idempotent(ephemeral_postgres: EphemeralPostgres) -> None:
    pg = ephemeral_postgres
    _prepare(pg)
    expected = {
        (table, privilege)
        for table in (*_TABLES, "acl_neighbor")
        for privilege in ("SELECT", "INSERT", "UPDATE")
    }
    assert _grants(pg) == expected
    neighbor = {grant for grant in expected if grant[0] == "acl_neighbor"}
    for _ in range(2):
        result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(_SUCCESSOR))
        assert result.returncode == 0, result.stderr
        assert _grants(pg) == neighbor
    with pg.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'omninode_internal'"
        )
        assert {row[0] for row in cursor.fetchall()} == {*_TABLES, "acl_neighbor"}


def test_non_grantor_cannot_report_a_successful_revoke(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    _prepare(pg)
    result = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE ROLE acl_runner LOGIN; "
        "GRANT USAGE ON SCHEMA omninode_internal TO acl_runner; "
        "GRANT SELECT ON ALL TABLES IN SCHEMA omninode_internal TO acl_runner;",
    )
    assert result.returncode == 0, result.stderr
    before = _grants(pg)
    result = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        str(_SUCCESSOR),
        user="acl_runner",
    )
    assert result.returncode != 0
    assert "division by zero" in result.stderr, result.stderr
    assert _grants(pg) == before
