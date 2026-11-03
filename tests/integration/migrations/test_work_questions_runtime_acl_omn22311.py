# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Apply the ACL repair to a disposable PostgreSQL cluster with real grants."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.slow]

_ROOT = Path(__file__).resolve().parents[3]
_MIGRATION = (
    _ROOT
    / "docker/migrations/forward/nodes/node_projection_work_questions"
    / "0001_revoke_undeclared_runtime_grants.sql"
)
_TABLES = (
    "work_outcome_units",
    "orchestration_style_outcomes",
    "lane_drift_and_stalls",
    "brief_quality_vs_outcome",
    "subagent_waste",
)


def _sql(pg: EphemeralPostgres, sql: str) -> str:
    result = pg.psql("-At", "-c", sql)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _prepare(pg: EphemeralPostgres, *, foreign_last_table: bool = False) -> None:
    _sql(
        pg,
        "CREATE ROLE omninode_runtime; CREATE ROLE work_questions_owner LOGIN; "
        "CREATE ROLE work_questions_foreign; CREATE ROLE work_questions_reader; "
        "CREATE SCHEMA omninode_internal AUTHORIZATION work_questions_owner; "
        "GRANT USAGE, CREATE ON SCHEMA omninode_internal TO work_questions_foreign; "
        "SET ROLE work_questions_owner; "
        "ALTER DEFAULT PRIVILEGES IN SCHEMA omninode_internal "
        "GRANT SELECT, INSERT, UPDATE ON TABLES TO omninode_runtime; "
        "RESET ROLE;",
    )
    for index, table in enumerate((*_TABLES, "unrelated_table")):
        owner = (
            "work_questions_foreign"
            if foreign_last_table and index == len(_TABLES) - 1
            else "work_questions_owner"
        )
        _sql(
            pg,
            f"SET ROLE {owner}; "
            f"CREATE TABLE omninode_internal.{table} (id integer); "
            f"GRANT SELECT, INSERT, UPDATE ON omninode_internal.{table} "
            "TO omninode_runtime; "
            f"GRANT SELECT ON omninode_internal.{table} TO work_questions_reader;",
        )
        if owner == "work_questions_foreign":
            # Enough access for REVOKE to warn rather than raise permission
            # denied, but no grant option: the post-condition must catch it.
            _sql(
                pg,
                f"SET ROLE {owner}; GRANT SELECT ON omninode_internal.{table} "
                "TO work_questions_owner;",
            )


def _grants(pg: EphemeralPostgres) -> str:
    return _sql(
        pg,
        "SELECT c.relname || ':' || r.rolname || ':' || a.privilege_type "
        "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace, "
        "aclexplode(c.relacl) a JOIN pg_roles r ON r.oid = a.grantee "
        "WHERE n.nspname = 'omninode_internal' "
        "AND r.rolname IN ('omninode_runtime', 'work_questions_reader') "
        "ORDER BY 1",
    )


def test_revoke_is_scoped_and_idempotent(ephemeral_postgres: EphemeralPostgres) -> None:
    pg = ephemeral_postgres
    _prepare(pg)
    before = _grants(pg).splitlines()
    assert len(before) == 24
    expected = [
        row
        for row in before
        if row.startswith("unrelated_table:") or ":work_questions_reader:" in row
    ]
    for _ in range(2):
        applied = pg.psql(
            "-v", "ON_ERROR_STOP=1", "-f", str(_MIGRATION), user="work_questions_owner"
        )
        assert applied.returncode == 0, applied.stderr
        assert _grants(pg).splitlines() == expected


def test_unrevokable_grant_refuses_and_rolls_back(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    _prepare(pg, foreign_last_table=True)
    before = _grants(pg)
    applied = pg.psql(
        "-v", "ON_ERROR_STOP=1", "-f", str(_MIGRATION), user="work_questions_owner"
    )
    assert applied.returncode != 0
    assert "division by zero" in applied.stderr
    assert _grants(pg) == before
