# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Apply the budget readings SQL and exercise its constraints and runtime ACLs."""

from pathlib import Path

import pytest
from psycopg2 import sql

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.ephemeral_pg]

_CHAIN = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/nodes/node_projection_delegation_budget_readings"
)
_INSERT = (
    "INSERT INTO omninode_internal.delegation_budget_readings (correlation_id, status, task_type, observed_at) "
    "VALUES ('00000000-0000-0000-0000-000000000001', 'completed', 'code', NOW())"
)


def _sql(
    pg: EphemeralPostgres, statement: str | sql.Composable, *, succeeds: bool = True
) -> str:
    if isinstance(statement, sql.Composable):
        with pg.connect() as conn:
            statement = statement.as_string(conn)
    result = pg.psql("-X", "-v", "ON_ERROR_STOP=1", "-At", "-c", statement)
    assert (result.returncode == 0) == succeeds, result.stderr
    return result.stdout.strip()


def _apply(pg: EphemeralPostgres, *, succeeds: bool = True) -> None:
    paths = sorted(_CHAIN.glob("*.sql"))
    assert len(paths) == 2
    for path in paths:
        result = pg.psql("-X", "-v", "ON_ERROR_STOP=1", "-f", str(path))
        assert (result.returncode == 0) == succeeds, result.stderr
        if not succeeds:
            return


def _provision(pg: EphemeralPostgres) -> None:
    _sql(pg, "CREATE SCHEMA omninode_internal; CREATE ROLE omninode_runtime NOLOGIN;")
    _apply(pg)


@pytest.mark.parametrize(
    "drift",
    [
        "ALTER COLUMN status DROP NOT NULL",
        "ALTER COLUMN task_type DROP NOT NULL",
        "ALTER COLUMN observed_at DROP NOT NULL",
        "ALTER COLUMN projected_at DROP NOT NULL",
        "ALTER COLUMN projected_at DROP DEFAULT",
        "ALTER COLUMN projection_cursor DROP NOT NULL",
        "ALTER COLUMN projection_cursor DROP DEFAULT",
        "DROP CONSTRAINT pk_delegation_budget_readings",
        "DROP CONSTRAINT ck_delegation_budget_readings_status",
        "DROP COLUMN correlation_id",
        "DROP COLUMN task_type",
        "DROP COLUMN projection_cursor",
    ],
)
def test_reapply_restores_empty_table_constraints(
    ephemeral_postgres: EphemeralPostgres, drift: str
) -> None:
    pg = ephemeral_postgres
    _provision(pg)
    _sql(
        pg,
        sql.SQL("ALTER TABLE omninode_internal.delegation_budget_readings {}").format(
            sql.SQL(drift)
        ),
    )
    _apply(pg)
    _sql(pg, _INSERT)
    assert (
        _sql(
            pg,
            "SELECT projected_at IS NOT NULL FROM omninode_internal.delegation_budget_readings",
        )
        == "t"
    )
    for column in (
        "correlation_id",
        "status",
        "task_type",
        "observed_at",
        "projected_at",
        "projection_cursor",
    ):
        _sql(
            pg,
            sql.SQL(
                "UPDATE omninode_internal.delegation_budget_readings SET {} = NULL"
            ).format(sql.Identifier(column)),
            succeeds=False,
        )
    _sql(
        pg,
        "UPDATE omninode_internal.delegation_budget_readings SET status = 'unknown'",
        succeeds=False,
    )
    _sql(pg, _INSERT, succeeds=False)


def test_reapply_preserves_populated_rows_cursors_and_sequences(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    _provision(pg)
    _sql(pg, _INSERT)
    before = _sql(
        pg, "SELECT row_to_json(t) FROM omninode_internal.delegation_budget_readings t"
    )
    sequences = _sql(pg, "SELECT oid, relname FROM pg_class WHERE relkind = 'S'")
    _apply(pg)
    assert (
        _sql(
            pg,
            "SELECT row_to_json(t) FROM omninode_internal.delegation_budget_readings t",
        )
        == before
    )
    assert (
        _sql(pg, "SELECT oid, relname FROM pg_class WHERE relkind = 'S'") == sequences
    )


def test_populated_table_missing_required_values_refuses_without_partial_repair(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    _provision(pg)
    _sql(pg, _INSERT)
    _sql(
        pg,
        "ALTER TABLE omninode_internal.delegation_budget_readings DROP COLUMN observed_at; ALTER TABLE omninode_internal.delegation_budget_readings DROP COLUMN terminal_topic",
    )
    _apply(pg, succeeds=False)
    assert (
        _sql(pg, "SELECT count(*) FROM omninode_internal.delegation_budget_readings")
        == "1"
    )
    assert (
        _sql(
            pg,
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema = 'omninode_internal' AND table_name = 'delegation_budget_readings' "
            "AND column_name IN ('observed_at', 'terminal_topic')",
        )
        == "0"
    )


def test_runtime_can_upsert_but_cannot_delete_or_widen_privileges(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    _provision(pg)
    assert (
        _sql(
            pg,
            "SELECT string_agg(privilege_type, ',' ORDER BY privilege_type) "
            "FROM information_schema.role_table_grants "
            "WHERE table_schema = 'omninode_internal' AND table_name = 'delegation_budget_readings' "
            "AND grantee = 'omninode_runtime'",
        )
        == "INSERT,SELECT,UPDATE"
    )
    # Include PUBLIC, grant options and any other role, not only runtime's rows.
    assert (
        _sql(
            pg,
            "SELECT count(*) FROM pg_class c, aclexplode(c.relacl) a "
            "WHERE c.oid = 'omninode_internal.delegation_budget_readings'::regclass AND a.grantee <> c.relowner "
            "AND (a.grantee <> 'omninode_runtime'::regrole OR a.is_grantable)",
        )
        == "0"
    )
    assert _sql(pg, "SELECT count(*) FROM pg_default_acl") == "0"
    assert (
        _sql(
            pg,
            "SELECT has_schema_privilege('omninode_runtime', 'omninode_internal', 'USAGE'), "
            "has_schema_privilege('omninode_runtime', 'omninode_internal', 'CREATE'), "
            "has_sequence_privilege('omninode_runtime', pg_get_serial_sequence('omninode_internal.delegation_budget_readings', 'projection_cursor'), 'USAGE'), "
            "has_sequence_privilege('omninode_runtime', pg_get_serial_sequence('omninode_internal.delegation_budget_readings', 'projection_cursor'), 'UPDATE')",
        )
        == "t|f|t|f"
    )
    assert (
        _sql(
            pg,
            "SELECT privilege_type FROM pg_class c, aclexplode(c.relacl) a "
            "WHERE c.oid = pg_get_serial_sequence('omninode_internal.delegation_budget_readings', 'projection_cursor')::regclass "
            "AND a.grantee <> c.relowner AND a.grantee = 'omninode_runtime'::regrole "
            "AND NOT a.is_grantable",
        )
        == "USAGE"
    )
    _sql(pg, "SET ROLE omninode_runtime; " + _INSERT)
    assert (
        _sql(
            pg,
            "SET ROLE omninode_runtime; "
            "INSERT INTO omninode_internal.delegation_budget_readings "
            "(correlation_id, status, task_type, observed_at) "
            "VALUES ('00000000-0000-0000-0000-000000000001', 'failed', 'code', NOW()) "
            "ON CONFLICT (correlation_id) "
            "DO UPDATE SET status = EXCLUDED.status RETURNING projection_cursor; "
            "SELECT status, projection_cursor FROM omninode_internal.delegation_budget_readings",
        ).splitlines()[-1]
        == "failed|1"
    )
    _sql(
        pg,
        "SET ROLE omninode_runtime; DELETE FROM omninode_internal.delegation_budget_readings",
        succeeds=False,
    )
    _sql(
        pg,
        "SET ROLE omninode_runtime; GRANT DELETE ON omninode_internal.delegation_budget_readings TO PUBLIC",
    )
    assert (
        _sql(
            pg,
            "SELECT count(*) FROM pg_class c, aclexplode(c.relacl) a "
            "WHERE c.oid = 'omninode_internal.delegation_budget_readings'::regclass "
            "AND a.grantee = 0",
        )
        == "0"
    )
