# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-apply proofs for the vendored local_identity and local_alerts projections.

local_identity (OMN-19986) holds one row per tenant: the install's identity and
when it was minted. local_alerts (OMN-19991) holds one row per tenant, alert code
and subject. These proofs apply each real vendored chain to a real database and
exercise the tenant boundary with a real non-superuser writer.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import sql

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

REPO_ROOT = Path(__file__).resolve().parents[3]
NODES = REPO_ROOT / "docker" / "migrations" / "forward" / "nodes"
TENANT_A = "00000000-0000-0000-0000-000000000001"
TENANT_B = "00000000-0000-0000-0000-000000000002"
TABLES = ("local_identity", "local_alerts")


def _select(column: str, table: str) -> sql.Composed:
    target = sql.SQL("*") if column == "*" else sql.Identifier(column)
    return sql.SQL("SELECT {} FROM {}").format(target, sql.Identifier("public", table))


def _migrations(table: str) -> tuple[Path, ...]:
    node = NODES / f"node_projection_{table}"
    return (
        node / f"0000_create_{table}.sql",
        node / f"0001_grant_tenant_projection_writer_{table}.sql",
        node / f"0002_force_rls_{table}.sql",
    )


def _apply_migrations(pg: EphemeralPostgres) -> None:
    for table in TABLES:
        for migration in _migrations(table):
            result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
            assert result.returncode == 0, result.stderr


@pytest.fixture
def applied(
    ephemeral_postgres: EphemeralPostgres,
) -> Iterator[psycopg2.extensions.connection]:
    """Provision cluster roles as superuser, then apply both real migration chains."""
    provisioned = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_catalog.pg_roles
                WHERE rolname = 'tenant_projection_writer'
            ) THEN
                CREATE ROLE tenant_projection_writer NOLOGIN;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM pg_catalog.pg_roles
                WHERE rolname = 'app_dashboard'
            ) THEN
                CREATE ROLE app_dashboard NOLOGIN;
            END IF;
        END $$;
        """,
    )
    assert provisioned.returncode == 0, provisioned.stderr
    _apply_migrations(ephemeral_postgres)
    conn = ephemeral_postgres.connect()
    try:
        yield conn
    finally:
        conn.close()


def _insert_identity(
    conn: psycopg2.extensions.connection,
    *,
    tenant_id: str = TENANT_A,
    subject: str = "",
) -> None:
    del subject  # the identity is one row per tenant
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.local_identity (tenant_id, minted_at, event_id) "
            "VALUES (%s, NOW(), %s)",
            (tenant_id, str(uuid.uuid4())),
        )
    conn.commit()


def _insert_alert(
    conn: psycopg2.extensions.connection,
    *,
    tenant_id: str = TENANT_A,
    subject: str = "compose/postgres",
) -> None:
    mode, service = subject.split("/")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.local_alerts (
                tenant_id, code, subject, mode, service, state, severity,
                raised_at, evidence, last_transition_id, last_transition_at
            ) VALUES (
                %s, 'SERVICE_DOWN', %s, %s, %s, 'raised', 'critical',
                NOW(), '{}'::jsonb, %s, NOW()
            )
            """,
            (tenant_id, subject, mode, service, str(uuid.uuid4())),
        )
    conn.commit()


INSERTS: dict[str, Callable[..., None]] = {
    "local_identity": _insert_identity,
    "local_alerts": _insert_alert,
}
KEYS = {
    "local_identity": ["tenant_id"],
    "local_alerts": ["tenant_id", "code", "subject"],
}


@pytest.mark.parametrize("table", TABLES)
def test_reapplying_all_migrations_converges(
    table: str,
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """A second apply preserves a populated row, including its cursor."""
    INSERTS[table](applied)
    with applied.cursor() as cur:
        cur.execute(_select("*", table))
        before = cur.fetchall()
    assert len(before) == 1
    applied.commit()

    _apply_migrations(ephemeral_postgres)

    with applied.cursor() as cur:
        cur.execute(_select("*", table))
        assert cur.fetchall() == before


@pytest.mark.parametrize("table", TABLES)
def test_primary_key_is_the_declared_key(
    table: str, applied: psycopg2.extensions.connection
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            """
            SELECT array_agg(a.attname ORDER BY k.ordinality)
            FROM pg_catalog.pg_constraint AS c
            CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ordinality)
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = c.conrelid AND a.attnum = k.attnum
            WHERE c.conrelid = %s::regclass AND c.contype = 'p'
            """,
            (f"public.{table}",),
        )
        assert cur.fetchone() == (KEYS[table],)

    INSERTS[table](applied)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        INSERTS[table](applied)
    applied.rollback()


def test_a_second_alert_subject_is_its_own_row(
    applied: psycopg2.extensions.connection,
) -> None:
    _insert_alert(applied)
    _insert_alert(applied, subject="compose/valkey")
    with applied.cursor() as cur:
        cur.execute("SELECT subject FROM public.local_alerts ORDER BY subject")
        assert cur.fetchall() == [("compose/postgres",), ("compose/valkey",)]


def test_a_cleared_alert_carries_its_clear_time(
    applied: psycopg2.extensions.connection,
) -> None:
    with applied.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute(
            """
            INSERT INTO public.local_alerts (
                tenant_id, code, subject, mode, service, state, severity,
                raised_at, evidence, last_transition_id, last_transition_at
            ) VALUES (
                %s, 'SERVICE_DOWN', 'compose/postgres', 'compose', 'postgres',
                'cleared', 'critical', NOW(), '{}'::jsonb, %s, NOW()
            )
            """,
            (TENANT_A, str(uuid.uuid4())),
        )
    applied.rollback()


def test_an_empty_tenant_identity_is_refused(
    applied: psycopg2.extensions.connection,
) -> None:
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert_identity(applied, tenant_id="")
    applied.rollback()


@pytest.mark.parametrize("table", TABLES)
def test_row_level_security_is_enabled_and_forced(
    table: str, applied: psycopg2.extensions.connection
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_catalog.pg_class "
            "WHERE oid = %s::regclass",
            (f"public.{table}",),
        )
        assert cur.fetchone() == (True, True)


@pytest.mark.parametrize("table", TABLES)
def test_writer_can_insert_only_its_tenant_and_cannot_read_another(
    table: str,
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """Use a real non-superuser login and SET ROLE to exercise both RLS clauses."""
    member = f"{table}_test_member"
    INSERTS[table](applied, tenant_id=TENANT_B)
    with applied.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE ROLE {} WITH LOGIN NOSUPERUSER NOBYPASSRLS INHERIT").format(
                sql.Identifier(member)
            )
        )
        cur.execute(
            sql.SQL("GRANT tenant_projection_writer TO {}").format(
                sql.Identifier(member)
            )
        )
    applied.commit()

    writer = ephemeral_postgres.connect(user=member)
    try:
        writer.autocommit = True
        with writer.cursor() as cur:
            cur.execute("SET ROLE tenant_projection_writer")
            cur.execute("SELECT set_config('app.tenant_id', %s, false)", (TENANT_A,))

        INSERTS[table](writer)
        with pytest.raises(
            psycopg2.errors.InsufficientPrivilege, match="row-level security policy"
        ):
            INSERTS[table](writer, tenant_id=TENANT_B, subject="compose/valkey")

        with writer.cursor() as cur:
            cur.execute(_select("tenant_id", table))
            assert cur.fetchall() == [(TENANT_A,)]
    finally:
        writer.close()


@pytest.mark.parametrize("table", TABLES)
def test_writer_has_only_required_table_and_sequence_privileges(
    table: str, applied: psycopg2.extensions.connection
) -> None:
    with applied.cursor() as cur:
        for privilege, expected in (
            ("SELECT", True),
            ("INSERT", True),
            ("UPDATE", True),
            ("DELETE", False),
        ):
            cur.execute(
                "SELECT has_table_privilege('tenant_projection_writer', %s, %s)",
                (f"public.{table}", privilege),
            )
            assert cur.fetchone() == (expected,), privilege
        cur.execute(
            "SELECT has_sequence_privilege('tenant_projection_writer', %s, 'USAGE')",
            (f"public.{table}_projection_cursor_seq",),
        )
        assert cur.fetchone() == (True,)
        cur.execute(
            "SELECT has_table_privilege('app_dashboard', %s, 'SELECT')",
            (f"public.{table}",),
        )
        assert cur.fetchone() == (True,)
