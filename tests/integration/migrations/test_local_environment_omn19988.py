# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-apply proofs for the vendored local environment projection (OMN-19988).

The local_environment table holds the latest report of each local service, per
tenant and mode, with its verdict and open alert. These proofs apply the real
vendored chain to a real database and exercise the tenant boundary with a real
non-superuser writer: another tenant's row for the same mode and service can be
neither written nor read.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import psycopg2
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_local_environment"
)
MIGRATIONS = (
    FORWARD / "0000_create_local_environment.sql",
    FORWARD / "0001_grant_tenant_projection_writer_local_environment.sql",
    FORWARD / "0002_force_rls_local_environment.sql",
)
TENANT_A = "00000000-0000-0000-0000-000000000001"
TENANT_B = "00000000-0000-0000-0000-000000000002"


def _apply_migrations(pg: EphemeralPostgres) -> None:
    for migration in MIGRATIONS:
        result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
        assert result.returncode == 0, result.stderr


@pytest.fixture
def applied(
    ephemeral_postgres: EphemeralPostgres,
) -> Iterator[psycopg2.extensions.connection]:
    """Provision cluster roles as superuser, then apply the real migration chain.

    Backend availability and skips are owned by the shared ephemeral fixture.
    """
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


def _insert(
    conn: psycopg2.extensions.connection,
    *,
    tenant_id: str = TENANT_A,
    service: str = "postgres",
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.local_environment (
                tenant_id, mode, service, observed_state, reason, last_seen,
                last_change, verdict, verdict_reason, report_id
            ) VALUES (
                %s, 'compose', %s, 'running', 'healthy', NOW(), NOW(),
                'running', 'reported running', %s
            )
            """,
            (tenant_id, service, f"report-{tenant_id}-{service}"),
        )
    conn.commit()


def test_reapplying_all_three_migrations_converges(
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """A second apply preserves a populated row, including its cursor."""
    _insert(applied)
    with applied.cursor() as cur:
        cur.execute("SELECT * FROM public.local_environment")
        before = cur.fetchall()
    assert len(before) == 1
    # Release ACCESS SHARE before the separate psql session needs table locks.
    applied.commit()

    _apply_migrations(ephemeral_postgres)

    with applied.cursor() as cur:
        cur.execute("SELECT * FROM public.local_environment")
        assert cur.fetchall() == before


def test_primary_key_is_tenant_mode_and_service(
    applied: psycopg2.extensions.connection,
) -> None:
    """The key rejects a second row for one service and keeps another service."""
    with applied.cursor() as cur:
        cur.execute(
            """
            SELECT array_agg(a.attname ORDER BY k.ordinality)
            FROM pg_catalog.pg_constraint AS c
            CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ordinality)
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = c.conrelid AND a.attnum = k.attnum
            WHERE c.conrelid = 'public.local_environment'::regclass
              AND c.contype = 'p'
            """
        )
        assert cur.fetchone() == (["tenant_id", "mode", "service"],)

    _insert(applied)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        _insert(applied)
    applied.rollback()
    _insert(applied, service="valkey")

    with applied.cursor() as cur:
        cur.execute("SELECT service FROM public.local_environment ORDER BY service")
        assert cur.fetchall() == [("postgres",), ("valkey",)]


def test_an_alert_is_stored_whole_or_not_at_all(
    applied: psycopg2.extensions.connection,
) -> None:
    """An alert code without its severity and start is refused."""
    with applied.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute(
            """
            INSERT INTO public.local_environment (
                tenant_id, mode, service, observed_state, reason, last_seen,
                last_change, verdict, verdict_reason, report_id, alert_code
            ) VALUES (
                %s, 'compose', 'postgres', 'down', 'exited', NOW(), NOW(),
                'down', 'reported down', 'report-1', 'SERVICE_DOWN'
            )
            """,
            (TENANT_A,),
        )
    applied.rollback()


def test_row_level_security_is_enabled_and_forced(
    applied: psycopg2.extensions.connection,
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            """
            SELECT relrowsecurity, relforcerowsecurity
            FROM pg_catalog.pg_class
            WHERE oid = 'public.local_environment'::regclass
            """
        )
        assert cur.fetchone() == (True, True)


def test_writer_can_insert_only_its_tenant_and_cannot_read_another(
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """Use a real non-superuser login and SET ROLE to exercise both RLS clauses."""
    _insert(applied, tenant_id=TENANT_B)
    with applied.cursor() as cur:
        cur.execute(
            "CREATE ROLE local_environment_test_member "
            "WITH LOGIN NOSUPERUSER NOBYPASSRLS INHERIT"
        )
        cur.execute("GRANT tenant_projection_writer TO local_environment_test_member")
    applied.commit()

    writer = ephemeral_postgres.connect(user="local_environment_test_member")
    try:
        writer.autocommit = True
        with writer.cursor() as cur:
            cur.execute("SET ROLE tenant_projection_writer")
            cur.execute("SELECT set_config('app.tenant_id', %s, false)", (TENANT_A,))
            cur.execute(
                "SELECT current_user, rolsuper, rolbypassrls "
                "FROM pg_catalog.pg_roles WHERE rolname = current_user"
            )
            assert cur.fetchone() == ("tenant_projection_writer", False, False)

        _insert(writer)
        with pytest.raises(
            psycopg2.errors.InsufficientPrivilege, match="row-level security policy"
        ):
            # Same mode, a free service: only the tenant differs.
            _insert(writer, tenant_id=TENANT_B, service="valkey")

        with writer.cursor() as cur:
            cur.execute("SELECT tenant_id FROM public.local_environment")
            assert cur.fetchall() == [(TENANT_A,)]
            cur.execute(
                "SELECT count(*) FROM public.local_environment WHERE tenant_id = %s",
                (TENANT_B,),
            )
            assert cur.fetchone() == (0,)
    finally:
        writer.close()


def test_writer_has_only_required_table_and_sequence_privileges(
    applied: psycopg2.extensions.connection,
) -> None:
    with applied.cursor() as cur:
        for privilege, expected in (
            ("SELECT", True),
            ("INSERT", True),
            ("UPDATE", True),
            ("DELETE", False),
        ):
            cur.execute(
                "SELECT has_table_privilege("
                "'tenant_projection_writer', 'public.local_environment', %s)",
                (privilege,),
            )
            assert cur.fetchone() == (expected,), privilege
        cur.execute(
            "SELECT has_sequence_privilege('tenant_projection_writer', "
            "'public.local_environment_projection_cursor_seq', 'USAGE')"
        )
        assert cur.fetchone() == (True,)
