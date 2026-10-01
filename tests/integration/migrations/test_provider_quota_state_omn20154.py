# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-apply proofs for the vendored provider quota state (OMN-20154)."""

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
    / "node_projection_provider_quota"
)
MIGRATIONS = (
    FORWARD / "0000_create_provider_quota_state.sql",
    FORWARD / "0001_grant_tenant_projection_writer_provider_quota_state.sql",
    FORWARD / "0002_force_rls_provider_quota_state.sql",
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
    model_scope: str = "*",
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.provider_quota_state (
                tenant_id, credential_ref, provider_id, model_scope, observed_at
            ) VALUES (%s, %s, %s, %s, NOW())
            """,
            (tenant_id, "test-credential-ref", "test-provider", model_scope),
        )
    conn.commit()


def test_reapplying_all_three_migrations_converges(
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """A second apply preserves a populated row, including its cursor."""
    _insert(applied)
    with applied.cursor() as cur:
        cur.execute("SELECT * FROM public.provider_quota_state")
        before = cur.fetchall()
    assert len(before) == 1
    # Release ACCESS SHARE before the separate psql session needs table locks.
    applied.commit()

    _apply_migrations(ephemeral_postgres)

    with applied.cursor() as cur:
        cur.execute("SELECT * FROM public.provider_quota_state")
        assert cur.fetchall() == before


def test_primary_key_distinguishes_provider_wide_and_model_rows(
    applied: psycopg2.extensions.connection,
) -> None:
    """The four-column key rejects duplicates but retains distinct scopes."""
    with applied.cursor() as cur:
        cur.execute(
            """
            SELECT array_agg(a.attname ORDER BY k.ordinality)
            FROM pg_catalog.pg_constraint AS c
            CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ordinality)
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = c.conrelid AND a.attnum = k.attnum
            WHERE c.conrelid = 'public.provider_quota_state'::regclass
              AND c.contype = 'p'
            """
        )
        assert cur.fetchone() == (
            ["tenant_id", "credential_ref", "provider_id", "model_scope"],
        )

    _insert(applied)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        _insert(applied)
    applied.rollback()
    _insert(applied, model_scope="test-model")

    with applied.cursor() as cur:
        cur.execute(
            "SELECT model_scope FROM public.provider_quota_state ORDER BY model_scope"
        )
        assert cur.fetchall() == [("*",), ("test-model",)]


def test_row_level_security_is_enabled_and_forced(
    applied: psycopg2.extensions.connection,
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            """
            SELECT relrowsecurity, relforcerowsecurity
            FROM pg_catalog.pg_class
            WHERE oid = 'public.provider_quota_state'::regclass
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
            "CREATE ROLE provider_quota_test_member "
            "WITH LOGIN NOSUPERUSER NOBYPASSRLS INHERIT"
        )
        cur.execute("GRANT tenant_projection_writer TO provider_quota_test_member")
    applied.commit()

    writer = ephemeral_postgres.connect(user="provider_quota_test_member")
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
            _insert(writer, tenant_id=TENANT_B, model_scope="forbidden-model")

        with writer.cursor() as cur:
            cur.execute("SELECT tenant_id::text FROM public.provider_quota_state")
            assert cur.fetchall() == [(TENANT_A,)]
            cur.execute(
                "SELECT count(*) FROM public.provider_quota_state WHERE tenant_id = %s",
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
                "'tenant_projection_writer', 'public.provider_quota_state', %s)",
                (privilege,),
            )
            assert cur.fetchone() == (expected,), privilege
        cur.execute(
            "SELECT has_sequence_privilege('tenant_projection_writer', "
            "'public.provider_quota_state_projection_cursor_seq', 'USAGE')"
        )
        assert cur.fetchone() == (True,)
