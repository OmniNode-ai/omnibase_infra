# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-apply proofs for the vendored budget applied-events table (OMN-20613).

0056 creates delegation_budget_applied_events and grants the projection
writer; 0057 forces its row-level security. Both are vendored verbatim from
omnimarket, so these proofs drive the real files through a throwaway cluster.
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
    / "node_projection_delegation"
)
MIGRATIONS = (
    FORWARD / "0056_delegation_budget_applied_events.sql",
    FORWARD / "0057_delegation_budget_applied_events_rls.sql",
)
TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


def _apply_migrations(pg: EphemeralPostgres) -> None:
    for migration in MIGRATIONS:
        result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
        assert result.returncode == 0, result.stderr


def _provision_roles(pg: EphemeralPostgres) -> None:
    provisioned = pg.psql(
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


@pytest.fixture
def applied(
    ephemeral_postgres: EphemeralPostgres,
) -> Iterator[psycopg2.extensions.connection]:
    """Provision cluster roles as superuser, then apply 0056 and 0057."""
    _provision_roles(ephemeral_postgres)
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
    correlation_id: str = "corr-1",
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.delegation_budget_applied_events (
                tenant_id, cost_tier_name, budget_period, correlation_id
            ) VALUES (%s, %s, %s, %s)
            """,
            (tenant_id, "standard", "2026-10", correlation_id),
        )
    conn.commit()


def test_without_writer_role_0056_refuses_to_apply(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(MIGRATIONS[0]))
    assert result.returncode != 0
    assert "tenant_projection_writer role missing" in result.stderr


def test_reapply_converges_and_preserves_rows(
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _insert(applied)
    with applied.cursor() as cur:
        cur.execute("SELECT * FROM public.delegation_budget_applied_events")
        before = cur.fetchall()
    assert len(before) == 1
    applied.commit()

    _apply_migrations(ephemeral_postgres)

    with applied.cursor() as cur:
        cur.execute("SELECT * FROM public.delegation_budget_applied_events")
        assert cur.fetchall() == before


def test_drifted_table_converges_to_declared_shape(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """A pre-existing table missing columns and its key reconciles in place."""
    _provision_roles(ephemeral_postgres)
    drifted = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE TABLE public.delegation_budget_applied_events (tenant_id TEXT)",
    )
    assert drifted.returncode == 0, drifted.stderr

    _apply_migrations(ephemeral_postgres)

    conn = ephemeral_postgres.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT attname, attnotnull
                FROM pg_catalog.pg_attribute
                WHERE attrelid = 'public.delegation_budget_applied_events'::regclass
                  AND attnum > 0 AND NOT attisdropped
                ORDER BY attnum
                """
            )
            assert cur.fetchall() == [
                ("tenant_id", True),
                ("cost_tier_name", True),
                ("budget_period", True),
                ("correlation_id", True),
                ("applied_at", True),
            ]
    finally:
        conn.close()


def test_event_identity_is_the_primary_key(
    applied: psycopg2.extensions.connection,
) -> None:
    """A replayed event identity is refused; a new correlation id is not."""
    _insert(applied)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        _insert(applied)
    applied.rollback()
    _insert(applied, correlation_id="corr-2")


def test_row_level_security_is_enabled_and_forced(
    applied: psycopg2.extensions.connection,
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            """
            SELECT relrowsecurity, relforcerowsecurity
            FROM pg_catalog.pg_class
            WHERE oid = 'public.delegation_budget_applied_events'::regclass
            """
        )
        assert cur.fetchone() == (True, True)


def test_writer_privileges_and_tenant_isolation(
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _insert(applied, tenant_id=TENANT_B)
    with applied.cursor() as cur:
        for privilege, expected in (
            ("SELECT", True),
            ("INSERT", True),
            ("UPDATE", True),
            ("DELETE", False),
        ):
            cur.execute(
                "SELECT has_table_privilege('tenant_projection_writer', "
                "'public.delegation_budget_applied_events', %s)",
                (privilege,),
            )
            assert cur.fetchone() == (expected,), privilege
        cur.execute(
            "CREATE ROLE budget_applied_test_member "
            "WITH LOGIN NOSUPERUSER NOBYPASSRLS INHERIT"
        )
        cur.execute("GRANT tenant_projection_writer TO budget_applied_test_member")
    applied.commit()

    writer = ephemeral_postgres.connect(user="budget_applied_test_member")
    try:
        writer.autocommit = True
        with writer.cursor() as cur:
            cur.execute("SET ROLE tenant_projection_writer")
            cur.execute("SELECT set_config('app.tenant_id', %s, false)", (TENANT_A,))
        _insert(writer)
        with pytest.raises(
            psycopg2.errors.InsufficientPrivilege, match="row-level security policy"
        ):
            _insert(writer, tenant_id=TENANT_B, correlation_id="forbidden")
        with writer.cursor() as cur:
            cur.execute("SELECT tenant_id FROM public.delegation_budget_applied_events")
            assert cur.fetchall() == [(TENANT_A,)]
    finally:
        writer.close()
