# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-apply proofs for the vendored delegation capability map projection (OMN-20169).

delegation_capability_cells holds one capability cell per tenant, map run,
model, backend, configuration, task class and size band. These proofs apply the
real vendored chain to a real database and exercise the tenant boundary with a
real non-superuser writer: another tenant's cell can be neither written nor read.
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
    / "node_projection_delegation_capability_map"
)
MIGRATIONS = (
    FORWARD / "0000_create_delegation_capability_cells.sql",
    FORWARD / "0001_grant_tenant_projection_writer_delegation_capability_cells.sql",
    FORWARD / "0002_force_rls_delegation_capability_cells.sql",
)
TENANT_A = "00000000-0000-0000-0000-000000000001"
TENANT_B = "00000000-0000-0000-0000-000000000002"
KEY = [
    "tenant_id",
    "map_run_id",
    "model_key",
    "backend_id",
    "configuration_id",
    "task_class",
    "size_band",
]


def _apply_migrations(pg: EphemeralPostgres) -> None:
    for migration in MIGRATIONS:
        result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
        assert result.returncode == 0, result.stderr


@pytest.fixture
def applied(
    ephemeral_postgres: EphemeralPostgres,
) -> Iterator[psycopg2.extensions.connection]:
    """Provision cluster roles as superuser, then apply the real migration chain."""
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
    size_band: str = "S",
    n: int = 100,
    passes: int = 97,
    verdict: str = "MET",
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.delegation_capability_cells (
                tenant_id, map_run_id, model_key, backend_id, configuration_id,
                task_class, size_band, configuration_parameters, n, passes,
                pass_rate, wilson_low, wilson_high, decision_lower_bound, floor,
                required_n, verdict, observed_at
            ) VALUES (
                %s, 'run-1', 'qwen3-coder-30b', 'local-url', 'cfg-0123456789abcdef',
                'code_review', %s, '{"context_strategy_id": "full_context"}'::jsonb,
                %s, %s, 0.97, 0.92, 0.99, 0.91, 0.9, 89, %s, NOW()
            )
            """,
            (tenant_id, size_band, n, passes, verdict),
        )
    conn.commit()


def test_reapplying_all_three_migrations_converges(
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """A second apply preserves a populated row."""
    _insert(applied)
    with applied.cursor() as cur:
        cur.execute("SELECT * FROM public.delegation_capability_cells")
        before = cur.fetchall()
    assert len(before) == 1
    applied.commit()

    _apply_migrations(ephemeral_postgres)

    with applied.cursor() as cur:
        cur.execute("SELECT * FROM public.delegation_capability_cells")
        assert cur.fetchall() == before


def test_primary_key_is_the_cell_identity(
    applied: psycopg2.extensions.connection,
) -> None:
    """The key refuses a second row for one cell and keeps another band."""
    with applied.cursor() as cur:
        cur.execute(
            """
            SELECT array_agg(a.attname ORDER BY k.ordinality)
            FROM pg_catalog.pg_constraint AS c
            CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ordinality)
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = c.conrelid AND a.attnum = k.attnum
            WHERE c.conrelid = 'public.delegation_capability_cells'::regclass
              AND c.contype = 'p'
            """
        )
        assert cur.fetchone() == (KEY,)

    _insert(applied)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        _insert(applied)
    applied.rollback()
    _insert(applied, size_band="M")

    with applied.cursor() as cur:
        cur.execute(
            "SELECT size_band FROM public.delegation_capability_cells ORDER BY size_band"
        )
        assert cur.fetchall() == [("M",), ("S",)]


@pytest.mark.parametrize(
    ("size_band", "verdict", "n", "passes"),
    [
        ("XL", "MET", 100, 97),
        ("S", "PASS", 100, 97),
        ("S", "REFUSED", 0, 0),
        ("S", "REFUSED", 10, 11),
    ],
)
def test_a_malformed_cell_is_refused(
    applied: psycopg2.extensions.connection,
    size_band: str,
    verdict: str,
    n: int,
    passes: int,
) -> None:
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert(applied, size_band=size_band, verdict=verdict, n=n, passes=passes)
    applied.rollback()


def test_row_level_security_is_enabled_and_forced(
    applied: psycopg2.extensions.connection,
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            """
            SELECT relrowsecurity, relforcerowsecurity
            FROM pg_catalog.pg_class
            WHERE oid = 'public.delegation_capability_cells'::regclass
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
            "CREATE ROLE capability_cells_test_member "
            "WITH LOGIN NOSUPERUSER NOBYPASSRLS INHERIT"
        )
        cur.execute("GRANT tenant_projection_writer TO capability_cells_test_member")
    applied.commit()

    writer = ephemeral_postgres.connect(user="capability_cells_test_member")
    try:
        writer.autocommit = True
        with writer.cursor() as cur:
            cur.execute("SET ROLE tenant_projection_writer")
            cur.execute("SELECT set_config('app.tenant_id', %s, false)", (TENANT_A,))

        _insert(writer)
        with pytest.raises(
            psycopg2.errors.InsufficientPrivilege, match="row-level security policy"
        ):
            _insert(writer, tenant_id=TENANT_B, size_band="L")

        with writer.cursor() as cur:
            cur.execute("SELECT tenant_id FROM public.delegation_capability_cells")
            assert cur.fetchall() == [(TENANT_A,)]
    finally:
        writer.close()


def test_writer_has_only_required_table_privileges(
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
                "'tenant_projection_writer', 'public.delegation_capability_cells', %s)",
                (privilege,),
            )
            assert cur.fetchone() == (expected,), privilege
