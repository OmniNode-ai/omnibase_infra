# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-apply proofs for the vendored generation-stability columns (OMN-19795).

delegation_eval_results gains each class's flip rate, its Wilson interval and
the measuring backend. These proofs apply the real eval-run chain 0003-0007 to a
real database: the four columns arrive nullable, a row written before them keeps
its values with the new columns NULL, a second apply changes nothing, and the
tenant writer can set them.
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
    / "node_projection_delegation_eval"
)
BEFORE = (
    FORWARD / "0003_create_delegation_eval_item_verdicts.sql",
    FORWARD / "0004_create_delegation_eval_results.sql",
    FORWARD / "0005_grant_tenant_projection_writer_delegation_eval_run_tables.sql",
    FORWARD / "0006_force_rls_delegation_eval_run_tables.sql",
)
ADD = FORWARD / "0007_add_generation_stability_to_delegation_eval_results.sql"
COLUMNS = {
    "flip_rate": "double precision",
    "flip_rate_wilson_low": "double precision",
    "flip_rate_wilson_high": "double precision",
    "flip_measured_backend_id": "text",
}
TENANT = "00000000-0000-0000-0000-000000000001"


def _apply(pg: EphemeralPostgres, *migrations: Path) -> None:
    for migration in migrations:
        result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
        assert result.returncode == 0, result.stderr


@pytest.fixture
def before(
    ephemeral_postgres: EphemeralPostgres,
) -> Iterator[psycopg2.extensions.connection]:
    provisioned = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles
                           WHERE rolname = 'tenant_projection_writer') THEN
                CREATE ROLE tenant_projection_writer NOLOGIN;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles
                           WHERE rolname = 'app_dashboard') THEN
                CREATE ROLE app_dashboard NOLOGIN;
            END IF;
        END $$;
        """,
    )
    assert provisioned.returncode == 0, provisioned.stderr
    _apply(ephemeral_postgres, *BEFORE)
    conn = ephemeral_postgres.connect()
    try:
        yield conn
    finally:
        conn.close()


def _insert_row(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.delegation_eval_results (
                tenant_id, eval_run_id, task_class, stratum, arm, manifest_id,
                label_set_sha256, gate_version, rater_role, rubric_version,
                total_n, accepted_n, false_pass_count, observed_at
            ) VALUES (
                %s, '735b18b8-734e-5c7d-bc00-47926ac70963', 'document', 'all',
                'recorded', 'manifest-1', 'sha', 'gate-1', 'rater', 'v1',
                32, 20, 10, NOW()
            )
            """,
            (TENANT,),
        )
    conn.commit()


def _columns(conn: psycopg2.extensions.connection) -> dict[str, tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'delegation_eval_results'
            """
        )
        return {name: (kind, nullable) for name, kind, nullable in cur.fetchall()}


def test_the_four_columns_arrive_nullable_and_keep_an_earlier_row(
    before: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    assert not set(COLUMNS) & set(_columns(before))
    _insert_row(before)
    _apply(ephemeral_postgres, ADD)
    columns = _columns(before)
    for name, kind in COLUMNS.items():
        assert columns[name] == (kind, "YES")
    with before.cursor() as cur:
        cur.execute(
            "SELECT accepted_n, false_pass_count, flip_rate, flip_rate_wilson_low,"
            " flip_rate_wilson_high, flip_measured_backend_id"
            " FROM public.delegation_eval_results"
        )
        assert cur.fetchall() == [(20, 10, None, None, None, None)]
    before.commit()


def test_a_second_apply_changes_nothing(
    before: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply(ephemeral_postgres, ADD)
    _insert_row(before)
    with before.cursor() as cur:
        cur.execute(
            "UPDATE public.delegation_eval_results SET flip_rate = 0.25,"
            " flip_rate_wilson_low = 0.1, flip_rate_wilson_high = 0.47,"
            " flip_measured_backend_id = 'local-heavy-reasoning'"
        )
        cur.execute("SELECT * FROM public.delegation_eval_results")
        kept = cur.fetchall()
    before.commit()
    _apply(ephemeral_postgres, ADD)
    with before.cursor() as cur:
        cur.execute("SELECT * FROM public.delegation_eval_results")
        assert cur.fetchall() == kept


def test_the_tenant_writer_can_set_the_columns(
    before: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply(ephemeral_postgres, ADD)
    with before.cursor() as cur:
        cur.execute(
            "SELECT has_column_privilege('tenant_projection_writer',"
            " 'public.delegation_eval_results', 'flip_rate', 'UPDATE')"
        )
        assert cur.fetchone() == (True,)
