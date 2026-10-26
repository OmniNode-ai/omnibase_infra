# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-apply proofs for the vendored harness identity columns (OMN-21396).

The verdicts and results tables gain nullable harness identity columns. These
proofs apply the real eval-run chain 0003-0008 to a real database: an earlier
result keeps its values with the new columns NULL, a second apply changes
nothing and keeps one index, negative durations are refused, and the tenant
writer can set the columns.
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
    FORWARD / "0007_add_generation_stability_to_delegation_eval_results.sql",
)
ADD = FORWARD / "0008_harness_eval_identity.sql"
COLUMNS = {
    "delegation_eval_item_verdicts": {
        "harness": "text",
        "model": "text",
        "configuration": "text",
        "duration_s": "double precision",
        "failure_kind": "text",
    },
    "delegation_eval_results": {
        "harness": "text",
        "model": "text",
        "configuration": "text",
    },
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


def _insert_verdict(conn: psycopg2.extensions.connection, duration_s: float) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.delegation_eval_item_verdicts (
                tenant_id, eval_run_id, item_key, manifest_id, gate_version,
                rater_role, rubric_version, observed_at, duration_s
            ) VALUES (
                %s, '735b18b8-734e-5c7d-bc00-47926ac70963', 'item-1',
                'manifest-1', 'gate-1', 'rater', 'v1', NOW(), %s
            )
            """,
            (TENANT, duration_s),
        )
    conn.commit()


def _columns(
    conn: psycopg2.extensions.connection, table: str
) -> dict[str, tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            """,
            (table,),
        )
        return {name: (kind, nullable) for name, kind, nullable in cur.fetchall()}


def test_the_columns_arrive_nullable_and_keep_an_earlier_row(
    before: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    for table, expected in COLUMNS.items():
        assert not set(expected) & set(_columns(before, table))
    _insert_row(before)
    _apply(ephemeral_postgres, ADD)
    for table, expected in COLUMNS.items():
        columns = _columns(before, table)
        for name, kind in expected.items():
            assert columns[name] == (kind, "YES")
    with before.cursor() as cur:
        cur.execute(
            "SELECT accepted_n, false_pass_count, harness, model, configuration"
            " FROM public.delegation_eval_results"
        )
        assert cur.fetchall() == [(20, 10, None, None, None)]
    before.commit()


def test_a_second_apply_changes_nothing_and_keeps_one_index(
    before: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply(ephemeral_postgres, ADD)
    _insert_row(before)
    _insert_verdict(before, 0.5)
    with before.cursor() as cur:
        cur.execute(
            "UPDATE public.delegation_eval_results SET harness = 'local',"
            " model = 'model-1', configuration = 'default'"
        )
        cur.execute("SELECT * FROM public.delegation_eval_results")
        kept_results = cur.fetchall()
        cur.execute("SELECT * FROM public.delegation_eval_item_verdicts")
        kept_verdicts = cur.fetchall()
    before.commit()
    _apply(ephemeral_postgres, ADD)
    with before.cursor() as cur:
        cur.execute("SELECT * FROM public.delegation_eval_results")
        assert cur.fetchall() == kept_results
        cur.execute("SELECT * FROM public.delegation_eval_item_verdicts")
        assert cur.fetchall() == kept_verdicts
        cur.execute(
            "SELECT COUNT(*) FROM pg_indexes"
            " WHERE schemaname = 'public' AND tablename = 'delegation_eval_results'"
            " AND indexname = 'delegation_eval_results_harness_class_idx'"
        )
        assert cur.fetchone() == (1,)


@pytest.mark.parametrize("operation", ["insert", "update"])
def test_negative_duration_is_refused(
    before: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
    operation: str,
) -> None:
    _apply(ephemeral_postgres, ADD)
    if operation == "update":
        _insert_verdict(before, 0.5)
    try:
        with pytest.raises(psycopg2.errors.CheckViolation):
            if operation == "insert":
                _insert_verdict(before, -1)
            else:
                with before.cursor() as cur:
                    cur.execute(
                        "UPDATE public.delegation_eval_item_verdicts SET duration_s = -1"
                    )
    finally:
        before.rollback()


def test_the_tenant_writer_can_set_the_columns(
    before: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply(ephemeral_postgres, ADD)
    with before.cursor() as cur:
        cur.execute(
            "SELECT has_column_privilege('tenant_projection_writer',"
            " 'public.delegation_eval_results', 'harness', 'UPDATE'),"
            " has_column_privilege('tenant_projection_writer',"
            " 'public.delegation_eval_item_verdicts', 'duration_s', 'UPDATE')"
        )
        assert cur.fetchone() == (True, True)
