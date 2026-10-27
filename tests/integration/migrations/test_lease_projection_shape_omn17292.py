# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Apply the lease migration to fresh and drifted PostgreSQL tables."""

from pathlib import Path

import psycopg2
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = pytest.mark.integration

MIGRATION = (
    Path(__file__).parents[3]
    / "docker/migrations/forward/nodes/node_projection_lease"
    / "0000_create_lease_projection.sql"
)


@pytest.mark.parametrize("drifted", [False, True], ids=["fresh", "drifted"])
def test_lease_projection_converges_and_preserves_constraints(
    ephemeral_postgres: EphemeralPostgres, drifted: bool
) -> None:
    pg = ephemeral_postgres
    setup = "CREATE ROLE tenant_projection_writer;"
    if drifted:
        setup += "CREATE TABLE public.lease_projection (kind TEXT NOT NULL);"
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-c", setup)
    assert result.returncode == 0, result.stderr

    # Reapplying the production file must be safe on both paths.
    for _ in range(2):
        result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION))
        assert result.returncode == 0, result.stderr

    with pg.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'lease_projection' "
            "ORDER BY ordinal_position"
        )
        assert cur.fetchall() == [
            (name, "NO")
            for name in (
                "kind",
                "resource",
                "lease_id",
                "fence",
                "holder",
                "state",
                "updated_at",
            )
        ]
        cur.execute(
            "SELECT contype, pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = 'public.lease_projection'::regclass ORDER BY contype, conname"
        )
        constraints = cur.fetchall()
        assert len(constraints) == 3, constraints
        assert ("p", "PRIMARY KEY (kind, resource)") in constraints
        assert sum(kind == "c" for kind, _ in constraints) == 2
        cur.execute("SET ROLE tenant_projection_writer")
        cur.execute(
            "INSERT INTO public.lease_projection "
            "(kind, resource, lease_id, fence, holder, state) "
            "VALUES ('exclusive', 'r', '00000000-0000-0000-0000-000000000001', "
            "1, 'holder', 'held') "
            "ON CONFLICT (kind, resource) DO UPDATE SET fence = EXCLUDED.fence"
        )
        cur.execute("UPDATE public.lease_projection SET fence = 2 WHERE resource = 'r'")
        cur.execute("SELECT * FROM public.lease_projection")
        before = cur.fetchall()
        assert before[0][3] == 2

    result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION))
    assert result.returncode == 0, result.stderr
    with pg.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM public.lease_projection")
        assert cur.fetchall() == before

    for fence, state in ((0, "held"), (1, "invalid")):
        with (
            pytest.raises(psycopg2.errors.CheckViolation),
            pg.connect() as conn,
            conn.cursor() as cur,
        ):
            cur.execute(
                "INSERT INTO public.lease_projection "
                "(kind, resource, lease_id, fence, holder, state) "
                "VALUES ('exclusive', 'invalid', "
                "'00000000-0000-0000-0000-000000000002', %s, 'h', %s)",
                (fence, state),
            )


def test_incomplete_populated_projection_requires_a_data_ruling(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    result = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE ROLE tenant_projection_writer; "
        "CREATE TABLE public.lease_projection (kind TEXT NOT NULL); "
        "INSERT INTO public.lease_projection VALUES ('keep-me');",
    )
    assert result.returncode == 0, result.stderr
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION))
    assert result.returncode != 0, "migration invented a missing lease identity"
    assert 'column "resource"' in result.stderr
    assert "contains null values" in result.stderr
    with pg.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT kind FROM public.lease_projection")
        assert cur.fetchall() == [("keep-me",)]
