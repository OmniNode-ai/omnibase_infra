# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21970 AC1 on a real Postgres: the vendored per-day view serves the whole history, not the newest 500 runs.

The decisions exposure served only its newest 500 rows, so the Overview drew about a day and a half of a longer
history. 0066 (projection_delegation_daily) folds delegation_events into one row per tenant with every UTC day, and
0065 gives each row the integer walk cursor. These proofs apply both files the way the forward runner does
(``psql -v ON_ERROR_STOP=1 -f``) to a throwaway cluster and read the result back. Failure modes, one test each:

  R1 600 runs over 20 days serve 20 days whose runs sum to 600, more than the 500-row cap could hold;
  R2 a day whose runs carry no token counts is served as null tokens, never 0;
  R3 every row, existing and inserted after 0065, has a distinct non-null cursor;
  R4 0065 applied a second time succeeds and leaves exactly one cursor sequence (the hostile review asked);
  R5 control: the same sum check fails on a capped 500-row source, so R1 is not always-pass.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = pytest.mark.integration

NODE = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/nodes/node_projection_delegation"
)
CURSOR = NODE / "0065_delegation_events_projection_cursor.sql"
DAILY = NODE / "0066_delegation_daily_view.sql"
TENANT = "00000000-0000-0000-0000-0000000021c0"
START = datetime(2026, 10, 1, tzinfo=UTC)

# The columns 0065 and 0066 read, as delegation_events carries them; the roles their grants name.
SCHEMA = """
CREATE ROLE tenant_projection_writer;
CREATE ROLE app_dashboard;
CREATE TABLE public.delegation_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    written_at timestamptz,
    terminal_ok boolean,
    quality_gate_passed boolean,
    tokens_input bigint,
    tokens_output bigint
);
"""


def _apply(pg: EphemeralPostgres, path: Path) -> None:
    out = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(path))
    assert out.returncode == 0, f"{path.name} failed: {out.stderr}"


def _seed(
    pg: EphemeralPostgres,
    runs_per_day: int,
    days: int,
    tokenless_day: int | None = None,
) -> None:
    rows = [
        (
            TENANT,
            START + timedelta(days=d, minutes=r),
            True,
            r % 5 != 0,
            None if d == tokenless_day else 100 + r,
            None if d == tokenless_day else 10 + r,
        )
        for d in range(days)
        for r in range(runs_per_day)
    ]
    with pg.connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO public.delegation_events "
            "(tenant_id, written_at, terminal_ok, quality_gate_passed, tokens_input, tokens_output) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            rows,
        )


def _served(pg: EphemeralPostgres) -> tuple[list[dict], int]:
    with pg.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT days, total_runs FROM public.projection_delegation_daily WHERE tenant_id = %s",
            (TENANT,),
        )
        row = cur.fetchone()
    assert row is not None, "the view serves no row for the tenant"
    return row[0], row[1]


@pytest.fixture
def cluster(ephemeral_postgres: EphemeralPostgres) -> EphemeralPostgres:
    with ephemeral_postgres.connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
    return ephemeral_postgres


def test_r1_600_runs_over_20_days_are_served_whole(cluster: EphemeralPostgres) -> None:
    _seed(cluster, runs_per_day=30, days=20)
    _apply(cluster, CURSOR)
    _apply(cluster, DAILY)
    days, total = _served(cluster)
    assert len(days) == 20, f"served {len(days)} days, wanted all 20"
    assert sum(day["runs"] for day in days) == 600 == total
    assert [day["day"] for day in days] == [
        (START + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(20)
    ]


def test_r2_a_tokenless_day_is_null_never_zero(cluster: EphemeralPostgres) -> None:
    _seed(cluster, runs_per_day=30, days=20, tokenless_day=7)
    _apply(cluster, CURSOR)
    _apply(cluster, DAILY)
    days, _ = _served(cluster)
    tokenless = days[7]
    assert tokenless["tokensIn"] is None and tokenless["tokensOut"] is None, tokenless
    assert tokenless["tokensInUnmeasured"] == 30
    assert all(day["tokensIn"] is not None for i, day in enumerate(days) if i != 7)


def test_r3_every_row_has_a_distinct_cursor(cluster: EphemeralPostgres) -> None:
    _seed(cluster, runs_per_day=30, days=10)
    _apply(cluster, CURSOR)
    _seed(cluster, runs_per_day=5, days=2)
    with cluster.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), count(projection_cursor), count(DISTINCT projection_cursor) FROM public.delegation_events"
        )
        rows, with_cursor, distinct = cur.fetchone()
    assert rows == 310 and with_cursor == 310 and distinct == 310, (
        rows,
        with_cursor,
        distinct,
    )


def test_r4_a_second_apply_succeeds_and_leaves_one_sequence(
    cluster: EphemeralPostgres,
) -> None:
    _seed(cluster, runs_per_day=3, days=2)
    _apply(cluster, CURSOR)
    _apply(cluster, CURSOR)
    with cluster.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_class WHERE relkind = 'S' AND relname LIKE 'delegation_events_projection_cursor%'"
        )
        (sequences,) = cur.fetchone()
    assert sequences == 1, f"{sequences} cursor sequences after a second apply"


def test_r5_control_a_capped_source_fails_the_sum(cluster: EphemeralPostgres) -> None:
    _seed(cluster, runs_per_day=30, days=20)
    _apply(cluster, CURSOR)
    _apply(cluster, DAILY)
    with cluster.connect() as conn, conn.cursor() as cur:
        # The pre-fix read: the newest 500 rows, grouped the way the Overview drew them.
        cur.execute(
            "SELECT count(DISTINCT (written_at AT TIME ZONE 'UTC')::date), count(*) FROM ("
            "SELECT written_at FROM public.delegation_events ORDER BY written_at DESC LIMIT 500) newest"
        )
        capped_days, capped_runs = cur.fetchone()
    assert (capped_days, capped_runs) != (20, 600), (
        "the capped read must not pass the whole-history check"
    )
    assert capped_runs == 500 and capped_days < 20
