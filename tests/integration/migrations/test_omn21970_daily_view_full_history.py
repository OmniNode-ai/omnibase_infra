# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21970 AC1 on a real Postgres: the vendored per-day view serves the whole history, not the newest 500 runs.

The decisions exposure served only its newest 500 rows, so the Overview drew about a day and a half of a longer
history. 0066 (projection_delegation_daily) folds delegation_events into one row per tenant with every UTC day, and
0065 gives each row the integer walk cursor. These proofs apply both files the way the forward runner does
(``psql -v ON_ERROR_STOP=1 -f``) to a throwaway cluster whose delegation_events carries the production tenant
policy (0023/0032: ENABLE + FORCE row level security, tenant_isolation on app.tenant_id), and read the result back.
Failure modes, one test each:

  R1 600 runs over 20 days serve 20 days whose runs sum to 600, more than the 500-row cap could hold, with each day's
     passed, failed and unrecorded counts as the view's rules give them;
  R2 a day whose runs carry no token counts is served as null tokens, never 0;
  R3 every row, existing and inserted after 0065, has a distinct non-null cursor;
  R4 0065 applied a second time succeeds and leaves exactly one cursor sequence (the hostile review asked);
  R5 control through the view itself: with only the newest 500 runs in the source, the same whole-history check
     fails, so R1 is not always-pass;
  R7 after a second apply, the writer role itself inserts from two concurrent transactions that commit in reverse
     order: both succeed through the sequence GRANT 0065 carries, every row gets a distinct cursor, the sequence
     stays owned by the column and the cursor index stays unique;
  R6 security_invoker: the dashboard role reads only its own tenant's days through the view, and nothing at all when
     app.tenant_id is unset (the policy fails closed).
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
OTHER = "00000000-0000-0000-0000-0000000021c1"
START = datetime(2026, 10, 1, tzinfo=UTC)

# The columns 0065 and 0066 read, the roles their grants name, and the production tenant policy (0023, 0032).
SCHEMA = """
CREATE ROLE tenant_projection_writer LOGIN;
CREATE ROLE app_dashboard LOGIN;
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
GRANT SELECT ON public.delegation_events TO app_dashboard;
GRANT SELECT, INSERT ON public.delegation_events TO tenant_projection_writer;
ALTER TABLE public.delegation_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.delegation_events FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON public.delegation_events
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);
"""


def _outcome(r: int) -> tuple[bool | None, bool | None]:
    """(terminal_ok, quality_gate_passed) for the r-th run of a day: a mix of every rule the view applies."""
    if r % 10 == 0:
        return False, True  # terminal not ok: failed, whatever the gate said
    if r % 7 == 0:
        return True, None  # gate never recorded: unrecorded
    return True, r % 3 != 0  # the gate decides


def _expected(runs_per_day: int) -> tuple[int, int, int]:
    passed = failed = unrecorded = 0
    for r in range(runs_per_day):
        terminal, gate = _outcome(r)
        if terminal is False or gate is False:
            failed += 1
        elif gate is True:
            passed += 1
        else:
            unrecorded += 1
    return passed, failed, unrecorded


def _apply(pg: EphemeralPostgres, path: Path) -> None:
    out = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(path))
    assert out.returncode == 0, f"{path.name} failed: {out.stderr}"


def _seed(
    pg: EphemeralPostgres,
    runs_per_day: int,
    days: int,
    tokenless_day: int | None = None,
    tenant: str = TENANT,
) -> None:
    rows = [
        (
            tenant,
            START + timedelta(days=d, minutes=r),
            *_outcome(r),
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


def _served(pg: EphemeralPostgres, tenant: str = TENANT) -> tuple[list[dict], int]:
    with pg.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT days, total_runs FROM public.projection_delegation_daily WHERE tenant_id = %s",
            (tenant,),
        )
        row = cur.fetchone()
    assert row is not None, "the view serves no row for the tenant"
    return row[0], row[1]


def _whole_history(
    days: list[dict], total: int, want_days: int, want_runs: int
) -> bool:
    return len(days) == want_days and sum(d["runs"] for d in days) == want_runs == total


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
    assert _whole_history(days, total, 20, 600), (len(days), total)
    assert [d["day"] for d in days] == [
        (START + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(20)
    ]
    passed, failed, unrecorded = _expected(30)
    for day in days:
        assert (day["passed"], day["failed"], day["unrecorded"]) == (
            passed,
            failed,
            unrecorded,
        ), day
        assert day["passed"] + day["failed"] + day["unrecorded"] == day["runs"] == 30


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
            "SELECT count(*), count(projection_cursor), count(DISTINCT projection_cursor) "
            "FROM public.delegation_events"
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
            "SELECT count(*) FROM pg_class "
            "WHERE relkind = 'S' AND relname LIKE 'delegation_events_projection_cursor%'"
        )
        (sequences,) = cur.fetchone()
    assert sequences == 1, f"{sequences} cursor sequences after a second apply"


def test_r5_control_a_capped_source_fails_the_whole_history_check(
    cluster: EphemeralPostgres,
) -> None:
    _seed(cluster, runs_per_day=30, days=20)
    _apply(cluster, CURSOR)
    _apply(cluster, DAILY)
    with cluster.connect() as conn, conn.cursor() as cur:
        # What a 500-row read leaves: only the newest 500 runs in the source the view folds.
        cur.execute(
            "DELETE FROM public.delegation_events WHERE id NOT IN ("
            "SELECT id FROM public.delegation_events ORDER BY written_at DESC LIMIT 500)"
        )
    days, total = _served(cluster)
    assert total == 500 and len(days) < 20, (len(days), total)
    assert not _whole_history(days, total, 20, 600), (
        "the check must fail on a capped source"
    )


def test_r6_the_dashboard_reads_only_its_own_tenant(cluster: EphemeralPostgres) -> None:
    _seed(cluster, runs_per_day=4, days=3, tenant=TENANT)
    _seed(cluster, runs_per_day=6, days=5, tenant=OTHER)
    _apply(cluster, CURSOR)
    _apply(cluster, DAILY)
    with cluster.connect(user="app_dashboard") as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT tenant_id::text, total_runs FROM public.projection_delegation_daily"
        )
        unset = cur.fetchall()
        cur.execute("SELECT set_config('app.tenant_id', %s, false)", (TENANT,))
        cur.execute(
            "SELECT tenant_id::text, total_runs FROM public.projection_delegation_daily"
        )
        own = cur.fetchall()
    assert unset == [], f"with no tenant set the policy must show nothing, saw {unset}"
    assert own == [(TENANT, 12)], f"tenant {TENANT} saw {own}"


def test_r7_concurrent_writer_inserts_after_a_second_apply(
    cluster: EphemeralPostgres,
) -> None:
    _seed(cluster, runs_per_day=3, days=2)
    _apply(cluster, CURSOR)
    _apply(cluster, CURSOR)
    insert = (
        "INSERT INTO public.delegation_events (tenant_id, written_at, terminal_ok, quality_gate_passed) "
        "VALUES (%s, %s, true, true)"
    )
    first = cluster.connect(user="tenant_projection_writer")
    second = cluster.connect(user="tenant_projection_writer")
    try:
        with first.cursor() as a, second.cursor() as b:
            for cur in (a, b):
                cur.execute("SELECT set_config('app.tenant_id', %s, false)", (TENANT,))
            first.commit()
            second.commit()
            # Interleaved and still open: the first transaction takes the lower numbers but commits last.
            for n in range(5):
                a.execute(insert, (TENANT, START + timedelta(hours=1, minutes=n)))
                b.execute(insert, (TENANT, START + timedelta(hours=2, minutes=n)))
            second.commit()
            first.commit()
    finally:
        first.close()
        second.close()
    with cluster.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), count(projection_cursor), count(DISTINCT projection_cursor) "
            "FROM public.delegation_events"
        )
        rows, with_cursor, distinct = cur.fetchone()
    assert rows == 16 and with_cursor == 16 and distinct == 16, (
        rows,
        with_cursor,
        distinct,
    )
    with cluster.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT pg_get_serial_sequence('public.delegation_events', 'projection_cursor')"
        )
        (owned,) = cur.fetchone()
        cur.execute(
            "SELECT indisunique FROM pg_index "
            "WHERE indexrelid = 'public.idx_delegation_events_projection_cursor'::regclass"
        )
        (unique,) = cur.fetchone()
    assert owned == "public.delegation_events_projection_cursor_seq", owned
    assert unique is True, "the cursor index must stay unique after a second apply"
