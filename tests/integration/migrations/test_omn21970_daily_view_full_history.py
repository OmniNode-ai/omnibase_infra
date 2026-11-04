# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21970 AC1 on a real Postgres: the vendored per-day view serves the whole history, not the newest 500 runs.

The decisions exposure served only its newest 500 rows, so the Overview drew about a day and a half of a longer
history. 0066 (projection_delegation_daily) folds delegation_events into one row per tenant with every UTC day, and
0065 gives each row the integer walk cursor. Each proof builds delegation_events on a throwaway cluster from the
node's real forward chain, 0007 through 0064 (all 75 columns, the correlation_id unique key, the data_source check,
ENABLE + FORCE row level security and the tenant_isolation policy), seeds it, applies 0065 and 0066 the way the
forward runner does (``psql -v ON_ERROR_STOP=1 -f``) and reads the result back. Failure modes, one test each:

  R1 600 runs over 20 days serve 20 days whose runs sum to 600, more than the 500-row cap could hold, with each day's
     passed and failed counts as the view's rules give them, and no unrecorded run (the gate column is NOT NULL);
  R3 every row, existing and inserted after 0065, has a distinct non-null cursor;
  R4 0065 applied a second time succeeds and leaves exactly one cursor sequence (the hostile review asked);
  R5 control through the view itself: with only the newest 500 runs in the source, the same whole-history check
     fails, so R1 is not always-pass;
  R7 after a second apply, the writer role itself inserts from two concurrent transactions that commit in reverse
     order: both succeed through the sequence GRANT 0065 carries, every row gets a distinct cursor, the sequence
     stays owned by the column and the cursor index stays unique;
  R6 security_invoker: the dashboard role reads only its own tenant's days through the view, and nothing at all when
     app.tenant_id is unset (the policy fails closed);
  R8 a day is the UTC day of written_at, or of created_at when written_at is null: runs a second either side of
     midnight UTC land on their own days, read in a session whose TimeZone is not UTC.
"""

from __future__ import annotations

import uuid
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

#: Superseded by omnibase_infra's own 0032 and fenced (OMN-15349), as in the OMN-18159 corpus test.
FENCED = {"0031_delegation_events_tenant_id_to_uuid.sql"}

# Roles the chain grants to but does not create; a lane provisions them elsewhere (forward 103 for the writer).
# LOGIN is the fixture's one addition: R6 and R7 connect as these roles.
ROLES = (
    "DO $$ BEGIN "
    "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_dashboard') "
    "THEN CREATE ROLE app_dashboard LOGIN; END IF; "
    "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tenant_projection_writer') THEN "
    "CREATE ROLE tenant_projection_writer WITH LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE "
    "NOREPLICATION; END IF; "
    "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'omninode_runtime') THEN "
    "CREATE ROLE omninode_runtime WITH NOLOGIN NOSUPERUSER NOBYPASSRLS; END IF; "
    "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'role_omnidash') "
    "THEN CREATE ROLE role_omnidash; END IF; "
    "END$$;"
)
# The writer's table grant, issued by node_projection_delegation_inference_response/0004 on every lane.
WRITER_GRANT = "GRANT SELECT, INSERT, UPDATE ON public.delegation_events TO tenant_projection_writer"


def _outcome(r: int) -> tuple[bool | None, bool]:
    """(terminal_ok, quality_gate_passed) for the r-th run of a day: a mix of every rule the view applies.

    quality_gate_passed is NOT NULL DEFAULT FALSE (0007), so the real table holds no run the view could count as
    unrecorded; R1 asserts that bucket stays 0.
    """
    if r % 10 == 0:
        return False, True  # terminal not ok: failed, whatever the gate said
    if r % 7 == 0:
        return None, False  # terminal never recorded counts as ok; the gate fails it
    if r % 11 == 0:
        return None, True  # terminal never recorded counts as ok: the gate passes it
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
    tenant: str = TENANT,
) -> None:
    rows = [
        (
            tenant,
            str(uuid.uuid4()),
            START + timedelta(days=d, minutes=r),
            *_outcome(r),
            100 + r,
            10 + r,
        )
        for d in range(days)
        for r in range(runs_per_day)
    ]
    with pg.connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO public.delegation_events "
            "(tenant_id, correlation_id, written_at, terminal_ok, quality_gate_passed, "
            "tokens_input, tokens_output) VALUES (%s, %s, %s, %s, %s, %s, %s)",
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
        cur.execute(ROLES)
        # omninode_internal is provisioned by omnibase_infra forward 098 on every lane; 0051 asserts it.
        cur.execute("CREATE SCHEMA IF NOT EXISTS omninode_internal")
        for path in sorted(NODE.glob("*.sql")):
            if path.name >= CURSOR.name:
                break
            if path.name in FENCED:
                continue
            # CONCURRENTLY cannot run in the transaction psycopg2 opens; a property of the driver.
            cur.execute(
                path.read_text(encoding="utf-8").replace(
                    "CREATE INDEX CONCURRENTLY", "CREATE INDEX"
                )
            )
        cur.execute(WRITER_GRANT)
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
    assert unrecorded == 0 and passed and failed, (passed, failed, unrecorded)
    for day in days:
        assert (day["passed"], day["failed"], day["unrecorded"]) == (
            passed,
            failed,
            unrecorded,
        ), day
        assert day["passed"] + day["failed"] + day["unrecorded"] == day["runs"] == 30


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
        "INSERT INTO public.delegation_events "
        "(tenant_id, correlation_id, written_at, terminal_ok, quality_gate_passed) "
        "VALUES (%s, %s, %s, true, true)"
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
                a.execute(
                    insert,
                    (TENANT, str(uuid.uuid4()), START + timedelta(hours=1, minutes=n)),
                )
                b.execute(
                    insert,
                    (TENANT, str(uuid.uuid4()), START + timedelta(hours=2, minutes=n)),
                )
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


def test_r8_days_are_utc_days_of_written_at_or_created_at(
    cluster: EphemeralPostgres,
) -> None:
    edge = datetime(2026, 10, 2, tzinfo=UTC)
    rows = [
        (
            str(uuid.uuid4()),
            edge - timedelta(seconds=1),
            edge - timedelta(seconds=1),
        ),  # 10-01, last second
        (str(uuid.uuid4()), edge, edge),  # 10-02, first second
        # Written at 10-03 23:30Z but created on 10-02: the day is written_at's.
        (
            str(uuid.uuid4()),
            edge + timedelta(days=1, hours=23, minutes=30),
            edge + timedelta(hours=1),
        ),
        (
            str(uuid.uuid4()),
            None,
            edge + timedelta(days=3, hours=12),
        ),  # never written: created_at's day, 10-05
    ]
    with cluster.connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO public.delegation_events (tenant_id, correlation_id, written_at, created_at, "
            "terminal_ok, quality_gate_passed) VALUES (%s, %s, %s, %s, true, true)",
            [(TENANT, c, w, k) for c, w, k in rows],
        )
    _apply(cluster, CURSOR)
    _apply(cluster, DAILY)
    with cluster.connect() as conn, conn.cursor() as cur:
        # +05:30: in this zone the 10-01 23:59:59Z run is already 10-02 and the 10-03 23:30Z run is 10-04.
        cur.execute("SET TIME ZONE 'Asia/Kolkata'")
        cur.execute(
            "SELECT days FROM public.projection_delegation_daily WHERE tenant_id = %s",
            (TENANT,),
        )
        (days,) = cur.fetchone()
    assert [(d["day"], d["runs"]) for d in days] == [
        ("2026-10-01", 1),
        ("2026-10-02", 1),
        ("2026-10-03", 1),
        ("2026-10-05", 1),
    ], days
