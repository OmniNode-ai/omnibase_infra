# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-apply proof for the gateway_link_health_status verdict logic (OMN-15570).

These tests apply ``100_create_gateway_link_health.sql`` to a throwaway
Postgres cluster and read ``health_status`` back out of the real view. They
exist because the verdict is SQL, not Python: a unit test that greps the
migration text proves the string is present, not that the CASE arms evaluate
the way the ticket requires.

The specific regression they lock down: before OMN-15742/G2 the heartbeat had
no self-reported status, so health was derived from recency alone. Once G2
added ``status``, an edge could heartbeat perfectly on schedule while
reporting itself degraded — and a recency-only view scored that HEALTHY. A
health surface that renders a self-reported-degraded edge as HEALTHY is worse
than no health surface, because the wrong answer is indistinguishable from
the right one. ``test_degraded_self_report_beats_recency`` is the arm that
must never regress.

Precedence under test (stale > self-reported > lag > healthy) is asserted
directly, including the ambiguous case where two arms are simultaneously true.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
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
    / "100_create_gateway_link_health.sql"
)
ROLLBACK = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_100_create_gateway_link_health.sql"
)

# Matches the migration's hardcoded silence window. Anything older than this
# is stale; the drift guard in tests/unit/db/test_migration_100.py is what
# keeps that literal pinned to the contract.
SILENCE_WINDOW = timedelta(seconds=60)


@pytest.fixture
def applied(
    ephemeral_postgres: EphemeralPostgres,
) -> Iterator[psycopg2.extensions.connection]:
    """Apply the real forward migration, yield an open connection."""
    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(FORWARD))
    assert result.returncode == 0, result.stderr
    conn = ephemeral_postgres.connect()
    try:
        yield conn
    finally:
        conn.close()


def _upsert(
    conn: psycopg2.extensions.connection,
    *,
    tenant_id: str,
    age: timedelta,
    reported_status: str = "active",
    consecutive_failures: int = 0,
    lag_messages: int | None = None,
    lag_seconds: float | None = None,
) -> None:
    """Write one row, with last_seen_at placed `age` in the past."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.gateway_link_health (
                tenant_id, principal_id, local_transport_flavor, last_seen_at,
                reported_status, consecutive_failures, lag_messages, lag_seconds
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id) DO UPDATE SET
                last_seen_at         = EXCLUDED.last_seen_at,
                reported_status      = EXCLUDED.reported_status,
                consecutive_failures = EXCLUDED.consecutive_failures,
                lag_messages         = EXCLUDED.lag_messages,
                lag_seconds          = EXCLUDED.lag_seconds
            """,
            (
                tenant_id,
                f"t-{tenant_id}",
                "containerized",
                datetime.now(UTC) - age,
                reported_status,
                consecutive_failures,
                lag_messages,
                lag_seconds,
            ),
        )
    conn.commit()


def _verdict(conn: psycopg2.extensions.connection, tenant_id: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT health_status FROM public.gateway_link_health_status "
            "WHERE tenant_id = %s",
            (tenant_id,),
        )
        row = cur.fetchone()
    assert row is not None, f"no projection row for {tenant_id}"
    return str(row[0])


def test_fresh_active_edge_is_healthy(
    applied: psycopg2.extensions.connection,
) -> None:
    _upsert(applied, tenant_id="fresh", age=timedelta(seconds=5))

    assert _verdict(applied, "fresh") == "HEALTHY"


def test_silent_edge_is_unhealthy(
    applied: psycopg2.extensions.connection,
) -> None:
    """Absence of progress is visible as a stale row, not a missing one."""
    _upsert(applied, tenant_id="silent", age=SILENCE_WINDOW + timedelta(seconds=30))

    assert _verdict(applied, "silent") == "UNHEALTHY"

    with applied.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.gateway_link_health_status")
        assert cur.fetchone()[0] == 1, "the stale row must still be present"


def test_degraded_self_report_beats_recency(
    applied: psycopg2.extensions.connection,
) -> None:
    """THE regression this file exists for.

    The edge is heartbeating well inside the silence window — recency alone
    would score it HEALTHY — but it reports itself degraded. It must not read
    as healthy.
    """
    _upsert(
        applied,
        tenant_id="degraded",
        age=timedelta(seconds=2),
        reported_status="degraded",
        consecutive_failures=3,
    )

    assert _verdict(applied, "degraded") == "DEGRADED_SELF_REPORTED"


def test_unrecognised_status_is_not_scored_healthy(
    applied: psycopg2.extensions.connection,
) -> None:
    """A status the producer adds later must fail toward not-healthy.

    The column is free TEXT precisely so an unknown value cannot crash the
    projection; the view's `<> 'active'` test is what stops it being read as
    healthy by omission.
    """
    _upsert(
        applied,
        tenant_id="draining",
        age=timedelta(seconds=2),
        reported_status="draining",
    )

    assert _verdict(applied, "draining") == "DEGRADED_SELF_REPORTED"


def test_stale_outranks_degraded_when_both_are_true(
    applied: psycopg2.extensions.connection,
) -> None:
    """Deterministic precedence for simultaneous signals.

    A stale edge's last self-report is itself stale, so staleness wins. The
    point is that the answer is fixed, not that either verdict is nicer.
    """
    _upsert(
        applied,
        tenant_id="both",
        age=SILENCE_WINDOW + timedelta(seconds=30),
        reported_status="degraded",
        consecutive_failures=9,
    )

    assert _verdict(applied, "both") == "UNHEALTHY"


def test_self_report_outranks_lag_when_both_are_true(
    applied: psycopg2.extensions.connection,
) -> None:
    """A first-party statement outranks a metric inferred about the edge."""
    _upsert(
        applied,
        tenant_id="degraded-and-lagging",
        age=timedelta(seconds=2),
        reported_status="degraded",
        lag_messages=5_000,
    )

    assert _verdict(applied, "degraded-and-lagging") == "DEGRADED_SELF_REPORTED"


def test_lag_breach_still_reported_for_an_active_edge(
    applied: psycopg2.extensions.connection,
) -> None:
    """The lag arms stay reachable — adding the status arm did not shadow them."""
    _upsert(
        applied,
        tenant_id="lagging",
        age=timedelta(seconds=2),
        lag_messages=5_000,
    )

    assert _verdict(applied, "lagging") == "DEGRADED_LAG"


def test_null_lag_never_produces_a_verdict(
    applied: psycopg2.extensions.connection,
) -> None:
    """NULL lag is neither TRUE nor FALSE in SQL — it must not decide anything."""
    _upsert(applied, tenant_id="nolag", age=timedelta(seconds=2))

    assert _verdict(applied, "nolag") == "HEALTHY"


def test_recovery_flips_the_verdict_back(
    applied: psycopg2.extensions.connection,
) -> None:
    """Degraded is not sticky: the next healthy heartbeat clears it.

    Same row throughout — this is a latest-known-state projection, so recovery
    is an UPDATE of the one row, never a second row or a delete.
    """
    _upsert(
        applied,
        tenant_id="recovers",
        age=timedelta(seconds=2),
        reported_status="degraded",
        consecutive_failures=4,
    )
    assert _verdict(applied, "recovers") == "DEGRADED_SELF_REPORTED"

    _upsert(
        applied,
        tenant_id="recovers",
        age=timedelta(seconds=1),
        reported_status="active",
        consecutive_failures=0,
    )

    assert _verdict(applied, "recovers") == "HEALTHY"
    with applied.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM public.gateway_link_health WHERE tenant_id = %s",
            ("recovers",),
        )
        assert cur.fetchone()[0] == 1, "recovery must update in place, not insert"


def test_silence_then_return_flips_back_from_unhealthy(
    applied: psycopg2.extensions.connection,
) -> None:
    """An edge that goes quiet and comes back is HEALTHY again, same row."""
    _upsert(
        applied,
        tenant_id="returns",
        age=SILENCE_WINDOW + timedelta(seconds=30),
    )
    assert _verdict(applied, "returns") == "UNHEALTHY"

    _upsert(applied, tenant_id="returns", age=timedelta(seconds=1))

    assert _verdict(applied, "returns") == "HEALTHY"


def test_consecutive_failures_is_recorded_but_drives_no_verdict(
    applied: psycopg2.extensions.connection,
) -> None:
    """Evidence, not an input.

    The producer already folds the failure count into `status`; re-deriving a
    verdict from it here would let the two disagree about the same edge.
    """
    _upsert(
        applied,
        tenant_id="blips",
        age=timedelta(seconds=2),
        reported_status="active",
        consecutive_failures=99,
    )

    assert _verdict(applied, "blips") == "HEALTHY"
    with applied.cursor() as cur:
        cur.execute(
            "SELECT consecutive_failures FROM public.gateway_link_health_status "
            "WHERE tenant_id = %s",
            ("blips",),
        )
        assert cur.fetchone()[0] == 99, "the evidence must still be readable"


def test_forward_migration_is_idempotent(
    ephemeral_postgres: EphemeralPostgres,
    applied: psycopg2.extensions.connection,
) -> None:
    """Re-applying must not fail — the runner may replay it."""
    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(FORWARD))

    assert result.returncode == 0, result.stderr


def test_rollback_removes_both_objects(
    ephemeral_postgres: EphemeralPostgres,
    applied: psycopg2.extensions.connection,
) -> None:
    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(ROLLBACK))
    assert result.returncode == 0, result.stderr

    with applied.cursor() as cur:
        cur.execute("SELECT to_regclass('public.gateway_link_health')")
        assert cur.fetchone()[0] is None
        cur.execute("SELECT to_regclass('public.gateway_link_health_status')")
        assert cur.fetchone()[0] is None
