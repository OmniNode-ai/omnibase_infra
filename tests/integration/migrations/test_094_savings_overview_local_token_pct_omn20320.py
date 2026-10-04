# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The savings overview measures ``local_token_pct`` (OMN-20320).

``projection_cost_savings_overview.local_token_pct`` was the literal
``0::float`` through 093, so the dashboard read 0% local while
``projection_delegation_savings_series`` -- over the same ``delegation_events``
-- read 0.88 to 0.98 local share. 094 computes it from ``cost_tier_name`` with
the series' own tier bucketing.

The proof is a pair on one corpus: stopped at 093 the view reads 0.0 for rows
that are three-quarters local by tokens; with 094 it reads 0.75.
"""

from __future__ import annotations

import uuid

import psycopg2
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres
from tests.integration.migrations.test_088_savings_views_invoker_scoped_omn18159 import (
    SAVINGS,
    _apply,
)

PREVIOUS_MIGRATION = "093_savings_series_persisted_baseline.sql"
TENANT = "820272f9-4aaf-5add-a2df-0af942852ab2"

#: (cost_tier_name, tokens_input, tokens_output); '' is the untiered value. 300 local of 400 tiered tokens;
#: the untiered row is outside the denominator, as it is in the series.
_RUNS = (
    ("local", 100, 100),
    ("local", 50, 50),
    ("claude", 60, 40),
    ("cheap_cloud", 0, 0),
    ("", 500, 500),
)


def _seed(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        for tier, tokens_in, tokens_out in _RUNS:
            run = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO public.delegation_events "
                "(correlation_id, session_id, tenant_id, cost_tier_name, "
                "tokens_input, tokens_output, cost_usd, cost_savings_usd, "
                "cost_measurement_source) "
                "VALUES (%s, %s, %s, %s, %s, %s, 0, 0, 'metered')",
                (run, run, TENANT, tier, tokens_in, tokens_out),
            )


def _overview(conn: psycopg2.extensions.connection) -> tuple[float, list[str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT local_token_pct, warnings "
            "FROM public.projection_cost_savings_overview WHERE tenant_id = %s",
            (TENANT,),
        )
        row = cur.fetchone()
    assert row is not None, "positive control: the seeded tenant must have a row"
    return row[0], row[1]


def _database(pg: EphemeralPostgres, name: str) -> psycopg2.extensions.connection:
    admin = pg.connect(dbname="postgres")
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE {name}")
    finally:
        admin.close()
    conn = pg.connect(dbname=name)
    conn.autocommit = True
    return conn


@pytest.mark.integration
def test_094_measures_local_token_pct_and_093_reported_zero(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    conn = _database(ephemeral_postgres, "before_094")
    try:
        _apply(conn, stop_after=PREVIOUS_MIGRATION)
        _seed(conn)
        before, _ = _overview(conn)
    finally:
        conn.close()
    assert before == 0.0, "093 hardcodes the zero, or 094 proves nothing"

    conn = _database(ephemeral_postgres, "after_094")
    try:
        _apply(conn, stop_after=None)
        _seed(conn)
        after, warnings = _overview(conn)
        assert after == pytest.approx(0.75)
        assert not any("unmeasured" in w for w in warnings)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT option_value FROM pg_options_to_table("
                "(SELECT reloptions FROM pg_class WHERE oid = "
                "'public.projection_cost_savings_overview'::regclass)) "
                "WHERE option_name = 'security_invoker'"
            )
            assert cur.fetchone() == ("true",)
    finally:
        conn.close()


@pytest.mark.integration
def test_094_reports_unmeasured_when_no_run_carries_a_tier(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    conn = _database(ephemeral_postgres, "untiered_094")
    try:
        _apply(conn, stop_after=None)
        with conn.cursor() as cur:
            run = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO public.delegation_events "
                "(correlation_id, session_id, tenant_id, cost_tier_name, "
                "tokens_input, tokens_output, cost_usd, cost_savings_usd, "
                "cost_measurement_source) "
                "VALUES (%s, %s, %s, '', 10, 10, 0, 0, 'metered')",
                (run, run, TENANT),
            )
        pct, warnings = _overview(conn)
    finally:
        conn.close()
    assert pct == 0.0
    assert any("unmeasured" in w for w in warnings)


def test_094_is_the_last_savings_migration() -> None:
    names = sorted(p.name for p in SAVINGS.glob("09*.sql"))
    # Namespaced migrations are keyed by the full filename. OMN-20008 adds a
    # forward 094 successor without editing the applied local-share migration.
    assert names[-1].startswith("094_savings_overview_")
