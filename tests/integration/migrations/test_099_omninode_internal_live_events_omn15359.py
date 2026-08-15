# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-15359 -- physically create omninode_internal.live_events.

THE GAP (live evidence, mergesweep-0809-projplane ground phase + adversarial
verify, rolling ledger 2026-08-09)

    `098_create_omninode_internal_schema.sql` created the empty
    `omninode_internal` schema. `node_projection_live_events/contract.yaml`
    declares `db_io.db_tables[0].schema: omninode_internal`, and
    `handler_wiring._resolve_projection_database_target` uses that
    contract-declared schema literally as the SQL write target -- so the
    runtime has been issuing `INSERT INTO omninode_internal.live_events`
    since before 098 merged. No migration had ever physically created that
    relation, so every insert failed `UndefinedTable`
    (`relation "omninode_internal.live_events" does not exist`), live-
    reconfirmed on onex-dev at ~10s cadence.

This proof applies `099_create_omninode_internal_live_events.sql` through the
same `psql -v ON_ERROR_STOP=1 -f` invocation `scripts/run-forward-migrations.sh`
uses (matching `test_098_omninode_internal_schema_omn15359.py`), against a
real ephemeral cluster seeded with representative pre-existing
`public.live_events` rows, and asserts real catalog + row-level state -- not
string matching the SQL text.

OMN-15838 AMENDMENT: 099 originally also transform-copied every row from
`public.live_events` into `omninode_internal.live_events` in the same file,
then RAISE EXCEPTIONed unless the two tables' row counts matched exactly.
That reconciliation was a non-atomic race on any lane with a concurrent
writer to `public.live_events` (the stability-test lane runs at ~24
writes/min): the INSERT...SELECT snapshots the source, then two separate
follow-up `SELECT count(*)` statements re-read both tables, so any row
committed to the source in between makes the counts diverge and the DO block
RAISE deterministically -- not intermittently. Because
`scripts/run-forward-migrations.sh` runs with `ON_ERROR_STOP=1`, that RAISE
aborted the whole migration run before it was ever recorded applied, so every
subsequent refresh retried and failed identically (963965 vs 963977 rows,
OMN-15838). Data delivery for this table is independently owned by the
node-owned replacement migration
(`docker/migrations/forward/nodes/node_projection_live_events/
0002_create_omninode_internal_live_events.sql`, OMN-15819), so 099's own copy
of that logic was also redundant on top of being racy. 099 now only creates
schema shape (table, grants, indexes) and asserts it -- it never reads
`public.live_events` at all. The row-copy/idempotent-copy/reconciliation
tests that covered the removed logic are replaced below by tests proving the
removed behavior stays removed: 099 no longer copies rows, and a diverging
public/internal row count no longer fails the migration.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
ROLLBACK_DIR = REPO_ROOT / "docker" / "migrations" / "rollback"
SCHEMA_MIGRATION = FORWARD_DIR / "098_create_omninode_internal_schema.sql"
MIGRATION = FORWARD_DIR / "099_create_omninode_internal_live_events.sql"
ROLLBACK = ROLLBACK_DIR / "rollback_099_create_omninode_internal_live_events.sql"
ANALYTICS_DB = "omnidash_analytics"


def _apply(
    pg: EphemeralPostgres, path: Path, *, dbname: str = "postgres"
) -> subprocess.CompletedProcess[str]:
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(path), dbname=dbname)
    return result


def _apply_ok(
    pg: EphemeralPostgres, path: Path, *, dbname: str = "postgres"
) -> subprocess.CompletedProcess[str]:
    result = _apply(pg, path, dbname=dbname)
    assert result.returncode == 0, (
        f"{path.name} failed to apply against {dbname}:\n{result.stderr}"
    )
    return result


def _seed_analytics_database_with_source_table(pg: EphemeralPostgres) -> None:
    """Build omnidash_analytics with a public.live_events shaped and seeded
    exactly like the live table (0000_create_live_events.sql shape)."""
    bootstrap = pg.connect()
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{ANALYTICS_DB}"')
    bootstrap.close()

    conn = pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cur.execute(
            """
            CREATE TABLE public.live_events (
              id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
              event_id       TEXT        UNIQUE NOT NULL,
              type           TEXT        NOT NULL DEFAULT 'ACTION',
              timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              source         TEXT        NOT NULL DEFAULT 'platform',
              topic          TEXT        NOT NULL DEFAULT '',
              summary        TEXT        NOT NULL DEFAULT '',
              payload        TEXT        NOT NULL DEFAULT '{}',
              correlation_id TEXT,
              created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        for i in range(25):
            cur.execute(
                """
                INSERT INTO public.live_events
                  (event_id, type, source, topic, summary, payload, correlation_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    f"evt-{i:04d}-{uuid.uuid4()}",
                    "ACTION" if i % 2 == 0 else "ERROR",
                    "platform",
                    f"onex.evt.omnimarket.projection-live-events-applied.v1#{i}",
                    f"summary {i}",
                    f'{{"i": {i}}}',
                    str(uuid.uuid4()) if i % 3 == 0 else None,
                ),
            )
    conn.close()


def _table_exists(pg: EphemeralPostgres, schema: str, table: str) -> bool:
    conn = pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = %s",
            (schema, table),
        )
        exists = cur.fetchone() is not None
    conn.close()
    return exists


def _row_count(pg: EphemeralPostgres, schema: str, table: str) -> int:
    conn = pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{schema}"."{table}"')  # noqa: S608
        (count,) = cur.fetchone()
    conn.close()
    return int(count)


def _insert_out_of_band_row(pg: EphemeralPostgres, event_id: str) -> None:
    """Insert directly into omninode_internal.live_events with an event_id
    that does not exist in public.live_events -- simulating the dual-write
    bleed OMN-15838 observed (a row landing in one table but not the other)."""
    conn = pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO omninode_internal.live_events (event_id, topic) "
            "VALUES (%s, %s)",
            (event_id, "onex.evt.platform.node-heartbeat.v1"),
        )
    conn.close()


@pytest.fixture
def seeded_pg(
    ephemeral_postgres: EphemeralPostgres,
) -> EphemeralPostgres:
    _seed_analytics_database_with_source_table(ephemeral_postgres)
    _apply_ok(ephemeral_postgres, SCHEMA_MIGRATION)
    return ephemeral_postgres


def test_099_creates_omninode_internal_live_events(
    seeded_pg: EphemeralPostgres,
) -> None:
    """RED-before: absent before 099; created (additively) by 099."""
    assert not _table_exists(seeded_pg, "omninode_internal", "live_events"), (
        "test setup invariant: the physical table must not pre-exist"
    )

    _apply_ok(seeded_pg, MIGRATION)

    assert _table_exists(seeded_pg, "omninode_internal", "live_events")


def test_099_preserves_public_live_events(seeded_pg: EphemeralPostgres) -> None:
    """OMN-15359 AC: source relation must survive until parity is reproven."""
    before_count = _row_count(seeded_pg, "public", "live_events")

    _apply_ok(seeded_pg, MIGRATION)

    assert _table_exists(seeded_pg, "public", "live_events")
    assert _row_count(seeded_pg, "public", "live_events") == before_count


def test_099_does_not_copy_any_rows_even_when_source_has_data(
    seeded_pg: EphemeralPostgres,
) -> None:
    """OMN-15838: 099 is schema-shape + grants only. It must never read
    public.live_events, regardless of how many rows the source carries."""
    src_count = _row_count(seeded_pg, "public", "live_events")
    assert src_count == 25, "test setup invariant"

    _apply_ok(seeded_pg, MIGRATION)

    assert _row_count(seeded_pg, "omninode_internal", "live_events") == 0


def test_099_succeeds_when_public_and_internal_row_counts_diverge(
    seeded_pg: EphemeralPostgres,
) -> None:
    """OMN-15838 regression: the removed transform-copy/reconciliation block
    used to RAISE EXCEPTION whenever public.live_events and
    omninode_internal.live_events had different row counts -- exactly the
    condition the stability-test lane hit under concurrent writes (963965 vs
    963977). Manufacture that divergence directly (an out-of-band row in the
    destination with no counterpart in the source, mirroring the dual-write
    bleed) and prove a re-apply of 099 still succeeds and leaves the
    divergence untouched -- it no longer compares the two tables at all.
    """
    _apply_ok(seeded_pg, MIGRATION)
    _insert_out_of_band_row(seeded_pg, f"evt-out-of-band-{uuid.uuid4()}")
    _insert_out_of_band_row(seeded_pg, f"evt-out-of-band-{uuid.uuid4()}")

    src_count = _row_count(seeded_pg, "public", "live_events")
    dst_count_before = _row_count(seeded_pg, "omninode_internal", "live_events")
    assert dst_count_before != src_count, "test setup invariant: counts diverge"

    result = _apply(seeded_pg, MIGRATION)

    assert result.returncode == 0, (
        f"099 must not fail on a diverging row count:\n{result.stderr}"
    )
    assert _row_count(seeded_pg, "omninode_internal", "live_events") == (
        dst_count_before
    ), "099 must not mutate existing rows in either table on re-apply"


def test_099_schema_shape_is_idempotent_on_reapply(
    seeded_pg: EphemeralPostgres,
) -> None:
    _apply_ok(seeded_pg, MIGRATION)
    _apply_ok(seeded_pg, MIGRATION)

    assert _table_exists(seeded_pg, "omninode_internal", "live_events")


def test_099_rollback_drops_the_table_and_preserves_the_source(
    seeded_pg: EphemeralPostgres,
) -> None:
    _apply_ok(seeded_pg, MIGRATION)
    assert _table_exists(seeded_pg, "omninode_internal", "live_events")

    _apply_ok(seeded_pg, ROLLBACK, dbname="postgres")

    assert not _table_exists(seeded_pg, "omninode_internal", "live_events")
    assert _table_exists(seeded_pg, "public", "live_events")
    assert _row_count(seeded_pg, "public", "live_events") == 25


def test_099_succeeds_when_public_live_events_does_not_exist(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """Matches omnibase_infra's standalone "Migration Integration Test" CI gate,
    which applies only the numbered top-level docker/migrations/forward/*.sql
    files (not the nodes/ subtree that vendors public.live_events's own
    CREATE TABLE) against a fresh database. public.live_events genuinely does
    not exist in that scope -- 099 must still succeed, creating
    omninode_internal.live_events empty (as it always does now, OMN-15838)
    rather than failing UndefinedTable.
    """
    bootstrap = ephemeral_postgres.connect()
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{ANALYTICS_DB}"')
    bootstrap.close()

    _apply_ok(ephemeral_postgres, SCHEMA_MIGRATION)

    assert not _table_exists(ephemeral_postgres, "public", "live_events")

    _apply_ok(ephemeral_postgres, MIGRATION)

    assert _table_exists(ephemeral_postgres, "omninode_internal", "live_events")
    assert _row_count(ephemeral_postgres, "omninode_internal", "live_events") == 0
