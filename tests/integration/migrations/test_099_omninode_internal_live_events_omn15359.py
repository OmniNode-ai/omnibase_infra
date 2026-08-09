# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-15359 -- physically create omninode_internal.live_events and
transform-copy the existing public.live_events rows.

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


def _event_ids(pg: EphemeralPostgres, schema: str, table: str) -> set[str]:
    conn = pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f'SELECT event_id FROM "{schema}"."{table}"')  # noqa: S608
        rows = {row[0] for row in cur.fetchall()}
    conn.close()
    return rows


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


def test_099_copies_every_source_row(seeded_pg: EphemeralPostgres) -> None:
    _apply_ok(seeded_pg, MIGRATION)

    src_count = _row_count(seeded_pg, "public", "live_events")
    dst_count = _row_count(seeded_pg, "omninode_internal", "live_events")
    assert src_count == 25
    assert dst_count == src_count

    src_ids = _event_ids(seeded_pg, "public", "live_events")
    dst_ids = _event_ids(seeded_pg, "omninode_internal", "live_events")
    assert src_ids == dst_ids


def test_099_preserves_public_live_events(seeded_pg: EphemeralPostgres) -> None:
    """OMN-15359 AC: source relation must survive until parity is reproven."""
    before_count = _row_count(seeded_pg, "public", "live_events")

    _apply_ok(seeded_pg, MIGRATION)

    assert _table_exists(seeded_pg, "public", "live_events")
    assert _row_count(seeded_pg, "public", "live_events") == before_count


def test_099_row_content_is_faithfully_reproduced(
    seeded_pg: EphemeralPostgres,
) -> None:
    """Every copied column value matches the source row exactly, not just the key."""
    _apply_ok(seeded_pg, MIGRATION)

    conn = seeded_pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, event_id, type, source, topic, summary, payload, "
            "correlation_id, timestamp, created_at "
            "FROM public.live_events ORDER BY event_id"
        )
        source_rows = cur.fetchall()
        cur.execute(
            "SELECT id, event_id, type, source, topic, summary, payload, "
            "correlation_id, timestamp, created_at "
            "FROM omninode_internal.live_events ORDER BY event_id"
        )
        dest_rows = cur.fetchall()
    conn.close()

    assert source_rows == dest_rows


def test_099_is_idempotent_on_reapply(seeded_pg: EphemeralPostgres) -> None:
    _apply_ok(seeded_pg, MIGRATION)
    _apply_ok(seeded_pg, MIGRATION)

    assert _row_count(seeded_pg, "omninode_internal", "live_events") == 25


def test_099_reapply_after_new_source_rows_copies_only_the_delta(
    seeded_pg: EphemeralPostgres,
) -> None:
    """A later, unrelated re-apply must pick up rows written to the source
    after the first apply (e.g. a retry window) without duplicating existing
    ones."""
    _apply_ok(seeded_pg, MIGRATION)
    assert _row_count(seeded_pg, "omninode_internal", "live_events") == 25

    conn = seeded_pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.live_events (event_id, topic) VALUES (%s, %s)",
            (f"evt-late-{uuid.uuid4()}", "onex.evt.platform.node-heartbeat.v1"),
        )
    conn.close()

    _apply_ok(seeded_pg, MIGRATION)

    assert _row_count(seeded_pg, "public", "live_events") == 26
    assert _row_count(seeded_pg, "omninode_internal", "live_events") == 26
    assert _event_ids(seeded_pg, "public", "live_events") == _event_ids(
        seeded_pg, "omninode_internal", "live_events"
    )


def test_099_reconciliation_fails_closed_on_a_corrupted_partial_copy(
    seeded_pg: EphemeralPostgres,
) -> None:
    """If the destination table exists but a row was corrupted/dropped before
    the reconciliation check runs, the migration must abort loudly rather than
    silently accept the mismatch. Simulated by hand-creating the destination
    table with only a subset of rows and a content divergence, then re-running
    099's own reconciliation logic via the full migration file (which is
    idempotent CREATE TABLE + copy + reconcile in one file): corrupt one row's
    payload directly in the destination after a first successful apply, then
    prove a second apply (which re-derives hashes from live state on every
    run) detects it.
    """
    _apply_ok(seeded_pg, MIGRATION)

    conn = seeded_pg.connect(dbname=ANALYTICS_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE omninode_internal.live_events "
            "SET payload = payload || '-corrupted' "
            "WHERE event_id = (SELECT event_id FROM public.live_events LIMIT 1)"
        )
    conn.close()

    result = _apply(seeded_pg, MIGRATION)

    assert result.returncode != 0
    assert "content hash mismatch" in result.stderr


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
    omninode_internal.live_events empty rather than failing UndefinedTable.
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
