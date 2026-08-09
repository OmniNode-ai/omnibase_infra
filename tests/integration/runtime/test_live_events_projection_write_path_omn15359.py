# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-15359 cross-boundary regression: the live_events projection write path
against a REAL physical omninode_internal.live_events table.

THE GAP THIS CLOSES (flagged by the mergesweep-0809-projplane-verify
adversarial verify, rolling ledger 2026-08-09T18:45Z)

    Every prior OMN-15359 proof for this table was one of:
      (a) a unit test resolving ``_resolve_projection_database_target``
          against a topology fixture with NO live database at all, or
      (b) an integration test that PATCHES ``_build_projection_db_adapter``
          with a ``FakeDb``/``MagicMock`` (e.g.
          ``tests/integration/runtime/test_projection_handler_db_injection_integration.py``),
          which proves the dispatch *bridge* shape but never touches real SQL.

    Neither proves the thing that was actually broken: that
    ``handler_wiring._resolve_projection_database_target`` resolves the
    contract-declared ``schema: omninode_internal`` write target, that the
    REAL ``ProjectionDatabaseOperations`` adapter (unpatched, real psycopg2,
    real connection-identity attestation) issues
    ``INSERT INTO "omninode_internal"."live_events"`` against a database where
    that relation actually exists, and that the row is durably readable back
    from that exact physical location afterward.

THIS TEST

    Drives a real event through the REAL infra dispatch bridge
    (``_make_projection_dispatch_callback`` -> real, unpatched
    ``_build_projection_db_adapter`` -> real ``ProjectionDatabaseOperations``)
    against an ephemeral, real PostgreSQL 16 cluster that has had
    098 + 099 applied via the same ``psql -f`` path production uses, with a
    real ``omninode_runtime`` LOGIN role carrying exactly the grants the
    shipped topology derives (mirroring
    ``docker/domain-adapter-proof/prove.py``'s real-role-identity pattern).
    Nothing about the DB adapter is mocked. The handler under test is a
    minimal stand-in (the real ``HandlerProjectionLiveEvents`` lives in the
    omnimarket repo and is not importable here) that performs the exact
    ``_db.upsert("live_events", "event_id", row)`` call the golden path in
    ``node_projection_live_events/contract.yaml`` documents -- the infra
    dispatch/adapter code under test is 100% real either way.

    Assertions:
      1. The row lands in AND reads back from ``omninode_internal.live_events``
         -- verified through an INDEPENDENT, freshly-opened connection (not
         reusing the writer's connection), proving the write is durably
         committed to the real physical location, not merely visible in-session.
      2. ``physical_grant_schema_for_table('omninode_internal', 'live_events')``
         agrees with the resolved write-target schema post-migration (both
         'omninode_internal', with live_events removed from the physical
         bridge in this same PR).
"""

from __future__ import annotations

import asyncio
import subprocess
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import psycopg2
import pytest

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_projection_dispatch_callback,
    _resolve_projection_database_target,
)
from omnibase_infra.topology.physical_schema_mapping import (
    physical_grant_schema_for_table,
)
from tests.helpers.application_db_topology import application_topology
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
SCHEMA_MIGRATION = FORWARD_DIR / "098_create_omninode_internal_schema.sql"
TABLE_MIGRATION = FORWARD_DIR / "099_create_omninode_internal_live_events.sql"
ANALYTICS_DB = "omnidash_analytics"
INTERNAL_ROLE = "omninode_runtime"
ROLE_PASSWORD = "live-events-write-path-proof-only"  # pragma: allowlist secret


def _apply(pg: EphemeralPostgres, path: Path) -> None:
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(path), dbname="postgres")
    assert result.returncode == 0, f"{path.name} failed to apply:\n{result.stderr}"


def _admin_dsn(pg: EphemeralPostgres) -> str:
    return f"host={pg.socket_dir} port={pg.port} user=postgres dbname={ANALYTICS_DB}"


def _internal_role_dsn(pg: EphemeralPostgres) -> str:
    return (
        f"host={pg.socket_dir} port={pg.port} user={INTERNAL_ROLE} "
        f"password={ROLE_PASSWORD} dbname={ANALYTICS_DB}"
    )


@pytest.fixture
def live_events_pg(ephemeral_postgres: EphemeralPostgres) -> EphemeralPostgres:
    """Ephemeral cluster with 098 + 099 applied and a real omninode_runtime
    LOGIN role carrying exactly the shipped topology's live_events grant."""
    bootstrap = ephemeral_postgres.connect()
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{ANALYTICS_DB}"')
    bootstrap.close()

    # 099 transform-copies FROM public.live_events -- seed the (empty) source
    # relation with the same shape production carries, matching
    # docker/migrations/forward/nodes/node_projection_live_events/0000_create_live_events.sql.
    # This fixture intentionally starts empty: the copy-migration's own
    # correctness (count/key/hash reconciliation over real rows) is proven in
    # tests/integration/migrations/test_099_omninode_internal_live_events_omn15359.py;
    # this file's job is the write-PATH seam, not the historical-row copy.
    seed = psycopg2.connect(_admin_dsn(ephemeral_postgres))
    seed.autocommit = True
    with seed.cursor() as cur:
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
    seed.close()

    _apply(ephemeral_postgres, SCHEMA_MIGRATION)
    _apply(ephemeral_postgres, TABLE_MIGRATION)

    admin = psycopg2.connect(_admin_dsn(ephemeral_postgres))
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(
            f"CREATE ROLE {INTERNAL_ROLE} LOGIN PASSWORD %s NOSUPERUSER NOBYPASSRLS",
            (ROLE_PASSWORD,),
        )
        cur.execute(f"GRANT CONNECT ON DATABASE {ANALYTICS_DB} TO {INTERNAL_ROLE}")
        cur.execute(f"GRANT USAGE ON SCHEMA omninode_internal TO {INTERNAL_ROLE}")
        cur.execute(
            f"GRANT SELECT, INSERT, UPDATE ON omninode_internal.live_events "
            f"TO {INTERNAL_ROLE}"
        )
    admin.close()
    return ephemeral_postgres


def _read_back_independently(
    pg: EphemeralPostgres, event_id: str
) -> tuple[str, str, str] | None:
    """Open a brand-new connection (never the writer's) and read the row back
    directly from the physical location -- proves durable commit, not merely
    in-session visibility."""
    conn = psycopg2.connect(_admin_dsn(pg))
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT event_id, topic, summary "
                "FROM omninode_internal.live_events WHERE event_id = %s",
                (event_id,),
            )
            row = cur.fetchone()
            return tuple(row) if row is not None else None
    finally:
        conn.close()


class _StandInProjectionHandler:
    """Minimal stand-in for HandlerProjectionLiveEvents (omnimarket repo, not
    importable from omnibase_infra). Exercises the same
    ``_db.upsert("live_events", "event_id", row)`` call the node's golden
    path documents -- everything downstream of this call (the adapter, the
    connection, the SQL, the real Postgres cluster) is 100% real."""

    def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
        db = input_data.pop("_db")
        input_data.pop("_event_type", None)
        row = {
            "event_id": input_data["event_id"],
            "topic": input_data.get("topic", ""),
            "summary": input_data.get("summary", ""),
            "source": input_data.get("source", "platform"),
            "type": input_data.get("type", "ACTION"),
            "payload": input_data.get("payload", "{}"),
        }
        assert db.upsert("live_events", "event_id", row) is True
        return {"rows_upserted": 1}


def test_live_events_write_lands_in_and_reads_back_from_omninode_internal(
    live_events_pg: EphemeralPostgres,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", _internal_role_dsn(live_events_pg))

    declaration = ModelDbTableDeclaration(
        name="live_events",
        database_ref="application",
        schema="omninode_internal",
        migration="docker/migrations/forward/099_create_omninode_internal_live_events.sql",
        access="write",
        role="live_events",
    )
    target = _resolve_projection_database_target((declaration,), application_topology())

    assert target.table_targets[0].schema == "omninode_internal"
    assert target.table_targets[0].write_binding is not None
    assert (
        target.table_targets[0].write_binding.binding_ref == "omninode_runtime_service"
    )

    handler = _StandInProjectionHandler()
    callback = _make_projection_dispatch_callback(
        handler, target, ("onex.evt.platform.node-heartbeat.v1",)
    )

    event_id = f"evt-write-path-{uuid.uuid4()}"
    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-heartbeat.v1"
    envelope.payload = {
        "event_id": event_id,
        "topic": "onex.evt.platform.node-heartbeat.v1",
        "summary": "cross-boundary write-path proof",
        "source": "platform",
        "type": "ACTION",
        "payload": "{}",
    }

    asyncio.run(callback(envelope))

    row = _read_back_independently(live_events_pg, event_id)
    assert row is not None, (
        "row did not land in omninode_internal.live_events -- the real "
        "dispatch/adapter write path failed silently"
    )
    assert row[0] == event_id
    assert row[1] == "onex.evt.platform.node-heartbeat.v1"
    assert row[2] == "cross-boundary write-path proof"


def test_grant_derivation_schema_agrees_with_the_insert_target_schema() -> None:
    """Post-099 + bridge removal: the grant-privilege check and the real SQL
    INSERT target must resolve to the identical physical schema -- the exact
    seam a bridge-set omission would silently reintroduce. Pure topology
    resolution, no live database required; paired here with the write-path
    test above so both halves of the seam are proven in one file."""
    declaration = ModelDbTableDeclaration(
        name="live_events",
        database_ref="application",
        schema="omninode_internal",
        migration="docker/migrations/forward/099_create_omninode_internal_live_events.sql",
        access="write",
        role="live_events",
    )
    target = _resolve_projection_database_target((declaration,), application_topology())
    insert_target_schema = target.table_targets[0].schema

    grant_check_schema = physical_grant_schema_for_table(
        "omninode_internal", "live_events"
    )

    assert grant_check_schema == "omninode_internal"
    assert grant_check_schema == insert_target_schema
