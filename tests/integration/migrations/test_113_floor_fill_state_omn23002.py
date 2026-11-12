# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration 113 applied to a real Postgres: ``floor_fill_state`` (OMN-23002).

omnimarket's ``node_floor_fill_orchestrator`` declares
``state_io: {database: omnibase_infra, table: floor_fill_state, key: scope}``.
The runtime loads that row before each leg and CAS-writes it after, through
``StateStoreAdapter``. This module applies the migration to a throwaway
Postgres and drives the real adapter against the real table:

* AC1  the migration creates ``floor_fill_state`` keyed on ``scope`` with the
  adapter's columns and its two partial indexes and trigger;
* AC3  applied twice, it raises nothing and changes nothing;
* AC4  ``seed`` inserts, a second ``seed`` of the same scope loses, a
  ``cas_update`` on the stored version updates one row and bumps the version, a
  ``cas_update`` on the stale version updates zero rows, ``load`` reads the
  winner's payload and version, and the ``updated_at`` trigger fires;
* AC5  the same adapter calls with the migration NOT applied fail red: the
  table is absent and the adapter raises, so the proof above is not vacuous.

The Postgres is the one named by ``FLOOR_FILL_PROOF_DSN`` when set, else a
throwaway container the migrations conftest starts (skipped, or failed under
``ONEX_MIGRATION_PROOF_REQUIRE_PG=1``, when Docker is unusable).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import asyncpg
import psycopg2
import pytest
from psycopg2 import sql

from omnibase_infra.errors.repository import RepositoryExecutionError
from omnibase_infra.runtime.state_io.state_store_adapter import StateStoreAdapter
from tests.integration.migrations.conftest import (
    _REQUIRE_ENV,
    _docker_cluster,
    _docker_usable,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = (
    REPO_ROOT / "docker" / "migrations" / "forward" / "113_create_floor_fill_state.sql"
)
TABLE = "floor_fill_state"
KEY = "scope"
EXPECTED_COLUMNS = {
    "scope": ("text", "NO"),
    "tenant_id": ("text", "NO"),
    "state": ("text", "NO"),
    "in_flight": ("boolean", "NO"),
    "payload": ("jsonb", "NO"),
    "version": ("integer", "NO"),
    "pending_emissions": ("jsonb", "YES"),
    "publish_attempts": ("integer", "NO"),
    "created_at": ("timestamp with time zone", "NO"),
    "updated_at": ("timestamp with time zone", "NO"),
}
# The shape StateIoCodec writes: ModelFloorFillState (seq, entries). It carries no
# tenant_id / state / in_flight, so the wiring seam seeds '' / '' / False.
FIRST_PAYLOAD = {"seq": 0, "entries": {}}
SECOND_PAYLOAD = {"seq": 1, "entries": {}}


def _database_dsns() -> Iterator[str]:
    env_dsn = os.environ.get("FLOOR_FILL_PROOF_DSN")
    if env_dsn:
        yield env_dsn
        return
    if not _docker_usable():
        if os.environ.get(_REQUIRE_ENV) == "1":
            pytest.fail(f"{_REQUIRE_ENV}=1 but no Postgres backend is usable")
        pytest.skip("no FLOOR_FILL_PROOF_DSN and Docker is unusable")
    for cluster in _docker_cluster():
        yield f"postgresql://postgres@{cluster.socket_dir}:{cluster.port}/postgres"


@pytest.fixture
def fresh_dsn() -> Iterator[str]:
    """A DSN to a database that has no ``floor_fill_state`` yet."""
    for dsn in _database_dsns():
        _drop(dsn)
        try:
            yield dsn
        finally:
            _drop(dsn)


def _drop(dsn: str) -> None:
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS public.{TABLE}")
        cur.execute(
            "DROP FUNCTION IF EXISTS public.refresh_floor_fill_state_updated_at()"
        )


def _apply_migration(dsn: str) -> None:
    migration_sql = MIGRATION.read_text(encoding="utf-8")
    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(migration_sql)
    finally:
        conn.close()


@dataclass(frozen=True)
class Catalog:
    columns: dict[str, tuple[str, str]]
    primary_key: list[str]
    index_names: list[str]
    index_definitions: list[tuple[str, str]]
    triggers: list[str]


@dataclass(frozen=True)
class AdapterRun:
    absent: tuple[str, int] | None
    seeded: bool
    seeded_again: bool
    cas_on_stored_version: int
    cas_on_stale_version: int
    loaded: tuple[str, int] | None


def _catalog(dsn: str) -> Catalog:
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s ORDER BY column_name",
            (TABLE,),
        )
        columns = {name: (kind, nullable) for name, kind, nullable in cur.fetchall()}
        cur.execute(
            "SELECT a.attname FROM pg_index i JOIN pg_attribute a "
            "ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "WHERE i.indrelid = %s::regclass AND i.indisprimary",
            (f"public.{TABLE}",),
        )
        primary_key = [row[0] for row in cur.fetchall()]
        cur.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename = %s ORDER BY indexname",
            (TABLE,),
        )
        indexes = cur.fetchall()
        cur.execute(
            "SELECT tgname FROM pg_trigger WHERE tgrelid = %s::regclass "
            "AND NOT tgisinternal ORDER BY tgname",
            (f"public.{TABLE}",),
        )
        triggers = [row[0] for row in cur.fetchall()]
    return Catalog(
        columns=columns,
        primary_key=primary_key,
        index_names=[name for name, _ in indexes],
        index_definitions=indexes,
        triggers=triggers,
    )


async def _adapter_proof(dsn: str) -> AdapterRun:
    """The StateStoreAdapter-shaped proof: seed, CAS update, stale CAS, load."""
    adapter = StateStoreAdapter(
        dsn,
        table=TABLE,
        key_column=KEY,
        pool_factory=lambda: asyncpg.create_pool(dsn, min_size=1, max_size=2),
    )
    scope = "floor-fill-omn23002"
    try:
        absent = await adapter.load(scope)
        seeded = await adapter.seed(
            scope,
            tenant_id="",
            state="",
            in_flight=False,
            payload_json=json.dumps(FIRST_PAYLOAD),
        )
        seeded_again = await adapter.seed(
            scope,
            tenant_id="",
            state="",
            in_flight=False,
            payload_json=json.dumps(FIRST_PAYLOAD),
        )
        won = await adapter.cas_update(
            scope,
            tenant_id="",
            state="",
            in_flight=False,
            payload_json=json.dumps(SECOND_PAYLOAD),
            expected_version=0,
        )
        stale = await adapter.cas_update(
            scope,
            tenant_id="",
            state="",
            in_flight=False,
            payload_json=json.dumps({"seq": 99, "entries": {}}),
            expected_version=0,
        )
        loaded = await adapter.load(scope)
    finally:
        await adapter.close()
    return AdapterRun(
        absent=absent,
        seeded=seeded,
        seeded_again=seeded_again,
        cas_on_stored_version=won,
        cas_on_stale_version=stale,
        loaded=loaded,
    )


def test_ac1_the_migration_creates_the_table_keyed_on_scope(fresh_dsn: str) -> None:
    _apply_migration(fresh_dsn)
    catalog = _catalog(fresh_dsn)
    assert catalog.columns == EXPECTED_COLUMNS
    assert catalog.primary_key == [KEY]
    assert catalog.index_names == [
        "floor_fill_state_pkey",
        "ix_floor_fill_state_recoverable_batches",
        "ix_floor_fill_state_stale_sweep",
    ]
    assert catalog.triggers == ["trg_floor_fill_state_updated_at"]


def test_ac3_the_migration_applied_twice_is_a_no_op(fresh_dsn: str) -> None:
    _apply_migration(fresh_dsn)
    first = _catalog(fresh_dsn)
    _apply_migration(fresh_dsn)
    assert _catalog(fresh_dsn) == first


async def test_ac4_seed_cas_update_and_load_work_on_the_real_table(
    fresh_dsn: str,
) -> None:
    _apply_migration(fresh_dsn)
    result = await _adapter_proof(fresh_dsn)
    assert result.absent is None
    assert result.seeded is True
    assert result.seeded_again is False
    assert result.cas_on_stored_version == 1
    # A conditional write against a stale version updates zero rows.
    assert result.cas_on_stale_version == 0
    assert result.loaded is not None
    payload_json, version = result.loaded
    assert json.loads(payload_json) == SECOND_PAYLOAD
    assert version == 1
    with psycopg2.connect(fresh_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT updated_at > created_at FROM {} WHERE {} = %s").format(
                sql.Identifier("public", TABLE), sql.Identifier(KEY)
            ),
            ("floor-fill-omn23002",),
        )
        assert cur.fetchone() == (True,)


async def test_ac5_the_same_proof_fails_without_the_migration(fresh_dsn: str) -> None:
    """Failing control: the table is absent, so the adapter's first call raises."""
    with pytest.raises(RepositoryExecutionError, match="state_io load failed"):
        await _adapter_proof(fresh_dsn)
    with psycopg2.connect(fresh_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"public.{TABLE}",))
        assert cur.fetchone() == (None,)
