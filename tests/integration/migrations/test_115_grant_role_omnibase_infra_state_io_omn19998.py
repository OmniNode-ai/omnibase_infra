# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration 115 grants ``role_omnibase_infra`` on the state_io tables (OMN-24037).

omnimarket's state_io nodes declare ``database: omnibase_infra`` and write their
rows through ``StateStoreAdapter`` as the runtime login role
``role_omnibase_infra``. The tables are created by the postgres superuser, so on
a warm dev volume the runtime account holds nothing and the floor fill state
node cannot write ``floor_fill_state``. Migration 115 GRANTs exactly the
privileges the topology declares (SELECT, INSERT, UPDATE on the public
state tables) and re-asserts them through a post-condition that fails the
migration unless every privilege is actually held.

This module applies the migration to a throwaway Postgres and proves the grant
chain end to end, driving the real adapter against the real table AS the
granted role:

* AC1  before the migration the role holds nothing and the adapter is refused
  (the first call, ``load``, fails with a permission-denied error);
* AC2  after the migration the role holds exactly SELECT, INSERT, UPDATE (and
  not DELETE) on ``floor_fill_state``;
* AC3  after the migration the adapter proof works as the role: seed, a second
  seed of the same scope loses, a ``cas_update`` on the stored version updates
  one row and bumps the version, a ``cas_update`` on the stale version updates
  zero rows, and ``load`` reads the winner's payload and version;
* AC4  applied twice, the ACL is unchanged (the GRANT is idempotent);
* AC5  without the role the migration applies cleanly and grants nothing.

The Postgres is the one named by ``STATE_IO_GRANT_PROOF_DSN`` when set, else a
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

from omnibase_infra.errors.repository import RepositoryExecutionError
from omnibase_infra.runtime.state_io.state_store_adapter import StateStoreAdapter
from tests.integration.migrations.conftest import (
    _REQUIRE_ENV,
    _docker_cluster,
    _docker_usable,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

REPO_ROOT = Path(__file__).resolve().parents[3]
GRANT_MIGRATION = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "115_grant_role_omnibase_infra_state_io_omn19998.sql"
)
FLOOR_FILL_MIGRATION = (
    REPO_ROOT / "docker" / "migrations" / "forward" / "113_create_floor_fill_state.sql"
)
ROLE = "role_omnibase_infra"
TABLE = "floor_fill_state"
KEY = "scope"
# The state_io tables with a public ownership declaration that the grant covers. The grant proof
# only needs the relations to exist so ``has_table_privilege`` can answer; the
# five non-floor tables are created as one-column stubs, and floor_fill_state is
# created by its real migration so the adapter can actually drive it.
STATE_TABLES = (
    "code_eval_trial_state",
    "floor_fill_state",
    "pr_landing_workflow_state",
    "session_phase_state",
)
# The shape StateIoCodec writes: ModelFloorFillState (seq, entries). It carries
# no tenant_id / state / in_flight, so the wiring seam seeds '' / '' / False.
FIRST_PAYLOAD = {"seq": 0, "entries": {}}
SECOND_PAYLOAD = {"seq": 1, "entries": {}}


def _database_dsns() -> Iterator[str]:
    env_dsn = os.environ.get("STATE_IO_GRANT_PROOF_DSN")
    if env_dsn:
        yield env_dsn
        return
    if not _docker_usable():
        if os.environ.get(_REQUIRE_ENV) == "1":
            pytest.fail(f"{_REQUIRE_ENV}=1 but no Postgres backend is usable")
        pytest.skip("no STATE_IO_GRANT_PROOF_DSN and Docker is unusable")
    for cluster in _docker_cluster():
        yield f"postgresql://postgres@{cluster.socket_dir}:{cluster.port}/postgres"


def _drop_tables_and_role(dsn: str) -> None:
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        for name in STATE_TABLES:
            cur.execute(f"DROP TABLE IF EXISTS public.{name}")
        cur.execute(
            "DROP FUNCTION IF EXISTS public.refresh_floor_fill_state_updated_at()"
        )
        cur.execute(f"DROP ROLE IF EXISTS {ROLE}")


def _grant_dsns(*, create_role: bool) -> Iterator[str]:
    """Superuser DSNs whose state tables exist and whose role is, or is not, present."""
    for dsn in _database_dsns():
        _drop_tables_and_role(dsn)
        try:
            with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
                if create_role:
                    cur.execute(
                        f"CREATE ROLE {ROLE} LOGIN NOSUPERUSER "
                        "NOCREATEDB NOCREATEROLE NOBYPASSRLS"
                    )
                cur.execute(FLOOR_FILL_MIGRATION.read_text(encoding="utf-8"))
                for name in STATE_TABLES:
                    if name != TABLE:
                        # The grant proof needs only the relation to exist so
                        # has_table_privilege can answer; the adapter drives
                        # floor_fill_state, which the real migration created.
                        cur.execute(f"CREATE TABLE public.{name} (id integer)")
            yield dsn
        finally:
            _drop_tables_and_role(dsn)


@pytest.fixture
def grant_dsn() -> Iterator[str]:
    """A DSN to a database with the state tables and the runtime role."""
    yield from _grant_dsns(create_role=True)


@pytest.fixture
def grant_dsn_without_role() -> Iterator[str]:
    """The same database with the state tables and NO runtime role."""
    yield from _grant_dsns(create_role=False)


def _apply_migration(dsn: str) -> None:
    migration_sql = GRANT_MIGRATION.read_text(encoding="utf-8")
    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(migration_sql)
    finally:
        conn.close()


def _as_role(dsn: str) -> str:
    """The same DSN but connecting as ``role_omnibase_infra`` instead of postgres.

    The throwaway cluster is trust-auth over a unix socket, so the DSN looks like
    ``postgresql://postgres@<socket_dir>:<port>/postgres`` and the user part is
    the only thing that must change to run the adapter as the granted role.
    """
    return dsn.replace("//postgres@", f"//{ROLE}@", 1)


def _privileges(dsn: str) -> dict[str, bool]:
    """privilege -> bool for SELECT, INSERT, UPDATE, DELETE on floor_fill_state."""
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        out: dict[str, bool] = {}
        for priv in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            cur.execute(
                "SELECT has_table_privilege(%s, 'public.floor_fill_state', %s)",
                (ROLE, priv),
            )
            out[priv] = bool(cur.fetchone()[0])
    return out


def _acl(dsn: str) -> str:
    """The relacl text for floor_fill_state."""
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT relacl::text FROM pg_class "
            "WHERE oid = 'public.floor_fill_state'::regclass"
        )
        row = cur.fetchone()
    return row[0] if row and row[0] is not None else ""


@dataclass(frozen=True)
class AdapterRun:
    absent: tuple[str, int] | None
    seeded: bool
    seeded_again: bool
    cas_on_stored_version: int
    cas_on_stale_version: int
    loaded: tuple[str, int] | None


async def _adapter_proof(dsn: str) -> AdapterRun:
    """The StateStoreAdapter-shaped proof: seed, CAS update, stale CAS, load."""
    adapter = StateStoreAdapter(
        dsn,
        table=TABLE,
        key_column=KEY,
        pool_factory=lambda: asyncpg.create_pool(dsn, min_size=1, max_size=2),
    )
    scope = "floor-fill-omn24037"
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


async def test_ac1_before_the_migration_the_role_holds_nothing_and_the_adapter_is_refused(
    grant_dsn: str,
) -> None:
    assert _privileges(grant_dsn) == {
        "SELECT": False,
        "INSERT": False,
        "UPDATE": False,
        "DELETE": False,
    }
    with pytest.raises(RepositoryExecutionError, match="state_io load failed"):
        await _adapter_proof(_as_role(grant_dsn))


def test_ac2_after_the_migration_the_role_holds_exactly_select_insert_update(
    grant_dsn: str,
) -> None:
    _apply_migration(grant_dsn)
    assert _privileges(grant_dsn) == {
        "SELECT": True,
        "INSERT": True,
        "UPDATE": True,
        "DELETE": False,
    }


async def test_ac3_after_the_migration_the_adapter_proof_works_as_the_role(
    grant_dsn: str,
) -> None:
    _apply_migration(grant_dsn)
    result = await _adapter_proof(_as_role(grant_dsn))
    assert result.absent is None
    assert result.seeded is True
    assert result.seeded_again is False
    assert result.cas_on_stored_version == 1
    assert result.cas_on_stale_version == 0
    assert result.loaded is not None
    payload_json, version = result.loaded
    assert json.loads(payload_json) == SECOND_PAYLOAD
    assert version == 1


def test_ac4_applied_twice_leaves_the_acl_unchanged(grant_dsn: str) -> None:
    _apply_migration(grant_dsn)
    first = _acl(grant_dsn)
    _apply_migration(grant_dsn)
    assert _acl(grant_dsn) == first


def test_ac5_without_the_role_the_migration_applies_and_grants_nothing(
    grant_dsn_without_role: str,
) -> None:
    _apply_migration(grant_dsn_without_role)
    with psycopg2.connect(grant_dsn_without_role) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regrole(%s)", (ROLE,))
        assert cur.fetchone()[0] is None
