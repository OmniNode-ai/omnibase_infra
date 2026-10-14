# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20737: the work-ledger parity check's read-only role can read the state table.

``nodes/node_projection_work_ledger/0004_grant_work_ledger_ro_work_ledger_state.sql``
is applied with ``psql -v ON_ERROR_STOP=1 -f`` after the vendored 0000 and 0001
on a throwaway PostgreSQL. ``work_ledger_ro`` is created by the test as a
NOSUPERUSER NOBYPASSRLS login holding what the dev lane's role holds before the
grant: USAGE on the schema and SELECT on ``work_ledger_rows``. The login's
password is generated per run and never leaves the test.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import psycopg2
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_NODE = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/nodes/node_projection_work_ledger"
)
_BEFORE = (
    "0000_create_work_ledger.sql",
    "0001_grant_omninode_runtime_work_ledger.sql",
)
_GRANT = "0004_grant_work_ledger_ro_work_ledger_state.sql"
_ROLE = "work_ledger_ro"
_STATE = "omninode_internal.work_ledger_state"


def _apply(pg: EphemeralPostgres, filename: str) -> None:
    migration = _NODE / filename
    assert migration.is_file(), f"missing migration: {migration}"
    applied = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
    assert applied.returncode == 0, applied.stdout + applied.stderr


def _privileges(pg: EphemeralPostgres) -> dict[str, bool]:
    with pg.connect() as admin, admin.cursor() as cur:
        cur.execute(
            "SELECT has_table_privilege(%s, %s, 'SELECT'), "
            "has_table_privilege(%s, %s, 'INSERT'), "
            "has_table_privilege(%s, %s, 'UPDATE'), "
            "has_table_privilege(%s, %s, 'DELETE')",
            (_ROLE, _STATE) * 4,
        )
        select, insert, update, delete = cur.fetchone()
    return {"SELECT": select, "INSERT": insert, "UPDATE": update, "DELETE": delete}


def test_grant_lets_the_read_only_role_read_state_and_nothing_more(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    password = secrets.token_hex(16)
    provisioned = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE SCHEMA omninode_internal; "
        "CREATE ROLE omninode_runtime NOLOGIN NOSUPERUSER NOBYPASSRLS; "
        f"CREATE ROLE {_ROLE} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD '{password}'; "
        f"GRANT USAGE ON SCHEMA omninode_internal TO {_ROLE};",
    )
    assert provisioned.returncode == 0, provisioned.stderr
    for filename in _BEFORE:
        _apply(pg, filename)
    granted = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "GRANT SELECT ON omninode_internal.work_ledger_rows TO work_ledger_ro;",
    )
    assert granted.returncode == 0, granted.stderr
    seeded = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "INSERT INTO omninode_internal.work_ledger_state "
        "(entity_key, kind, projected_at) "
        "VALUES ('claim:omn20737', 'claim', NOW());",
    )
    assert seeded.returncode == 0, seeded.stderr

    # RED control: before 0004 the role reads the rows table and not the state
    # table, which is the live failure ("permission denied for table").
    assert _privileges(pg) == {
        "SELECT": False,
        "INSERT": False,
        "UPDATE": False,
        "DELETE": False,
    }
    with (
        pg.connect(user=_ROLE, password=password, dbname="postgres") as reader,
        reader.cursor() as cur,
    ):
        cur.execute("SELECT count(*) FROM omninode_internal.work_ledger_rows")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("SELECT count(*) FROM omninode_internal.work_ledger_state")

    _apply(pg, _GRANT)

    assert _privileges(pg) == {
        "SELECT": True,
        "INSERT": False,
        "UPDATE": False,
        "DELETE": False,
    }
    with (
        pg.connect(user=_ROLE, password=password, dbname="postgres") as reader,
        reader.cursor() as cur,
    ):
        cur.execute("SELECT count(*) FROM omninode_internal.work_ledger_state")
        assert cur.fetchone() == (1,)
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("DELETE FROM omninode_internal.work_ledger_state")

    # A replay is a no-op, so the runner can apply it on a warm volume.
    _apply(pg, _GRANT)
    assert _privileges(pg)["SELECT"] is True


def test_grant_is_a_noop_where_the_role_does_not_exist(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """A fresh volume, a slot or another lane has no work_ledger_ro.

    An unguarded GRANT would fail the file there and block every migration
    behind it.
    """
    pg = ephemeral_postgres
    provisioned = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE SCHEMA omninode_internal; "
        "CREATE ROLE omninode_runtime NOLOGIN NOSUPERUSER NOBYPASSRLS;",
    )
    assert provisioned.returncode == 0, provisioned.stderr
    for filename in (*_BEFORE, _GRANT):
        _apply(pg, filename)
    with pg.connect() as admin, admin.cursor() as cur:
        cur.execute("SELECT count(*) FROM pg_roles WHERE rolname = %s", (_ROLE,))
        assert cur.fetchone() == (0,)
