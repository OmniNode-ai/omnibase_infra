# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Execute the compaction A/B migration chain through the production psql path."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import psycopg2
import psycopg2.errors
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [
    pytest.mark.integration,
    pytest.mark.postgres,
    pytest.mark.ephemeral_pg,
    pytest.mark.timeout(60),
]

_CHAIN = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/nodes/node_projection_compaction_ab"
)
_CREATE = _CHAIN / "0000_create_compaction_ab.sql"
_GRANT = _CHAIN / "0001_grant_omninode_runtime_compaction_ab.sql"
_REPAIR = _CHAIN / "0002_repair_compaction_ab_constraints_and_grants.sql"
_ROWS = "omninode_internal.compaction_ab_rows"
_SERIES = "omninode_internal.compaction_ab_build_series"
_ROW = (
    "INSERT INTO omninode_internal.compaction_ab_rows (experiment_id, session_id, boundary_uuid, hl1_events) "
    "VALUES ('experiment', 'session', 'boundary', 2)"
)
_BUILD = (
    "INSERT INTO omninode_internal.compaction_ab_build_series (candidate_id, node_commit, split, recorded_at) "
    "VALUES ('candidate', 'commit', 'calibration', '2026-11-04T00:00:00Z')"
)


def _apply(pg: EphemeralPostgres, *migrations: Path) -> None:
    for migration in migrations:
        result = pg.psql(
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            "SET lock_timeout = '5s'; SET statement_timeout = '15s'",
            "-f",
            str(migration),
        )
        assert result.returncode == 0, f"{migration.name}: {result.stderr}"


def _provision(pg: EphemeralPostgres, *, legacy: bool) -> None:
    sql = "CREATE SCHEMA omninode_internal; CREATE ROLE omninode_runtime NOLOGIN;"
    if legacy:
        sql += (
            "CREATE TABLE omninode_internal.compaction_ab_rows (experiment_id TEXT, session_id TEXT, boundary_uuid TEXT);"
            "CREATE TABLE omninode_internal.compaction_ab_build_series (candidate_id TEXT, node_commit TEXT);"
        )
    result = pg.psql("-X", "-v", "ON_ERROR_STOP=1", "-c", sql)
    assert result.returncode == 0, result.stderr
    _apply(pg, _CREATE, _GRANT)


def _snapshot(conn: psycopg2.extensions.connection) -> list[tuple[object, ...]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT conrelid::regclass::text, contype, convalidated, "
            "pg_catalog.pg_get_constraintdef(oid) FROM pg_catalog.pg_constraint "
            "WHERE conrelid IN (%s::regclass, %s::regclass) ORDER BY 1, 2, 4",
            (_ROWS, _SERIES),
        )
        return cur.fetchall()


def test_repair_preserves_reordered_legacy_primary_keys(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    _provision(pg, legacy=True)
    result = pg.psql(
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "ALTER TABLE omninode_internal.compaction_ab_rows "
        "ADD CONSTRAINT legacy_rows_identity "
        "PRIMARY KEY (boundary_uuid, session_id, experiment_id); "
        "ALTER TABLE omninode_internal.compaction_ab_build_series "
        "ADD CONSTRAINT legacy_build_identity PRIMARY KEY (node_commit, candidate_id);",
    )
    assert result.returncode == 0, result.stderr
    _apply(pg, _REPAIR)
    with closing(pg.connect()) as conn:
        conn.autocommit = True
        before = _snapshot(conn)
        with conn.cursor() as cur:
            cur.execute("SET ROLE omninode_runtime")
            cur.execute(_ROW)
            cur.execute(_BUILD)
            cur.execute(
                _BUILD + " ON CONFLICT (candidate_id, node_commit) "
                "DO UPDATE SET h1_recall = 0.9 RETURNING h1_recall"
            )
            assert cur.fetchone() == (0.9,)
        _apply(pg, _REPAIR)
        assert _snapshot(conn) == before
        assert {row[3] for row in before if row[1] == "p"} == {
            "PRIMARY KEY (boundary_uuid, session_id, experiment_id)",
            "PRIMARY KEY (node_commit, candidate_id)",
        }


@pytest.mark.parametrize(
    ("table", "columns"),
    [(_ROWS, "experiment_id, session_id"), (_SERIES, "candidate_id")],
)
def test_repair_refuses_a_different_legacy_primary_key(
    ephemeral_postgres: EphemeralPostgres, table: str, columns: str
) -> None:
    pg = ephemeral_postgres
    _provision(pg, legacy=True)
    result = pg.psql(
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        f"ALTER TABLE {table} ADD PRIMARY KEY ({columns})",
    )
    assert result.returncode == 0, result.stderr
    with closing(pg.connect()) as conn:
        conn.autocommit = True
        before = _snapshot(conn)
        result = pg.psql(
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-v",
            "VERBOSITY=verbose",
            "-f",
            str(_REPAIR),
        )
        assert result.returncode != 0
        assert "42P16" in result.stderr, result.stderr
        assert _snapshot(conn) == before
        with conn.cursor() as cur:
            cur.execute(
                "SELECT has_table_privilege('omninode_runtime', %s, 'UPDATE')",
                (_SERIES,),
            )
            assert cur.fetchone() == (False,)


@pytest.mark.parametrize(
    ("legacy", "named_keys"),
    [(False, False), (True, False), (True, True)],
    ids=["fresh", "legacy", "legacy-named-keys"],
)
def test_chain_restores_constraints_grants_and_replays_without_changing_rows(
    ephemeral_postgres: EphemeralPostgres, legacy: bool, named_keys: bool
) -> None:
    pg = ephemeral_postgres
    _provision(pg, legacy=legacy)
    with closing(pg.connect()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Negative controls: 0000/0001 leave the exact defects repaired by 0002.
            cur.execute(
                "SELECT has_table_privilege('omninode_runtime', %s, 'UPDATE')",
                (_SERIES,),
            )
            assert cur.fetchone() == (False,)
            if legacy:
                assert _snapshot(conn) == []
            cur.execute(_ROW)
            cur.execute(_BUILD)
            if named_keys:
                cur.execute(
                    "ALTER TABLE omninode_internal.compaction_ab_rows "
                    "ADD CONSTRAINT legacy_rows_identity "
                    "PRIMARY KEY (experiment_id, session_id, boundary_uuid)"
                )
                cur.execute(
                    "ALTER TABLE omninode_internal.compaction_ab_build_series "
                    "ADD CONSTRAINT legacy_build_identity PRIMARY KEY (candidate_id, node_commit)"
                )
        _apply(pg, _REPAIR)
        before = _snapshot(conn)
        assert {(row[0], row[1], row[2], row[3]) for row in before} == {
            (
                _ROWS,
                "p",
                True,
                "PRIMARY KEY (experiment_id, session_id, boundary_uuid)",
            ),
            (_SERIES, "p", True, "PRIMARY KEY (candidate_id, node_commit)"),
            (_SERIES, "c", True, "CHECK ((candidate_id <> ''::text))"),
            (_SERIES, "c", True, "CHECK ((split = 'calibration'::text))"),
        }
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = 'omninode_internal' AND is_nullable = 'NO' "
                "AND table_name IN ('compaction_ab_rows', 'compaction_ab_build_series')"
            )
            assert set(cur.fetchall()) == {
                ("compaction_ab_rows", column)
                for column in ("experiment_id", "session_id", "boundary_uuid")
            } | {
                ("compaction_ab_build_series", column)
                for column in ("candidate_id", "node_commit", "split", "recorded_at")
            }
            for table in (_ROWS, _SERIES):
                cur.execute(
                    "SELECT privilege_type FROM information_schema.role_table_grants "
                    "WHERE table_schema = 'omninode_internal' AND table_name = %s "
                    "AND grantee = 'omninode_runtime'",
                    (table.split(".")[1],),
                )
                assert {row[0] for row in cur.fetchall()} == {
                    "SELECT",
                    "INSERT",
                    "UPDATE",
                }
            cur.execute("SELECT * FROM omninode_internal.compaction_ab_rows")
            rows_before = cur.fetchall()
            cur.execute("SELECT * FROM omninode_internal.compaction_ab_build_series")
            builds_before = cur.fetchall()
        _apply(pg, _CREATE, _GRANT, _REPAIR)
        assert _snapshot(conn) == before
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM omninode_internal.compaction_ab_rows")
            assert cur.fetchall() == rows_before
            cur.execute("SELECT * FROM omninode_internal.compaction_ab_build_series")
            assert cur.fetchall() == builds_before
            cur.execute("SET ROLE omninode_runtime")
            cur.execute(
                "SELECT current_user, rolsuper, rolbypassrls FROM pg_catalog.pg_roles "
                "WHERE rolname = current_user"
            )
            assert cur.fetchone() == ("omninode_runtime", False, False)
            cur.execute(
                _ROW + " ON CONFLICT (experiment_id, session_id, boundary_uuid) "
                "DO UPDATE SET hl1_events = EXCLUDED.hl1_events + 1 RETURNING hl1_events"
            )
            assert cur.fetchone() == (3,)
            cur.execute(
                _BUILD + " ON CONFLICT (candidate_id, node_commit) "
                "DO UPDATE SET h1_recall = 0.9 RETURNING h1_recall"
            )
            assert cur.fetchone() == (0.9,)
            for statement, error in (
                (_ROW, psycopg2.errors.UniqueViolation),
                (_BUILD, psycopg2.errors.UniqueViolation),
                (
                    _BUILD.replace("'candidate'", "NULL"),
                    psycopg2.errors.NotNullViolation,
                ),
                (_BUILD.replace("'candidate'", "''"), psycopg2.errors.CheckViolation),
                (
                    _BUILD.replace("'calibration'", "'live'"),
                    psycopg2.errors.CheckViolation,
                ),
                (
                    "DELETE FROM omninode_internal.compaction_ab_rows",
                    psycopg2.errors.InsufficientPrivilege,
                ),
                (
                    "DELETE FROM omninode_internal.compaction_ab_build_series",
                    psycopg2.errors.InsufficientPrivilege,
                ),
            ):
                with pytest.raises(error):
                    cur.execute(statement)


@pytest.mark.parametrize(
    ("invalid", "sqlstate"),
    [
        (_ROW.replace("'experiment'", "NULL"), "23502"),
        (_ROW + ";" + _ROW, "23505"),
        (_BUILD.replace("'candidate'", "NULL"), "23502"),
        (_BUILD.replace("'candidate'", "''"), "23514"),
        (_BUILD.replace("'calibration'", "'live'"), "23514"),
        (_BUILD + ";" + _BUILD, "23505"),
    ],
    ids=[
        "null-row-key",
        "duplicate-row",
        "null-candidate",
        "empty-candidate",
        "split",
        "duplicate-build",
    ],
)
def test_invalid_legacy_history_refuses_repair_atomically(
    ephemeral_postgres: EphemeralPostgres, invalid: str, sqlstate: str
) -> None:
    pg = ephemeral_postgres
    _provision(pg, legacy=True)
    with closing(pg.connect()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(invalid)
            cur.execute("SELECT * FROM omninode_internal.compaction_ab_rows")
            rows_before = cur.fetchall()
            cur.execute("SELECT * FROM omninode_internal.compaction_ab_build_series")
            builds_before = cur.fetchall()
        result = pg.psql(
            "-X", "-v", "ON_ERROR_STOP=1", "-v", "VERBOSITY=verbose", "-f", str(_REPAIR)
        )
        assert result.returncode != 0
        assert sqlstate in result.stderr, result.stderr
        assert _snapshot(conn) == []
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM omninode_internal.compaction_ab_rows")
            assert cur.fetchall() == rows_before
            cur.execute("SELECT * FROM omninode_internal.compaction_ab_build_series")
            assert cur.fetchall() == builds_before
            cur.execute(
                "SELECT has_table_privilege('omninode_runtime', %s, 'UPDATE')",
                (_SERIES,),
            )
            assert cur.fetchone() == (False,)
            cur.execute(
                "SELECT count(*) FROM information_schema.columns WHERE "
                "table_schema = 'omninode_internal' AND is_nullable = 'NO' "
                "AND table_name IN ('compaction_ab_rows', 'compaction_ab_build_series')"
            )
            assert cur.fetchone() == (0,)
