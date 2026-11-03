# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-apply proofs for the transcript_mirror_runs projection migrations (OMN-22260).

The two migrations are applied the way production applies them, ``psql -v
ON_ERROR_STOP=1 -f``, to a real database in two starting states: an empty
``omnidash_analytics`` and one already at the head of other projection chains.
The proofs:

* the table has exactly the columns of the reducer's row model and the key
  (run id, attempt id),
* applied twice, the second pass changes nothing (schema, ACL and rows),
* a table that already exists with only its key columns converges to the same
  columns,
* a schema diff across the pass shows exactly one table, its key index and its
  start-time index, and dropping the table by hand leaves the catalog as it was
  (nothing is left behind, so the expand-only class has no rollback barrier),
* the declared checks and key refuse malformed rows, and
* the writer principal is the runtime role and no other: it holds SELECT, INSERT
  and UPDATE and is refused DELETE and TRUNCATE, and every other role with schema
  access is refused every privilege on the table.
"""

from __future__ import annotations

from pathlib import Path

import psycopg2
import psycopg2.errors
import psycopg2.extensions
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD = REPO_ROOT / "docker" / "migrations" / "forward"
NODES = FORWARD / "nodes"
NODE_DIR = NODES / "node_projection_transcript_mirror_run"
CREATE = NODE_DIR / "0000_create_transcript_mirror_runs.sql"
GRANT = NODE_DIR / "0001_grant_omninode_runtime_transcript_mirror_runs.sql"
SCHEMA_MIGRATION = FORWARD / "098_create_omninode_internal_schema.sql"
DATABASE = "omnidash_analytics"
TABLE = "omninode_internal.transcript_mirror_runs"
WRITER = "omninode_runtime"
OTHER_ROLES = ("app_dashboard", "role_omnidash", "tenant_projection_writer")
PREVIOUS_HEAD = (
    NODES / "node_projection_work_ledger" / "0000_create_work_ledger.sql",
    NODES / "node_projection_pr_state" / "0000_create_pr_state.sql",
)
EXPECTED_COLUMNS = [
    ("run_id", "text", "NO"),
    ("attempt_id", "text", "NO"),
    ("state", "text", "NO"),
    ("started_at", "timestamp with time zone", "NO"),
    ("finished_at", "timestamp with time zone", "YES"),
    ("files_seen", "bigint", "NO"),
    ("stored", "bigint", "NO"),
    ("would_store", "bigint", "NO"),
    ("unchanged", "bigint", "NO"),
    ("destination_ahead", "bigint", "NO"),
    ("refused_stale", "bigint", "NO"),
    ("refused_conflict", "bigint", "NO"),
    ("source_unreadable", "bigint", "NO"),
    ("bytes_put", "bigint", "NO"),
]

_SNAPSHOT_QUERIES = {
    "relations": """
        SELECT n.nspname || '.' || c.relname || ':' || c.relkind::text,
               coalesce(c.relacl::text, '')
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
    """,
    "columns": """
        SELECT n.nspname || '.' || c.relname || '.' || a.attname,
               format_type(a.atttypid, a.atttypmod) || ' notnull=' || a.attnotnull::text
               || ' default=' || coalesce(pg_get_expr(d.adbin, d.adrelid), '')
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_catalog.pg_attrdef d
               ON d.adrelid = a.attrelid AND d.adnum = a.attnum
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
          AND c.relkind <> 'i' AND a.attnum > 0 AND NOT a.attisdropped
    """,
    "constraints": """
        SELECT n.nspname || '.' || c.relname || '.' || k.conname,
               pg_get_constraintdef(k.oid)
        FROM pg_catalog.pg_constraint k
        JOIN pg_catalog.pg_class c ON c.oid = k.conrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
    """,
    "indexes": """
        SELECT schemaname || '.' || tablename || '.' || indexname, indexdef
        FROM pg_catalog.pg_indexes
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
    """,
    "other": """
        SELECT 'schema:' || nspname || ':' || coalesce(nspacl::text, ''), ''
        FROM pg_catalog.pg_namespace
        UNION ALL SELECT 'role:' || rolname, '' FROM pg_catalog.pg_roles
        UNION ALL SELECT 'trigger:' || tgname, '' FROM pg_catalog.pg_trigger
                  WHERE NOT tgisinternal
        UNION ALL SELECT 'policy:' || policyname, '' FROM pg_catalog.pg_policies
        UNION ALL SELECT 'function:' || p.proname, ''
                  FROM pg_catalog.pg_proc p
                  JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
                  WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
        UNION ALL SELECT 'type:' || t.typname, ''
                  FROM pg_catalog.pg_type t
                  JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
                  WHERE n.nspname = 'omninode_internal'
    """,
}

Snapshot = dict[str, dict[str, str]]


def _snapshot(conn: psycopg2.extensions.connection) -> Snapshot:
    snapshot: Snapshot = {}
    with conn.cursor() as cur:
        for kind, query in _SNAPSHOT_QUERIES.items():
            cur.execute(query)
            snapshot[kind] = {str(key): str(value) for key, value in cur.fetchall()}
    conn.rollback()
    return snapshot


def _apply(pg: EphemeralPostgres, *paths: Path) -> None:
    for path in paths:
        result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(path), dbname=DATABASE)
        assert result.returncode == 0, result.stderr


def _prepare_database(pg: EphemeralPostgres) -> None:
    created = pg.psql("-v", "ON_ERROR_STOP=1", "-c", f"CREATE DATABASE {DATABASE}")
    assert created.returncode == 0, created.stderr
    schema = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(SCHEMA_MIGRATION))
    assert schema.returncode == 0, schema.stderr
    # The applying runtime provisions these roles; the grant file does not create them.
    # Every role gets USAGE on the schema so that a refusal on the table can only come
    # from the table's own privileges, never from missing schema access.
    for role in (WRITER, *OTHER_ROLES):
        made = pg.psql("-v", "ON_ERROR_STOP=1", "-c", f"CREATE ROLE {role} NOLOGIN")
        assert made.returncode == 0, made.stderr
        usage = pg.psql(
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            f"GRANT USAGE ON SCHEMA omninode_internal TO {role}",
            dbname=DATABASE,
        )
        assert usage.returncode == 0, usage.stderr


@pytest.fixture(params=["empty-database", "previous-head"])
def database(
    request: pytest.FixtureRequest, ephemeral_postgres: EphemeralPostgres
) -> EphemeralPostgres:
    _prepare_database(ephemeral_postgres)
    if request.param == "previous-head":
        _apply(ephemeral_postgres, *PREVIOUS_HEAD)
    return ephemeral_postgres


def _connect(pg: EphemeralPostgres) -> psycopg2.extensions.connection:
    return pg.connect(dbname=DATABASE)


def _start_row(run_id: str, attempt_id: str) -> tuple[str, str]:
    return (run_id, attempt_id)


_INSERT_STARTED = (
    "INSERT INTO omninode_internal.transcript_mirror_runs "
    "(run_id, attempt_id, state, started_at) "
    "VALUES (%s, %s, 'started', '2026-11-03T12:00:00Z')"
)


def test_the_table_has_the_row_models_columns_and_its_key(
    database: EphemeralPostgres,
) -> None:
    _apply(database, CREATE, GRANT)
    conn = _connect(database)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type, is_nullable "
                "FROM information_schema.columns "
                "WHERE table_schema = 'omninode_internal' "
                "AND table_name = 'transcript_mirror_runs' ORDER BY ordinal_position"
            )
            assert cur.fetchall() == EXPECTED_COLUMNS
            cur.execute(
                "SELECT a.attname FROM pg_catalog.pg_index i "
                "JOIN pg_catalog.pg_attribute a "
                "  ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey) "
                "WHERE i.indrelid = %s::regclass AND i.indisprimary "
                "ORDER BY array_position(i.indkey::int2[], a.attnum)",
                (TABLE,),
            )
            assert [row[0] for row in cur.fetchall()] == ["run_id", "attempt_id"]
            cur.execute(
                "SELECT indexdef FROM pg_catalog.pg_indexes "
                "WHERE schemaname = 'omninode_internal' "
                "AND tablename = 'transcript_mirror_runs' "
                "AND indexname = 'idx_transcript_mirror_runs_started_at'"
            )
            (definition,) = cur.fetchone() or ("",)
            assert "(started_at DESC)" in definition
    finally:
        conn.close()


def test_second_pass_changes_nothing(database: EphemeralPostgres) -> None:
    conn = _connect(database)
    try:
        _apply(database, CREATE, GRANT)
        with conn.cursor() as cur:
            cur.execute(_INSERT_STARTED, _start_row("run-1", "attempt-1"))
            cur.execute(
                "UPDATE omninode_internal.transcript_mirror_runs SET state = 'completed', "
                "finished_at = '2026-11-03T12:01:00Z', files_seen = 3, stored = 3, "
                "bytes_put = 900 WHERE run_id = 'run-1' AND attempt_id = 'attempt-1'"
            )
        conn.commit()
        after_first = _snapshot(conn)
        _apply(database, CREATE, GRANT)
        assert _snapshot(conn) == after_first
        with conn.cursor() as cur:
            cur.execute(
                "SELECT run_id, attempt_id, state, files_seen, stored, bytes_put "
                "FROM omninode_internal.transcript_mirror_runs"
            )
            assert cur.fetchall() == [("run-1", "attempt-1", "completed", 3, 3, 900)]
    finally:
        conn.close()


def test_a_table_that_exists_with_only_its_key_converges(
    database: EphemeralPostgres,
) -> None:
    created = database.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE TABLE omninode_internal.transcript_mirror_runs "
        "(run_id TEXT, attempt_id TEXT)",
        dbname=DATABASE,
    )
    assert created.returncode == 0, created.stderr
    _apply(database, CREATE, GRANT)
    conn = _connect(database)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'omninode_internal' "
                "AND table_name = 'transcript_mirror_runs' ORDER BY ordinal_position"
            )
            assert cur.fetchall() == [
                (name, kind) for name, kind, _ in EXPECTED_COLUMNS
            ]
    finally:
        conn.close()


def test_schema_diff_is_one_table_and_dropping_it_leaves_nothing_behind(
    database: EphemeralPostgres,
) -> None:
    conn = _connect(database)
    try:
        before = _snapshot(conn)
        _apply(database, CREATE, GRANT)
        after = _snapshot(conn)
        for kind in before:
            for key, value in before[kind].items():
                if kind == "other" and key.startswith("schema:omninode_internal"):
                    continue  # USAGE for the runtime role is the grant file's one schema change
                assert after[kind].get(key) == value, f"{kind} {key} changed"
        added_relations = {
            key for key in after["relations"] if key not in before["relations"]
        }
        assert added_relations == {
            "omninode_internal.transcript_mirror_runs:r",
            "omninode_internal.pk_transcript_mirror_runs:i",
            "omninode_internal.idx_transcript_mirror_runs_started_at:i",
        }
        for kind in ("columns", "constraints", "indexes"):
            owners = {
                key.split(".")[1] for key in after[kind] if key not in before[kind]
            }
            assert owners <= {"transcript_mirror_runs"}, (kind, owners)
        # The only other additions are the table's own row type and its array type
        # (and, at most, the schema ACL line); no role, function, trigger or policy.
        assert {k for k in after["other"] if k not in before["other"]} <= {
            "type:transcript_mirror_runs",
            "type:_transcript_mirror_runs",
            *(k for k in after["other"] if k.startswith("schema:omninode_internal")),
        }
        # The rollback of an expand-only table: dropping it restores the catalog.
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE {TABLE}")
        conn.commit()
        restored = _snapshot(conn)
        assert restored["relations"] == before["relations"]
        assert restored["columns"] == before["columns"]
        assert restored["constraints"] == before["constraints"]
        assert restored["indexes"] == before["indexes"]
        assert {k for k in restored["other"] if not k.startswith("schema:")} == {
            k for k in before["other"] if not k.startswith("schema:")
        }
    finally:
        conn.close()


def test_checks_and_key_refuse_malformed_rows(database: EphemeralPostgres) -> None:
    _apply(database, CREATE, GRANT)
    conn = _connect(database)
    try:
        with conn.cursor() as cur:
            cur.execute(_INSERT_STARTED, _start_row("run-1", "attempt-1"))
            conn.commit()
            # Same run, other attempt, is a different row.
            cur.execute(_INSERT_STARTED, _start_row("run-1", "attempt-2"))
            conn.commit()
            for statement, params in (
                (_INSERT_STARTED, _start_row("run-1", "attempt-1")),  # duplicate key
                (_INSERT_STARTED, _start_row("", "attempt-1")),  # empty run id
                (_INSERT_STARTED, _start_row("run-1", "")),  # empty attempt id
                (
                    "INSERT INTO omninode_internal.transcript_mirror_runs "
                    "(run_id, attempt_id, state, started_at) "
                    "VALUES ('r', 'a', 'done', now())",  # not a state
                    (),
                ),
                (
                    "INSERT INTO omninode_internal.transcript_mirror_runs "
                    "(run_id, attempt_id, state, started_at, finished_at) "
                    "VALUES ('r', 'a', 'started', now(), now())",  # started yet finished
                    (),
                ),
                (
                    "INSERT INTO omninode_internal.transcript_mirror_runs "
                    "(run_id, attempt_id, state, started_at) "
                    "VALUES ('r', 'a', 'completed', now())",  # terminal, no finish time
                    (),
                ),
                (
                    "INSERT INTO omninode_internal.transcript_mirror_runs "
                    "(run_id, attempt_id, state, started_at, stored) "
                    "VALUES ('r', 'a', 'started', now(), -1)",  # negative count
                    (),
                ),
            ):
                with pytest.raises(psycopg2.errors.IntegrityError):
                    cur.execute(statement, params)
                conn.rollback()
            cur.execute("SELECT count(*) FROM omninode_internal.transcript_mirror_runs")
            assert cur.fetchone() == (2,)
    finally:
        conn.close()


def _privileges(conn: psycopg2.extensions.connection, role: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p FROM unnest(ARRAY['SELECT','INSERT','UPDATE','DELETE','TRUNCATE',"
            "'REFERENCES','TRIGGER']) AS p WHERE has_table_privilege(%s, %s, p)",
            (role, TABLE),
        )
        found = {row[0] for row in cur.fetchall()}
    conn.rollback()
    return found


def test_the_writer_principal_is_the_runtime_role_and_no_other(
    database: EphemeralPostgres,
) -> None:
    _apply(database, CREATE, GRANT)
    conn = _connect(database)
    try:
        assert _privileges(conn, WRITER) == {"SELECT", "INSERT", "UPDATE"}
        for role in OTHER_ROLES:
            assert _privileges(conn, role) == set(), role
        with conn.cursor() as cur:
            # The ACL names the table owner and the runtime role, and nothing else:
            # no other role and no PUBLIC entry (grantee 0).
            cur.execute(
                "SELECT DISTINCT acl.grantee::regrole::text "
                "FROM pg_catalog.pg_class c, aclexplode(c.relacl) AS acl "
                "WHERE c.oid = %s::regclass",
                (TABLE,),
            )
            grantees = {row[0] for row in cur.fetchall()}
            assert grantees == {"postgres", WRITER}
        conn.rollback()
        # The runtime role writes the way the reducer does: insert the start row, then
        # update it with the terminal state and the counts.
        with conn.cursor() as cur:
            cur.execute(f"SET ROLE {WRITER}")
            cur.execute(_INSERT_STARTED, _start_row("run-1", "attempt-1"))
            cur.execute(
                "UPDATE omninode_internal.transcript_mirror_runs "
                "SET state = 'failed', finished_at = now() "
                "WHERE run_id = 'run-1' AND attempt_id = 'attempt-1'"
            )
            cur.execute(
                "SELECT state FROM omninode_internal.transcript_mirror_runs "
                "WHERE run_id = 'run-1'"
            )
            assert cur.fetchone() == ("failed",)
        conn.commit()
        for statement in (
            "DELETE FROM omninode_internal.transcript_mirror_runs",
            "TRUNCATE omninode_internal.transcript_mirror_runs",
        ):
            with conn.cursor() as cur:
                cur.execute(f"SET ROLE {WRITER}")
                with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                    cur.execute(statement)
            conn.rollback()
        for role in OTHER_ROLES:
            for statement in (
                "SELECT 1 FROM omninode_internal.transcript_mirror_runs",
                _INSERT_STARTED % ("'x'", "'y'"),
                "UPDATE omninode_internal.transcript_mirror_runs SET stored = 1",
                "DELETE FROM omninode_internal.transcript_mirror_runs",
            ):
                with conn.cursor() as cur:
                    cur.execute(f"SET ROLE {role}")
                    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                        cur.execute(statement)
                conn.rollback()
    finally:
        conn.close()
