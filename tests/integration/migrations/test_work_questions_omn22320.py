# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-apply proofs for the work-questions projection migration (OMN-22320).

The migration is applied the way production applies it, ``psql -v
ON_ERROR_STOP=1 -f``, to a real database in two starting states: an empty
``omnidash_analytics`` and one already at the head of other projection chains.
The proofs are the three that matter for a table-only migration:

* applied twice, the second pass changes nothing (schema and rows),
* a schema diff across the first pass shows exactly the five tables (with their
  own columns, constraints and indexes) and no other change, and
* the declared checks and primary keys refuse malformed rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg2
import psycopg2.extensions
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD = REPO_ROOT / "docker" / "migrations" / "forward"
NODES = FORWARD / "nodes"
MIGRATION = NODES / "node_projection_work_questions" / "0000_create_work_questions.sql"
SCHEMA_MIGRATION = FORWARD / "098_create_omninode_internal_schema.sql"
DATABASE = "omnidash_analytics"
FIVE_TABLES = frozenset(
    {
        "work_outcome_units",
        "orchestration_style_outcomes",
        "lane_drift_and_stalls",
        "brief_quality_vs_outcome",
        "subagent_waste",
    }
)
PREVIOUS_HEAD = (
    NODES / "node_projection_work_ledger" / "0000_create_work_ledger.sql",
    NODES / "node_projection_pr_state" / "0000_create_pr_state.sql",
)

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
        SELECT 'schema:' || nspname, '' FROM pg_catalog.pg_namespace
        UNION ALL SELECT 'role:' || rolname, '' FROM pg_catalog.pg_roles
        UNION ALL SELECT 'trigger:' || tgname, '' FROM pg_catalog.pg_trigger
                  WHERE NOT tgisinternal
        UNION ALL SELECT 'policy:' || policyname, '' FROM pg_catalog.pg_policies
        UNION ALL SELECT 'function:' || p.proname, ''
                  FROM pg_catalog.pg_proc p
                  JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
                  WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
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


def _apply(pg: EphemeralPostgres, path: Path) -> None:
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(path), dbname=DATABASE)
    assert result.returncode == 0, result.stderr


def _prepare_database(pg: EphemeralPostgres) -> None:
    created = pg.psql("-v", "ON_ERROR_STOP=1", "-c", f"CREATE DATABASE {DATABASE}")
    assert created.returncode == 0, created.stderr
    schema = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(SCHEMA_MIGRATION))
    assert schema.returncode == 0, schema.stderr


@pytest.fixture(params=["empty-database", "previous-head"])
def database(
    request: pytest.FixtureRequest, ephemeral_postgres: EphemeralPostgres
) -> EphemeralPostgres:
    _prepare_database(ephemeral_postgres)
    if request.param == "previous-head":
        for migration in PREVIOUS_HEAD:
            _apply(ephemeral_postgres, migration)
    return ephemeral_postgres


def _connect(pg: EphemeralPostgres) -> psycopg2.extensions.connection:
    return pg.connect(dbname=DATABASE)


def _added(before: Snapshot, after: Snapshot) -> Snapshot:
    return {
        kind: {k: v for k, v in after[kind].items() if k not in before[kind]}
        for kind in after
    }


def test_second_pass_changes_nothing(database: EphemeralPostgres) -> None:
    conn = _connect(database)
    try:
        _apply(database, MIGRATION)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO omninode_internal.subagent_waste (generation_id, "
                "session_id, unit_id, label, waste_streams, surviving_streams, "
                "answer_use_unknown_streams, unknown_streams, pending_streams, "
                "no_written_paths_streams) VALUES ('g1', 's1', 'u1', 'x', 1, 2, 3, 4, 5, 6)"
            )
        conn.commit()
        after_first = _snapshot(conn)
        _apply(database, MIGRATION)
        assert _snapshot(conn) == after_first
        with conn.cursor() as cur:
            cur.execute(
                "SELECT generation_id, session_id, waste_streams, pending_streams "
                "FROM omninode_internal.subagent_waste"
            )
            assert cur.fetchall() == [("g1", "s1", 1, 5)]
    finally:
        conn.close()


def test_schema_diff_is_exactly_the_five_tables(database: EphemeralPostgres) -> None:
    conn = _connect(database)
    try:
        before = _snapshot(conn)
        _apply(database, MIGRATION)
        after = _snapshot(conn)
    finally:
        conn.close()
    # Nothing that existed was dropped or altered.
    for kind in before:
        for key, value in before[kind].items():
            assert after[kind].get(key) == value, f"{kind} {key} changed"
    added = _added(before, after)
    tables = {
        key.removeprefix("omninode_internal.").removesuffix(":r")
        for key in added["relations"]
        if key.endswith(":r")
    }
    assert tables == FIVE_TABLES
    # Every other added relation is a primary-key index of one of the five.
    others = {key for key in added["relations"] if not key.endswith(":r")}
    assert others == {
        f"omninode_internal.pk_{table}:i" for table in sorted(FIVE_TABLES)
    }
    # Every added column, constraint and index belongs to one of the five.
    for kind in ("columns", "constraints", "indexes"):
        owners = {key.split(".")[1] for key in added[kind]}
        assert owners <= FIVE_TABLES, (kind, owners - FIVE_TABLES)
    assert added["other"] == {}


def test_checks_and_keys_refuse_malformed_rows(database: EphemeralPostgres) -> None:
    _apply(database, MIGRATION)
    conn = _connect(database)
    base = (
        "INSERT INTO omninode_internal.work_outcome_units "
        "(generation_id, unit_id, unit_kind, state, style, model, effort{extra}) "
        "VALUES ('g1', %s, 'lane', %s, 'solo', 'sonnet', 'high'{vals})"
    )
    try:
        with conn.cursor() as cur:
            cur.execute(base.format(extra="", vals=""), ("u1", "resolved"))
            conn.commit()
            for statement, params in (
                (base.format(extra="", vals=""), ("u1", "resolved")),
                (base.format(extra="", vals=""), ("u2", "done")),
                (
                    base.format(
                        extra=", cost_measured_amount, cost_unknown",
                        vals=", 1.5, TRUE",
                    ),
                    ("u3", "resolved"),
                ),
            ):
                with pytest.raises(psycopg2.errors.IntegrityError):
                    cur.execute(statement, params)
                conn.rollback()
            cur.execute(
                "INSERT INTO omninode_internal.work_outcome_units "
                "(generation_id, unit_id, unit_kind, state, style, model, effort, "
                "pr_roles, generation_source_versions) VALUES "
                "('g2', 'u1', 'lane', 'unknown', 'solo', 'sonnet', 'high', %s, %s)",
                (json.dumps({"7": "own"}), json.dumps({"ledger": 11})),
            )
            conn.commit()
            cur.execute(
                "SELECT count(*), max((generation_source_versions->>'ledger')::int) "
                "FROM omninode_internal.work_outcome_units"
            )
            assert cur.fetchone() == (2, 11)
    finally:
        conn.close()
