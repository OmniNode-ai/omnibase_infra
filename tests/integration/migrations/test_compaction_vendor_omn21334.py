# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live-apply proofs for the barrier and receipts migration chains (OMN-21334)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import sql

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.ephemeral_pg]

_FORWARD = Path(__file__).resolve().parents[3] / "docker/migrations/forward/nodes"
_CHAINS = (
    ("node_projection_session_content", 1, 2),
    ("node_projection_claude_hook_events", 0, 8),
    ("node_projection_compaction_receipts", 0, 1),
)
_TABLES = ("session_content", "claude_hook_events", "compaction_receipts")
_BARRIER_COLUMNS = {
    "content_record_id": "text",
    "stream": "text",
    "producer_seq": "bigint",
    "record_type": "text",
    "barrier_id": "text",
    "high_water": "bigint",
    "content_capture": "text",
}


def _apply_migrations(pg: EphemeralPostgres) -> None:
    for node, first, last in _CHAINS:
        migrations = sorted((_FORWARD / node).glob("*.sql"))
        migrations = [p for p in migrations if int(p.name[:4]) <= last]
        assert [int(p.name[:4]) for p in migrations] == list(range(first, last + 1))
        for migration in migrations:
            result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
            assert result.returncode == 0, f"{migration.name}: {result.stderr}"


@pytest.fixture
def applied(
    ephemeral_postgres: EphemeralPostgres,
) -> Iterator[psycopg2.extensions.connection]:
    """Provision the schema and runtime role, then apply the real vendored SQL."""
    provisioned = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE SCHEMA omninode_internal; CREATE ROLE omninode_runtime NOLOGIN;",
    )
    assert provisioned.returncode == 0, provisioned.stderr
    _apply_migrations(ephemeral_postgres)
    conn = ephemeral_postgres.connect()
    try:
        yield conn
    finally:
        conn.close()


def test_second_apply_preserves_populated_rows_and_cursors(
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            """
            INSERT INTO omninode_internal.session_content (
                event_id, session_id, content_kind, content, emitted_at,
                content_record_id, stream, producer_seq, record_type,
                barrier_id, high_water, content_capture
            ) VALUES (
                'event-1', 'session-1', 'barrier', 'retained', NOW(),
                'record-1', 'content', 42, 'barrier', 'boundary-1', 41, 'full'
            );
            INSERT INTO omninode_internal.claude_hook_events (
                event_id, session_id, is_subagent, hook_event_name,
                correlation_id, emitted_at, source_topic, producer_seq
            ) VALUES (
                '00000000-0000-0000-0000-000000000001', 'session-1', FALSE,
                'PreCompact', '00000000-0000-0000-0000-000000000002',
                NOW(), 'hooks', 43
            );
            INSERT INTO omninode_internal.compaction_receipts (
                session_id, receipt_key, boundary_uuid, boundary_kind,
                activation_id, verdict, reason, installed_messages_hash,
                observed_hash, source_topic, projected_at
            ) VALUES (
                'session-1', 'receipt-1', 'boundary-1', 'compaction',
                '00000000-0000-0000-0000-000000000003', 'verified',
                'matched', 'hash-1', 'hash-1', 'receipts', NOW()
            );
            """
        )
        before = {}
        for table in _TABLES:
            cur.execute(
                sql.SQL("SELECT * FROM omninode_internal.{}").format(
                    sql.Identifier(table)
                )
            )
            before[table] = cur.fetchall()
            assert len(before[table]) == 1
    # The separate psql apply needs table locks: release this transaction first.
    applied.commit()
    _apply_migrations(ephemeral_postgres)
    with applied.cursor() as cur:
        for table in _TABLES:
            cur.execute(
                sql.SQL("SELECT * FROM omninode_internal.{}").format(
                    sql.Identifier(table)
                )
            )
            assert cur.fetchall() == before[table], table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ("session_content", _BARRIER_COLUMNS),
        ("claude_hook_events", {"producer_seq": "bigint"}),
        (
            "compaction_receipts",
            {
                "session_id": "text",
                "receipt_key": "text",
                "boundary_uuid": "text",
                "boundary_kind": "text",
                "activation_id": "uuid",
                "verdict": "text",
                "reason": "text",
                "installed_messages_hash": "text",
                "observed_hash": "text",
                "source_topic": "text",
                "projection_cursor": "bigint",
                "projected_at": "timestamp with time zone",
            },
        ),
    ],
)
def test_column_types(
    applied: psycopg2.extensions.connection, table: str, expected: dict[str, str]
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'omninode_internal' AND table_name = %s",
            (table,),
        )
        columns = {name: (typ, nullable) for name, typ, nullable in cur.fetchall()}
        assert {name: columns[name][0] for name in expected} == expected
        if table != "compaction_receipts":
            assert all(columns[name][1] == "YES" for name in expected)
        else:
            assert set(columns) == set(expected)


def test_runtime_has_exact_receipts_table_schema_and_sequence_privileges(
    applied: psycopg2.extensions.connection,
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            "SELECT privilege_type FROM information_schema.role_table_grants "
            "WHERE table_schema = 'omninode_internal' AND table_name = 'compaction_receipts' "
            "AND grantee = 'omninode_runtime'"
        )
        assert {row[0] for row in cur.fetchall()} == {"SELECT", "INSERT", "UPDATE"}
        for privilege in (
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "TRUNCATE",
            "REFERENCES",
            "TRIGGER",
        ):
            cur.execute(
                "SELECT has_table_privilege('omninode_runtime', "
                "'omninode_internal.compaction_receipts', %s)",
                (privilege,),
            )
            assert cur.fetchone() == (privilege in {"SELECT", "INSERT", "UPDATE"},)
        for privilege in ("USAGE", "SELECT", "UPDATE"):
            cur.execute(
                "SELECT has_sequence_privilege('omninode_runtime', "
                "'omninode_internal.compaction_receipts_projection_cursor_seq', %s)",
                (privilege,),
            )
            assert cur.fetchone() == (privilege == "USAGE",)
        cur.execute(
            "SELECT has_schema_privilege('omninode_runtime', 'omninode_internal', 'USAGE'), "
            "has_schema_privilege('omninode_runtime', 'omninode_internal', 'CREATE')"
        )
        assert cur.fetchone() == (True, False)


def test_runtime_writer_can_upsert_and_return_a_receipt_with_its_cursor(
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            "CREATE ROLE compaction_receipts_test_writer LOGIN NOSUPERUSER NOBYPASSRLS "
            "IN ROLE omninode_runtime"
        )
    applied.commit()
    writer = ephemeral_postgres.connect(user="compaction_receipts_test_writer")
    try:
        writer.autocommit = True
        with writer.cursor() as cur:
            cur.execute("SET ROLE omninode_runtime")
            cur.execute(
                "SELECT current_user, rolsuper, rolbypassrls FROM pg_catalog.pg_roles "
                "WHERE rolname = current_user"
            )
            assert cur.fetchone() == ("omninode_runtime", False, False)
            for verdict in ("uncertain", "verified"):
                cur.execute(
                    "INSERT INTO omninode_internal.compaction_receipts "
                    "(session_id, receipt_key, verdict, projected_at) "
                    "VALUES ('session-1', 'receipt-1', %s, NOW()) "
                    "ON CONFLICT (session_id, receipt_key) DO UPDATE SET verdict = EXCLUDED.verdict "
                    "RETURNING verdict, projection_cursor",
                    (verdict,),
                )
                assert cur.fetchone() == (verdict, 1)
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("DELETE FROM omninode_internal.compaction_receipts")
    finally:
        writer.close()
