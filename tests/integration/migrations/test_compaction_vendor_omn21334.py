# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live-apply proofs for the barrier migration chains (OMN-21334)."""

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
)
_TABLES = ("session_content", "claude_hook_events")
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


def test_second_apply_preserves_populated_rows(
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
        assert all(columns[name][1] == "YES" for name in expected)
