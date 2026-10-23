# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live-apply proofs for the vendored compaction receipts chain (OMN-21336)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import psycopg2
import psycopg2.errors
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.ephemeral_pg]

_CHAIN = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/nodes/node_projection_compaction_receipts"
)
_COLUMNS = {
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
}


def _apply_migrations(pg: EphemeralPostgres) -> None:
    migrations = sorted(_CHAIN.glob("*.sql"))
    assert [int(p.name[:4]) for p in migrations] == [0, 1]
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


def test_second_apply_preserves_a_populated_receipt_and_its_cursor(
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            """
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
        cur.execute("SELECT * FROM omninode_internal.compaction_receipts")
        before = cur.fetchall()
        assert len(before) == 1
    # The separate psql apply needs table locks: release this transaction first.
    applied.commit()
    _apply_migrations(ephemeral_postgres)
    with applied.cursor() as cur:
        cur.execute("SELECT * FROM omninode_internal.compaction_receipts")
        assert cur.fetchall() == before


def test_column_types_and_no_extra_columns(
    applied: psycopg2.extensions.connection,
) -> None:
    with applied.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'omninode_internal' "
            "AND table_name = 'compaction_receipts'"
        )
        assert dict(cur.fetchall()) == _COLUMNS


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
