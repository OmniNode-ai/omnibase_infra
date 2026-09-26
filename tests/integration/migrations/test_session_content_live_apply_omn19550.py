# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live application proof for the session_content migration (OMN-19550)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)
from tests.integration.migrations.conftest import EphemeralPostgres

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION_RELATIVE_PATH = Path(
    "docker/migrations/forward/nodes/node_projection_session_content/"
    "0001_create_session_content.sql"
)
MIGRATION_FILE = REPO_ROOT / MIGRATION_RELATIVE_PATH
SESSION_CONTENT_COLUMNS = frozenset(
    {
        "event_id",
        "session_id",
        "turn_id",
        "correlation_id",
        "tool_use_id",
        "tool_name",
        "content_kind",
        "chunk_index",
        "chunk_count",
        "content",
        "command",
        "content_sha256",
        "original_chars",
        "truncated",
        "redaction_state",
        "producer_redaction",
        "hook_source",
        "emitted_at",
        "source_topic",
        "ingested_at",
    }
)


def _apply(pg: EphemeralPostgres) -> subprocess.CompletedProcess[str]:
    """Use the forward runner's psql invocation for the real vendored file."""
    return pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_FILE))


def _bootstrap(pg: EphemeralPostgres, *, runtime_role: bool = True) -> None:
    statements = ["CREATE SCHEMA omninode_internal;"]
    if runtime_role:
        statements.append("CREATE ROLE omninode_runtime NOLOGIN;")
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-c", "\n".join(statements))
    assert result.returncode == 0, result.stderr


def _column_names(pg: EphemeralPostgres) -> frozenset[str]:
    connection = pg.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'omninode_internal'
                  AND table_name = 'session_content'
                """
            )
            return frozenset(str(row[0]) for row in cursor.fetchall())
    finally:
        connection.close()


def _table_exists(pg: EphemeralPostgres) -> bool:
    connection = pg.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT to_regclass('omninode_internal.session_content') IS NOT NULL
                """
            )
            row = cursor.fetchone()
    finally:
        connection.close()
    assert row is not None
    return bool(row[0])


@pytest.mark.integration
def test_session_content_live_apply_creates_shape_and_runtime_acl(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """A fresh cluster gets the real table and exactly its writer privileges."""
    _bootstrap(ephemeral_postgres)

    result = _apply(ephemeral_postgres)

    assert result.returncode == 0, result.stderr
    assert _table_exists(ephemeral_postgres)
    assert _column_names(ephemeral_postgres) == SESSION_CONTENT_COLUMNS

    connection = ephemeral_postgres.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT array_agg(kcu.column_name::text ORDER BY kcu.ordinal_position)
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON kcu.constraint_name = tc.constraint_name
                 AND kcu.constraint_schema = tc.constraint_schema
                WHERE tc.table_schema = 'omninode_internal'
                  AND tc.table_name = 'session_content'
                  AND tc.constraint_type = 'PRIMARY KEY'
                GROUP BY tc.constraint_name
                """
            )
            primary_keys = cursor.fetchall()
            cursor.execute(
                """
                SELECT
                    has_schema_privilege(
                        'omninode_runtime', 'omninode_internal', 'USAGE'
                    ),
                    has_table_privilege(
                        'omninode_runtime',
                        'omninode_internal.session_content',
                        'SELECT'
                    ),
                    has_table_privilege(
                        'omninode_runtime',
                        'omninode_internal.session_content',
                        'INSERT'
                    ),
                    has_table_privilege(
                        'omninode_runtime',
                        'omninode_internal.session_content',
                        'UPDATE'
                    ),
                    has_table_privilege(
                        'omninode_runtime',
                        'omninode_internal.session_content',
                        'DELETE'
                    )
                """
            )
            privileges = cursor.fetchone()
    finally:
        connection.close()

    assert primary_keys == [(["event_id"],)]
    assert privileges == (True, True, True, True, False)


@pytest.mark.integration
def test_session_content_apply_is_idempotent(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """Reapplying the vendored file is a successful no-op on its column set."""
    _bootstrap(ephemeral_postgres)
    first = _apply(ephemeral_postgres)
    assert first.returncode == 0, first.stderr
    columns_after_first_apply = _column_names(ephemeral_postgres)

    second = _apply(ephemeral_postgres)

    assert second.returncode == 0, second.stderr
    assert _column_names(ephemeral_postgres) == columns_after_first_apply
    assert columns_after_first_apply == SESSION_CONTENT_COLUMNS


@pytest.mark.integration
def test_session_content_apply_refuses_a_drifted_table_before_any_grant(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """A row-holding drifted table fails the post-conditions, and gets no grant.

    The guarded adds can only add the missing NOT NULL columns as nullable, so
    the NOT NULL post-condition refuses the drifted shape under ON_ERROR_STOP.
    That refusal must land BEFORE the GRANT statements: the runtime role never
    receives write access to a relation whose shape the migration rejected.
    This is also why the corpus-wide OMN-15376 drift proof leaves
    omninode_internal tables out of its seed.
    """
    _bootstrap(ephemeral_postgres)
    seed = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        """
        CREATE TABLE omninode_internal.session_content (
            event_id TEXT PRIMARY KEY
        );
        INSERT INTO omninode_internal.session_content (event_id) VALUES ('drifted');
        """,
    )
    assert seed.returncode == 0, seed.stderr

    result = _apply(ephemeral_postgres)

    assert result.returncode != 0
    assert "division by zero" in result.stderr, result.stderr
    # The guarded adds ran (the reconciliation region is live) ...
    assert _column_names(ephemeral_postgres) == SESSION_CONTENT_COLUMNS
    # ... and the refusal came before the grants.
    connection = ephemeral_postgres.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    has_table_privilege(
                        'omninode_runtime',
                        'omninode_internal.session_content',
                        'INSERT'
                    ),
                    has_schema_privilege(
                        'omninode_runtime', 'omninode_internal', 'USAGE'
                    )
                """
            )
            privileges = cursor.fetchone()
    finally:
        connection.close()
    assert privileges == (False, False)


@pytest.mark.integration
def test_session_content_apply_rejects_missing_runtime_role(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """The role precondition fails before the migration can create the table."""
    _bootstrap(ephemeral_postgres, runtime_role=False)

    result = _apply(ephemeral_postgres)

    assert result.returncode != 0
    assert not _table_exists(ephemeral_postgres)


@pytest.mark.integration
def test_session_content_legacy_declaration_points_at_vendored_migration() -> None:
    """The temporary grant-derivation bridge identifies the file this applies."""
    declarations = [
        declaration
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if declaration.table.name == "session_content"
    ]

    assert len(declarations) == 1
    declaration = declarations[0]
    assert declaration.table.schema == "omninode_internal"
    assert declaration.table.database_ref == "application"
    assert declaration.table.access == "write"
    assert declaration.table.migration is not None
    assert REPO_ROOT / declaration.table.migration == MIGRATION_FILE
    assert (REPO_ROOT / declaration.table.migration).is_file()
    assert REPO_ROOT / declaration.contract_path == MIGRATION_FILE
