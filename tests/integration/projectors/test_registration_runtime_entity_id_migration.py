# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test for migration 117 on registration_projections.

Verifies (M1) that docker/migrations/forward/117_add_registration_runtime_entity_id.sql
adds the nullable ``runtime_entity_id UUID`` column (OMN-24495) to a table
created in the pre-117 shape without disturbing existing rows:

1. The table is rebuilt in its pre-117 shape (migration 001 plus the topic
   columns of migration 049), and rows are inserted with the old column set.
2. Migration 117 adds the column as nullable uuid with its partial index.
3. Every pre-existing row reads NULL for the column and is otherwise
   unchanged; the reader maps the old row's runtime_entity_id to None.
4. Re-running migration 117 is idempotent.

The table is dropped and re-created from the current schema file afterwards
so later tests in the session see the canonical shape.

Related Tickets:
    - OMN-24495: Add runtime_entity_id to registration projections
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumRegistrationState

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.projectors import ProjectionReaderRegistration

# Test markers (same convention as test_registration_projection_integration.py)
pytestmark = [
    pytest.mark.asyncio,
]


REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Full current schema (the same file the pg_pool fixture applies); used to
# restore the table for later tests after the migration is exercised.
SCHEMA_FILE = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "schemas"
    / "schema_registration_projection.sql"
)

# Pre-117 shape: migration 001 creates the table (and the registration_state
# enum, idempotently via DO $$ ... IF NOT EXISTS), migration 049 adds the
# topic columns the production table carried before 117 (OMN-2746).
OLD_SCHEMA_FILE = (
    REPO_ROOT / "docker" / "migrations" / "forward" / "001_registration_projection.sql"
)
TOPIC_COLUMNS_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "049_add_topic_columns_to_node_registrations.sql"
)
MIGRATION_117_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "117_add_registration_runtime_entity_id.sql"
)

# Column metadata lookup for the new column (empty before 117, present after).
_RUNTIME_ENTITY_ID_COLUMN_SQL = """
    SELECT data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'registration_projections'
      AND column_name = 'runtime_entity_id'
"""


class TestMigration117RuntimeEntityId:
    """M1: migration 117 adds the nullable column without touching old rows."""

    async def test_migration_adds_nullable_runtime_entity_id_preserving_rows(
        self,
        pg_pool: asyncpg.Pool,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify migration 117 on a pre-117 table: nullable column, rows kept.

        Rebuilds the table in its pre-117 shape, inserts old-shape rows, runs
        migration 117, and asserts the column exists (uuid, nullable), the
        partial index exists, all old rows read NULL for the new column and
        are otherwise unchanged, the reader maps an old row to None, and the
        migration is idempotent. Finally restores the canonical schema.
        """
        old_schema_sql = OLD_SCHEMA_FILE.read_text()
        topic_columns_sql = TOPIC_COLUMNS_FILE.read_text()
        migration_117_sql = MIGRATION_117_FILE.read_text()

        # -- Replace the fixture-created table with the pre-117 shape --------
        async with pg_pool.acquire() as conn:
            await conn.execute("DROP TABLE registration_projections CASCADE")
            # 001 re-creates the registration_state enum idempotently, so
            # running it after the fixture schema is safe.
            await conn.execute(old_schema_sql)
            await conn.execute(topic_columns_sql)

            pre_column = await conn.fetchrow(_RUNTIME_ENTITY_ID_COLUMN_SQL)
            assert pre_column is None, (
                "pre-117 table must not have a runtime_entity_id column"
            )

            # -- Insert rows using the old column set --------------------------
            now = datetime.now(UTC)
            inserted: dict[UUID, UUID] = {}
            for _ in range(2):
                entity_id = uuid4()
                event_id = uuid4()
                inserted[entity_id] = event_id
                await conn.execute(
                    """
                    INSERT INTO registration_projections (
                        entity_id,
                        domain,
                        current_state,
                        node_type,
                        node_version,
                        capabilities,
                        last_applied_event_id,
                        last_applied_offset,
                        registered_at,
                        updated_at,
                        correlation_id
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    entity_id,
                    "registration",
                    "active",
                    "effect",
                    "1.0.0",
                    '{"postgres": true, "read": true}',
                    event_id,
                    100,
                    now,
                    now,
                    uuid4(),
                )

            # -- Apply migration 117 --------------------------------------------
            await conn.execute(migration_117_sql)

            column = await conn.fetchrow(_RUNTIME_ENTITY_ID_COLUMN_SQL)
            assert column is not None, (
                "migration 117 must add the runtime_entity_id column"
            )
            assert column["data_type"] == "uuid"
            assert column["is_nullable"] == "YES"

            index = await conn.fetchrow(
                """
                SELECT indexdef
                FROM pg_indexes
                WHERE tablename = 'registration_projections'
                  AND indexname = 'idx_registration_runtime_entity_id'
                """
            )
            assert index is not None, (
                "migration 117 must create idx_registration_runtime_entity_id"
            )
            assert "runtime_entity_id IS NOT NULL" in index["indexdef"], (
                "the runtime parent index must be partial (non-null rows only)"
            )

            # -- Old rows read NULL for the new column, else unchanged ---------
            rows = await conn.fetch(
                """
                SELECT entity_id,
                       current_state,
                       node_type,
                       registered_at,
                       updated_at,
                       last_applied_event_id,
                       runtime_entity_id
                FROM registration_projections
                """
            )
            assert len(rows) == len(inserted)
            rows_by_entity = {row["entity_id"]: row for row in rows}
            assert set(rows_by_entity) == set(inserted)
            for entity_id, event_id in inserted.items():
                row = rows_by_entity[entity_id]
                assert row["runtime_entity_id"] is None
                assert row["current_state"] == "active"
                assert row["node_type"] == "effect"
                assert row["last_applied_event_id"] == event_id
                assert row["registered_at"] == now
                assert row["updated_at"] == now

        # -- Read one old row back through the reader --------------------------
        first_entity_id = next(iter(inserted))
        projection = await reader.get_entity_state(first_entity_id)
        assert projection is not None
        assert projection.runtime_entity_id is None
        assert projection.current_state == EnumRegistrationState.ACTIVE
        assert projection.node_type == "effect"

        # -- Idempotency: re-running migration 117 must be a no-op -------------
        async with pg_pool.acquire() as conn:
            await conn.execute(migration_117_sql)

            count = await conn.fetchval("SELECT count(*) FROM registration_projections")
            assert count == len(inserted)

            column = await conn.fetchrow(_RUNTIME_ENTITY_ID_COLUMN_SQL)
            assert column is not None
            assert column["data_type"] == "uuid"
            assert column["is_nullable"] == "YES"

        # -- Restore the table for later tests ----------------------------------
        async with pg_pool.acquire() as conn:
            await conn.execute("DROP TABLE registration_projections CASCADE")
            await conn.execute(SCHEMA_FILE.read_text())

            restored = await conn.fetchrow(_RUNTIME_ENTITY_ID_COLUMN_SQL)
            assert restored is not None, (
                "schema file must restore the table with the new column"
            )
