# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S608
# Note: S608 (SQL injection) is disabled for this test file. All table names
# are UUID-generated locally by test fixtures, not from user input.
"""Integration tests for HandlerDb against remote PostgreSQL infrastructure.

These tests validate HandlerDb behavior against actual PostgreSQL infrastructure
running on the remote infrastructure server. They require proper database
credentials and will be skipped gracefully if the database is not available.

See tests/infrastructure_config.py for the default REMOTE_INFRA_HOST value.

CI/CD Graceful Skip Behavior
============================

These tests skip gracefully in CI/CD environments without database access:

Skip Conditions:
    - Skips if POSTGRES_HOST not set
    - Skips if POSTGRES_PASSWORD not set
    - Module-level ``pytestmark`` with ``pytest.mark.skipif`` used

Example CI/CD Output::

    $ pytest tests/integration/handlers/test_db_handler_integration.py -v
    test_db_describe SKIPPED (PostgreSQL not available - POSTGRES_PASSWORD not set)
    test_db_query_simple SKIPPED (PostgreSQL not available - POSTGRES_PASSWORD not set)

Test Categories
===============

- Connection Tests: Validate basic connectivity and handler lifecycle
- Query Tests: Verify SELECT operations with various inputs
- Execute Tests: Verify INSERT/UPDATE/DELETE and DDL operations
- Error Handling Tests: Validate proper error responses for invalid inputs

Single-Statement SQL Design
===========================

HandlerDb only supports **single SQL statements per call** due to asyncpg's
``execute()`` and ``fetch()`` methods. Multi-statement SQL (statements
separated by semicolons) will raise an error.

Tests that require multiple SQL operations (e.g., CREATE TABLE then INSERT)
must use separate ``execute()`` calls for each statement. This is intentional
and documented in the handler module docstring.

Example pattern used in these tests::

    # Step 1: Create table (separate call)
    create_envelope = {"operation": "db.execute", "payload": {"sql": "CREATE TABLE..."}}
    await handler.execute(create_envelope)

    # Step 2: Insert data (separate call)
    insert_envelope = {"operation": "db.execute", "payload": {"sql": "INSERT INTO..."}}
    await handler.execute(insert_envelope)

Environment Variables
=====================

    POSTGRES_HOST: PostgreSQL server hostname (required - skip if not set)
        Example: localhost or ${REMOTE_INFRA_HOST}
    POSTGRES_PORT: PostgreSQL server port (default: 5432)
    POSTGRES_DATABASE: Database name (default: omninode_bridge)
    POSTGRES_USER: Database username (default: postgres)
    POSTGRES_PASSWORD: Database password (required - tests skip if not set)

    See tests/infrastructure_config.py for REMOTE_INFRA_HOST default.

Related Ticket: OMN-816 - Create handler integration tests
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from omnibase_core.container import ModelONEXContainer
from tests.integration.handlers.conftest import POSTGRES_AVAILABLE

if TYPE_CHECKING:
    from omnibase_core.types import JsonType
    from omnibase_infra.handlers import HandlerDb


# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Handler default configuration values
# These match the defaults in HandlerDb and are tested to ensure consistency
DB_HANDLER_DEFAULT_POOL_SIZE = 5
DB_HANDLER_VERSION = "0.1.0-mvp"

# Module-level markers - skip all tests if PostgreSQL is not available
pytestmark = [
    pytest.mark.database,
    pytest.mark.postgres,
    pytest.mark.skipif(
        not POSTGRES_AVAILABLE,
        reason="PostgreSQL not available (POSTGRES_PASSWORD not set)",
    ),
]


# =============================================================================
# Connection Tests - Validate basic connectivity
# =============================================================================


class TestHandlerDbConnection:
    """Tests for HandlerDb connection and lifecycle management."""

    @pytest.mark.asyncio
    async def test_db_describe(
        self, db_config: dict[str, JsonType], mock_container: MagicMock
    ) -> None:
        """Verify describe() returns correct handler metadata."""
        from omnibase_infra.handlers import HandlerDb

        handler = HandlerDb(mock_container)
        await handler.initialize(db_config)

        try:
            description = handler.describe()

            assert description.handler_type == "database"
            assert "db.query" in description.supported_operations
            assert "db.execute" in description.supported_operations
            assert description.pool_size == DB_HANDLER_DEFAULT_POOL_SIZE
            assert description.initialized is True
            assert description.version == DB_HANDLER_VERSION
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_db_shutdown_cleans_up(
        self, db_config: dict[str, JsonType], mock_container: MagicMock
    ) -> None:
        """Verify shutdown properly closes connection pool.

        After shutdown, the handler should reject execute() calls with
        a RuntimeHostError indicating it is not initialized.
        """
        from omnibase_infra.errors import RuntimeHostError
        from omnibase_infra.handlers import HandlerDb

        handler = HandlerDb(mock_container)
        await handler.initialize(db_config)

        # Verify initialized by executing a simple query
        envelope = {
            "operation": "db.query",
            "payload": {"sql": "SELECT 1", "parameters": []},
        }
        result = await handler.execute(envelope)
        assert result.result.status == "success"

        # Shutdown
        await handler.shutdown()

        # Verify cleaned up - execute should fail after shutdown
        with pytest.raises(RuntimeHostError, match="not initialized"):
            await handler.execute(envelope)


# =============================================================================
# Query Tests - Validate SELECT operations
# =============================================================================


class TestHandlerDbQuery:
    """Tests for HandlerDb db.query operations (SELECT statements)."""

    @pytest.mark.asyncio
    async def test_db_query_simple(self, initialized_db_handler: HandlerDb) -> None:
        """Verify simple SELECT 1 query works.

        This is the most basic query test - if this fails, the database
        connection is likely misconfigured.
        """
        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": "SELECT 1 AS one",
                "parameters": [],
            },
        }

        result = await initialized_db_handler.execute(envelope)

        assert result.result.status == "success"
        assert result.result.payload.row_count == 1
        assert len(result.result.payload.rows) == 1
        assert result.result.payload.rows[0]["one"] == 1

    @pytest.mark.asyncio
    async def test_db_query_with_parameters(
        self, initialized_db_handler: HandlerDb
    ) -> None:
        """Verify parameterized query works correctly.

        Tests that query parameters are properly bound to prevent
        SQL injection and ensure correct value handling.
        """
        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": "SELECT $1::text AS name, $2::int AS value",
                "parameters": ["test_name", 42],
            },
        }

        result = await initialized_db_handler.execute(envelope)

        assert result.result.status == "success"
        assert result.result.payload.row_count == 1
        rows = result.result.payload.rows
        assert rows[0]["name"] == "test_name"
        assert rows[0]["value"] == 42

    @pytest.mark.asyncio
    async def test_db_query_multiple_rows(
        self, initialized_db_handler: HandlerDb
    ) -> None:
        """Verify query returning multiple rows works correctly."""
        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": """
                    SELECT * FROM (
                        VALUES (1, 'a'), (2, 'b'), (3, 'c')
                    ) AS t(num, letter)
                """,
                "parameters": [],
            },
        }

        result = await initialized_db_handler.execute(envelope)

        assert result.result.status == "success"
        assert result.result.payload.row_count == 3
        rows = result.result.payload.rows
        assert rows[0]["num"] == 1
        assert rows[1]["letter"] == "b"
        assert rows[2]["num"] == 3

    @pytest.mark.asyncio
    async def test_db_query_empty_result(
        self, initialized_db_handler: HandlerDb
    ) -> None:
        """Verify query returning no rows handles empty result correctly."""
        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": "SELECT 1 WHERE FALSE",
                "parameters": [],
            },
        }

        result = await initialized_db_handler.execute(envelope)

        assert result.result.status == "success"
        assert result.result.payload.row_count == 0
        assert len(result.result.payload.rows) == 0

    @pytest.mark.asyncio
    async def test_db_query_with_null_values(
        self, initialized_db_handler: HandlerDb
    ) -> None:
        """Verify NULL values are handled correctly in query results."""
        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": "SELECT NULL::text AS nullable_col, 'present'::text AS normal_col",
                "parameters": [],
            },
        }

        result = await initialized_db_handler.execute(envelope)

        assert result.result.status == "success"
        assert result.result.payload.row_count == 1
        row = result.result.payload.rows[0]
        assert row["nullable_col"] is None
        assert row["normal_col"] == "present"


# =============================================================================
# Execute Tests - Validate INSERT/UPDATE/DELETE and DDL operations
# =============================================================================


class TestHandlerDbExecute:
    """Tests for HandlerDb db.execute operations (INSERT/UPDATE/DELETE/DDL)."""

    @pytest.mark.asyncio
    async def test_db_execute_create_and_drop_table(
        self,
        initialized_db_handler: HandlerDb,
        unique_table_name: str,
    ) -> None:
        """Verify CREATE TABLE and DROP TABLE DDL operations work.

        Creates a test table, verifies it exists, then drops it.
        Uses unique table name to avoid conflicts with parallel tests.
        """
        # Create table
        create_envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": f"""
                    CREATE TABLE "{unique_table_name}" (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """,
                "parameters": [],
            },
        }

        create_result = await initialized_db_handler.execute(create_envelope)
        assert create_result.result.status == "success"

        try:
            # Verify table exists by querying information_schema
            check_envelope = {
                "operation": "db.query",
                "payload": {
                    "sql": """
                        SELECT table_name FROM information_schema.tables
                        WHERE table_name = $1
                    """,
                    "parameters": [unique_table_name],
                },
            }

            check_result = await initialized_db_handler.execute(check_envelope)
            assert check_result.result.payload.row_count == 1
            assert (
                check_result.result.payload.rows[0]["table_name"] == unique_table_name
            )

        finally:
            # Drop table (cleanup)
            drop_envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f'DROP TABLE IF EXISTS "{unique_table_name}"',
                    "parameters": [],
                },
            }
            await initialized_db_handler.execute(drop_envelope)

    @pytest.mark.asyncio
    async def test_db_execute_insert_and_query(
        self,
        initialized_db_handler: HandlerDb,
        unique_table_name: str,
    ) -> None:
        """Verify INSERT followed by SELECT returns inserted data.

        Creates a table, inserts rows, queries them, then cleans up.
        Tests the complete write-then-read cycle.
        """
        # Create table
        create_envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": f"""
                    CREATE TABLE "{unique_table_name}" (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        value INT
                    )
                """,
                "parameters": [],
            },
        }
        await initialized_db_handler.execute(create_envelope)

        try:
            # Insert rows
            insert_envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f"""
                        INSERT INTO "{unique_table_name}" (name, value)
                        VALUES ($1, $2), ($3, $4), ($5, $6)
                    """,
                    "parameters": ["alice", 10, "bob", 20, "charlie", 30],
                },
            }

            insert_result = await initialized_db_handler.execute(insert_envelope)
            assert insert_result.result.status == "success"
            # asyncpg returns "INSERT 0 3" for 3 rows inserted
            assert insert_result.result.payload.row_count == 3

            # Query inserted rows
            query_envelope = {
                "operation": "db.query",
                "payload": {
                    "sql": f"""
                        SELECT name, value FROM "{unique_table_name}"
                        ORDER BY value
                    """,
                    "parameters": [],
                },
            }

            query_result = await initialized_db_handler.execute(query_envelope)
            assert query_result.result.status == "success"
            assert query_result.result.payload.row_count == 3

            rows = query_result.result.payload.rows
            assert rows[0]["name"] == "alice"
            assert rows[0]["value"] == 10
            assert rows[1]["name"] == "bob"
            assert rows[2]["name"] == "charlie"

        finally:
            # Cleanup
            drop_envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f'DROP TABLE IF EXISTS "{unique_table_name}"',
                    "parameters": [],
                },
            }
            await initialized_db_handler.execute(drop_envelope)

    @pytest.mark.asyncio
    async def test_db_execute_update(
        self,
        initialized_db_handler: HandlerDb,
        unique_table_name: str,
    ) -> None:
        """Verify UPDATE operation modifies existing rows.

        Note: This test uses separate execute() calls for CREATE TABLE and INSERT
        because HandlerDb only supports single-statement SQL per call (asyncpg
        limitation). See module docstring "Single-Statement SQL Design" for details.
        """
        # Step 1: Create table (asyncpg requires single statement per execute call)
        create_envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": f"""
                    CREATE TABLE "{unique_table_name}" (
                        id SERIAL PRIMARY KEY,
                        status TEXT NOT NULL
                    )
                """,
                "parameters": [],
            },
        }
        await initialized_db_handler.execute(create_envelope)

        # Step 2: Insert initial data (asyncpg requires single statement per execute call)
        insert_envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": f"""
                    INSERT INTO "{unique_table_name}" (status)
                    VALUES ('pending'), ('pending'), ('done')
                """,
                "parameters": [],
            },
        }
        await initialized_db_handler.execute(insert_envelope)

        try:
            # Update rows
            update_envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f"""
                        UPDATE "{unique_table_name}"
                        SET status = $1
                        WHERE status = $2
                    """,
                    "parameters": ["completed", "pending"],
                },
            }

            update_result = await initialized_db_handler.execute(update_envelope)
            assert update_result.result.status == "success"
            # 2 rows updated: the two 'pending' rows from INSERT above changed to 'completed'
            assert update_result.result.payload.row_count == 2

            # Verify update
            query_envelope = {
                "operation": "db.query",
                "payload": {
                    "sql": f"""
                        SELECT status, COUNT(*) as cnt
                        FROM "{unique_table_name}"
                        GROUP BY status
                        ORDER BY status
                    """,
                    "parameters": [],
                },
            }

            query_result = await initialized_db_handler.execute(query_envelope)
            rows = query_result.result.payload.rows
            # Should have 2 'completed' and 1 'done'
            assert any(r["status"] == "completed" and r["cnt"] == 2 for r in rows)
            assert any(r["status"] == "done" and r["cnt"] == 1 for r in rows)

        finally:
            # Cleanup
            drop_envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f'DROP TABLE IF EXISTS "{unique_table_name}"',
                    "parameters": [],
                },
            }
            await initialized_db_handler.execute(drop_envelope)

    @pytest.mark.asyncio
    async def test_db_execute_delete(
        self,
        initialized_db_handler: HandlerDb,
        unique_table_name: str,
    ) -> None:
        """Verify DELETE operation removes rows.

        Note: This test uses separate execute() calls for CREATE TABLE and INSERT
        because HandlerDb only supports single-statement SQL per call (asyncpg
        limitation). See module docstring "Single-Statement SQL Design" for details.
        """
        # Step 1: Create table (asyncpg requires single statement per execute call)
        create_envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": f"""
                    CREATE TABLE "{unique_table_name}" (
                        id SERIAL PRIMARY KEY,
                        keep BOOLEAN NOT NULL
                    )
                """,
                "parameters": [],
            },
        }
        await initialized_db_handler.execute(create_envelope)

        # Step 2: Insert initial data (asyncpg requires single statement per execute call)
        insert_envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": f"""
                    INSERT INTO "{unique_table_name}" (keep)
                    VALUES (TRUE), (FALSE), (FALSE), (TRUE)
                """,
                "parameters": [],
            },
        }
        await initialized_db_handler.execute(insert_envelope)

        try:
            # Delete rows
            delete_envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f"""
                        DELETE FROM "{unique_table_name}"
                        WHERE keep = FALSE
                    """,
                    "parameters": [],
                },
            }

            delete_result = await initialized_db_handler.execute(delete_envelope)
            assert delete_result.result.status == "success"
            # 2 rows deleted: the two FALSE rows from INSERT above (TRUE, FALSE, FALSE, TRUE)
            assert delete_result.result.payload.row_count == 2

            # Verify delete
            query_envelope = {
                "operation": "db.query",
                "payload": {
                    "sql": f'SELECT COUNT(*) as cnt FROM "{unique_table_name}"',
                    "parameters": [],
                },
            }

            query_result = await initialized_db_handler.execute(query_envelope)
            # 2 rows remain: the two TRUE rows from original INSERT (TRUE, FALSE, FALSE, TRUE)
            assert query_result.result.payload.rows[0]["cnt"] == 2

        finally:
            # Cleanup
            drop_envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f'DROP TABLE IF EXISTS "{unique_table_name}"',
                    "parameters": [],
                },
            }
            await initialized_db_handler.execute(drop_envelope)


# =============================================================================
# Error Handling Tests - Validate proper error responses
# =============================================================================


class TestHandlerDbErrors:
    """Tests for HandlerDb error handling."""

    @pytest.mark.asyncio
    async def test_db_query_syntax_error(
        self, initialized_db_handler: HandlerDb
    ) -> None:
        """Verify SQL syntax error is properly reported."""
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": "SELECTT * FROM nonexistent",  # Typo in SELECT
                "parameters": [],
            },
        }

        with pytest.raises(RuntimeHostError, match="SQL syntax error"):
            await initialized_db_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_db_query_table_not_found(
        self, initialized_db_handler: HandlerDb
    ) -> None:
        """Verify table not found error is properly reported."""
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": "SELECT * FROM nonexistent_table_xyz_12345",
                "parameters": [],
            },
        }

        with pytest.raises(RuntimeHostError, match="Table not found"):
            await initialized_db_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_db_query_column_not_found(
        self,
        initialized_db_handler: HandlerDb,
        unique_table_name: str,
    ) -> None:
        """Verify column not found error is properly reported."""
        from omnibase_infra.errors import RuntimeHostError

        # Create simple table
        create_envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": f'CREATE TABLE "{unique_table_name}" (id INT)',
                "parameters": [],
            },
        }
        await initialized_db_handler.execute(create_envelope)

        try:
            # Query nonexistent column
            envelope = {
                "operation": "db.query",
                "payload": {
                    "sql": f'SELECT nonexistent_column FROM "{unique_table_name}"',
                    "parameters": [],
                },
            }

            with pytest.raises(RuntimeHostError, match="Column not found"):
                await initialized_db_handler.execute(envelope)

        finally:
            # Cleanup
            drop_envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f'DROP TABLE IF EXISTS "{unique_table_name}"',
                    "parameters": [],
                },
            }
            await initialized_db_handler.execute(drop_envelope)

    @pytest.mark.asyncio
    async def test_db_execute_not_initialized(self, mock_container: MagicMock) -> None:
        """Verify execute fails when handler not initialized."""
        from omnibase_infra.errors import RuntimeHostError
        from omnibase_infra.handlers import HandlerDb

        handler = HandlerDb(mock_container)  # Not initialized

        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": "SELECT 1",
                "parameters": [],
            },
        }

        with pytest.raises(RuntimeHostError, match="not initialized"):
            await handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_db_execute_missing_operation(
        self, initialized_db_handler: HandlerDb
    ) -> None:
        """Verify missing operation in envelope is properly reported."""
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            # Missing "operation" key
            "payload": {
                "sql": "SELECT 1",
                "parameters": [],
            },
        }

        with pytest.raises(RuntimeHostError, match="operation"):
            await initialized_db_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_db_execute_unsupported_operation(
        self, initialized_db_handler: HandlerDb
    ) -> None:
        """Verify unsupported operation is properly reported."""
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            "operation": "db.transaction",  # Not supported in MVP
            "payload": {
                "sql": "BEGIN",
                "parameters": [],
            },
        }

        with pytest.raises(RuntimeHostError, match="not supported"):
            await initialized_db_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_db_execute_missing_sql(
        self, initialized_db_handler: HandlerDb
    ) -> None:
        """Verify missing SQL in payload is properly reported."""
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            "operation": "db.query",
            "payload": {
                # Missing "sql" key
                "parameters": [],
            },
        }

        with pytest.raises(RuntimeHostError, match="sql"):
            await initialized_db_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_db_execute_unique_constraint_violation(
        self,
        initialized_db_handler: HandlerDb,
        unique_table_name: str,
    ) -> None:
        """Verify unique constraint violation is properly reported."""
        from omnibase_infra.errors import RuntimeHostError

        # Create table with unique constraint
        create_envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": f"""
                    CREATE TABLE "{unique_table_name}" (
                        id INT PRIMARY KEY,
                        name TEXT UNIQUE
                    )
                """,
                "parameters": [],
            },
        }
        await initialized_db_handler.execute(create_envelope)

        try:
            # Insert first row
            insert1_envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f'INSERT INTO "{unique_table_name}" (id, name) VALUES ($1, $2)',
                    "parameters": [1, "unique_name"],
                },
            }
            await initialized_db_handler.execute(insert1_envelope)

            # Try to insert duplicate
            insert2_envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f'INSERT INTO "{unique_table_name}" (id, name) VALUES ($1, $2)',
                    "parameters": [2, "unique_name"],  # Same name
                },
            }

            with pytest.raises(RuntimeHostError, match="Unique constraint violation"):
                await initialized_db_handler.execute(insert2_envelope)

        finally:
            # Cleanup
            drop_envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f'DROP TABLE IF EXISTS "{unique_table_name}"',
                    "parameters": [],
                },
            }
            await initialized_db_handler.execute(drop_envelope)


# =============================================================================
# Correlation ID Tests - Validate request tracing
# =============================================================================


class TestHandlerDbCorrelationId:
    """Tests for correlation ID handling in HandlerDb."""

    @pytest.mark.asyncio
    async def test_db_query_preserves_correlation_id(
        self, initialized_db_handler: HandlerDb
    ) -> None:
        """Verify correlation_id from envelope is preserved in response."""
        from uuid import uuid4

        test_correlation_id = uuid4()

        envelope = {
            "operation": "db.query",
            "correlation_id": str(test_correlation_id),
            "payload": {
                "sql": "SELECT 1 AS one",
                "parameters": [],
            },
        }

        result = await initialized_db_handler.execute(envelope)

        assert result.correlation_id == test_correlation_id
        assert result.result.correlation_id == test_correlation_id

    @pytest.mark.asyncio
    async def test_db_query_generates_correlation_id_if_missing(
        self, initialized_db_handler: HandlerDb
    ) -> None:
        """Verify correlation_id is generated when not provided."""
        from uuid import UUID

        envelope = {
            "operation": "db.query",
            # No correlation_id provided
            "payload": {
                "sql": "SELECT 1 AS one",
                "parameters": [],
            },
        }

        result = await initialized_db_handler.execute(envelope)

        # Should have a generated correlation_id
        assert result.correlation_id is not None
        assert isinstance(result.correlation_id, UUID)
