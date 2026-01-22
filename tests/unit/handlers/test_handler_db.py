# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for HandlerDb.

Comprehensive test suite covering initialization, query/execute operations,
error handling, describe, and lifecycle management.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import asyncpg
import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.handlers.handler_db import HandlerDb
from tests.helpers import filter_handler_warnings


@pytest.fixture
def mock_container() -> MagicMock:
    """Create mock ONEX container for HandlerDb tests."""
    return MagicMock(spec=ModelONEXContainer)


class TestHandlerDbInitialization:
    """Test suite for HandlerDb initialization."""

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    def test_handler_init_default_state(self, handler: HandlerDb) -> None:
        """Test handler initializes in uninitialized state."""
        assert handler._initialized is False
        assert handler._pool is None
        assert handler._pool_size == 5
        assert handler._timeout == 30.0

    def test_handler_type_returns_infra_handler(self, handler: HandlerDb) -> None:
        """Test handler_type property returns EnumHandlerType.INFRA_HANDLER."""
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category_returns_effect(self, handler: HandlerDb) -> None:
        """Test handler_category property returns EnumHandlerTypeCategory.EFFECT."""
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    @pytest.mark.asyncio
    async def test_initialize_missing_dsn_raises_error(
        self, handler: HandlerDb
    ) -> None:
        """Test initialize without DSN raises RuntimeHostError."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.initialize({})

        assert "dsn" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_empty_dsn_raises_error(self, handler: HandlerDb) -> None:
        """Test initialize with empty DSN raises RuntimeHostError."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.initialize({"dsn": ""})

        assert "dsn" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_creates_pool(self, handler: HandlerDb) -> None:
        """Test initialize creates asyncpg connection pool."""
        mock_pool = MagicMock(spec=asyncpg.Pool)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            config: dict[str, object] = {"dsn": "postgresql://user:pass@localhost/db"}
            await handler.initialize(config)

            assert handler._initialized is True
            assert handler._pool is mock_pool
            mock_create.assert_called_once_with(
                dsn="postgresql://user:pass@localhost/db",
                min_size=1,
                max_size=5,
                command_timeout=30.0,
            )

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_custom_timeout(self, handler: HandlerDb) -> None:
        """Test initialize respects custom timeout."""
        mock_pool = MagicMock(spec=asyncpg.Pool)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            config: dict[str, object] = {
                "dsn": "postgresql://localhost/db",
                "timeout": 60.0,
            }
            await handler.initialize(config)

            assert handler._timeout == 60.0
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["command_timeout"] == 60.0

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_connection_error_raises_infra_error(
        self, handler: HandlerDb
    ) -> None:
        """Test connection error during initialize raises InfraConnectionError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = OSError("Connection refused")

            config: dict[str, object] = {"dsn": "postgresql://localhost/db"}

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.initialize(config)

            assert "connect" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_invalid_password_raises_error(
        self, handler: HandlerDb
    ) -> None:
        """Test invalid password raises InfraAuthenticationError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = asyncpg.InvalidPasswordError("Invalid password")

            config: dict[str, object] = {"dsn": "postgresql://localhost/db"}

            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.initialize(config)

            assert "authentication" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_invalid_database_raises_error(
        self, handler: HandlerDb
    ) -> None:
        """Test invalid database name raises RuntimeHostError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = asyncpg.InvalidCatalogNameError(
                "Database not found"
            )

            config: dict[str, object] = {"dsn": "postgresql://localhost/nonexistent"}

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.initialize(config)

            assert "database" in str(exc_info.value).lower()


class TestHandlerDbQueryOperations:
    """Test suite for db.query operations."""

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        return MagicMock(spec=asyncpg.Pool)

    @pytest.mark.asyncio
    async def test_query_successful_response(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test successful query returns correct response structure."""
        # Setup mock connection and rows
        mock_conn = AsyncMock()
        mock_rows = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        # Setup pool context manager
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            correlation_id = uuid4()
            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELECT id, name FROM users"},
                "correlation_id": correlation_id,
            }

            output = await handler.execute(envelope)
            result = output.result  # ModelDbQueryResponse

            assert result.status == "success"
            assert result.payload.row_count == 2
            assert len(result.payload.rows) == 2
            assert result.payload.rows[0] == {"id": 1, "name": "Alice"}
            assert result.payload.rows[1] == {"id": 2, "name": "Bob"}
            assert result.correlation_id == correlation_id
            assert output.correlation_id == correlation_id

            mock_conn.fetch.assert_called_once_with("SELECT id, name FROM users")

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_query_with_parameters(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test query with parameterized SQL."""
        mock_conn = AsyncMock()
        mock_rows = [{"id": 1, "name": "Alice"}]
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {
                    "sql": "SELECT id, name FROM users WHERE id = $1",
                    "parameters": [1],
                },
            }

            output = await handler.execute(envelope)
            result = output.result  # ModelDbQueryResponse

            mock_conn.fetch.assert_called_once_with(
                "SELECT id, name FROM users WHERE id = $1", 1
            )

            assert result.payload.row_count == 1

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_query_empty_result(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test query returning no rows."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELECT * FROM empty_table"},
            }

            output = await handler.execute(envelope)
            result = output.result  # ModelDbQueryResponse

            assert result.payload.row_count == 0
            assert result.payload.rows == []

            await handler.shutdown()


class TestHandlerDbExecuteOperations:
    """Test suite for db.execute operations."""

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        return MagicMock(spec=asyncpg.Pool)

    @pytest.mark.asyncio
    async def test_execute_insert_successful(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test successful INSERT returns correct row count."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.execute",
                "payload": {
                    "sql": "INSERT INTO users (name) VALUES ($1)",
                    "parameters": ["Charlie"],
                },
            }

            output = await handler.execute(envelope)
            result = output.result  # ModelDbQueryResponse

            assert result.status == "success"
            assert result.payload.row_count == 1
            assert result.payload.rows == []

            mock_conn.execute.assert_called_once_with(
                "INSERT INTO users (name) VALUES ($1)", "Charlie"
            )

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_update_multiple_rows(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test UPDATE affecting multiple rows."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 5")

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.execute",
                "payload": {
                    "sql": "UPDATE users SET active = $1 WHERE status = $2",
                    "parameters": [True, "pending"],
                },
            }

            output = await handler.execute(envelope)
            result = output.result  # ModelDbQueryResponse

            assert result.payload.row_count == 5

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_delete(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test DELETE statement."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 3")

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.execute",
                "payload": {"sql": "DELETE FROM users WHERE inactive = true"},
            }

            output = await handler.execute(envelope)
            result = output.result  # ModelDbQueryResponse

            assert result.payload.row_count == 3

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_no_rows_affected(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test execute with no rows affected."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 0")

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.execute",
                "payload": {
                    "sql": "UPDATE users SET name = $1 WHERE id = $2",
                    "parameters": ["Test", 99999],
                },
            }

            output = await handler.execute(envelope)
            result = output.result  # ModelDbQueryResponse

            assert result.payload.row_count == 0

            await handler.shutdown()


class TestHandlerDbErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        return MagicMock(spec=asyncpg.Pool)

    @pytest.mark.asyncio
    async def test_query_timeout_raises_infra_timeout(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test query timeout raises InfraTimeoutError."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            side_effect=asyncpg.QueryCanceledError("query timeout")
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELECT * FROM slow_query"},
            }

            with pytest.raises(InfraTimeoutError) as exc_info:
                await handler.execute(envelope)

            assert "timed out" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_connection_lost_raises_infra_connection(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test connection loss raises InfraConnectionError."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            side_effect=asyncpg.PostgresConnectionError("connection lost")
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELECT 1"},
            }

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.execute(envelope)

            assert "connection" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_syntax_error_raises_runtime_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test SQL syntax error raises RuntimeHostError."""
        mock_conn = AsyncMock()
        error = asyncpg.PostgresSyntaxError("syntax error")
        error.message = "syntax error at or near 'SELEKT'"
        mock_conn.fetch = AsyncMock(side_effect=error)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELEKT * FROM users"},
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "syntax" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_undefined_table_raises_runtime_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test undefined table raises RuntimeHostError."""
        mock_conn = AsyncMock()
        error = asyncpg.UndefinedTableError("table not found")
        error.message = 'relation "nonexistent" does not exist'
        mock_conn.fetch = AsyncMock(side_effect=error)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELECT * FROM nonexistent"},
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "table" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_unique_violation_raises_runtime_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test unique constraint violation raises RuntimeHostError."""
        mock_conn = AsyncMock()
        error = asyncpg.UniqueViolationError("unique violation")
        error.message = "duplicate key value violates unique constraint"
        mock_conn.execute = AsyncMock(side_effect=error)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.execute",
                "payload": {
                    "sql": "INSERT INTO users (email) VALUES ($1)",
                    "parameters": ["duplicate@example.com"],
                },
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "unique" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_foreign_key_violation_raises_runtime_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test foreign key constraint violation raises RuntimeHostError."""
        mock_conn = AsyncMock()
        error = asyncpg.ForeignKeyViolationError("foreign key violation")
        error.message = "insert or update on table violates foreign key constraint"
        mock_conn.execute = AsyncMock(side_effect=error)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.execute",
                "payload": {
                    "sql": "INSERT INTO orders (user_id) VALUES ($1)",
                    "parameters": [99999],  # Non-existent user
                },
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "foreign key" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_not_null_violation_raises_runtime_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test not null constraint violation raises RuntimeHostError."""
        mock_conn = AsyncMock()
        error = asyncpg.NotNullViolationError("not null violation")
        error.message = "null value in column violates not-null constraint"
        mock_conn.execute = AsyncMock(side_effect=error)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.execute",
                "payload": {
                    "sql": "INSERT INTO users (name) VALUES ($1)",
                    "parameters": [None],  # Null for required field
                },
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "not null" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_check_violation_raises_runtime_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test check constraint violation raises RuntimeHostError."""
        mock_conn = AsyncMock()
        error = asyncpg.CheckViolationError("check violation")
        error.message = "new row violates check constraint"
        mock_conn.execute = AsyncMock(side_effect=error)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.execute",
                "payload": {
                    "sql": "INSERT INTO products (price) VALUES ($1)",
                    "parameters": [-10],  # Negative price violates check constraint
                },
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "check" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_unsupported_operation_raises_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test unsupported operation raises RuntimeHostError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.transaction",
                "payload": {"sql": "BEGIN"},
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "db.transaction" in str(exc_info.value)
            assert "not supported" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_missing_sql_raises_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test missing SQL field raises RuntimeHostError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"parameters": [1, 2]},  # No SQL
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "sql" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_empty_sql_raises_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test empty SQL field raises RuntimeHostError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "  "},  # Whitespace only
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "sql" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_invalid_parameters_type_raises_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test invalid parameters type raises RuntimeHostError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {
                    "sql": "SELECT * FROM users WHERE id = $1",
                    "parameters": "not-a-list",  # Invalid type
                },
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "parameters" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_missing_operation_raises_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test missing operation field raises RuntimeHostError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "payload": {"sql": "SELECT 1"},
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "operation" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_missing_payload_raises_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test missing payload field raises RuntimeHostError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.query",
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "payload" in str(exc_info.value).lower()

            await handler.shutdown()


class TestHandlerDbDescribe:
    """Test suite for describe operations."""

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    def test_describe_returns_handler_metadata(self, handler: HandlerDb) -> None:
        """Test describe returns correct handler metadata."""
        description = handler.describe()

        assert description.handler_type == "infra_handler"
        assert description.handler_category == "effect"
        assert description.pool_size == 5
        assert description.timeout_seconds == 30.0
        assert description.version == "0.1.0-mvp"
        assert description.initialized is False

    def test_describe_lists_supported_operations(self, handler: HandlerDb) -> None:
        """Test describe lists supported operations."""
        description = handler.describe()

        assert "db.query" in description.supported_operations
        assert "db.execute" in description.supported_operations
        assert len(description.supported_operations) == 2

    @pytest.mark.asyncio
    async def test_describe_reflects_initialized_state(
        self, handler: HandlerDb
    ) -> None:
        """Test describe shows correct initialized state."""
        mock_pool = MagicMock(spec=asyncpg.Pool)

        assert handler.describe().initialized is False

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})
            assert handler.describe().initialized is True

            await handler.shutdown()
            assert handler.describe().initialized is False


class TestHandlerDbLifecycle:
    """Test suite for lifecycle management."""

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        pool = MagicMock(spec=asyncpg.Pool)
        pool.close = AsyncMock()
        return pool

    @pytest.mark.asyncio
    async def test_shutdown_closes_pool(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test shutdown closes the connection pool properly."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            await handler.shutdown()

            mock_pool.close.assert_called_once()
            assert handler._pool is None
            assert handler._initialized is False

    @pytest.mark.asyncio
    async def test_execute_after_shutdown_raises_error(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test execute after shutdown raises RuntimeHostError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})
            await handler.shutdown()

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELECT 1"},
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_before_initialize_raises_error(
        self, handler: HandlerDb
    ) -> None:
        """Test execute before initialize raises RuntimeHostError."""
        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": {"sql": "SELECT 1"},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_multiple_shutdown_calls_safe(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test multiple shutdown calls are safe (idempotent)."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})
            await handler.shutdown()
            await handler.shutdown()  # Second call should not raise

            assert handler._initialized is False
            assert handler._pool is None

    @pytest.mark.asyncio
    async def test_reinitialize_after_shutdown(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test handler can be reinitialized after shutdown."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})
            await handler.shutdown()

            assert handler._initialized is False

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            assert handler._initialized is True
            assert handler._pool is not None

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_called_once_per_lifecycle(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test that initialize creates pool exactly once per call.

        Acceptance criteria for OMN-252: Asserts handler initialized exactly once.
        Each call to initialize() should create a new pool via asyncpg.create_pool().
        """
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            # First initialize
            await handler.initialize({"dsn": "postgresql://localhost/db"})
            assert mock_create.call_count == 1

            # Shutdown and reinitialize
            await handler.shutdown()
            await handler.initialize({"dsn": "postgresql://localhost/db"})
            assert mock_create.call_count == 2  # Called again for reinit

            await handler.shutdown()


class TestHandlerDbCorrelationId:
    """Test suite for correlation ID handling."""

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        return MagicMock(spec=asyncpg.Pool)

    @pytest.mark.asyncio
    async def test_correlation_id_from_envelope_uuid(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test correlation ID extracted from envelope as UUID."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            correlation_id = uuid4()
            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELECT 1"},
                "correlation_id": correlation_id,
            }

            output = await handler.execute(envelope)
            result = output.result  # ModelDbQueryResponse

            assert result.correlation_id == correlation_id
            assert output.correlation_id == correlation_id

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_correlation_id_from_envelope_string(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test correlation ID extracted from envelope as string."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            correlation_id = str(uuid4())
            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELECT 1"},
                "correlation_id": correlation_id,
            }

            output = await handler.execute(envelope)
            result = output.result  # ModelDbQueryResponse

            # String correlation_id is converted to UUID by handler
            assert result.correlation_id == UUID(correlation_id)
            assert output.correlation_id == UUID(correlation_id)

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_correlation_id_generated_when_missing(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test correlation ID generated when not in envelope."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELECT 1"},
            }

            output = await handler.execute(envelope)
            result = output.result  # ModelDbQueryResponse

            # Should have a generated UUID
            assert isinstance(result.correlation_id, UUID)
            assert isinstance(output.correlation_id, UUID)
            # Correlation IDs should match between output wrapper and result
            assert output.correlation_id == result.correlation_id

            await handler.shutdown()


class TestHandlerDbDsnSecurity:
    """Test suite for DSN security and sanitization.

    Security Policy: DSN contains credentials and must NEVER be exposed in:
    - Error messages
    - Log output
    - describe() metadata

    See HandlerDb class docstring "Security Policy - DSN Handling" for full policy.
    """

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    def test_sanitize_dsn_removes_password(self, handler: HandlerDb) -> None:
        """Test _sanitize_dsn replaces password with asterisks."""
        # Standard format with password
        dsn = "postgresql://user:secret123@localhost:5432/mydb"
        sanitized = handler._sanitize_dsn(dsn)
        assert "secret123" not in sanitized
        assert "***" in sanitized
        assert "user" in sanitized
        assert "localhost" in sanitized

    def test_sanitize_dsn_handles_special_characters(self, handler: HandlerDb) -> None:
        """Test _sanitize_dsn handles passwords with special characters."""
        dsn = "postgresql://admin:p@ss!word#123@db.example.com:5432/prod"
        sanitized = handler._sanitize_dsn(dsn)
        assert "p@ss!word#123" not in sanitized
        assert "***" in sanitized

    def test_sanitize_dsn_preserves_structure(self, handler: HandlerDb) -> None:
        """Test _sanitize_dsn preserves DSN structure for debugging."""
        dsn = "postgresql://user:password@host:5432/database"
        sanitized = handler._sanitize_dsn(dsn)
        # Should preserve user, host, port, database
        assert sanitized == "postgresql://user:***@host:5432/database"

    @pytest.mark.asyncio
    async def test_connection_error_does_not_expose_dsn(
        self, handler: HandlerDb
    ) -> None:
        """Test that connection errors do NOT expose DSN credentials."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = OSError("Connection refused")

            secret_password = "my_super_secret_password_12345"
            dsn = f"postgresql://user:{secret_password}@localhost/db"

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.initialize({"dsn": dsn})

            error_str = str(exc_info.value)
            # Password must NOT appear in error message
            assert secret_password not in error_str
            # DSN must NOT appear in error message
            assert dsn not in error_str
            # Generic message should be present
            assert "check host and port" in error_str.lower()

    @pytest.mark.asyncio
    async def test_auth_error_does_not_expose_dsn(self, handler: HandlerDb) -> None:
        """Test that authentication errors do NOT expose DSN credentials."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = asyncpg.InvalidPasswordError("Invalid password")

            secret_password = "my_super_secret_password_67890"
            dsn = f"postgresql://user:{secret_password}@localhost/db"

            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.initialize({"dsn": dsn})

            error_str = str(exc_info.value)
            # Password must NOT appear in error message
            assert secret_password not in error_str
            # DSN must NOT appear in error message
            assert dsn not in error_str
            # Generic message should be present
            assert "check credentials" in error_str.lower()

    def test_describe_does_not_expose_dsn(self, handler: HandlerDb) -> None:
        """Test that describe() does NOT include DSN."""
        description = handler.describe()

        # DSN must NOT be in describe response
        desc_str = str(description)
        assert "dsn" not in desc_str.lower()
        assert "password" not in desc_str.lower()
        assert "postgresql://" not in desc_str


class TestHandlerDbRowCountParsing:
    """Test suite for row count parsing."""

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    def test_parse_insert_row_count(self, handler: HandlerDb) -> None:
        """Test parsing INSERT row count."""
        assert handler._parse_row_count("INSERT 0 1") == 1
        assert handler._parse_row_count("INSERT 0 5") == 5
        assert handler._parse_row_count("INSERT 0 100") == 100

    def test_parse_update_row_count(self, handler: HandlerDb) -> None:
        """Test parsing UPDATE row count."""
        assert handler._parse_row_count("UPDATE 1") == 1
        assert handler._parse_row_count("UPDATE 10") == 10
        assert handler._parse_row_count("UPDATE 0") == 0

    def test_parse_delete_row_count(self, handler: HandlerDb) -> None:
        """Test parsing DELETE row count."""
        assert handler._parse_row_count("DELETE 3") == 3
        assert handler._parse_row_count("DELETE 0") == 0

    def test_parse_invalid_returns_zero(self, handler: HandlerDb) -> None:
        """Test invalid result string returns 0."""
        assert handler._parse_row_count("") == 0
        assert handler._parse_row_count("INVALID") == 0
        assert handler._parse_row_count("INSERT") == 0


class TestHandlerDbLogWarnings:
    """Test suite for log warning assertions (OMN-252 acceptance criteria).

    These tests verify that:
    1. Normal operations produce no unexpected warnings
    2. Expected warnings are logged only in specific error conditions
    """

    # Module name used for filtering log warnings
    HANDLER_MODULE = "omnibase_infra.handlers.handler_db"

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        pool = MagicMock(spec=asyncpg.Pool)
        pool.close = AsyncMock()
        return pool

    @pytest.mark.asyncio
    async def test_no_unexpected_warnings_during_normal_operation(
        self, handler: HandlerDb, mock_pool: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that normal operations produce no unexpected warnings.

        This test verifies the OMN-252 acceptance criteria: "Asserts no unexpected
        warnings in logs" during normal handler lifecycle and execution.
        """
        import logging

        # Setup mock connection and rows
        mock_conn = AsyncMock()
        mock_rows = [{"id": 1, "name": "Alice"}]
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with caplog.at_level(logging.WARNING):
            with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_pool

                # Initialize
                await handler.initialize({"dsn": "postgresql://localhost/db"})

                # Perform normal query operation
                correlation_id = uuid4()
                envelope: dict[str, object] = {
                    "operation": "db.query",
                    "payload": {"sql": "SELECT id, name FROM users"},
                    "correlation_id": correlation_id,
                }

                output = await handler.execute(envelope)
                result = output.result  # ModelDbQueryResponse
                assert result.status == "success"

                # Shutdown
                await handler.shutdown()

        # Filter for warnings from our handler module using helper
        handler_warnings = filter_handler_warnings(caplog.records, self.HANDLER_MODULE)
        assert len(handler_warnings) == 0, (
            f"Unexpected warnings: {[w.message for w in handler_warnings]}"
        )


class TestHandlerDbMapPostgresError:
    """Test suite for _map_postgres_error helper function.

    Tests the extracted error mapping helper that centralizes exception-to-error
    mapping logic, reducing complexity in _execute_statement and _execute_query.
    """

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    @pytest.fixture
    def error_context(self) -> ModelInfraErrorContext:
        """Create error context fixture."""
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import ModelInfraErrorContext

        return ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="db.execute",
            target_name="db_handler",
            correlation_id=uuid4(),
        )

    def test_query_canceled_returns_timeout_error(
        self, handler: HandlerDb, error_context: ModelInfraErrorContext
    ) -> None:
        """Test QueryCanceledError maps to InfraTimeoutError."""
        exc = asyncpg.QueryCanceledError("query timeout")

        result = handler._map_postgres_error(exc, error_context)

        assert isinstance(result, InfraTimeoutError)
        assert "timed out" in str(result).lower()
        # timeout_seconds is passed as extra context to the error
        assert str(handler._timeout) in str(result)

    def test_connection_error_returns_infra_connection_error(
        self, handler: HandlerDb, error_context: ModelInfraErrorContext
    ) -> None:
        """Test PostgresConnectionError maps to InfraConnectionError."""
        exc = asyncpg.PostgresConnectionError("connection lost")

        result = handler._map_postgres_error(exc, error_context)

        assert isinstance(result, InfraConnectionError)
        assert "connection" in str(result).lower()

    def test_syntax_error_returns_runtime_error(
        self, handler: HandlerDb, error_context: ModelInfraErrorContext
    ) -> None:
        """Test PostgresSyntaxError maps to RuntimeHostError with syntax prefix."""
        exc = asyncpg.PostgresSyntaxError("syntax error near 'SELEKT'")
        exc.message = "syntax error at or near 'SELEKT'"

        result = handler._map_postgres_error(exc, error_context)

        assert isinstance(result, RuntimeHostError)
        assert "SQL syntax error" in str(result)
        assert "SELEKT" in str(result)

    def test_undefined_table_returns_runtime_error(
        self, handler: HandlerDb, error_context: ModelInfraErrorContext
    ) -> None:
        """Test UndefinedTableError maps to RuntimeHostError with table prefix."""
        exc = asyncpg.UndefinedTableError("table not found")
        exc.message = 'relation "nonexistent" does not exist'

        result = handler._map_postgres_error(exc, error_context)

        assert isinstance(result, RuntimeHostError)
        assert "Table not found" in str(result)
        assert "nonexistent" in str(result)

    def test_undefined_column_returns_runtime_error(
        self, handler: HandlerDb, error_context: ModelInfraErrorContext
    ) -> None:
        """Test UndefinedColumnError maps to RuntimeHostError with column prefix."""
        exc = asyncpg.UndefinedColumnError("column not found")
        exc.message = 'column "unknown_col" does not exist'

        result = handler._map_postgres_error(exc, error_context)

        assert isinstance(result, RuntimeHostError)
        assert "Column not found" in str(result)
        assert "unknown_col" in str(result)

    def test_unique_violation_returns_runtime_error(
        self, handler: HandlerDb, error_context: ModelInfraErrorContext
    ) -> None:
        """Test UniqueViolationError maps to RuntimeHostError with unique prefix."""
        exc = asyncpg.UniqueViolationError("unique violation")
        exc.message = "duplicate key value violates unique constraint"

        result = handler._map_postgres_error(exc, error_context)

        assert isinstance(result, RuntimeHostError)
        assert "Unique constraint violation" in str(result)

    def test_foreign_key_violation_returns_runtime_error(
        self, handler: HandlerDb, error_context: ModelInfraErrorContext
    ) -> None:
        """Test ForeignKeyViolationError maps to RuntimeHostError with FK prefix."""
        exc = asyncpg.ForeignKeyViolationError("foreign key violation")
        exc.message = "insert violates foreign key constraint"

        result = handler._map_postgres_error(exc, error_context)

        assert isinstance(result, RuntimeHostError)
        assert "Foreign key constraint violation" in str(result)

    def test_not_null_violation_returns_runtime_error(
        self, handler: HandlerDb, error_context: ModelInfraErrorContext
    ) -> None:
        """Test NotNullViolationError maps to RuntimeHostError with NOT NULL prefix."""
        exc = asyncpg.NotNullViolationError("not null violation")
        exc.message = "null value in column violates not-null constraint"

        result = handler._map_postgres_error(exc, error_context)

        assert isinstance(result, RuntimeHostError)
        assert "Not null constraint violation" in str(result)

    def test_check_violation_returns_runtime_error(
        self, handler: HandlerDb, error_context: ModelInfraErrorContext
    ) -> None:
        """Test CheckViolationError maps to RuntimeHostError with check prefix."""
        exc = asyncpg.CheckViolationError("check violation")
        exc.message = "new row violates check constraint"

        result = handler._map_postgres_error(exc, error_context)

        assert isinstance(result, RuntimeHostError)
        assert "Check constraint violation" in str(result)

    def test_unknown_postgres_error_returns_runtime_error_with_default_prefix(
        self, handler: HandlerDb, error_context: ModelInfraErrorContext
    ) -> None:
        """Test unknown PostgresError maps to RuntimeHostError with default prefix."""
        # Use a generic PostgresError that's not specifically mapped
        exc = asyncpg.PostgresError("some error")

        result = handler._map_postgres_error(exc, error_context)

        assert isinstance(result, RuntimeHostError)
        assert "Database error" in str(result)

    def test_error_without_message_attribute_uses_type_name(
        self, handler: HandlerDb, error_context: ModelInfraErrorContext
    ) -> None:
        """Test error without message attribute uses exception type name."""
        # PostgresError by default doesn't have a message attribute
        exc = asyncpg.PostgresError("generic error")
        # Verify no message attribute exists (or it has no value)
        # getattr with default will return the type name fallback
        assert not hasattr(exc, "message") or exc.message is None

        result = handler._map_postgres_error(exc, error_context)

        assert isinstance(result, RuntimeHostError)
        # Should contain the type name as fallback
        assert "PostgresError" in str(result)


class TestHandlerDbTransientErrorClassification:
    """Test suite for intelligent error classification (_is_transient_error).

    The circuit breaker should only trip on TRANSIENT errors (infrastructure issues)
    and NOT on PERMANENT errors (application bugs like constraint violations).

    PostgreSQL SQLSTATE class codes:
    - Transient (should trip circuit): 08, 53, 57, 58
    - Permanent (should NOT trip circuit): 22, 23, 28, 42
    """

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    # --- Transient error tests (should return True) ---

    def test_connection_error_is_transient(self, handler: HandlerDb) -> None:
        """Test PostgresConnectionError is classified as transient."""
        error = asyncpg.PostgresConnectionError("connection lost")
        assert handler._is_transient_error(error) is True

    def test_query_canceled_is_transient(self, handler: HandlerDb) -> None:
        """Test QueryCanceledError (timeout) is classified as transient."""
        error = asyncpg.QueryCanceledError("query timeout")
        assert handler._is_transient_error(error) is True

    def test_class_08_connection_exception_is_transient(
        self, handler: HandlerDb
    ) -> None:
        """Test Class 08 (Connection Exception) is classified as transient."""
        # Create error with SQLSTATE 08000 (connection_exception)
        error = asyncpg.PostgresError("connection exception")
        error.sqlstate = "08000"
        assert handler._is_transient_error(error) is True

        # Test 08003 (connection_does_not_exist)
        error.sqlstate = "08003"
        assert handler._is_transient_error(error) is True

        # Test 08006 (connection_failure)
        error.sqlstate = "08006"
        assert handler._is_transient_error(error) is True

    def test_class_53_insufficient_resources_is_transient(
        self, handler: HandlerDb
    ) -> None:
        """Test Class 53 (Insufficient Resources) is classified as transient."""
        error = asyncpg.PostgresError("out of memory")
        # 53000 (insufficient_resources)
        error.sqlstate = "53000"
        assert handler._is_transient_error(error) is True

        # 53100 (disk_full)
        error.sqlstate = "53100"
        assert handler._is_transient_error(error) is True

        # 53200 (out_of_memory)
        error.sqlstate = "53200"
        assert handler._is_transient_error(error) is True

        # 53300 (too_many_connections)
        error.sqlstate = "53300"
        assert handler._is_transient_error(error) is True

    def test_class_57_operator_intervention_is_transient(
        self, handler: HandlerDb
    ) -> None:
        """Test Class 57 (Operator Intervention) is classified as transient."""
        error = asyncpg.PostgresError("server shutdown")
        # 57000 (operator_intervention)
        error.sqlstate = "57000"
        assert handler._is_transient_error(error) is True

        # 57014 (query_canceled)
        error.sqlstate = "57014"
        assert handler._is_transient_error(error) is True

        # 57P01 (admin_shutdown)
        error.sqlstate = "57P01"
        assert handler._is_transient_error(error) is True

        # 57P02 (crash_shutdown)
        error.sqlstate = "57P02"
        assert handler._is_transient_error(error) is True

        # 57P03 (cannot_connect_now)
        error.sqlstate = "57P03"
        assert handler._is_transient_error(error) is True

    def test_class_58_system_error_is_transient(self, handler: HandlerDb) -> None:
        """Test Class 58 (System Error) is classified as transient."""
        error = asyncpg.PostgresError("I/O error")
        # 58000 (system_error)
        error.sqlstate = "58000"
        assert handler._is_transient_error(error) is True

        # 58030 (io_error)
        error.sqlstate = "58030"
        assert handler._is_transient_error(error) is True

    # --- Permanent error tests (should return False) ---

    def test_class_23_integrity_constraint_is_permanent(
        self, handler: HandlerDb
    ) -> None:
        """Test Class 23 (Integrity Constraint Violation) is classified as permanent."""
        error = asyncpg.PostgresError("constraint violation")
        # 23000 (integrity_constraint_violation)
        error.sqlstate = "23000"
        assert handler._is_transient_error(error) is False

        # 23502 (not_null_violation)
        error.sqlstate = "23502"
        assert handler._is_transient_error(error) is False

        # 23503 (foreign_key_violation)
        error.sqlstate = "23503"
        assert handler._is_transient_error(error) is False

        # 23505 (unique_violation)
        error.sqlstate = "23505"
        assert handler._is_transient_error(error) is False

        # 23514 (check_violation)
        error.sqlstate = "23514"
        assert handler._is_transient_error(error) is False

    def test_class_42_syntax_error_is_permanent(self, handler: HandlerDb) -> None:
        """Test Class 42 (Syntax Error or Access Rule Violation) is permanent."""
        error = asyncpg.PostgresError("syntax error")
        # 42000 (syntax_error_or_access_rule_violation)
        error.sqlstate = "42000"
        assert handler._is_transient_error(error) is False

        # 42601 (syntax_error)
        error.sqlstate = "42601"
        assert handler._is_transient_error(error) is False

        # 42P01 (undefined_table)
        error.sqlstate = "42P01"
        assert handler._is_transient_error(error) is False

        # 42703 (undefined_column)
        error.sqlstate = "42703"
        assert handler._is_transient_error(error) is False

    def test_class_28_invalid_authorization_is_permanent(
        self, handler: HandlerDb
    ) -> None:
        """Test Class 28 (Invalid Authorization Specification) is permanent."""
        error = asyncpg.PostgresError("invalid authorization")
        # 28000 (invalid_authorization_specification)
        error.sqlstate = "28000"
        assert handler._is_transient_error(error) is False

        # 28P01 (invalid_password)
        error.sqlstate = "28P01"
        assert handler._is_transient_error(error) is False

    def test_class_22_data_exception_is_permanent(self, handler: HandlerDb) -> None:
        """Test Class 22 (Data Exception) is classified as permanent."""
        error = asyncpg.PostgresError("data exception")
        # 22000 (data_exception)
        error.sqlstate = "22000"
        assert handler._is_transient_error(error) is False

        # 22012 (division_by_zero)
        error.sqlstate = "22012"
        assert handler._is_transient_error(error) is False

        # 22001 (string_data_right_truncation)
        error.sqlstate = "22001"
        assert handler._is_transient_error(error) is False

    def test_foreign_key_violation_exception_is_permanent(
        self, handler: HandlerDb
    ) -> None:
        """Test ForeignKeyViolationError (specific exception type) is permanent."""
        error = asyncpg.ForeignKeyViolationError("FK violation")
        error.sqlstate = "23503"
        assert handler._is_transient_error(error) is False

    def test_not_null_violation_exception_is_permanent(
        self, handler: HandlerDb
    ) -> None:
        """Test NotNullViolationError (specific exception type) is permanent."""
        error = asyncpg.NotNullViolationError("NOT NULL violation")
        error.sqlstate = "23502"
        assert handler._is_transient_error(error) is False

    def test_syntax_error_exception_is_permanent(self, handler: HandlerDb) -> None:
        """Test PostgresSyntaxError (specific exception type) is permanent."""
        error = asyncpg.PostgresSyntaxError("syntax error")
        error.sqlstate = "42601"
        assert handler._is_transient_error(error) is False

    def test_undefined_table_exception_is_permanent(self, handler: HandlerDb) -> None:
        """Test UndefinedTableError (specific exception type) is permanent."""
        error = asyncpg.UndefinedTableError("table not found")
        error.sqlstate = "42P01"
        assert handler._is_transient_error(error) is False

    # --- Edge cases ---

    def test_error_without_sqlstate_falls_back_to_type_check(
        self, handler: HandlerDb
    ) -> None:
        """Test errors without sqlstate fall back to exception type classification.

        Note: asyncpg exception types like PostgresConnectionError have a default
        sqlstate (e.g., '08000' for connection errors). This test verifies that
        even with a mock generic error that has no sqlstate, the fallback logic
        works correctly for known exception types.
        """
        # PostgresConnectionError is always transient, regardless of sqlstate
        # (it has default sqlstate='08000' which is also transient)
        conn_error = asyncpg.PostgresConnectionError("connection lost")
        assert handler._is_transient_error(conn_error) is True

        # QueryCanceledError is always transient
        # (it has default sqlstate='57014' which is also transient)
        timeout_error = asyncpg.QueryCanceledError("timeout")
        assert handler._is_transient_error(timeout_error) is True

        # Test a generic PostgresError with sqlstate explicitly set to None
        # to verify fallback behavior
        generic_error = asyncpg.PostgresError("generic error")
        # Generic error without sqlstate defaults to permanent (conservative)
        assert handler._is_transient_error(generic_error) is False

    def test_unknown_error_without_sqlstate_defaults_to_permanent(
        self, handler: HandlerDb
    ) -> None:
        """Test unknown errors without sqlstate default to permanent (conservative)."""
        error = asyncpg.PostgresError("unknown error")
        # No sqlstate set
        assert handler._is_transient_error(error) is False

    def test_unknown_sqlstate_class_defaults_to_permanent(
        self, handler: HandlerDb
    ) -> None:
        """Test unknown SQLSTATE class defaults to permanent (conservative)."""
        error = asyncpg.PostgresError("unknown error")
        # Class XX is not in our known transient or permanent lists
        error.sqlstate = "XX123"
        assert handler._is_transient_error(error) is False


class TestHandlerDbCircuitBreakerErrorClassification:
    """Test suite for circuit breaker behavior with intelligent error classification.

    Verifies that only transient errors (infrastructure issues) trip the circuit
    breaker, while permanent errors (application bugs) do NOT affect circuit state.
    """

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerDb:
        """Create HandlerDb fixture."""
        return HandlerDb(mock_container)

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        pool = MagicMock(spec=asyncpg.Pool)
        pool.close = AsyncMock()
        return pool

    @pytest.mark.asyncio
    async def test_transient_error_trips_circuit_breaker(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test that transient errors (connection, timeout) trip circuit breaker."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            side_effect=asyncpg.PostgresConnectionError("connection lost")
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            # Circuit breaker should start at 0 failures
            assert handler._circuit_breaker_failures == 0

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELECT 1"},
            }

            with pytest.raises(InfraConnectionError):
                await handler.execute(envelope)

            # Circuit breaker failure count should increase
            assert handler._circuit_breaker_failures == 1

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_permanent_error_does_not_trip_circuit_breaker(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test that permanent errors (FK, NOT NULL) do NOT trip circuit breaker."""
        mock_conn = AsyncMock()
        fk_error = asyncpg.ForeignKeyViolationError("FK violation")
        fk_error.message = "violates foreign key constraint"
        fk_error.sqlstate = "23503"  # FK violation
        mock_conn.execute = AsyncMock(side_effect=fk_error)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            # Circuit breaker should start at 0 failures
            assert handler._circuit_breaker_failures == 0

            envelope: dict[str, object] = {
                "operation": "db.execute",
                "payload": {
                    "sql": "INSERT INTO orders (user_id) VALUES ($1)",
                    "parameters": [99999],
                },
            }

            with pytest.raises(RuntimeHostError):
                await handler.execute(envelope)

            # Circuit breaker failure count should NOT increase for FK violation
            assert handler._circuit_breaker_failures == 0

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_not_null_violation_does_not_trip_circuit_breaker(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test that NOT NULL violation does NOT trip circuit breaker."""
        mock_conn = AsyncMock()
        nn_error = asyncpg.NotNullViolationError("NOT NULL violation")
        nn_error.message = "null value violates not-null constraint"
        nn_error.sqlstate = "23502"  # NOT NULL violation
        mock_conn.execute = AsyncMock(side_effect=nn_error)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})
            assert handler._circuit_breaker_failures == 0

            envelope: dict[str, object] = {
                "operation": "db.execute",
                "payload": {
                    "sql": "INSERT INTO users (name) VALUES ($1)",
                    "parameters": [None],
                },
            }

            with pytest.raises(RuntimeHostError):
                await handler.execute(envelope)

            # Circuit breaker failure count should NOT increase
            assert handler._circuit_breaker_failures == 0

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_syntax_error_does_not_trip_circuit_breaker(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test that syntax error does NOT trip circuit breaker."""
        mock_conn = AsyncMock()
        syntax_error = asyncpg.PostgresSyntaxError("syntax error")
        syntax_error.message = "syntax error at or near 'SELEKT'"
        syntax_error.sqlstate = "42601"  # Syntax error
        mock_conn.fetch = AsyncMock(side_effect=syntax_error)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})
            assert handler._circuit_breaker_failures == 0

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELEKT * FROM users"},
            }

            with pytest.raises(RuntimeHostError):
                await handler.execute(envelope)

            # Circuit breaker failure count should NOT increase
            assert handler._circuit_breaker_failures == 0

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_resource_exhaustion_trips_circuit_breaker(
        self, handler: HandlerDb, mock_pool: MagicMock
    ) -> None:
        """Test that resource exhaustion (Class 53) trips circuit breaker."""
        mock_conn = AsyncMock()
        resource_error = asyncpg.PostgresError("out of memory")
        resource_error.sqlstate = "53200"  # Out of memory
        mock_conn.fetch = AsyncMock(side_effect=resource_error)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})
            assert handler._circuit_breaker_failures == 0

            envelope: dict[str, object] = {
                "operation": "db.query",
                "payload": {"sql": "SELECT * FROM huge_table"},
            }

            with pytest.raises(RuntimeHostError):
                await handler.execute(envelope)

            # Circuit breaker failure count SHOULD increase for resource exhaustion
            assert handler._circuit_breaker_failures == 1

            await handler.shutdown()


__all__: list[str] = [
    "TestHandlerDbInitialization",
    "TestHandlerDbQueryOperations",
    "TestHandlerDbExecuteOperations",
    "TestHandlerDbErrorHandling",
    "TestHandlerDbDescribe",
    "TestHandlerDbLifecycle",
    "TestHandlerDbCorrelationId",
    "TestHandlerDbDsnSecurity",
    "TestHandlerDbRowCountParsing",
    "TestHandlerDbLogWarnings",
    "TestHandlerDbMapPostgresError",
    "TestHandlerDbTransientErrorClassification",
    "TestHandlerDbCircuitBreakerErrorClassification",
]
