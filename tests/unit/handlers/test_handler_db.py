# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for DbHandler.

Comprehensive test suite covering initialization, query/execute operations,
error handling, health checks, describe, and lifecycle management.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import asyncpg
import pytest
from omnibase_core.enums.enum_handler_type import EnumHandlerType

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    RuntimeHostError,
)
from omnibase_infra.handlers.handler_db import DbHandler
from omnibase_infra.handlers.models import (
    ModelDbHealthResponse,
)
from tests.helpers import filter_handler_warnings


class TestDbHandlerInitialization:
    """Test suite for DbHandler initialization."""

    @pytest.fixture
    def handler(self) -> DbHandler:
        """Create DbHandler fixture."""
        return DbHandler()

    def test_handler_init_default_state(self, handler: DbHandler) -> None:
        """Test handler initializes in uninitialized state."""
        assert handler._initialized is False
        assert handler._pool is None
        assert handler._pool_size == 5
        assert handler._timeout == 30.0

    def test_handler_type_returns_database(self, handler: DbHandler) -> None:
        """Test handler_type property returns EnumHandlerType.DATABASE."""
        assert handler.handler_type == EnumHandlerType.DATABASE

    @pytest.mark.asyncio
    async def test_initialize_missing_dsn_raises_error(
        self, handler: DbHandler
    ) -> None:
        """Test initialize without DSN raises RuntimeHostError."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.initialize({})

        assert "dsn" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_empty_dsn_raises_error(self, handler: DbHandler) -> None:
        """Test initialize with empty DSN raises RuntimeHostError."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.initialize({"dsn": ""})

        assert "dsn" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_creates_pool(self, handler: DbHandler) -> None:
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
    async def test_initialize_with_custom_timeout(self, handler: DbHandler) -> None:
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
        self, handler: DbHandler
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
        self, handler: DbHandler
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
        self, handler: DbHandler
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


class TestDbHandlerQueryOperations:
    """Test suite for db.query operations."""

    @pytest.fixture
    def handler(self) -> DbHandler:
        """Create DbHandler fixture."""
        return DbHandler()

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        return MagicMock(spec=asyncpg.Pool)

    @pytest.mark.asyncio
    async def test_query_successful_response(
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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


class TestDbHandlerExecuteOperations:
    """Test suite for db.execute operations."""

    @pytest.fixture
    def handler(self) -> DbHandler:
        """Create DbHandler fixture."""
        return DbHandler()

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        return MagicMock(spec=asyncpg.Pool)

    @pytest.mark.asyncio
    async def test_execute_insert_successful(
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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


class TestDbHandlerErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def handler(self) -> DbHandler:
        """Create DbHandler fixture."""
        return DbHandler()

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        return MagicMock(spec=asyncpg.Pool)

    @pytest.mark.asyncio
    async def test_query_timeout_raises_infra_timeout(
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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


class TestDbHandlerHealthCheck:
    """Test suite for health check operations."""

    @pytest.fixture
    def handler(self) -> DbHandler:
        """Create DbHandler fixture."""
        return DbHandler()

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        return MagicMock(spec=asyncpg.Pool)

    @pytest.mark.asyncio
    async def test_health_check_structure(
        self, handler: DbHandler, mock_pool: MagicMock
    ) -> None:
        """Test health_check returns correct structure."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            health = await handler.health_check()

            # Verify health response has all required fields
            assert isinstance(health, ModelDbHealthResponse)
            assert hasattr(health, "healthy")
            assert hasattr(health, "initialized")
            assert hasattr(health, "handler_type")
            assert hasattr(health, "pool_size")
            assert hasattr(health, "timeout_seconds")

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_initialized(
        self, handler: DbHandler, mock_pool: MagicMock
    ) -> None:
        """Test health_check shows healthy=True when initialized and DB responds."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            health = await handler.health_check()

            assert health.healthy is True
            assert health.initialized is True
            assert health.handler_type == "database"
            assert health.pool_size == 5
            assert health.timeout_seconds == 30.0

            mock_conn.fetchval.assert_called_once_with("SELECT 1")

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_not_initialized(
        self, handler: DbHandler
    ) -> None:
        """Test health_check shows healthy=False when not initialized."""
        health = await handler.health_check()

        assert health.healthy is False
        assert health.initialized is False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_db_unreachable(
        self, handler: DbHandler, mock_pool: MagicMock
    ) -> None:
        """Test health_check shows healthy=False when DB check fails."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=Exception("Connection failed"))

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            health = await handler.health_check()

            assert health.healthy is False
            assert health.initialized is True

            await handler.shutdown()


class TestDbHandlerDescribe:
    """Test suite for describe operations."""

    @pytest.fixture
    def handler(self) -> DbHandler:
        """Create DbHandler fixture."""
        return DbHandler()

    def test_describe_returns_handler_metadata(self, handler: DbHandler) -> None:
        """Test describe returns correct handler metadata."""
        description = handler.describe()

        assert description.handler_type == "database"
        assert description.pool_size == 5
        assert description.timeout_seconds == 30.0
        assert description.version == "0.1.0-mvp"
        assert description.initialized is False

    def test_describe_lists_supported_operations(self, handler: DbHandler) -> None:
        """Test describe lists supported operations."""
        description = handler.describe()

        assert "db.query" in description.supported_operations
        assert "db.execute" in description.supported_operations
        assert len(description.supported_operations) == 2

    @pytest.mark.asyncio
    async def test_describe_reflects_initialized_state(
        self, handler: DbHandler
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


class TestDbHandlerLifecycle:
    """Test suite for lifecycle management."""

    @pytest.fixture
    def handler(self) -> DbHandler:
        """Create DbHandler fixture."""
        return DbHandler()

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        pool = MagicMock(spec=asyncpg.Pool)
        pool.close = AsyncMock()
        return pool

    @pytest.mark.asyncio
    async def test_shutdown_closes_pool(
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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


class TestDbHandlerCorrelationId:
    """Test suite for correlation ID handling."""

    @pytest.fixture
    def handler(self) -> DbHandler:
        """Create DbHandler fixture."""
        return DbHandler()

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        return MagicMock(spec=asyncpg.Pool)

    @pytest.mark.asyncio
    async def test_correlation_id_from_envelope_uuid(
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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
        self, handler: DbHandler, mock_pool: MagicMock
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


class TestDbHandlerDsnSecurity:
    """Test suite for DSN security and sanitization.

    Security Policy: DSN contains credentials and must NEVER be exposed in:
    - Error messages
    - Log output
    - Health check responses
    - describe() metadata

    See DbHandler class docstring "Security Policy - DSN Handling" for full policy.
    """

    @pytest.fixture
    def handler(self) -> DbHandler:
        """Create DbHandler fixture."""
        return DbHandler()

    def test_sanitize_dsn_removes_password(self, handler: DbHandler) -> None:
        """Test _sanitize_dsn replaces password with asterisks."""
        # Standard format with password
        dsn = "postgresql://user:secret123@localhost:5432/mydb"
        sanitized = handler._sanitize_dsn(dsn)
        assert "secret123" not in sanitized
        assert "***" in sanitized
        assert "user" in sanitized
        assert "localhost" in sanitized

    def test_sanitize_dsn_handles_special_characters(self, handler: DbHandler) -> None:
        """Test _sanitize_dsn handles passwords with special characters."""
        dsn = "postgresql://admin:p@ss!word#123@db.example.com:5432/prod"
        sanitized = handler._sanitize_dsn(dsn)
        assert "p@ss!word#123" not in sanitized
        assert "***" in sanitized

    def test_sanitize_dsn_preserves_structure(self, handler: DbHandler) -> None:
        """Test _sanitize_dsn preserves DSN structure for debugging."""
        dsn = "postgresql://user:password@host:5432/database"
        sanitized = handler._sanitize_dsn(dsn)
        # Should preserve user, host, port, database
        assert sanitized == "postgresql://user:***@host:5432/database"

    @pytest.mark.asyncio
    async def test_connection_error_does_not_expose_dsn(
        self, handler: DbHandler
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
    async def test_auth_error_does_not_expose_dsn(self, handler: DbHandler) -> None:
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

    @pytest.mark.asyncio
    async def test_health_check_does_not_expose_dsn(self, handler: DbHandler) -> None:
        """Test that health check response does NOT include DSN."""
        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            secret_password = "health_check_secret_password"
            dsn = f"postgresql://user:{secret_password}@localhost/db"
            await handler.initialize({"dsn": dsn})

            health = await handler.health_check()

            # DSN or password must NOT be in health response
            health_str = str(health)
            assert secret_password not in health_str
            assert dsn not in health_str
            assert "dsn" not in health_str.lower()

            await handler.shutdown()

    def test_describe_does_not_expose_dsn(self, handler: DbHandler) -> None:
        """Test that describe() does NOT include DSN."""
        description = handler.describe()

        # DSN must NOT be in describe response
        desc_str = str(description)
        assert "dsn" not in desc_str.lower()
        assert "password" not in desc_str.lower()
        assert "postgresql://" not in desc_str


class TestDbHandlerRowCountParsing:
    """Test suite for row count parsing."""

    @pytest.fixture
    def handler(self) -> DbHandler:
        """Create DbHandler fixture."""
        return DbHandler()

    def test_parse_insert_row_count(self, handler: DbHandler) -> None:
        """Test parsing INSERT row count."""
        assert handler._parse_row_count("INSERT 0 1") == 1
        assert handler._parse_row_count("INSERT 0 5") == 5
        assert handler._parse_row_count("INSERT 0 100") == 100

    def test_parse_update_row_count(self, handler: DbHandler) -> None:
        """Test parsing UPDATE row count."""
        assert handler._parse_row_count("UPDATE 1") == 1
        assert handler._parse_row_count("UPDATE 10") == 10
        assert handler._parse_row_count("UPDATE 0") == 0

    def test_parse_delete_row_count(self, handler: DbHandler) -> None:
        """Test parsing DELETE row count."""
        assert handler._parse_row_count("DELETE 3") == 3
        assert handler._parse_row_count("DELETE 0") == 0

    def test_parse_invalid_returns_zero(self, handler: DbHandler) -> None:
        """Test invalid result string returns 0."""
        assert handler._parse_row_count("") == 0
        assert handler._parse_row_count("INVALID") == 0
        assert handler._parse_row_count("INSERT") == 0


class TestDbHandlerLogWarnings:
    """Test suite for log warning assertions (OMN-252 acceptance criteria).

    These tests verify that:
    1. Normal operations produce no unexpected warnings
    2. Expected warnings are logged only in specific error conditions
    """

    # Module name used for filtering log warnings
    HANDLER_MODULE = "omnibase_infra.handlers.handler_db"

    @pytest.fixture
    def handler(self) -> DbHandler:
        """Create DbHandler fixture."""
        return DbHandler()

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create mock asyncpg pool fixture."""
        pool = MagicMock(spec=asyncpg.Pool)
        pool.close = AsyncMock()
        return pool

    @pytest.mark.asyncio
    async def test_no_unexpected_warnings_during_normal_operation(
        self, handler: DbHandler, mock_pool: MagicMock, caplog: pytest.LogCaptureFixture
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

    @pytest.mark.asyncio
    async def test_health_check_logs_warning_on_failure(
        self, handler: DbHandler, mock_pool: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that health check failure produces expected warning.

        When the health check query fails (e.g., connection lost), the handler
        should log a warning indicating the health check failed.
        """
        import logging

        # Setup mock connection that fails on health check
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=Exception("Connection lost"))

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            with caplog.at_level(logging.WARNING):
                # Perform health check that will fail
                health = await handler.health_check()

                # Health check should return unhealthy
                assert health.healthy is False
                assert health.initialized is True

            await handler.shutdown()

        # Should have exactly one warning about health check failure
        handler_warnings = filter_handler_warnings(caplog.records, self.HANDLER_MODULE)
        assert len(handler_warnings) == 1
        assert "Health check failed" in handler_warnings[0].message

    @pytest.mark.asyncio
    async def test_no_warnings_on_successful_health_check(
        self, handler: DbHandler, mock_pool: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that successful health check produces no warnings.

        A successful health check should not log any warnings.
        """
        import logging

        # Setup mock connection that succeeds on health check
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await handler.initialize({"dsn": "postgresql://localhost/db"})

            with caplog.at_level(logging.WARNING):
                # Perform health check that will succeed
                health = await handler.health_check()

                # Health check should return healthy
                assert health.healthy is True
                assert health.initialized is True

            await handler.shutdown()

        # Should have no warnings
        handler_warnings = filter_handler_warnings(caplog.records, self.HANDLER_MODULE)
        assert len(handler_warnings) == 0, (
            f"Unexpected warnings: {[w.message for w in handler_warnings]}"
        )


__all__: list[str] = [
    "TestDbHandlerInitialization",
    "TestDbHandlerQueryOperations",
    "TestDbHandlerExecuteOperations",
    "TestDbHandlerErrorHandling",
    "TestDbHandlerHealthCheck",
    "TestDbHandlerDescribe",
    "TestDbHandlerLifecycle",
    "TestDbHandlerCorrelationId",
    "TestDbHandlerDsnSecurity",
    "TestDbHandlerRowCountParsing",
    "TestDbHandlerLogWarnings",
]
