"""
Comprehensive tests for PostgreSQL Connection Manager.

Tests connection pooling, query execution, metrics collection,
and error handling.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from asyncpg import Connection, Pool, Record
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from pydantic import SecretStr

from omnibase_infra.infrastructure.postgres.connection_manager import (
    PostgresConnectionManager,
)
from omnibase_infra.models.postgres import (
    ModelPostgresConnectionConfig,
    ModelPostgresConnectionStats,
    ModelPostgresQueryMetrics,
)


@pytest.fixture
def mock_config():
    """Create a mock PostgreSQL connection configuration."""
    return ModelPostgresConnectionConfig(
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password=SecretStr("test_password"),
        schema="test_schema",
        min_connections=2,
        max_connections=10,
        max_inactive_connection_lifetime=300.0,
        max_queries=50000,
        command_timeout=60.0,
        ssl_mode="disable",
    )


@pytest.fixture
def manager(mock_config):
    """Create a PostgresConnectionManager instance."""
    return PostgresConnectionManager(config=mock_config)


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    pool = AsyncMock(spec=Pool)
    pool.get_size = Mock(return_value=5)
    return pool


@pytest.fixture
def mock_connection():
    """Create a mock asyncpg connection."""
    conn = AsyncMock(spec=Connection)
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock()
    conn.fetchval = AsyncMock()
    return conn


class TestPostgresConnectionManagerInit:
    """Test connection manager initialization."""

    def test_init_with_config(self, mock_config):
        """Test initialization with provided configuration."""
        manager = PostgresConnectionManager(config=mock_config)
        assert manager.config == mock_config
        assert manager.pool is None
        assert manager.is_initialized is False
        assert manager.query_metrics == []
        assert isinstance(manager.connection_stats, ModelPostgresConnectionStats)

    @patch.object(ModelPostgresConnectionConfig, "from_environment")
    def test_init_without_config(self, mock_from_env):
        """Test initialization with environment configuration."""
        env_config = ModelPostgresConnectionConfig(
            host="env_host",
            port=5432,
            database="env_db",
            user="env_user",
            password=SecretStr("env_password"),
        )
        mock_from_env.return_value = env_config

        manager = PostgresConnectionManager()
        assert manager.config == env_config
        mock_from_env.assert_called_once()


class TestPostgresConnectionManagerInitialization:
    """Test connection pool initialization."""

    @pytest.mark.asyncio
    async def test_successful_initialization(self, manager, mock_config, mock_connection):
        """Test successful pool initialization."""
        mock_connection.fetchval = AsyncMock(return_value=mock_config.schema)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_pool = AsyncMock(spec=Pool)
            mock_pool.acquire = AsyncMock(return_value=mock_connection)
            mock_pool.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_pool.__aexit__ = AsyncMock()
            mock_create_pool.return_value = mock_pool

            await manager.initialize()

            assert manager.is_initialized is True
            assert manager.pool == mock_pool
            mock_create_pool.assert_called_once()

            # Verify connection parameters
            call_kwargs = mock_create_pool.call_args.kwargs
            assert call_kwargs["host"] == mock_config.host
            assert call_kwargs["port"] == mock_config.port
            assert call_kwargs["database"] == mock_config.database
            assert call_kwargs["user"] == mock_config.user
            assert call_kwargs["password"] == mock_config.password.get_secret_value()

    @pytest.mark.asyncio
    async def test_initialization_already_initialized(self, manager):
        """Test that re-initialization is skipped."""
        manager.is_initialized = True
        manager.pool = Mock()

        await manager.initialize()

        # Should not create a new pool
        assert manager.pool is not None

    @pytest.mark.asyncio
    async def test_initialization_schema_mismatch(self, manager, mock_config):
        """Test initialization failure when schema cannot be set."""
        mock_connection = AsyncMock(spec=Connection)
        mock_connection.fetchval = AsyncMock(return_value="wrong_schema")

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_pool = AsyncMock(spec=Pool)
            mock_pool.acquire = AsyncMock(return_value=mock_connection)
            mock_pool.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_pool.__aexit__ = AsyncMock()
            mock_create_pool.return_value = mock_pool

            with pytest.raises(OnexError) as exc_info:
                await manager.initialize()

            assert exc_info.value.code == CoreErrorCode.DATABASE_CONNECTION_FAILED
            assert "Failed to set schema" in exc_info.value.message
            assert manager.connection_stats.failed_connections == 1

    @pytest.mark.asyncio
    async def test_initialization_connection_failure(self, manager):
        """Test initialization failure on connection error."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection refused")

            with pytest.raises(OnexError) as exc_info:
                await manager.initialize()

            assert exc_info.value.code == CoreErrorCode.DATABASE_CONNECTION_FAILED
            assert "Failed to initialize PostgreSQL connection pool" in exc_info.value.message
            assert manager.connection_stats.failed_connections == 1

    @pytest.mark.asyncio
    async def test_initialization_with_ssl(self, mock_config):
        """Test initialization with SSL configuration."""
        mock_config.ssl_mode = "require"
        mock_config.ssl_cert_file = "/path/to/cert.pem"
        mock_config.ssl_key_file = "/path/to/key.pem"
        mock_config.ssl_ca_file = "/path/to/ca.pem"

        manager = PostgresConnectionManager(config=mock_config)

        mock_connection = AsyncMock(spec=Connection)
        mock_connection.fetchval = AsyncMock(return_value=mock_config.schema)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_pool = AsyncMock(spec=Pool)
            mock_pool.acquire = AsyncMock(return_value=mock_connection)
            mock_pool.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_pool.__aexit__ = AsyncMock()
            mock_create_pool.return_value = mock_pool

            await manager.initialize()

            # Verify SSL parameters
            call_kwargs = mock_create_pool.call_args.kwargs
            assert call_kwargs["ssl"] == "require"
            assert call_kwargs["ssl_cert"] == "/path/to/cert.pem"
            assert call_kwargs["ssl_key"] == "/path/to/key.pem"
            assert call_kwargs["ssl_ca"] == "/path/to/ca.pem"


class TestPostgresConnectionManagerClose:
    """Test connection pool closing."""

    @pytest.mark.asyncio
    async def test_close_pool(self, manager, mock_pool):
        """Test closing the connection pool."""
        manager.pool = mock_pool
        manager.is_initialized = True

        await manager.close()

        mock_pool.close.assert_called_once()
        assert manager.pool is None
        assert manager.is_initialized is False

    @pytest.mark.asyncio
    async def test_close_no_pool(self, manager):
        """Test closing when no pool exists."""
        manager.pool = None
        manager.is_initialized = False

        await manager.close()

        # Should not raise any errors
        assert manager.pool is None


class TestPostgresConnectionManagerAcquireConnection:
    """Test connection acquisition from pool."""

    @pytest.mark.asyncio
    async def test_acquire_connection_success(self, manager, mock_pool, mock_connection):
        """Test successful connection acquisition."""
        manager.pool = mock_pool
        manager.is_initialized = True
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()

        async with manager.acquire_connection() as conn:
            assert conn == mock_connection
            mock_connection.execute.assert_called_once()  # SET search_path

        mock_pool.acquire.assert_called_once()
        mock_pool.release.assert_called_once_with(mock_connection)
        assert manager.connection_stats.checked_out == 1
        assert manager.connection_stats.checked_in == 1

    @pytest.mark.asyncio
    async def test_acquire_connection_auto_initialize(self, manager, mock_config, mock_connection):
        """Test auto-initialization when acquiring connection."""
        mock_connection.fetchval = AsyncMock(return_value=mock_config.schema)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_pool = AsyncMock(spec=Pool)
            mock_pool.acquire = AsyncMock(return_value=mock_connection)
            mock_pool.release = AsyncMock()
            mock_pool.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_pool.__aexit__ = AsyncMock()
            mock_create_pool.return_value = mock_pool

            async with manager.acquire_connection() as conn:
                assert conn == mock_connection

            assert manager.is_initialized is True

    @pytest.mark.asyncio
    async def test_acquire_connection_failure(self, manager, mock_pool):
        """Test connection acquisition failure."""
        manager.pool = mock_pool
        manager.is_initialized = True
        mock_pool.acquire = AsyncMock(side_effect=Exception("Pool exhausted"))

        with pytest.raises(OnexError) as exc_info:
            async with manager.acquire_connection():
                pass

        assert exc_info.value.code == CoreErrorCode.DATABASE_OPERATION_ERROR
        assert "Database connection error" in exc_info.value.message
        assert manager.connection_stats.failed_connections == 1

    @pytest.mark.asyncio
    async def test_acquire_connection_release_on_error(self, manager, mock_pool, mock_connection):
        """Test connection is released even if error occurs during usage."""
        manager.pool = mock_pool
        manager.is_initialized = True
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()

        with pytest.raises(ValueError):
            async with manager.acquire_connection():
                raise ValueError("User error")

        # Connection should still be released
        mock_pool.release.assert_called_once_with(mock_connection)
        assert manager.connection_stats.checked_in == 1


class TestPostgresConnectionManagerExecuteQuery:
    """Test query execution."""

    @pytest.mark.asyncio
    async def test_execute_select_query(self, manager, mock_pool, mock_connection):
        """Test executing SELECT query."""
        manager.pool = mock_pool
        manager.is_initialized = True
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()

        mock_records = [
            Mock(spec=Record),
            Mock(spec=Record),
        ]
        mock_connection.fetch = AsyncMock(return_value=mock_records)

        result = await manager.execute_query("SELECT * FROM users")

        assert result == mock_records
        assert len(manager.query_metrics) == 1
        metric = manager.query_metrics[0]
        assert metric.rows_affected == 2
        assert metric.was_successful is True
        assert metric.error_message is None

    @pytest.mark.asyncio
    async def test_execute_insert_query(self, manager, mock_pool, mock_connection):
        """Test executing INSERT query."""
        manager.pool = mock_pool
        manager.is_initialized = True
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()

        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        result = await manager.execute_query("INSERT INTO users VALUES ($1, $2)", "user1", "email@test.com")

        assert result == "INSERT 0 1"
        assert len(manager.query_metrics) == 1
        metric = manager.query_metrics[0]
        assert metric.rows_affected == 1
        assert metric.was_successful is True

    @pytest.mark.asyncio
    async def test_execute_update_query(self, manager, mock_pool, mock_connection):
        """Test executing UPDATE query."""
        manager.pool = mock_pool
        manager.is_initialized = True
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()

        mock_connection.execute = AsyncMock(return_value="UPDATE 5")

        result = await manager.execute_query("UPDATE users SET active = true")

        assert result == "UPDATE 5"
        metric = manager.query_metrics[0]
        assert metric.rows_affected == 5

    @pytest.mark.asyncio
    async def test_execute_query_with_timeout(self, manager, mock_pool, mock_connection):
        """Test query execution with custom timeout."""
        manager.pool = mock_pool
        manager.is_initialized = True
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()

        mock_connection.fetch = AsyncMock(return_value=[])

        await manager.execute_query("SELECT * FROM users", timeout=10.0)

        mock_connection.fetch.assert_called_once()
        call_kwargs = mock_connection.fetch.call_args.kwargs
        assert call_kwargs.get("timeout") == 10.0

    @pytest.mark.asyncio
    async def test_execute_query_no_metrics(self, manager, mock_pool, mock_connection):
        """Test query execution without metrics collection."""
        manager.pool = mock_pool
        manager.is_initialized = True
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()

        mock_connection.execute = AsyncMock(return_value="DELETE 3")

        await manager.execute_query("DELETE FROM users WHERE id = $1", 123, record_metrics=False)

        assert len(manager.query_metrics) == 0

    @pytest.mark.asyncio
    async def test_execute_query_failure(self, manager, mock_pool, mock_connection):
        """Test query execution failure."""
        manager.pool = mock_pool
        manager.is_initialized = True
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()

        mock_connection.execute = AsyncMock(side_effect=Exception("Syntax error"))

        with pytest.raises(OnexError) as exc_info:
            await manager.execute_query("INVALID SQL")

        assert exc_info.value.code == CoreErrorCode.DATABASE_QUERY_ERROR
        assert "Query execution failed" in exc_info.value.message

        # Metrics should still be recorded
        assert len(manager.query_metrics) == 1
        metric = manager.query_metrics[0]
        assert metric.was_successful is False
        assert metric.error_message is not None

    @pytest.mark.asyncio
    async def test_execute_with_query(self, manager, mock_pool, mock_connection):
        """Test executing WITH (CTE) query."""
        manager.pool = mock_pool
        manager.is_initialized = True
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()

        mock_records = [Mock(spec=Record)]
        mock_connection.fetch = AsyncMock(return_value=mock_records)

        result = await manager.execute_query("WITH cte AS (SELECT 1) SELECT * FROM cte")

        assert result == mock_records
        mock_connection.fetch.assert_called_once()


class TestPostgresConnectionManagerMetrics:
    """Test metrics collection and reporting."""

    def test_get_connection_stats(self, manager, mock_pool):
        """Test retrieving connection statistics."""
        manager.pool = mock_pool
        manager.connection_stats.checked_out = 10
        manager.connection_stats.checked_in = 8
        manager.connection_stats.failed_connections = 2

        stats = manager.get_connection_stats()

        assert isinstance(stats, ModelPostgresConnectionStats)
        assert stats.size == 5  # From mock_pool.get_size()
        assert stats.total_connections == 5
        assert stats.checked_out == 10
        assert stats.checked_in == 8
        assert stats.failed_connections == 2

    def test_get_connection_stats_no_pool(self, manager):
        """Test getting stats when pool doesn't exist."""
        manager.pool = None

        stats = manager.get_connection_stats()

        assert stats.size == 0
        assert stats.total_connections == 0

    def test_get_query_metrics(self, manager):
        """Test retrieving query metrics."""
        # Add some metrics
        for i in range(5):
            manager.query_metrics.append(
                ModelPostgresQueryMetrics(
                    query_hash=f"hash_{i}",
                    execution_time_ms=float(i * 10),
                    rows_affected=i,
                    connection_id=f"conn_{i}",
                    timestamp=float(i),
                    was_successful=True,
                )
            )

        metrics = manager.get_query_metrics(limit=3)

        assert len(metrics) == 3
        assert metrics[0].query_hash == "hash_2"
        assert metrics[-1].query_hash == "hash_4"

    def test_get_query_metrics_all(self, manager):
        """Test retrieving all query metrics."""
        for i in range(50):
            manager.query_metrics.append(
                ModelPostgresQueryMetrics(
                    query_hash=f"hash_{i}",
                    execution_time_ms=float(i),
                    rows_affected=0,
                    connection_id="conn",
                    timestamp=float(i),
                    was_successful=True,
                )
            )

        metrics = manager.get_query_metrics()

        assert len(metrics) == 50

    def test_clear_metrics(self, manager):
        """Test clearing metrics."""
        manager.query_metrics.append(
            ModelPostgresQueryMetrics(
                query_hash="hash",
                execution_time_ms=10.0,
                rows_affected=1,
                connection_id="conn",
                timestamp=1.0,
                was_successful=True,
            )
        )
        manager.connection_stats.query_count = 5
        manager.connection_stats.checked_out = 10

        manager.clear_metrics()

        assert len(manager.query_metrics) == 0
        assert manager.connection_stats.query_count == 0
        assert manager.connection_stats.checked_out == 0

    def test_record_query_metrics(self, manager):
        """Test internal metric recording."""
        manager._record_query_metrics(
            query_hash="test_hash",
            execution_time_ms=25.5,
            rows_affected=3,
            connection_id="conn_123",
            was_successful=True,
            error_message=None,
        )

        assert len(manager.query_metrics) == 1
        metric = manager.query_metrics[0]
        assert metric.query_hash == "test_hash"
        assert metric.execution_time_ms == 25.5
        assert metric.rows_affected == 3
        assert metric.connection_id == "conn_123"
        assert metric.was_successful is True
        assert metric.error_message is None
        assert manager.connection_stats.query_count == 1
        assert manager.connection_stats.average_response_time_ms == 25.5

    def test_record_query_metrics_rolling_average(self, manager):
        """Test rolling average calculation for response time."""
        manager._record_query_metrics("hash1", 10.0, 1, "conn", True)
        manager._record_query_metrics("hash2", 20.0, 1, "conn", True)
        manager._record_query_metrics("hash3", 30.0, 1, "conn", True)

        assert manager.connection_stats.query_count == 3
        assert manager.connection_stats.average_response_time_ms == 20.0

    def test_record_query_metrics_limit_size(self, manager):
        """Test metric storage size limit."""
        # Add more than 1000 metrics
        for i in range(1100):
            manager._record_query_metrics(f"hash_{i}", float(i), 1, "conn", True)

        # Should only keep last 1000
        assert len(manager.query_metrics) == 1000
        assert manager.query_metrics[0].query_hash == "hash_100"
        assert manager.query_metrics[-1].query_hash == "hash_1099"


class TestPostgresConnectionManagerIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, manager, mock_config, mock_connection):
        """Test complete lifecycle: init -> query -> close."""
        mock_connection.fetchval = AsyncMock(return_value=mock_config.schema)
        mock_connection.fetch = AsyncMock(return_value=[Mock(spec=Record)])

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_pool = AsyncMock(spec=Pool)
            mock_pool.acquire = AsyncMock(return_value=mock_connection)
            mock_pool.release = AsyncMock()
            mock_pool.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_pool.__aexit__ = AsyncMock()
            mock_pool.get_size = Mock(return_value=5)
            mock_create_pool.return_value = mock_pool

            # Initialize
            await manager.initialize()
            assert manager.is_initialized is True

            # Execute query
            result = await manager.execute_query("SELECT 1")
            assert len(result) == 1

            # Get stats
            stats = manager.get_connection_stats()
            assert stats.query_count == 1

            # Close
            await manager.close()
            assert manager.is_initialized is False
            assert manager.pool is None

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, manager, mock_pool, mock_connection):
        """Test concurrent query execution."""
        manager.pool = mock_pool
        manager.is_initialized = True
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()
        mock_connection.execute = AsyncMock(return_value="UPDATE 1")

        # Execute multiple queries concurrently
        tasks = [
            manager.execute_query(f"UPDATE users SET name = 'user{i}'")
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert len(manager.query_metrics) == 5
        assert manager.connection_stats.query_count == 5
