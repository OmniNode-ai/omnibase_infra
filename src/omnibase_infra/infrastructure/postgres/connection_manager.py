"""
PostgreSQL Connection Manager for Infrastructure.

Provides asyncpg-based connection pooling and query execution utilities
for ONEX infrastructure components. This is a utility class used by
PostgreSQL adapter nodes for actual database operations.
"""

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from asyncpg import Connection, Pool, Record

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_infra.models.postgres import (
    ModelPostgresConnectionConfig,
    ModelPostgresConnectionStats,
    ModelPostgresQueryMetrics,
)


class PostgresConnectionManager:
    """
    Asyncpg connection pool manager for PostgreSQL infrastructure.

    This is a UTILITY class that manages asyncpg connection pools.
    It should be used by PostgreSQL adapter nodes, not directly by consumers.

    Features:
    - Connection pooling with configurable min/max connections
    - Query metrics collection and performance monitoring
    - Schema-aware operations with search path management
    - Automatic connection lifecycle management

    Note: Transaction management and health checking should be handled
    by adapter nodes, not by this utility class.
    """

    def __init__(self, config: ModelPostgresConnectionConfig | None = None):
        """
        Initialize connection manager with configuration.

        Args:
            config: Connection configuration (uses environment if None)
        """
        self.config = config or ModelPostgresConnectionConfig.from_environment()
        self.pool: Pool | None = None
        self.is_initialized = False
        self.query_metrics: list[ModelPostgresQueryMetrics] = []
        self.connection_stats = ModelPostgresConnectionStats()

    async def initialize(self) -> None:
        """
        Initialize the connection pool.

        Raises:
            OnexError: If pool initialization fails
        """
        if self.is_initialized:
            return

        try:
            # Build connection parameters
            connection_params = {
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "user": self.config.user,
                "password": self.config.password.get_secret_value(),
                "min_size": self.config.min_connections,
                "max_size": self.config.max_connections,
                "max_inactive_connection_lifetime": self.config.max_inactive_connection_lifetime,
                "max_queries": self.config.max_queries,
                "command_timeout": self.config.command_timeout,
                "server_settings": self.config.server_settings or {},
            }

            # Add SSL configuration if specified
            if self.config.ssl_mode != "disable":
                connection_params["ssl"] = self.config.ssl_mode
                if self.config.ssl_cert_file:
                    connection_params["ssl_cert"] = self.config.ssl_cert_file
                if self.config.ssl_key_file:
                    connection_params["ssl_key"] = self.config.ssl_key_file
                if self.config.ssl_ca_file:
                    connection_params["ssl_ca"] = self.config.ssl_ca_file

            # Create connection pool
            self.pool = await asyncpg.create_pool(**connection_params)

            # Test connection and set search path
            async with self.pool.acquire() as conn:
                await conn.execute(f"SET search_path TO {self.config.schema}, public")
                result = await conn.fetchval("SELECT current_schema()")
                if result != self.config.schema:
                    raise OnexError(
                        code=CoreErrorCode.DATABASE_CONNECTION_FAILED,
                        message=f"Failed to set schema to {self.config.schema}, got {result}",
                    )

            self.is_initialized = True

        except OnexError:
            self.connection_stats.failed_connections += 1
            raise
        except Exception as e:
            self.connection_stats.failed_connections += 1
            raise OnexError(
                code=CoreErrorCode.DATABASE_CONNECTION_FAILED,
                message=f"Failed to initialize PostgreSQL connection pool: {e!s}",
            ) from e

    async def close(self) -> None:
        """Close the connection pool and cleanup resources."""
        if self.pool:
            await self.pool.close()
            self.pool = None
        self.is_initialized = False

    @asynccontextmanager
    async def acquire_connection(self) -> AsyncIterator[Connection]:
        """
        Acquire a connection from the pool with automatic cleanup.

        Yields:
            Connection: asyncpg connection with schema path set

        Raises:
            OnexError: If connection acquisition fails

        Example:
            async with manager.acquire_connection() as conn:
                result = await conn.fetchval("SELECT 1")
        """
        if not self.is_initialized:
            await self.initialize()

        connection = None
        try:
            connection = await self.pool.acquire()
            self.connection_stats.checked_out += 1

            # Set search path for this connection
            await connection.execute(f"SET search_path TO {self.config.schema}, public")

            yield connection

        except OnexError:
            self.connection_stats.failed_connections += 1
            raise
        except Exception as e:
            self.connection_stats.failed_connections += 1
            raise OnexError(
                code=CoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Database connection error: {e!s}",
            ) from e
        finally:
            if connection:
                await self.pool.release(connection)
                self.connection_stats.checked_in += 1

    async def execute_query(
        self,
        query: str,
        *args,
        timeout: float | None = None,
        record_metrics: bool = True,
    ) -> str | list[Record]:
        """
        Execute a query with metrics collection.

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout (uses default if None)
            record_metrics: Whether to record query metrics

        Returns:
            Query result (status for non-SELECT, records for SELECT)

        Raises:
            OnexError: If query execution fails
        """
        start_time = time.perf_counter()
        query_hash = str(hash(query))[:12]
        connection_id = str(uuid.uuid4())[:8]
        error_message = None
        rows_affected = 0
        was_successful = False

        try:
            async with self.acquire_connection() as conn:
                if query.strip().upper().startswith(("SELECT", "WITH")):
                    result = await conn.fetch(query, *args, timeout=timeout)
                    rows_affected = len(result)
                    was_successful = True
                    return result
                result = await conn.execute(query, *args, timeout=timeout)
                # Parse rows affected from result string (e.g., "UPDATE 5")
                if result and result.split():
                    try:
                        rows_affected = int(result.split()[-1])
                    except (ValueError, IndexError):
                        rows_affected = 0
                was_successful = True
                return result

        except OnexError:
            error_message = str(OnexError)
            raise
        except Exception as e:
            error_message = str(e)
            raise OnexError(
                code=CoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Query execution failed: {e!s}",
            ) from e
        finally:
            if record_metrics:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._record_query_metrics(
                    query_hash,
                    execution_time_ms,
                    rows_affected,
                    connection_id,
                    was_successful,
                    error_message,
                )

    def get_connection_stats(self) -> ModelPostgresConnectionStats:
        """Get current connection pool statistics."""
        if self.pool:
            self.connection_stats.size = self.pool.get_size()
            self.connection_stats.total_connections = self.pool.get_size()

        return self.connection_stats

    def get_query_metrics(self, limit: int = 100) -> list[ModelPostgresQueryMetrics]:
        """Get recent query metrics."""
        return self.query_metrics[-limit:]

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self.query_metrics.clear()
        self.connection_stats = ModelPostgresConnectionStats()

    def _record_query_metrics(
        self,
        query_hash: str,
        execution_time_ms: float,
        rows_affected: int,
        connection_id: str,
        was_successful: bool,
        error_message: str | None = None,
    ) -> None:
        """Record query execution metrics."""
        metric = ModelPostgresQueryMetrics(
            query_hash=query_hash,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            connection_id=connection_id,
            timestamp=time.time(),
            was_successful=was_successful,
            error_message=error_message,
        )

        self.query_metrics.append(metric)
        self.connection_stats.query_count += 1

        # Update rolling average response time
        total_time = (
            self.connection_stats.average_response_time_ms
            * (self.connection_stats.query_count - 1)
            + execution_time_ms
        )
        self.connection_stats.average_response_time_ms = (
            total_time / self.connection_stats.query_count
        )

        # Keep only recent metrics (last 1000 queries)
        if len(self.query_metrics) > 1000:
            self.query_metrics = self.query_metrics[-1000:]
