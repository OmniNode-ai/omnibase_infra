"""
PostgreSQL Connection Manager for Infrastructure.

Provides enterprise-grade connection pooling, transaction management,
and high-availability database operations for the ONEX infrastructure system.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Union

import asyncpg
from asyncpg import Connection, Pool, Record

# Updated import to use omnibase-core
from omnibase_core.exceptions.base_onex_error import OnexError
from omnibase_core.core.errors.core_errors import CoreErrorCode


@dataclass
class ConnectionConfig:
    """PostgreSQL connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "omnibase_infrastructure"
    user: str = "postgres"
    password: str = ""
    schema: str = "infrastructure"

    # Pool configuration
    min_connections: int = 5
    max_connections: int = 50
    max_inactive_connection_lifetime: float = 300.0  # 5 minutes
    max_queries: int = 50000

    # Connection timeouts
    command_timeout: float = 60.0
    server_settings: Optional[Dict[str, str]] = None

    # SSL configuration
    ssl_mode: str = "prefer"
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None
    ssl_ca_file: Optional[str] = None

    @classmethod
    def from_environment(cls) -> "ConnectionConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DATABASE", "omnibase_infrastructure"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            schema=os.getenv("POSTGRES_SCHEMA", "infrastructure"),
            min_connections=int(os.getenv("POSTGRES_MIN_CONNECTIONS", "5")),
            max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "50")),
            max_inactive_connection_lifetime=float(
                os.getenv("POSTGRES_MAX_INACTIVE_LIFETIME", "300.0")
            ),
            command_timeout=float(os.getenv("POSTGRES_COMMAND_TIMEOUT", "60.0")),
            ssl_mode=os.getenv("POSTGRES_SSL_MODE", "prefer"),
            ssl_cert_file=os.getenv("POSTGRES_SSL_CERT_FILE"),
            ssl_key_file=os.getenv("POSTGRES_SSL_KEY_FILE"),
            ssl_ca_file=os.getenv("POSTGRES_SSL_CA_FILE"),
        )


@dataclass
class ConnectionStats:
    """Connection pool statistics."""

    size: int
    checked_out: int
    overflow: int
    checked_in: int
    total_connections: int
    failed_connections: int
    reconnect_count: int
    query_count: int
    average_response_time_ms: float


@dataclass
class QueryMetrics:
    """Query execution metrics."""

    query_hash: str
    execution_time_ms: float
    rows_affected: int
    connection_id: str
    timestamp: float
    was_successful: bool
    error_message: Optional[str] = None


class PostgresConnectionManager:
    """
    Enterprise PostgreSQL connection manager with pooling and monitoring.

    Features:
    - Connection pooling with configurable min/max connections
    - Automatic reconnection and failover handling
    - Query metrics collection and performance monitoring
    - Transaction management with proper isolation levels
    - Schema-aware operations with search path management
    - Health checking and connection validation
    """

    def __init__(self, config: Optional[ConnectionConfig] = None):
        """Initialize connection manager with configuration."""
        self.config = config or ConnectionConfig.from_environment()
        self.pool: Optional[Pool] = None
        self.is_initialized = False
        self.query_metrics: List[QueryMetrics] = []
        self.connection_stats = ConnectionStats(
            size=0,
            checked_out=0,
            overflow=0,
            checked_in=0,
            total_connections=0,
            failed_connections=0,
            reconnect_count=0,
            query_count=0,
            average_response_time_ms=0.0,
        )

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self.is_initialized:
            return

        try:
            # Build connection parameters
            connection_params = {
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "user": self.config.user,
                "password": self.config.password,
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
                        code=CoreErrorCode.DATABASE_CONNECTION_ERROR,
                        message=f"Failed to set schema to {self.config.schema}, got {result}",
                    )

            self.is_initialized = True

        except Exception as e:
            self.connection_stats.failed_connections += 1
            raise OnexError(
                code=CoreErrorCode.DATABASE_CONNECTION_ERROR,
                message=f"Failed to initialize PostgreSQL connection pool: {str(e)}",
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

        Usage:
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

        except Exception as e:
            self.connection_stats.failed_connections += 1
            raise OnexError(
                code=CoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Database connection error: {str(e)}",
            ) from e
        finally:
            if connection:
                await self.pool.release(connection)
                self.connection_stats.checked_in += 1

    @asynccontextmanager
    async def transaction(
        self,
        isolation: str = "read_committed",
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncIterator[Connection]:
        """
        Execute operations within a database transaction.

        Args:
            isolation: Transaction isolation level
            readonly: Whether transaction is read-only
            deferrable: Whether transaction can be deferred

        Usage:
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO table VALUES ($1)", value)
                await conn.execute("UPDATE other_table SET col = $1", value)
        """
        async with self.acquire_connection() as conn:
            async with conn.transaction(
                isolation=isolation, readonly=readonly, deferrable=deferrable
            ):
                yield conn

    async def execute_query(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None,
        record_metrics: bool = True,
    ) -> Union[str, List[Record]]:
        """
        Execute a query with metrics collection.

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout (uses default if None)
            record_metrics: Whether to record query metrics

        Returns:
            Query result (status for non-SELECT, records for SELECT)
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
                else:
                    result = await conn.execute(query, *args, timeout=timeout)
                    # Parse rows affected from result string (e.g., "UPDATE 5")
                    if result and result.split():
                        try:
                            rows_affected = int(result.split()[-1])
                        except (ValueError, IndexError):
                            rows_affected = 0
                    was_successful = True
                    return result

        except Exception as e:
            error_message = str(e)
            raise OnexError(
                code=CoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Query execution failed: {str(e)}",
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

    async def fetch_one(
        self, query: str, *args, timeout: Optional[float] = None
    ) -> Optional[Record]:
        """
        Fetch a single record from a query.

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout

        Returns:
            Single record or None if no results
        """
        try:
            async with self.acquire_connection() as conn:
                return await conn.fetchrow(query, *args, timeout=timeout)
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Single record fetch failed: {str(e)}",
            ) from e

    async def fetch_value(
        self, query: str, *args, timeout: Optional[float] = None
    ) -> Union[List[Record], Record, str, int, float, bool]:
        """
        Fetch a single value from a query.

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout

        Returns:
            Single value from first column of first row
        """
        try:
            async with self.acquire_connection() as conn:
                return await conn.fetchval(query, *args, timeout=timeout)
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Value fetch failed: {str(e)}",
            ) from e

    async def call_function(
        self,
        function_name: str,
        *args,
        schema: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> List[Record]:
        """
        Call a PostgreSQL function.

        Args:
            function_name: Name of the function to call
            *args: Function arguments
            schema: Schema name (uses configured schema if None)
            timeout: Query timeout

        Returns:
            Function result as list of records
        """
        schema_prefix = f"{schema or self.config.schema}."
        placeholders = ", ".join(f"${i+1}" for i in range(len(args)))
        query = f"SELECT * FROM {schema_prefix}{function_name}({placeholders})"

        return await self.execute_query(query, *args, timeout=timeout)

    async def health_check(
        self,
    ) -> Dict[str, Union[str, bool, int, float, Dict[str, Union[str, int, float]]]]:
        """
        Perform comprehensive health check of the database connection.

        Returns:
            Health check results with status and metrics
        """
        health_status = {
            "status": "unhealthy",
            "timestamp": time.time(),
            "connection_pool": {},
            "database_info": {},
            "schema_info": {},
            "performance": {},
            "errors": [],
        }

        try:
            if not self.is_initialized:
                health_status["errors"].append("Connection manager not initialized")
                return health_status

            # Test basic connectivity
            start_time = time.perf_counter()
            async with self.acquire_connection() as conn:
                # Basic connectivity test
                version = await conn.fetchval("SELECT version()")
                current_schema = await conn.fetchval("SELECT current_schema()")
                connection_count = await conn.fetchval(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = $1",
                    self.config.database,
                )

                # Schema validation
                schema_exists = await conn.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = $1)",
                    self.config.schema,
                )

            response_time_ms = (time.perf_counter() - start_time) * 1000

            # Update health status
            health_status.update(
                {
                    "status": "healthy",
                    "connection_pool": {
                        "size": self.pool.get_size(),
                        "min_size": self.pool.get_min_size(),
                        "max_size": self.pool.get_max_size(),
                        "idle_connections": self.pool.get_idle_size(),
                    },
                    "database_info": {
                        "version": version,
                        "current_schema": current_schema,
                        "active_connections": connection_count,
                    },
                    "schema_info": {
                        "schema": self.config.schema,
                        "exists": schema_exists,
                    },
                    "performance": {
                        "response_time_ms": response_time_ms,
                        "total_queries": self.connection_stats.query_count,
                        "failed_connections": self.connection_stats.failed_connections,
                        "average_response_time_ms": self.connection_stats.average_response_time_ms,
                    },
                }
            )

            if not schema_exists:
                health_status["errors"].append(
                    f"Schema '{self.config.schema}' does not exist"
                )
                health_status["status"] = "degraded"

        except Exception as e:
            health_status["errors"].append(f"Health check failed: {str(e)}")
            health_status["status"] = "unhealthy"

        return health_status

    def get_connection_stats(self) -> ConnectionStats:
        """Get current connection pool statistics."""
        if self.pool:
            self.connection_stats.size = self.pool.get_size()
            self.connection_stats.total_connections = self.pool.get_size()

        return self.connection_stats

    def get_query_metrics(self, limit: int = 100) -> List[QueryMetrics]:
        """Get recent query metrics."""
        return self.query_metrics[-limit:]

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self.query_metrics.clear()
        self.connection_stats = ConnectionStats(
            size=0,
            checked_out=0,
            overflow=0,
            checked_in=0,
            total_connections=0,
            failed_connections=0,
            reconnect_count=0,
            query_count=0,
            average_response_time_ms=0.0,
        )

    def _record_query_metrics(
        self,
        query_hash: str,
        execution_time_ms: float,
        rows_affected: int,
        connection_id: str,
        was_successful: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Record query execution metrics."""
        metric = QueryMetrics(
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


# Global connection manager instance
_connection_manager: Optional[PostgresConnectionManager] = None


def get_connection_manager() -> PostgresConnectionManager:
    """Get the global connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = PostgresConnectionManager()
    return _connection_manager


async def initialize_database() -> None:
    """Initialize the global database connection."""
    manager = get_connection_manager()
    await manager.initialize()


async def close_database() -> None:
    """Close the global database connection."""
    global _connection_manager
    if _connection_manager:
        await _connection_manager.close()
        _connection_manager = None