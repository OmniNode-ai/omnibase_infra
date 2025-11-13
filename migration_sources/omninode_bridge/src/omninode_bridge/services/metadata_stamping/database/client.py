# === OmniNode:Tool_Metadata ===
# metadata_version: 0.1
# name: metadata_stamping_database_client
# title: MetadataStampingService Database Client
# version: 0.1.0
# namespace: omninode.services.metadata
# category: service.infrastructure.stamping
# kind: service
# role: database_client
# description: |
#   PostgreSQL database client for metadata stamping operations with advanced
#   connection pooling, prepared statements, batch operations, and comprehensive
#   error handling for high-performance data persistence.
# tags: [database, client, postgresql, stamping, metadata, persistence]
# author: OmniNode Development Team
# license: MIT
# entrypoint: client.py
# protocols_supported: [O.N.E. v0.1]
# runtime_constraints: {sandboxed: false, privileged: false, requires_network: true, requires_gpu: false}
# dependencies: [{"name": "asyncpg", "version": "^0.29.0"}]
# environment: [python>=3.11, postgresql>=13]
# === /OmniNode:Tool_Metadata ===

"""PostgreSQL client for metadata stamping service.

High-performance PostgreSQL client with connection pooling and optimization
for metadata stamping operations.

ROBUST ERROR HANDLING:
- Comprehensive None checking throughout all database operations
- Custom exception hierarchy for clear error classification
- Circuit breaker pattern for resilience
- Validation of all inputs and outputs
- Graceful degradation and error recovery
- Performance monitoring with failure tracking

This client acts as "nunchucks for crashes" - preventing runtime failures
from None returns and database operation errors.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import asyncpg
import orjson
from asyncpg import Pool

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base database error for metadata stamping service."""

    pass


class DatabaseOperationError(DatabaseError):
    """Error raised when database operation returns None or invalid result."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Error raised when database connection fails."""

    pass


class ConnectionState(Enum):
    """Database connection state enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class DatabaseConfig:
    """Database configuration with performance optimization."""

    host: str
    port: int
    database: str
    user: str
    password: str
    min_connections: int = 10
    max_connections: int = 50
    command_timeout: float = 30.0
    connection_timeout: float = 10.0
    ssl_enabled: bool = False
    ssl_config: Optional[dict[str, Any]] = None


class CircuitBreaker:
    """Circuit breaker pattern for database resilience."""

    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: int,
        expected_exception: Union[type, tuple],
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Exception type or tuple of exception types to handle
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    async def __aenter__(self):
        """Enter circuit breaker context."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                # Use the first exception type if it's a tuple, otherwise use the single type
                exception_to_raise = (
                    self.expected_exception[0]
                    if isinstance(self.expected_exception, tuple)
                    else self.expected_exception
                )
                raise exception_to_raise("Circuit breaker is open")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit circuit breaker context."""
        if exc_type and issubclass(exc_type, self.expected_exception):
            await self.record_failure()
        else:
            await self.record_success()

    async def record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    async def record_success(self):
        """Record a success and potentially close the circuit."""
        self.failure_count = 0
        self.state = "closed"


class MetadataStampingPostgresClient:
    """High-performance PostgreSQL client for metadata stamping service."""

    def __init__(self, config: DatabaseConfig):
        """Initialize the database client.

        Args:
            config: Database configuration
        """
        self.config = config
        self.pool: Optional[Pool] = None
        self.state = ConnectionState.DISCONNECTED

        # Performance tracking
        self.connection_metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "connection_errors": 0,
            "query_count": 0,
            "avg_query_time": 0.0,
        }

        # Rely on asyncpg's automatic statement caching for optimal performance

        # Circuit breaker for resilience - only trigger on database/connection errors
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=(asyncpg.PostgresError, asyncio.TimeoutError),
        )

    async def initialize(self) -> bool:
        """Initialize database connection pool with optimization.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.state = ConnectionState.CONNECTING

            # Validate configuration
            if not self.config:
                raise DatabaseConnectionError("Database configuration is None")

            required_config_fields = ["host", "port", "database", "user", "password"]
            for field in required_config_fields:
                if not getattr(self.config, field, None):
                    raise DatabaseConnectionError(
                        f"Database configuration missing required field: {field}"
                    )

            # Build connection parameters
            connection_params = {
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "user": self.config.user,
                "password": self.config.password,
                "min_size": self.config.min_connections,
                "max_size": self.config.max_connections,
                "command_timeout": self.config.command_timeout,
                "server_settings": {
                    "application_name": "metadata_stamping_service",
                    "tcp_keepalives_idle": "600",
                    "tcp_keepalives_interval": "30",
                    "tcp_keepalives_count": "3",
                },
            }

            # Create connection pool
            self.pool = await asyncpg.create_pool(**connection_params)

            if self.pool is None:
                raise DatabaseConnectionError(
                    "Failed to create connection pool - returned None"
                )

            # Verify pool connection
            async with self.pool.acquire() as connection:
                if connection is None:
                    raise DatabaseConnectionError("Failed to acquire test connection")
                # Test basic connectivity
                await connection.fetchval("SELECT 1")

            self.state = ConnectionState.CONNECTED
            self.connection_metrics["total_connections"] += self.config.max_connections

            logger.info(
                f"Database pool initialized: {self.config.min_connections}-{self.config.max_connections} connections, using asyncpg automatic statement caching"
            )
            return True

        except asyncpg.exceptions.PostgresError as e:
            self.state = ConnectionState.ERROR
            self.connection_metrics["connection_errors"] += 1
            logger.error(f"PostgreSQL error during initialization: {e}")
            return False

        except Exception as e:
            self.state = ConnectionState.ERROR
            self.connection_metrics["connection_errors"] += 1
            logger.error(f"Failed to initialize database pool: {e}")
            return False

    @asynccontextmanager
    async def acquire_connection(self):
        """Acquire database connection with automatic cleanup and metrics.

        Yields:
            Database connection

        Raises:
            DatabaseConnectionError: If database not connected or connection fails
        """
        if self.state != ConnectionState.CONNECTED:
            raise DatabaseConnectionError("Database not connected")

        if self.pool is None:
            raise DatabaseConnectionError("Database pool is None")

        start_time = time.perf_counter()
        connection = None

        try:
            connection = await self.pool.acquire()
            if connection is None:
                raise DatabaseConnectionError("Pool returned None connection")

            self.connection_metrics["active_connections"] += 1
            yield connection

        except asyncpg.exceptions.PostgresError as e:
            self.connection_metrics["connection_errors"] += 1
            await self.circuit_breaker.record_failure()
            logger.error(f"PostgreSQL connection error: {e}")
            raise DatabaseConnectionError(
                f"Failed to acquire database connection: {e}"
            ) from e

        except Exception as e:
            self.connection_metrics["connection_errors"] += 1
            await self.circuit_breaker.record_failure()
            logger.error(f"Unexpected connection error: {e}")
            raise DatabaseConnectionError(f"Unexpected connection error: {e}") from e

        finally:
            if connection:
                try:
                    await self.pool.release(connection)
                except Exception as e:
                    logger.warning(f"Failed to release connection: {e}")
                finally:
                    self.connection_metrics["active_connections"] -= 1

            # Update query metrics
            execution_time = (time.perf_counter() - start_time) * 1000
            self.connection_metrics["query_count"] += 1

            # Rolling average calculation with None check
            if self.connection_metrics.get("avg_query_time") is not None:
                current_avg = self.connection_metrics["avg_query_time"]
                count = self.connection_metrics["query_count"]
                self.connection_metrics["avg_query_time"] = (
                    (current_avg * (count - 1)) + execution_time
                ) / count
            else:
                self.connection_metrics["avg_query_time"] = execution_time

    async def execute_query(self, query: str, *args, fetch_mode: str = "all") -> Any:
        """Execute query with performance monitoring and error handling.

        Args:
            query: SQL query to execute
            *args: Query arguments
            fetch_mode: One of "all", "one", "val", or "execute"

        Returns:
            Query results based on fetch mode

        Raises:
            DatabaseOperationError: If database operation returns None unexpectedly
            DatabaseConnectionError: If database connection fails
        """
        async with self.circuit_breaker:
            async with self.acquire_connection() as connection:
                start_time = time.perf_counter()

                try:
                    if fetch_mode == "all":
                        result = await connection.fetch(query, *args)
                        if result is None:
                            raise DatabaseOperationError(
                                f"Database fetch operation returned None for query: {query[:100]}..."
                            )
                    elif fetch_mode == "one":
                        result = await connection.fetchrow(query, *args)
                        # Note: fetchrow can legitimately return None when no rows found
                    elif fetch_mode == "val":
                        result = await connection.fetchval(query, *args)
                        # Note: fetchval can legitimately return None when value is NULL
                    else:  # execute
                        result = await connection.execute(query, *args)
                        if result is None:
                            raise DatabaseOperationError(
                                f"Database execute operation returned None for query: {query[:100]}..."
                            )

                    execution_time = (time.perf_counter() - start_time) * 1000

                    # Log slow queries
                    if execution_time > 100:  # > 100ms
                        logger.warning(
                            f"Slow query detected: {execution_time:.2f}ms - {query[:100]}..."
                        )

                    return result

                except asyncpg.exceptions.PostgresError as e:
                    execution_time = (time.perf_counter() - start_time) * 1000
                    logger.error(
                        f"Database query failed after {execution_time:.2f}ms: {e}"
                    )
                    raise DatabaseConnectionError(f"PostgreSQL error: {e}") from e
                except DatabaseOperationError:
                    # Re-raise our custom database operation errors
                    raise
                except Exception as e:
                    execution_time = (time.perf_counter() - start_time) * 1000
                    logger.error(f"Query failed after {execution_time:.2f}ms: {e}")
                    raise DatabaseError(f"Unexpected database error: {e}") from e

    async def execute_transaction(self, operations: list[tuple]) -> list[Any]:
        """Execute multiple operations in a single transaction with rollback support.

        Args:
            operations: List of (query, args) tuples

        Returns:
            List of results from each operation

        Raises:
            DatabaseOperationError: If any operation returns None unexpectedly
            DatabaseError: If transaction fails
        """
        if not operations:
            raise DatabaseOperationError("Cannot execute empty transaction")

        async with self.acquire_connection() as connection:
            try:
                async with connection.transaction():
                    results = []
                    for i, operation in enumerate(operations):
                        if not operation:
                            raise DatabaseOperationError(
                                f"Operation {i} is None or empty"
                            )

                        query, args = operation[0], (
                            operation[1:] if len(operation) > 1 else ()
                        )
                        if not query:
                            raise DatabaseOperationError(f"Query {i} is None or empty")

                        result = await connection.fetch(query, *args)
                        if result is None:
                            raise DatabaseOperationError(
                                f"Transaction operation {i} returned None for query: {query[:100]}..."
                            )
                        results.append(result)

                    return results
            except asyncpg.exceptions.PostgresError as e:
                logger.error(f"Transaction failed with PostgreSQL error: {e}")
                raise DatabaseConnectionError(f"Transaction failed: {e}") from e
            except DatabaseOperationError:
                # Re-raise our custom database operation errors
                raise
            except Exception as e:
                logger.error(f"Transaction failed with unexpected error: {e}")
                raise DatabaseError(f"Transaction failed: {e}") from e

    # High-level operations for metadata stamping

    async def create_metadata_stamp(
        self,
        file_hash: str,
        file_path: str,
        file_size: int,
        content_type: str,
        stamp_data: dict,
        protocol_version: str = "1.0",
        intelligence_data: dict = None,
        version: int = 1,
        op_id: str = None,
        namespace: str = "omninode.services.metadata",
        metadata_version: str = "0.1",
    ) -> dict:
        """Create a new metadata stamp with optimized insertion.

        Args:
            file_hash: BLAKE3 hash of the file
            file_path: Path to the file
            file_size: Size of the file in bytes
            content_type: MIME content type
            stamp_data: Stamp metadata as dictionary
            protocol_version: Protocol version
            intelligence_data: Intelligence data as dictionary (optional)
            version: Schema version (default: 1)
            op_id: Operation ID (generated if None)
            namespace: Namespace (default: omninode.services.metadata)
            metadata_version: Metadata version (default: 0.1)

        Returns:
            Created stamp information

        Raises:
            DatabaseOperationError: If stamp creation returns None or invalid result
            ValueError: If required parameters are None or invalid
        """
        # Validate required parameters
        if not file_hash:
            raise ValueError("file_hash cannot be None or empty")
        if not file_path:
            raise ValueError("file_path cannot be None or empty")
        if stamp_data is None:
            raise ValueError("stamp_data cannot be None")
        if not namespace:
            raise ValueError("namespace cannot be None or empty")

        # Set defaults for optional compliance fields
        if intelligence_data is None:
            intelligence_data = {}
        if op_id is None:
            import uuid

            op_id = str(uuid.uuid4())

        query = """
            INSERT INTO metadata_stamps (
                file_hash, file_path, file_size, content_type, stamp_data, protocol_version,
                intelligence_data, version, op_id, namespace, metadata_version
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id, created_at, op_id
        """

        # Convert stamp_data and intelligence_data to JSON
        try:
            stamp_data_json = orjson.dumps(stamp_data).decode("utf-8")
            intelligence_data_json = orjson.dumps(intelligence_data).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to serialize data to JSON: {e}")

        result = await self.execute_query(
            query,
            file_hash,
            file_path,
            file_size,
            content_type,
            stamp_data_json,
            protocol_version,
            intelligence_data_json,
            version,
            op_id,
            namespace,
            metadata_version,
            fetch_mode="one",
        )

        if result is None:
            raise DatabaseOperationError(
                "Failed to create metadata stamp - database returned None"
            )

        if "id" not in result or "created_at" not in result or "op_id" not in result:
            raise DatabaseOperationError(
                "Failed to create metadata stamp - missing required fields in result"
            )

        return {
            "id": str(result["id"]),
            "file_hash": file_hash,
            "op_id": str(result["op_id"]),
            "namespace": namespace,
            "version": version,
            "metadata_version": metadata_version,
            "created_at": result["created_at"],
        }

    async def get_metadata_stamp(self, file_hash: str) -> Optional[dict]:
        """Retrieve metadata stamp by file hash.

        Args:
            file_hash: BLAKE3 hash to look up

        Returns:
            Stamp record if found, None otherwise

        Raises:
            ValueError: If file_hash is None or empty
        """
        if not file_hash:
            raise ValueError("file_hash cannot be None or empty")

        query = """
            SELECT id, file_hash, file_path, file_size, content_type, stamp_data, protocol_version,
                   intelligence_data, version, op_id, namespace, metadata_version, created_at, updated_at
            FROM metadata_stamps
            WHERE file_hash = $1
        """

        result = await self.execute_query(query, file_hash, fetch_mode="one")

        # Note: result can legitimately be None when no record found
        if result:
            stamp_dict = dict(result)
            # Parse JSONB columns if they are strings
            if isinstance(stamp_dict.get("stamp_data"), str):
                import orjson

                stamp_dict["stamp_data"] = orjson.loads(stamp_dict["stamp_data"])
            if isinstance(stamp_dict.get("intelligence_data"), str):
                import orjson

                stamp_dict["intelligence_data"] = orjson.loads(
                    stamp_dict["intelligence_data"]
                )
            return stamp_dict
        return None

    async def batch_insert_stamps(self, stamps_data: list[dict]) -> list[str]:
        """Efficiently insert multiple metadata stamps using batch operations.

        Args:
            stamps_data: List of stamp data dictionaries

        Returns:
            List of inserted stamp IDs

        Raises:
            DatabaseOperationError: If batch operation returns None or invalid result
            ValueError: If stamps_data is None or contains invalid data
        """
        if stamps_data is None:
            raise ValueError("stamps_data cannot be None")

        if not stamps_data:
            return []

        # Validate and prepare batch data
        batch_values = []
        for i, stamp in enumerate(stamps_data):
            if stamp is None:
                raise ValueError(f"Stamp data at index {i} is None")

            required_fields = [
                "file_hash",
                "file_path",
                "file_size",
                "content_type",
                "stamp_data",
            ]

            # Optional compliance fields with defaults
            optional_fields = {
                "intelligence_data": {},
                "version": 1,
                "op_id": None,
                "namespace": "omninode.services.metadata",
                "metadata_version": "0.1",
            }
            for field in required_fields:
                if field not in stamp:
                    raise ValueError(
                        f"Stamp data at index {i} missing required field: {field}"
                    )
                if field != "stamp_data" and not stamp[field]:
                    raise ValueError(
                        f"Stamp data at index {i} has None/empty value for field: {field}"
                    )

            # Apply defaults for optional fields
            for field, default_value in optional_fields.items():
                if field not in stamp:
                    stamp[field] = default_value

            # Generate op_id if not provided
            if stamp["op_id"] is None:
                import uuid

                stamp["op_id"] = str(uuid.uuid4())

            try:
                stamp_data_json = orjson.dumps(stamp["stamp_data"]).decode("utf-8")
                intelligence_data_json = orjson.dumps(
                    stamp["intelligence_data"]
                ).decode("utf-8")
            except Exception as e:
                raise ValueError(f"Failed to serialize data at index {i} to JSON: {e}")

            batch_values.append(
                (
                    stamp["file_hash"],
                    stamp["file_path"],
                    stamp["file_size"],
                    stamp["content_type"],
                    stamp_data_json,
                    stamp.get("protocol_version", "1.0"),
                    intelligence_data_json,
                    stamp["version"],
                    stamp["op_id"],
                    stamp["namespace"],
                    stamp["metadata_version"],
                )
            )

        # Execute batch insertion with transaction
        async with self.acquire_connection() as connection:
            try:
                async with connection.transaction():
                    query = """
                        INSERT INTO metadata_stamps (
                            file_hash, file_path, file_size, content_type, stamp_data, protocol_version,
                            intelligence_data, version, op_id, namespace, metadata_version
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (file_hash) DO NOTHING
                        RETURNING id
                    """

                    # Use executemany for optimal batch performance
                    results = await connection.executemany(query, batch_values)

                    # Critical None check: the "nunchucks for crashes"
                    if results is None:
                        raise DatabaseOperationError(
                            "Batch insert operation returned None - database executemany failed"
                        )

                    # Extract inserted IDs with additional validation
                    inserted_ids = []
                    for i, result in enumerate(results):
                        if result is not None:
                            try:
                                inserted_ids.append(str(result))
                            except (TypeError, ValueError) as e:
                                logger.warning(
                                    f"Failed to convert result {i} to string: {e}"
                                )

                    logger.info(
                        f"Batch inserted {len(inserted_ids)} metadata stamps out of {len(stamps_data)} attempted"
                    )
                    return inserted_ids

            except asyncpg.exceptions.PostgresError as e:
                logger.error(f"Batch insert failed with PostgreSQL error: {e}")
                raise DatabaseConnectionError(f"Batch insert failed: {e}") from e
            except DatabaseOperationError:
                # Re-raise our custom database operation errors
                raise
            except Exception as e:
                logger.error(f"Batch insert failed with unexpected error: {e}")
                raise DatabaseError(f"Batch insert failed: {e}") from e

    async def record_performance_metric(
        self,
        operation_type: str,
        execution_time_ms: float,
        file_size_bytes: int,
        cpu_usage: float,
        memory_usage: int,
    ):
        """Record performance metrics for monitoring.

        Args:
            operation_type: Type of operation
            execution_time_ms: Execution time in milliseconds
            file_size_bytes: File size in bytes
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage in MB

        Raises:
            ValueError: If required parameters are None or invalid
            DatabaseOperationError: If metric recording fails
        """
        # Validate required parameters
        if not operation_type:
            raise ValueError("operation_type cannot be None or empty")
        if execution_time_ms is None or execution_time_ms < 0:
            raise ValueError("execution_time_ms must be a non-negative number")
        if file_size_bytes is None or file_size_bytes < 0:
            raise ValueError("file_size_bytes must be a non-negative number")

        query = """
            INSERT INTO hash_metrics (operation_type, execution_time_ms, file_size_bytes, cpu_usage_percent, memory_usage_mb)
            VALUES ($1, $2, $3, $4, $5)
        """

        # Execute INSERT query - result intentionally ignored for INSERT operations
        await self.execute_query(
            query,
            operation_type,
            execution_time_ms,
            file_size_bytes,
            cpu_usage,
            memory_usage,
            fetch_mode="execute",
        )

        # Note: execute mode can return None for INSERT operations without RETURNING clause
        # This is expected behavior, so we don't check for None here

    async def get_performance_statistics(self) -> list[dict]:
        """Get aggregated performance statistics.

        Returns:
            List of performance statistics by operation type

        Raises:
            DatabaseOperationError: If statistics retrieval fails
        """
        query = """
            SELECT
                operation_type,
                COUNT(*) as operation_count,
                AVG(execution_time_ms) as avg_execution_time,
                MAX(execution_time_ms) as max_execution_time,
                AVG(cpu_usage_percent) as avg_cpu_usage,
                AVG(memory_usage_mb) as avg_memory_usage
            FROM hash_metrics
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            GROUP BY operation_type
        """

        results = await self.execute_query(query, fetch_mode="all")

        # Note: results can legitimately be empty list if no metrics found
        if results is None:
            raise DatabaseOperationError("Performance statistics query returned None")

        return [dict(row) for row in results if row is not None]

    async def health_check(self) -> dict[str, Any]:
        """Perform database health check with connection and performance metrics.

        Returns:
            Health check results
        """
        try:
            # Simple connectivity test
            start_time = time.perf_counter()
            result = await self.execute_query("SELECT 1", fetch_mode="val")
            response_time = (time.perf_counter() - start_time) * 1000

            # Validate connectivity test result
            if result is None:
                raise DatabaseOperationError(
                    "Health check query returned None - database connectivity issue"
                )

            if result != 1:
                raise DatabaseOperationError(
                    f"Health check query returned unexpected value: {result}"
                )

            # Get pool statistics with None checks
            if self.pool:
                try:
                    pool_stats = {
                        "size": self.pool.get_size(),
                        "min_size": self.pool.get_min_size(),
                        "max_size": self.pool.get_max_size(),
                        "idle_connections": self.pool.get_idle_size(),
                    }
                except Exception as e:
                    logger.warning(f"Failed to get pool statistics: {e}")
                    pool_stats = {"error": f"Pool stats unavailable: {e}"}
            else:
                pool_stats = {"error": "Pool not initialized"}

            return {
                "status": "healthy",
                "state": self.state.value,
                "response_time_ms": response_time,
                "pool_statistics": pool_stats,
                "connection_metrics": self.connection_metrics or {},
                "statement_caching": "asyncpg_automatic",
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "state": self.state.value,
                "error": str(e),
                "connection_metrics": self.connection_metrics or {},
                "error_type": type(e).__name__,
            }

    async def close(self):
        """Close database connection pool gracefully.

        Raises:
            DatabaseError: If pool closure fails
        """
        if self.pool:
            try:
                await self.pool.close()
                self.state = ConnectionState.DISCONNECTED
                self.pool = None
                logger.info("Database connection pool closed successfully")
            except Exception as e:
                self.state = ConnectionState.ERROR
                logger.error(f"Failed to close database pool: {e}")
                raise DatabaseError(f"Failed to close database pool: {e}") from e
        else:
            logger.info("Database pool already closed or not initialized")
