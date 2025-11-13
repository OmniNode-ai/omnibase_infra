"""
PostgreSQL Connection Manager with Connection Pooling.

Provides high-performance PostgreSQL connection pooling following omnibase_infra
architecture patterns with async/await, context managers, metrics collection,
and health monitoring.

Features:
- asyncpg connection pooling (5-50 connections)
- Context managers for connections and transactions
- Query execution with automatic SELECT/DML detection
- Metrics collection for performance monitoring
- Health monitoring and pool statistics
- Docker secrets support for password security
- Schema search_path configuration

Implementation: Following omnibase_infra PostgresConnectionManager patterns
Reference: docs/DATABASE_ADAPTER_PATTERNS.md Section 3
"""

import hashlib
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import asyncpg
    from asyncpg import Connection, Record
    from asyncpg.pool import Pool

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    # Fallback for type hints when asyncpg is not installed
    Connection = Any  # type: ignore
    Record = Any  # type: ignore
    Pool = Any  # type: ignore

from pydantic import BaseModel, Field, field_validator

try:
    from omninode_bridge.security.validation import InputSanitizer
except ImportError:
    # Fallback for when validation is not available
    class InputSanitizer:  # type: ignore[no-redef]
        @staticmethod
        def validate_sql_identifier(value: str, max_length: int = 63) -> str:
            if not value or not isinstance(value, str):
                raise ValueError("SQL identifier must be a non-empty string")
            if len(value) > max_length:
                raise ValueError(
                    f"SQL identifier too long. Maximum length: {max_length}"
                )
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", value):
                raise ValueError(
                    "SQL identifier contains invalid characters or invalid start character"
                )
            return value


# Direct imports - omnibase_core is required
from omnibase_core import EnumCoreErrorCode, ModelOnexError

OnexError = ModelOnexError

OMNIBASE_CORE_AVAILABLE = True


logger = logging.getLogger(__name__)


# Configuration Models


class ModelPostgresConfig(BaseModel):
    """
    PostgreSQL connection configuration model.

    Supports loading from environment variables with Docker secrets support
    for secure password management.

    Environment Variables:
        - POSTGRES_HOST: Database host (default: localhost)
        - POSTGRES_PORT: Database port (default: 5432)
        - POSTGRES_DATABASE: Database name (required)
        - POSTGRES_USER: Database user (required)
        - POSTGRES_PASSWORD: Database password (fallback, not recommended)
        - POSTGRES_PASSWORD_FILE: Path to Docker secret file (recommended)
        - POSTGRES_SCHEMA: Default schema (default: public)
        - POSTGRES_MIN_CONNECTIONS: Minimum pool size (default: 5)
        - POSTGRES_MAX_CONNECTIONS: Maximum pool size (default: 50)
        - POSTGRES_COMMAND_TIMEOUT: Query timeout in seconds (default: 60)
        - POSTGRES_MAX_INACTIVE_CONNECTION_LIFETIME: Connection lifetime (default: 300)
        - POSTGRES_POOL_EXHAUSTION_THRESHOLD: Pool utilization threshold (default: 0.90 for 90%)
        - POSTGRES_POOL_EXHAUSTION_LOG_INTERVAL: Seconds between exhaustion warnings (default: 60)
        - POSTGRES_MAX_METRICS_STORED: Maximum query metrics to store (default: 10000, range: 1000-1000000)

    Docker Secrets:
        For production deployments, use POSTGRES_PASSWORD_FILE pointing to a
        Docker secret file instead of POSTGRES_PASSWORD environment variable.
    """

    host: str = Field(default="localhost", description="PostgreSQL host address")
    port: int = Field(default=5432, description="PostgreSQL port", ge=1, le=65535)
    database: str = Field(..., description="Database name (required)")
    user: str = Field(..., description="Database user (required)")
    password: str = Field(default="", description="Database password")
    schema: str = Field(default="public", description="Default schema name")

    # Connection pool configuration
    min_connections: int = Field(
        default=5, description="Minimum pool size", ge=1, le=100
    )
    max_connections: int = Field(
        default=50, description="Maximum pool size", ge=5, le=200
    )
    max_inactive_connection_lifetime: float = Field(
        default=300.0, description="Max connection lifetime in seconds", ge=60.0
    )
    command_timeout: float = Field(
        default=60.0, description="Query timeout in seconds", ge=1.0
    )
    max_queries: int = Field(
        default=50000, description="Max queries per connection", ge=1000
    )

    # Pool exhaustion monitoring configuration
    pool_exhaustion_threshold: float = Field(
        default=0.90,
        description="Pool utilization threshold for exhaustion warnings (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    pool_exhaustion_log_interval: int = Field(
        default=60,
        description="Minimum seconds between exhaustion log warnings",
        ge=1,
        le=3600,
    )

    # Query metrics configuration
    max_metrics_stored: int = Field(
        default=10000,
        description="Maximum query metrics to store in circular buffer",
        ge=1000,
        le=1000000,
    )

    @field_validator("max_connections")
    @classmethod
    def validate_max_connections(cls, v: int, info) -> int:
        """Validate that max_connections >= min_connections."""
        if "min_connections" in info.data and v < info.data["min_connections"]:
            raise ValueError(
                f"max_connections ({v}) must be >= min_connections ({info.data['min_connections']})"
            )
        return v

    @classmethod
    def _normalize_threshold(cls, value: float) -> float:
        """
        Normalize pool exhaustion threshold to 0.0-1.0 range.

        Handles both percentage format (e.g., 90 for 90%) and fraction format (e.g., 0.9).

        Args:
            value: Threshold value (either percentage or fraction)

        Returns:
            float: Normalized value between 0.0 and 1.0

        Examples:
            _normalize_threshold(90) -> 0.9
            _normalize_threshold(0.9) -> 0.9
        """
        if value > 1.0:
            # Assume percentage format (e.g., 90 means 90%)
            return value / 100.0
        return value

    @classmethod
    def from_environment(cls) -> "ModelPostgresConfig":
        """
        Load configuration from environment variables.

        Supports Docker secrets via POSTGRES_PASSWORD_FILE for secure
        password management in production environments.

        Returns:
            ModelPostgresConfig instance with values from environment

        Raises:
            OnexError: If required configuration is missing or invalid
        """
        # Load password from Docker secrets if available
        password = ""
        password_file = os.getenv("POSTGRES_PASSWORD_FILE")
        if password_file and os.path.exists(password_file):
            try:
                with open(password_file) as f:
                    password = f.read().strip()
                logger.info(
                    "Loaded PostgreSQL password from Docker secrets",
                    extra={"password_file": password_file},
                )
            except Exception as e:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                    message=f"Failed to read password from Docker secrets: {e}",
                    context={"password_file": password_file},
                ) from e
        else:
            # Fallback to environment variable (not recommended for production)
            password = os.getenv("POSTGRES_PASSWORD", "")
            if password:
                logger.warning(
                    "Using POSTGRES_PASSWORD from environment - consider using POSTGRES_PASSWORD_FILE for production"
                )

        # Validate required fields
        database = os.getenv("POSTGRES_DATABASE")
        user = os.getenv("POSTGRES_USER")

        if not database or not user:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.MISSING_REQUIRED_PARAMETER,
                message="POSTGRES_DATABASE and POSTGRES_USER are required",
                context={
                    "database_set": bool(database),
                    "user_set": bool(user),
                },
            )

        # Build configuration
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=database,
            user=user,
            password=password,
            schema=os.getenv("POSTGRES_SCHEMA", "public"),
            min_connections=int(os.getenv("POSTGRES_MIN_CONNECTIONS", "5")),
            max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "50")),
            max_inactive_connection_lifetime=float(
                os.getenv("POSTGRES_MAX_INACTIVE_CONNECTION_LIFETIME", "300.0")
            ),
            command_timeout=float(os.getenv("POSTGRES_COMMAND_TIMEOUT", "60.0")),
            max_queries=int(os.getenv("POSTGRES_MAX_QUERIES", "50000")),
            pool_exhaustion_threshold=cls._normalize_threshold(
                float(os.getenv("POSTGRES_POOL_EXHAUSTION_THRESHOLD", "0.90"))
            ),
            pool_exhaustion_log_interval=int(
                os.getenv("POSTGRES_POOL_EXHAUSTION_LOG_INTERVAL", "60")
            ),
            max_metrics_stored=int(os.getenv("POSTGRES_MAX_METRICS_STORED", "10000")),
        )


# Metrics and Statistics Models


@dataclass
class ConnectionStats:
    """Connection pool statistics for monitoring."""

    checked_out: int = 0
    checked_in: int = 0
    pool_size: int = 0
    pool_free: int = 0
    pool_max: int = 0


@dataclass
class QueryMetrics:
    """Query execution metrics."""

    query_hash: str
    execution_time_ms: float
    query_type: str  # SELECT, INSERT, UPDATE, DELETE, etc.
    timestamp: float = field(default_factory=time.time)
    performance_category: str = "unknown"  # fast, slow

    def __post_init__(self):
        """Calculate performance category based on execution time."""
        if self.execution_time_ms < 100:
            self.performance_category = "fast"
        else:
            self.performance_category = "slow"


# PostgreSQL Connection Manager


class PostgresConnectionManager:
    """
    High-performance PostgreSQL connection manager with connection pooling.

    Implements asyncpg connection pooling with context managers, query execution,
    metrics collection, and health monitoring following omnibase_infra patterns.

    Features:
        - Connection pooling (5-50 connections by default)
        - Context managers for automatic resource cleanup
        - Transaction management with ACID compliance
        - Query metrics collection for performance monitoring
        - Health monitoring and pool statistics
        - Schema search_path configuration
        - Docker secrets support for password security

    Usage:
        ```python
        # Initialize connection manager
        config = ModelPostgresConfig.from_environment()
        manager = PostgresConnectionManager(config)
        await manager.initialize()

        # Execute query with automatic connection management
        results = await manager.execute_query("SELECT * FROM users WHERE id = $1", 1)

        # Use context manager for explicit connection control
        async with manager.acquire_connection() as conn:
            result = await conn.fetchrow("SELECT * FROM users WHERE id = $1", 1)

        # Use transactions for ACID compliance
        async with manager.transaction() as conn:
            await conn.execute("INSERT INTO users (name) VALUES ($1)", "Alice")
            await conn.execute("UPDATE accounts SET balance = balance + 100 WHERE user_id = $1", 1)

        # Cleanup
        await manager.close()
        ```

    Reference: omnibase_infra PostgresConnectionManager
    """

    # Maximum number of query metrics to store (circular buffer)
    MAX_METRICS_STORED = 10000

    def __init__(self, config: ModelPostgresConfig):
        """
        Initialize PostgreSQL connection manager.

        Args:
            config: PostgreSQL configuration

        Raises:
            OnexError: If asyncpg is not available
        """
        # Check asyncpg availability
        if not ASYNCPG_AVAILABLE:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.DEPENDENCY_ERROR,
                message="asyncpg not installed - required for PostgreSQL connection pooling",
                context={"install_command": "pip install asyncpg"},
            )

        self.config = config
        self.pool: Optional[Pool] = None
        self.connection_stats = ConnectionStats()
        self._query_metrics: list[QueryMetrics] = []
        self._is_initialized = False

        # Metrics configuration
        self._max_metrics_stored = config.max_metrics_stored

        # Pool exhaustion monitoring
        self._last_exhaustion_warning: float = 0.0
        self._exhaustion_warning_count: int = 0

        logger.info(
            "PostgresConnectionManager initialized",
            extra={
                "host": config.host,
                "port": config.port,
                "database": config.database,
                "schema": config.schema,
                "min_connections": config.min_connections,
                "max_connections": config.max_connections,
            },
        )

    async def initialize(self) -> None:
        """
        Initialize connection pool.

        Creates asyncpg connection pool with configured parameters.

        Raises:
            OnexError: If pool initialization fails
        """
        if self._is_initialized:
            logger.warning("Connection pool already initialized")
            return

        try:
            logger.info(
                "Initializing PostgreSQL connection pool",
                extra={
                    "min_size": self.config.min_connections,
                    "max_size": self.config.max_connections,
                    "database": self.config.database,
                },
            )

            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
                max_queries=self.config.max_queries,
                command_timeout=self.config.command_timeout,
            )

            # Update pool statistics
            self.connection_stats.pool_size = self.pool.get_size()
            self.connection_stats.pool_max = self.config.max_connections

            self._is_initialized = True

            logger.info(
                "PostgreSQL connection pool initialized successfully",
                extra={
                    "pool_size": self.connection_stats.pool_size,
                    "pool_max": self.connection_stats.pool_max,
                },
            )

        except asyncpg.PostgresError as e:
            logger.error(
                f"PostgreSQL connection pool initialization failed: {e}", exc_info=True
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Failed to initialize PostgreSQL connection pool: {e}",
                context={
                    "host": self.config.host,
                    "port": self.config.port,
                    "database": self.config.database,
                },
            ) from e
        except OSError as e:
            logger.error(f"Database connection failed: {e}", exc_info=True)
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Unexpected error initializing connection pool: {e}",
                context={
                    "host": self.config.host,
                    "port": self.config.port,
                    "database": self.config.database,
                },
            ) from e
        except Exception as e:
            logger.error(
                f"Unexpected error initializing connection pool: {e}", exc_info=True
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                message=f"Unexpected error initializing connection pool: {e}",
            ) from e

    @asynccontextmanager
    async def acquire_connection(self) -> AsyncIterator[Connection]:
        """
        Acquire a connection from the pool with automatic cleanup.

        Context manager that acquires a connection, sets schema search_path,
        and automatically releases the connection when done.

        Yields:
            asyncpg.Connection instance

        Raises:
            OnexError: If pool not initialized or connection acquisition fails

        Example:
            async with manager.acquire_connection() as conn:
                result = await conn.fetchrow("SELECT * FROM users WHERE id = $1", 1)
        """
        if not self._is_initialized or not self.pool:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="Connection pool not initialized - call initialize() first",
            )

        connection = None
        try:
            # Acquire connection from pool
            connection = await self.pool.acquire()
            self.connection_stats.checked_out += 1

            # Set schema search_path
            validated_schema = InputSanitizer.validate_sql_identifier(
                self.config.schema
            )
            await connection.execute(f"SET search_path TO {validated_schema}, public")

            # Update pool statistics
            self.connection_stats.pool_size = self.pool.get_size()
            self.connection_stats.pool_free = self.pool.get_idle_size()

            # Check for pool exhaustion at connection acquisition
            self._check_pool_exhaustion()

            yield connection

        except asyncpg.PostgresError as e:
            logger.error(f"Connection acquisition failed: {e}", exc_info=True)
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Failed to acquire database connection: {e}",
            ) from e
        finally:
            if connection:
                # Release connection back to pool
                await self.pool.release(connection)
                self.connection_stats.checked_in += 1

                # Update pool statistics
                if self.pool:
                    self.connection_stats.pool_free = self.pool.get_idle_size()

    @asynccontextmanager
    async def transaction(
        self,
        isolation: str = "read_committed",
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncIterator[Connection]:
        """
        Execute operations within a database transaction.

        Context manager that acquires a connection and starts a transaction
        with specified isolation level and characteristics.

        **Transaction Boundaries:**
            - **BEGIN**: Transaction starts automatically when entering context manager
            - **COMMIT**: Transaction commits automatically when exiting context normally
            - **ROLLBACK**: Transaction rolls back automatically on any exception

        **Transaction Lifecycle:**
            1. Context manager entered → Connection acquired → BEGIN TRANSACTION
            2. Operations executed within context
            3. Context manager exited successfully → COMMIT
            4. Exception raised within context → ROLLBACK + re-raise exception
            5. Connection released back to pool

        **Isolation Levels:**
            - read_uncommitted: Lowest isolation, allows dirty reads
            - read_committed: Default, prevents dirty reads
            - repeatable_read: Prevents non-repeatable reads
            - serializable: Highest isolation, prevents phantom reads

        **Use Cases:**
            - Multi-step operations requiring atomicity (all-or-nothing)
            - Financial transactions (debits/credits must succeed together)
            - Data consistency across related tables
            - Batch updates that must complete atomically

        Args:
            isolation: Transaction isolation level
                      (read_uncommitted, read_committed, repeatable_read, serializable)
            readonly: Whether transaction is read-only
            deferrable: Whether transaction can be deferred (for read-only serializable)

        Yields:
            asyncpg.Connection instance within transaction context

        Raises:
            OnexError: If transaction setup fails

        Example:
            # Successful transaction - commits automatically
            async with manager.transaction(isolation="serializable") as conn:
                await conn.execute("INSERT INTO users (name) VALUES ($1)", "Alice")
                await conn.execute("UPDATE accounts SET balance = balance + 100")
            # COMMIT executed here

            # Failed transaction - rolls back automatically
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO users (name) VALUES ($1)", "Bob")
                raise ValueError("Simulated error")  # ROLLBACK executed here
        """
        async with self.acquire_connection() as conn:
            try:
                async with conn.transaction(
                    isolation=isolation, readonly=readonly, deferrable=deferrable
                ):
                    yield conn
            except asyncpg.PostgresError as e:
                logger.error(f"Transaction failed: {e}", exc_info=True)
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                    message=f"Database transaction failed: {e}",
                    context={
                        "isolation": isolation,
                        "readonly": readonly,
                    },
                ) from e

    async def execute_query(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None,
        record_metrics: bool = True,
    ) -> list[Record] | str:
        """
        Execute a query with metrics collection.

        Automatically detects SELECT vs DML queries and uses appropriate
        execution method (fetch for SELECT, execute for DML).

        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Query timeout in seconds (overrides config)
            record_metrics: Whether to record execution metrics

        Returns:
            For SELECT queries: list of asyncpg.Record objects
            For DML queries: string with execution result (e.g., "INSERT 0 1")

        Raises:
            OnexError: If query execution fails

        Example:
            # SELECT query
            users = await manager.execute_query("SELECT * FROM users WHERE age > $1", 18)

            # INSERT query
            result = await manager.execute_query(
                "INSERT INTO users (name, age) VALUES ($1, $2)",
                "Alice", 25
            )
        """
        if not self._is_initialized or not self.pool:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="Connection pool not initialized - call initialize() first",
            )

        start_time = time.perf_counter()
        query_hash = self._compute_query_hash(query)
        query_type = self._detect_query_type(query)

        try:
            async with self.acquire_connection() as conn:
                # Execute query based on type
                if query_type == "SELECT":
                    result = await conn.fetch(query, *args, timeout=timeout)
                else:
                    result = await conn.execute(query, *args, timeout=timeout)

                # Record metrics
                if record_metrics:
                    execution_time_ms = (time.perf_counter() - start_time) * 1000
                    self._record_query_metrics(
                        query_hash, execution_time_ms, query_type
                    )

                return result

        except asyncpg.PostgresError as e:
            logger.error(
                f"Query execution failed: {e}",
                exc_info=True,
                extra={
                    "query_type": query_type,
                    "query_hash": query_hash,
                },
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Query execution failed: {e}",
                context={
                    "query_type": query_type,
                    "query_hash": query_hash,
                },
            ) from e

    def _compute_query_hash(self, query: str) -> str:
        """
        Compute hash of query for deduplication and metrics.

        Args:
            query: SQL query string

        Returns:
            12-character hash string
        """
        query_normalized = " ".join(query.split())  # Normalize whitespace
        hash_obj = hashlib.sha256(query_normalized.encode("utf-8"))
        return hash_obj.hexdigest()[:12]

    def _detect_query_type(self, query: str) -> str:
        """
        Detect query type from SQL statement.

        Args:
            query: SQL query string

        Returns:
            Query type: SELECT, INSERT, UPDATE, DELETE, WITH, or OTHER
        """
        query_upper = query.strip().upper()

        if query_upper.startswith("SELECT"):
            return "SELECT"
        elif query_upper.startswith("WITH"):
            return "SELECT"  # CTEs are typically SELECT queries
        elif query_upper.startswith("INSERT"):
            # Check if INSERT has RETURNING clause
            if "RETURNING" in query_upper:
                return "SELECT"  # Treat as SELECT to use fetch()
            return "INSERT"
        elif query_upper.startswith("UPDATE"):
            # Check if UPDATE has RETURNING clause
            if "RETURNING" in query_upper:
                return "SELECT"  # Treat as SELECT to use fetch()
            return "UPDATE"
        elif query_upper.startswith("DELETE"):
            # Check if DELETE has RETURNING clause
            if "RETURNING" in query_upper:
                return "SELECT"  # Treat as SELECT to use fetch()
            return "DELETE"
        else:
            return "OTHER"

    def _record_query_metrics(
        self, query_hash: str, execution_time_ms: float, query_type: str
    ) -> None:
        """
        Record query execution metrics with circular buffer to prevent memory leak.

        Args:
            query_hash: Query hash for deduplication
            execution_time_ms: Execution time in milliseconds
            query_type: Query type (SELECT, INSERT, etc.)
        """
        metric = QueryMetrics(
            query_hash=query_hash,
            execution_time_ms=execution_time_ms,
            query_type=query_type,
        )

        self._query_metrics.append(metric)

        # Keep only last N metrics (circular buffer)
        if len(self._query_metrics) > self._max_metrics_stored:
            self._query_metrics = self._query_metrics[-self._max_metrics_stored :]

        # Log slow queries
        if metric.performance_category == "slow":
            logger.warning(
                f"Slow query detected: {execution_time_ms:.2f}ms",
                extra={
                    "query_hash": query_hash,
                    "query_type": query_type,
                    "execution_time_ms": execution_time_ms,
                },
            )

    def _check_pool_exhaustion(self) -> None:
        """
        Check for pool exhaustion and log warnings when threshold exceeded.

        Monitors pool utilization and logs warnings when usage exceeds the configured
        threshold (default 90%). Implements rate limiting to prevent log spam.

        Pool Exhaustion Detection:
            - Calculates: utilization = used_connections / max_connections
            - Triggers warning when: utilization >= pool_exhaustion_threshold
            - Rate limited by: pool_exhaustion_log_interval (default 60s)

        Tracking Metrics:
            - _last_exhaustion_warning: Timestamp of last warning (prevents spam)
            - _exhaustion_warning_count: Total number of exhaustion events

        Performance:
            - Minimal overhead: Simple arithmetic calculation
            - No blocking operations
            - Rate limited logging prevents performance impact
        """
        if not self.pool:
            return

        # Calculate pool utilization
        pool_size = self.pool.get_size()
        free_connections = self.pool.get_idle_size()
        used_connections = pool_size - free_connections
        max_connections = self.config.max_connections

        if max_connections <= 0:
            return

        utilization = used_connections / max_connections

        # Check if threshold exceeded
        if utilization >= self.config.pool_exhaustion_threshold:
            current_time = time.time()

            # Check if enough time has passed since last warning (rate limiting)
            time_since_last_warning = current_time - self._last_exhaustion_warning

            if time_since_last_warning >= self.config.pool_exhaustion_log_interval:
                # Update tracking
                self._last_exhaustion_warning = current_time
                self._exhaustion_warning_count += 1

                # Log warning with structured context
                logger.warning(
                    f"Connection pool exhaustion detected: {utilization:.1%} utilization",
                    extra={
                        "event": "pool_exhaustion",
                        "utilization_percent": round(utilization * 100, 1),
                        "used_connections": used_connections,
                        "free_connections": free_connections,
                        "pool_size": pool_size,
                        "pool_max": max_connections,
                        "threshold_percent": round(
                            self.config.pool_exhaustion_threshold * 100, 1
                        ),
                        "exhaustion_count": self._exhaustion_warning_count,
                        "time_since_last_warning_seconds": round(
                            time_since_last_warning, 1
                        ),
                        "recommendation": "Consider increasing max_connections or investigating connection leaks",
                    },
                )

    def get_pool_stats(self) -> dict[str, Any]:
        """
        Get connection pool statistics with exhaustion monitoring.

        Retrieves current pool statistics and checks for pool exhaustion conditions.
        Monitors utilization and logs warnings when threshold is exceeded.

        Returns:
            Dictionary with pool statistics including:
            - Pool size, free connections, max connections
            - Checked out/in counts
            - Utilization percentage
            - Exhaustion warning count
            - Query metrics count

        Example:
            stats = manager.get_pool_stats()
            print(f"Pool size: {stats['pool_size']}/{stats['pool_max']}")
            print(f"Utilization: {stats['utilization_percent']:.1f}%")
            print(f"Free connections: {stats['pool_free']}")
        """
        if not self.pool:
            return {
                "initialized": False,
                "pool_size": 0,
                "pool_free": 0,
                "pool_max": 0,
                "utilization_percent": 0.0,
            }

        # Get pool statistics
        pool_size = self.pool.get_size()
        pool_free = self.pool.get_idle_size()
        used_connections = pool_size - pool_free
        max_connections = self.config.max_connections

        # Calculate utilization
        utilization_percent = (
            (used_connections / max_connections * 100) if max_connections > 0 else 0.0
        )

        # Check for pool exhaustion (with rate-limited warnings)
        self._check_pool_exhaustion()

        return {
            "initialized": self._is_initialized,
            "pool_size": pool_size,
            "pool_free": pool_free,
            "pool_max": max_connections,
            "used_connections": used_connections,
            "utilization_percent": round(utilization_percent, 1),
            "exhaustion_threshold_percent": round(
                self.config.pool_exhaustion_threshold * 100, 1
            ),
            "exhaustion_warning_count": self._exhaustion_warning_count,
            "checked_out": self.connection_stats.checked_out,
            "checked_in": self.connection_stats.checked_in,
            "query_metrics_count": len(self._query_metrics),
        }

    async def health_check(self) -> bool:
        """
        Perform health check on database connection.

        Executes a simple SELECT 1 query to verify database connectivity.

        Returns:
            True if database is healthy, False otherwise
        """
        if not self._is_initialized or not self.pool:
            logger.warning("Health check failed - connection pool not initialized")
            return False

        try:
            async with self.acquire_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                is_healthy = result == 1

                if is_healthy:
                    logger.debug("Database health check passed")
                else:
                    logger.warning(
                        f"Database health check failed - unexpected result: {result}"
                    )

                return is_healthy

        except Exception as e:
            logger.error(f"Database health check failed: {e}", exc_info=True)
            return False

    def get_query_metrics(
        self, limit: Optional[int] = None, query_type: Optional[str] = None
    ) -> list[QueryMetrics]:
        """
        Get query execution metrics.

        Args:
            limit: Maximum number of metrics to return (most recent)
            query_type: Filter by query type (SELECT, INSERT, etc.)

        Returns:
            List of QueryMetrics objects
        """
        metrics = self._query_metrics

        # Filter by query type
        if query_type:
            metrics = [m for m in metrics if m.query_type == query_type]

        # Apply limit
        if limit:
            metrics = metrics[-limit:]

        return metrics

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Get performance summary from query metrics.

        Returns:
            Dictionary with performance statistics including:
            - Total queries executed
            - Average execution time
            - Fast/slow query counts
            - Query type distribution
        """
        if not self._query_metrics:
            return {
                "total_queries": 0,
                "avg_execution_time_ms": 0.0,
                "fast_queries": 0,
                "slow_queries": 0,
                "query_types": {},
            }

        total_queries = len(self._query_metrics)
        total_time = sum(m.execution_time_ms for m in self._query_metrics)
        avg_time = total_time / total_queries if total_queries > 0 else 0.0

        fast_queries = sum(
            1 for m in self._query_metrics if m.performance_category == "fast"
        )
        slow_queries = sum(
            1 for m in self._query_metrics if m.performance_category == "slow"
        )

        # Count by query type
        query_types: dict[str, int] = {}
        for metric in self._query_metrics:
            query_types[metric.query_type] = query_types.get(metric.query_type, 0) + 1

        return {
            "total_queries": total_queries,
            "avg_execution_time_ms": round(avg_time, 2),
            "fast_queries": fast_queries,
            "slow_queries": slow_queries,
            "query_types": query_types,
        }

    async def close(self) -> None:
        """
        Close connection pool and cleanup resources.

        Safe to call multiple times.
        """
        if not self.pool:
            logger.debug("Connection pool already closed or never initialized")
            return

        try:
            logger.info(
                "Closing PostgreSQL connection pool",
                extra={
                    "pool_size": self.pool.get_size(),
                    "queries_executed": len(self._query_metrics),
                },
            )

            await self.pool.close()
            self._is_initialized = False

            logger.info("PostgreSQL connection pool closed successfully")

        except Exception as e:
            logger.error(f"Error closing connection pool: {e}", exc_info=True)
            # Don't raise - cleanup should be best-effort
        finally:
            self.pool = None

    @property
    def is_initialized(self) -> bool:
        """Check if connection pool is initialized."""
        return self._is_initialized

    @property
    def pool_size(self) -> int:
        """Get current pool size."""
        return self.pool.get_size() if self.pool else 0

    @property
    def pool_free(self) -> int:
        """Get number of free connections in pool."""
        return self.pool.get_idle_size() if self.pool else 0
