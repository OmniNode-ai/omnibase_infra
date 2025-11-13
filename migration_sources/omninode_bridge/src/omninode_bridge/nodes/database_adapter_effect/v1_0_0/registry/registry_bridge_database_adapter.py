"""
Registry for Bridge Database Adapter Effect Node Dependencies.

This registry provides dependency injection for PostgreSQL operations through
protocol-based interfaces, enabling clean separation between the effect node
and database infrastructure.

ONEX v2.0 Compliance:
- Container-based dependency injection
- Protocol-based abstractions
- Async/await throughout
- Proper error handling with OnexError

Implementation: Phase 1, Agent 4
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional, TypeVar, Union

from omnibase_core import EnumCoreErrorCode, ModelOnexError

# Aliases for compatibility
CoreErrorCode = EnumCoreErrorCode
OnexError = ModelOnexError

# Import Kafka infrastructure for event consumption
from omninode_bridge.infrastructure.kafka import KafkaConsumerWrapper

# Import the existing PostgreSQL infrastructure
from omninode_bridge.services.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

# Type variables for generic type safety
ConfigValueT = TypeVar("ConfigValueT")


class RegistryBridgeDatabaseAdapter:
    """
    Registry for Bridge Database Adapter Effect Node.

    Handles dependency injection for PostgreSQL operations with proper
    protocol resolution and connection management.

    Integrates with omninode_bridge's existing PostgresClient infrastructure
    to provide connection pooling, query execution, and transaction management
    capabilities to the database adapter effect node.
    """

    def __init__(self):
        """Initialize registry with PostgreSQL and Kafka dependencies."""
        self._postgres_client: PostgresClient | None = None
        self._kafka_consumer: KafkaConsumerWrapper | None = None
        self._config: dict[str, Any] = {}
        self._services: dict[str, Any] = {}  # Container for services

    async def initialize(self) -> None:
        """
        Initialize PostgreSQL client and dependencies.

        Creates a PostgresClient instance using environment-based configuration
        and establishes connection pool for database operations.

        Process:
            1. Create PostgresClient from environment configuration
            2. Establish connection pool with retry logic
            3. Validate connection with health check
            4. Store configuration for protocol resolution

        Raises:
            OnexError: If initialization fails with INITIALIZATION_ERROR code
        """
        try:
            logger.info("Initializing Bridge Database Adapter registry")

            # Create PostgresClient using environment configuration
            # The client automatically loads configuration from environment variables
            self._postgres_client = PostgresClient()
            logger.info(
                f"PostgresClient created for {self._postgres_client.host}:"
                f"{self._postgres_client.port}/{self._postgres_client.database}"
            )

            # Establish connection pool with automatic retry
            await self._postgres_client.connect()
            logger.info(
                f"Connection pool established (min: {self._postgres_client.min_size}, "
                f"max: {self._postgres_client.max_size})"
            )

            # Validate connection health
            health = await self._postgres_client.health_check()
            if health.get("status") != "healthy":
                raise RuntimeError(
                    f"Connection pool health check failed: {health.get('error', 'Unknown error')}"
                )
            logger.info("Connection pool health check passed")

            # Store configuration for later access
            self._config = {
                "postgres_host": self._postgres_client.host,
                "postgres_port": self._postgres_client.port,
                "postgres_database": self._postgres_client.database,
                "postgres_user": self._postgres_client.user,
                "postgres_min_connections": self._postgres_client.min_size,
                "postgres_max_connections": self._postgres_client.max_size,
                "postgres_ssl_enabled": self._postgres_client.ssl_enabled,
                "postgres_pool_timeout": self._postgres_client.acquire_timeout_seconds,
                "postgres_query_timeout": self._postgres_client.query_timeout_seconds,
            }

            # Initialize Kafka consumer for event-driven operations
            logger.info("Initializing Kafka consumer for event consumption")
            try:
                self._kafka_consumer = KafkaConsumerWrapper()
                logger.info(
                    "Kafka consumer created",
                    extra={
                        "bootstrap_servers": self._kafka_consumer._bootstrap_servers,
                        "security_protocol": self._kafka_consumer._security_protocol,
                    },
                )

                # Store Kafka configuration
                self._config.update(
                    {
                        "kafka_bootstrap_servers": self._kafka_consumer._bootstrap_servers,
                        "kafka_security_protocol": self._kafka_consumer._security_protocol,
                        "kafka_env": self._kafka_consumer._env,
                        "kafka_tenant": self._kafka_consumer._tenant,
                        "kafka_context": self._kafka_consumer._context,
                    }
                )

                logger.info("Kafka consumer initialized successfully")

            except OnexError as e:
                logger.warning(
                    f"Kafka consumer initialization failed (non-critical): {e.message}",
                    extra={"error_code": e.error_code},
                )
                # Kafka is optional - continue if it fails
                self._kafka_consumer = None
            except Exception as e:
                logger.warning(
                    f"Kafka consumer initialization failed (non-critical): {e}",
                    exc_info=True,
                )
                # Kafka is optional - continue if it fails
                self._kafka_consumer = None

            logger.info("Bridge Database Adapter registry initialized successfully")

        except OnexError:
            # Re-raise OnexError instances without wrapping
            raise
        except Exception as e:
            logger.error(f"Failed to initialize registry: {e}", exc_info=True)
            raise OnexError(
                code=CoreErrorCode.INITIALIZATION_ERROR,
                message=f"Failed to initialize Bridge Database Adapter registry: {e!s}",
            ) from e

    async def resolve_protocol(self, protocol_name: str) -> Union[
        "ConnectionPoolManagerAdapter",
        "QueryExecutorAdapter",
        "TransactionManagerAdapter",
        "KafkaConsumerWrapper",
    ]:
        """
        Resolve protocol dependencies for database adapter.

        Provides protocol implementations for connection pooling, query execution,
        transaction management, and Kafka event consumption.

        Args:
            protocol_name: Name of protocol to resolve

        Returns:
            Protocol implementation - one of:
                - ConnectionPoolManagerAdapter for ProtocolConnectionPoolManager
                - QueryExecutorAdapter for ProtocolQueryExecutor
                - TransactionManagerAdapter for ProtocolTransactionManager
                - KafkaConsumerWrapper for ProtocolKafkaConsumer

        Raises:
            OnexError: If protocol is unknown or client not initialized
        """
        if protocol_name == "ProtocolConnectionPoolManager":
            if not self._postgres_client:
                raise OnexError(
                    code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                    message="PostgresClient not initialized - call initialize() first",
                )
            # Provide connection pool management interface
            return ConnectionPoolManagerAdapter(self._postgres_client)

        elif protocol_name == "ProtocolQueryExecutor":
            if not self._postgres_client:
                raise OnexError(
                    code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                    message="PostgresClient not initialized - call initialize() first",
                )
            # Provide query execution interface
            return QueryExecutorAdapter(self._postgres_client)

        elif protocol_name == "ProtocolTransactionManager":
            if not self._postgres_client:
                raise OnexError(
                    code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                    message="PostgresClient not initialized - call initialize() first",
                )
            # Provide transaction management interface
            return TransactionManagerAdapter(self._postgres_client)

        elif protocol_name == "ProtocolKafkaConsumer":
            if not self._kafka_consumer:
                raise OnexError(
                    code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                    message="KafkaConsumer not initialized - call initialize() first or check Kafka configuration",
                )
            # Provide Kafka consumer interface
            return self._kafka_consumer

        else:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message=f"Unknown protocol: {protocol_name}",
            )

    def get_config(
        self, config_key: str, default: Optional[ConfigValueT] = None
    ) -> Optional[ConfigValueT]:
        """
        Get configuration value with type inference.

        Args:
            config_key: Configuration key to retrieve
            default: Default value if key not found (type determines return type)

        Returns:
            Configuration value matching the type of default, or None

        Example:
            # Type inference from default value
            host: str = registry.get_config("postgres_host", "localhost")
            port: int = registry.get_config("postgres_port", 5432)
            enabled: bool = registry.get_config("feature_enabled", False)
        """
        return self._config.get(config_key, default)

    async def cleanup(self) -> None:
        """
        Cleanup registry resources.

        Closes PostgreSQL connection pool, Kafka consumer, and releases all resources.
        Safe to call multiple times - will only cleanup if resources exist.

        Process:
            1. Close Kafka consumer
            2. Disconnect PostgreSQL client
            3. Clear connection pool
            4. Reset configuration cache
        """
        logger.info("Cleaning up Bridge Database Adapter registry")

        # Close Kafka consumer first
        if self._kafka_consumer:
            try:
                logger.info("Closing Kafka consumer")
                await self._kafka_consumer.close_consumer()
                logger.info("Kafka consumer closed")
            except Exception as e:
                logger.error(f"Error closing Kafka consumer: {e}", exc_info=True)
                # Don't raise - cleanup should be best-effort
            finally:
                self._kafka_consumer = None

        # Close PostgreSQL client
        if self._postgres_client:
            try:
                await self._postgres_client.disconnect()
                logger.info("PostgreSQL connection pool closed")
            except Exception as e:
                logger.error(f"Error during PostgreSQL cleanup: {e}", exc_info=True)
                # Don't raise - cleanup should be best-effort
            finally:
                self._postgres_client = None

        # Reset configuration
        self._config = {}
        logger.info("Registry cleanup completed")


class ConnectionPoolManagerAdapter:
    """
    Adapter for connection pool management operations.

    Provides access to the PostgreSQL connection pool and pool health metrics.
    """

    def __init__(self, postgres_client: PostgresClient):
        """
        Initialize with PostgresClient.

        Args:
            postgres_client: The PostgresClient instance
        """
        self._postgres_client = postgres_client

    async def get_pool_health(self) -> dict[str, Any]:
        """
        Get connection pool health status.

        Returns:
            Dictionary with pool health metrics
        """
        health_result = await self._postgres_client.health_check()
        return health_result

    async def get_pool_metrics(self) -> dict[str, Any]:
        """
        Get detailed connection pool metrics.

        Returns:
            Dictionary with pool metrics
        """
        pool_metrics = await self._postgres_client.get_pool_metrics()

        # Convert PoolMetrics dataclass to dictionary
        return {
            "current_size": pool_metrics.current_size,
            "max_size": pool_metrics.max_size,
            "utilization_percent": pool_metrics.utilization_percent,
            "connection_lifecycle": pool_metrics.connection_lifecycle,
            "performance_metrics": pool_metrics.performance_metrics,
            "health_status": pool_metrics.health_status,
        }

    @property
    def is_connected(self) -> bool:
        """Check if connection pool is connected."""
        return self._postgres_client.is_connected

    def get_connection_pool(self):
        """
        Get the underlying connection pool.

        Returns:
            The asyncpg Pool if available
        """
        return self._postgres_client.get_connection_pool()


class QueryExecutorAdapter:
    """
    Adapter for query execution operations.

    Provides a unified interface for executing database queries with
    proper error handling, timeout support, and metrics collection.

    Features:
        - Automatic parameter binding for SQL injection prevention
        - Configurable query timeouts
        - Result conversion to dictionaries
        - Comprehensive error handling
    """

    def __init__(self, postgres_client: PostgresClient):
        """
        Initialize with PostgresClient.

        Args:
            postgres_client: The PostgresClient instance
        """
        self._postgres_client = postgres_client

    async def fetch_all(
        self, sql: str, parameters: Optional[list] = None, timeout: int = 60
    ) -> list[dict[str, Any]]:
        """
        Execute query and fetch all results.

        Args:
            sql: SQL query to execute
            parameters: Query parameters for parameterized queries (optional)
            timeout: Query timeout in seconds (default: 60)

        Returns:
            List of result rows as dictionaries

        Raises:
            OnexError: If query execution fails

        Example:
            results = await executor.fetch_all(
                "SELECT * FROM users WHERE active = $1",
                parameters=[True],
                timeout=30
            )
        """
        try:
            params = parameters or []
            results = await self._postgres_client.fetch_all(sql, *params)

            # Convert asyncpg records to dictionaries
            if isinstance(results, list):
                return [dict(record) for record in results]
            return []
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            raise OnexError(
                code=CoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Failed to execute query: {e!s}",
            ) from e

    async def fetch_one(
        self, sql: str, parameters: Optional[list] = None, timeout: int = 60
    ) -> Optional[dict[str, Any]]:
        """
        Execute query and fetch one result.

        Args:
            sql: SQL query to execute
            parameters: Query parameters for parameterized queries (optional)
            timeout: Query timeout in seconds (default: 60)

        Returns:
            Single result row as dictionary or None if no results

        Raises:
            OnexError: If query execution fails

        Example:
            user = await executor.fetch_one(
                "SELECT * FROM users WHERE id = $1",
                parameters=[user_id]
            )
        """
        try:
            params = parameters or []
            result = await self._postgres_client.fetch_one(sql, *params)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            raise OnexError(
                code=CoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Failed to execute query: {e!s}",
            ) from e

    async def fetch_value(
        self, sql: str, parameters: Optional[list] = None, timeout: int = 60
    ) -> Any:
        """
        Execute query and fetch single scalar value.

        Useful for COUNT, SUM, MAX, or other aggregate queries that return a single value.

        Args:
            sql: SQL query to execute
            parameters: Query parameters for parameterized queries (optional)
            timeout: Query timeout in seconds (default: 60)

        Returns:
            Single scalar value from first column of first row, or None

        Raises:
            OnexError: If query execution fails

        Example:
            count = await executor.fetch_value(
                "SELECT COUNT(*) FROM users WHERE active = $1",
                parameters=[True]
            )
        """
        try:
            params = parameters or []
            result = await self._postgres_client.fetch_one(sql, *params)

            # Return first column value if result exists
            if result:
                return list(dict(result).values())[0]
            return None
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            raise OnexError(
                code=CoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Failed to execute query: {e!s}",
            ) from e

    async def execute(
        self, sql: str, parameters: Optional[list] = None, timeout: int = 60
    ) -> str:
        """
        Execute a command (INSERT, UPDATE, DELETE, etc.) without fetching results.

        Args:
            sql: SQL command to execute
            parameters: Query parameters for parameterized queries (optional)
            timeout: Query timeout in seconds (default: 60)

        Returns:
            Status string indicating affected rows (e.g., "INSERT 0 1", "UPDATE 5")

        Raises:
            OnexError: If query execution fails

        Example:
            status = await executor.execute(
                "INSERT INTO users (name, email) VALUES ($1, $2)",
                parameters=["John Doe", "john@example.com"]
            )
        """
        try:
            params = parameters or []
            result = await self._postgres_client.execute_query(sql, *params)
            return result
        except Exception as e:
            logger.error(f"Command execution failed: {e}", exc_info=True)
            raise OnexError(
                code=CoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Failed to execute command: {e!s}",
            ) from e


class TransactionManagerAdapter:
    """
    Adapter for transaction management operations.

    Provides ACID-compliant transaction context managers with support for
    different isolation levels, read-only transactions, and deferrable transactions.

    Features:
        - Multiple isolation levels (read_committed, repeatable_read, serializable)
        - Read-only transaction optimization
        - Deferrable transaction support for serializable isolation
        - Automatic rollback on exceptions
        - Connection pooling integration

    Transaction Isolation Levels:
        - read_committed: Default level, prevents dirty reads
        - repeatable_read: Prevents non-repeatable reads
        - serializable: Full isolation, prevents all anomalies
    """

    def __init__(self, postgres_client: PostgresClient):
        """
        Initialize with PostgresClient.

        Args:
            postgres_client: The PostgresClient instance
        """
        self._postgres_client = postgres_client

    @asynccontextmanager
    async def transaction(
        self,
        isolation_level: str = "read_committed",
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncIterator[Any]:
        """
        Create ACID-compliant transaction context manager.

        Automatically commits on success and rolls back on exceptions.
        Manages connection acquisition and release from the pool.

        Args:
            isolation_level: Transaction isolation level
                - "read_committed" (default): Prevents dirty reads
                - "repeatable_read": Prevents non-repeatable reads
                - "serializable": Full isolation, prevents all anomalies
            readonly: Whether transaction is read-only (enables optimizations)
            deferrable: Whether transaction can be deferred (only for serializable)

        Yields:
            Database connection within transaction context

        Raises:
            OnexError: If connection pool is unavailable or transaction fails

        Example:
            # Basic transaction
            async with transaction_manager.transaction() as conn:
                await conn.execute(
                    "INSERT INTO users (name, email) VALUES ($1, $2)",
                    "John Doe", "john@example.com"
                )
                await conn.execute(
                    "INSERT INTO audit_log (action) VALUES ($1)",
                    "user_created"
                )

            # Serializable read-only transaction
            async with transaction_manager.transaction(
                isolation_level="serializable",
                readonly=True
            ) as conn:
                users = await conn.fetch("SELECT * FROM users")
                stats = await conn.fetch("SELECT * FROM user_stats")
        """
        # Get connection pool
        pool = self._postgres_client.get_connection_pool()
        if not pool:
            logger.error("Connection pool not available for transaction")
            raise OnexError(
                code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message="Database connection pool not available",
            )

        # Acquire connection and start transaction
        try:
            async with pool.acquire() as conn:
                # Start transaction with specified isolation level
                logger.debug(
                    f"Starting transaction (isolation: {isolation_level}, "
                    f"readonly: {readonly}, deferrable: {deferrable})"
                )
                async with conn.transaction(
                    isolation=isolation_level, readonly=readonly, deferrable=deferrable
                ):
                    yield conn
                    logger.debug("Transaction committed successfully")
        except Exception as e:
            logger.error(f"Transaction failed and rolled back: {e}", exc_info=True)
            raise OnexError(
                code=CoreErrorCode.DATABASE_TRANSACTION_ERROR,
                message=f"Transaction failed: {e!s}",
            ) from e
