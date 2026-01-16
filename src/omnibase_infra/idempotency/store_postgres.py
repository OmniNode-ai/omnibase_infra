# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S608
# S608 disabled: All SQL f-strings use table_name which is validated via:
# 1. Pydantic regex pattern ^[a-zA-Z_][a-zA-Z0-9_]*$ in ModelPostgresIdempotencyStoreConfig
# 2. Runtime validation in _validate_table_name() (defense-in-depth)
# This defense-in-depth approach ensures only valid PostgreSQL identifiers are used,
# preventing SQL injection even if config validation is somehow bypassed.
"""PostgreSQL-based Idempotency Store Implementation.

This module provides a PostgreSQL-based implementation of the
ProtocolIdempotencyStore protocol for tracking processed messages
and preventing duplicate processing in distributed systems.

The store uses atomic INSERT ... ON CONFLICT DO NOTHING for thread-safe
idempotency checking and asyncpg for async database operations.

Table Schema:
    CREATE TABLE IF NOT EXISTS idempotency_records (
        id UUID PRIMARY KEY,
        domain VARCHAR(255),
        message_id UUID NOT NULL,
        correlation_id UUID,
        processed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        UNIQUE (domain, message_id)
    );
    CREATE INDEX IF NOT EXISTS idx_idempotency_processed_at ON idempotency_records(processed_at);
    CREATE INDEX IF NOT EXISTS idx_idempotency_domain ON idempotency_records(domain);
    CREATE INDEX IF NOT EXISTS idx_idempotency_correlation_id ON idempotency_records(correlation_id)
        WHERE correlation_id IS NOT NULL;

Clock Skew Considerations:
    In distributed systems, nodes may have slightly different system clocks.
    This can cause issues with TTL-based cleanup:

    Problem Scenario:
        - Node A processes message at 10:00:01 (its clock)
        - Node B's clock is 1 second behind (shows 10:00:00)
        - Cleanup runs on Node B using now() - it might delete records
          that Node A considers still valid

    Solution:
        The store applies a clock_skew_tolerance_seconds buffer (default: 60s)
        to all TTL calculations during cleanup. The effective TTL becomes:
        effective_ttl = ttl_seconds + clock_skew_tolerance_seconds

        This ensures records are only cleaned up after ALL nodes in the
        distributed system would consider them expired.

    Production Recommendations:
        1. Use NTP (Network Time Protocol) on all nodes to minimize clock drift
        2. Set clock_skew_tolerance_seconds to at least the maximum expected
           clock drift between nodes (default 60s is conservative for NTP-synced systems)
        3. Monitor NTP synchronization status across your infrastructure
        4. Consider higher tolerance values (e.g., 300s) for multi-datacenter deployments

Security Note:
    - DSN contains credentials - never log the raw value
    - Use parameterized queries to prevent SQL injection
    - Connection pool handles credential management
    - Table names are validated at both config and runtime level (defense-in-depth)
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.idempotency.models import (
    ModelIdempotencyStoreHealthCheckResult,
    ModelIdempotencyStoreMetrics,
    ModelPostgresIdempotencyStoreConfig,
)
from omnibase_infra.idempotency.protocol_idempotency_store import (
    ProtocolIdempotencyStore,
)
from omnibase_infra.utils.util_datetime import is_timezone_aware

logger = logging.getLogger(__name__)

# Regex pattern for valid PostgreSQL table names (defense-in-depth validation)
# Must start with letter or underscore, followed by letters, digits, or underscores
_TABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class StoreIdempotencyPostgres(ProtocolIdempotencyStore):
    """PostgreSQL-based idempotency store using asyncpg connection pool.

    This implementation provides exactly-once semantics by using PostgreSQL's
    INSERT ... ON CONFLICT DO NOTHING pattern for atomic check-and-record
    operations.

    Features:
        - Atomic check_and_record using INSERT ON CONFLICT
        - Connection pooling via asyncpg
        - TTL-based cleanup for expired records
        - Composite key (domain, message_id) for domain-isolated deduplication
        - Full correlation ID support for distributed tracing
        - Defense-in-depth table name validation for SQL injection prevention

    Security:
        Table names are validated at two levels:
        1. Pydantic config model: regex pattern on table_name field
        2. Runtime validation: _validate_table_name() in __init__

        This defense-in-depth approach ensures SQL injection prevention even
        if the config validation is somehow bypassed (e.g., through direct
        attribute assignment or deserialization from untrusted sources).

    Concurrency Safety:
        This store is coroutine-safe for asyncio concurrent access. The
        underlying asyncpg pool handles connection management and concurrent
        coroutine access safely. All metrics updates are protected by
        ``_metrics_lock`` (asyncio.Lock) to ensure atomic read-modify-write
        operations for observability counters.

        Note: This is not thread-safe. For multi-threaded access, additional
        synchronization would be required (e.g., threading.Lock or
        thread-safe connection pooling).

    Example:
        >>> from uuid import uuid4
        >>> config = ModelPostgresIdempotencyStoreConfig(
        ...     dsn="postgresql://user:pass@localhost:5432/mydb",
        ...     table_name="idempotency_records",
        ... )
        >>> store = StoreIdempotencyPostgres(config)
        >>> await store.initialize()
        >>> try:
        ...     is_new = await store.check_and_record(
        ...         message_id=uuid4(),
        ...         domain="registration",
        ...     )
        ...     if is_new:
        ...         print("Processing message...")
        ... finally:
        ...     await store.shutdown()
    """

    def __init__(self, config: ModelPostgresIdempotencyStoreConfig) -> None:
        """Initialize the PostgreSQL idempotency store.

        Args:
            config: Configuration model containing DSN, pool settings, and TTL options.

        Raises:
            ProtocolConfigurationError: If table_name contains invalid characters
                (defense-in-depth validation for SQL injection prevention).
        """
        self._config = config
        self._pool: asyncpg.Pool | None = None
        self._initialized: bool = False
        self._metrics = ModelIdempotencyStoreMetrics()
        self._metrics_lock = asyncio.Lock()

        # Defense-in-depth: Validate table name at runtime even though
        # Pydantic config already validates it. This protects against:
        # - Direct attribute assignment bypassing Pydantic validation
        # - Deserialization from untrusted sources
        # - Future code changes that might bypass config validation
        self._validate_table_name(config.table_name)

    @property
    def is_initialized(self) -> bool:
        """Return True if the store has been initialized."""
        return self._initialized

    async def get_metrics(self) -> ModelIdempotencyStoreMetrics:
        """Get current store metrics for observability.

        Returns a copy of the current metrics to prevent external mutation.
        Metrics include:
            - total_checks: Total check_and_record calls
            - duplicate_count: Number of duplicates detected
            - error_count: Number of failed checks
            - duplicate_rate: Ratio of duplicates to total checks
            - error_rate: Ratio of errors to total checks
            - total_cleanup_deleted: Total records cleaned up
            - last_cleanup_deleted: Records deleted in last cleanup
            - last_cleanup_at: Timestamp of last cleanup

        Concurrency Safety:
            This method acquires ``_metrics_lock`` (asyncio.Lock) to return
            a consistent snapshot. Safe for concurrent coroutine access.

        Returns:
            Copy of current metrics.
        """
        async with self._metrics_lock:
            return self._metrics.model_copy()

    def _validate_table_name(self, table_name: str) -> None:
        """Validate table name for SQL injection prevention (defense-in-depth).

        This method provides runtime validation of the table name pattern,
        complementing the Pydantic field validation in the config model.
        Together they form a defense-in-depth approach to prevent SQL injection.

        Args:
            table_name: The table name to validate.

        Raises:
            ProtocolConfigurationError: If table_name doesn't match the
                expected pattern ^[a-zA-Z_][a-zA-Z0-9_]*$
        """
        if not _TABLE_NAME_PATTERN.match(table_name):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="validate_table_name",
                target_name="postgres_idempotency_store",
            )
            raise ProtocolConfigurationError(
                f"Invalid table name: {table_name}. "
                "Must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$ "
                "(letters, digits, underscores only, must start with letter or underscore)",
                context=context,
                parameter="table_name",
                value=table_name,
            )

    async def initialize(self) -> None:
        """Initialize the connection pool and ensure table exists.

        Creates the asyncpg connection pool and verifies (or creates)
        the idempotency_records table with proper schema.

        Raises:
            InfraConnectionError: If database connection fails.
            RuntimeHostError: If pool creation or table setup fails.
        """
        if self._initialized:
            return

        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="initialize",
            target_name="postgres_idempotency_store",
            correlation_id=correlation_id,
        )

        try:
            self._pool = await asyncpg.create_pool(
                dsn=self._config.dsn,
                min_size=self._config.pool_min_size,
                max_size=self._config.pool_max_size,
                command_timeout=self._config.command_timeout,
            )

            # Ensure table exists with proper schema
            await self._ensure_table_exists()

            self._initialized = True
            logger.info(
                "StoreIdempotencyPostgres initialized",
                extra={
                    "table_name": self._config.table_name,
                    "pool_min_size": self._config.pool_min_size,
                    "pool_max_size": self._config.pool_max_size,
                },
            )
        except asyncpg.InvalidPasswordError as e:
            # Clean up pool if it was created before table setup failed
            if self._pool is not None:
                await self._pool.close()
                self._pool = None
            raise InfraConnectionError(
                "Database authentication failed - check credentials",
                context=context,
            ) from e
        except asyncpg.InvalidCatalogNameError as e:
            # Clean up pool if it was created before table setup failed
            if self._pool is not None:
                await self._pool.close()
                self._pool = None
            raise InfraConnectionError(
                "Database not found - check database name",
                context=context,
            ) from e
        except OSError as e:
            # Clean up pool if it was created before table setup failed
            if self._pool is not None:
                await self._pool.close()
                self._pool = None
            raise InfraConnectionError(
                "Failed to connect to database - check host and port",
                context=context,
            ) from e
        except Exception as e:
            # Clean up pool if it was created before table setup failed
            if self._pool is not None:
                await self._pool.close()
                self._pool = None
            raise RuntimeHostError(
                f"Failed to initialize idempotency store: {type(e).__name__}",
                context=context,
            ) from e

    async def _ensure_table_exists(self) -> None:
        """Create the idempotency table if it doesn't exist.

        Creates the table with:
            - UUID primary key
            - Composite unique constraint on (domain, message_id)
            - Index on processed_at for efficient TTL cleanup
            - Index on domain for efficient is_processed queries
            - Partial index on correlation_id for distributed tracing queries
        """
        if self._pool is None:
            raise RuntimeHostError(
                "Pool not initialized - call initialize() first",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="_ensure_table_exists",
                    target_name="postgres_idempotency_store",
                ),
            )

        # Note: Table name is validated in config (alphanumeric + underscore only)
        # so safe to use in SQL. We still use parameterized queries for data values.
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self._config.table_name} (
                id UUID PRIMARY KEY,
                domain VARCHAR(255),
                message_id UUID NOT NULL,
                correlation_id UUID,
                processed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                UNIQUE (domain, message_id)
            )
        """

        create_processed_at_index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{self._config.table_name}_processed_at
            ON {self._config.table_name}(processed_at)
        """

        # Index on domain for efficient is_processed queries filtering by domain
        create_domain_index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{self._config.table_name}_domain
            ON {self._config.table_name}(domain)
        """

        # Partial index on correlation_id for distributed tracing queries
        # Uses partial index (WHERE NOT NULL) to save space since correlation_id is optional
        create_correlation_index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{self._config.table_name}_correlation_id
            ON {self._config.table_name}(correlation_id)
            WHERE correlation_id IS NOT NULL
        """

        async with self._pool.acquire() as conn:
            await conn.execute(create_table_sql)
            await conn.execute(create_processed_at_index_sql)
            await conn.execute(create_domain_index_sql)
            await conn.execute(create_correlation_index_sql)

    async def shutdown(self) -> None:
        """Close the connection pool and release resources."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False
        logger.info("StoreIdempotencyPostgres shutdown complete")

    async def check_and_record(
        self,
        message_id: UUID,
        domain: str | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Atomically check if message was processed and record if not.

        Uses INSERT ... ON CONFLICT DO NOTHING for atomic operation:
        - If insert succeeds, message is new -> return True
        - If insert conflicts, message is duplicate -> return False

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace for isolated deduplication.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            True if message is new (should be processed).
            False if message is duplicate (should be skipped).

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            RuntimeHostError: If store is not initialized.
        """
        op_correlation_id = correlation_id or uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="check_and_record",
            target_name="postgres_idempotency_store",
            correlation_id=op_correlation_id,
        )

        if not self._initialized or self._pool is None:
            raise RuntimeHostError(
                "Store not initialized - call initialize() first",
                context=context,
            )

        record_id = uuid4()
        processed_at = datetime.now(UTC)

        # INSERT ... ON CONFLICT DO NOTHING returns affected row count
        # 1 = insert succeeded (new message), 0 = conflict (duplicate)
        # table_name is validated via regex in ModelPostgresIdempotencyStoreConfig
        insert_sql = f"""  # noqa: S608
            INSERT INTO {self._config.table_name}
                (id, domain, message_id, correlation_id, processed_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (domain, message_id) DO NOTHING
        """

        is_new: bool = False  # Initialize before try block for finally access
        metrics_updated: bool = False  # Track if exception handler updated metrics
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    insert_sql,
                    record_id,
                    domain,
                    message_id,
                    correlation_id,
                    processed_at,
                )
                # asyncpg returns "INSERT 0 1" for success, "INSERT 0 0" for conflict
                is_new = str(result).endswith(" 1")

                if is_new:
                    logger.debug(
                        "Recorded new message",
                        extra={
                            "message_id": str(message_id),
                            "domain": domain,
                            "correlation_id": str(correlation_id)
                            if correlation_id
                            else None,
                        },
                    )
                else:
                    logger.debug(
                        "Duplicate message detected",
                        extra={
                            "message_id": str(message_id),
                            "domain": domain,
                        },
                    )

                return is_new

        except asyncpg.QueryCanceledError as e:
            async with self._metrics_lock:
                self._metrics.total_checks += 1
                self._metrics.error_count += 1
            metrics_updated = True
            raise InfraTimeoutError(
                f"Check and record timed out after {self._config.command_timeout}s",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=self._config.command_timeout,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._metrics_lock:
                self._metrics.total_checks += 1
                self._metrics.error_count += 1
            metrics_updated = True
            raise InfraConnectionError(
                "Database connection lost during check_and_record",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._metrics_lock:
                self._metrics.total_checks += 1
                self._metrics.error_count += 1
            metrics_updated = True
            raise RuntimeHostError(
                f"Database error during check_and_record: {type(e).__name__}",
                context=context,
            ) from e
        finally:
            # Always update metrics for success path, even if logging fails.
            # Exception handlers above set metrics_updated=True before re-raising,
            # so we only update here for the success path (no exception caught).
            if not metrics_updated:
                async with self._metrics_lock:
                    self._metrics.total_checks += 1
                    if not is_new:
                        self._metrics.duplicate_count += 1

    async def is_processed(
        self,
        message_id: UUID,
        domain: str | None = None,
    ) -> bool:
        """Check if a message was already processed (read-only).

        This is a read-only query that does not modify the store.

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace.

        Returns:
            True if the message has been processed.
            False if the message has not been processed or has expired.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If query times out.
            RuntimeHostError: If store is not initialized.
        """
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="is_processed",
            target_name="postgres_idempotency_store",
        )

        if not self._initialized or self._pool is None:
            raise RuntimeHostError(
                "Store not initialized - call initialize() first",
                context=context,
            )

        # table_name is validated via regex in ModelPostgresIdempotencyStoreConfig
        query_sql = f"""  # noqa: S608
            SELECT 1 FROM {self._config.table_name}
            WHERE domain IS NOT DISTINCT FROM $1 AND message_id = $2
            LIMIT 1
        """

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query_sql, domain, message_id)
                return row is not None

        except asyncpg.QueryCanceledError as e:
            raise InfraTimeoutError(
                f"Processed check query timed out after {self._config.command_timeout}s",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=self._config.command_timeout,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                "Database connection lost during processed check query",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            raise RuntimeHostError(
                f"Database error during is_processed: {type(e).__name__}",
                context=context,
            ) from e

    async def mark_processed(
        self,
        message_id: UUID,
        domain: str | None = None,
        correlation_id: UUID | None = None,
        processed_at: datetime | None = None,
    ) -> None:
        """Mark a message as processed (upsert).

        Records a message as processed. If the record already exists,
        updates the processed_at timestamp.

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace for isolated deduplication.
            correlation_id: Optional correlation ID for tracing.
            processed_at: Optional timestamp. If None, uses datetime.now(timezone.utc).
                Must be timezone-aware.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            RuntimeHostError: If store is not initialized or if processed_at
                is a naive (timezone-unaware) datetime.
        """
        op_correlation_id = correlation_id or uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="mark_processed",
            target_name="postgres_idempotency_store",
            correlation_id=op_correlation_id,
        )

        if not self._initialized or self._pool is None:
            raise RuntimeHostError(
                "Store not initialized - call initialize() first",
                context=context,
            )

        # Validate timezone awareness - fail fast on naive datetime
        # Note: This guards against external callers passing naive datetimes.
        # Our internal datetime.now(UTC) is always timezone-aware.
        if processed_at is not None and not is_timezone_aware(processed_at):
            raise RuntimeHostError(
                "processed_at must be timezone-aware (got naive datetime)",
                context=context,
            )

        effective_processed_at = processed_at or datetime.now(UTC)
        record_id = uuid4()

        # Use ON CONFLICT ... DO UPDATE to ensure idempotent upsert
        # table_name is validated via regex in ModelPostgresIdempotencyStoreConfig
        upsert_sql = f"""  # noqa: S608
            INSERT INTO {self._config.table_name}
                (id, domain, message_id, correlation_id, processed_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (domain, message_id) DO UPDATE
            SET processed_at = EXCLUDED.processed_at,
                correlation_id = COALESCE(EXCLUDED.correlation_id, {self._config.table_name}.correlation_id)
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    upsert_sql,
                    record_id,
                    domain,
                    message_id,
                    correlation_id,
                    effective_processed_at,
                )
                logger.debug(
                    "Marked message as processed",
                    extra={
                        "message_id": str(message_id),
                        "domain": domain,
                        "correlation_id": str(correlation_id)
                        if correlation_id
                        else None,
                    },
                )

        except asyncpg.QueryCanceledError as e:
            raise InfraTimeoutError(
                f"Mark processed timed out after {self._config.command_timeout}s",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=self._config.command_timeout,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                "Database connection lost during mark processed",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            raise RuntimeHostError(
                f"Database error during mark_processed: {type(e).__name__}",
                context=context,
            ) from e

    async def cleanup_expired(
        self,
        ttl_seconds: int,
        batch_size: int | None = None,
        max_iterations: int | None = None,
    ) -> int:
        """Remove entries older than TTL using batched deletion with clock skew tolerance.

        Cleans up old idempotency records based on processed_at timestamp.
        Uses batched deletion to reduce lock contention on high-volume tables.

        Batched Deletion Benefits:
            - Reduces lock contention by breaking large deletes into smaller
              transactions
            - Prevents long-running transactions that can block other operations
            - Allows other database operations to interleave between batches
            - Limits transaction log growth by committing in smaller chunks

        Clock Skew Handling:
            In distributed systems, nodes may have slightly different system clocks.
            To prevent premature deletion of records that some nodes may still
            consider valid, we add a clock_skew_tolerance_seconds buffer to the TTL.

            Example scenario without tolerance:
                - Node A processes message at 10:00:01 (its clock)
                - Node B checks for duplicate at 10:00:00 (its clock is 1 second behind)
                - If cleanup runs on Node B using now(), it might delete records
                  that Node A just created (from Node B's perspective, they're from
                  the "future")

            With tolerance, effective_ttl = ttl_seconds + clock_skew_tolerance_seconds,
            ensuring records are only cleaned up after ALL nodes would consider them
            expired.

        Args:
            ttl_seconds: Time-to-live in seconds. Records older than
                (ttl_seconds + clock_skew_tolerance_seconds) are removed.
            batch_size: Number of records to delete per batch. Defaults to
                config.cleanup_batch_size (10000). Use larger values for faster
                cleanup at the cost of longer locks, smaller values for better
                concurrency.
            max_iterations: Maximum number of batch iterations. Defaults to
                config.cleanup_max_iterations (100). Prevents runaway cleanup
                loops. Total max records = batch_size * max_iterations.

        Returns:
            Total number of entries removed across all batches.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If cleanup times out.
            RuntimeHostError: If store is not initialized.

        Example:
            >>> # Standard cleanup using config defaults
            >>> removed = await store.cleanup_expired(ttl_seconds=86400)
            >>> print(f"Removed {removed} expired records")

            >>> # High-concurrency system: smaller batches
            >>> removed = await store.cleanup_expired(
            ...     ttl_seconds=86400,
            ...     batch_size=1000,
            ... )

            >>> # Bulk cleanup with large batches
            >>> removed = await store.cleanup_expired(
            ...     ttl_seconds=86400,
            ...     batch_size=50000,
            ...     max_iterations=10,
            ... )
        """
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="cleanup_expired",
            target_name="postgres_idempotency_store",
        )

        if not self._initialized or self._pool is None:
            raise RuntimeHostError(
                "Store not initialized - call initialize() first",
                context=context,
            )

        # Use config defaults if not specified
        effective_batch_size = batch_size or self._config.cleanup_batch_size
        effective_max_iterations = max_iterations or self._config.cleanup_max_iterations

        # Apply clock skew tolerance to prevent premature deletion
        # effective_ttl = ttl_seconds + clock_skew_tolerance
        effective_ttl = ttl_seconds + self._config.clock_skew_tolerance_seconds

        # Batched delete using subquery with LIMIT
        # This pattern:
        # 1. Selects up to batch_size expired record IDs
        # 2. Deletes only those specific records
        # 3. Repeats until no more records are found or max_iterations reached
        #
        # table_name is validated via regex in ModelPostgresIdempotencyStoreConfig
        delete_batch_sql = f"""  # noqa: S608
            DELETE FROM {self._config.table_name}
            WHERE id IN (
                SELECT id FROM {self._config.table_name}
                WHERE processed_at < now() - interval '1 second' * $1
                LIMIT $2
            )
        """

        total_removed = 0
        iteration = 0

        try:
            while iteration < effective_max_iterations:
                iteration += 1

                async with self._pool.acquire() as conn:
                    result = await conn.execute(
                        delete_batch_sql, effective_ttl, effective_batch_size
                    )
                    # Parse "DELETE N" to get count
                    batch_removed = int(result.split()[-1]) if result else 0

                    total_removed += batch_removed

                    logger.debug(
                        "Cleanup batch completed",
                        extra={
                            "batch_removed": batch_removed,
                            "total_removed": total_removed,
                            "iteration": iteration,
                            "batch_size": effective_batch_size,
                        },
                    )

                    # Log progress every 10 batches for visibility during large cleanups
                    if iteration % 10 == 0:
                        logger.info(
                            "Cleanup progress",
                            extra={
                                "total_removed": total_removed,
                                "iteration": iteration,
                                "batch_size": effective_batch_size,
                            },
                        )

                    # If we deleted fewer than batch_size, we're done
                    if batch_removed < effective_batch_size:
                        break

            # Update cleanup metrics (protected by lock for thread safety)
            async with self._metrics_lock:
                self._metrics.total_cleanup_deleted += total_removed
                self._metrics.last_cleanup_deleted = total_removed
                self._metrics.last_cleanup_at = datetime.now(UTC)

            logger.info(
                "Cleaned up expired idempotency records",
                extra={
                    "total_removed": total_removed,
                    "ttl_seconds": ttl_seconds,
                    "clock_skew_tolerance_seconds": self._config.clock_skew_tolerance_seconds,
                    "effective_ttl_seconds": effective_ttl,
                    "table_name": self._config.table_name,
                    "iterations": iteration,
                    "batch_size": effective_batch_size,
                },
            )

            return total_removed

        except asyncpg.QueryCanceledError as e:
            raise InfraTimeoutError(
                f"Cleanup timed out after {self._config.command_timeout}s",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=self._config.command_timeout,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                "Database connection lost during cleanup",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            raise RuntimeHostError(
                f"Database error during cleanup: {type(e).__name__}",
                context=context,
            ) from e

    async def health_check(self) -> ModelIdempotencyStoreHealthCheckResult:
        """Check if the store is healthy and can accept operations.

        Performs read verification and table existence check to ensure
        the database is operational without writing data.

        Returns:
            ModelIdempotencyStoreHealthCheckResult with health status and diagnostics:
            - healthy: bool - True if store is healthy
            - reason: str - "ok", "not_initialized", "table_not_found", or "check_failed"
            - error_type: str | None - Exception type if check failed
        """
        if not self._initialized or self._pool is None:
            return ModelIdempotencyStoreHealthCheckResult(
                healthy=False, reason="not_initialized"
            )

        try:
            async with self._pool.acquire() as conn:
                # Step 1: Verify read access
                await conn.fetchval("SELECT 1")

                # Step 2: Verify table exists and is accessible
                # This confirms schema setup without writes
                check_table_sql = """
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = $1
                    LIMIT 1
                """
                table_exists = await conn.fetchval(
                    check_table_sql, self._config.table_name
                )

                if table_exists is None:
                    return ModelIdempotencyStoreHealthCheckResult(
                        healthy=False, reason="table_not_found"
                    )
                return ModelIdempotencyStoreHealthCheckResult(healthy=True, reason="ok")

        except Exception as e:
            return ModelIdempotencyStoreHealthCheckResult(
                healthy=False,
                reason="check_failed",
                error_type=type(e).__name__,
            )


__all__: list[str] = ["StoreIdempotencyPostgres"]
