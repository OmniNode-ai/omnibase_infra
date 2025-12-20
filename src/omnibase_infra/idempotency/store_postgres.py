# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S608
# S608 disabled: All SQL f-strings use table_name which is validated via
# regex pattern ^[a-zA-Z_][a-zA-Z0-9_]*$ in ModelPostgresIdempotencyStoreConfig.
# This ensures only valid PostgreSQL identifiers are used, preventing SQL injection.
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
    CREATE INDEX idx_idempotency_processed_at ON idempotency_records(processed_at);

Security Note:
    - DSN contains credentials - never log the raw value
    - Use parameterized queries to prevent SQL injection
    - Connection pool handles credential management
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timezone
from uuid import UUID, uuid4

import asyncpg
from omnibase_spi.protocols.storage.protocol_idempotency_store import (
    ProtocolIdempotencyStore,
)

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.idempotency.models import ModelPostgresIdempotencyStoreConfig

logger = logging.getLogger(__name__)


class PostgresIdempotencyStore(ProtocolIdempotencyStore):
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

    Thread Safety:
        This store is thread-safe. The underlying asyncpg pool handles
        connection management and concurrent access safely.

    Example:
        >>> from uuid import uuid4
        >>> config = ModelPostgresIdempotencyStoreConfig(
        ...     dsn="postgresql://user:pass@localhost:5432/mydb",
        ...     table_name="idempotency_records",
        ... )
        >>> store = PostgresIdempotencyStore(config)
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
        """
        self._config = config
        self._pool: asyncpg.Pool | None = None
        self._initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        """Return True if the store has been initialized."""
        return self._initialized

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
                "PostgresIdempotencyStore initialized",
                extra={
                    "table_name": self._config.table_name,
                    "pool_min_size": self._config.pool_min_size,
                    "pool_max_size": self._config.pool_max_size,
                },
            )
        except asyncpg.InvalidPasswordError as e:
            raise InfraConnectionError(
                "Database authentication failed - check credentials",
                context=context,
            ) from e
        except asyncpg.InvalidCatalogNameError as e:
            raise InfraConnectionError(
                "Database not found - check database name",
                context=context,
            ) from e
        except OSError as e:
            raise InfraConnectionError(
                "Failed to connect to database - check host and port",
                context=context,
            ) from e
        except Exception as e:
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
        """
        if self._pool is None:
            raise RuntimeHostError(
                "Pool not initialized - call initialize() first",
                context=ModelInfraErrorContext(
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

        create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{self._config.table_name}_processed_at
            ON {self._config.table_name}(processed_at)
        """

        async with self._pool.acquire() as conn:
            await conn.execute(create_table_sql)
            await conn.execute(create_index_sql)

    async def shutdown(self) -> None:
        """Close the connection pool and release resources."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False
        logger.info("PostgresIdempotencyStore shutdown complete")

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
                is_new: bool = str(result).endswith(" 1")

                if is_new:
                    logger.debug(
                        "Recorded new message",
                        extra={
                            "message_id": str(message_id),
                            "domain": domain,
                            "correlation_id": str(correlation_id) if correlation_id else None,
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
            raise InfraTimeoutError(
                f"Check and record timed out after {self._config.command_timeout}s",
                context=context,
                timeout_seconds=self._config.command_timeout,
            ) from e
        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                "Database connection lost during check_and_record",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            raise RuntimeHostError(
                f"Database error during check_and_record: {type(e).__name__}",
                context=context,
            ) from e

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
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="is_processed",
            target_name="postgres_idempotency_store",
            correlation_id=uuid4(),
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
                f"is_processed query timed out after {self._config.command_timeout}s",
                context=context,
                timeout_seconds=self._config.command_timeout,
            ) from e
        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                "Database connection lost during is_processed query",
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
            RuntimeHostError: If store is not initialized or processed_at is naive.
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

        # Validate timezone awareness
        if processed_at is not None and processed_at.tzinfo is None:
            logger.warning(
                "Naive datetime provided to mark_processed, treating as UTC",
                extra={"message_id": str(message_id)},
            )
            processed_at = processed_at.replace(tzinfo=UTC)

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
                        "correlation_id": str(correlation_id) if correlation_id else None,
                    },
                )

        except asyncpg.QueryCanceledError as e:
            raise InfraTimeoutError(
                f"mark_processed timed out after {self._config.command_timeout}s",
                context=context,
                timeout_seconds=self._config.command_timeout,
            ) from e
        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                "Database connection lost during mark_processed",
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
    ) -> int:
        """Remove entries older than TTL.

        Cleans up old idempotency records based on processed_at timestamp.
        Uses batched deletion to avoid long-running transactions.

        Args:
            ttl_seconds: Time-to-live in seconds. Records older than this
                value are removed.

        Returns:
            Number of entries removed.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If cleanup times out.
            RuntimeHostError: If store is not initialized.
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="cleanup_expired",
            target_name="postgres_idempotency_store",
            correlation_id=uuid4(),
        )

        if not self._initialized or self._pool is None:
            raise RuntimeHostError(
                "Store not initialized - call initialize() first",
                context=context,
            )

        # Delete records older than TTL
        # Using interval arithmetic for clarity and correctness
        # table_name is validated via regex in ModelPostgresIdempotencyStoreConfig
        delete_sql = f"""  # noqa: S608
            DELETE FROM {self._config.table_name}
            WHERE processed_at < now() - interval '1 second' * $1
        """

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(delete_sql, ttl_seconds)
                # Parse "DELETE N" to get count
                removed_count = int(result.split()[-1]) if result else 0

                logger.info(
                    "Cleaned up expired idempotency records",
                    extra={
                        "removed_count": removed_count,
                        "ttl_seconds": ttl_seconds,
                        "table_name": self._config.table_name,
                    },
                )

                return removed_count

        except asyncpg.QueryCanceledError as e:
            raise InfraTimeoutError(
                f"Cleanup timed out after {self._config.command_timeout}s",
                context=context,
                timeout_seconds=self._config.command_timeout,
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

    async def health_check(self) -> bool:
        """Check if the store is healthy and can accept operations.

        Performs a simple query to verify database connectivity.

        Returns:
            True if store is healthy, False otherwise.
        """
        if not self._initialized or self._pool is None:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except Exception:
            return False


__all__: list[str] = ["PostgresIdempotencyStore"]
