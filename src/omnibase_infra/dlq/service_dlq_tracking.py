# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S608
# S608 disabled: All SQL f-strings use storage_table which is validated via:
# 1. Pydantic regex pattern (PATTERN_TABLE_NAME) in ModelDlqTrackingConfig
# 2. Runtime validation in _validate_storage_table() (defense-in-depth)
# Both use the shared PATTERN_TABLE_NAME constant from constants_dlq.py.
# This defense-in-depth approach ensures only valid PostgreSQL identifiers are used,
# preventing SQL injection even if config validation is somehow bypassed.
"""DLQ Replay Tracking Service.

This module provides a PostgreSQL-based service for tracking DLQ replay
operations, enabling persistent history of replay attempts.

The service uses asyncpg for async database operations and provides
methods for recording replay attempts and querying replay history.

Table Schema:
    CREATE TABLE IF NOT EXISTS dlq_replay_history (
        id UUID PRIMARY KEY,
        original_message_id UUID NOT NULL,
        replay_correlation_id UUID NOT NULL,
        original_topic VARCHAR(255) NOT NULL,
        target_topic VARCHAR(255) NOT NULL,
        replay_status VARCHAR(50) NOT NULL,
        replay_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        success BOOLEAN NOT NULL,
        error_message TEXT,
        dlq_offset BIGINT NOT NULL,
        dlq_partition INTEGER NOT NULL,
        retry_count INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_dlq_replay_message_id ON dlq_replay_history(original_message_id);
    CREATE INDEX IF NOT EXISTS idx_dlq_replay_timestamp ON dlq_replay_history(replay_timestamp);

Security Note:
    - DSN contains credentials - never log the raw value
    - Use parameterized queries to prevent SQL injection
    - Connection pool handles credential management
    - Table names are validated at both config and runtime level (defense-in-depth)

Related:
    - scripts/dlq_replay.py - CLI tool that uses this service
    - OMN-1032 - PostgreSQL tracking integration ticket
"""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.dlq.constants_dlq import PATTERN_TABLE_NAME, REGEX_TABLE_NAME
from omnibase_infra.dlq.models import (
    EnumReplayStatus,
    ModelDlqReplayRecord,
    ModelDlqTrackingConfig,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

logger = logging.getLogger(__name__)


class ServiceDlqTracking(MixinAsyncCircuitBreaker):
    """PostgreSQL-based service for tracking DLQ replay operations.

    This service provides persistent storage for DLQ replay history,
    enabling operators to track which messages have been replayed,
    when, and with what outcome.

    Features:
        - Async PostgreSQL operations via asyncpg
        - Connection pooling for efficient database access
        - Replay history tracking with correlation IDs
        - Query methods for replay history analysis
        - Defense-in-depth table name validation for SQL injection prevention
        - Circuit breaker pattern for fault tolerance (via MixinAsyncCircuitBreaker)

    Circuit Breaker Pattern (Production-Grade):
        - Uses MixinAsyncCircuitBreaker for consistent circuit breaker implementation
        - Prevents cascading failures to PostgreSQL service
        - Three states: CLOSED (normal), OPEN (blocking), HALF_OPEN (testing)
        - Configurable failure_threshold (default: 5 consecutive failures)
        - Configurable reset_timeout (default: 60 seconds)
        - Raises InfraUnavailableError when circuit is OPEN

    Security:
        Table names are validated at two levels:
        1. Pydantic config model: regex pattern on storage_table field
        2. Runtime validation: _validate_storage_table() in __init__

        This defense-in-depth approach ensures SQL injection prevention even
        if the config validation is somehow bypassed (e.g., through direct
        attribute assignment or deserialization from untrusted sources).

    Thread Safety:
        This service is thread-safe. The underlying asyncpg pool handles
        connection management and concurrent access safely. Circuit breaker
        operations are protected by async locks (via MixinAsyncCircuitBreaker).

    Example:
        >>> from uuid import uuid4
        >>> from datetime import datetime, timezone
        >>> config = ModelDlqTrackingConfig(
        ...     dsn="postgresql://user:pass@localhost:5432/mydb",
        ...     storage_table="dlq_replay_history",
        ... )
        >>> service = ServiceDlqTracking(config)
        >>> await service.initialize()
        >>> try:
        ...     record = ModelDlqReplayRecord(
        ...         id=uuid4(),
        ...         original_message_id=uuid4(),
        ...         replay_correlation_id=uuid4(),
        ...         original_topic="dev.orders.command.v1",
        ...         target_topic="dev.orders.command.v1",
        ...         replay_status=EnumReplayStatus.COMPLETED,
        ...         replay_timestamp=datetime.now(timezone.utc),
        ...         success=True,
        ...         dlq_offset=12345,
        ...         dlq_partition=0,
        ...         retry_count=1,
        ...     )
        ...     await service.record_replay_attempt(record)
        ... finally:
        ...     await service.shutdown()
    """

    def __init__(self, config: ModelDlqTrackingConfig) -> None:
        """Initialize the DLQ tracking service.

        Args:
            config: Configuration model containing DSN and pool settings.

        Raises:
            ProtocolConfigurationError: If storage_table contains invalid characters
                (defense-in-depth validation for SQL injection prevention).
        """
        self._config = config
        self._pool: asyncpg.Pool | None = None
        self._initialized: bool = False

        # Initialize circuit breaker for PostgreSQL fault tolerance
        # Uses MixinAsyncCircuitBreaker for consistent implementation across infra services
        cb_config = ModelCircuitBreakerConfig.from_env(
            service_name="dlq_tracking_service",
            transport_type=EnumInfraTransportType.DATABASE,
        )
        self._init_circuit_breaker_from_config(cb_config)

        # Defense-in-depth: Validate table name at runtime even though
        # Pydantic config already validates it. This protects against:
        # - Direct attribute assignment bypassing Pydantic validation
        # - Deserialization from untrusted sources
        # - Future code changes that might bypass config validation
        self._validate_storage_table(config.storage_table)

    @property
    def is_initialized(self) -> bool:
        """Return True if the service has been initialized."""
        return self._initialized

    @property
    def is_tracking_enabled(self) -> bool:
        """Return True if the service is initialized and ready to track replays.

        This is an alias for is_initialized that provides clearer semantics
        when the service is used specifically for replay tracking.

        Returns:
            True if tracking is available, False otherwise.
        """
        return self._initialized and self._pool is not None

    def _validate_storage_table(self, storage_table: str) -> None:
        """Validate storage table name for SQL injection prevention (defense-in-depth).

        This method provides runtime validation of the storage table pattern,
        complementing the Pydantic field validation in the config model.
        Together they form a defense-in-depth approach to prevent SQL injection.

        Both validations use the shared PATTERN_TABLE_NAME constant from
        constants_dlq.py to ensure consistency. See that module for details
        on why both validation layers are intentional and required.

        Args:
            storage_table: The storage table name to validate.

        Raises:
            ProtocolConfigurationError: If storage_table doesn't match the
                expected pattern (PATTERN_TABLE_NAME constant).
        """
        if not REGEX_TABLE_NAME.match(storage_table):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="validate_storage_table",
                target_name="dlq_tracking_service",
            )
            raise ProtocolConfigurationError(
                f"Invalid storage table: {storage_table}. "
                f"Must match pattern {PATTERN_TABLE_NAME} "
                "(letters, digits, underscores only, must start with letter or underscore)",
                context=context,
                parameter="storage_table",
                value=storage_table,
            )

    async def initialize(self) -> None:
        """Initialize the connection pool and ensure table exists.

        Creates the asyncpg connection pool and verifies (or creates)
        the dlq_replay_history table with proper schema.

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
            target_name="dlq_tracking_service",
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
                "ServiceDlqTracking initialized",
                extra={
                    "table_name": self._config.storage_table,
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
                f"Failed to initialize DLQ tracking service: {type(e).__name__}",
                context=context,
            ) from e
        finally:
            # Cleanup pool if initialization failed
            if not self._initialized and self._pool is not None:
                logger.debug("Cleaning up connection pool after initialization failure")
                await self._pool.close()
                self._pool = None

    async def _ensure_table_exists(self) -> None:
        """Create the DLQ replay history table if it doesn't exist.

        Creates the table with:
            - UUID primary key
            - Indexes on original_message_id and replay_timestamp
            - All required columns for replay tracking
        """
        if self._pool is None:
            raise RuntimeHostError(
                "Pool not initialized - call initialize() first",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="_ensure_table_exists",
                    target_name="dlq_tracking_service",
                ),
            )

        # Note: Table name is validated in config (alphanumeric + underscore only)
        # so safe to use in SQL. We still use parameterized queries for data values.
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self._config.storage_table} (
                id UUID PRIMARY KEY,
                original_message_id UUID NOT NULL,
                replay_correlation_id UUID NOT NULL,
                original_topic VARCHAR(255) NOT NULL,
                target_topic VARCHAR(255) NOT NULL,
                replay_status VARCHAR(50) NOT NULL,
                replay_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                success BOOLEAN NOT NULL,
                error_message TEXT,
                dlq_offset BIGINT NOT NULL,
                dlq_partition INTEGER NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0
            )
        """

        create_message_id_index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{self._config.storage_table}_message_id
            ON {self._config.storage_table}(original_message_id)
        """

        create_timestamp_index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{self._config.storage_table}_timestamp
            ON {self._config.storage_table}(replay_timestamp)
        """

        async with self._pool.acquire() as conn:
            # Wrap DDL in transaction for atomicity - if index creation fails,
            # the table creation will be rolled back to avoid partial schema state
            async with conn.transaction():
                await conn.execute(create_table_sql)
                await conn.execute(create_message_id_index_sql)
                await conn.execute(create_timestamp_index_sql)

    async def shutdown(self) -> None:
        """Close the connection pool and release resources."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False
        logger.info("ServiceDlqTracking shutdown complete")

    async def record_replay_attempt(self, record: ModelDlqReplayRecord) -> None:
        """Record a DLQ replay attempt.

        Inserts a new replay record into the database. Each replay attempt
        gets its own record, enabling full audit trail of replay operations.

        Circuit breaker integration:
            - Checks circuit state before execution (raises InfraUnavailableError if OPEN)
            - Records success/failure for circuit state management
            - Allows test request in HALF_OPEN state

        Args:
            record: The replay record to persist.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is OPEN.
            RuntimeHostError: If service is not initialized.
        """
        correlation_id = record.replay_correlation_id
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="record_replay_attempt",
            target_name="dlq_tracking_service",
            correlation_id=correlation_id,
        )

        if not self._initialized or self._pool is None:
            raise RuntimeHostError(
                "Service not initialized - call initialize() first",
                context=context,
            )

        # Circuit breaker check (caller-held lock pattern per ONEX circuit breaker pattern)
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("record_replay_attempt", correlation_id)

        # storage_table is validated via regex in ModelDlqTrackingConfig
        insert_sql = f"""
            INSERT INTO {self._config.storage_table}
                (id, original_message_id, replay_correlation_id, original_topic,
                 target_topic, replay_status, replay_timestamp, success,
                 error_message, dlq_offset, dlq_partition, retry_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    insert_sql,
                    record.id,
                    record.original_message_id,
                    record.replay_correlation_id,
                    record.original_topic,
                    record.target_topic,
                    record.replay_status.value,
                    record.replay_timestamp,
                    record.success,
                    record.error_message,
                    record.dlq_offset,
                    record.dlq_partition,
                    record.retry_count,
                )
                logger.debug(
                    "Recorded replay attempt",
                    extra={
                        "record_id": str(record.id),
                        "original_message_id": str(record.original_message_id),
                        "replay_status": record.replay_status.value,
                    },
                )

            # Circuit breaker success (caller-held lock pattern per ONEX circuit breaker pattern)
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

        except asyncpg.QueryCanceledError as e:
            # Circuit breaker failure (caller-held lock pattern per ONEX circuit breaker pattern)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "record_replay_attempt", correlation_id
                )
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="record_replay_attempt",
                target_name="dlq_tracking_service",
                correlation_id=correlation_id,
                timeout_seconds=self._config.command_timeout,
            )
            raise InfraTimeoutError(
                f"Record replay attempt timed out after {self._config.command_timeout}s",
                context=timeout_ctx,
            ) from e
        except asyncpg.PostgresConnectionError as e:
            # Circuit breaker failure (caller-held lock pattern per ONEX circuit breaker pattern)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "record_replay_attempt", correlation_id
                )
            raise InfraConnectionError(
                "Database connection lost during record_replay_attempt",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            # Circuit breaker failure (caller-held lock pattern per ONEX circuit breaker pattern)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "record_replay_attempt", correlation_id
                )
            raise RuntimeHostError(
                f"Database error during record_replay_attempt: {type(e).__name__}",
                context=context,
            ) from e

    async def get_replay_history(self, message_id: UUID) -> list[ModelDlqReplayRecord]:
        """Get replay history for a specific message.

        Retrieves all replay attempts for a given original message ID,
        ordered by replay timestamp (most recent first).

        Circuit breaker integration:
            - Checks circuit state before execution (raises InfraUnavailableError if OPEN)
            - Records success/failure for circuit state management
            - Allows test request in HALF_OPEN state

        Args:
            message_id: The original message correlation ID to query.

        Returns:
            List of replay records for the message, ordered by timestamp desc.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If query times out.
            InfraUnavailableError: If circuit breaker is OPEN.
            RuntimeHostError: If service is not initialized.
        """
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_replay_history",
            target_name="dlq_tracking_service",
            correlation_id=correlation_id,
        )

        if not self._initialized or self._pool is None:
            raise RuntimeHostError(
                "Service not initialized - call initialize() first",
                context=context,
            )

        # Circuit breaker check (caller-held lock pattern per ONEX circuit breaker pattern)
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("get_replay_history", correlation_id)

        # storage_table is validated via regex in ModelDlqTrackingConfig
        query_sql = f"""
            SELECT id, original_message_id, replay_correlation_id, original_topic,
                   target_topic, replay_status, replay_timestamp, success,
                   error_message, dlq_offset, dlq_partition, retry_count
            FROM {self._config.storage_table}
            WHERE original_message_id = $1
            ORDER BY replay_timestamp DESC
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql, message_id)

                records = []
                for row in rows:
                    records.append(
                        ModelDlqReplayRecord(
                            id=row["id"],
                            original_message_id=row["original_message_id"],
                            replay_correlation_id=row["replay_correlation_id"],
                            original_topic=row["original_topic"],
                            target_topic=row["target_topic"],
                            replay_status=EnumReplayStatus(row["replay_status"]),
                            replay_timestamp=row["replay_timestamp"],
                            success=row["success"],
                            error_message=row["error_message"],
                            dlq_offset=row["dlq_offset"],
                            dlq_partition=row["dlq_partition"],
                            retry_count=row["retry_count"],
                        )
                    )

            # Circuit breaker success (caller-held lock pattern per ONEX circuit breaker pattern)
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return records

        except asyncpg.QueryCanceledError as e:
            # Circuit breaker failure (caller-held lock pattern per ONEX circuit breaker pattern)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_replay_history", correlation_id)
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="get_replay_history",
                target_name="dlq_tracking_service",
                correlation_id=correlation_id,
                timeout_seconds=self._config.command_timeout,
            )
            raise InfraTimeoutError(
                f"Get replay history timed out after {self._config.command_timeout}s",
                context=timeout_ctx,
            ) from e
        except asyncpg.PostgresConnectionError as e:
            # Circuit breaker failure (caller-held lock pattern per ONEX circuit breaker pattern)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_replay_history", correlation_id)
            raise InfraConnectionError(
                "Database connection lost during get_replay_history",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            # Circuit breaker failure (caller-held lock pattern per ONEX circuit breaker pattern)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_replay_history", correlation_id)
            raise RuntimeHostError(
                f"Database error during get_replay_history: {type(e).__name__}",
                context=context,
            ) from e

    async def health_check(self) -> bool:
        """Check if the service is healthy and can accept operations.

        Performs a simple query to verify database connectivity
        and table existence.

        Returns:
            True if the service is healthy, False otherwise.
        """
        if not self._initialized or self._pool is None:
            return False

        try:
            async with self._pool.acquire() as conn:
                # Verify read access
                await conn.fetchval("SELECT 1")

                # Verify table exists and is accessible
                check_table_sql = """
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = $1
                    LIMIT 1
                """
                table_exists = await conn.fetchval(
                    check_table_sql, self._config.storage_table
                )

                return table_exists is not None

        except Exception:
            logger.debug("Health check failed", exc_info=True)
            return False


__all__: list[str] = ["ServiceDlqTracking"]
