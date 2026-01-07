# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Registration Storage Handler.

This module provides a PostgreSQL-backed implementation of the registration
storage handler protocol, wrapping existing PostgreSQL functionality with
circuit breaker resilience.

Connection Pooling:
    - Uses asyncpg connection pool for efficient database access
    - Configurable pool size (default: 10)
    - Pool gracefully closed on handler shutdown

Circuit Breaker:
    - Uses MixinAsyncCircuitBreaker for consistent resilience
    - Three states: CLOSED (normal), OPEN (blocking), HALF_OPEN (testing)
    - Configurable failure threshold and reset timeout
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.enums.enum_node_kind import EnumNodeKind

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
)
from omnibase_infra.handlers.registration_storage.models import (
    ModelRegistrationRecord,
    ModelStorageResult,
    ModelUpsertResult,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.nodes.effects.protocol_postgres_adapter import (
        ProtocolPostgresAdapter,
    )

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
DEFAULT_CIRCUIT_BREAKER_RESET_TIMEOUT = 30.0
DEFAULT_POOL_SIZE = 10
DEFAULT_TIMEOUT_SECONDS = 30.0

# SQL statements
SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS node_registrations (
    node_id UUID PRIMARY KEY,
    node_type VARCHAR(64) NOT NULL,
    node_version VARCHAR(32) NOT NULL,
    endpoints JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

SQL_UPSERT = """
INSERT INTO node_registrations (node_id, node_type, node_version, endpoints, metadata, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (node_id) DO UPDATE SET
    node_type = EXCLUDED.node_type,
    node_version = EXCLUDED.node_version,
    endpoints = EXCLUDED.endpoints,
    metadata = EXCLUDED.metadata,
    updated_at = EXCLUDED.updated_at
RETURNING (xmax = 0) AS was_insert;
"""

SQL_QUERY_BASE = """
SELECT node_id, node_type, node_version, endpoints, metadata, created_at, updated_at
FROM node_registrations
"""

SQL_QUERY_COUNT = """
SELECT COUNT(*) FROM node_registrations
"""

SQL_UPDATE = """
UPDATE node_registrations SET
    endpoints = COALESCE($2, endpoints),
    metadata = COALESCE($3, metadata),
    updated_at = NOW()
WHERE node_id = $1
RETURNING node_id;
"""

SQL_DELETE = """
DELETE FROM node_registrations WHERE node_id = $1 RETURNING node_id;
"""


class PostgresRegistrationStorageHandler(MixinAsyncCircuitBreaker):
    """PostgreSQL implementation of ProtocolRegistrationStorageHandler.

    Wraps existing PostgreSQL adapter functionality with circuit breaker
    resilience and proper error handling.

    Thread Safety:
        This handler is coroutine-safe. All database operations use
        asyncpg's connection pool, and circuit breaker state is protected
        by asyncio.Lock.

    Attributes:
        handler_type: Returns "postgresql" identifier.

    Example:
        >>> handler = PostgresRegistrationStorageHandler(
        ...     postgres_adapter=postgres_adapter,
        ...     circuit_breaker_config={"threshold": 5, "reset_timeout": 30.0},
        ... )
        >>> result = await handler.store_registration(record)
    """

    def __init__(
        self,
        postgres_adapter: ProtocolPostgresAdapter | None = None,
        dsn: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "omninode_bridge",
        user: str = "postgres",
        password: str | None = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        circuit_breaker_config: dict[str, object] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize PostgresRegistrationStorageHandler.

        Args:
            postgres_adapter: Optional existing PostgreSQL adapter (ProtocolPostgresAdapter).
                If not provided, a new asyncpg connection pool will be created.
            dsn: Optional PostgreSQL connection DSN (overrides host/port/etc).
            host: PostgreSQL server hostname (default: "localhost").
            port: PostgreSQL server port (default: 5432).
            database: Database name (default: "omninode_bridge").
            user: Database user (default: "postgres").
            password: Optional database password.
            pool_size: Connection pool size (default: 10).
            circuit_breaker_config: Optional circuit breaker configuration with:
                - threshold: Max failures before opening (default: 5)
                - reset_timeout: Seconds before reset (default: 30.0)
                - service_name: Service identifier (default: "postgres.storage")
            timeout_seconds: Operation timeout in seconds (default: 30.0).
        """
        config = circuit_breaker_config or {}
        _threshold_raw = config.get("threshold", DEFAULT_CIRCUIT_BREAKER_THRESHOLD)
        threshold = (
            int(_threshold_raw)
            if isinstance(_threshold_raw, (int, float, str))
            else DEFAULT_CIRCUIT_BREAKER_THRESHOLD
        )
        _reset_timeout_raw = config.get(
            "reset_timeout", DEFAULT_CIRCUIT_BREAKER_RESET_TIMEOUT
        )
        reset_timeout = (
            float(_reset_timeout_raw)
            if isinstance(_reset_timeout_raw, (int, float, str))
            else DEFAULT_CIRCUIT_BREAKER_RESET_TIMEOUT
        )
        _service_name_raw = config.get("service_name", "postgres.storage")
        service_name = (
            str(_service_name_raw)
            if _service_name_raw is not None
            else "postgres.storage"
        )

        self._init_circuit_breaker(
            threshold=threshold,
            reset_timeout=reset_timeout,
            service_name=service_name,
            transport_type=EnumInfraTransportType.DATABASE,
        )

        # Store configuration
        self._dsn = dsn
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._pool_size = pool_size
        self._timeout_seconds = timeout_seconds

        # Connection pool (initialized on first use)
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()
        self._initialized = False

        # External adapter (if provided)
        self._postgres_adapter = postgres_adapter

        logger.info(
            "PostgresRegistrationStorageHandler created",
            extra={
                "host": host,
                "port": port,
                "database": database,
                "pool_size": pool_size,
            },
        )

    @property
    def handler_type(self) -> str:
        """Return the handler type identifier.

        Returns:
            "postgresql" identifier string.
        """
        return "postgresql"

    async def _ensure_pool(self) -> asyncpg.Pool:
        """Ensure connection pool is initialized.

        Returns:
            The asyncpg connection pool.

        Raises:
            InfraConnectionError: If pool creation fails.
        """
        if self._pool is not None:
            return self._pool

        async with self._pool_lock:
            # Double-check after acquiring lock
            if self._pool is not None:
                return self._pool

            try:
                import asyncpg

                if self._dsn:
                    self._pool = await asyncpg.create_pool(
                        dsn=self._dsn,
                        min_size=1,
                        max_size=self._pool_size,
                    )
                else:
                    self._pool = await asyncpg.create_pool(
                        host=self._host,
                        port=self._port,
                        database=self._database,
                        user=self._user,
                        password=self._password,
                        min_size=1,
                        max_size=self._pool_size,
                    )

                # Create table if not exists
                async with self._pool.acquire() as conn:
                    await conn.execute(SQL_CREATE_TABLE)

                self._initialized = True

                logger.info(
                    "PostgreSQL connection pool initialized",
                    extra={
                        "host": self._host,
                        "port": self._port,
                        "database": self._database,
                    },
                )

                return self._pool

            except Exception as e:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="initialize_pool",
                    target_name="postgres.storage",
                )
                raise InfraConnectionError(
                    f"Failed to initialize PostgreSQL pool: {type(e).__name__}",
                    context=context,
                ) from e

    async def store_registration(
        self,
        record: ModelRegistrationRecord,
        correlation_id: UUID | None = None,
    ) -> ModelUpsertResult:
        """Store a registration record in PostgreSQL.

        Args:
            record: Registration record to store.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelUpsertResult with upsert outcome.

        Raises:
            InfraConnectionError: If connection to PostgreSQL fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="store_registration",
                correlation_id=correlation_id,
            )

        try:
            pool = await self._ensure_pool()

            now = datetime.now(UTC)
            endpoints_json = json.dumps(record.endpoints)
            metadata_json = json.dumps(record.metadata)

            async with pool.acquire() as conn:
                result = await asyncio.wait_for(
                    conn.fetchrow(
                        SQL_UPSERT,
                        record.node_id,
                        record.node_type.value,
                        record.node_version,
                        endpoints_json,
                        metadata_json,
                        record.created_at or now,
                        now,
                    ),
                    timeout=self._timeout_seconds,
                )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            was_insert = result["was_insert"] if result else False
            duration_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "Registration stored in PostgreSQL",
                extra={
                    "node_id": str(record.node_id),
                    "was_insert": was_insert,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelUpsertResult(
                success=True,
                node_id=record.node_id,
                was_insert=was_insert,
                duration_ms=duration_ms,
                backend_type=self.handler_type,
                correlation_id=correlation_id,
            )

        except TimeoutError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="store_registration",
                    correlation_id=correlation_id,
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="store_registration",
                target_name="postgres.storage",
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"PostgreSQL upsert timed out after {self._timeout_seconds}s",
                context=context,
                timeout_seconds=self._timeout_seconds,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="store_registration",
                    correlation_id=correlation_id,
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="store_registration",
                target_name="postgres.storage",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"PostgreSQL upsert failed: {type(e).__name__}",
                context=context,
            ) from e

    async def query_registrations(
        self,
        node_type: EnumNodeKind | None = None,
        node_version: str | None = None,
        limit: int = 100,
        offset: int = 0,
        correlation_id: UUID | None = None,
    ) -> ModelStorageResult:
        """Query registration records from PostgreSQL.

        Args:
            node_type: Optional node type to filter by.
            node_version: Optional version pattern to filter by.
            limit: Maximum number of records to return.
            offset: Number of records to skip (for pagination).
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelStorageResult with list of matching records.

        Raises:
            InfraConnectionError: If connection to PostgreSQL fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="query_registrations",
                correlation_id=correlation_id,
            )

        try:
            pool = await self._ensure_pool()

            # Build query with filters
            conditions: list[str] = []
            params: list[object] = []
            param_idx = 1

            if node_type is not None:
                conditions.append(f"node_type = ${param_idx}")
                params.append(node_type.value)
                param_idx += 1

            if node_version is not None:
                conditions.append(f"node_version LIKE ${param_idx}")
                params.append(f"{node_version}%")
                param_idx += 1

            where_clause = ""
            if conditions:
                where_clause = " WHERE " + " AND ".join(conditions)

            # Query for records
            query = f"{SQL_QUERY_BASE}{where_clause} ORDER BY updated_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            params.extend([limit, offset])

            # Query for total count
            count_query = f"{SQL_QUERY_COUNT}{where_clause}"
            count_params = params[:-2]  # Exclude limit and offset

            async with pool.acquire() as conn:
                rows, count_result = await asyncio.gather(
                    asyncio.wait_for(
                        conn.fetch(query, *params),
                        timeout=self._timeout_seconds,
                    ),
                    asyncio.wait_for(
                        conn.fetchval(count_query, *count_params),
                        timeout=self._timeout_seconds,
                    ),
                )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # Convert rows to records
            records: list[ModelRegistrationRecord] = []
            for row in rows:
                endpoints = json.loads(row["endpoints"]) if row["endpoints"] else {}
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}

                records.append(
                    ModelRegistrationRecord(
                        node_id=row["node_id"],
                        node_type=EnumNodeKind(row["node_type"]),
                        node_version=row["node_version"],
                        endpoints=endpoints,
                        metadata=metadata,
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        correlation_id=correlation_id,
                    )
                )

            duration_ms = (time.monotonic() - start_time) * 1000
            total_count = count_result or 0

            logger.info(
                "Registration query completed",
                extra={
                    "record_count": len(records),
                    "total_count": total_count,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelStorageResult(
                success=True,
                records=tuple(records),
                total_count=total_count,
                duration_ms=duration_ms,
                backend_type=self.handler_type,
                correlation_id=correlation_id,
            )

        except TimeoutError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="query_registrations",
                    correlation_id=correlation_id,
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="query_registrations",
                target_name="postgres.storage",
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"PostgreSQL query timed out after {self._timeout_seconds}s",
                context=context,
                timeout_seconds=self._timeout_seconds,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="query_registrations",
                    correlation_id=correlation_id,
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="query_registrations",
                target_name="postgres.storage",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"PostgreSQL query failed: {type(e).__name__}",
                context=context,
            ) from e

    async def update_registration(
        self,
        node_id: UUID,
        endpoints: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelUpsertResult:
        """Update an existing registration record.

        Args:
            node_id: ID of the node to update.
            endpoints: Optional new endpoints dict.
            metadata: Optional new metadata dict.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelUpsertResult with update outcome.

        Raises:
            InfraConnectionError: If connection to PostgreSQL fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="update_registration",
                correlation_id=correlation_id,
            )

        try:
            pool = await self._ensure_pool()

            endpoints_json = json.dumps(endpoints) if endpoints else None
            metadata_json = json.dumps(metadata) if metadata else None

            async with pool.acquire() as conn:
                result = await asyncio.wait_for(
                    conn.fetchval(SQL_UPDATE, node_id, endpoints_json, metadata_json),
                    timeout=self._timeout_seconds,
                )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            success = result is not None
            duration_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "Registration updated",
                extra={
                    "node_id": str(node_id),
                    "success": success,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelUpsertResult(
                success=success,
                node_id=node_id,
                was_insert=False,
                error="Record not found" if not success else None,
                duration_ms=duration_ms,
                backend_type=self.handler_type,
                correlation_id=correlation_id,
            )

        except TimeoutError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="update_registration",
                    correlation_id=correlation_id,
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="update_registration",
                target_name="postgres.storage",
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"PostgreSQL update timed out after {self._timeout_seconds}s",
                context=context,
                timeout_seconds=self._timeout_seconds,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="update_registration",
                    correlation_id=correlation_id,
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="update_registration",
                target_name="postgres.storage",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"PostgreSQL update failed: {type(e).__name__}",
                context=context,
            ) from e

    async def delete_registration(
        self,
        node_id: UUID,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Delete a registration record from PostgreSQL.

        Args:
            node_id: ID of the node to delete.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            True if record was deleted, False if not found.

        Raises:
            InfraConnectionError: If connection to PostgreSQL fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="delete_registration",
                correlation_id=correlation_id,
            )

        try:
            pool = await self._ensure_pool()

            async with pool.acquire() as conn:
                result = await asyncio.wait_for(
                    conn.fetchval(SQL_DELETE, node_id),
                    timeout=self._timeout_seconds,
                )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            deleted = result is not None
            duration_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "Registration deletion completed",
                extra={
                    "node_id": str(node_id),
                    "deleted": deleted,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return deleted

        except TimeoutError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="delete_registration",
                    correlation_id=correlation_id,
                )
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="delete_registration",
                target_name="postgres.storage",
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"PostgreSQL delete timed out after {self._timeout_seconds}s",
                context=context,
                timeout_seconds=self._timeout_seconds,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="delete_registration",
                    correlation_id=correlation_id,
                )
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="delete_registration",
                target_name="postgres.storage",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"PostgreSQL delete failed: {type(e).__name__}",
                context=context,
            ) from e

    async def health_check(
        self,
        correlation_id: UUID | None = None,
    ) -> dict[str, object]:
        """Perform a health check on the PostgreSQL connection.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Dict with health status information.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        try:
            pool = await self._ensure_pool()

            async with pool.acquire() as conn:
                result = await asyncio.wait_for(
                    conn.fetchval("SELECT 1"),
                    timeout=5.0,  # Short timeout for health check
                )

            duration_ms = (time.monotonic() - start_time) * 1000

            return {
                "healthy": True,
                "backend_type": self.handler_type,
                "host": self._host,
                "port": self._port,
                "database": self._database,
                "pool_size": self._pool_size,
                "circuit_breaker_open": self._circuit_breaker_open,
                "duration_ms": duration_ms,
                "correlation_id": str(correlation_id),
            }

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            return {
                "healthy": False,
                "backend_type": self.handler_type,
                "host": self._host,
                "port": self._port,
                "database": self._database,
                "error": f"Health check failed: {type(e).__name__}",
                "circuit_breaker_open": self._circuit_breaker_open,
                "duration_ms": duration_ms,
                "correlation_id": str(correlation_id),
            }

    async def shutdown(self) -> None:
        """Shutdown the handler and release resources."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

        self._initialized = False
        logger.info("PostgresRegistrationStorageHandler shutdown complete")


__all__ = ["PostgresRegistrationStorageHandler"]
