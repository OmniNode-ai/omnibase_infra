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

SQL Security:
    All SQL queries in this module use parameterized queries with positional
    placeholders ($1, $2, etc.) to prevent SQL injection attacks. The asyncpg
    library handles proper escaping and type conversion for all parameters.

    Query parameters are ALWAYS passed as separate arguments to execute/fetch
    methods, never interpolated into SQL strings:

    SAFE:
        await conn.execute("SELECT * FROM users WHERE id = $1", user_id)

    UNSAFE (never do this):
        await conn.execute(f"SELECT * FROM users WHERE id = {user_id}")  # WRONG!

    Dynamic WHERE clauses (e.g., in query_registrations) are built by appending
    conditions with parameterized placeholders, not by string interpolation of
    user values.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

# Import asyncpg at module level to avoid redundant imports inside methods
import asyncpg

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
)
from omnibase_infra.handlers.registration_storage.models import (
    ModelDeleteRegistrationRequest,
    ModelUpdateRegistrationRequest,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelDeleteResult,
    ModelRegistrationRecord,
    ModelStorageHealthCheckDetails,
    ModelStorageHealthCheckResult,
    ModelStorageQuery,
    ModelStorageResult,
    ModelUpsertResult,
)

if TYPE_CHECKING:
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
    capabilities JSONB NOT NULL DEFAULT '[]',
    endpoints JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

SQL_UPSERT = """
INSERT INTO node_registrations (node_id, node_type, node_version, capabilities, endpoints, metadata, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (node_id) DO UPDATE SET
    node_type = EXCLUDED.node_type,
    node_version = EXCLUDED.node_version,
    capabilities = EXCLUDED.capabilities,
    endpoints = EXCLUDED.endpoints,
    metadata = EXCLUDED.metadata,
    updated_at = EXCLUDED.updated_at
RETURNING (xmax = 0) AS was_insert;
"""

SQL_QUERY_BASE = """
SELECT node_id, node_type, node_version, capabilities, endpoints, metadata, created_at, updated_at
FROM node_registrations
"""

SQL_QUERY_COUNT = """
SELECT COUNT(*) FROM node_registrations
"""

SQL_UPDATE = """
UPDATE node_registrations SET
    capabilities = COALESCE($2, capabilities),
    endpoints = COALESCE($3, endpoints),
    metadata = COALESCE($4, metadata),
    node_version = COALESCE($5, node_version),
    updated_at = NOW()
WHERE node_id = $1
RETURNING node_id;
"""

SQL_DELETE = """
DELETE FROM node_registrations WHERE node_id = $1 RETURNING node_id;
"""


class HandlerRegistrationStoragePostgres(MixinAsyncCircuitBreaker):
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
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer(...)
        >>> handler = HandlerRegistrationStoragePostgres(
        ...     container=container,
        ...     postgres_adapter=postgres_adapter,
        ...     circuit_breaker_config={"threshold": 5, "reset_timeout": 30.0},
        ... )
        >>> result = await handler.store_registration(record)
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        postgres_adapter: ProtocolPostgresAdapter | None = None,
        dsn: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "omninode_bridge",
        user: str = "postgres",
        password: str | None = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        circuit_breaker_config: ModelCircuitBreakerConfig
        | dict[str, object]
        | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        auto_create_schema: bool = False,
    ) -> None:
        """Initialize HandlerRegistrationStoragePostgres.

        Args:
            container: ONEX dependency injection container (required).
            postgres_adapter: Optional existing PostgreSQL adapter (ProtocolPostgresAdapter).
                If not provided, a new asyncpg connection pool will be created.
            dsn: Optional PostgreSQL connection DSN (overrides host/port/etc).
            host: PostgreSQL server hostname (default: "localhost").
            port: PostgreSQL server port (default: 5432).
            database: Database name (default: "omninode_bridge").
            user: Database user (default: "postgres").
            password: Optional database password.
            pool_size: Connection pool size (default: 10).
            circuit_breaker_config: Optional circuit breaker configuration.
                Accepts ModelCircuitBreakerConfig or dict with keys:
                - threshold: Max failures before opening (default: 5)
                - reset_timeout_seconds: Seconds before reset (default: 60.0)
                - service_name: Service identifier (default: "postgres.storage")
            timeout_seconds: Operation timeout in seconds (default: 30.0).
            auto_create_schema: If True, automatically create the node_registrations
                table on first connection. Default is False. Production deployments
                should use database migrations instead of auto-creation.
        """
        self._container = container
        # Normalize circuit breaker config to ModelCircuitBreakerConfig
        if isinstance(circuit_breaker_config, ModelCircuitBreakerConfig):
            cb_config = circuit_breaker_config
        elif circuit_breaker_config is not None:
            # Handle dict with legacy key names (reset_timeout -> reset_timeout_seconds)
            config_dict = dict(circuit_breaker_config)
            if (
                "reset_timeout" in config_dict
                and "reset_timeout_seconds" not in config_dict
            ):
                config_dict["reset_timeout_seconds"] = config_dict.pop("reset_timeout")
            # Set defaults for service_name and transport_type if not provided
            config_dict.setdefault("service_name", "postgres.storage")
            config_dict.setdefault("transport_type", EnumInfraTransportType.DATABASE)
            cb_config = ModelCircuitBreakerConfig(**config_dict)
        else:
            cb_config = ModelCircuitBreakerConfig(
                service_name="postgres.storage",
                transport_type=EnumInfraTransportType.DATABASE,
            )

        self._init_circuit_breaker(
            threshold=cb_config.threshold,
            reset_timeout=cb_config.reset_timeout_seconds,
            service_name=cb_config.service_name,
            transport_type=cb_config.transport_type,
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
        self._auto_create_schema = auto_create_schema

        # Connection pool (initialized on first use)
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()
        self._initialized = False

        # External adapter (if provided)
        self._postgres_adapter = postgres_adapter

        logger.info(
            "HandlerRegistrationStoragePostgres created",
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

    async def _ensure_pool(
        self,
        correlation_id: UUID | None = None,
    ) -> asyncpg.Pool:
        """Ensure connection pool is initialized.

        Args:
            correlation_id: Optional correlation ID for tracing.

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

                # Create table only if auto_create_schema is enabled
                # Production deployments should use database migrations
                if self._auto_create_schema:
                    async with self._pool.acquire() as conn:
                        await conn.execute(SQL_CREATE_TABLE)

                self._initialized = True

                logger.info(
                    "PostgreSQL connection pool initialized",
                    extra={
                        "host": self._host,
                        "port": self._port,
                        "database": self._database,
                        "auto_create_schema": self._auto_create_schema,
                    },
                )

                return self._pool

            except Exception as e:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="initialize_pool",
                    target_name="postgres.storage",
                    correlation_id=correlation_id,
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
            pool = await self._ensure_pool(correlation_id=correlation_id)

            now = datetime.now(UTC)
            capabilities_json = json.dumps(record.capabilities)
            endpoints_json = json.dumps(record.endpoints)
            metadata_json = json.dumps(record.metadata)

            async with pool.acquire() as conn:
                result = await asyncio.wait_for(
                    conn.fetchrow(
                        SQL_UPSERT,
                        record.node_id,
                        record.node_type.value,
                        record.node_version,
                        capabilities_json,
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
            operation = "insert" if was_insert else "update"
            duration_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "Registration stored in PostgreSQL",
                extra={
                    "node_id": str(record.node_id),
                    "operation": operation,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelUpsertResult(
                success=True,
                node_id=record.node_id,
                operation=operation,
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
            raise InfraTimeoutError(
                f"PostgreSQL upsert timed out after {self._timeout_seconds}s",
                context=ModelTimeoutErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="store_registration",
                    target_name="postgres.storage",
                    correlation_id=correlation_id,
                    timeout_seconds=self._timeout_seconds,
                ),
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
        query: ModelStorageQuery,
        correlation_id: UUID | None = None,
    ) -> ModelStorageResult:
        """Query registration records from PostgreSQL.

        Args:
            query: ModelStorageQuery containing filter and pagination parameters.
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
            pool = await self._ensure_pool(correlation_id=correlation_id)

            # Build query with parameterized filters
            # NOTE: All filter values use positional parameters ($1, $2, etc.)
            # to prevent SQL injection. The param_idx tracks parameter positions.
            # User values are NEVER interpolated into SQL strings.
            conditions: list[str] = []
            params: list[object] = []
            param_idx = 1

            # Filter by node_id if specified (exact match)
            if query.node_id is not None:
                conditions.append(f"node_id = ${param_idx}")
                params.append(query.node_id)
                param_idx += 1

            # Filter by node_type if specified
            if query.node_type is not None:
                conditions.append(f"node_type = ${param_idx}")
                params.append(query.node_type.value)
                param_idx += 1

            # Filter by capability (JSONB array contains match)
            if query.capability_filter is not None:
                # Use JSONB containment operator to check if capability exists in array
                conditions.append(f"capabilities @> ${param_idx}::jsonb")
                params.append(json.dumps([query.capability_filter]))
                param_idx += 1

            where_clause = ""
            if conditions:
                where_clause = " WHERE " + " AND ".join(conditions)

            # Query for records with pagination
            sql_query = f"{SQL_QUERY_BASE}{where_clause} ORDER BY updated_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            params.extend([query.limit, query.offset])

            # Query for total count
            count_query = f"{SQL_QUERY_COUNT}{where_clause}"
            count_params = params[:-2]  # Exclude limit and offset

            async with pool.acquire() as conn:
                rows, count_result = await asyncio.gather(
                    asyncio.wait_for(
                        conn.fetch(sql_query, *params),
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
                capabilities = (
                    json.loads(row["capabilities"]) if row["capabilities"] else []
                )
                endpoints = json.loads(row["endpoints"]) if row["endpoints"] else {}
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}

                records.append(
                    ModelRegistrationRecord(
                        node_id=row["node_id"],
                        node_type=EnumNodeKind(row["node_type"]),
                        node_version=row["node_version"],
                        capabilities=capabilities,
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
            raise InfraTimeoutError(
                f"PostgreSQL query timed out after {self._timeout_seconds}s",
                context=ModelTimeoutErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="query_registrations",
                    target_name="postgres.storage",
                    correlation_id=correlation_id,
                    timeout_seconds=self._timeout_seconds,
                ),
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
        request: ModelUpdateRegistrationRequest,
    ) -> ModelUpsertResult:
        """Update an existing registration record.

        Args:
            request: ModelUpdateRegistrationRequest containing:
                - node_id: ID of the node to update
                - updates: ModelRegistrationUpdate with fields to update
                  (only non-None fields will be applied)
                - correlation_id: Optional correlation ID for tracing

        Returns:
            ModelUpsertResult with update outcome.

        Raises:
            InfraConnectionError: If connection to PostgreSQL fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        # Extract fields from request model
        node_id = request.node_id
        updates = request.updates
        correlation_id = request.correlation_id or uuid4()
        start_time = time.monotonic()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="update_registration",
                correlation_id=correlation_id,
            )

        try:
            pool = await self._ensure_pool(correlation_id=correlation_id)

            # Extract fields from the update model
            # Use `is not None` checks to allow explicitly clearing fields with empty
            # lists/dicts. Truthiness checks would treat [] and {} as "no update".
            capabilities_json = (
                json.dumps(updates.capabilities)
                if updates.capabilities is not None
                else None
            )
            endpoints_json = (
                json.dumps(updates.endpoints) if updates.endpoints is not None else None
            )
            metadata_json = (
                json.dumps(updates.metadata) if updates.metadata is not None else None
            )
            node_version = updates.node_version

            async with pool.acquire() as conn:
                result = await asyncio.wait_for(
                    conn.fetchval(
                        SQL_UPDATE,
                        node_id,
                        capabilities_json,
                        endpoints_json,
                        metadata_json,
                        node_version,
                    ),
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
                operation="update",
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
            raise InfraTimeoutError(
                f"PostgreSQL update timed out after {self._timeout_seconds}s",
                context=ModelTimeoutErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="update_registration",
                    target_name="postgres.storage",
                    correlation_id=correlation_id,
                    timeout_seconds=self._timeout_seconds,
                ),
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
        request: ModelDeleteRegistrationRequest,
    ) -> ModelDeleteResult:
        """Delete a registration record from PostgreSQL.

        Args:
            request: ModelDeleteRegistrationRequest containing:
                - node_id: ID of the node to delete
                - correlation_id: Optional correlation ID for tracing

        Returns:
            ModelDeleteResult with deletion outcome.

        Raises:
            InfraConnectionError: If connection to PostgreSQL fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        # Extract fields from request model
        node_id = request.node_id
        correlation_id = request.correlation_id or uuid4()
        start_time = time.monotonic()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="delete_registration",
                correlation_id=correlation_id,
            )

        try:
            pool = await self._ensure_pool(correlation_id=correlation_id)

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

            return ModelDeleteResult(
                success=True,
                node_id=node_id,
                deleted=deleted,
                duration_ms=duration_ms,
                backend_type=self.handler_type,
                correlation_id=correlation_id,
            )

        except TimeoutError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="delete_registration",
                    correlation_id=correlation_id,
                )
            raise InfraTimeoutError(
                f"PostgreSQL delete timed out after {self._timeout_seconds}s",
                context=ModelTimeoutErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="delete_registration",
                    target_name="postgres.storage",
                    correlation_id=correlation_id,
                    timeout_seconds=self._timeout_seconds,
                ),
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
    ) -> ModelStorageHealthCheckResult:
        """Perform a health check on the PostgreSQL connection.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelStorageHealthCheckResult with health status information.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        try:
            pool = await self._ensure_pool(correlation_id=correlation_id)

            async with pool.acquire() as conn:
                await asyncio.wait_for(
                    conn.fetchval("SELECT 1"),
                    timeout=5.0,  # Short timeout for health check
                )

            duration_ms = (time.monotonic() - start_time) * 1000

            return ModelStorageHealthCheckResult(
                healthy=True,
                backend_type=self.handler_type,
                latency_ms=duration_ms,
                reason="ok",
                details=ModelStorageHealthCheckDetails(
                    pool_size=self._pool_size,
                    database_name=self._database,
                ),
                correlation_id=correlation_id,
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            return ModelStorageHealthCheckResult(
                healthy=False,
                backend_type=self.handler_type,
                latency_ms=duration_ms,
                reason=f"Health check failed: {type(e).__name__}",
                error_type=type(e).__name__,
                details=ModelStorageHealthCheckDetails(
                    pool_size=self._pool_size,
                    database_name=self._database,
                ),
                correlation_id=correlation_id,
            )

    async def shutdown(self) -> None:
        """Shutdown the handler and release resources."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

        self._initialized = False
        logger.info("HandlerRegistrationStoragePostgres shutdown complete")


__all__ = ["HandlerRegistrationStoragePostgres"]
