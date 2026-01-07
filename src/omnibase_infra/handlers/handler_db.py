# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Database Handler - MVP implementation using asyncpg async client.

Supports query and execute operations with fixed pool size (5).
Transaction support deferred to Beta. Configurable pool size deferred to Beta.

All queries MUST use parameterized statements for SQL injection protection.

Single-Statement SQL Limitation
===============================

This handler uses asyncpg's ``execute()`` and ``fetch()`` methods, which only
support **single SQL statements per call**. Multi-statement SQL (statements
separated by semicolons) is NOT supported and will raise an error.

**Example - Incorrect (will fail):**

.. code-block:: python

    # This will fail - multiple statements in one call
    envelope = {
        "operation": "db.execute",
        "payload": {
            "sql": "CREATE TABLE foo (id INT); INSERT INTO foo VALUES (1);",
            "parameters": [],
        },
    }

**Example - Correct (split into separate calls):**

.. code-block:: python

    # Execute each statement separately
    create_envelope = {
        "operation": "db.execute",
        "payload": {"sql": "CREATE TABLE foo (id INT)", "parameters": []},
    }
    await handler.execute(create_envelope)

    insert_envelope = {
        "operation": "db.execute",
        "payload": {"sql": "INSERT INTO foo VALUES (1)", "parameters": []},
    }
    await handler.execute(insert_envelope)

This is a deliberate design choice for security and clarity:
1. Prevents SQL injection through statement concatenation
2. Provides clear error attribution per statement
3. Enables proper row count tracking per operation
4. Aligns with asyncpg's native API design

For multi-statement operations requiring atomicity, use the ``db.transaction``
operation (planned for Beta release).

Note:
    Environment variable configuration (ONEX_DB_POOL_SIZE, ONEX_DB_TIMEOUT) is parsed
    at module import time, not at handler instantiation. This means:

    - Changes to environment variables require application restart to take effect
    - Tests should use ``unittest.mock.patch.dict(os.environ, ...)`` before importing,
      or use ``importlib.reload()`` to re-import the module after patching
    - This is an intentional design choice for startup-time validation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import asyncpg
from omnibase_core.models.dispatch import ModelHandlerOutput

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.handlers.models import (
    ModelDbDescribeResponse,
    ModelDbQueryPayload,
    ModelDbQueryResponse,
)
from omnibase_infra.mixins import MixinEnvelopeExtraction
from omnibase_infra.utils.util_env_parsing import parse_env_float, parse_env_int

if TYPE_CHECKING:
    from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)

# MVP pool size fixed at 5 connections.
# Note: Recommended range is 10-20 for production workloads.
# Configurable pool size deferred to Beta release.
_DEFAULT_POOL_SIZE: int = parse_env_int(
    "ONEX_DB_POOL_SIZE",
    5,
    min_value=1,
    max_value=100,
    transport_type=EnumInfraTransportType.DATABASE,
    service_name="db_handler",
)

# Handler ID for ModelHandlerOutput
HANDLER_ID_DB: str = "db-handler"
_DEFAULT_TIMEOUT_SECONDS: float = parse_env_float(
    "ONEX_DB_TIMEOUT",
    30.0,
    min_value=0.1,
    max_value=3600.0,
    transport_type=EnumInfraTransportType.DATABASE,
    service_name="db_handler",
)
_SUPPORTED_OPERATIONS: frozenset[str] = frozenset({"db.query", "db.execute"})

# Error message prefixes for PostgreSQL errors
# Used by _map_postgres_error to build descriptive error messages
_POSTGRES_ERROR_PREFIXES: dict[type[asyncpg.PostgresError], str] = {
    asyncpg.PostgresSyntaxError: "SQL syntax error",
    asyncpg.UndefinedTableError: "Table not found",
    asyncpg.UndefinedColumnError: "Column not found",
    asyncpg.UniqueViolationError: "Unique constraint violation",
    asyncpg.ForeignKeyViolationError: "Foreign key constraint violation",
    asyncpg.NotNullViolationError: "Not null constraint violation",
    asyncpg.CheckViolationError: "Check constraint violation",
}


class DbHandler(MixinEnvelopeExtraction):
    """PostgreSQL database handler using asyncpg connection pool (MVP: query, execute only).

    Security Policy - DSN Handling:
        The database connection string (DSN) contains sensitive credentials and is
        treated as a secret throughout this handler. The following security measures
        are enforced:

        1. DSN is stored internally in ``_dsn`` but NEVER logged or exposed in errors
        2. All error messages use generic descriptions (e.g., "check host and port")
           rather than exposing connection details
        3. The ``_sanitize_dsn()`` method is available if DSN info ever needs to be
           logged for debugging, but should only be used in development environments
        4. The ``describe()`` method returns capabilities without credentials

        See CLAUDE.md "Error Sanitization Guidelines" for the full security policy
        on what information is safe vs unsafe to include in errors and logs.

    Production Database Safety:
        When connecting to production databases, ensure the following safeguards:

        1. **Use read-only credentials** when possible to prevent accidental mutations
        2. **Connection isolation**: Use separate DSNs for read and write operations
        3. **Query timeouts**: Configure appropriate timeouts to prevent long-running
           queries from exhausting connection pools (default: 30 seconds)
        4. **Pool limits**: Production workloads should use 10-20 connections
           (currently fixed at 5 for MVP - see Beta roadmap)
        5. **SSL/TLS**: Always use encrypted connections (sslmode=require/verify-full)
        6. **Audit logging**: Enable PostgreSQL statement logging for compliance
        7. **Connection pooling**: Consider PgBouncer for high-traffic scenarios

        WARNING: This handler executes arbitrary SQL. Ensure all queries use
        parameterized statements to prevent SQL injection. Multi-statement SQL
        is intentionally blocked for security.

    TODO(OMN-42): Consider implementing circuit breaker pattern for connection
    resilience. See CLAUDE.md "Error Recovery Patterns" for implementation guidance.
    """

    def __init__(self) -> None:
        """Initialize DbHandler in uninitialized state."""
        self._pool: asyncpg.Pool | None = None
        self._pool_size: int = _DEFAULT_POOL_SIZE
        self._timeout: float = _DEFAULT_TIMEOUT_SECONDS
        self._initialized: bool = False
        self._dsn: str = ""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            protocol/transport handler (as opposed to NODE_HANDLER for event
            processing, PROJECTION_HANDLER for read models, or COMPUTE_HANDLER
            for pure computation).

        Note:
            handler_type determines lifecycle, protocol selection, and runtime
            invocation patterns. It answers "what is this handler in the architecture?"

        See Also:
            - handler_category: Behavioral classification (EFFECT/COMPUTE)
            - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.EFFECT - This handler performs side-effecting
            I/O operations (database queries and mutations). EFFECT handlers are
            not deterministic and interact with external systems.

        Note:
            handler_category determines security rules, determinism guarantees,
            replay safety, and permissions. It answers "how does this handler
            behave at runtime?"

            Categories:
            - COMPUTE: Pure, deterministic transformations (no side effects)
            - EFFECT: Side-effecting I/O (database, HTTP, service calls)
            - NONDETERMINISTIC_COMPUTE: Pure but not deterministic (UUID, random)

        See Also:
            - handler_type: Architectural role (INFRA_HANDLER/NODE_HANDLER/etc.)
            - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md
        """
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, JsonType]) -> None:
        """Initialize database connection pool with fixed size (5).

        Args:
            config: Configuration dict containing:
                - dsn: PostgreSQL connection string (required)
                - timeout: Optional timeout in seconds (default: 30.0)

        Raises:
            RuntimeHostError: If DSN is missing or pool creation fails.
        """
        # Generate correlation_id for initialization tracing
        init_correlation_id = uuid4()

        logger.info(
            "Initializing %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
            },
        )

        dsn = config.get("dsn")
        if not isinstance(dsn, str) or not dsn:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="initialize",
                target_name="db_handler",
                correlation_id=init_correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'dsn' in config - PostgreSQL connection string required",
                context=ctx,
            )

        timeout_raw = config.get("timeout", _DEFAULT_TIMEOUT_SECONDS)
        if isinstance(timeout_raw, (int, float)):
            self._timeout = float(timeout_raw)

        try:
            self._pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=1,
                max_size=self._pool_size,
                command_timeout=self._timeout,
            )
            self._dsn = dsn
            # Note: DSN stored internally but never logged or exposed in errors.
            # Use _sanitize_dsn() if DSN info ever needs to be logged.
            self._initialized = True
            logger.info(
                "%s initialized successfully",
                self.__class__.__name__,
                extra={
                    "handler": self.__class__.__name__,
                    "pool_min_size": 1,
                    "pool_max_size": self._pool_size,
                    "timeout_seconds": self._timeout,
                    "correlation_id": str(init_correlation_id),
                },
            )
        except asyncpg.InvalidPasswordError as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="initialize",
                target_name="db_handler",
                correlation_id=init_correlation_id,
            )
            raise InfraAuthenticationError(
                "Database authentication failed - check credentials", context=ctx
            ) from e
        except asyncpg.InvalidCatalogNameError as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="initialize",
                target_name="db_handler",
                correlation_id=init_correlation_id,
            )
            raise RuntimeHostError(
                "Database not found - check database name", context=ctx
            ) from e
        except OSError as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="initialize",
                target_name="db_handler",
                correlation_id=init_correlation_id,
            )
            raise InfraConnectionError(
                "Failed to connect to database - check host and port", context=ctx
            ) from e
        except Exception as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="initialize",
                target_name="db_handler",
                correlation_id=init_correlation_id,
            )
            raise RuntimeHostError(
                f"Failed to initialize database pool: {type(e).__name__}", context=ctx
            ) from e

    async def shutdown(self) -> None:
        """Close database connection pool and release resources."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False
        logger.info("DbHandler shutdown complete")

    async def execute(
        self, envelope: dict[str, JsonType]
    ) -> ModelHandlerOutput[ModelDbQueryResponse]:
        """Execute database operation (db.query or db.execute) from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: "db.query" or "db.execute"
                - payload: dict with "sql" (required) and "parameters" (optional list)
                - correlation_id: Optional correlation ID for tracing
                - envelope_id: Optional envelope ID for causality tracking

        Returns:
            ModelHandlerOutput[ModelDbQueryResponse] containing:
                - result: ModelDbQueryResponse with status, payload, and correlation_id
                - input_envelope_id: UUID for causality tracking
                - correlation_id: UUID for request/response correlation
                - handler_id: "db-handler"

        Raises:
            RuntimeHostError: If handler not initialized or invalid input.
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If query times out.
        """
        correlation_id = self._extract_correlation_id(envelope)
        input_envelope_id = self._extract_envelope_id(envelope)

        if not self._initialized or self._pool is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="execute",
                target_name="db_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "DbHandler not initialized. Call initialize() first.", context=ctx
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="execute",
                target_name="db_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'operation' in envelope", context=ctx
            )

        if operation not in _SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation=operation,
                target_name="db_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Operation '{operation}' not supported in MVP. Available: {', '.join(sorted(_SUPPORTED_OPERATIONS))}",
                context=ctx,
            )

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation=operation,
                target_name="db_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope", context=ctx
            )

        sql = payload.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation=operation,
                target_name="db_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError("Missing or invalid 'sql' in payload", context=ctx)

        parameters = self._extract_parameters(payload, operation, correlation_id)

        if operation == "db.query":
            return await self._execute_query(
                sql, parameters, correlation_id, input_envelope_id
            )
        else:  # db.execute
            return await self._execute_statement(
                sql, parameters, correlation_id, input_envelope_id
            )

    def _sanitize_dsn(self, dsn: str) -> str:
        """Sanitize DSN by removing password for safe logging.

        SECURITY: This method exists to support debugging scenarios where
        connection information may be helpful, while ensuring credentials
        are never exposed. The raw DSN should NEVER be logged directly.

        Uses urllib.parse for robust parsing instead of regex, handling
        edge cases like IPv6 addresses and URL-encoded passwords.

        Args:
            dsn: Raw PostgreSQL connection string containing credentials.

        Returns:
            Sanitized DSN with password replaced by '***'.

        Example:
            >>> handler._sanitize_dsn("postgresql://user:secret@host:5432/db")
            'postgresql://user:***@host:5432/db'

            >>> handler._sanitize_dsn("postgresql://user:p%40ss@[::1]:5432/db")
            'postgresql://user:***@[::1]:5432/db'

        Note:
            This method is intentionally NOT used in production error paths.
            It exists as a utility for development/debugging only. See class
            docstring "Security Policy - DSN Handling" for full policy.
        """
        from omnibase_infra.utils.util_dsn_validation import sanitize_dsn

        return sanitize_dsn(dsn)

    def _extract_parameters(
        self, payload: dict[str, JsonType], operation: str, correlation_id: UUID
    ) -> list[object]:
        """Extract and validate parameters from payload."""
        params_raw = payload.get("parameters")
        if params_raw is None:
            return []
        if isinstance(params_raw, list):
            return list(params_raw)
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation=operation,
            target_name="db_handler",
            correlation_id=correlation_id,
        )
        raise RuntimeHostError(
            "Invalid 'parameters' in payload - must be a list", context=ctx
        )

    async def _execute_query(
        self,
        sql: str,
        parameters: list[object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelDbQueryResponse]:
        """Execute SELECT query and return rows."""
        if self._pool is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="db.query",
                target_name="db_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "DbHandler not initialized - call initialize() first", context=ctx
            )

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="db.query",
            target_name="db_handler",
            correlation_id=correlation_id,
        )

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *parameters)
                return self._build_response(
                    [dict(row) for row in rows],
                    len(rows),
                    correlation_id,
                    input_envelope_id,
                )
        except asyncpg.QueryCanceledError as e:
            raise InfraTimeoutError(
                f"Query timed out after {self._timeout}s",
                context=ctx,
                timeout_seconds=self._timeout,
            ) from e
        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                "Database connection lost during query", context=ctx
            ) from e
        except asyncpg.PostgresSyntaxError as e:
            raise RuntimeHostError(f"SQL syntax error: {e.message}", context=ctx) from e
        except asyncpg.UndefinedTableError as e:
            raise RuntimeHostError(f"Table not found: {e.message}", context=ctx) from e
        except asyncpg.UndefinedColumnError as e:
            raise RuntimeHostError(f"Column not found: {e.message}", context=ctx) from e
        except asyncpg.PostgresError as e:
            raise RuntimeHostError(
                f"Database error: {type(e).__name__}", context=ctx
            ) from e

    async def _execute_statement(
        self,
        sql: str,
        parameters: list[object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelDbQueryResponse]:
        """Execute INSERT/UPDATE/DELETE statement and return affected row count."""
        if self._pool is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="db.execute",
                target_name="db_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "DbHandler not initialized - call initialize() first", context=ctx
            )

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="db.execute",
            target_name="db_handler",
            correlation_id=correlation_id,
        )

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(sql, *parameters)
                # asyncpg returns string like "INSERT 0 1" or "UPDATE 5"
                row_count = self._parse_row_count(result)
                return self._build_response(
                    [], row_count, correlation_id, input_envelope_id
                )
        except asyncpg.PostgresError as e:
            raise self._map_postgres_error(e, ctx) from e

    def _parse_row_count(self, result: str) -> int:
        """Parse row count from asyncpg execute result string.

        asyncpg returns strings like:
        - "INSERT 0 1" -> 1 row inserted
        - "UPDATE 5" -> 5 rows updated
        - "DELETE 3" -> 3 rows deleted
        """
        try:
            parts = result.split()
            if len(parts) >= 2:
                return int(parts[-1])
        except (ValueError, IndexError):
            pass
        return 0

    def _map_postgres_error(
        self,
        exc: asyncpg.PostgresError,
        ctx: ModelInfraErrorContext,
    ) -> RuntimeHostError | InfraTimeoutError | InfraConnectionError:
        """Map asyncpg exception to ONEX infrastructure error.

        This helper reduces complexity of _execute_statement and _execute_query
        by centralizing exception-to-error mapping logic.

        Args:
            exc: The asyncpg exception that was raised.
            ctx: Error context with transport type, operation, and correlation ID.

        Returns:
            Appropriate ONEX infrastructure error based on exception type.
        """
        exc_type = type(exc)

        # Special cases requiring specific error types or additional arguments
        if exc_type is asyncpg.QueryCanceledError:
            return InfraTimeoutError(
                f"Statement timed out after {self._timeout}s",
                context=ctx,
                timeout_seconds=self._timeout,
            )

        if exc_type is asyncpg.PostgresConnectionError:
            return InfraConnectionError(
                "Database connection lost during statement execution",
                context=ctx,
            )

        # All other errors map to RuntimeHostError with descriptive message
        prefix = _POSTGRES_ERROR_PREFIXES.get(exc_type, "Database error")
        # Use message attribute if available and non-empty, else use type name
        message = getattr(exc, "message", None) or type(exc).__name__
        return RuntimeHostError(f"{prefix}: {message}", context=ctx)

    def _build_response(
        self,
        rows: list[dict[str, JsonType]],
        row_count: int,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelDbQueryResponse]:
        """Build response wrapped in ModelHandlerOutput from query/execute result."""
        result = ModelDbQueryResponse(
            status="success",
            payload=ModelDbQueryPayload(rows=rows, row_count=row_count),
            correlation_id=correlation_id,
        )
        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_DB,
            result=result,
        )

    def describe(self) -> ModelDbDescribeResponse:
        """Return handler metadata and capabilities for introspection.

        This method exposes the handler's type classification along with its
        operational configuration and capabilities.

        Returns:
            ModelDbDescribeResponse containing:
                - handler_type: Architectural role from handler_type property
                  (e.g., "infra_handler"). See EnumHandlerType for valid values.
                - handler_category: Behavioral classification from handler_category
                  property (e.g., "effect"). See EnumHandlerTypeCategory for valid values.
                - supported_operations: List of supported operations
                - pool_size: Connection pool size
                - timeout_seconds: Query timeout in seconds
                - initialized: Whether the handler is initialized
                - version: Handler version string

        Note:
            The handler_type and handler_category fields form the handler
            classification system:

            1. handler_type (architectural role): Determines lifecycle and invocation
               patterns. This handler is INFRA_HANDLER (protocol/transport handler).

            2. handler_category (behavioral classification): Determines security rules
               and replay safety. This handler is EFFECT (side-effecting I/O).

            The transport type for this handler is DATABASE (PostgreSQL).

        See Also:
            - handler_type property: Full documentation of architectural role
            - handler_category property: Full documentation of behavioral classification
            - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md
        """
        return ModelDbDescribeResponse(
            handler_type=self.handler_type.value,
            handler_category=self.handler_category.value,
            supported_operations=sorted(_SUPPORTED_OPERATIONS),
            pool_size=self._pool_size,
            timeout_seconds=self._timeout,
            initialized=self._initialized,
            version="0.1.0-mvp",
        )


__all__: list[str] = ["DbHandler"]
