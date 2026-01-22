# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Database Handler - MVP implementation using asyncpg async client.

Supports query and execute operations with fixed pool size (5).
Transaction support deferred to Beta. Configurable pool size deferred to Beta.

All queries MUST use parameterized statements for SQL injection protection.

Envelope-Based Routing:
    This handler uses envelope-based operation routing. See CLAUDE.md section
    "Intent Model Architecture > Envelope-Based Handler Routing" for the full
    design pattern and how orchestrators translate intents to handler envelopes.

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
from uuid import UUID, uuid4

import asyncpg

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_core.types import JsonType
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumResponseStatus,
)
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    RuntimeHostError,
)
from omnibase_infra.handlers.models import (
    ModelDbDescribeResponse,
    ModelDbQueryPayload,
    ModelDbQueryResponse,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker, MixinEnvelopeExtraction
from omnibase_infra.utils.util_env_parsing import parse_env_float, parse_env_int

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

# PostgreSQL SQLSTATE class codes for error classification
# See: https://www.postgresql.org/docs/current/errcodes-appendix.html
#
# TRANSIENT errors (should trip circuit breaker):
#   - Class 08: Connection Exception (database unreachable, connection lost)
#   - Class 53: Insufficient Resources (out of memory, disk full, too many connections)
#   - Class 57: Operator Intervention (admin shutdown, crash recovery, cannot connect now)
#   - Class 58: System Error (I/O error, undefined file, duplicate file)
#
# PERMANENT errors (should NOT trip circuit breaker):
#   - Class 23: Integrity Constraint Violation (FK, NOT NULL, unique, check)
#   - Class 42: Syntax Error or Access Rule Violation (bad SQL, undefined table/column)
#   - Class 28: Invalid Authorization Specification (bad credentials)
#   - Class 22: Data Exception (division by zero, string data truncation)
#   - Class 40: Transaction Rollback (serialization failure, deadlock detected)
#
#     DESIGN DECISION: Classified as PERMANENT despite deadlocks being retry-able.
#
#     Rationale:
#     1. Deadlocks indicate transaction contention, not infrastructure failure
#     2. The database is healthy - it correctly detected and resolved the deadlock
#     3. Retrying at application level (with backoff) typically succeeds
#     4. Tripping the circuit would block ALL queries, not just the conflicting ones
#     5. High deadlock rates indicate application design issues (lock ordering,
#        transaction scope) that should be fixed in code, not masked by circuit breaker
#
#     Note: If sustained deadlock storms occur, this is a symptom of application
#     issues or schema contention that monitoring/alerting should surface, but
#     the circuit breaker is not the right mitigation tool.
#
# The key insight: transient errors indicate the DATABASE INFRASTRUCTURE is unhealthy,
# while permanent errors indicate the QUERY/APPLICATION is invalid. The circuit breaker
# protects against infrastructure failures, not application bugs.
_TRANSIENT_SQLSTATE_CLASSES: frozenset[str] = frozenset({"08", "53", "57", "58"})
_PERMANENT_SQLSTATE_CLASSES: frozenset[str] = frozenset({"22", "23", "28", "42"})


class HandlerDb(MixinAsyncCircuitBreaker, MixinEnvelopeExtraction):
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

    Circuit Breaker:
        This handler uses MixinAsyncCircuitBreaker for connection resilience.
        The circuit breaker is initialized after the connection pool is created
        in the initialize() method. Only connection-related errors (PostgresConnectionError,
        QueryCanceledError) trip the circuit - application errors (syntax errors,
        missing tables/columns) do not affect the circuit state.

        States:
        - CLOSED: Normal operation, requests allowed
        - OPEN: Circuit tripped after threshold failures, requests blocked
        - HALF_OPEN: Testing recovery after reset timeout, limited requests allowed
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize HandlerDb with ONEX container for dependency injection.

        Args:
            container: ONEX container for dependency injection.
        """
        self._container = container
        self._pool: asyncpg.Pool | None = None
        self._pool_size: int = _DEFAULT_POOL_SIZE
        self._timeout: float = _DEFAULT_TIMEOUT_SECONDS
        self._initialized: bool = False
        self._dsn: str = ""
        self._circuit_breaker_initialized: bool = False

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

    async def initialize(self, config: dict[str, object]) -> None:
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
        if isinstance(timeout_raw, int | float):
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

            # Initialize circuit breaker after pool creation succeeds
            self._init_circuit_breaker(
                threshold=5,
                reset_timeout=30.0,
                service_name="db_handler",
                transport_type=EnumInfraTransportType.DATABASE,
            )
            self._circuit_breaker_initialized = True

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
        # Reset circuit breaker state
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()
            self._circuit_breaker_initialized = False

        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False
        logger.info("HandlerDb shutdown complete")

    async def execute(
        self, envelope: dict[str, object]
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
                "HandlerDb not initialized. Call initialize() first.", context=ctx
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
        self, payload: dict[str, object], operation: str, correlation_id: UUID
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

    def _is_transient_error(self, error: asyncpg.PostgresError) -> bool:
        """Determine if a PostgreSQL error is transient (should trip circuit breaker).

        Transient errors indicate infrastructure issues where retrying later may succeed:
        - Connection failures (Class 08)
        - Resource exhaustion (Class 53)
        - Server shutdown/restart (Class 57)
        - System I/O errors (Class 58)

        Permanent errors indicate application/query bugs that won't be fixed by retrying:
        - Constraint violations (Class 23): FK, NOT NULL, unique, check
        - Syntax errors (Class 42): bad SQL, undefined table/column
        - Authorization failures (Class 28): invalid credentials
        - Data exceptions (Class 22): division by zero, truncation

        The circuit breaker ONLY trips on transient errors because:
        1. Transient errors suggest the database infrastructure is unhealthy
        2. Opening the circuit prevents cascading failures and gives the DB time to recover
        3. Permanent errors are application bugs that should be fixed in code, not retried

        Args:
            error: The asyncpg PostgresError exception to classify.

        Returns:
            True if the error is transient (should increment circuit breaker failure count),
            False if the error is permanent (should NOT affect circuit breaker state).

        Examples:
            >>> # Connection lost - transient, should trip circuit
            >>> handler._is_transient_error(asyncpg.PostgresConnectionError())
            True

            >>> # FK violation - permanent, should NOT trip circuit
            >>> handler._is_transient_error(asyncpg.ForeignKeyViolationError())
            False
        """
        # Get SQLSTATE code from exception (5-character code like '23503')
        sqlstate = getattr(error, "sqlstate", None)

        if sqlstate is None:
            # No SQLSTATE available - fall back to exception type classification
            # Connection-related errors are always transient
            if isinstance(error, asyncpg.PostgresConnectionError):
                return True
            # Query canceled (timeout) is transient - indicates server overload
            if isinstance(error, asyncpg.QueryCanceledError):
                return True
            # Default: assume permanent (don't trip circuit for unknown errors)
            # This is conservative - we'd rather miss a transient error than
            # incorrectly trip the circuit on application bugs
            logger.debug(
                "No SQLSTATE for PostgreSQL error, defaulting to permanent classification",
                extra={
                    "error_type": type(error).__name__,
                },
            )
            return False

        # Extract class code (first 2 characters of SQLSTATE)
        # e.g., '23503' -> '23' (Integrity Constraint Violation)
        sqlstate_class = sqlstate[:2]

        # Check if class is explicitly transient
        if sqlstate_class in _TRANSIENT_SQLSTATE_CLASSES:
            logger.debug(
                "Classified PostgreSQL error as TRANSIENT (will trip circuit)",
                extra={
                    "sqlstate": sqlstate,
                    "sqlstate_class": sqlstate_class,
                    "error_type": type(error).__name__,
                },
            )
            return True

        # Check if class is explicitly permanent
        if sqlstate_class in _PERMANENT_SQLSTATE_CLASSES:
            logger.debug(
                "Classified PostgreSQL error as PERMANENT (will NOT trip circuit)",
                extra={
                    "sqlstate": sqlstate,
                    "sqlstate_class": sqlstate_class,
                    "error_type": type(error).__name__,
                },
            )
            return False

        # Unknown class - log warning and default to permanent (conservative)
        # Unknown errors are more likely to be application bugs than infrastructure issues
        logger.warning(
            "Unknown PostgreSQL SQLSTATE class, defaulting to permanent classification",
            extra={
                "sqlstate": sqlstate,
                "sqlstate_class": sqlstate_class,
                "error_type": type(error).__name__,
            },
        )
        return False

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
                "HandlerDb not initialized - call initialize() first", context=ctx
            )

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="db.query",
            target_name="db_handler",
            correlation_id=correlation_id,
        )

        # Check circuit breaker before operation
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker(
                    operation="db.query",
                    correlation_id=correlation_id,
                )

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *parameters)

                # Reset circuit breaker immediately after successful operation,
                # within connection context to avoid race conditions
                if self._circuit_breaker_initialized:
                    async with self._circuit_breaker_lock:
                        await self._reset_circuit_breaker()

            return self._build_response(
                [dict(row) for row in rows],
                len(rows),
                correlation_id,
                input_envelope_id,
            )
        except asyncpg.QueryCanceledError as e:
            # Record failure for timeout errors (database overloaded)
            if self._circuit_breaker_initialized:
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure(
                        operation="db.query",
                        correlation_id=correlation_id,
                    )
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="db.query",
                target_name="db_handler",
                correlation_id=correlation_id,
                timeout_seconds=self._timeout,
            )
            raise InfraTimeoutError(
                f"Query timed out after {self._timeout}s",
                context=timeout_ctx,
            ) from e
        except asyncpg.PostgresConnectionError as e:
            # Record failure for connection errors (database unavailable)
            if self._circuit_breaker_initialized:
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure(
                        operation="db.query",
                        correlation_id=correlation_id,
                    )
            raise InfraConnectionError(
                "Database connection lost during query", context=ctx
            ) from e
        except asyncpg.PostgresSyntaxError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(f"SQL syntax error: {e.message}", context=ctx) from e
        except asyncpg.UndefinedTableError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(f"Table not found: {e.message}", context=ctx) from e
        except asyncpg.UndefinedColumnError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(f"Column not found: {e.message}", context=ctx) from e
        except asyncpg.ForeignKeyViolationError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(
                f"Foreign key constraint violation: {e.message}", context=ctx
            ) from e
        except asyncpg.NotNullViolationError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(
                f"Not null constraint violation: {e.message}", context=ctx
            ) from e
        except asyncpg.UniqueViolationError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(
                f"Unique constraint violation: {e.message}", context=ctx
            ) from e
        except asyncpg.PostgresError as e:
            # Generic PostgreSQL error - use intelligent classification based on SQLSTATE
            # to determine if this is a transient infrastructure issue or permanent app bug
            if self._is_transient_error(e):
                # Transient error (e.g., resource exhaustion, system error)
                # Record failure to potentially trip circuit breaker
                if self._circuit_breaker_initialized:
                    async with self._circuit_breaker_lock:
                        await self._record_circuit_failure(
                            operation="db.query",
                            correlation_id=correlation_id,
                        )
            # Re-raise as RuntimeHostError regardless of classification
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
                "HandlerDb not initialized - call initialize() first", context=ctx
            )

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="db.execute",
            target_name="db_handler",
            correlation_id=correlation_id,
        )

        # Check circuit breaker before operation
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker(
                    operation="db.execute",
                    correlation_id=correlation_id,
                )

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(sql, *parameters)

                # Reset circuit breaker immediately after successful operation,
                # within connection context to avoid race conditions
                if self._circuit_breaker_initialized:
                    async with self._circuit_breaker_lock:
                        await self._reset_circuit_breaker()

            # asyncpg returns string like "INSERT 0 1" or "UPDATE 5"
            row_count = self._parse_row_count(result)
            return self._build_response(
                [], row_count, correlation_id, input_envelope_id
            )
        except asyncpg.QueryCanceledError as e:
            # Record failure for timeout errors (database overloaded)
            if self._circuit_breaker_initialized:
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure(
                        operation="db.execute",
                        correlation_id=correlation_id,
                    )
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="db.execute",
                target_name="db_handler",
                correlation_id=correlation_id,
                timeout_seconds=self._timeout,
            )
            raise InfraTimeoutError(
                f"Statement timed out after {self._timeout}s",
                context=timeout_ctx,
            ) from e
        except asyncpg.PostgresConnectionError as e:
            # Record failure for connection errors (database unavailable)
            if self._circuit_breaker_initialized:
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure(
                        operation="db.execute",
                        correlation_id=correlation_id,
                    )
            raise InfraConnectionError(
                "Database connection lost during statement execution", context=ctx
            ) from e
        except asyncpg.PostgresSyntaxError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(f"SQL syntax error: {e.message}", context=ctx) from e
        except asyncpg.UndefinedTableError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(f"Table not found: {e.message}", context=ctx) from e
        except asyncpg.UndefinedColumnError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(f"Column not found: {e.message}", context=ctx) from e
        except asyncpg.ForeignKeyViolationError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(
                f"Foreign key constraint violation: {e.message}", context=ctx
            ) from e
        except asyncpg.NotNullViolationError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(
                f"Not null constraint violation: {e.message}", context=ctx
            ) from e
        except asyncpg.UniqueViolationError as e:
            # Application error - do NOT trip circuit
            raise RuntimeHostError(
                f"Unique constraint violation: {e.message}", context=ctx
            ) from e
        except asyncpg.PostgresError as e:
            # Generic PostgreSQL error - use intelligent classification based on SQLSTATE
            # to determine if this is a transient infrastructure issue or permanent app bug
            if self._is_transient_error(e):
                # Transient error (e.g., resource exhaustion, system error)
                # Record failure to potentially trip circuit breaker
                if self._circuit_breaker_initialized:
                    async with self._circuit_breaker_lock:
                        await self._record_circuit_failure(
                            operation="db.execute",
                            correlation_id=correlation_id,
                        )
            # Re-raise as RuntimeHostError regardless of classification
            raise RuntimeHostError(
                f"Database error: {type(e).__name__}", context=ctx
            ) from e

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
            # Convert ModelInfraErrorContext to ModelTimeoutErrorContext for stricter typing
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=ctx.transport_type or EnumInfraTransportType.DATABASE,
                operation=ctx.operation or "db.statement",
                target_name=ctx.target_name,
                correlation_id=ctx.correlation_id,
                timeout_seconds=self._timeout,
            )
            return InfraTimeoutError(
                f"Statement timed out after {self._timeout}s",
                context=timeout_ctx,
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
        rows: list[dict[str, object]],
        row_count: int,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelDbQueryResponse]:
        """Build response wrapped in ModelHandlerOutput from query/execute result."""
        result = ModelDbQueryResponse(
            status=EnumResponseStatus.SUCCESS,
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

        Security Consideration:
            The circuit_breaker field exposes operational state including failure counts.
            This information is intended for internal monitoring and observability.

            WARNING: If describe() is exposed via external APIs, failure counts could
            reveal information useful to attackers (e.g., timing attacks when circuit
            is about to open). For external exposure, consider:

            1. Restricting describe() to authenticated admin/monitoring endpoints only
            2. Sanitizing output to show only state (OPEN/CLOSED/HALF_OPEN), not counts
            3. Rate-limiting describe() calls to prevent information harvesting

            The current implementation assumes describe() is used for internal
            monitoring dashboards and health checks, not public APIs.

        See Also:
            - handler_type property: Full documentation of architectural role
            - handler_category property: Full documentation of behavioral classification
            - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md
        """
        # Get circuit breaker state if initialized
        cb_state: dict[str, JsonType] | None = None
        if self._circuit_breaker_initialized:
            cb_state = self._get_circuit_breaker_state()

        return ModelDbDescribeResponse(
            handler_type=self.handler_type.value,
            handler_category=self.handler_category.value,
            supported_operations=sorted(_SUPPORTED_OPERATIONS),
            pool_size=self._pool_size,
            timeout_seconds=self._timeout,
            initialized=self._initialized,
            version="0.1.0-mvp",
            circuit_breaker=cb_state,
        )


__all__: list[str] = ["HandlerDb"]
