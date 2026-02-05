# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Repository Runtime.

This module provides a generic runtime for executing repository contracts
against PostgreSQL databases. The runtime enforces safety constraints,
deterministic query ordering, and configurable operation limits.

Key Features:
    - Contract-driven: All operations defined in ModelDbRepositoryContract
    - Positional parameters: Uses $1, $2, ... (no named param rewriting)
    - Determinism enforcement: ORDER BY injection for multi-row queries
    - Limit enforcement: LIMIT injection with configurable maximum
    - Operation validation: Allowlist-based operation control
    - Timeout enforcement: asyncio.wait_for() for query cancellation

Usage Example:
    >>> import asyncpg
    >>> from omnibase_infra.runtime.db import (
    ...     ModelDbRepositoryContract,
    ...     ModelDbOperation,
    ...     ModelDbReturn,
    ...     ModelRepositoryRuntimeConfig,
    ... )
    >>> from omnibase_infra.runtime.db.postgres_repository_runtime import (
    ...     PostgresRepositoryRuntime,
    ... )
    >>>
    >>> # Create contract
    >>> contract = ModelDbRepositoryContract(
    ...     name="users",
    ...     database_ref="primary",
    ...     ops={
    ...         "find_by_id": ModelDbOperation(
    ...             mode="select",
    ...             sql="SELECT * FROM users WHERE id = $1",
    ...             params=["user_id"],
    ...             returns=ModelDbReturn(many=False),
    ...         ),
    ...     },
    ... )
    >>>
    >>> # Create runtime (with pool)
    >>> pool = await asyncpg.create_pool(...)
    >>> runtime = PostgresRepositoryRuntime(pool, contract)
    >>>
    >>> # Execute operation
    >>> user = await runtime.call("find_by_id", 123)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.repository import (
    RepositoryContractError,
    RepositoryExecutionError,
    RepositoryTimeoutError,
    RepositoryValidationError,
)
from omnibase_infra.models.errors import ModelInfraErrorContext
from omnibase_infra.models.ledger import (
    ModelDbQueryFailed,
    ModelDbQueryRequested,
    ModelDbQuerySucceeded,
    ModelLedgerEventBase,
)
from omnibase_infra.runtime.db.models import (
    ModelDbOperation,
    ModelDbRepositoryContract,
    ModelRepositoryRuntimeConfig,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.protocols import ProtocolLedgerSink

logger = logging.getLogger(__name__)

# =============================================================================
# SQL Clause Detection Patterns
# =============================================================================
#
# These regex patterns provide simple ORDER BY and LIMIT clause detection
# for determinism enforcement. They use word boundaries (\b) and case-insensitive
# matching to identify SQL keywords.
#
# KNOWN LIMITATIONS:
# -----------------
# These patterns use simple regex matching, NOT a full SQL parser. As such,
# they can produce false positives in certain edge cases:
#
# 1. String Literals: Patterns inside quoted strings will match:
#    - SELECT description FROM items WHERE note = 'sort ORDER BY priority'
#    - SELECT * FROM logs WHERE message LIKE '%LIMIT 10 reached%'
#
# 2. Subqueries: Patterns in nested queries will match the outer detection:
#    - SELECT * FROM (SELECT id FROM users ORDER BY created_at LIMIT 5) sub
#    - The outer query appears to have ORDER BY/LIMIT, but doesn't
#
# 3. Comments: Patterns inside SQL comments will match:
#    - SELECT * FROM users -- ORDER BY id for debugging
#    - SELECT * FROM users /* LIMIT 100 was here */
#
# WHY THIS IS ACCEPTABLE:
# ----------------------
# 1. Contract SQL should be simple, predictable queries. Complex queries with
#    subqueries, dynamic string construction, or embedded SQL in literals
#    indicate contract design that should be reconsidered.
#
# 2. This detection is defense-in-depth, not primary validation. The contract
#    author has explicit control over the SQL and can always add explicit
#    ORDER BY and LIMIT clauses to avoid injection entirely.
#
# 3. False positives (detecting ORDER BY/LIMIT when not present at outer level)
#    are safer than false negatives. A false positive skips injection, leaving
#    the query unchanged. A false negative would inject duplicate clauses.
#
# RECOMMENDATION FOR COMPLEX QUERIES:
# ----------------------------------
# If your contract requires complex SQL with subqueries or string operations
# that contain SQL keywords, explicitly include ORDER BY and LIMIT in the
# outer query. This bypasses regex detection entirely:
#
#   GOOD: "SELECT * FROM (SELECT id FROM users ORDER BY id) sub ORDER BY id LIMIT 100"
#   AVOID: Relying on injection for queries with embedded SQL-like strings
#
# =============================================================================
_ORDER_BY_PATTERN = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
# Pattern to detect numeric LIMIT for validation (e.g., LIMIT 100)
_LIMIT_NUMERIC_PATTERN = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
# Pattern to detect parameterized LIMIT (e.g., LIMIT $1) - cannot validate at build time
_LIMIT_PARAM_PATTERN = re.compile(r"\bLIMIT\s+\$\d+\b", re.IGNORECASE)


class PostgresRepositoryRuntime:
    """Runtime for executing repository contracts against PostgreSQL.

    Executes operations defined in a ModelDbRepositoryContract with
    safety constraints, determinism guarantees, and configurable limits.

    Thread Safety:
        This class is NOT thread-safe for concurrent modifications.
        The pool itself handles connection-level concurrency.
        Multiple coroutines may call() concurrently on the same runtime.

    Attributes:
        pool: asyncpg connection pool for database access.
        contract: Repository contract defining available operations.
        config: Runtime configuration for safety and behavior.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> runtime = PostgresRepositoryRuntime(pool, contract)
        >>> results = await runtime.call("find_all")
    """

    __slots__ = (
        "_config",
        "_contract",
        "_contract_fingerprint",
        "_ledger_sink",
        "_pool",
    )

    def __init__(
        self,
        pool: asyncpg.Pool,
        contract: ModelDbRepositoryContract,
        config: ModelRepositoryRuntimeConfig | None = None,
        ledger_sink: ProtocolLedgerSink | None = None,
    ) -> None:
        """Initialize the repository runtime.

        Args:
            pool: asyncpg connection pool for database access.
            contract: Repository contract defining available operations.
            config: Optional runtime configuration. If None, uses defaults.
            ledger_sink: Optional ledger sink for emitting traceability events.
                If provided, events are emitted on call() entry, success, and failure.

        Example:
            >>> runtime = PostgresRepositoryRuntime(
            ...     pool=pool,
            ...     contract=contract,
            ...     config=ModelRepositoryRuntimeConfig(max_row_limit=100),
            ...     ledger_sink=FileSpoolLedgerSink("/var/log/ledger"),
            ... )
        """
        self._pool = pool
        self._contract = contract
        self._config = config or ModelRepositoryRuntimeConfig()
        self._ledger_sink = ledger_sink
        self._contract_fingerprint = self._compute_contract_fingerprint()

    @property
    def contract(self) -> ModelDbRepositoryContract:
        """Get the repository contract."""
        return self._contract

    @property
    def config(self) -> ModelRepositoryRuntimeConfig:
        """Get the runtime configuration."""
        return self._config

    @property
    def contract_fingerprint(self) -> str:
        """Get the SHA256 fingerprint of the contract."""
        return self._contract_fingerprint

    def _compute_contract_fingerprint(self) -> str:
        """Compute SHA256 fingerprint of canonical contract JSON.

        Uses sorted keys and stable serialization for determinism.
        Does not hash raw YAML to avoid formatting churn.
        """
        # Serialize contract to canonical JSON (sorted keys)
        contract_dict = self._contract.model_dump(mode="json")
        canonical_json = json.dumps(
            contract_dict, sort_keys=True, separators=(",", ":")
        )
        return f"sha256:{hashlib.sha256(canonical_json.encode()).hexdigest()}"

    def _compute_query_fingerprint(self, op_name: str, args: tuple[object, ...]) -> str:
        """Compute fingerprint of query (operation + param shape, NOT values).

        Security: Does NOT include raw SQL or param values.
        Only includes operation name and param type names for shape.
        """
        # Build param shape: list of type names (not values)
        param_shape = [type(arg).__name__ for arg in args]
        fingerprint_data = f"{op_name}:{param_shape}"
        return f"sha256:{hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]}"

    async def _emit_ledger_event(self, event: ModelLedgerEventBase) -> None:
        """Emit a ledger event if sink is configured.

        Errors during emission are logged but not propagated (ledger is
        observability, not critical path).
        """
        if self._ledger_sink is None:
            return
        try:
            await self._ledger_sink.emit(event)
        except Exception as e:
            logger.warning(f"Failed to emit ledger event: {e}", exc_info=True)

    def _is_retriable_error(self, exc: Exception) -> bool:
        """Determine if a database error is retriable.

        Connection errors and transient failures are retriable.
        Constraint violations, syntax errors, etc. are NOT retriable.

        Args:
            exc: The exception to classify.

        Returns:
            True if the error is transient and retriable, False otherwise.

        Note:
            Uses isinstance() for built-in exceptions but string-based checking
            for asyncpg exceptions because asyncpg is imported under TYPE_CHECKING.
            This avoids runtime import requirements when asyncpg may not be installed.
        """
        # Check built-in exceptions first (use isinstance for type safety)
        if isinstance(
            exc,
            (
                TimeoutError,
                ConnectionRefusedError,
                ConnectionResetError,
                BrokenPipeError,
                OSError,
            ),
        ):
            return True

        # asyncpg-specific exceptions (string-based to avoid import dependency)
        # asyncpg is only imported under TYPE_CHECKING for type hints
        error_type = type(exc).__name__
        asyncpg_retriable_types = {
            "ConnectionDoesNotExistError",
            "InterfaceError",
            "InterfaceWarning",
            "CannotConnectNowError",
            "TooManyConnectionsError",
        }

        if error_type in asyncpg_retriable_types:
            return True

        # Fallback: check error message for connection-related keywords
        error_str = str(exc).lower()
        retriable_keywords = [
            "connection",
            "connect",
            "network",
            "timeout",
            "temporarily unavailable",
            "too many connections",
            "server closed",
            "broken pipe",
        ]

        if any(keyword in error_str for keyword in retriable_keywords):
            return True

        # Default: not retriable (constraint violations, syntax errors, etc.)
        return False

    async def call(
        self, op_name: str, *args: object, correlation_id: UUID | None = None
    ) -> list[dict[str, object]] | dict[str, object] | None:
        """Execute a named operation from the contract.

        Validates the operation exists, checks allowed operations,
        validates argument count, applies determinism and limit
        constraints, and executes with timeout enforcement.

        If a ledger_sink is configured, emits traceability events:
        - db.query.requested: At entry, before execution
        - db.query.succeeded: On successful completion
        - db.query.failed: On any exception

        Args:
            op_name: Operation name as defined in contract.ops.
            *args: Positional arguments matching contract params order.
            correlation_id: Optional correlation ID for distributed tracing.
                If not provided, one is auto-generated.

        Returns:
            For many=True: list of dicts (possibly empty)
            For many=False: single dict or None if no row found

        Raises:
            RepositoryContractError: Operation not found, forbidden mode,
                or determinism constraint violation (no PK for multi-row).
            RepositoryValidationError: Argument count mismatch.
            RepositoryExecutionError: Database execution error.
            RepositoryTimeoutError: Query exceeded timeout.

        Example:
            >>> # Single row lookup
            >>> user = await runtime.call("find_by_id", 123)
            >>> # Multi-row query
            >>> users = await runtime.call("find_by_status", "active")
        """
        start_time = time.monotonic()

        # Generate or use provided correlation_id (must be before context creation)
        corr_id = correlation_id or uuid4()

        # Create error context with correlation_id for distributed tracing
        context = self._create_error_context(op_name, correlation_id=corr_id)

        # Lookup operation in contract
        operation = self._get_operation(op_name, context)

        # Validate operation is allowed
        self._validate_operation_allowed(operation, op_name, context)

        # Validate argument count
        self._validate_arg_count(operation, args, op_name, context)

        # Build final SQL with determinism and limit constraints
        sql = self._build_sql(operation, op_name, context)

        # Emit db.query.requested event
        if self._ledger_sink is not None:
            requested_event = ModelDbQueryRequested(
                event_id=uuid4(),
                correlation_id=corr_id,
                idempotency_key=ModelLedgerEventBase.build_idempotency_key(
                    corr_id, op_name, "db.query.requested"
                ),
                contract_id=self._contract.name,
                contract_fingerprint=self._contract_fingerprint,
                operation_name=op_name,
                query_fingerprint=self._compute_query_fingerprint(op_name, args),
                emitted_at=datetime.now(UTC),
            )
            await self._emit_ledger_event(requested_event)

        # Execute with timeout
        try:
            result = await self._execute_with_timeout(
                sql, args, operation, op_name, context
            )
        except TimeoutError as e:
            # Emit failure event before re-raising
            elapsed_ms = (time.monotonic() - start_time) * 1000
            if self._ledger_sink is not None:
                failed_event = ModelDbQueryFailed(
                    event_id=uuid4(),
                    correlation_id=corr_id,
                    idempotency_key=ModelLedgerEventBase.build_idempotency_key(
                        corr_id, op_name, "db.query.failed"
                    ),
                    contract_id=self._contract.name,
                    contract_fingerprint=self._contract_fingerprint,
                    operation_name=op_name,
                    emitted_at=datetime.now(UTC),
                    duration_ms=elapsed_ms,
                    error_type="RepositoryTimeoutError",
                    error_message=f"Query exceeded timeout of {self._config.timeout_ms}ms",
                    retriable=True,
                )
                await self._emit_ledger_event(failed_event)

            timeout_seconds = self._config.timeout_ms / 1000.0
            raise RepositoryTimeoutError(
                f"Query '{op_name}' exceeded timeout of {timeout_seconds}s",
                op_name=op_name,
                table=self._get_primary_table(),
                timeout_seconds=timeout_seconds,
                sql_fingerprint=self._fingerprint_sql(sql),
                context=context,
            ) from e
        except Exception as e:
            # Emit failure event for any other exception
            elapsed_ms = (time.monotonic() - start_time) * 1000
            if self._ledger_sink is not None:
                # Classify error as retriable based on error type/message
                retriable = self._is_retriable_error(e)
                failed_event = ModelDbQueryFailed(
                    event_id=uuid4(),
                    correlation_id=corr_id,
                    idempotency_key=ModelLedgerEventBase.build_idempotency_key(
                        corr_id, op_name, "db.query.failed"
                    ),
                    contract_id=self._contract.name,
                    contract_fingerprint=self._contract_fingerprint,
                    operation_name=op_name,
                    emitted_at=datetime.now(UTC),
                    duration_ms=elapsed_ms,
                    error_type=type(e).__name__,
                    error_message=sanitize_error_string(str(e), max_length=200),
                    retriable=retriable,
                )
                await self._emit_ledger_event(failed_event)
            raise

        # Calculate metrics
        elapsed_ms = (time.monotonic() - start_time) * 1000
        row_count = len(result) if isinstance(result, list) else (1 if result else 0)

        # Emit db.query.succeeded event
        if self._ledger_sink is not None:
            succeeded_event = ModelDbQuerySucceeded(
                event_id=uuid4(),
                correlation_id=corr_id,
                idempotency_key=ModelLedgerEventBase.build_idempotency_key(
                    corr_id, op_name, "db.query.succeeded"
                ),
                contract_id=self._contract.name,
                contract_fingerprint=self._contract_fingerprint,
                operation_name=op_name,
                emitted_at=datetime.now(UTC),
                duration_ms=elapsed_ms,
                rows_returned=row_count,
            )
            await self._emit_ledger_event(succeeded_event)

        # Log metrics if enabled
        if self._config.emit_metrics:
            logger.info(
                "Repository operation completed",
                extra={
                    "op_name": op_name,
                    "duration_ms": round(elapsed_ms, 2),
                    "rows_returned": row_count,
                    "repository": self._contract.name,
                },
            )

        return result

    def _create_error_context(
        self, op_name: str, correlation_id: UUID | None = None
    ) -> ModelInfraErrorContext:
        """Create error context for infrastructure errors.

        Args:
            op_name: Operation name for context.
            correlation_id: Optional correlation ID for distributed tracing.
                If provided, uses this ID. If None, auto-generates one.

        Returns:
            Error context with correlation ID for tracing.
        """
        return ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.DATABASE,
            operation=f"repository.{op_name}",
            target_name=self._contract.name,
        )

    def _get_operation(
        self, op_name: str, context: ModelInfraErrorContext
    ) -> ModelDbOperation:
        """Get operation from contract, raising error if not found."""
        operation = self._contract.ops.get(op_name)
        if operation is None:
            available_ops = list(self._contract.ops.keys())
            raise RepositoryContractError(
                f"Unknown operation '{op_name}' not defined in contract '{self._contract.name}'. "
                f"Available operations: {available_ops}",
                op_name=op_name,
                table=self._get_primary_table(),
                context=context,
            )
        return operation

    def _validate_operation_allowed(
        self,
        operation: ModelDbOperation,
        op_name: str,
        context: ModelInfraErrorContext,
    ) -> None:
        """Validate operation mode is allowed by config.

        The contract uses 'read' or 'write' modes (validated by omnibase_core
        validators at contract load time to ensure SQL verb matching).
        """
        mode = operation.mode

        # Check write operations against feature flag
        if mode == "write" and not self._config.allow_write_operations:
            raise RepositoryContractError(
                f"Operation '{op_name}' uses 'write' mode which is disabled. "
                "Set allow_write_operations=True in config to enable.",
                op_name=op_name,
                table=self._get_primary_table(),
                context=context,
            )

        # Check mode against allowlist
        if mode not in self._config.allowed_modes:
            raise RepositoryContractError(
                f"Operation mode '{mode}' for '{op_name}' is not in allowed_modes. "
                f"Allowed: {set(self._config.allowed_modes)}",
                op_name=op_name,
                table=self._get_primary_table(),
                context=context,
            )

    def _validate_arg_count(
        self,
        operation: ModelDbOperation,
        args: tuple[object, ...],
        op_name: str,
        context: ModelInfraErrorContext,
    ) -> None:
        """Validate argument count matches contract params.

        Contract params is a dict[str, ModelDbParam] where keys are param names.
        """
        param_names = list(operation.params.keys())
        expected = len(param_names)
        actual = len(args)
        if actual != expected:
            raise RepositoryValidationError(
                f"Operation '{op_name}' expects {expected} argument(s) ({param_names}), "
                f"but received {actual}",
                op_name=op_name,
                table=self._get_primary_table(),
                context=context,
                expected_args=expected,
                actual_args=actual,
                param_names=param_names,
            )

    def _build_sql(
        self,
        operation: ModelDbOperation,
        op_name: str,
        context: ModelInfraErrorContext,
    ) -> str:
        """Build final SQL with determinism and limit constraints.

        Applies ORDER BY injection for multi-row queries without ORDER BY.
        Applies LIMIT injection or validation based on config.

        Only applies constraints to 'read' mode operations (SELECT).
        """
        sql = operation.sql
        is_read = operation.mode == "read"
        is_multi_row = operation.returns.many

        # Only apply constraints to read operations
        if not is_read:
            return sql

        # Apply determinism constraints for multi-row reads
        if is_multi_row:
            sql = self._inject_order_by(sql, op_name, context)

        # Apply limit constraints for multi-row reads
        if is_multi_row:
            sql = self._inject_limit(sql, op_name, context)

        return sql

    def _inject_order_by(
        self,
        sql: str,
        op_name: str,
        context: ModelInfraErrorContext,
    ) -> str:
        """Inject ORDER BY clause for deterministic multi-row results.

        Rules:
            - If ORDER BY exists: no injection needed
            - If no ORDER BY and PK declared: inject ORDER BY {pk}
            - If no ORDER BY and no PK: HARD ERROR

        When injecting ORDER BY, the clause is inserted BEFORE any existing
        LIMIT clause to produce valid SQL. For example:
            - Input:  "SELECT * FROM users LIMIT $1"
            - Output: "SELECT * FROM users ORDER BY id LIMIT $1"

        Args:
            sql: The SQL query to potentially modify.
            op_name: Operation name for error context.
            context: Error context for exception raising.

        Returns:
            SQL with ORDER BY clause (injected or original).

        Raises:
            RepositoryContractError: No ORDER BY and no primary_key_column.
        """
        has_order_by = bool(_ORDER_BY_PATTERN.search(sql))
        if has_order_by:
            return sql

        # No ORDER BY - check if we can inject
        pk_column = self._config.primary_key_column
        if pk_column is None:
            raise RepositoryContractError(
                f"Multi-row query '{op_name}' has no ORDER BY clause and "
                "primary_key_column is not configured. Deterministic results "
                "cannot be guaranteed. Either add ORDER BY to the SQL or "
                "set primary_key_column in config.",
                op_name=op_name,
                table=self._get_primary_table(),
                sql_fingerprint=self._fingerprint_sql(sql),
                context=context,
            )

        # Inject ORDER BY using configured order or just PK
        order_by = self._config.default_order_by or pk_column

        # Check if LIMIT exists - ORDER BY must be inserted BEFORE LIMIT
        param_match = _LIMIT_PARAM_PATTERN.search(sql)
        numeric_match = _LIMIT_NUMERIC_PATTERN.search(sql)
        limit_match = param_match or numeric_match

        if limit_match:
            # Insert ORDER BY before LIMIT
            limit_start = limit_match.start()
            return (
                f"{sql[:limit_start].rstrip()} ORDER BY {order_by} {sql[limit_start:]}"
            )
        else:
            # No LIMIT, append at end
            return f"{sql.rstrip().rstrip(';')} ORDER BY {order_by}"

    def _inject_limit(
        self,
        sql: str,
        op_name: str,
        context: ModelInfraErrorContext,
    ) -> str:
        """Inject or validate LIMIT clause for multi-row results.

        Rules:
            - If parameterized LIMIT (e.g., $1): OK (no change, can't validate at build time)
            - If numeric LIMIT > max_row_limit: HARD ERROR
            - If numeric LIMIT <= max_row_limit: OK (no change)
            - If no LIMIT: inject LIMIT {max_row_limit}

        Args:
            sql: The SQL query to potentially modify.
            op_name: Operation name for error context.
            context: Error context for exception raising.

        Returns:
            SQL with LIMIT clause (injected or original).

        Raises:
            RepositoryContractError: Numeric LIMIT exceeds max_row_limit.
        """
        max_limit = self._config.max_row_limit

        # Check for parameterized LIMIT (e.g., LIMIT $1) - can't validate at build time
        if _LIMIT_PARAM_PATTERN.search(sql):
            return sql

        # Check for numeric LIMIT (e.g., LIMIT 100) - can validate
        limit_match = _LIMIT_NUMERIC_PATTERN.search(sql)
        if limit_match:
            # Existing numeric LIMIT - validate it
            existing_limit = int(limit_match.group(1))
            if existing_limit > max_limit:
                raise RepositoryContractError(
                    f"Query '{op_name}' has LIMIT {existing_limit} which exceeds "
                    f"max_row_limit of {max_limit}. Reduce the LIMIT or increase "
                    "max_row_limit in config.",
                    op_name=op_name,
                    table=self._get_primary_table(),
                    sql_fingerprint=self._fingerprint_sql(sql),
                    context=context,
                    existing_limit=existing_limit,
                    max_row_limit=max_limit,
                )
            return sql

        # No LIMIT - inject one
        return f"{sql.rstrip().rstrip(';')} LIMIT {max_limit}"

    async def _execute_with_timeout(
        self,
        sql: str,
        args: tuple[object, ...],
        operation: ModelDbOperation,
        op_name: str,
        context: ModelInfraErrorContext,
    ) -> list[dict[str, object]] | dict[str, object] | None:
        """Execute query with timeout enforcement.

        Uses asyncio.wait_for() to enforce timeout.
        Uses fetch() for many=True, fetchrow() for many=False.

        Args:
            sql: Final SQL query to execute.
            args: Positional arguments for the query.
            operation: Operation specification.
            op_name: Operation name for error context.
            context: Error context for exception raising.

        Returns:
            Query results as appropriate type.

        Raises:
            asyncio.TimeoutError: Query exceeded timeout (caught by caller).
            RepositoryExecutionError: Database execution error.
        """
        timeout_seconds = self._config.timeout_ms / 1000.0

        try:
            async with self._pool.acquire() as conn:
                if operation.returns.many:
                    # Multi-row: use fetch()
                    coro = conn.fetch(sql, *args)
                    records = await asyncio.wait_for(coro, timeout=timeout_seconds)
                    return [dict(record) for record in records]
                else:
                    # Single-row: use fetchrow()
                    coro = conn.fetchrow(sql, *args)
                    record = await asyncio.wait_for(coro, timeout=timeout_seconds)
                    return dict(record) if record is not None else None
        except TimeoutError:
            # Re-raise for caller to handle
            raise
        except Exception as e:
            # Wrap all other exceptions
            raise RepositoryExecutionError(
                f"Failed to execute operation '{op_name}': {e}",
                op_name=op_name,
                table=self._get_primary_table(),
                sql_fingerprint=self._fingerprint_sql(sql),
                context=context,
            ) from e

    def _get_primary_table(self) -> str | None:
        """Get the primary table from contract for error context."""
        return self._contract.tables[0] if self._contract.tables else None

    def _fingerprint_sql(self, sql: str) -> str:
        """Create a safe fingerprint of SQL for logging/errors.

        Truncates long SQL and removes potentially sensitive values.
        """
        # Simple approach: truncate to reasonable length
        max_len = 200
        if len(sql) <= max_len:
            return sql
        return sql[:max_len] + "..."


__all__: list[str] = ["PostgresRepositoryRuntime"]
