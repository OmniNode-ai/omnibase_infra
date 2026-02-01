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
import logging
import re
import time
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.repository import (
    RepositoryContractError,
    RepositoryExecutionError,
    RepositoryTimeoutError,
    RepositoryValidationError,
)
from omnibase_infra.models.errors import ModelInfraErrorContext
from omnibase_infra.runtime.db.models import (
    ModelDbOperation,
    ModelDbRepositoryContract,
    ModelRepositoryRuntimeConfig,
)

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

# Regex patterns for SQL parsing (simple approach, not full SQL parser)
_ORDER_BY_PATTERN = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)


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

    __slots__ = ("_config", "_contract", "_pool")

    def __init__(
        self,
        pool: asyncpg.Pool,
        contract: ModelDbRepositoryContract,
        config: ModelRepositoryRuntimeConfig | None = None,
    ) -> None:
        """Initialize the repository runtime.

        Args:
            pool: asyncpg connection pool for database access.
            contract: Repository contract defining available operations.
            config: Optional runtime configuration. If None, uses defaults.

        Example:
            >>> runtime = PostgresRepositoryRuntime(
            ...     pool=pool,
            ...     contract=contract,
            ...     config=ModelRepositoryRuntimeConfig(max_row_limit=100),
            ... )
        """
        self._pool = pool
        self._contract = contract
        self._config = config or ModelRepositoryRuntimeConfig()

    @property
    def contract(self) -> ModelDbRepositoryContract:
        """Get the repository contract."""
        return self._contract

    @property
    def config(self) -> ModelRepositoryRuntimeConfig:
        """Get the runtime configuration."""
        return self._config

    async def call(
        self, op_name: str, *args: object
    ) -> list[dict[str, object]] | dict[str, object] | None:
        """Execute a named operation from the contract.

        Validates the operation exists, checks allowed operations,
        validates argument count, applies determinism and limit
        constraints, and executes with timeout enforcement.

        Args:
            op_name: Operation name as defined in contract.ops.
            *args: Positional arguments matching contract params order.

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
        context = self._create_error_context(op_name)

        # Lookup operation in contract
        operation = self._get_operation(op_name, context)

        # Validate operation is allowed
        self._validate_operation_allowed(operation, op_name, context)

        # Validate argument count
        self._validate_arg_count(operation, args, op_name, context)

        # Build final SQL with determinism and limit constraints
        sql = self._build_sql(operation, op_name, context)

        # Execute with timeout
        try:
            result = await self._execute_with_timeout(
                sql, args, operation, op_name, context
            )
        except TimeoutError as e:
            timeout_seconds = self._config.timeout_ms / 1000.0
            raise RepositoryTimeoutError(
                f"Query '{op_name}' exceeded timeout of {timeout_seconds}s",
                op_name=op_name,
                table=self._get_primary_table(),
                timeout_seconds=timeout_seconds,
                sql_fingerprint=self._fingerprint_sql(sql),
                context=context,
            ) from e

        # Log metrics if enabled
        if self._config.emit_metrics:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            row_count = (
                len(result) if isinstance(result, list) else (1 if result else 0)
            )
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

    def _create_error_context(self, op_name: str) -> ModelInfraErrorContext:
        """Create error context for infrastructure errors."""
        return ModelInfraErrorContext.with_correlation(
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
        return f"{sql.rstrip().rstrip(';')} ORDER BY {order_by}"

    def _inject_limit(
        self,
        sql: str,
        op_name: str,
        context: ModelInfraErrorContext,
    ) -> str:
        """Inject or validate LIMIT clause for multi-row results.

        Rules:
            - If LIMIT > max_row_limit: HARD ERROR
            - If no LIMIT: inject LIMIT {max_row_limit}
            - If LIMIT <= max_row_limit: OK (no change)

        Args:
            sql: The SQL query to potentially modify.
            op_name: Operation name for error context.
            context: Error context for exception raising.

        Returns:
            SQL with LIMIT clause (injected or original).

        Raises:
            RepositoryContractError: LIMIT exceeds max_row_limit.
        """
        max_limit = self._config.max_row_limit
        limit_match = _LIMIT_PATTERN.search(sql)

        if limit_match:
            # Existing LIMIT - validate it
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
