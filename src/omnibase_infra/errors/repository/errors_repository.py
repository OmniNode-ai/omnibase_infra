# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Repository Error Classes for PostgresRepositoryRuntime.

This module defines error classes specific to repository operations,
providing granular error handling for contract-driven database access.

Error Hierarchy:
    RuntimeHostError (from error_infra)
    └── RepositoryError (base repository error)
        ├── RepositoryContractError (contract-level errors)
        ├── RepositoryValidationError (validation errors)
        ├── RepositoryExecutionError (execution errors)
        └── RepositoryTimeoutError (query timeout errors)

All errors:
    - Extend RuntimeHostError for infrastructure consistency
    - Include repository-specific fields: op_name, table, retriable
    - Support optional sql_fingerprint for query tracking
    - Use EnumCoreErrorCode for error classification
    - Support correlation IDs for distributed tracing

Retriability Guidelines:
    - RepositoryContractError: NOT retriable (contract/configuration issue)
    - RepositoryValidationError: NOT retriable (data validation issue)
    - RepositoryExecutionError: Generally retriable (transient failures)
    - RepositoryTimeoutError: Retriable (timeout may be transient)

Example::

    from omnibase_infra.errors.repository import (
        RepositoryContractError,
        RepositoryExecutionError,
        RepositoryTimeoutError,
        RepositoryValidationError,
    )
    from omnibase_infra.models.errors import ModelInfraErrorContext
    from omnibase_infra.enums import EnumInfraTransportType

    # Contract error - unknown operation
    context = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.DATABASE,
        operation="execute_operation",
    )
    raise RepositoryContractError(
        "Unknown operation 'invalid_op' not defined in contract",
        op_name="invalid_op",
        table="users",
        context=context,
    )

    # Execution error with SQL fingerprint
    raise RepositoryExecutionError(
        "Connection pool exhausted",
        op_name="find_by_id",
        table="users",
        sql_fingerprint="SELECT * FROM users WHERE id = $1",
        context=context,
    ) from e
"""

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class RepositoryError(RuntimeHostError):
    """Base error class for all repository operations.

    Provides common structured fields for repository-specific errors:
        - op_name: The operation name from the repository contract
        - table: The target table for the operation
        - retriable: Whether the operation can be retried
        - sql_fingerprint: Optional SQL fingerprint for query tracking

    Subclasses set default retriability based on error category:
        - Contract errors: NOT retriable
        - Validation errors: NOT retriable
        - Execution errors: Generally retriable
        - Timeout errors: Retriable

    Example:
        >>> context = ModelInfraErrorContext.with_correlation(
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ...     operation="execute_operation",
        ... )
        >>> raise RepositoryError(
        ...     "Repository operation failed",
        ...     op_name="find_by_id",
        ...     table="users",
        ...     context=context,
        ... )
    """

    # Default retriability for base class (conservative: not retriable)
    _default_retriable: bool = False

    def __init__(
        self,
        message: str,
        *,
        op_name: str | None = None,
        table: str | None = None,
        retriable: bool | None = None,
        sql_fingerprint: str | None = None,
        error_code: EnumCoreErrorCode | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize RepositoryError with repository-specific fields.

        Args:
            message: Human-readable error message
            op_name: Operation name from the repository contract
            table: Target table for the operation
            retriable: Whether the operation can be retried. If None, uses
                class default (_default_retriable)
            sql_fingerprint: SQL fingerprint for query tracking (sanitized)
            error_code: Error code (defaults to DATABASE_OPERATION_ERROR)
            context: Bundled infrastructure context
            **extra_context: Additional context information
        """
        # Add repository-specific fields to extra_context
        if op_name is not None:
            extra_context["op_name"] = op_name
        if table is not None:
            extra_context["table"] = table
        if sql_fingerprint is not None:
            extra_context["sql_fingerprint"] = sql_fingerprint

        # Resolve retriability: explicit > class default
        resolved_retriable = (
            retriable if retriable is not None else self._default_retriable
        )
        extra_context["retriable"] = resolved_retriable

        # Store as instance attributes for programmatic access
        self.op_name = op_name
        self.table = table
        self.retriable = resolved_retriable
        self.sql_fingerprint = sql_fingerprint

        super().__init__(
            message=message,
            error_code=error_code or EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
            context=context,
            **extra_context,
        )


class RepositoryContractError(RepositoryError):
    """Raised for contract-level errors in repository operations.

    Used when:
        - Operation name (op_name) is not defined in the contract
        - Required parameters are missing for the operation
        - Operation is explicitly forbidden in the contract
        - Contract schema validation fails

    Contract errors are NOT retriable - they indicate a configuration
    or programming error that requires code or contract changes to fix.

    Example:
        >>> context = ModelInfraErrorContext.with_correlation(
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ...     operation="execute_operation",
        ... )
        >>> raise RepositoryContractError(
        ...     "Operation 'drop_table' is forbidden in contract",
        ...     op_name="drop_table",
        ...     table="users",
        ...     context=context,
        ... )
    """

    _default_retriable: bool = False

    def __init__(
        self,
        message: str,
        *,
        op_name: str | None = None,
        table: str | None = None,
        retriable: bool | None = None,
        sql_fingerprint: str | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize RepositoryContractError.

        Args:
            message: Human-readable error message
            op_name: Operation name that caused the contract error
            table: Target table (if applicable)
            retriable: Override default (False). Contract errors are
                generally NOT retriable.
            sql_fingerprint: SQL fingerprint (if applicable)
            context: Bundled infrastructure context
            **extra_context: Additional context information
        """
        super().__init__(
            message=message,
            op_name=op_name,
            table=table,
            retriable=retriable,
            sql_fingerprint=sql_fingerprint,
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
            context=context,
            **extra_context,
        )


class RepositoryValidationError(RepositoryError):
    """Raised for validation errors in repository operations.

    Used when:
        - Parameter type mismatches (e.g., string passed for int column)
        - Constraint violations (e.g., null for non-nullable column)
        - Value out of allowed range
        - Invalid data format

    Validation errors are NOT retriable - they indicate invalid input
    data that must be corrected before retrying.

    Example:
        >>> context = ModelInfraErrorContext.with_correlation(
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ...     operation="execute_operation",
        ... )
        >>> raise RepositoryValidationError(
        ...     "Parameter 'user_id' expected int, got str",
        ...     op_name="find_by_id",
        ...     table="users",
        ...     context=context,
        ...     param_name="user_id",
        ...     expected_type="int",
        ...     actual_type="str",
        ... )
    """

    _default_retriable: bool = False

    def __init__(
        self,
        message: str,
        *,
        op_name: str | None = None,
        table: str | None = None,
        retriable: bool | None = None,
        sql_fingerprint: str | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize RepositoryValidationError.

        Args:
            message: Human-readable error message
            op_name: Operation name that triggered validation
            table: Target table (if applicable)
            retriable: Override default (False). Validation errors are
                generally NOT retriable.
            sql_fingerprint: SQL fingerprint (if applicable)
            context: Bundled infrastructure context
            **extra_context: Additional context (param_name, expected_type, etc.)
        """
        super().__init__(
            message=message,
            op_name=op_name,
            table=table,
            retriable=retriable,
            sql_fingerprint=sql_fingerprint,
            error_code=EnumCoreErrorCode.VALIDATION_ERROR,
            context=context,
            **extra_context,
        )


class RepositoryExecutionError(RepositoryError):
    """Raised for execution errors in repository operations.

    Used when:
        - asyncpg errors during query execution
        - Connection pool exhaustion
        - Database connection lost mid-operation
        - Deadlock detected
        - Serialization failures

    Execution errors are generally retriable - they often indicate
    transient failures that may succeed on retry with backoff.

    Example:
        >>> context = ModelInfraErrorContext.with_correlation(
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ...     operation="execute_operation",
        ... )
        >>> try:
        ...     result = await pool.execute(sql, *params)
        ... except asyncpg.PostgresError as e:
        ...     raise RepositoryExecutionError(
        ...         f"Query execution failed: {e}",
        ...         op_name="create_user",
        ...         table="users",
        ...         sql_fingerprint="INSERT INTO users (...) VALUES (...)",
        ...         context=context,
        ...     ) from e
    """

    _default_retriable: bool = True

    def __init__(
        self,
        message: str,
        *,
        op_name: str | None = None,
        table: str | None = None,
        retriable: bool | None = None,
        sql_fingerprint: str | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize RepositoryExecutionError.

        Args:
            message: Human-readable error message
            op_name: Operation name that failed during execution
            table: Target table (if applicable)
            retriable: Override default (True). Set False for non-retriable
                execution errors (e.g., constraint violations from DB).
            sql_fingerprint: SQL fingerprint for query tracking
            context: Bundled infrastructure context
            **extra_context: Additional context (asyncpg error details, etc.)
        """
        super().__init__(
            message=message,
            op_name=op_name,
            table=table,
            retriable=retriable,
            sql_fingerprint=sql_fingerprint,
            error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
            context=context,
            **extra_context,
        )


class RepositoryTimeoutError(RepositoryError):
    """Raised when a repository query exceeds its timeout.

    Used when:
        - Query exceeds statement_timeout
        - Connection acquisition times out
        - Transaction times out

    Timeout errors are retriable - the same query may succeed
    under different load conditions or with adjusted timeouts.

    Example:
        >>> context = ModelInfraErrorContext.with_correlation(
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ...     operation="execute_operation",
        ... )
        >>> raise RepositoryTimeoutError(
        ...     "Query exceeded 30s timeout",
        ...     op_name="complex_report",
        ...     table="analytics",
        ...     timeout_seconds=30.0,
        ...     sql_fingerprint="SELECT ... FROM analytics ...",
        ...     context=context,
        ... )
    """

    _default_retriable: bool = True

    def __init__(
        self,
        message: str,
        *,
        op_name: str | None = None,
        table: str | None = None,
        retriable: bool | None = None,
        sql_fingerprint: str | None = None,
        timeout_seconds: float | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize RepositoryTimeoutError.

        Args:
            message: Human-readable error message
            op_name: Operation name that timed out
            table: Target table (if applicable)
            retriable: Override default (True). Timeout errors are
                generally retriable.
            sql_fingerprint: SQL fingerprint for query tracking
            timeout_seconds: The timeout value that was exceeded
            context: Bundled infrastructure context
            **extra_context: Additional context information
        """
        if timeout_seconds is not None:
            extra_context["timeout_seconds"] = timeout_seconds

        # Store for programmatic access
        self.timeout_seconds = timeout_seconds

        super().__init__(
            message=message,
            op_name=op_name,
            table=table,
            retriable=retriable,
            sql_fingerprint=sql_fingerprint,
            error_code=EnumCoreErrorCode.TIMEOUT_ERROR,
            context=context,
            **extra_context,
        )


__all__ = [
    "RepositoryContractError",
    "RepositoryError",
    "RepositoryExecutionError",
    "RepositoryTimeoutError",
    "RepositoryValidationError",
]
