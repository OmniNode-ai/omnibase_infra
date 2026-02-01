# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Repository Error Classes Module.

This module provides error classes specific to repository operations,
enabling typed exception handling for the PostgresRepositoryRuntime.

Error Hierarchy:
    RuntimeHostError (from omnibase_infra.errors)
    └── RepositoryError (base repository error)
        ├── RepositoryContractError (contract-level errors)
        ├── RepositoryValidationError (validation errors)
        ├── RepositoryExecutionError (execution errors)
        └── RepositoryTimeoutError (query timeout errors)

Exports:
    RepositoryError: Base error for all repository operations
    RepositoryContractError: Bad op_name, missing params, forbidden op
    RepositoryValidationError: Param type mismatch, constraint violation
    RepositoryExecutionError: asyncpg errors, connection issues
    RepositoryTimeoutError: Query timeout exceeded

Common Fields (all exceptions):
    op_name: str | None - Operation name from repository contract
    table: str | None - Target table for the operation
    retriable: bool - Whether the operation can be retried
    sql_fingerprint: str | None - SQL fingerprint for query tracking

Example Usage::

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
        "Unknown operation 'invalid_op'",
        op_name="invalid_op",
        table="users",
        context=context,
    )

    # Check retriability in error handling
    try:
        result = await runtime.execute("find_all", {})
    except RepositoryError as e:
        if e.retriable:
            # Can retry with backoff
            pass
        else:
            # Non-retriable, propagate
            raise
"""

from omnibase_infra.errors.repository.errors_repository import (
    RepositoryContractError,
    RepositoryError,
    RepositoryExecutionError,
    RepositoryTimeoutError,
    RepositoryValidationError,
)

__all__ = [
    "RepositoryContractError",
    "RepositoryError",
    "RepositoryExecutionError",
    "RepositoryTimeoutError",
    "RepositoryValidationError",
]
