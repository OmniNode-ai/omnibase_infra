# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for PostgreSQL contract record upsert.

This handler encapsulates PostgreSQL-specific persistence logic for the
NodeContractPersistenceEffect node, following the declarative node pattern where
handlers are extracted for testability and separation of concerns.

Architecture:
    HandlerPostgresContractUpsert is responsible for:
    - Executing upsert operations against the PostgreSQL contracts table
    - Serializing contract_yaml dict to YAML string before INSERT
    - Timing operation duration for observability
    - Sanitizing error messages before inclusion in results
    - Returning structured ModelBackendResult

    This extraction supports the declarative node pattern where
    NodeContractPersistenceEffect delegates backend-specific operations
    to dedicated handlers.

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different payload instances. Thread-safety depends on the
    underlying asyncpg connection pool implementation.

SQL Security:
    All SQL queries use parameterized queries with positional placeholders
    ($1, $2, etc.) to prevent SQL injection attacks. The asyncpg library
    handles proper escaping and type conversion for all parameters.

Related:
    - NodeContractPersistenceEffect: Parent effect node that coordinates handlers
    - ModelPayloadUpsertContract: Input payload model
    - ModelBackendResult: Structured result model for backend operations
    - OMN-1845: Implementation ticket
    - OMN-1653: ContractRegistryReducer ticket (source of intents)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID

import yaml

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumPostgresErrorCode,
)
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    RepositoryExecutionError,
)
from omnibase_infra.nodes.effects.models import ModelBackendResult
from omnibase_infra.utils import sanitize_backend_error, sanitize_error_message

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.nodes.contract_registry_reducer.models import (
        ModelPayloadUpsertContract,
    )

logger = logging.getLogger(__name__)

# SQL statement for contract upsert with ON CONFLICT for idempotency.
# Uses RETURNING to confirm the operation was executed.
SQL_UPSERT_CONTRACT = """
INSERT INTO contracts (
    contract_id, node_name, version_major, version_minor, version_patch,
    contract_hash, contract_yaml, is_active, registered_at, last_seen_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (contract_id) DO UPDATE SET
    contract_hash = EXCLUDED.contract_hash,
    contract_yaml = EXCLUDED.contract_yaml,
    is_active = EXCLUDED.is_active,
    last_seen_at = EXCLUDED.last_seen_at,
    updated_at = NOW()
RETURNING contract_id, (xmax = 0) AS was_insert;
"""


class HandlerPostgresContractUpsert:
    """Handler for PostgreSQL contract record upsert.

    Encapsulates all PostgreSQL-specific persistence logic for contract
    record upserts. The handler provides a clean interface for executing
    upsert operations with proper timing and error sanitization.

    This handler never raises exceptions - all errors are captured and
    returned in the ModelBackendResult with appropriate error codes.

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> import asyncpg
        >>> pool = await asyncpg.create_pool(dsn="...")
        >>> handler = HandlerPostgresContractUpsert(pool)
        >>> result = await handler.handle(payload, correlation_id)
        >>> result.success
        True

    See Also:
        - NodeContractPersistenceEffect: Parent node that uses this handler
        - ModelPayloadUpsertContract: Input payload model
        - contract.yaml: Handler routing configuration
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize handler with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool for database operations.
                The pool should be pre-configured and ready for use.
        """
        self._pool = pool

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role of this handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification of this handler."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        payload: ModelPayloadUpsertContract,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute PostgreSQL contract record upsert.

        Performs the upsert operation against the contracts table with:
        - Contract YAML serialization (dict to YAML string)
        - Operation timing via time.perf_counter()
        - Parameterized SQL for injection prevention
        - Error sanitization for security
        - Structured result construction

        Args:
            payload: Upsert contract payload containing all contract fields
                including contract_id, node_name, version components,
                contract_hash, contract_yaml, and timestamps.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with:
                - success: True if upsert completed successfully
                - error: Sanitized error message if failed
                - error_code: Error code for programmatic handling
                - duration_ms: Operation duration in milliseconds
                - backend_id: Set to "postgres"
                - correlation_id: Passed through for tracing

        Note:
            This method never raises exceptions. All errors are captured
            and returned in the result model with appropriate error codes.

            Error messages are sanitized using sanitize_error_message to
            prevent exposure of connection strings, credentials, or other
            sensitive information in logs and responses.
        """
        start_time = time.perf_counter()

        try:
            # Serialize contract_yaml to YAML string if it's a dict
            # The PostgreSQL column is TEXT type, so we need a string representation
            contract_yaml_str: str
            if isinstance(payload.contract_yaml, dict):
                contract_yaml_str = yaml.safe_dump(
                    payload.contract_yaml,
                    default_flow_style=False,
                    sort_keys=True,
                    allow_unicode=True,
                )
            elif isinstance(payload.contract_yaml, str):
                contract_yaml_str = payload.contract_yaml
            else:
                # Handle unexpected types by converting to string representation
                contract_yaml_str = str(payload.contract_yaml)

            async with self._pool.acquire() as conn:
                result = await conn.fetchrow(
                    SQL_UPSERT_CONTRACT,
                    payload.contract_id,
                    payload.node_name,
                    payload.version_major,
                    payload.version_minor,
                    payload.version_patch,
                    payload.contract_hash,
                    contract_yaml_str,
                    payload.is_active,
                    payload.registered_at,
                    payload.last_seen_at,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if result is not None:
                was_insert = result["was_insert"]
                operation = "insert" if was_insert else "update"
                logger.info(
                    "Contract upsert completed",
                    extra={
                        "contract_id": payload.contract_id,
                        "node_name": payload.node_name,
                        "operation": operation,
                        "duration_ms": duration_ms,
                        "correlation_id": str(correlation_id),
                    },
                )
                return ModelBackendResult(
                    success=True,
                    duration_ms=duration_ms,
                    backend_id="postgres",
                    correlation_id=correlation_id,
                )
            else:
                # RETURNING clause should always return a row on success
                # If None, something unexpected happened
                logger.warning(
                    "Contract upsert returned no result",
                    extra={
                        "contract_id": payload.contract_id,
                        "duration_ms": duration_ms,
                        "correlation_id": str(correlation_id),
                    },
                )
                return ModelBackendResult(
                    success=False,
                    error="postgres operation failed: no result returned",
                    error_code=EnumPostgresErrorCode.UPSERT_ERROR,
                    duration_ms=duration_ms,
                    backend_id="postgres",
                    correlation_id=correlation_id,
                )

        except (TimeoutError, InfraTimeoutError) as e:
            # Timeout during upsert - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            logger.warning(
                "Contract upsert timed out",
                extra={
                    "contract_id": payload.contract_id,
                    "error": sanitized_error,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.TIMEOUT_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraAuthenticationError as e:
            # Authentication failure - non-retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            logger.exception(
                "Contract upsert authentication failed",
                extra={
                    "contract_id": payload.contract_id,
                    "error": sanitized_error,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.AUTH_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraConnectionError as e:
            # Connection failure - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            logger.warning(
                "Contract upsert connection failed",
                extra={
                    "contract_id": payload.contract_id,
                    "error": sanitized_error,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.CONNECTION_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except RepositoryExecutionError as e:
            # Query execution error - may be retriable
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            logger.warning(
                "Contract upsert execution failed",
                extra={
                    "contract_id": payload.contract_id,
                    "error": sanitized_error,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.UPSERT_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except (
            Exception
        ) as e:  # ONEX: catch-all - database adapter may raise unexpected exceptions
            # beyond typed infrastructure errors (e.g., driver errors, encoding errors,
            # connection pool errors, asyncpg-specific exceptions).
            # Required to sanitize errors and prevent credential exposure.
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_backend_error("postgres", e)
            logger.exception(
                "Contract upsert failed with unexpected error",
                extra={
                    "contract_id": payload.contract_id,
                    "error_type": type(e).__name__,
                    "error": sanitized_error,
                    "duration_ms": duration_ms,
                    "correlation_id": str(correlation_id),
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.UNKNOWN_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )


__all__: list[str] = ["HandlerPostgresContractUpsert"]
