# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for PostgreSQL registration record deactivation.

This handler encapsulates PostgreSQL-specific deactivation logic for the
NodeRegistryEffect node, following the declarative node pattern where handlers
are extracted for testability and separation of concerns.

Architecture:
    HandlerPostgresDeactivate is responsible for:
    - Executing deactivation operations against the PostgreSQL adapter
    - Timing operation duration for observability
    - Sanitizing error messages before inclusion in results
    - Returning structured ModelBackendResult

    The deactivation operation performs a soft delete by marking the registration
    record as inactive, preserving historical data while removing the node from
    active service discovery.

    This extraction supports the declarative node pattern where NodeRegistryEffect
    delegates backend-specific operations to dedicated handlers.

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different request instances. Thread-safety depends on the
    underlying ProtocolPostgresAdapter implementation.

Related:
    - NodeRegistryEffect: Parent effect node that coordinates handlers
    - ProtocolPostgresAdapter: Protocol defining the deactivate interface
    - ModelBackendResult: Structured result model for backend operations
    - ModelRegistryRequest: Input request model
    - OMN-1103: Refactoring ticket for handler extraction
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
)
from omnibase_infra.nodes.node_registry_effect.models import ModelBackendResult
from omnibase_infra.utils import sanitize_backend_error, sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.nodes.effects.protocol_postgres_adapter import (
        ProtocolPostgresAdapter,
    )
    from omnibase_infra.nodes.node_registry_effect.models import ModelRegistryRequest


class HandlerPostgresDeactivate:
    """Handler for PostgreSQL registration record deactivation.

    Encapsulates all PostgreSQL-specific deactivation logic extracted from
    NodeRegistryEffect for declarative node compliance. The handler provides
    a clean interface for executing deactivation operations with proper timing
    and error sanitization.

    The deactivation operation marks a node registration as inactive (soft delete)
    rather than performing a hard delete, preserving audit trails and enabling
    potential reactivation.

    Attributes:
        _postgres_adapter: Protocol-compliant PostgreSQL adapter.

    Example:
        >>> from unittest.mock import AsyncMock
        >>> adapter = AsyncMock()
        >>> adapter.deactivate.return_value = ModelBackendResult(success=True, backend_id="postgres")
        >>> handler = HandlerPostgresDeactivate(adapter)
        >>> result = await handler.handle(request, correlation_id)
        >>> result.success
        True

    See Also:
        - NodeRegistryEffect: Parent node that uses this handler
        - ProtocolPostgresAdapter: Protocol for PostgreSQL operations
    """

    def __init__(self, postgres_adapter: ProtocolPostgresAdapter) -> None:
        """Initialize handler with PostgreSQL adapter.

        Args:
            postgres_adapter: Protocol-compliant PostgreSQL adapter for
                executing registration record deactivation operations.
        """
        self._postgres_adapter = postgres_adapter

    async def handle(
        self,
        request: ModelRegistryRequest,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute PostgreSQL registration record deactivation.

        Performs the deactivation operation against the PostgreSQL adapter with:
        - Operation timing via time.perf_counter()
        - Protocol call to adapter.deactivate()
        - Error sanitization for security
        - Structured result construction

        The deactivation marks the registration record as inactive without
        deleting the underlying data, supporting audit requirements and
        potential reactivation scenarios.

        Args:
            request: Registration request containing the node_id to deactivate.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with:
                - success: True if deactivation completed successfully
                - error: Sanitized error message if failed
                - error_code: Error code for programmatic handling
                - duration_ms: Operation duration in milliseconds
                - backend_id: Set to "postgres"
                - correlation_id: Passed through for tracing

        Note:
            Error messages are sanitized using sanitize_backend_error to
            prevent exposure of connection strings, credentials, or other
            sensitive information in logs and responses.
        """
        start_time = time.perf_counter()

        try:
            result = await self._postgres_adapter.deactivate(
                node_id=request.node_id,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if result.success:
                return ModelBackendResult(
                    success=True,
                    duration_ms=duration_ms,
                    backend_id="postgres",
                    correlation_id=correlation_id,
                )
            else:
                # Sanitize backend error to avoid exposing secrets
                # (connection strings, credentials, internal hostnames)
                sanitized_error = sanitize_backend_error("postgres", result.error)
                return ModelBackendResult(
                    success=False,
                    error=sanitized_error,
                    error_code="POSTGRES_DEACTIVATION_ERROR",
                    duration_ms=duration_ms,
                    backend_id="postgres",
                    correlation_id=correlation_id,
                )

        except (TimeoutError, InfraTimeoutError) as e:
            # Timeout during deactivation - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="POSTGRES_TIMEOUT_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraAuthenticationError as e:
            # Authentication failure - non-retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="POSTGRES_AUTH_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraConnectionError as e:
            # Connection failure - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="POSTGRES_CONNECTION_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except (
            Exception
        ) as e:  # ONEX: catch-all - database adapter may raise unexpected exceptions
            # beyond typed infrastructure errors (e.g., driver errors, encoding errors,
            # connection pool errors). Required to sanitize errors and prevent credential exposure.
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="POSTGRES_UNKNOWN_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )


__all__: list[str] = ["HandlerPostgresDeactivate"]
