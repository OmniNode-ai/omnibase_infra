# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for partial failure retry operations.

This handler routes retry operations to the appropriate backend (Consul or PostgreSQL)
based on the target_backend field in the request. It is used to recover from partial
registration failures where one backend succeeded but another failed.

Architecture:
    This handler follows the ONEX handler pattern:
    - Receives typed input with target_backend specification
    - Routes to the appropriate backend based on target_backend field
    - Executes a single responsibility (targeted retry)
    - Returns typed output (ModelBackendResult)
    - Uses error sanitization for security

Handler Responsibilities:
    - Accept both Consul and PostgreSQL clients for routing flexibility
    - Route to correct backend based on target_backend field
    - Execute retry operation against targeted backend
    - Track operation timing
    - Sanitize errors to prevent credential exposure

Idempotency:
    This handler expects an idempotency_key in the request for safe retry semantics.
    The actual idempotency enforcement is handled by the caller or middleware,
    but the handler respects the key by delegating to idempotent backend operations.

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different request instances.

Related Tickets:
    - OMN-1103: NodeRegistryEffect refactoring to declarative pattern
    - OMN-954: Partial failure scenario testing
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.enums import EnumBackendType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
)
from omnibase_infra.nodes.node_registry_effect.models import ModelBackendResult
from omnibase_infra.utils import sanitize_backend_error, sanitize_error_message

if TYPE_CHECKING:
    from omnibase_core.enums.enum_node_kind import EnumNodeKind
    from omnibase_infra.nodes.effects.protocol_consul_client import ProtocolConsulClient
    from omnibase_infra.nodes.effects.protocol_postgres_adapter import (
        ProtocolPostgresAdapter,
    )


@runtime_checkable
class ProtocolPartialRetryRequest(Protocol):
    """Protocol for partial retry request objects.

    Defines the required attributes for a request to the HandlerPartialRetry handler.
    This allows duck typing with any request object that provides these fields.

    Attributes:
        node_id: Unique identifier for the node being registered.
        node_type: Type of ONEX node (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node.
        target_backend: Backend to retry (EnumBackendType.CONSUL or EnumBackendType.POSTGRES).
        idempotency_key: Optional key for idempotent retry semantics.
        service_name: Optional service name for Consul registration.
        tags: Tags for Consul service discovery.
        health_check_config: Optional Consul health check configuration.
        endpoints: Dict of endpoint type to URL for PostgreSQL.
        metadata: Additional metadata for PostgreSQL registration.
    """

    node_id: UUID
    node_type: EnumNodeKind
    node_version: ModelSemVer
    target_backend: EnumBackendType
    idempotency_key: str | None
    service_name: str | None
    tags: list[str]
    health_check_config: dict[str, str] | None
    endpoints: dict[str, str]
    metadata: dict[str, str]


class HandlerPartialRetry:
    """Handler for partial failure retry operations.

    Routes retry operations to the appropriate backend (Consul or PostgreSQL)
    based on the target_backend field in the request. This handler is used to
    recover from partial registration failures where one backend succeeded
    but another failed.

    Backend Routing:
        - target_backend=EnumBackendType.CONSUL: Routes to Consul registration
        - target_backend=EnumBackendType.POSTGRES: Routes to PostgreSQL upsert
        - Unknown values: Returns error result

    Error Handling:
        All errors are sanitized before inclusion in the result to prevent
        credential exposure. The sanitization uses an allowlist approach,
        only including known-safe error patterns.

    Attributes:
        _consul_client: Protocol-compliant Consul client for service registration.
        _postgres_adapter: Protocol-compliant PostgreSQL adapter for persistence.

    Example:
        >>> from unittest.mock import AsyncMock
        >>> consul_client = AsyncMock()
        >>> postgres_adapter = AsyncMock()
        >>> consul_client.register_service.return_value = ModelBackendResult(
        ...     success=True, duration_ms=0.0, backend_id="consul"
        ... )
        >>> handler = HandlerPartialRetry(consul_client, postgres_adapter)
        >>> # Call handler.handle(request, correlation_id) in async context
    """

    def __init__(
        self,
        consul_client: ProtocolConsulClient,
        postgres_adapter: ProtocolPostgresAdapter,
    ) -> None:
        """Initialize handler with both backend clients.

        Args:
            consul_client: Protocol-compliant Consul client for service registration.
                Must implement ProtocolConsulClient with async register_service method.
            postgres_adapter: Protocol-compliant PostgreSQL adapter for persistence.
                Must implement ProtocolPostgresAdapter with async upsert method.
        """
        self._consul_client = consul_client
        self._postgres_adapter = postgres_adapter

    async def handle(
        self,
        request: ProtocolPartialRetryRequest,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute partial failure retry operation.

        Routes to the appropriate backend based on target_backend field and
        executes the retry operation. Handles both success and failure cases
        with appropriate timing and error sanitization.

        Args:
            request: Retry request with target backend specification including:
                - node_id: UUID of the node to register
                - node_type: ONEX node type (effect, compute, reducer, orchestrator)
                - target_backend: EnumBackendType.CONSUL or EnumBackendType.POSTGRES
                - idempotency_key: Optional key for safe retry semantics
                - service_name: Optional custom service name (for Consul)
                - tags: List of service discovery tags (for Consul)
                - health_check_config: Optional health check configuration (for Consul)
                - endpoints: Dict of endpoint URLs (for PostgreSQL)
                - metadata: Additional metadata (for PostgreSQL)

            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with:
                - success: True if retry operation succeeded
                - duration_ms: Time taken for the operation
                - backend_id: The target backend ("consul" or "postgres")
                - correlation_id: The provided correlation ID
                - error: Sanitized error message (only on failure)
                - error_code: Error code for programmatic handling (only on failure)

        Note:
            This handler does not raise exceptions. All errors are captured
            and returned in the ModelBackendResult to support partial failure
            handling in dual-backend registration scenarios.
        """
        start_time = time.perf_counter()
        target_backend = request.target_backend

        if target_backend == EnumBackendType.CONSUL:
            return await self._retry_consul(request, correlation_id, start_time)
        elif target_backend == EnumBackendType.POSTGRES:
            return await self._retry_postgres(request, correlation_id, start_time)
        else:
            # Defensive: This branch handles unexpected enum values that may arise from
            # duck-typed Protocol usage, where callers could pass objects with a
            # target_backend attribute that isn't a valid EnumBackendType member.
            # While static typing prevents this in normal usage, the Protocol pattern
            # allows runtime duck typing that bypasses compile-time checks.
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = (
                f"Unknown target backend: {target_backend}. "
                "Expected 'consul' or 'postgres'."
            )
            return ModelBackendResult(
                success=False,
                error=error_msg,
                error_code="INVALID_TARGET_BACKEND",
                duration_ms=duration_ms,
                backend_id=target_backend.value
                if isinstance(target_backend, EnumBackendType)
                else str(target_backend),
                correlation_id=correlation_id,
            )

    async def _retry_consul(
        self,
        request: ProtocolPartialRetryRequest,
        correlation_id: UUID,
        start_time: float,
    ) -> ModelBackendResult:
        """Execute Consul registration retry.

        Args:
            request: Retry request with Consul registration parameters.
            correlation_id: Request correlation ID for distributed tracing.
            start_time: Operation start time for duration tracking.

        Returns:
            ModelBackendResult with Consul operation outcome.
        """
        # Generate service identifiers following ONEX naming convention
        service_id = f"onex-{request.node_type.value}-{request.node_id}"
        service_name = request.service_name or f"onex-{request.node_type.value}"

        try:
            result = await self._consul_client.register_service(
                service_id=service_id,
                service_name=service_name,
                tags=request.tags,
                health_check=request.health_check_config,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if result.success:
                return ModelBackendResult(
                    success=True,
                    duration_ms=duration_ms,
                    backend_id="consul",
                    correlation_id=correlation_id,
                )
            else:
                # Client returned failure - sanitize the error message
                sanitized_error = sanitize_backend_error("consul", result.error)
                return ModelBackendResult(
                    success=False,
                    error=sanitized_error,
                    error_code="CONSUL_REGISTRATION_ERROR",
                    duration_ms=duration_ms,
                    backend_id="consul",
                    correlation_id=correlation_id,
                )

        except (TimeoutError, InfraTimeoutError) as e:
            # Timeout during registration - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="CONSUL_TIMEOUT_ERROR",
                duration_ms=duration_ms,
                backend_id="consul",
                correlation_id=correlation_id,
            )

        except InfraAuthenticationError as e:
            # Authentication failure - non-retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="CONSUL_AUTH_ERROR",
                duration_ms=duration_ms,
                backend_id="consul",
                correlation_id=correlation_id,
            )

        except InfraConnectionError as e:
            # Connection failure - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="CONSUL_CONNECTION_ERROR",
                duration_ms=duration_ms,
                backend_id="consul",
                correlation_id=correlation_id,
            )

        except (
            Exception
        ) as e:  # ONEX: catch-all - external service may raise unexpected exceptions
            # beyond typed infrastructure errors (e.g., HTTP client errors, JSON decode errors,
            # network stack errors). Required to sanitize errors and prevent credential exposure.
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="CONSUL_UNKNOWN_ERROR",
                duration_ms=duration_ms,
                backend_id="consul",
                correlation_id=correlation_id,
            )

    async def _retry_postgres(
        self,
        request: ProtocolPartialRetryRequest,
        correlation_id: UUID,
        start_time: float,
    ) -> ModelBackendResult:
        """Execute PostgreSQL upsert retry.

        Args:
            request: Retry request with PostgreSQL upsert parameters.
            correlation_id: Request correlation ID for distributed tracing.
            start_time: Operation start time for duration tracking.

        Returns:
            ModelBackendResult with PostgreSQL operation outcome.
        """
        try:
            result = await self._postgres_adapter.upsert(
                node_id=request.node_id,
                node_type=request.node_type,
                node_version=request.node_version,
                endpoints=request.endpoints,
                metadata=request.metadata,
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
                sanitized_error = sanitize_backend_error("postgres", result.error)
                return ModelBackendResult(
                    success=False,
                    error=sanitized_error,
                    error_code="POSTGRES_UPSERT_ERROR",
                    duration_ms=duration_ms,
                    backend_id="postgres",
                    correlation_id=correlation_id,
                )

        except (TimeoutError, InfraTimeoutError) as e:
            # Timeout during upsert - retriable error
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


__all__: list[str] = ["HandlerPartialRetry", "ProtocolPartialRetryRequest"]
