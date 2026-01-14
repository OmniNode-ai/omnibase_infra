# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for Consul service registration.

This handler encapsulates all Consul-specific registration logic extracted from
the legacy NodeRegistryEffect implementation as part of the OMN-1103 refactoring.

Architecture:
    This handler follows the ONEX handler pattern:
    - Receives typed input (ModelRegistryRequest)
    - Executes a single responsibility (Consul registration)
    - Returns typed output (ModelBackendResult)
    - Uses error sanitization for security

Handler Responsibilities:
    - Generate service identifiers (service_id, service_name)
    - Execute Consul registration via protocol client
    - Track operation timing
    - Sanitize errors to prevent credential exposure

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different request instances.

Related Tickets:
    - OMN-1103: NodeRegistryEffect refactoring to declarative pattern
    - OMN-954: Partial failure scenario testing
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
    from omnibase_infra.nodes.effects.protocol_consul_client import ProtocolConsulClient
    from omnibase_infra.nodes.node_registry_effect.models import ModelRegistryRequest


class HandlerConsulRegister:
    """Handler for Consul service registration.

    Encapsulates all Consul-specific registration logic extracted from
    NodeRegistryEffect for declarative node compliance.

    Service Naming Convention:
        - service_id: `onex-{node_type}-{node_id}` (unique identifier)
        - service_name: `request.service_name` or `onex-{node_type}` (discovery name)

    Error Handling:
        All errors are sanitized before inclusion in the result to prevent
        credential exposure. The sanitization uses an allowlist approach,
        only including known-safe error patterns.

    Attributes:
        _consul_client: Protocol-compliant Consul client for service registration.

    Example:
        >>> from unittest.mock import AsyncMock
        >>> consul_client = AsyncMock()
        >>> consul_client.register_service.return_value = ModelBackendResult(
        ...     success=True, duration_ms=0.0, backend_id="consul"
        ... )
        >>> handler = HandlerConsulRegister(consul_client)
        >>> # Call handler.handle(request, correlation_id) in async context
    """

    def __init__(self, consul_client: ProtocolConsulClient) -> None:
        """Initialize handler with Consul client.

        Args:
            consul_client: Protocol-compliant Consul client for service registration.
                Must implement ProtocolConsulClient with async register_service method.
        """
        self._consul_client = consul_client

    async def handle(
        self,
        request: ModelRegistryRequest,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute Consul service registration.

        Generates service identifiers following ONEX naming conventions and
        delegates registration to the Consul client. Handles both success
        and failure cases with appropriate timing and error sanitization.

        Args:
            request: Registration request with node details including:
                - node_id: UUID of the node to register
                - node_type: ONEX node type (effect, compute, reducer, orchestrator)
                - service_name: Optional custom service name
                - tags: Optional list of service discovery tags
                - health_check_config: Optional Consul health check configuration

            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with:
                - success: True if registration succeeded
                - duration_ms: Time taken for the operation
                - backend_id: "consul" to identify the backend
                - correlation_id: The provided correlation ID
                - error: Sanitized error message (only on failure)
                - error_code: Error code for programmatic handling (only on failure)

        Note:
            This handler does not raise exceptions. All errors are captured
            and returned in the ModelBackendResult to support partial failure
            handling in dual-backend registration scenarios.
        """
        start_time = time.perf_counter()

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


__all__: list[str] = ["HandlerConsulRegister"]
