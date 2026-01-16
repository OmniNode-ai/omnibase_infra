# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul-Specific Infrastructure Error Class.

This module defines the InfraConsulError class for Consul-related infrastructure
errors. It extends InfraConnectionError to provide specialized error handling
for HashiCorp Consul operations.
"""

from omnibase_infra.errors.error_infra import InfraConnectionError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class InfraConsulError(InfraConnectionError):
    """Error communicating with Consul.

    Used for Consul client initialization failures, KV operations,
    service registration, and retry exhaustion scenarios.

    This error extends InfraConnectionError and is specifically designed
    for HashiCorp Consul operations. The context should use
    ``transport_type=EnumInfraTransportType.CONSUL``.

    Common use cases:
        - Consul client initialization failures
        - KV store read/write operation failures
        - Service registration/deregistration errors
        - Health check registration failures
        - Connection retry exhaustion
        - Session/lock operation errors

    Example:
        >>> from omnibase_infra.enums import EnumInfraTransportType
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.CONSUL,
        ...     operation="kv_get",
        ...     target_name="consul-primary",
        ... )
        >>> raise InfraConsulError(
        ...     "Failed to read key from Consul KV store",
        ...     context=context,
        ...     consul_key="config/database/connection",
        ... )

        >>> # Service registration failure
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.CONSUL,
        ...     operation="register_service",
        ...     target_name="consul-primary",
        ... )
        >>> raise InfraConsulError(
        ...     "Failed to register service with Consul",
        ...     context=context,
        ...     service_name="api-gateway",
        ...     service_id="api-gateway-1",
        ... )

        >>> # Client initialization failure
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.CONSUL,
        ...     operation="initialize_client",
        ...     target_name="consul-primary",
        ... )
        >>> raise InfraConsulError(
        ...     "Failed to initialize Consul client",
        ...     context=context,
        ...     host="consul.example.com",
        ...     port=8500,
        ...     retry_count=3,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        consul_key: str | None = None,
        service_name: str | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize InfraConsulError with Consul-specific context.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context (should use CONSUL transport_type)
            consul_key: Optional KV key that caused the error
            service_name: Optional service name for service registration errors
            **extra_context: Additional context information (e.g., host, port, retry_count)
        """
        # Include consul_key in extra_context if provided
        if consul_key is not None:
            extra_context["consul_key"] = consul_key

        # Include service_name in extra_context if provided
        if service_name is not None:
            extra_context["service_name"] = service_name

        super().__init__(
            message=message,
            context=context,
            **extra_context,
        )


__all__: list[str] = [
    "InfraConsulError",
]
