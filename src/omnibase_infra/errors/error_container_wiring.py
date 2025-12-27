# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Container Wiring Error Classes.

This module defines error classes specific to container wiring operations,
providing granular error handling for service registration and resolution.
"""

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode

from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class ContainerWiringError(RuntimeHostError):
    """Base error for container wiring operations.

    Used as base class for all container wiring-related errors.
    Provides common structured fields for container operations.

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.RUNTIME,
        ...     operation="wire_services",
        ... )
        >>> raise ContainerWiringError(
        ...     "Container wiring failed",
        ...     context=context,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize ContainerWiringError.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context
            **extra_context: Additional context information
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.OPERATION_FAILED,
            context=context,
            **extra_context,
        )


class ServiceRegistrationError(ContainerWiringError):
    """Raised when service registration fails.

    Used for failures during service instance registration with the container,
    including duplicate registrations, validation failures, or container API issues.

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.RUNTIME,
        ...     operation="register_service",
        ... )
        >>> raise ServiceRegistrationError(
        ...     "Failed to register PolicyRegistry",
        ...     service_name="PolicyRegistry",
        ...     context=context,
        ... )
    """

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize ServiceRegistrationError.

        Args:
            message: Human-readable error message
            service_name: Name of the service that failed to register
            context: Bundled infrastructure context
            **extra_context: Additional context information
        """
        if service_name is not None:
            extra_context["service_name"] = service_name

        super().__init__(
            message=message,
            context=context,
            **extra_context,
        )


class ServiceResolutionError(ContainerWiringError):
    """Raised when service resolution fails.

    Used for failures when attempting to resolve a service from the container,
    including unregistered services, type mismatches, or container API issues.

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.RUNTIME,
        ...     operation="resolve_service",
        ... )
        >>> raise ServiceResolutionError(
        ...     "PolicyRegistry not found in container",
        ...     service_name="PolicyRegistry",
        ...     context=context,
        ... )
    """

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize ServiceResolutionError.

        Args:
            message: Human-readable error message
            service_name: Name of the service that failed to resolve
            context: Bundled infrastructure context
            **extra_context: Additional context information
        """
        if service_name is not None:
            extra_context["service_name"] = service_name

        super().__init__(
            message=message,
            context=context,
            **extra_context,
        )


class ContainerValidationError(ContainerWiringError):
    """Raised when container validation fails.

    Used for pre-wiring validation failures, such as missing required
    container attributes or invalid container state.

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.RUNTIME,
        ...     operation="validate_container",
        ... )
        >>> raise ContainerValidationError(
        ...     "Container missing service_registry attribute",
        ...     context=context,
        ...     missing_attribute="service_registry",
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize ContainerValidationError.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context
            **extra_context: Additional context information
        """
        super().__init__(
            message=message,
            context=context,
            **extra_context,
        )


__all__ = [
    "ContainerWiringError",
    "ServiceRegistrationError",
    "ServiceResolutionError",
    "ContainerValidationError",
]
