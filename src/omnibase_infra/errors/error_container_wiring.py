# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Container Wiring Error Classes.

This module defines error classes specific to container wiring operations,
providing granular error handling for service registration and resolution.
"""

from omnibase_core.enums import EnumCoreErrorCode
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
        ...     "Failed to register RegistryPolicy",
        ...     service_name="RegistryPolicy",
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
        ...     "RegistryPolicy not found in container",
        ...     service_name="RegistryPolicy",
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


class ServiceRegistryUnavailableError(ContainerValidationError):
    """Raised when container.service_registry is None.

    This error indicates that the ModelONEXContainer was initialized
    without a service registry, either because:
    - enable_service_registry=False was passed
    - The ServiceRegistry module is not installed/available
    - Container initialization failed silently

    This is a distinct error from generic ContainerValidationError to provide
    clearer diagnostics and actionable error messages for service_registry issues.

    Example:
        >>> raise ServiceRegistryUnavailableError(
        ...     "Container service_registry is None",
        ...     operation="register RegistryPolicy",
        ...     hint="Check that enable_service_registry=True",
        ... )
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        hint: str | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize ServiceRegistryUnavailableError.

        Args:
            message: Primary error message.
            operation: The operation that was attempted (e.g., 'register_instance').
            hint: Actionable hint for fixing the issue.
            context: Bundled infrastructure context.
            **extra_context: Additional context information.
        """
        # Build full message with operation and hint
        full_message = message
        if operation:
            full_message = f"{message} (operation: {operation})"
            extra_context["operation"] = operation

        default_hint = (
            "Ensure ModelONEXContainer is created with enable_service_registry=True "
            "and that the ServiceRegistry module is installed."
        )
        actual_hint = hint or default_hint
        full_message = f"{full_message}\nHint: {actual_hint}"
        extra_context["hint"] = actual_hint

        super().__init__(
            message=full_message,
            context=context,
            **extra_context,
        )


__all__ = [
    "ContainerValidationError",
    "ContainerWiringError",
    "ServiceRegistrationError",
    "ServiceRegistryUnavailableError",
    "ServiceResolutionError",
]
