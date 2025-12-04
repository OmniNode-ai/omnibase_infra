# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infrastructure-Specific Error Classes.

This module defines infrastructure-specific error classes for the
omnibase_infra package. All error classes extend from ModelOnexError
(from omnibase_core) to maintain consistency with ONEX error handling patterns.

Error Hierarchy:
    ModelOnexError (from omnibase_core)
    └── RuntimeHostError (base infrastructure error)
        ├── ProtocolConfigurationError
        ├── SecretResolutionError
        ├── InfraConnectionError
        ├── InfraTimeoutError
        ├── InfraAuthenticationError
        └── InfraUnavailableError

All errors:
    - Extend ModelOnexError from omnibase_core
    - Use EnumCoreErrorCode for error classification
    - Support proper error chaining with `raise ... from e`
    - Include structured context for debugging
    - Support correlation IDs for request tracking
    - Accept ModelInfraErrorContext for bundled context parameters
"""

from typing import Optional

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.models.errors.model_onex_error import ModelOnexError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.model_infra_error_context import ModelInfraErrorContext


class RuntimeHostError(ModelOnexError):
    """Base error class for runtime host infrastructure errors.

    All infrastructure-specific errors should inherit from this class.
    Provides common structured fields for infrastructure operations.

    Structured Fields (via ModelInfraErrorContext):
        transport_type: Type of transport (http, db, kafka, etc.)
        operation: Operation being performed
        correlation_id: Request correlation ID for tracking
        target_name: Target resource/endpoint name

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.HTTP,
        ...     operation="process_request",
        ...     target_name="api-gateway",
        ... )
        >>> raise RuntimeHostError("Operation failed", context=context)

        # Or with extra context:
        >>> raise RuntimeHostError(
        ...     "Operation failed",
        ...     context=context,
        ...     retry_count=3,
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[EnumCoreErrorCode] = None,
        context: Optional[ModelInfraErrorContext] = None,
        **extra_context: object,
    ) -> None:
        """Initialize RuntimeHostError with structured fields.

        Args:
            message: Human-readable error message
            error_code: Error code (defaults to OPERATION_FAILED)
            context: Bundled infrastructure context (transport_type, operation, etc.)
            **extra_context: Additional context information
        """
        # Build structured context from model and extra kwargs
        structured_context: dict[str, object] = dict(extra_context)

        # Extract fields from context model if provided
        correlation_id = None
        if context is not None:
            if context.transport_type is not None:
                structured_context["transport_type"] = context.transport_type
            if context.operation is not None:
                structured_context["operation"] = context.operation
            if context.target_name is not None:
                structured_context["target_name"] = context.target_name
            correlation_id = context.correlation_id

        # Initialize base error with default error code
        super().__init__(
            message=message,
            error_code=error_code or EnumCoreErrorCode.OPERATION_FAILED,
            correlation_id=correlation_id,
            **structured_context,
        )


class ProtocolConfigurationError(RuntimeHostError):
    """Raised when protocol configuration validation fails.

    Used for configuration parsing errors, missing required fields,
    invalid configuration values, or schema validation failures.

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.HTTP,
        ...     operation="validate_config",
        ... )
        >>> raise ProtocolConfigurationError(
        ...     "Missing required field 'endpoint'",
        ...     context=context,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[ModelInfraErrorContext] = None,
        **extra_context: object,
    ) -> None:
        """Initialize ProtocolConfigurationError.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context
            **extra_context: Additional context information
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
            context=context,
            **extra_context,
        )


class SecretResolutionError(RuntimeHostError):
    """Raised when secret or credential resolution fails.

    Used for Vault connection failures, missing secrets, expired credentials,
    or permission issues accessing secret stores.

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.VAULT,
        ...     operation="get_secret",
        ...     target_name="vault-primary",
        ... )
        >>> raise SecretResolutionError(
        ...     "Secret not found in Vault",
        ...     context=context,
        ...     secret_key="database/postgres/password",
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[ModelInfraErrorContext] = None,
        **extra_context: object,
    ) -> None:
        """Initialize SecretResolutionError.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context
            **extra_context: Additional context information (e.g., secret_key, vault_path)
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.RESOURCE_NOT_FOUND,
            context=context,
            **extra_context,
        )


class InfraConnectionError(RuntimeHostError):
    """Raised when infrastructure connection fails.

    Used for database connection failures, mesh connectivity issues,
    message broker connection problems, or network-related errors.

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ...     operation="connect",
        ...     target_name="postgresql-primary",
        ... )
        >>> raise InfraConnectionError(
        ...     "Failed to connect to PostgreSQL",
        ...     context=context,
        ...     host="db.example.com",
        ...     port=5432,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[ModelInfraErrorContext] = None,
        **extra_context: object,
    ) -> None:
        """Initialize InfraConnectionError.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context
            **extra_context: Additional context information (e.g., host, port, retry_count)
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.DATABASE_CONNECTION_ERROR,
            context=context,
            **extra_context,
        )


class InfraTimeoutError(RuntimeHostError):
    """Raised when infrastructure operation exceeds timeout.

    Used for database query timeouts, HTTP request timeouts,
    message broker operation timeouts, or call deadlines.

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ...     operation="execute_query",
        ...     target_name="postgresql-primary",
        ... )
        >>> raise InfraTimeoutError(
        ...     "Database query exceeded timeout",
        ...     context=context,
        ...     timeout_seconds=30,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[ModelInfraErrorContext] = None,
        **extra_context: object,
    ) -> None:
        """Initialize InfraTimeoutError.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context
            **extra_context: Additional context information (e.g., timeout_seconds)
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.TIMEOUT_ERROR,
            context=context,
            **extra_context,
        )


class InfraAuthenticationError(RuntimeHostError):
    """Raised when infrastructure authentication or authorization fails.

    Used for invalid credentials, expired tokens, insufficient permissions,
    or authentication failures.

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.VAULT,
        ...     operation="authenticate",
        ...     target_name="vault-primary",
        ... )
        >>> raise InfraAuthenticationError(
        ...     "Invalid Vault token",
        ...     context=context,
        ...     auth_method="token",
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[ModelInfraErrorContext] = None,
        **extra_context: object,
    ) -> None:
        """Initialize InfraAuthenticationError.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context
            **extra_context: Additional context information (e.g., username, auth_method)
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.AUTHENTICATION_ERROR,
            context=context,
            **extra_context,
        )


class InfraUnavailableError(RuntimeHostError):
    """Raised when infrastructure resource is unavailable.

    Used for resource downtime, maintenance mode, circuit breaker states,
    or health check failures.

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.KAFKA,
        ...     operation="produce",
        ...     target_name="kafka-broker-1",
        ... )
        >>> raise InfraUnavailableError(
        ...     "Kafka broker unavailable",
        ...     context=context,
        ...     host="kafka.example.com",
        ...     port=9092,
        ...     retry_count=3,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[ModelInfraErrorContext] = None,
        **extra_context: object,
    ) -> None:
        """Initialize InfraUnavailableError.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context
            **extra_context: Additional context information (e.g., host, port, retry_count)
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.SERVICE_UNAVAILABLE,
            context=context,
            **extra_context,
        )


__all__ = [
    "RuntimeHostError",
    "ProtocolConfigurationError",
    "SecretResolutionError",
    "InfraConnectionError",
    "InfraTimeoutError",
    "InfraAuthenticationError",
    "InfraUnavailableError",
]
