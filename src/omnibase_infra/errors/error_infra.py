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

NOT_FOUND Classification Patterns:
    The ONEX infrastructure uses two distinct patterns for "not found" scenarios:

    1. **Error-based NOT_FOUND** (for exceptional conditions):
       When a resource SHOULD exist but doesn't, raise an error with
       ``EnumCoreErrorCode.RESOURCE_NOT_FOUND``. This is used by:
       - ``SecretResolutionError``: When a required secret is missing from Vault
       - Handler errors: When a required resource lookup fails

       Callers receive these as exceptions and should handle them via try/except:

       .. code-block:: python

           try:
               secret = await vault_handler.get_secret("db/password")
           except SecretResolutionError as e:
               if e.model.error_code == EnumCoreErrorCode.RESOURCE_NOT_FOUND:
                   logger.error(f"Secret not found: {e.model.message}")
                   # Handle missing secret (fail, use default, etc.)

    2. **Response-based NOT_FOUND** (for valid "empty" responses):
       When absence is a valid outcome (not an error), return a discriminated
       union variant. This is used by:
       - ``EnumConsulOperationType.KV_GET_NOT_FOUND``: When a Consul key doesn't exist
       - Handler responses with optional data fields

       Callers check the response type to determine if data was found:

       .. code-block:: python

           result = await consul_handler.kv_get("my/key")
           if result.operation_type == EnumConsulOperationType.KV_GET_NOT_FOUND:
               # Key doesn't exist - valid "empty" response
               return default_value
           else:
               return result.value

    **Choosing the Right Pattern**:
    - Use errors when absence indicates a configuration or operational problem
    - Use response variants when absence is an expected, valid state
    - Secrets, required configs: Use ``RESOURCE_NOT_FOUND`` error
    - Optional keys, cache lookups: Use discriminated union responses
"""

from typing import Any, cast
from uuid import uuid4

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.errors.model_timeout_error_context import (
    ModelTimeoutErrorContext,
)


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
        error_code: EnumCoreErrorCode | None = None,
        context: ModelInfraErrorContext | None = None,
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
            if context.namespace is not None:
                structured_context["namespace"] = context.namespace
            correlation_id = context.correlation_id

        # Auto-generate correlation_id if not provided (per CLAUDE.md guidelines)
        if correlation_id is None:
            correlation_id = uuid4()

        # Initialize base error with default error code
        # NOTE: Cast required for mypy - **dict[str, object] doesn't satisfy **context: Any
        super().__init__(
            message=message,
            error_code=error_code or EnumCoreErrorCode.OPERATION_FAILED,
            correlation_id=correlation_id,
            **cast("dict[str, Any]", structured_context),
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
        context: ModelInfraErrorContext | None = None,
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

    Security:
        Do NOT include full secret paths in error messages or extra_context.
        Use generic descriptions like "database credential" instead of
        revealing path structure like "database/postgres/password".

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.VAULT,
        ...     operation="get_secret",
        ...     target_name="vault-primary",
        ... )
        >>> # Use generic description, not the actual path
        >>> raise SecretResolutionError(
        ...     "Database credential not found in Vault",
        ...     context=context,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
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

    The error code is automatically selected based on the transport type
    in the context:
        - DATABASE -> DATABASE_CONNECTION_ERROR
        - HTTP, GRPC -> NETWORK_ERROR
        - KAFKA, CONSUL, VAULT, VALKEY, FILESYSTEM, QDRANT, GRAPH, MCP -> SERVICE_UNAVAILABLE
        - None (no context) -> SERVICE_UNAVAILABLE

    Example:
        >>> # Database connection with transport-specific error code
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

        >>> # HTTP connection uses NETWORK_ERROR
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.HTTP,
        ...     operation="request",
        ...     target_name="api-gateway",
        ... )
        >>> raise InfraConnectionError("API connection failed", context=context)

        >>> # Kafka connection uses SERVICE_UNAVAILABLE
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.KAFKA,
        ...     operation="produce",
        ...     target_name="kafka-broker",
        ... )
        >>> raise InfraConnectionError("Kafka connection failed", context=context)
    """

    # Transport type to error code mapping
    _TRANSPORT_ERROR_CODE_MAP: dict[
        EnumInfraTransportType | None, EnumCoreErrorCode
    ] = {
        EnumInfraTransportType.DATABASE: EnumCoreErrorCode.DATABASE_CONNECTION_ERROR,
        EnumInfraTransportType.HTTP: EnumCoreErrorCode.NETWORK_ERROR,
        EnumInfraTransportType.GRPC: EnumCoreErrorCode.NETWORK_ERROR,
        EnumInfraTransportType.KAFKA: EnumCoreErrorCode.SERVICE_UNAVAILABLE,
        EnumInfraTransportType.CONSUL: EnumCoreErrorCode.SERVICE_UNAVAILABLE,
        EnumInfraTransportType.VAULT: EnumCoreErrorCode.SERVICE_UNAVAILABLE,
        EnumInfraTransportType.VALKEY: EnumCoreErrorCode.SERVICE_UNAVAILABLE,
        EnumInfraTransportType.RUNTIME: EnumCoreErrorCode.OPERATION_FAILED,
        EnumInfraTransportType.INMEMORY: EnumCoreErrorCode.OPERATION_FAILED,
        EnumInfraTransportType.FILESYSTEM: EnumCoreErrorCode.SERVICE_UNAVAILABLE,
        EnumInfraTransportType.MCP: EnumCoreErrorCode.SERVICE_UNAVAILABLE,
        EnumInfraTransportType.QDRANT: EnumCoreErrorCode.SERVICE_UNAVAILABLE,
        EnumInfraTransportType.GRAPH: EnumCoreErrorCode.SERVICE_UNAVAILABLE,
        None: EnumCoreErrorCode.SERVICE_UNAVAILABLE,
    }

    @classmethod
    def _resolve_connection_error_code(
        cls, context: ModelInfraErrorContext | None
    ) -> EnumCoreErrorCode:
        """Resolve the appropriate error code based on transport type.

        Args:
            context: Infrastructure error context containing transport type

        Returns:
            Appropriate EnumCoreErrorCode for the transport type:
                - DATABASE -> DATABASE_CONNECTION_ERROR
                - HTTP, GRPC -> NETWORK_ERROR
                - KAFKA, CONSUL, VAULT, VALKEY, FILESYSTEM, QDRANT, GRAPH, MCP, None -> SERVICE_UNAVAILABLE
        """
        if context is None:
            return cls._TRANSPORT_ERROR_CODE_MAP[None]
        return cls._TRANSPORT_ERROR_CODE_MAP.get(
            context.transport_type,
            EnumCoreErrorCode.SERVICE_UNAVAILABLE,
        )

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize InfraConnectionError with transport-aware error code.

        The error code is automatically selected based on context.transport_type:
            - DATABASE -> DATABASE_CONNECTION_ERROR
            - HTTP, GRPC -> NETWORK_ERROR
            - KAFKA, CONSUL, VAULT, VALKEY, FILESYSTEM, QDRANT, GRAPH, MCP -> SERVICE_UNAVAILABLE
            - None (no context) -> SERVICE_UNAVAILABLE

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context (transport_type determines error code)
            **extra_context: Additional context information (e.g., host, port, retry_count)
        """
        super().__init__(
            message=message,
            error_code=self._resolve_connection_error_code(context),
            context=context,
            **extra_context,
        )


class InfraTimeoutError(RuntimeHostError):
    """Raised when infrastructure operation exceeds timeout.

    Used for database query timeouts, HTTP request timeouts,
    message broker operation timeouts, or call deadlines.

    Typing Requirements:
        Context is REQUIRED for timeout errors using ModelTimeoutErrorContext.
        This model enforces stricter typing with required correlation_id.

        ModelTimeoutErrorContext guarantees:
        - transport_type: Required (identifies the transport layer)
        - operation: Required (identifies the operation that timed out)
        - correlation_id: Required (auto-generated via default_factory if not provided)
        - target_name: Optional (target resource name)
        - timeout_seconds: Optional (the timeout value that was exceeded)

    Example:
        >>> context = ModelTimeoutErrorContext(
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ...     operation="execute_query",
        ...     target_name="postgresql-primary",
        ...     correlation_id=request.correlation_id,
        ...     timeout_seconds=30.0,
        ... )
        >>> raise InfraTimeoutError(
        ...     "Database query exceeded timeout",
        ...     context=context,
        ... )

        >>> # Auto-generated correlation_id (via default_factory)
        >>> context = ModelTimeoutErrorContext(
        ...     transport_type=EnumInfraTransportType.HTTP,
        ...     operation="fetch_resource",
        ...     timeout_seconds=10.0,
        ... )
        >>> raise InfraTimeoutError("Request timed out", context=context)
    """

    def __init__(
        self,
        message: str,
        context: ModelTimeoutErrorContext,
        **extra_context: object,
    ) -> None:
        """Initialize InfraTimeoutError with required ModelTimeoutErrorContext.

        Args:
            message: Human-readable error message
            context: Required timeout error context. ModelTimeoutErrorContext
                guarantees correlation_id is always present (auto-generated
                via default_factory if not explicitly provided).
            **extra_context: Additional context information

        Note:
            Context uses ModelTimeoutErrorContext which guarantees correlation_id
            at type-check time via its default_factory=uuid4. This provides
            compile-time safety that correlation_id is always present.
        """
        # Convert ModelTimeoutErrorContext to ModelInfraErrorContext for base class
        # This preserves all fields while maintaining compatibility with RuntimeHostError
        infra_context = ModelInfraErrorContext(
            transport_type=context.transport_type,
            operation=context.operation,
            target_name=context.target_name,
            correlation_id=context.correlation_id,
        )
        # Include timeout_seconds in extra_context if present
        if context.timeout_seconds is not None:
            extra_context = {
                "timeout_seconds": context.timeout_seconds,
                **extra_context,
            }

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.TIMEOUT_ERROR,
            context=infra_context,
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
        context: ModelInfraErrorContext | None = None,
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
        context: ModelInfraErrorContext | None = None,
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


class EnvelopeValidationError(RuntimeHostError):
    """Raised when envelope validation fails before dispatch.

    Used for:
    - Missing required fields (operation)
    - Missing required payload for data operations

    Note: Invalid correlation_id formats are normalized (not rejected).
    Invalid UUIDs are replaced with newly generated UUIDs during validation.

    This is a pre-dispatch validation error, NOT a handler-specific error.
    Handlers should NOT use this error class.

    Example:
        >>> raise EnvelopeValidationError(
        ...     "operation is required and must be non-empty string",
        ...     context=context,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize EnvelopeValidationError.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context
            **extra_context: Additional context information
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.INVALID_INPUT,
            context=context,
            **extra_context,
        )


class UnknownHandlerTypeError(RuntimeHostError):
    """Raised when an operation references an unknown handler type prefix.

    Used when dispatching envelopes with operation prefixes that don't
    map to any registered handler (e.g., "lolnope.query" when only
    "db" and "http" are registered).

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.RUNTIME,
        ...     operation="lolnope.query",
        ... )
        >>> raise UnknownHandlerTypeError(
        ...     "No handler registered for prefix: lolnope",
        ...     context=context,
        ...     prefix="lolnope",
        ...     registered_prefixes=["db", "http"],
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize UnknownHandlerTypeError.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context
            **extra_context: Additional context (prefix, registered_prefixes, etc.)
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.INVALID_INPUT,
            context=context,
            **extra_context,
        )


__all__: list[str] = [
    "EnvelopeValidationError",
    "InfraAuthenticationError",
    "InfraConnectionError",
    "InfraTimeoutError",
    "InfraUnavailableError",
    "ProtocolConfigurationError",
    "RuntimeHostError",
    "SecretResolutionError",
    "UnknownHandlerTypeError",
]
