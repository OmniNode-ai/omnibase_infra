# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Errors Module.

This module provides infrastructure-specific error classes and error handling
utilities for the omnibase_infra package. All errors extend from OnexError
to maintain consistency with the ONEX error handling patterns.

Exports:
    ModelInfraErrorContext: Configuration model for bundled error context
    RuntimeHostError: Base infrastructure error class
    ProtocolConfigurationError: Protocol configuration validation errors
    SecretResolutionError: Secret/credential resolution errors
    InfraConnectionError: Infrastructure connection errors
    InfraTimeoutError: Infrastructure timeout errors
    InfraAuthenticationError: Infrastructure authentication errors
    InfraUnavailableError: Infrastructure resource unavailable errors

Correlation ID Assignment:
    All infrastructure errors support correlation_id for distributed tracing.
    Follow these rules when assigning correlation IDs:

    - Always propagate correlation_id from incoming requests to error context
    - If no correlation_id exists in the request, generate one using uuid4()
    - Use UUID4 format for all new correlation IDs (from uuid import uuid4)
    - Include correlation_id in all error context for distributed tracing
    - Preserve correlation_id as UUID objects throughout the system (strong typing)

    Example::

        from uuid import UUID, uuid4
        from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
        from omnibase_infra.enums import EnumInfraTransportType

        # Propagate from request or generate new
        correlation_id = request.correlation_id or uuid4()

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="execute_query",
            target_name="postgresql-primary",
            correlation_id=correlation_id,
        )
        raise InfraConnectionError("Failed to connect", context=context) from e

Error Sanitization Guidelines:
    NEVER include in error messages or context:
        - Passwords, API keys, tokens, or secrets
        - Full connection strings with credentials
        - PII (names, emails, SSNs, phone numbers)
        - Internal IP addresses (in production logs)
        - Private keys or certificates
        - Session tokens or cookies

    SAFE to include:
        - Service names (e.g., "postgresql", "kafka")
        - Operation names (e.g., "connect", "query", "authenticate")
        - Correlation IDs (always include for tracing)
        - Error codes (e.g., EnumCoreErrorCode.DATABASE_CONNECTION_ERROR)
        - Sanitized hostnames (e.g., "db.example.com")
        - Port numbers
        - Retry counts and timeout values
        - Resource identifiers (non-sensitive)

    Example - BAD (exposes credentials)::

        raise InfraConnectionError(
            f"Failed to connect with password={password}",  # NEVER DO THIS
            context=context,
        )

    Example - GOOD (sanitized)::

        raise InfraConnectionError(
            "Failed to connect to database",
            context=context,
            host="db.example.com",
            port=5432,
            retry_count=3,
        )
"""

from omnibase_infra.errors.infra_errors import (
    EnvelopeValidationError,
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ProtocolConfigurationError,
    RuntimeHostError,
    SecretResolutionError,
    UnknownHandlerTypeError,
)
from omnibase_infra.errors.model_infra_error_context import ModelInfraErrorContext

__all__: list[str] = [
    # Configuration model
    "ModelInfraErrorContext",
    # Error classes
    "RuntimeHostError",
    "ProtocolConfigurationError",
    "SecretResolutionError",
    "InfraConnectionError",
    "InfraTimeoutError",
    "InfraAuthenticationError",
    "InfraUnavailableError",
    "EnvelopeValidationError",
    "UnknownHandlerTypeError",
]
