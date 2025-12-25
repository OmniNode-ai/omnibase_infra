# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Non-Retryable Error Category Enumeration.

Defines error categories that represent permanent failures and should NOT be retried.
Used by DLQ replay, event bus retry logic, and dispatch engine to determine retry
eligibility.

Design Rationale:
    Non-retryable errors are permanent failures where retrying would not succeed.
    These include:
    - Authentication/authorization failures (credentials are invalid)
    - Configuration/validation errors (input is malformed)
    - Schema mismatches (requires code changes)

    See docs/patterns/retry_backoff_compensation_strategy.md for the complete retry
    decision logic and rationale for each category.

Thread Safety:
    All enums in this module are immutable and thread-safe.
    Enum values can be safely shared across threads without synchronization.

Related:
    - OMN-1032: Centralize NON_RETRYABLE_ERRORS into shared enum
    - docs/patterns/retry_backoff_compensation_strategy.md
    - docs/dlq/DLQ_ARCHITECTURE.md
"""

from enum import Enum, unique


@unique
class EnumNonRetryableErrorCategory(str, Enum):
    """
    Error categories that should NOT be retried.

    These represent permanent failures where retrying would not succeed.
    Used by DLQ replay and event bus to determine retry eligibility.

    Categories:
        AUTHENTICATION_ERROR: Authentication/authorization failures (invalid credentials)
        CONFIGURATION_ERROR: Protocol or service configuration errors
        SECRET_RESOLUTION_ERROR: Secret/credential resolution failures (missing secrets)
        VALIDATION_ERROR: Input/schema validation errors (malformed data)

    Why These Are Non-Retryable:
        - AUTHENTICATION_ERROR: Credentials are invalid or permissions denied.
          Retrying with the same credentials will always fail. Requires
          credential refresh or permission changes.

        - CONFIGURATION_ERROR: Configuration is wrong or incompatible.
          Retrying will not fix malformed configuration. Requires code
          or configuration changes.

        - SECRET_RESOLUTION_ERROR: Secret does not exist or is inaccessible.
          Retrying will not create the missing secret. Requires secret
          provisioning or path correction.

        - VALIDATION_ERROR: Input data is malformed or violates schema.
          Retrying with the same data will always fail. Requires data
          correction at the source.

    Example:
        >>> from omnibase_infra.enums import EnumNonRetryableErrorCategory
        >>> EnumNonRetryableErrorCategory.is_non_retryable("ValidationError")
        True
        >>> EnumNonRetryableErrorCategory.is_non_retryable("InfraConnectionError")
        False
        >>> EnumNonRetryableErrorCategory.VALIDATION_ERROR.value
        'ValidationError'
    """

    AUTHENTICATION_ERROR = "InfraAuthenticationError"
    """Authentication/authorization failures from omnibase_infra.errors.

    Raised when:
    - Invalid credentials (401 Unauthorized)
    - Permission denied (403 Forbidden)
    - Token expired and cannot be refreshed
    - Encryption/decryption key mismatch

    Why non-retryable:
        Retrying with the same credentials will always fail.
        Requires credential refresh, permission changes, or key rotation.
    """

    CONFIGURATION_ERROR = "ProtocolConfigurationError"
    """Protocol or service configuration errors from omnibase_infra.errors.

    Raised when:
    - Invalid configuration values
    - Schema mismatch between services
    - Missing required configuration fields
    - Incompatible protocol versions

    Why non-retryable:
        Configuration is statically defined. Retrying will not fix
        malformed or incompatible configuration. Requires code or
        configuration changes and redeployment.
    """

    SECRET_RESOLUTION_ERROR = "SecretResolutionError"
    """Secret/credential resolution failures from omnibase_infra.errors.

    Raised when:
    - Secret path does not exist in Vault
    - Secret key not found within secret data
    - Vault backend unavailable (after retries exhausted)
    - Secret access denied by policy

    Why non-retryable:
        The secret either doesn't exist or is inaccessible by policy.
        Retrying will not create the missing secret. Requires secret
        provisioning or policy changes.
    """

    VALIDATION_ERROR = "ValidationError"
    """Input/schema validation errors from Pydantic or custom validators.

    Raised when:
    - Pydantic model validation fails
    - Request payload violates schema
    - Data type mismatches
    - Required fields missing
    - Constraint violations

    Why non-retryable:
        The input data itself is malformed. Retrying with the same
        malformed data will always fail. Requires correction at
        the data source.
    """

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value

    @classmethod
    def is_non_retryable(cls, error_type: str) -> bool:
        """
        Check if an error type is non-retryable.

        This is the primary method for determining retry eligibility.
        Used by DLQ replay and event bus retry logic.

        Args:
            error_type: The error type name (e.g., "ValidationError",
                "InfraAuthenticationError"). This typically comes from
                the error_type field in DLQ messages or exception class names.

        Returns:
            True if the error type is non-retryable, False otherwise

        Example:
            >>> EnumNonRetryableErrorCategory.is_non_retryable("ValidationError")
            True
            >>> EnumNonRetryableErrorCategory.is_non_retryable("InfraConnectionError")
            False
            >>> EnumNonRetryableErrorCategory.is_non_retryable("TimeoutError")
            False
        """
        return error_type in {member.value for member in cls}

    @classmethod
    def get_all_values(cls) -> frozenset[str]:
        """
        Get all non-retryable error type values as a frozenset.

        Returns a frozen (immutable) set of all error type values for efficient
        membership testing. This is useful when you need to perform multiple
        lookups or need to pass the set to external code.

        Returns:
            A frozenset containing all non-retryable error type values

        Example:
            >>> values = EnumNonRetryableErrorCategory.get_all_values()
            >>> "ValidationError" in values
            True
            >>> isinstance(values, frozenset)
            True
        """
        return frozenset(member.value for member in cls)

    @classmethod
    def get_description(cls, error_type: "EnumNonRetryableErrorCategory") -> str:
        """
        Get a human-readable description of why this error category is non-retryable.

        Args:
            error_type: The non-retryable error category to describe

        Returns:
            A human-readable description explaining why the error is non-retryable

        Example:
            >>> desc = EnumNonRetryableErrorCategory.get_description(
            ...     EnumNonRetryableErrorCategory.VALIDATION_ERROR
            ... )
            >>> "malformed data" in desc.lower()
            True
        """
        descriptions = {
            cls.AUTHENTICATION_ERROR: (
                "Authentication/authorization failure - credentials are invalid "
                "or permissions denied. Requires credential refresh or permission changes."
            ),
            cls.CONFIGURATION_ERROR: (
                "Configuration error - configuration is wrong or incompatible. "
                "Requires code or configuration changes and redeployment."
            ),
            cls.SECRET_RESOLUTION_ERROR: (
                "Secret resolution failure - secret does not exist or is inaccessible. "
                "Requires secret provisioning or policy changes."
            ),
            cls.VALIDATION_ERROR: (
                "Validation error - input data is malformed or violates schema. "
                "Requires data correction at the source."
            ),
        }
        return descriptions.get(error_type, "Unknown error category")


__all__ = ["EnumNonRetryableErrorCategory"]
