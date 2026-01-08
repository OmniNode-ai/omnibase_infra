# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Retry Error Category Enumeration.

Defines classification categories for errors during retry handling in
infrastructure handlers. Used to determine retry eligibility and circuit
breaker behavior.

Thread Safety:
    All enums in this module are immutable and thread-safe.
    Enum values can be safely shared across threads without synchronization.
"""

from enum import Enum, unique


@unique
class EnumRetryErrorCategory(str, Enum):
    """Classification of errors for retry decision making.

    Error categories determine:
    - Whether to retry the operation
    - Whether to record circuit breaker failure
    - Which error type to raise after retries exhausted

    Values:
        TIMEOUT: Operation timed out - retry eligible, circuit breaker failure on exhaustion.
        AUTHENTICATION: Authentication/permission error - no retry, immediate circuit breaker failure.
        CONNECTION: Connection/network error - retry eligible, circuit breaker failure on exhaustion.
        NOT_FOUND: Resource not found - no retry, no circuit breaker failure (user error).
        UNKNOWN: Unknown/unexpected error - retry eligible, circuit breaker failure on exhaustion.

    Example:
        >>> category = EnumRetryErrorCategory.TIMEOUT
        >>> category.is_retry_eligible()
        True
        >>> EnumRetryErrorCategory.AUTHENTICATION.is_retry_eligible()
        False
    """

    TIMEOUT = "timeout"
    """Operation timed out - retry eligible, circuit breaker failure on exhaustion."""

    AUTHENTICATION = "authentication"
    """Authentication/permission error - no retry, immediate circuit breaker failure."""

    CONNECTION = "connection"
    """Connection/network error - retry eligible, circuit breaker failure on exhaustion."""

    NOT_FOUND = "not_found"
    """Resource not found - no retry, no circuit breaker failure (user error)."""

    UNKNOWN = "unknown"
    """Unknown/unexpected error - retry eligible, circuit breaker failure on exhaustion."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value

    def is_retry_eligible(self) -> bool:
        """Check if this error category is eligible for retry.

        Returns:
            True if the error category allows retry, False otherwise.

        Example:
            >>> EnumRetryErrorCategory.CONNECTION.is_retry_eligible()
            True
            >>> EnumRetryErrorCategory.NOT_FOUND.is_retry_eligible()
            False
        """
        return self in {
            EnumRetryErrorCategory.TIMEOUT,
            EnumRetryErrorCategory.CONNECTION,
            EnumRetryErrorCategory.UNKNOWN,
        }

    def records_circuit_failure(self) -> bool:
        """Check if this error category should record circuit breaker failure.

        Returns:
            True if the error should be recorded as a circuit breaker failure.

        Example:
            >>> EnumRetryErrorCategory.AUTHENTICATION.records_circuit_failure()
            True
            >>> EnumRetryErrorCategory.NOT_FOUND.records_circuit_failure()
            False
        """
        return self in {
            EnumRetryErrorCategory.TIMEOUT,
            EnumRetryErrorCategory.AUTHENTICATION,
            EnumRetryErrorCategory.CONNECTION,
            EnumRetryErrorCategory.UNKNOWN,
        }


__all__: list[str] = ["EnumRetryErrorCategory"]
