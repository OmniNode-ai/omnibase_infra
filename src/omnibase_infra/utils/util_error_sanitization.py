# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Error message sanitization utilities.

This module provides functions to sanitize error messages before they are:
- Published to Dead Letter Queues (DLQ)
- Logged to external systems
- Included in API responses
- Stored in metrics or monitoring systems

Sanitization protects against leaking sensitive data such as:
- Passwords and credentials
- API keys and tokens
- Connection strings with embedded credentials
- Private keys and certificates

ONEX Error Sanitization Guidelines:
    NEVER include: Passwords, API keys, tokens, PII, credentials
    SAFE to include: Error types, correlation IDs, topic names, timestamps

See Also:
    docs/patterns/error_sanitization_patterns.md - Comprehensive sanitization guide
    docs/architecture/DLQ_MESSAGE_FORMAT.md - DLQ security considerations

Example:
    >>> from omnibase_infra.utils import sanitize_error_message
    >>> try:
    ...     raise ValueError("Auth failed with password=secret123")
    ... except Exception as e:
    ...     safe_msg = sanitize_error_message(e)
    >>> "password" not in safe_msg.lower()
    True
    >>> "secret123" not in safe_msg
    True
"""

from __future__ import annotations

# Patterns that may indicate sensitive data in error messages.
# These patterns are checked case-insensitively against the error message.
# When matched, the error message is redacted to prevent credential leakage.
SENSITIVE_PATTERNS: tuple[str, ...] = (
    # Credentials
    "password",
    "passwd",
    "pwd",
    # Secrets and keys
    "secret",
    "token",
    "api_key",
    "apikey",
    "api-key",
    "access_key",
    "accesskey",
    "access-key",
    "private_key",
    "privatekey",
    "private-key",
    # Authentication
    "credential",
    "auth",
    "bearer",
    "authorization",
    # Connection strings (often contain credentials)
    "connection_string",
    "connectionstring",
    "connection-string",
    "conn_str",
    "connstr",
    # Common credential parameter names
    "user:pass",
    "username:password",
    # AWS-specific
    "aws_secret",
    "aws_access",
    # Database-specific
    "db_password",
    "database_password",
    "pgpassword",
    "mysql_pwd",
    # Certificate and key material
    "-----begin",  # PEM format headers
    "-----end",
    # Database connection URI schemes (often contain credentials)
    "mongodb://",
    "postgres://",
    "postgresql://",
    "mysql://",
    "redis://",
    "rediss://",  # Redis with TLS
    "valkey://",  # Valkey (Redis-compatible)
    "amqp://",
    "kafka://",
)


def sanitize_error_string(error_str: str, max_length: int = 500) -> str:
    """Sanitize a raw error string for safe inclusion in logs and responses.

    This function removes or masks potentially sensitive information from
    error message strings. Use this when you have a raw error string rather
    than an exception object.

    SECURITY NOTE: This function is REQUIRED when logging error messages from
    Redis/Valkey connections, as these may include connection strings with
    embedded credentials.

    Sanitization rules:
        1. Check for common patterns indicating credentials/connection strings
        2. If sensitive patterns detected, return generic redacted message
        3. Truncate long messages to prevent excessive data exposure

    Args:
        error_str: The error string to sanitize
        max_length: Maximum length of the sanitized message (default 500)

    Returns:
        Sanitized error message safe for storage and logging.

    Example:
        >>> safe_msg = sanitize_error_string("Connection failed: redis://user:pass@host:6379")
        >>> "user:pass" not in safe_msg
        True
        >>> "[REDACTED" in safe_msg
        True
    """
    if not error_str:
        return ""

    # Check for sensitive patterns in the error message (case-insensitive)
    error_lower = error_str.lower()
    for pattern in SENSITIVE_PATTERNS:
        if pattern in error_lower:
            # Sensitive data detected - return generic message
            return "[REDACTED - potentially sensitive data]"

    # Truncate long messages to prevent data leakage through verbose errors
    if len(error_str) > max_length:
        return error_str[:max_length] + "... [truncated]"

    return error_str


def sanitize_error_message(exception: Exception, max_length: int = 500) -> str:
    """Sanitize an exception message for safe inclusion in logs, DLQ, and responses.

    This function removes or masks potentially sensitive information from
    exception messages before they are stored, logged, or published to DLQ.

    Sanitization rules:
        1. Check for common patterns indicating credentials/connection strings
        2. If sensitive patterns detected, return only the exception type
        3. Truncate long messages to prevent excessive data exposure
        4. Return sanitized error message with type prefix

    Args:
        exception: The exception to sanitize
        max_length: Maximum length of the sanitized message (default 500)

    Returns:
        Sanitized error message safe for storage and logging.
        Format: "{ExceptionType}: {sanitized_message}"

    Example:
        >>> try:
        ...     raise ValueError("Failed with password=secret123")
        ... except Exception as e:
        ...     safe_msg = sanitize_error_message(e)
        >>> "password" not in safe_msg.lower()
        True
        >>> safe_msg.startswith("ValueError:")
        True

        >>> try:
        ...     raise ConnectionError("Cannot connect to postgres://user:pass@db:5432")
        ... except Exception as e:
        ...     safe_msg = sanitize_error_message(e)
        >>> "user:pass" not in safe_msg
        True
        >>> "[REDACTED" in safe_msg
        True

    Note:
        This function is designed to be conservative - it will redact messages
        that might not actually contain credentials rather than risk exposing
        sensitive data. This follows the security principle of "when in doubt,
        redact."
    """
    exception_type = type(exception).__name__
    exception_str = str(exception)

    # Check for sensitive patterns in the exception message (case-insensitive)
    exception_lower = exception_str.lower()
    for pattern in SENSITIVE_PATTERNS:
        if pattern in exception_lower:
            # Sensitive data detected - return only the exception type
            return f"{exception_type}: [REDACTED - potentially sensitive data]"

    # Truncate long messages to prevent data leakage through verbose errors
    if len(exception_str) > max_length:
        exception_str = exception_str[:max_length] + "... [truncated]"

    return f"{exception_type}: {exception_str}"


__all__: list[str] = [
    "SENSITIVE_PATTERNS",
    "sanitize_error_message",
    "sanitize_error_string",
]
