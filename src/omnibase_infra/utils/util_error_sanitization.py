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
    # SSH credentials
    "ssh_key",
    "sshkey",
    "ssh-key",
    # OAuth/OIDC credentials
    "client_secret",
    "clientsecret",
    "client-secret",
    "access_token",
    "accesstoken",
    "access-token",
    "refresh_token",
    "refreshtoken",
    "refresh-token",
    "id_token",
    "idtoken",
    "id-token",
    # Session identifiers (hijacking prevention)
    "session_id",
    "sessionid",
    "session-id",
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
    # Vault-specific
    "vault_token",
    "vaulttoken",
    "vault-token",
    "x-vault-token",
    # Consul-specific
    "consul_token",
    "consultoken",
    "consul-token",
    "consul_http_token",
    "consul-http-token",
    "x-consul-token",
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


# Safe error patterns that don't contain secrets.
# These are checked in order - longer/more specific patterns should come first
# to ensure they match before shorter substrings.
# Used by sanitize_backend_error to extract safe portions of error messages.
SAFE_ERROR_PATTERNS: tuple[str, ...] = (
    # Connection patterns (longer first)
    "connection refused",
    "connection reset",
    "connection timeout",
    "connection closed",
    # Network patterns
    "network unreachable",
    "host not found",
    "dns lookup failed",
    # Availability patterns
    "service unavailable",
    "too many connections",
    "resource exhausted",
    # Auth patterns (type only, not details)
    "authentication failed",
    "permission denied",
    "access denied",
    # State patterns
    "already exists",
    "not found",
    "conflict",
    # Generic patterns (last, most generic)
    "timeout",
    "unavailable",
)


def sanitize_backend_error(backend_name: str, raw_error: object) -> str:
    """Sanitize a backend error message to avoid exposing secrets.

    Backend error messages (from Consul, PostgreSQL, etc.) may contain
    sensitive information like connection strings, credentials, or internal
    hostnames. This function extracts only safe, generic error information.

    This function uses an allowlist approach: it looks for known safe patterns
    and only includes those. Unknown error content is replaced with a generic
    message.

    Args:
        backend_name: Name of the backend (e.g., "consul", "postgres").
        raw_error: Raw error from the backend (string, exception, or any object).

    Returns:
        Sanitized error message safe for logging and user-facing responses.
        Format: "{backend_name} operation failed" or
                "{backend_name} operation failed: {safe_pattern}"

    Examples:
        >>> sanitize_backend_error("postgres", "connection refused")
        'postgres operation failed: connection refused'

        >>> sanitize_backend_error("consul", "auth failed: password=secret123")
        'consul operation failed'

        >>> sanitize_backend_error("consul", None)
        'consul operation failed'

        >>> sanitize_backend_error("postgres", {"error": "timeout"})
        'postgres operation failed: timeout'

    Security:
        This function is intentionally conservative. It only includes error
        patterns that are known to be safe. Any unrecognized error content
        is replaced with a generic message to prevent accidental secret exposure.
    """
    if raw_error is None:
        return f"{backend_name} operation failed"

    # Convert to string for analysis
    error_str = str(raw_error).lower().strip()

    if not error_str:
        return f"{backend_name} operation failed"

    # Check for safe, generic error patterns (checked in order - first match wins)
    for safe_pattern in SAFE_ERROR_PATTERNS:
        if safe_pattern in error_str:
            return f"{backend_name} operation failed: {safe_pattern}"

    # Default: don't expose the raw error, use generic message
    return f"{backend_name} operation failed"


def sanitize_secret_path(path: str | None) -> str | None:
    """Sanitize a Vault secret path to avoid exposing infrastructure details.

    Secret paths often reveal infrastructure topology, application names,
    and credential organization. This function masks sensitive portions
    while preserving enough information for debugging.

    Sanitization rules:
        1. If path is None or empty, return as-is
        2. Preserve the mount point (first segment)
        3. Mask subsequent path segments with asterisks
        4. Preserve the final segment (key name) in generic form

    Args:
        path: The secret path to sanitize (e.g., "secret/data/myapp/database/creds")

    Returns:
        Sanitized path safe for error messages and logging.
        Format: "{mount}/***/***" or "{mount}/***/***/{leaf}" for deep paths

    Examples:
        >>> sanitize_secret_path("secret/data/myapp/database/credentials")
        'secret/***/***'

        >>> sanitize_secret_path("kv/production/api-keys/stripe")
        'kv/***/***'

        >>> sanitize_secret_path("secret")
        'secret'

        >>> sanitize_secret_path(None)

        >>> sanitize_secret_path("")
        ''

    Security:
        This function prevents exposure of:
        - Application names and environments (e.g., "production", "myapp")
        - Service and database names (e.g., "postgres", "redis")
        - Credential types and purposes (e.g., "api-keys", "certificates")
    """
    if path is None:
        return None

    if not path:
        return ""

    # Split path into segments
    segments = path.split("/")

    if len(segments) <= 1:
        # Just mount point or single segment - return as-is
        return path

    # Keep first segment (mount point like "secret", "kv", etc.)
    # Replace everything else with masked indicator
    mount = segments[0]
    return f"{mount}/***/***"


def sanitize_consul_key(key: str | None) -> str | None:
    """Sanitize a Consul key path to avoid exposing infrastructure details.

    Consul keys often reveal infrastructure topology, service names,
    configuration structures, and potentially sensitive data paths.
    This function masks sensitive portions while preserving enough
    information for debugging.

    Sanitization rules:
        1. If key is None or empty, return as-is
        2. Preserve the first segment (typically namespace/service type)
        3. Mask subsequent path segments with asterisks
        4. Indicate depth with masked segments

    Args:
        key: The Consul key path to sanitize (e.g., "config/database/connection")

    Returns:
        Sanitized key safe for error messages and logging.
        Format: "{prefix}/***/***" for multi-segment keys

    Examples:
        >>> sanitize_consul_key("config/database/connection")
        'config/***/***'

        >>> sanitize_consul_key("services/api-gateway/endpoints")
        'services/***/***'

        >>> sanitize_consul_key("config")
        'config'

        >>> sanitize_consul_key(None)

        >>> sanitize_consul_key("")
        ''

    Security:
        This function prevents exposure of:
        - Service and application names (e.g., "api-gateway", "user-service")
        - Database and infrastructure names (e.g., "postgres-primary")
        - Configuration paths that reveal architecture
        - Environment identifiers (e.g., "production", "staging")
    """
    if key is None:
        return None

    if not key:
        return ""

    # Split key into segments
    segments = key.split("/")

    if len(segments) <= 1:
        # Just prefix or single segment - return as-is
        return key

    # Keep first segment (namespace like "config", "services", etc.)
    # Replace everything else with masked indicator
    prefix = segments[0]
    return f"{prefix}/***/***"


__all__: list[str] = [
    "SAFE_ERROR_PATTERNS",
    "SENSITIVE_PATTERNS",
    "sanitize_backend_error",
    "sanitize_consul_key",
    "sanitize_error_message",
    "sanitize_error_string",
    "sanitize_secret_path",
]
