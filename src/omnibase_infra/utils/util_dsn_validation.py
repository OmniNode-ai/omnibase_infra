# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL DSN validation utility.

This module provides robust DSN validation using urllib.parse instead of regex.
It handles edge cases like IPv6 addresses, URL-encoded passwords, query parameters,
and validates DSN structure comprehensively.

Security:
    - Never logs credentials in error messages
    - Returns [REDACTED] in error messages for sensitive values
    - Validates structure without exposing DSN contents
    - Validates database names to prevent injection attacks

Edge Cases Handled:
    - IPv6 addresses: postgresql://user:pass@[::1]:5432/db
    - URL-encoded passwords: user:p%40ssword@host (p@ssword)
    - Missing components: postgresql://localhost/db (no user/pass/port)
    - Query parameters: postgresql://host/db?sslmode=require
    - Unix sockets: postgresql:///db?host=/var/run/postgresql
    - Empty password: user:@host (different from no password)

Limitations:
    - Multiple hosts (host1:5432,host2:5433) are not validated
      (urllib.parse treats them as a single hostname)
    - If multi-host support is needed, use a PostgreSQL-specific parser
"""

from __future__ import annotations

import ipaddress
import re
from typing import Literal, cast
from urllib.parse import parse_qs, unquote, urlparse
from uuid import UUID

from omnibase_infra.types import ModelParsedDSN

# Database name validation pattern
# PostgreSQL allows more characters in quoted identifiers, but for security
# we restrict to a safe subset that prevents injection attacks.
#
# Allowed characters:
# - Letters (a-z, A-Z)
# - Numbers (0-9) - but not as first character
# - Underscores (_)
# - Hyphens (-) - commonly used but requires quoting in SQL
#
# Pattern explanation:
# ^[a-zA-Z_]      - Must start with letter or underscore
# [a-zA-Z0-9_-]*  - Followed by letters, numbers, underscores, or hyphens
# $               - End of string
#
# Max length is 63 characters (PostgreSQL identifier limit)
DATABASE_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,62}$")

# Characters that are explicitly forbidden in database names for security
# These could be used in SQL injection attacks or path traversal
FORBIDDEN_DATABASE_CHARS = frozenset("';\"\\`$(){}[]|&<>!@#%^*=+~/.")


def validate_database_name(
    database: str,
    *,
    correlation_id: UUID | None = None,
) -> str:
    """Validate database name for safety and correctness.

    This function validates that a database name follows safe patterns
    to prevent SQL injection attacks. It enforces PostgreSQL identifier
    conventions while rejecting potentially dangerous characters.

    Args:
        database: The database name to validate.
        correlation_id: Optional correlation ID for distributed tracing.
            If provided, the ID is propagated into error context for
            end-to-end request tracing. If ``None``, a new UUID is
            generated automatically.

    Returns:
        The validated database name if valid.

    Raises:
        ProtocolConfigurationError: If the database name contains
            forbidden characters or doesn't match the safe pattern.

    Security:
        This validation is defense-in-depth. While parameterized queries
        should prevent injection, validating identifiers at parse time
        provides an additional security layer.

    Valid names:
        - mydb, my_database, MyDB, _private
        - db123, test-db, prod_db_v2

    Invalid names:
        - 123db (starts with number)
        - my;db (contains semicolon - SQL injection risk)
        - my db (contains space)
        - my'db (contains quote - SQL injection risk)
        - empty string
        - names longer than 63 characters

    Example:
        >>> validate_database_name("mydb")
        'mydb'
        >>> validate_database_name("my;db")  # Raises ProtocolConfigurationError
    """
    # Lazy imports to avoid circular dependency
    from omnibase_infra.enums import EnumInfraTransportType
    from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

    context = ModelInfraErrorContext.with_correlation(
        correlation_id=correlation_id,
        transport_type=EnumInfraTransportType.DATABASE,
        operation="validate_database_name",
        target_name="dsn_validator",
    )

    # Check for forbidden characters first (provides clearer error message)
    forbidden_found = [char for char in database if char in FORBIDDEN_DATABASE_CHARS]
    if forbidden_found:
        # Don't expose the actual characters in production logs
        raise ProtocolConfigurationError(
            "Database name contains forbidden characters that could enable "
            "SQL injection. Use only letters, numbers, underscores, and hyphens.",
            context=context,
            parameter="dsn.database",
            value="[REDACTED]",
        )

    # Check for whitespace
    if any(c.isspace() for c in database):
        raise ProtocolConfigurationError(
            "Database name contains whitespace characters. "
            "Use only letters, numbers, underscores, and hyphens.",
            context=context,
            parameter="dsn.database",
            value="[REDACTED]",
        )

    # Check against the safe pattern
    if not DATABASE_NAME_PATTERN.match(database):
        raise ProtocolConfigurationError(
            "Database name must start with a letter or underscore, "
            "contain only letters, numbers, underscores, and hyphens, "
            "and be at most 63 characters long.",
            context=context,
            parameter="dsn.database",
            value="[REDACTED]",
        )

    return database


def _assert_postgres_scheme(
    scheme: str,
    *,
    correlation_id: UUID | None = None,
) -> Literal["postgresql", "postgres"]:
    """Type-safe scheme assertion for PostgreSQL DSN schemes.

    This helper enables proper type narrowing for the Literal type
    using typing.cast for explicit type assertion.

    Args:
        scheme: The scheme string to validate.
        correlation_id: Optional correlation ID for distributed tracing.
            If provided, the ID is propagated into error context. If
            ``None``, a new UUID is generated automatically.

    Returns:
        The scheme cast to the appropriate Literal type.

    Raises:
        ProtocolConfigurationError: If scheme is not 'postgresql' or 'postgres'.

    Note:
        This function should only be called AFTER validating
        that scheme is one of the valid values.
    """
    if scheme not in ("postgresql", "postgres"):
        # Lazy imports to avoid circular dependency (utils -> errors -> models -> utils)
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import (
            ModelInfraErrorContext,
            ProtocolConfigurationError,
        )

        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.DATABASE,
            operation="validate_dsn_scheme",
            target_name="dsn_validator",
        )
        raise ProtocolConfigurationError(
            f"Invalid scheme: expected 'postgresql' or 'postgres', got '{scheme}'",
            context=context,
            parameter="scheme",
            value=scheme,
        )
    return cast("Literal['postgresql', 'postgres']", scheme)


def parse_and_validate_dsn(
    dsn: object,
    *,
    correlation_id: UUID | None = None,
) -> ModelParsedDSN:
    """Parse and validate PostgreSQL DSN format using urllib.parse.

    This function provides comprehensive DSN validation that handles edge cases
    like IPv6 addresses, URL-encoded passwords, and query parameters. It uses
    urllib.parse for robust parsing instead of fragile regex patterns.

    Args:
        dsn: PostgreSQL connection string (any type - validated).
        correlation_id: Optional correlation ID for distributed tracing.
            If provided, the ID is propagated into error context and
            through to all sub-validation calls (database name validation,
            scheme assertion) for end-to-end request tracing. If ``None``,
            a new UUID is generated automatically.

    Returns:
        ModelParsedDSN with parsed components:
            - scheme: "postgresql" or "postgres" (Literal type)
            - username: Username or None
            - password: Password (URL-decoded) or None
            - hostname: Hostname or IP address or None
            - port: Port number or None
            - database: Database name
            - query: Dict of query parameters (str keys, str | list[str] values)

    Raises:
        ProtocolConfigurationError: If DSN format is invalid.

    Example:
        >>> result = parse_and_validate_dsn("postgresql://user:pass@localhost:5432/mydb")
        >>> assert result.hostname == "localhost"
        >>> assert result.database == "mydb"

    Security Note:
        Error messages never contain the actual DSN value. Sensitive information
        is replaced with [REDACTED] to prevent credential leakage in logs.
    """
    # Lazy imports to avoid circular dependency (utils -> errors -> models -> utils)
    from omnibase_infra.enums import EnumInfraTransportType
    from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

    context = ModelInfraErrorContext.with_correlation(
        correlation_id=correlation_id,
        transport_type=EnumInfraTransportType.DATABASE,
        operation="validate_dsn",
        target_name="dsn_validator",
    )

    # Capture the resolved correlation_id so sub-functions share the same trace
    resolved_correlation_id = context.correlation_id

    # Type validation
    if dsn is None:
        raise ProtocolConfigurationError(
            "Invalid dsn: expected a string, got None",
            context=context,
            parameter="dsn",
            value=None,
        )

    if not isinstance(dsn, str):
        raise ProtocolConfigurationError(
            f"Invalid dsn type: expected str, got {type(dsn).__name__}",
            context=context,
            parameter="dsn",
            value=type(dsn).__name__,
        )

    # Empty string validation
    if not dsn.strip():
        raise ProtocolConfigurationError(
            "Invalid dsn: expected a non-empty string, got empty string",
            context=context,
            parameter="dsn",
            value="",
        )

    dsn_str = dsn.strip()

    # Scheme validation (before parsing to provide clear error)
    valid_prefixes = ("postgresql://", "postgres://")
    if not dsn_str.startswith(valid_prefixes):
        raise ProtocolConfigurationError(
            f"dsn must start with one of {valid_prefixes}",
            context=context,
            parameter="dsn",
            value="[REDACTED]",  # Never log DSN contents
        )

    # Check for multi-host DSN format (not supported)
    # Multi-host format: postgresql://host1:port1,host2:port2/db
    # We check for comma after :// and before / (in the host portion)
    scheme_end = dsn_str.index("://") + 3
    path_start = dsn_str.find("/", scheme_end)
    host_portion = (
        dsn_str[scheme_end:path_start] if path_start != -1 else dsn_str[scheme_end:]
    )
    if "," in host_portion:
        raise ProtocolConfigurationError(
            "Multi-host DSNs are not supported. Use a single host or consider "
            "psycopg2.conninfo_to_dict for multi-host parsing.",
            context=context,
            parameter="dsn",
            value="[REDACTED]",
        )

    # Parse DSN using urllib.parse
    try:
        parsed = urlparse(dsn_str)
    except ValueError as e:
        # urlparse can raise ValueError for invalid ports, etc.
        raise ProtocolConfigurationError(
            f"Invalid DSN format: {e}",
            context=context,
            parameter="dsn",
            value="[REDACTED]",
        ) from e

    # Validate scheme (redundant but explicit)
    if parsed.scheme not in ("postgresql", "postgres"):
        raise ProtocolConfigurationError(
            f"Invalid scheme: expected 'postgresql' or 'postgres', got '{parsed.scheme}'",
            context=context,
            parameter="dsn",
            value="[REDACTED]",
        )

    # Validate port if present
    # Note: Accessing parsed.port can raise ValueError if port is invalid
    try:
        port = parsed.port
        if port is not None:
            if port < 1 or port > 65535:
                raise ProtocolConfigurationError(
                    f"Port must be between 1 and 65535, got {port}",
                    context=context,
                    parameter="dsn.port",
                    value=port,
                )
    except ValueError as e:
        # urlparse raises ValueError for invalid ports (non-numeric, out of range)
        raise ProtocolConfigurationError(
            f"Invalid port in DSN: {e}",
            context=context,
            parameter="dsn.port",
            value="[REDACTED]",
        ) from e

    # Validate database name
    # Path is "/dbname" or "///dbname" for Unix socket
    # Strip leading slashes to get database name
    database = parsed.path.lstrip("/") if parsed.path else ""

    # Database name is required (unless using Unix socket with query param)
    # For Unix sockets: postgresql:///dbname?host=/var/run/postgresql
    # For network: postgresql://host/dbname
    if not database:
        # Check if Unix socket is specified in query params
        query_params = parse_qs(parsed.query)
        if "host" not in query_params:
            raise ProtocolConfigurationError(
                "Database name is required in DSN path (e.g., postgresql://host:5432/dbname)",
                context=context,
                parameter="dsn.path",
                value="[REDACTED]",
            )
        # Unix socket case - database might be in query params or path
        # Allow empty database for now (will be validated at connection time)
    else:
        # Validate database name for security (prevent injection attacks)
        # This raises ProtocolConfigurationError if invalid
        validate_database_name(database, correlation_id=resolved_correlation_id)

    # Parse query parameters
    query_dict = {}
    if parsed.query:
        # parse_qs returns lists, flatten to single values
        parsed_qs = parse_qs(parsed.query)
        query_dict = {k: v[0] if len(v) == 1 else v for k, v in parsed_qs.items()}

    # Return parsed components
    # Note: urlparse does NOT decode URL-encoded passwords, so we use unquote()
    # Important: Check 'is not None' instead of truthiness to preserve empty strings
    return ModelParsedDSN(
        scheme=_assert_postgres_scheme(
            parsed.scheme, correlation_id=resolved_correlation_id
        ),
        username=unquote(parsed.username) if parsed.username is not None else None,
        password=unquote(parsed.password) if parsed.password is not None else None,
        hostname=parsed.hostname,
        port=port,
        database=database,
        query=query_dict,
    )


def sanitize_dsn(dsn: str) -> str:
    """Sanitize DSN by removing password for safe logging.

    SECURITY: This function should ONLY be used for development/debugging.
    Production code should NEVER log DSN values, even sanitized ones.

    Replaces the password portion of the DSN with asterisks using URL parsing
    instead of regex for robustness.

    Args:
        dsn: Raw PostgreSQL connection string containing credentials

    Returns:
        Sanitized DSN with password replaced by '***'

    Example:
        >>> sanitize_dsn("postgresql://user:secret@host:5432/db")
        'postgresql://user:***@host:5432/db'

        >>> sanitize_dsn("postgresql://user:p%40ss@host/db")
        'postgresql://user:***@host/db'

    Note:
        This function is intentionally NOT used in production error paths.
        It exists as a utility for development/debugging only.
    """
    try:
        parsed = urlparse(dsn)

        # Rebuild DSN with password masked
        # Format: scheme://[user[:password]@]host[:port]/path[?query][#fragment]
        netloc = parsed.hostname or ""

        # Add port if present (handle ValueError from invalid ports)
        try:
            port = parsed.port
            if port:
                netloc = f"{netloc}:{port}"
        except ValueError:
            # Invalid port - include raw port string from netloc
            # Extract port from raw netloc if present
            if ":" in (parsed.netloc or ""):
                # Keep original port notation even if invalid
                parts = parsed.netloc.split("@")[-1]  # Get host:port part
                if ":" in parts:
                    netloc = parts

        # Add username with masked password
        if parsed.username:
            if parsed.password:
                netloc = f"{parsed.username}:***@{netloc}"
            else:
                netloc = f"{parsed.username}@{netloc}"

        # Reconstruct URL
        sanitized = f"{parsed.scheme}://{netloc}{parsed.path}"

        # Add query string if present
        if parsed.query:
            sanitized = f"{sanitized}?{parsed.query}"

        # Add fragment if present
        if parsed.fragment:
            sanitized = f"{sanitized}#{parsed.fragment}"

        return sanitized

    except Exception:
        # If parsing fails, return a safe placeholder
        return "[INVALID_DSN]"


def is_private_ip(hostname: str) -> bool:
    """Check whether a hostname is a private or non-routable IP address.

    This function determines if a given hostname string represents a
    private, loopback, link-local, or otherwise non-routable IP address.
    It covers all standard RFC 1918 private ranges as well as additional
    non-routable ranges.

    Supported ranges:
        - **RFC 1918 private**: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
        - **Loopback**: 127.0.0.0/8 (IPv4), ::1 (IPv6)
        - **Link-local**: 169.254.0.0/16 (IPv4), fe80::/10 (IPv6)
        - **IPv6 private**: fc00::/7 (unique local addresses)

    If the hostname is not a valid IP address (e.g., a DNS hostname like
    ``db.example.com``), the function returns ``False`` since it cannot
    determine routability without DNS resolution.

    Args:
        hostname: The hostname or IP address string to check. May be an
            IPv4 address, IPv6 address, or DNS hostname.

    Returns:
        ``True`` if the hostname is a recognized private or non-routable
        IP address. ``False`` if it is a public IP address or if it is
        not a valid IP address (e.g., a DNS hostname).

    Example:
        >>> is_private_ip("192.168.1.100")
        True
        >>> is_private_ip("10.0.0.1")
        True
        >>> is_private_ip("172.16.0.1")
        True
        >>> is_private_ip("8.8.8.8")
        False
        >>> is_private_ip("db.example.com")
        False
        >>> is_private_ip("127.0.0.1")
        True
        >>> is_private_ip("::1")
        True
    """
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        # Not a valid IP address (likely a DNS hostname) -- cannot determine
        # routability without resolution, so return False conservatively.
        return False

    return addr.is_private or addr.is_loopback or addr.is_link_local


__all__: list[str] = [
    "DATABASE_NAME_PATTERN",
    "FORBIDDEN_DATABASE_CHARS",
    "ModelParsedDSN",
    "is_private_ip",
    "parse_and_validate_dsn",
    "sanitize_dsn",
    "validate_database_name",
]
