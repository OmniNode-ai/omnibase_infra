# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""SQL Identifier Utilities for Projector Models.

Provides shared utilities for validating and quoting PostgreSQL identifiers
used across projector model classes. Centralizes SQL safety logic to prevent
injection vulnerabilities.

Related Tickets:
    - OMN-1168: ProjectorPluginLoader contract discovery loading
"""

from __future__ import annotations

import re
from uuid import uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

# Valid PostgreSQL identifier pattern: starts with letter or underscore,
# followed by letters, digits, or underscores
IDENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_valid_identifier(name: str) -> bool:
    """Check if a name is a valid PostgreSQL identifier.

    Args:
        name: Identifier to validate.

    Returns:
        True if the identifier matches the safe pattern, False otherwise.
    """
    return bool(IDENT_PATTERN.match(name))


def quote_identifier(ident: str) -> str:
    """Quote a PostgreSQL identifier safely.

    Escapes embedded double quotes by doubling them and wraps the
    identifier in double quotes.

    Args:
        ident: Identifier to quote.

    Returns:
        Safely quoted identifier.

    Example:
        >>> quote_identifier("entity_id")
        '"entity_id"'
        >>> quote_identifier('weird"name')
        '"weird""name"'
    """
    escaped = ident.replace('"', '""')
    return f'"{escaped}"'


def validate_identifier(name: str, context_name: str = "identifier") -> str:
    """Validate that a name is a valid PostgreSQL identifier.

    Args:
        name: Identifier to validate.
        context_name: Description of what's being validated (for error messages).

    Returns:
        The validated identifier.

    Raises:
        ProtocolConfigurationError: If the identifier contains invalid characters.
    """
    if not IDENT_PATTERN.match(name):
        error_context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="validate_identifier",
            correlation_id=uuid4(),
        )
        raise ProtocolConfigurationError(
            f"Invalid {context_name} '{name}': must match pattern "
            "[A-Za-z_][A-Za-z0-9_]* (letters, digits, underscores only, "
            "starting with letter or underscore)",
            context=error_context,
        )
    return name


def escape_sql_string(value: str) -> str:
    """Escape a string for use in SQL string literals.

    Escapes single quotes by doubling them for safe use in SQL
    string contexts (e.g., COMMENT ON statements).

    Args:
        value: String value to escape.

    Returns:
        Escaped string suitable for SQL string literal.

    Example:
        >>> escape_sql_string("User's name")
        "User''s name"
        >>> escape_sql_string("normal text")
        "normal text"
    """
    return value.replace("'", "''")


__all__ = [
    "IDENT_PATTERN",
    "escape_sql_string",
    "is_valid_identifier",
    "quote_identifier",
    "validate_identifier",
]
