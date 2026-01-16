# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 OmniNode Team

"""Strongly-typed DSN parse result model.

This module provides a Pydantic model for representing parsed PostgreSQL
Data Source Name (DSN) connection strings with full type safety and
validation.

The model replaces loose dict[str, object] return types with a structured,
immutable, and validated representation of DSN components.

Example:
    >>> from omnibase_infra.types import ModelParsedDSN
    >>> dsn = ModelParsedDSN(
    ...     scheme="postgresql",
    ...     username="admin",
    ...     password="secret",
    ...     hostname="localhost",
    ...     port=5432,
    ...     database="mydb",
    ... )
    >>> dsn.hostname
    'localhost'
    >>> dsn.port
    5432

Note:
    The model is frozen (immutable) to ensure DSN components cannot be
    accidentally modified after parsing. This provides safety when passing
    DSN information through multiple layers of the application.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelParsedDSN"]


class ModelParsedDSN(BaseModel):
    """Strongly-typed DSN parse result for PostgreSQL connection strings.

    This model provides a structured representation of parsed DSN components
    with validation for port ranges and scheme restrictions. The model is
    immutable (frozen) to prevent accidental modification of connection
    parameters.

    Attributes:
        scheme: The database scheme/protocol. Must be 'postgresql' or 'postgres'.
        username: The database username for authentication. None if not specified.
        password: The database password for authentication. None if not specified.
            Note: Handle with care as this contains sensitive credentials.
        hostname: The database server hostname or IP address. None if not specified.
        port: The database server port number (1-65535). None if not specified.
        database: The name of the database to connect to.
        query: Additional connection parameters as key-value pairs. Values may be
            strings or lists of strings for multi-value parameters.

    Example:
        >>> dsn = ModelParsedDSN(
        ...     scheme="postgresql",
        ...     username="app_user",
        ...     hostname="db.example.com",
        ...     port=5432,
        ...     database="production",
        ...     query={"sslmode": "require"},
        ... )
        >>> dsn.scheme
        'postgresql'
        >>> dsn.query
        {'sslmode': 'require'}

    Note:
        The password field should be handled carefully in logging and
        error messages to avoid credential exposure. Use the sanitization
        utilities from util_dsn_validation for safe string representations.
    """

    scheme: Literal["postgresql", "postgres"] = Field(
        description="Database scheme/protocol. Must be 'postgresql' or 'postgres'."
    )
    username: str | None = Field(
        default=None,
        description="Database username for authentication.",
    )
    password: str | None = Field(
        default=None,
        description="Database password for authentication. Handle with care.",
    )
    hostname: str | None = Field(
        default=None,
        description="Database server hostname or IP address.",
    )
    port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Database server port number (valid range: 1-65535).",
    )
    database: str = Field(
        description="Name of the database to connect to.",
    )
    query: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description="Additional connection parameters as key-value pairs.",
    )

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    def __repr__(self) -> str:
        """Return string representation with masked password.

        This method overrides the default Pydantic __repr__ to prevent
        credential exposure in logs, debug output, and error messages.
        The actual password value remains accessible via the .password
        attribute for legitimate use cases.

        Returns:
            String representation with password shown as '[REDACTED]' if set,
            or None if not set.

        Example:
            >>> dsn = ModelParsedDSN(
            ...     scheme="postgresql",
            ...     username="admin",
            ...     password="secret123",
            ...     hostname="localhost",
            ...     port=5432,
            ...     database="mydb",
            ... )
            >>> repr(dsn)  # Shows '[REDACTED]' instead of 'secret123'
            "ModelParsedDSN(scheme='postgresql', username='admin', ...)"
        """
        password_display = "[REDACTED]" if self.password else None
        return (
            f"ModelParsedDSN(scheme={self.scheme!r}, username={self.username!r}, "
            f"password={password_display!r}, hostname={self.hostname!r}, "
            f"port={self.port!r}, database={self.database!r}, query={self.query!r})"
        )

    def __str__(self) -> str:
        """Return string representation with masked password.

        Delegates to __repr__ to ensure consistent password masking across
        all string conversion contexts (str(), print(), f-strings, etc.).

        Returns:
            String representation with password masked.
        """
        return self.__repr__()

    def to_sanitized_dict(self) -> dict[str, object]:
        """Return a dict representation with the password masked.

        Useful for logging and debugging without exposing credentials.

        Returns:
            dict[str, object]: Model data with password replaced by '[REDACTED]' if set.

        Example:
            >>> dsn = ModelParsedDSN(
            ...     scheme="postgresql", username="user", password="secret",
            ...     hostname="localhost", port=5432, database="mydb"
            ... )
            >>> dsn.to_sanitized_dict()
            {'scheme': 'postgresql', 'username': 'user', 'password': '[REDACTED]',
             'hostname': 'localhost', 'port': 5432, 'database': 'mydb', 'query': {}}
        """
        data = self.model_dump()
        if data.get("password"):
            data["password"] = "[REDACTED]"
        return data
