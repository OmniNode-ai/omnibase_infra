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

    model_config = ConfigDict(frozen=True)
