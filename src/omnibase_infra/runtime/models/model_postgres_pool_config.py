# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL connection pool configuration.

Part of OMN-1976: Contract dependency materialization.
Updated in OMN-2065: Per-service DB URL contract (DB-SPLIT-02).
"""

from __future__ import annotations

import os
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelPostgresPoolConfig(BaseModel):
    """PostgreSQL connection pool configuration.

    Sources configuration from a ``*_DB_URL`` environment variable.
    Fail-fast: raises ``ValueError`` when the required URL is missing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    user: str = Field(default="postgres", description="PostgreSQL user")
    password: str = Field(
        default="",
        repr=False,
        description="PostgreSQL password (never logged or included in repr)",
    )
    database: str = Field(
        description="PostgreSQL database name (required; use from_env() or from_dsn() factories)",
    )
    min_size: int = Field(default=2, ge=1, le=100, description="Minimum pool size")
    max_size: int = Field(default=10, ge=1, le=100, description="Maximum pool size")

    @model_validator(mode="after")
    def _check_pool_size_bounds(self) -> ModelPostgresPoolConfig:
        """Validate that min_size does not exceed max_size.

        Raises:
            ValueError: If min_size is greater than max_size.
        """
        if self.min_size > self.max_size:
            msg = (
                f"min_size ({self.min_size}) must not exceed max_size ({self.max_size})"
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_env(
        cls,
        db_url_var: str = "OMNIBASE_INFRA_DB_URL",
    ) -> ModelPostgresPoolConfig:
        """Create config from a ``*_DB_URL`` environment variable.

        Parses the DSN to extract host, port, user, password, and database.
        Pool-size overrides are still read from ``POSTGRES_POOL_*`` env vars.

        Args:
            db_url_var: Name of the environment variable holding the DSN.
                Defaults to ``OMNIBASE_INFRA_DB_URL``.

        Raises:
            ValueError: If the environment variable is not set (fail-fast)
                or contains an invalid DSN.
        """
        db_url = os.getenv(db_url_var)
        if db_url is not None:
            db_url = db_url.strip()
        if not db_url:
            msg = (
                f"{db_url_var} is required but not set. "
                f"Set it to a PostgreSQL DSN, e.g. "
                f"postgresql://user:pass@host:5432/dbname"
            )
            raise ValueError(msg)

        min_size_raw = os.getenv("POSTGRES_POOL_MIN_SIZE", "2")
        max_size_raw = os.getenv("POSTGRES_POOL_MAX_SIZE", "10")
        try:
            min_size = int(min_size_raw)
        except ValueError as e:
            msg = f"POSTGRES_POOL_MIN_SIZE must be an integer, got '{min_size_raw}'"
            raise ValueError(msg) from e
        try:
            max_size = int(max_size_raw)
        except ValueError as e:
            msg = f"POSTGRES_POOL_MAX_SIZE must be an integer, got '{max_size_raw}'"
            raise ValueError(msg) from e

        return cls.from_dsn(
            db_url,
            min_size=min_size,
            max_size=max_size,
        )

    @classmethod
    def from_dsn(
        cls,
        dsn: str,
        *,
        min_size: int = 2,
        max_size: int = 10,
    ) -> ModelPostgresPoolConfig:
        """Create config by parsing a PostgreSQL DSN string.

        Args:
            dsn: PostgreSQL connection string
                (``postgresql://user:pass@host:port/database``).
            min_size: Minimum pool size.
            max_size: Maximum pool size.

        Raises:
            ValueError: If the DSN is malformed or missing required parts.
        """
        # Security: all error messages below are sanitised to prevent credential leaks.
        # - Invalid scheme: shows only scheme prefix (no host/user/password)
        # - Missing database: shows scheme://host:port/??? (password omitted)
        parsed = urlparse(dsn)

        if parsed.scheme not in ("postgresql", "postgres"):
            msg = (
                f"Invalid DSN scheme '{parsed.scheme}', "
                f"expected 'postgresql' or 'postgres'"
            )
            raise ValueError(msg)

        database = (parsed.path or "").lstrip("/")
        if not database:
            # Sanitise DSN to avoid leaking credentials in error messages
            safe_dsn = (
                f"{parsed.scheme}://{parsed.hostname or '?'}:{parsed.port or '?'}/???"
            )
            msg = f"DSN is missing a database name: {safe_dsn}"
            raise ValueError(msg)

        # TODO(OMN-2065): DSN query params (sslmode, options, etc.) are currently
        # discarded during parsing. If needed, add a `query_params: str` field.
        #
        # NOTE: Missing password defaults to "" (the field default). This is
        # intentional — from_env() is the production entry point and requires a
        # fully-formed DSN with credentials.  from_dsn() is a lower-level
        # parser that tolerates password-less DSNs for dev/test flexibility.
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            user=unquote(parsed.username) if parsed.username else "postgres",
            password=unquote(parsed.password) if parsed.password else "",
            database=database,
            min_size=min_size,
            max_size=max_size,
        )


__all__ = ["ModelPostgresPoolConfig"]
