# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL connection pool configuration.

Part of OMN-1976: Contract dependency materialization.
Updated in OMN-2065: Per-service DB URL contract (DB-SPLIT-02).
"""

from __future__ import annotations

import logging
import os
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


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
        ...,
        description=(
            "PostgreSQL database name (required — no default). "
            "Use from_env() or from_dsn() factories which always provide this."
        ),
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

    @staticmethod
    def validate_dsn(dsn: str) -> str:
        """Validate a PostgreSQL DSN string (scheme, database name, sub-paths).

        Use this to validate a DSN without creating a full config object.
        Callers that need a different exception type should catch ``ValueError``
        and re-raise.

        Args:
            dsn: PostgreSQL connection string to validate.

        Returns:
            The validated (stripped) DSN string.

        Raises:
            ValueError: If the DSN has an invalid scheme, is missing a database
                name, or contains sub-paths in the database name.
        """
        dsn = dsn.strip()
        # Security: error messages omit credentials. Hostname/port are included
        # for diagnostics but are not considered secret in this context.
        parsed = urlparse(dsn)

        if parsed.scheme not in ("postgresql", "postgres"):
            msg = (
                f"Invalid DSN scheme '{parsed.scheme}', "
                f"expected 'postgresql' or 'postgres'"
            )
            raise ValueError(msg)

        database = (parsed.path or "").lstrip("/")
        if not database:
            try:
                port_str = str(parsed.port) if parsed.port else "?"
            except ValueError:
                port_str = "?"
            safe_dsn = f"{parsed.scheme}://{parsed.hostname or '?'}:{port_str}/???"
            msg = f"DSN is missing a database name: {safe_dsn}"
            raise ValueError(msg)
        if "/" in database:
            msg = (
                f"Invalid database name '{database}' extracted from DSN: "
                "sub-paths are not valid PostgreSQL database names"
            )
            raise ValueError(msg)

        return dsn

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
        dsn = cls.validate_dsn(dsn)

        parsed = urlparse(dsn)
        database = (parsed.path or "").lstrip("/")

        # DSN query params (sslmode, options, connect_timeout, etc.) are not
        # reflected in the returned config fields. Log a warning so operators
        # are aware of how parameters are (or are not) preserved.
        if parsed.query:
            logger.warning(
                "DSN query parameters detected but will not be reflected in "
                "config fields: %s. If the original DSN is passed directly to "
                "asyncpg, parameters will be preserved; if the DSN is "
                "reconstructed from config fields, parameters will be lost. "
                "To enforce SSL explicitly, configure it at the asyncpg pool level.",
                parsed.query,
            )
        #
        # NOTE: Missing password defaults to "" (the field default). This is
        # intentional — from_env() is the production entry point and requires a
        # fully-formed DSN with credentials.  from_dsn() is a lower-level
        # parser that tolerates password-less DSNs for dev/test flexibility.
        # NOTE: Credentials are stored *decoded* (unquote). If a DSN is later
        # reconstructed from these fields, the values must be re-encoded with
        # urllib.parse.quote_plus() to produce a valid connection string.
        # NOTE: parsed.hostname returns None for Unix-socket DSNs
        # (e.g., "postgresql:///dbname"). The fallback to "localhost" means
        # Unix-socket DSNs are silently rewritten to TCP connections.
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
