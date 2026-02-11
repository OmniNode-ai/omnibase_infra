# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL connection pool configuration.

Part of OMN-1976: Contract dependency materialization.
Part of OMN-2146: Per-service DB URL contract.

DB URL Contract
---------------
The preferred configuration method is a single ``OMNIBASE_INFRA_DB_URL``
environment variable containing a full PostgreSQL DSN::

    OMNIBASE_INFRA_DB_URL=postgresql://role_omnibase_infra:<pw>@<host>:5432/omnibase_infra

``from_env()`` reads ``OMNIBASE_INFRA_DB_URL`` first.  If unset it falls
back to individual ``POSTGRES_*`` variables **but raises** if no database
name can be resolved (no silent fallback to a shared database).
"""

from __future__ import annotations

import logging
import os
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class ModelPostgresPoolConfig(BaseModel):
    """PostgreSQL connection pool configuration.

    Sources configuration from ``OMNIBASE_INFRA_DB_URL`` (preferred) or
    individual ``POSTGRES_*`` environment variables.
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
    database: str = Field(default="", description="PostgreSQL database")
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

    @model_validator(mode="after")
    def _warn_empty_database(self) -> ModelPostgresPoolConfig:
        """Warn when database is empty (likely misconfiguration)."""
        if not self.database:
            logger.warning(
                "ModelPostgresPoolConfig created with empty database. "
                "This will fail at connection time. "
                "Use from_env() or from_db_url() for validated construction."
            )
        return self

    @classmethod
    def from_db_url(
        cls,
        url: str,
        *,
        min_size: int = 2,
        max_size: int = 10,
    ) -> ModelPostgresPoolConfig:
        """Create config by parsing a PostgreSQL DSN.

        Args:
            url: Full PostgreSQL DSN
                (e.g. ``postgresql://user:pass@host:5432/dbname``).
            min_size: Minimum connection pool size.
            max_size: Maximum connection pool size.

        Raises:
            ValueError: If the URL cannot be parsed or is missing required
                components (database name).
        """
        try:
            parsed = urlparse(url)
        except Exception as exc:
            msg = f"Invalid OMNIBASE_INFRA_DB_URL: {exc}"
            raise ValueError(msg) from exc

        database = unquote(parsed.path.lstrip("/")) if parsed.path else ""
        if not database:
            msg = (
                "OMNIBASE_INFRA_DB_URL is missing the database name "
                "(expected postgresql://…/<database>)"
            )
            raise ValueError(msg)

        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            user=unquote(parsed.username or "postgres"),
            password=unquote(parsed.password or ""),
            database=database,
            min_size=min_size,
            max_size=max_size,
        )

    @classmethod
    def from_env(cls) -> ModelPostgresPoolConfig:
        """Create config from environment variables.

        Resolution order:
        1. ``OMNIBASE_INFRA_DB_URL`` - full DSN (preferred).
        2. Individual ``POSTGRES_*`` variables (legacy fallback).

        Raises:
            ValueError: If no database name can be resolved or the
                environment configuration is otherwise invalid.
        """
        db_url = os.getenv("OMNIBASE_INFRA_DB_URL")
        if db_url:
            return cls.from_db_url(
                db_url,
                min_size=int(os.getenv("POSTGRES_POOL_MIN_SIZE", "2")),
                max_size=int(os.getenv("POSTGRES_POOL_MAX_SIZE", "10")),
            )

        # Legacy fallback – individual POSTGRES_* vars
        database = os.getenv("POSTGRES_DATABASE", "")
        if not database:
            msg = (
                "PostgreSQL database not configured. "
                "Set OMNIBASE_INFRA_DB_URL (preferred) or POSTGRES_DATABASE. "
                "The implicit 'omninode_bridge' default has been removed "
                "(see OMN-2146)."
            )
            raise ValueError(msg)

        try:
            return cls(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
                database=database,
                min_size=int(os.getenv("POSTGRES_POOL_MIN_SIZE", "2")),
                max_size=int(os.getenv("POSTGRES_POOL_MAX_SIZE", "10")),
            )
        except (ValueError, TypeError) as e:
            msg = f"Invalid PostgreSQL pool configuration: {e}"
            raise ValueError(msg) from e


__all__ = ["ModelPostgresPoolConfig"]
