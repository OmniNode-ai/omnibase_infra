"""Configuration for session snapshot storage.

Reads standard POSTGRES_* environment variables sourced from ~/.omnibase/.env
at shell startup. No env_prefix and no env_file — values come entirely from
the shell environment, consistent with the zero-repo-env policy (OMN-2287).

Note: This module intentionally uses individual POSTGRES_* env vars rather
than a single DSN. The session storage may target a different database than
the main OMNIBASE_INFRA_DB_URL. Migration to DSN-based configuration is
tracked separately from the OMN-2065 DB split.

Moved from omniclaude as part of OMN-1526 architectural cleanup.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigSessionStorage(BaseSettings):
    """Configuration for session snapshot PostgreSQL storage.

    Reads standard POSTGRES_* environment variables directly from the shell
    environment (no prefix). The env_file is explicitly disabled so that no
    repository-local .env file is silently discovered, in compliance with the
    zero-repo-env policy. Source ~/.omnibase/.env in your shell profile to
    supply the required values.

    Note: Using an empty prefix means any POSTGRES_* variables already set in the
    environment (e.g. by a test runner or CI matrix) will be used here. This is an
    intentional trade-off of the zero-repo-env policy; ensure POSTGRES_* variables
    in the shell match the intended session storage target.

    Example: export POSTGRES_HOST=db.example.com
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=None,  # No .env file — reads from shell env (sourced via ~/.omnibase/.env)
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL connection
    postgres_host: str = Field(
        default="localhost",
        description="PostgreSQL host",
    )
    postgres_port: int = Field(
        default=5436,
        ge=1,
        le=65535,
        description="PostgreSQL port",
    )
    postgres_database: str = Field(
        default="omnibase_infra",
        description="PostgreSQL database name",
    )
    postgres_user: str = Field(
        default="postgres",
        description="PostgreSQL user",
    )
    postgres_password: SecretStr = Field(
        ...,  # Required
        description="PostgreSQL password - set via POSTGRES_PASSWORD env var",
    )

    # Connection pool
    pool_min_size: int = Field(
        default=2,
        ge=1,
        le=100,
        description="Minimum connection pool size",
    )
    pool_max_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum connection pool size",
    )

    # Query timeouts
    query_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Query timeout in seconds",
    )

    @model_validator(mode="after")
    def validate_pool_sizes(self) -> ConfigSessionStorage:
        """Validate that pool_min_size <= pool_max_size.

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If pool_min_size > pool_max_size.
        """
        if self.pool_min_size > self.pool_max_size:
            raise ValueError(
                f"pool_min_size ({self.pool_min_size}) must be <= "
                f"pool_max_size ({self.pool_max_size})"
            )
        return self

    @staticmethod
    def _format_host(host: str) -> str:
        """Format host for DSN, wrapping IPv6 addresses in brackets.

        Uses ``ipaddress.IPv6Address`` for definitive detection rather than
        a ``":" in host`` heuristic, which would false-positive on strings
        like ``host:port`` accidentally passed as a bare hostname.

        Args:
            host: Hostname or IP address.

        Returns:
            Host string suitable for embedding in a DSN.
        """
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            return host
        return f"[{host}]"

    @property
    def dsn(self) -> str:
        """Build PostgreSQL DSN from components.

        Credentials, database name, and host are URL-encoded or formatted
        to handle special characters that would otherwise break the DSN.

        Returns:
            PostgreSQL connection string.
        """
        encoded_user = quote_plus(self.postgres_user, safe="")
        encoded_password = quote_plus(
            self.postgres_password.get_secret_value(), safe=""
        )
        encoded_database = quote_plus(self.postgres_database, safe="")
        host = self._format_host(self.postgres_host)
        return (
            f"postgresql://{encoded_user}:{encoded_password}"
            f"@{host}:{self.postgres_port}"
            f"/{encoded_database}"
        )

    @property
    def dsn_async(self) -> str:
        """Build async PostgreSQL DSN for asyncpg.

        Credentials, database name, and host are URL-encoded or formatted
        to handle special characters that would otherwise break the DSN.

        Returns:
            PostgreSQL connection string with postgresql+asyncpg scheme.
        """
        encoded_user = quote_plus(self.postgres_user, safe="")
        encoded_password = quote_plus(
            self.postgres_password.get_secret_value(), safe=""
        )
        encoded_database = quote_plus(self.postgres_database, safe="")
        host = self._format_host(self.postgres_host)
        return (
            f"postgresql+asyncpg://{encoded_user}:{encoded_password}"
            f"@{host}:{self.postgres_port}"
            f"/{encoded_database}"
        )

    @property
    def dsn_safe(self) -> str:
        """Build PostgreSQL DSN with password masked (safe for logging).

        Returns:
            PostgreSQL connection string with password replaced by ***.
        """
        encoded_user = quote_plus(self.postgres_user, safe="")
        encoded_database = quote_plus(self.postgres_database, safe="")
        host = self._format_host(self.postgres_host)
        return (
            f"postgresql://{encoded_user}:***"
            f"@{host}:{self.postgres_port}"
            f"/{encoded_database}"
        )

    def __repr__(self) -> str:
        """Safe string representation that doesn't expose credentials.

        Returns:
            String representation with masked password.
        """
        return f"ConfigSessionStorage(dsn={self.dsn_safe!r})"
