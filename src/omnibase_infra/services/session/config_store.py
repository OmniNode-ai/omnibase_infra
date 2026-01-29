"""Configuration for session snapshot storage.

Loads from environment variables with OMNIBASE_INFRA_SESSION_STORAGE_ prefix.

Moved from omniclaude as part of OMN-1526 architectural cleanup.
"""

from __future__ import annotations

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigSessionStorage(BaseSettings):
    """Configuration for session snapshot PostgreSQL storage.

    Environment variables use the OMNIBASE_INFRA_SESSION_STORAGE_ prefix.
    Example: OMNIBASE_INFRA_SESSION_STORAGE_POSTGRES_HOST=db.example.com
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_SESSION_STORAGE_",
        env_file=".env",
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
        default="omninode_bridge",
        description="PostgreSQL database name",
    )
    postgres_user: str = Field(
        default="postgres",
        description="PostgreSQL user",
    )
    postgres_password: SecretStr = Field(
        ...,  # Required
        description="PostgreSQL password - set via OMNIBASE_INFRA_SESSION_STORAGE_POSTGRES_PASSWORD env var",
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

    @property
    def dsn(self) -> str:
        """Build PostgreSQL DSN from components.

        Returns:
            PostgreSQL connection string.
        """
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_database}"
        )

    @property
    def dsn_async(self) -> str:
        """Build async PostgreSQL DSN for asyncpg.

        Returns:
            PostgreSQL connection string with postgresql+asyncpg scheme.
        """
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_database}"
        )

    @property
    def dsn_safe(self) -> str:
        """Build PostgreSQL DSN with password masked (safe for logging).

        Returns:
            PostgreSQL connection string with password replaced by ***.
        """
        return (
            f"postgresql://{self.postgres_user}:***"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_database}"
        )

    def __repr__(self) -> str:
        """Safe string representation that doesn't expose credentials.

        Returns:
            String representation with masked password.
        """
        return f"ConfigSessionStorage(dsn={self.dsn_safe!r})"
