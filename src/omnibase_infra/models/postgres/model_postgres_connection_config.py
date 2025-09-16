"""PostgreSQL connection configuration model."""

import os

from pydantic import BaseModel, Field


class ModelPostgresConnectionConfig(BaseModel):
    """PostgreSQL connection configuration."""

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    database: str = Field(default="omnibase_infrastructure", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")
    schema: str = Field(default="infrastructure", description="Default schema")

    # Pool configuration
    min_connections: int = Field(default=5, description="Minimum pool connections")
    max_connections: int = Field(default=50, description="Maximum pool connections")
    max_inactive_connection_lifetime: float = Field(
        default=300.0, description="Max inactive connection lifetime in seconds",
    )
    max_queries: int = Field(default=50000, description="Maximum queries per connection")

    # Connection timeouts
    command_timeout: float = Field(default=60.0, description="Command timeout in seconds")
    server_settings: dict[str, str] | None = Field(
        default=None, description="Additional server settings",
    )

    # SSL configuration
    ssl_mode: str = Field(default="prefer", description="SSL mode")
    ssl_cert_file: str | None = Field(default=None, description="SSL certificate file")
    ssl_key_file: str | None = Field(default=None, description="SSL key file")
    ssl_ca_file: str | None = Field(default=None, description="SSL CA file")

    @classmethod
    def from_environment(cls) -> "ModelPostgresConnectionConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DATABASE", "omnibase_infrastructure"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            schema=os.getenv("POSTGRES_SCHEMA", "infrastructure"),
            min_connections=int(os.getenv("POSTGRES_MIN_CONNECTIONS", "5")),
            max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "50")),
            max_inactive_connection_lifetime=float(
                os.getenv("POSTGRES_MAX_INACTIVE_LIFETIME", "300.0"),
            ),
            command_timeout=float(os.getenv("POSTGRES_COMMAND_TIMEOUT", "60.0")),
            ssl_mode=os.getenv("POSTGRES_SSL_MODE", "prefer"),
            ssl_cert_file=os.getenv("POSTGRES_SSL_CERT_FILE"),
            ssl_key_file=os.getenv("POSTGRES_SSL_KEY_FILE"),
            ssl_ca_file=os.getenv("POSTGRES_SSL_CA_FILE"),
        )
