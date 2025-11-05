"""PostgreSQL connection configuration model."""

import os

from pydantic import BaseModel, Field, SecretStr, field_serializer, field_validator


class ModelPostgresConnectionConfig(BaseModel):
    """
    PostgreSQL connection configuration with secure credential management.

    Used by PostgresConnectionManager for asyncpg pool configuration.
    Supports environment variable configuration and Docker secrets.
    """

    host: str = Field(description="PostgreSQL host address")
    port: int = Field(default=5432, description="PostgreSQL port", ge=1, le=65535)
    database: str = Field(description="Database name")
    user: str = Field(description="Database user")
    password: SecretStr = Field(description="Database password (securely stored)")
    schema: str = Field(default="infrastructure", description="Default schema")

    # Pool configuration
    min_connections: int = Field(default=5, description="Minimum pool connections", ge=1)
    max_connections: int = Field(default=50, description="Maximum pool connections", ge=1)
    max_inactive_connection_lifetime: float = Field(
        default=300.0,
        description="Max inactive connection lifetime in seconds",
        ge=0.0,
    )
    max_queries: int = Field(
        default=50000,
        description="Maximum queries per connection before recycling",
        ge=1,
    )

    # Connection timeouts
    command_timeout: float = Field(
        default=60.0,
        description="Command timeout in seconds",
        ge=0.0,
    )
    server_settings: dict[str, str] | None = Field(
        default=None,
        description="Additional server settings",
    )

    # SSL configuration
    ssl_mode: str = Field(default="prefer", description="SSL mode: disable, prefer, require")
    ssl_cert_file: str | None = Field(default=None, description="SSL certificate file path")
    ssl_key_file: str | None = Field(default=None, description="SSL key file path")
    ssl_ca_file: str | None = Field(default=None, description="SSL CA file path")

    @field_validator("password", mode="before")
    @classmethod
    def validate_password(cls, v):
        """Validate password without logging sensitive data."""
        if isinstance(v, str):
            if len(v) > 200:
                msg = "Database password too long (max 200 characters)"
                raise ValueError(msg)
            return SecretStr(v)
        return v

    @field_validator("ssl_mode")
    @classmethod
    def validate_ssl_mode(cls, v):
        """Validate SSL mode."""
        valid_modes = {"disable", "prefer", "require", "verify-ca", "verify-full"}
        if v not in valid_modes:
            msg = f"Invalid SSL mode: {v}. Must be one of {valid_modes}"
            raise ValueError(msg)
        return v

    @field_serializer("password")
    def serialize_password(self, value: SecretStr) -> str:
        """Serialize password securely - excludes actual value."""
        return "***REDACTED***"

    @classmethod
    def from_environment(cls) -> "ModelPostgresConnectionConfig":
        """
        Create configuration from environment variables with ONEX standards.

        Required environment variables:
        - POSTGRES_HOST
        - POSTGRES_PORT
        - POSTGRES_DATABASE
        - POSTGRES_USER
        - POSTGRES_PASSWORD_FILE (Docker secret) or POSTGRES_PASSWORD

        Optional environment variables:
        - POSTGRES_SCHEMA (default: infrastructure)
        - POSTGRES_MIN_CONNECTIONS (default: 5)
        - POSTGRES_MAX_CONNECTIONS (default: 50)
        - POSTGRES_MAX_INACTIVE_LIFETIME (default: 300.0)
        - POSTGRES_COMMAND_TIMEOUT (default: 60.0)
        - POSTGRES_SSL_MODE (default: prefer)
        - POSTGRES_SSL_CERT_FILE
        - POSTGRES_SSL_KEY_FILE
        - POSTGRES_SSL_CA_FILE
        """
        from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

        # Read password from Docker secrets file or environment variable
        password = ""
        password_file = os.getenv("POSTGRES_PASSWORD_FILE")
        if password_file and os.path.exists(password_file):
            try:
                with open(password_file) as f:
                    password = f.read().strip()
            except Exception as e:
                raise OnexError(
                    code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                    message=f"Failed to read PostgreSQL password from file {password_file}: {e!s}",
                ) from e
        else:
            password = os.getenv("POSTGRES_PASSWORD", "")
            if not password:
                raise OnexError(
                    code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                    message="POSTGRES_PASSWORD or POSTGRES_PASSWORD_FILE environment variable is required",
                )

        # Validate required environment variables
        required_vars = {
            "POSTGRES_HOST": os.getenv("POSTGRES_HOST"),
            "POSTGRES_PORT": os.getenv("POSTGRES_PORT"),
            "POSTGRES_DATABASE": os.getenv("POSTGRES_DATABASE"),
            "POSTGRES_USER": os.getenv("POSTGRES_USER"),
        }

        missing_vars = [k for k, v in required_vars.items() if not v]
        if missing_vars:
            raise OnexError(
                code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                message=f"Missing required environment variables: {', '.join(missing_vars)}",
            )

        return cls(
            host=required_vars["POSTGRES_HOST"],
            port=int(required_vars["POSTGRES_PORT"]),
            database=required_vars["POSTGRES_DATABASE"],
            user=required_vars["POSTGRES_USER"],
            password=SecretStr(password),
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
