"""Comprehensive Pydantic validation schemas for all external inputs.

This module provides validation for all external data entering the OmniNode Bridge system,
including environment variables, CLI inputs, webhook payloads, and configuration files.
"""

import re
from pathlib import Path
from typing import Any, Protocol, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


# Type-safe structures for common validation patterns
class ServiceConfigDict(TypedDict, total=False):
    """Service configuration structure."""

    host: str
    port: int
    timeout_ms: int
    max_retries: int
    enabled: bool


class DatabaseConfigDict(TypedDict, total=False):
    """Database configuration structure."""

    host: str
    port: int
    database: str
    user: str
    password: str
    pool_min: int
    pool_max: int
    ssl_enabled: bool


class KafkaConfigDict(TypedDict, total=False):
    """Kafka configuration structure."""

    bootstrap_servers: str
    topic_prefix: str
    consumer_group: str
    enable_auto_commit: bool
    session_timeout_ms: int


class SecurityConfigDict(TypedDict, total=False):
    """Security configuration structure."""

    api_key_enabled: bool
    jwt_enabled: bool
    tls_enabled: bool
    allowed_origins: list[str]
    rate_limit_enabled: bool


class SupportsValidation(Protocol):
    """Protocol for objects that support validation."""

    def validate(self) -> bool:
        """Validate the object."""
        ...


# Type alias for environment variable values
EnvVarValue = Union[str, int, bool, None]


class EnvironmentVariablesSchema(BaseModel):
    """Validation schema for environment variables at startup."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",  # Prevent unknown environment variables
    )

    # Environment and service identification
    environment: str = Field(
        ...,
        pattern=r"^(development|staging|production|test)$",
        description="Deployment environment",
    )
    service_version: str = Field(
        ..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic version"
    )
    log_level: str = Field(
        ...,
        pattern=r"^(debug|info|warning|error|critical)$",
        description="Logging level",
    )

    # Database connection (required)
    postgres_host: str = Field(
        ..., min_length=1, max_length=255, description="PostgreSQL host"
    )
    postgres_port: int = Field(..., ge=1, le=65535, description="PostgreSQL port")
    postgres_database: str = Field(
        ...,
        min_length=1,
        max_length=63,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Database name",
    )
    postgres_user: str = Field(
        ..., min_length=1, max_length=63, description="Database user"
    )
    postgres_password: str | None = Field(
        None, min_length=8, max_length=128, description="Database password"
    )

    # Kafka configuration (required)
    kafka_bootstrap_servers: str = Field(
        ..., min_length=1, description="Kafka bootstrap servers"
    )
    kafka_workflow_topic: str = Field(
        ..., min_length=1, max_length=249, description="Kafka workflow topic"
    )
    kafka_task_events_topic: str = Field(
        ..., min_length=1, max_length=249, description="Kafka task events topic"
    )

    # Security configuration
    api_key: str | None = Field(
        None, min_length=16, max_length=256, description="API key for authentication"
    )
    jwt_secret: str | None = Field(
        None, min_length=32, max_length=512, description="JWT signing secret"
    )
    webhook_signing_secret: str | None = Field(
        None, min_length=16, max_length=256, description="Webhook signing secret"
    )

    # Service endpoints (optional, with defaults)
    hook_receiver_url: str | None = Field(
        None, pattern=r"^https?://.*", description="Hook receiver URL"
    )
    model_metrics_url: str | None = Field(
        None, pattern=r"^https?://.*", description="Model metrics URL"
    )

    # Vault configuration (optional)
    vault_enabled: bool = Field(False, description="Enable HashiCorp Vault")
    vault_addr: str | None = Field(
        None, pattern=r"^https?://.*", description="Vault server address"
    )
    vault_token: str | None = Field(
        None, min_length=20, description="Vault authentication token"
    )

    @field_validator("postgres_password")
    @classmethod
    def validate_postgres_password(
        cls, v: str | None, info: ValidationInfo
    ) -> str | None:
        """Validate database password based on environment."""
        environment = info.data.get("environment", "development")
        if environment == "production" and not v:
            raise ValueError("Database password is required in production environment")
        if v and len(v) < 12 and environment == "production":
            raise ValueError(
                "Database password must be at least 12 characters in production"
            )
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key_strength(
        cls, v: str | None, info: ValidationInfo
    ) -> str | None:
        """Validate API key strength."""
        if v:
            weak_keys = [
                "omninode-bridge-api-key-2024",
                "default-api-key",
                "test-api-key",
                "development-key",
            ]
            if v in weak_keys:
                environment = info.data.get("environment", "development")
                if environment == "production":
                    raise ValueError("Weak or default API key detected in production")
        return v

    @field_validator("kafka_bootstrap_servers")
    @classmethod
    def validate_kafka_servers(cls, v: str) -> str:
        """Validate Kafka bootstrap servers format."""
        servers = v.split(",")
        for server in servers:
            server = server.strip()
            if ":" not in server:
                raise ValueError(
                    f"Invalid Kafka server format: {server} (missing port)"
                )
            host, port_str = server.rsplit(":", 1)
            try:
                port = int(port_str)
                if not (1 <= port <= 65535):
                    raise ValueError(f"Invalid port for Kafka server {server}: {port}")
            except ValueError:
                raise ValueError(
                    f"Invalid port format for Kafka server {server}: {port_str}"
                )
        return v


class CLIInputSchema(BaseModel):
    """Validation schema for CLI command inputs."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    # Workflow submission
    workflow_file_path: str | None = Field(
        None, description="Path to workflow definition file"
    )
    workflow_name: str | None = Field(
        None,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\s-]+$",
        description="Workflow name",
    )
    priority: int | None = Field(None, ge=1, le=10, description="Workflow priority")

    # Service control
    service_name: str | None = Field(
        None, pattern=r"^[a-zA-Z0-9_-]+$", description="Service name"
    )
    action: str | None = Field(
        None,
        pattern=r"^(start|stop|restart|status|health)$",
        description="Service action",
    )

    @field_validator("workflow_file_path")
    @classmethod
    def validate_workflow_file_path(cls, v: str | None) -> str | None:
        """Validate workflow file path."""
        if v:
            path = Path(v)
            if path.suffix not in [".json", ".yaml", ".yml"]:
                raise ValueError("Workflow file must be JSON or YAML format")
            if not path.exists():
                raise ValueError(f"Workflow file not found: {v}")
            # Prevent path traversal attacks
            if ".." in str(path) or str(path).startswith("/"):
                if not path.resolve().is_relative_to(Path.cwd()):
                    raise ValueError(
                        "Workflow file path not allowed (path traversal detected)"
                    )
        return v


class WebhookPayloadSchema(BaseModel):
    """Validation schema for incoming webhook payloads."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",  # Allow extra fields for flexible webhook support
    )

    # Common webhook fields
    event: str | None = Field(
        None, min_length=1, max_length=50, description="Event type"
    )
    action: str | None = Field(
        None, min_length=1, max_length=50, description="Action performed"
    )
    timestamp: str | None = Field(None, description="Event timestamp")
    signature: str | None = Field(
        None, description="Webhook signature for verification"
    )

    # GitHub webhook specific
    repository: dict[str, Any] | None = Field(
        None, description="GitHub repository data"
    )
    sender: dict[str, Any] | None = Field(None, description="GitHub sender data")

    # Generic payload data
    data: dict[str, Any] | None = Field(None, description="Webhook payload data")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_format(cls, v: str | None) -> str | None:
        """Validate timestamp format."""
        if v:
            # Support common timestamp formats
            import re

            patterns = [
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",  # ISO format
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$",  # ISO with milliseconds
                r"^\d{10}$",  # Unix timestamp
                r"^\d{13}$",  # Unix timestamp with milliseconds
            ]
            if not any(re.match(pattern, v) for pattern in patterns):
                raise ValueError("Invalid timestamp format")
        return v

    @field_validator("signature")
    @classmethod
    def validate_signature_format(cls, v: str | None) -> str | None:
        """Validate webhook signature format."""
        if v:
            # Common signature formats: sha256=..., sha1=..., etc.
            if not re.match(r"^(sha1|sha256|md5)=[a-f0-9]+$", v):
                raise ValueError("Invalid webhook signature format")
        return v


class FileUploadSchema(BaseModel):
    """Validation schema for file uploads."""

    model_config = ConfigDict(validate_assignment=True)

    filename: str = Field(
        ..., min_length=1, max_length=255, description="Upload filename"
    )
    content_type: str = Field(..., min_length=1, description="File content type")
    size_bytes: int = Field(
        ..., ge=0, le=50_000_000, description="File size in bytes (max 50MB)"
    )
    file_hash: str | None = Field(
        None, min_length=32, max_length=128, description="File hash for integrity"
    )

    @field_validator("filename")
    @classmethod
    def validate_filename_security(cls, v: str) -> str:
        """Validate filename for security."""
        # Prevent path traversal and dangerous filenames
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Filename contains invalid characters")

        # Prevent dangerous extensions
        dangerous_extensions = [
            ".exe",
            ".bat",
            ".cmd",
            ".com",
            ".scr",
            ".pif",
            ".vbs",
            ".js",
            ".jar",
            ".sh",
            ".py",
            ".php",
            ".asp",
            ".jsp",
            ".cgi",
            ".pl",
        ]
        file_ext = Path(v).suffix.lower()
        if file_ext in dangerous_extensions:
            raise ValueError(f"File extension not allowed: {file_ext}")

        return v

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """Validate content type."""
        allowed_types = [
            "application/json",
            "application/yaml",
            "text/yaml",
            "text/plain",
            "application/octet-stream",
            "text/csv",
            "application/xml",
            "text/xml",
        ]
        if v not in allowed_types:
            raise ValueError(f"Content type not allowed: {v}")
        return v


class ConfigurationFileSchema(BaseModel):
    """Validation schema for configuration files."""

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid"  # Strict validation for config files
    )

    # Service configuration - more specific typing
    services: Union[dict[str, ServiceConfigDict], dict[str, Any]] = Field(
        ..., description="Service configuration"
    )
    database: Union[DatabaseConfigDict, dict[str, Any]] = Field(
        ..., description="Database configuration"
    )
    kafka: Union[KafkaConfigDict, dict[str, Any]] = Field(
        ..., description="Kafka configuration"
    )
    security: Union[SecurityConfigDict, dict[str, Any]] = Field(
        ..., description="Security configuration"
    )

    # Optional sections
    cache: Union[dict[str, Any], None] = Field(None, description="Cache configuration")
    monitoring: Union[dict[str, Any], None] = Field(
        None, description="Monitoring configuration"
    )
    logging: Union[dict[str, Any], None] = Field(
        None, description="Logging configuration"
    )

    @field_validator("services", "database", "kafka", "security")
    @classmethod
    def validate_config_sections(
        cls,
        v: Union[
            dict[str, Any],
            ServiceConfigDict,
            DatabaseConfigDict,
            KafkaConfigDict,
            SecurityConfigDict,
        ],
    ) -> Union[
        dict[str, Any],
        ServiceConfigDict,
        DatabaseConfigDict,
        KafkaConfigDict,
        SecurityConfigDict,
    ]:
        """Validate configuration sections are not empty."""
        if not v:
            raise ValueError("Configuration section cannot be empty")
        return v


def validate_environment_variables(
    env_vars: dict[str, str]
) -> EnvironmentVariablesSchema:
    """Validate environment variables with comprehensive error reporting.

    Args:
        env_vars: Raw environment variables as string key-value pairs

    Returns:
        Validated and typed EnvironmentVariablesSchema instance

    Raises:
        ValueError: If validation fails with detailed error information
    """
    try:
        # Convert string values to appropriate types
        processed_vars: dict[str, EnvVarValue] = {}
        for key, value in env_vars.items():
            # Convert common boolean values
            if value.lower() in ("true", "false", "1", "0", "yes", "no"):
                if value.lower() in ("true", "1", "yes"):
                    processed_vars[key] = True
                else:
                    processed_vars[key] = False
            # Convert integer values for ports
            elif key.endswith("_PORT") or key.endswith("_port"):
                try:
                    processed_vars[key] = int(value)
                except ValueError:
                    processed_vars[key] = (
                        value  # Keep as string, let Pydantic handle validation
                    )
            else:
                processed_vars[key] = value

        return EnvironmentVariablesSchema(**processed_vars)

    except Exception as e:
        raise ValueError(f"Environment variable validation failed: {e}") from e


def validate_cli_input(
    cli_args: Union[dict[str, str], dict[str, Any]]
) -> CLIInputSchema:
    """Validate CLI command arguments.

    Args:
        cli_args: Command-line arguments dictionary

    Returns:
        Validated CLIInputSchema instance
    """
    return CLIInputSchema(**cli_args)


def validate_webhook_payload(
    payload: Union[dict[str, Any], WebhookPayloadSchema]
) -> WebhookPayloadSchema:
    """Validate incoming webhook payload.

    Args:
        payload: Raw webhook payload data

    Returns:
        Validated WebhookPayloadSchema instance
    """
    if isinstance(payload, WebhookPayloadSchema):
        return payload
    return WebhookPayloadSchema(**payload)


def validate_file_upload(
    file_info: Union[dict[str, Any], FileUploadSchema]
) -> FileUploadSchema:
    """Validate file upload information.

    Args:
        file_info: File upload metadata and information

    Returns:
        Validated FileUploadSchema instance
    """
    if isinstance(file_info, FileUploadSchema):
        return file_info
    return FileUploadSchema(**file_info)


def validate_config_file(
    config_data: Union[dict[str, Any], ConfigurationFileSchema]
) -> ConfigurationFileSchema:
    """Validate configuration file contents.

    Args:
        config_data: Configuration file data structure

    Returns:
        Validated ConfigurationFileSchema instance
    """
    if isinstance(config_data, ConfigurationFileSchema):
        return config_data
    return ConfigurationFileSchema(**config_data)
