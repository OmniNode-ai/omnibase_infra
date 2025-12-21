# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka Event Bus configuration model.

Provides a Pydantic configuration model for KafkaEventBus with support for
environment variable overrides, YAML configuration loading, and sensible
defaults for production deployment.

Features:
    - Strong typing with comprehensive validation
    - Environment variable override support with type conversion
    - YAML configuration file loading
    - Sensible defaults for production resilience patterns
    - Circuit breaker and retry configuration
    - Warning logs for invalid environment variable values

Environment Variables:
    All environment variables are optional and fall back to defaults if not set
    or if parsing fails. Invalid values log warnings and use defaults.

    Connection Settings:
        KAFKA_BOOTSTRAP_SERVERS: Kafka broker addresses (comma-separated)
            Default: "localhost:9092"
            Example: "kafka1:9092,kafka2:9092"

        KAFKA_ENVIRONMENT: Environment identifier for message routing
            Default: "local"
            Example: "dev", "staging", "prod"

        KAFKA_GROUP: Consumer group identifier
            Default: "default"
            Example: "my-service-group"

    Timeout and Retry Settings (with validation):
        KAFKA_TIMEOUT_SECONDS: Timeout for operations (integer, 1-300)
            Default: 30
            Example: "60"
            Warning: Logs warning if not a valid integer, uses default

        KAFKA_MAX_RETRY_ATTEMPTS: Maximum retry attempts (integer, 0-10)
            Default: 3
            Example: "5"
            Warning: Logs warning if not a valid integer, uses default

        KAFKA_RETRY_BACKOFF_BASE: Base exponential backoff delay (float, 0.1-60.0)
            Default: 1.0
            Example: "2.0"
            Warning: Logs warning if not a valid float, uses default

    Circuit Breaker Settings (with validation):
        KAFKA_CIRCUIT_BREAKER_THRESHOLD: Failures before circuit opens (integer, 1-100)
            Default: 5
            Example: "10"
            Warning: Logs warning if not a valid integer, uses default

        KAFKA_CIRCUIT_BREAKER_RESET_TIMEOUT: Reset timeout in seconds (float, 1.0-3600.0)
            Default: 30.0
            Example: "60.0"
            Warning: Logs warning if not a valid float, uses default

    Consumer Settings:
        KAFKA_CONSUMER_SLEEP_INTERVAL: Poll interval in seconds (float, 0.01-10.0)
            Default: 0.1
            Example: "0.2"
            Warning: Logs warning if not a valid float, uses default

        KAFKA_AUTO_OFFSET_RESET: Offset reset policy
            Default: "latest"
            Options: "earliest", "latest"

        KAFKA_ENABLE_AUTO_COMMIT: Auto-commit consumer offsets (boolean)
            Default: true
            True values: "true", "1", "yes", "on" (case-insensitive)
            False values: "false", "0", "no", "off" (case-insensitive)
            Warning: Logs warning if unexpected value, treats as False

    Producer Settings:
        KAFKA_ACKS: Producer acknowledgment policy
            Default: "all"
            Options: "all", "1", "0"

        KAFKA_ENABLE_IDEMPOTENCE: Enable idempotent producer (boolean)
            Default: true
            True values: "true", "1", "yes", "on" (case-insensitive)
            False values: "false", "0", "no", "off" (case-insensitive)
            Warning: Logs warning if unexpected value, treats as False

    Dead Letter Queue Settings:
        KAFKA_DEAD_LETTER_TOPIC: Topic name for failed messages (optional)
            Default: None (DLQ disabled)
            Example: "dlq-events"

Parsing Behavior:
    - Integer/Float fields: Logs warning and uses default if parsing fails
    - Boolean fields: Logs warning if value not in expected set, treats as False
    - String fields: No validation, accepts any string value
    - All warnings include the environment variable name, invalid value, and
      the field name that will use the default value
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

logger = logging.getLogger(__name__)


class ModelKafkaEventBusConfig(BaseModel):
    """Configuration model for KafkaEventBus.

    Defines all required configuration options for KafkaEventBus including
    connection settings, resilience patterns (circuit breaker, retry),
    and Kafka producer/consumer options.

    Attributes:
        bootstrap_servers: Kafka bootstrap servers (host:port format)
        environment: Environment identifier for message routing
        group: Consumer group identifier for message routing
        timeout_seconds: Timeout for Kafka operations in seconds
        max_retry_attempts: Maximum retry attempts for publish operations
        retry_backoff_base: Base delay in seconds for exponential backoff
        circuit_breaker_threshold: Number of consecutive failures before circuit opens
        circuit_breaker_reset_timeout: Seconds before circuit breaker resets to half-open
        consumer_sleep_interval: Sleep interval in seconds for consumer loop polling
        acks: Producer acknowledgment policy ("all", "1", "0")
        enable_idempotence: Enable producer idempotence for exactly-once semantics
        auto_offset_reset: Consumer offset reset policy ("earliest", "latest")
        enable_auto_commit: Enable auto-commit for consumer offsets
        dead_letter_topic: Dead letter queue topic for failed messages (optional)

    Example:
        ```python
        # Using defaults with environment overrides
        config = ModelKafkaEventBusConfig.default()

        # From YAML file
        config = ModelKafkaEventBusConfig.from_yaml(Path("kafka_config.yaml"))

        # Manual construction
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="kafka:9092",
            environment="prod",
            timeout_seconds=60,
        )
        ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # Connection settings
    bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers (host:port format, comma-separated for multiple)",
        min_length=1,
    )
    environment: str = Field(
        default="local",
        description="Environment identifier for message routing (e.g., 'local', 'dev', 'prod')",
        min_length=1,
    )
    group: str = Field(
        default="default",
        description="Consumer group identifier for message routing",
        min_length=1,
    )
    timeout_seconds: int = Field(
        default=30,
        description="Timeout for Kafka operations in seconds",
        ge=1,
        le=300,
    )

    # Retry configuration
    max_retry_attempts: int = Field(
        default=3,
        description="Maximum retry attempts for publish operations",
        ge=0,
        le=10,
    )
    retry_backoff_base: float = Field(
        default=1.0,
        description="Base delay in seconds for exponential backoff",
        ge=0.1,
        le=60.0,
    )

    # Circuit breaker configuration
    circuit_breaker_threshold: int = Field(
        default=5,
        description="Number of consecutive failures before circuit opens",
        ge=1,
        le=100,
    )
    circuit_breaker_reset_timeout: float = Field(
        default=30.0,
        description="Seconds before circuit breaker resets to half-open state",
        ge=1.0,
        le=3600.0,
    )

    # Consumer configuration
    consumer_sleep_interval: float = Field(
        default=0.1,
        description="Sleep interval in seconds for consumer loop polling",
        ge=0.01,
        le=10.0,
    )

    # Kafka producer settings
    acks: str = Field(
        default="all",
        description="Producer acknowledgment policy ('all', '1', '0')",
        pattern=r"^(all|0|1)$",
    )
    enable_idempotence: bool = Field(
        default=True,
        description="Enable producer idempotence for exactly-once semantics",
    )

    # Kafka consumer settings
    auto_offset_reset: str = Field(
        default="latest",
        description="Consumer offset reset policy ('earliest', 'latest')",
        pattern=r"^(earliest|latest)$",
    )
    enable_auto_commit: bool = Field(
        default=True,
        description="Enable auto-commit for consumer offsets",
    )

    # Dead letter queue configuration
    dead_letter_topic: str | None = Field(
        default=None,
        description="Dead letter queue topic for failed messages (optional)",
    )

    @field_validator("bootstrap_servers", mode="before")
    @classmethod
    def validate_bootstrap_servers(cls, v: object) -> str:
        """Validate bootstrap servers format.

        Args:
            v: Bootstrap servers value (any type before Pydantic conversion)

        Returns:
            Validated bootstrap servers string

        Raises:
            ProtocolConfigurationError: If bootstrap servers format is invalid
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="validate_config",
            target_name="kafka_config",
            correlation_id=uuid4(),
        )

        if v is None:
            raise ProtocolConfigurationError(
                "bootstrap_servers cannot be None",
                context=context,
                parameter="bootstrap_servers",
                value=None,
            )
        if not isinstance(v, str):
            raise ProtocolConfigurationError(
                f"bootstrap_servers must be a string, got {type(v).__name__}",
                context=context,
                parameter="bootstrap_servers",
                value=type(v).__name__,
            )
        if not v.strip():
            raise ProtocolConfigurationError(
                "bootstrap_servers cannot be empty",
                context=context,
                parameter="bootstrap_servers",
                value=v,
            )

        # Validate host:port format for each server
        servers = v.strip().split(",")
        for server in servers:
            server = server.strip()
            if not server:
                raise ProtocolConfigurationError(
                    "bootstrap_servers cannot contain empty entries",
                    context=context,
                    parameter="bootstrap_servers",
                    value=v,
                )
            if ":" not in server:
                raise ProtocolConfigurationError(
                    f"Invalid bootstrap server format '{server}'. "
                    "Expected 'host:port' (e.g., 'localhost:9092')",
                    context=context,
                    parameter="bootstrap_servers",
                    value=server,
                )
            host, port_str = server.rsplit(":", 1)
            if not host:
                raise ProtocolConfigurationError(
                    f"Invalid bootstrap server format '{server}'. Host cannot be empty",
                    context=context,
                    parameter="bootstrap_servers",
                    value=server,
                )
            try:
                port = int(port_str)
                if port < 1 or port > 65535:
                    raise ProtocolConfigurationError(
                        f"Invalid port {port} in '{server}'. Port must be between 1 and 65535",
                        context=context,
                        parameter="bootstrap_servers",
                        value=server,
                    )
            except ValueError as e:
                raise ProtocolConfigurationError(
                    f"Invalid port '{port_str}' in '{server}'. Port must be a valid integer",
                    context=context,
                    parameter="bootstrap_servers",
                    value=server,
                ) from e
            except ProtocolConfigurationError:
                raise

        return v.strip()

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: object) -> str:
        """Validate environment identifier.

        Args:
            v: Environment value (any type before Pydantic conversion)

        Returns:
            Validated environment string

        Raises:
            ProtocolConfigurationError: If environment is empty or invalid type
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="validate_config",
            target_name="kafka_config",
            correlation_id=uuid4(),
        )

        if v is None:
            raise ProtocolConfigurationError(
                "environment cannot be None",
                context=context,
                parameter="environment",
                value=None,
            )
        if not isinstance(v, str):
            raise ProtocolConfigurationError(
                f"environment must be a string, got {type(v).__name__}",
                context=context,
                parameter="environment",
                value=type(v).__name__,
            )
        if not v.strip():
            raise ProtocolConfigurationError(
                "environment cannot be empty",
                context=context,
                parameter="environment",
                value=v,
            )
        return v.strip()

    @field_validator("group", mode="before")
    @classmethod
    def validate_group(cls, v: object) -> str:
        """Validate consumer group identifier.

        Args:
            v: Group value (any type before Pydantic conversion)

        Returns:
            Validated group string

        Raises:
            ProtocolConfigurationError: If group is empty, invalid type, or contains invalid characters
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="validate_config",
            target_name="kafka_config",
            correlation_id=uuid4(),
        )

        if v is None:
            raise ProtocolConfigurationError(
                "group cannot be None",
                context=context,
                parameter="group",
                value=None,
            )
        if not isinstance(v, str):
            raise ProtocolConfigurationError(
                f"group must be a string, got {type(v).__name__}",
                context=context,
                parameter="group",
                value=type(v).__name__,
            )

        group_name = v.strip()
        if not group_name:
            raise ProtocolConfigurationError(
                "group cannot be empty",
                context=context,
                parameter="group",
                value=v,
            )

        # Kafka group names have similar restrictions to topic names
        # but are generally more permissive
        if len(group_name) > 255:
            raise ProtocolConfigurationError(
                f"group name '{group_name}' exceeds maximum length of 255 characters",
                context=context,
                parameter="group",
                value=group_name,
            )

        # Check for invalid characters (control characters, null bytes)
        for char in group_name:
            if ord(char) < 32 or char == "\x7f":
                raise ProtocolConfigurationError(
                    f"group name '{group_name}' contains invalid control character",
                    context=context,
                    parameter="group",
                    value=group_name,
                )

        return group_name

    def apply_environment_overrides(self) -> ModelKafkaEventBusConfig:
        """Apply environment variable overrides to configuration.

        Environment variables are mapped as follows:
            - KAFKA_BOOTSTRAP_SERVERS -> bootstrap_servers
            - KAFKA_TIMEOUT_SECONDS -> timeout_seconds
            - KAFKA_ENVIRONMENT -> environment
            - KAFKA_GROUP -> group
            - KAFKA_MAX_RETRY_ATTEMPTS -> max_retry_attempts
            - KAFKA_CIRCUIT_BREAKER_THRESHOLD -> circuit_breaker_threshold

        Returns:
            New configuration instance with environment overrides applied
        """
        overrides: dict[str, object] = {}

        env_mappings: dict[str, str] = {
            "KAFKA_BOOTSTRAP_SERVERS": "bootstrap_servers",
            "KAFKA_TIMEOUT_SECONDS": "timeout_seconds",
            "KAFKA_ENVIRONMENT": "environment",
            "KAFKA_GROUP": "group",
            "KAFKA_MAX_RETRY_ATTEMPTS": "max_retry_attempts",
            "KAFKA_CIRCUIT_BREAKER_THRESHOLD": "circuit_breaker_threshold",
            "KAFKA_CIRCUIT_BREAKER_RESET_TIMEOUT": "circuit_breaker_reset_timeout",
            "KAFKA_RETRY_BACKOFF_BASE": "retry_backoff_base",
            "KAFKA_CONSUMER_SLEEP_INTERVAL": "consumer_sleep_interval",
            "KAFKA_ACKS": "acks",
            "KAFKA_ENABLE_IDEMPOTENCE": "enable_idempotence",
            "KAFKA_AUTO_OFFSET_RESET": "auto_offset_reset",
            "KAFKA_ENABLE_AUTO_COMMIT": "enable_auto_commit",
            "KAFKA_DEAD_LETTER_TOPIC": "dead_letter_topic",
        }

        # Integer fields for type conversion
        int_fields = {
            "timeout_seconds",
            "max_retry_attempts",
            "circuit_breaker_threshold",
        }

        # Float fields for type conversion
        float_fields = {
            "circuit_breaker_reset_timeout",
            "retry_backoff_base",
            "consumer_sleep_interval",
        }

        # Boolean fields for type conversion
        bool_fields = {
            "enable_idempotence",
            "enable_auto_commit",
        }

        for env_var, field_name in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                if field_name in int_fields:
                    try:
                        overrides[field_name] = int(env_value)
                    except ValueError:
                        logger.warning(
                            "Failed to parse integer environment variable %s='%s', "
                            "using default value for %s",
                            env_var,
                            env_value,
                            field_name,
                        )
                        continue
                elif field_name in float_fields:
                    try:
                        overrides[field_name] = float(env_value)
                    except ValueError:
                        logger.warning(
                            "Failed to parse float environment variable %s='%s', "
                            "using default value for %s",
                            env_var,
                            env_value,
                            field_name,
                        )
                        continue
                elif field_name in bool_fields:
                    # Boolean conversion with explicit falsy value handling
                    # True values: "true", "1", "yes", "on" (case-insensitive)
                    # False values: All other values (including "false", "0", "no", "off")
                    parsed_value = env_value.lower() in ("true", "1", "yes", "on")
                    if env_value.lower() not in (
                        "true",
                        "1",
                        "yes",
                        "on",
                        "false",
                        "0",
                        "no",
                        "off",
                    ):
                        logger.warning(
                            "Boolean environment variable %s='%s' has unexpected value. "
                            "Valid values are: true/1/yes/on (True) or false/0/no/off (False). "
                            "Treating as False.",
                            env_var,
                            env_value,
                        )
                    overrides[field_name] = parsed_value
                else:
                    overrides[field_name] = env_value

        if overrides:
            current_data = self.model_dump()
            current_data.update(overrides)
            return ModelKafkaEventBusConfig(**current_data)

        return self

    @classmethod
    def default(cls) -> ModelKafkaEventBusConfig:
        """Create default configuration with environment overrides.

        Returns a canonical default configuration for development, testing,
        and CLI fallback use, with environment variable overrides applied.

        Returns:
            Default configuration instance with environment overrides
        """
        base_config = cls(
            bootstrap_servers="localhost:9092",
            environment="local",
            group="default",
            timeout_seconds=30,
            max_retry_attempts=3,
            retry_backoff_base=1.0,
            circuit_breaker_threshold=5,
            circuit_breaker_reset_timeout=30.0,
            consumer_sleep_interval=0.1,
            acks="all",
            enable_idempotence=True,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            dead_letter_topic=None,
        )
        return base_config.apply_environment_overrides()

    @classmethod
    def from_yaml(cls, path: Path) -> ModelKafkaEventBusConfig:
        """Load configuration from YAML file.

        Loads configuration from a YAML file and applies environment
        variable overrides on top.

        Args:
            path: Path to YAML configuration file

        Returns:
            Configuration instance loaded from YAML with env overrides

        Raises:
            FileNotFoundError: If the YAML file does not exist
            ProtocolConfigurationError: If the YAML content is invalid

        Example YAML:
            ```yaml
            bootstrap_servers: "kafka:9092"
            environment: "prod"
            group: "my-service"
            timeout_seconds: 60
            max_retry_attempts: 5
            circuit_breaker_threshold: 10
            ```
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        if not isinstance(data, dict):
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="load_yaml_config",
                target_name="kafka_config",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                f"YAML content must be a dictionary, got {type(data)}",
                context=context,
                parameter="yaml_content",
                value=type(data).__name__,
            )

        config = cls(**data)
        return config.apply_environment_overrides()


__all__: list[str] = ["ModelKafkaEventBusConfig"]
