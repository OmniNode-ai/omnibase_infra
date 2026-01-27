# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka Event Bus configuration model.

Provides a Pydantic configuration model for EventBusKafka with support for
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
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from omnibase_infra.enums import EnumInfraTransportType, EnumKafkaAcks
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

logger = logging.getLogger(__name__)


class ModelKafkaEventBusConfig(BaseModel):
    """Configuration model for EventBusKafka.

    Defines all required configuration options for EventBusKafka including
    connection settings, resilience patterns (circuit breaker, retry),
    and Kafka producer/consumer options.

    Attributes:
        bootstrap_servers: Kafka bootstrap servers (host:port format)
        environment: Environment identifier for message routing
        timeout_seconds: Timeout for Kafka operations in seconds
        max_retry_attempts: Maximum retry attempts for publish operations
        retry_backoff_base: Base delay in seconds for exponential backoff
        circuit_breaker_threshold: Number of consecutive failures before circuit opens
        circuit_breaker_reset_timeout: Seconds before circuit breaker resets to half-open
        consumer_sleep_interval: Sleep interval in seconds for consumer loop polling
        acks: Producer acknowledgment policy (EnumKafkaAcks.ALL, LEADER, NONE, ALL_REPLICAS)
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
        ge=0.001,  # Allow very short backoffs for testing (minimum 1ms)
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
        ge=0.01,  # Allow short timeouts for testing (minimum 10ms)
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
    acks: EnumKafkaAcks = Field(
        default=EnumKafkaAcks.ALL,
        description="Producer acknowledgment policy (ALL, LEADER, NONE, ALL_REPLICAS)",
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
        description=(
            "Dead letter queue topic for failed messages (optional). "
            "If not set, use get_dlq_topic() to build a topic name following "
            "ONEX conventions: <env>.dlq.<category>.v1 "
            "(e.g., 'dev.dlq.intents.v1', 'prod.dlq.events.v1')"
        ),
    )

    # NOTE: mypy reports "prop-decorator" error because it doesn't understand that
    # Pydantic's @computed_field transforms the @property into a computed field.
    # This is a known mypy/Pydantic v2 interaction - the code works correctly at runtime.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def acks_aiokafka(self) -> int | str:
        """Get acks value in aiokafka-compatible format.

        aiokafka's AIOKafkaProducer expects:
        - The string "all" for all-replica acknowledgment
        - Integer values (0, 1, -1) for numeric ack levels

        Returns:
            The acks value converted to the format expected by aiokafka:
            - "all" (str) for EnumKafkaAcks.ALL
            - 0 (int) for EnumKafkaAcks.NONE
            - 1 (int) for EnumKafkaAcks.LEADER
            - -1 (int) for EnumKafkaAcks.ALL_REPLICAS

        Example:
            >>> config = ModelKafkaEventBusConfig(acks=EnumKafkaAcks.LEADER)
            >>> config.acks_aiokafka
            1
        """
        return self.acks.to_aiokafka()

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

    def apply_environment_overrides(self) -> ModelKafkaEventBusConfig:
        """Apply environment variable overrides to configuration.

        Environment variables are mapped as follows:
            - KAFKA_BOOTSTRAP_SERVERS -> bootstrap_servers
            - KAFKA_TIMEOUT_SECONDS -> timeout_seconds
            - KAFKA_ENVIRONMENT -> environment
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

        # Enum fields with their valid values mapping
        # Maps field_name -> (enum_class, value_to_enum_mapping)
        acks_mapping = {
            "all": EnumKafkaAcks.ALL,
            "0": EnumKafkaAcks.NONE,
            "1": EnumKafkaAcks.LEADER,
            "-1": EnumKafkaAcks.ALL_REPLICAS,
        }

        for env_var, field_name in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                if field_name == "acks":
                    # Special handling for acks enum - fail-fast on invalid values
                    if env_value in acks_mapping:
                        overrides[field_name] = acks_mapping[env_value]
                    else:
                        valid_values = ", ".join(acks_mapping.keys())
                        raise ProtocolConfigurationError(
                            f"Invalid value for environment variable {env_var}='{env_value}'. "
                            f"Valid values are: {valid_values}",
                            context=ModelInfraErrorContext.with_correlation(
                                transport_type=EnumInfraTransportType.KAFKA,
                                operation="apply_environment_overrides",
                            ),
                        )
                elif field_name in int_fields:
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
            # Exclude computed field to avoid validation error
            current_data = self.model_dump(exclude={"acks_aiokafka"})
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
            timeout_seconds=30,
            max_retry_attempts=3,
            retry_backoff_base=1.0,
            circuit_breaker_threshold=5,
            circuit_breaker_reset_timeout=30.0,
            consumer_sleep_interval=0.1,
            acks=EnumKafkaAcks.ALL,
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
            ProtocolConfigurationError: If the file does not exist, cannot be read,
                contains invalid YAML, or has invalid content structure. Error includes
                correlation_id for tracing and detailed context for debugging.

        Example YAML:
            ```yaml
            bootstrap_servers: "kafka:9092"
            environment: "prod"
            timeout_seconds: 60
            max_retry_attempts: 5
            circuit_breaker_threshold: 10
            ```
        """
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="load_yaml_config",
            target_name=str(path),
            correlation_id=correlation_id,
        )

        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise ProtocolConfigurationError(
                f"Configuration file not found: {path}",
                context=context,
                config_path=str(path),
            ) from e
        except yaml.YAMLError as e:
            raise ProtocolConfigurationError(
                f"Failed to parse YAML from {path}: {e}",
                context=context,
                config_path=str(path),
                error_details=str(e),
            ) from e
        except UnicodeDecodeError as e:
            raise ProtocolConfigurationError(
                f"Configuration file contains binary or non-UTF-8 content: {path}",
                context=context,
                config_path=str(path),
                error_details=f"Encoding error at position {e.start}-{e.end}: {e.reason}",
            ) from e
        except OSError as e:
            raise ProtocolConfigurationError(
                f"Failed to read configuration file: {path}: {e}",
                context=context,
                config_path=str(path),
                error_details=str(e),
            ) from e

        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise ProtocolConfigurationError(
                f"YAML content must be a dictionary, got {type(data)}",
                context=context,
                config_path=str(path),
                parameter="yaml_content",
                value=type(data).__name__,
            )

        config = cls(**data)
        return config.apply_environment_overrides()

    def get_dlq_topic(self, category: str = "intents") -> str:
        """Get the DLQ topic for this configuration.

        If dead_letter_topic is explicitly set, returns that value.
        Otherwise, builds a DLQ topic name following ONEX conventions
        using the configuration's environment.

        DLQ Topic Naming Convention:
            Format: <env>.dlq.<category>.v1
            Examples:
                - dev.dlq.intents.v1 (for permanently failed intents)
                - prod.dlq.events.v1 (for permanently failed events)
                - staging.dlq.commands.v1 (for permanently failed commands)

        Args:
            category: Message category for DLQ routing. Valid values:
                - 'intent' or 'intents' (default)
                - 'event' or 'events'
                - 'command' or 'commands'

        Returns:
            The DLQ topic name (either explicit or generated).

        Raises:
            ValueError: If category is not a valid message category.

        Example:
            >>> config = ModelKafkaEventBusConfig(environment="prod")
            >>> config.get_dlq_topic()
            'prod.dlq.intents.v1'
            >>> config.get_dlq_topic("events")
            'prod.dlq.events.v1'
            >>> # Explicit topic takes precedence
            >>> config = ModelKafkaEventBusConfig(
            ...     environment="prod",
            ...     dead_letter_topic="custom-dlq"
            ... )
            >>> config.get_dlq_topic()
            'custom-dlq'
        """
        if self.dead_letter_topic:
            return self.dead_letter_topic

        # Import here to avoid circular imports
        from omnibase_infra.event_bus.topic_constants import build_dlq_topic

        return build_dlq_topic(self.environment, category)


__all__: list[str] = ["ModelKafkaEventBusConfig"]
