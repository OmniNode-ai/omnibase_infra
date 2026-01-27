# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Scheduler Configuration Model.

Provides a Pydantic configuration model for the runtime tick scheduler that emits
RuntimeTick events at configurable intervals. Supports environment variable overrides
for deployment flexibility and restart-safe behavior through sequence number tracking.

Features:
    - Strong typing with comprehensive validation
    - Environment variable override support with type conversion
    - Sensible defaults for production deployment
    - Circuit breaker configuration for publish failure resilience
    - Restart-safe sequence number persistence

Environment Variables:
    All environment variables are optional and fall back to defaults if not set
    or if parsing fails. Invalid values log warnings and use defaults.

    Core Settings:
        ONEX_RUNTIME_SCHEDULER_TICK_INTERVAL_MS: Tick interval in milliseconds (integer, 10-60000)
            Default: 1000 (1 second)
            Example: "5000" (5 seconds)
            Warning: Logs warning if not a valid integer, uses default

        ONEX_RUNTIME_SCHEDULER_ID: Unique scheduler identifier
            Default: "runtime-scheduler-default"
            Example: "runtime-scheduler-prod-1"

        ONEX_RUNTIME_SCHEDULER_TICK_TOPIC: Kafka topic for publishing ticks
            Default: SUFFIX_RUNTIME_TICK from omnibase_infra.topics
            Example: "prod.runtime.tick.v1"

    Restart-Safety Settings:
        ONEX_RUNTIME_SCHEDULER_PERSIST_SEQUENCE: Enable sequence number persistence (boolean)
            Default: true
            True values: "true", "1", "yes", "on" (case-insensitive)
            False values: "false", "0", "no", "off" (case-insensitive)

        ONEX_RUNTIME_SCHEDULER_SEQUENCE_KEY: Key for sequence number storage
            Default: "runtime_scheduler_sequence"
            Example: "scheduler_seq_prod"

    Performance Settings:
        ONEX_RUNTIME_SCHEDULER_MAX_JITTER_MS: Maximum jitter in milliseconds (integer, 0-10000)
            Default: 100
            Example: "50"
            Warning: Logs warning if not a valid integer, uses default

    Circuit Breaker Settings:
        ONEX_RUNTIME_SCHEDULER_CB_THRESHOLD: Failures before circuit opens (integer, 1-100)
            Default: 5
            Example: "10"
            Warning: Logs warning if not a valid integer, uses default

        ONEX_RUNTIME_SCHEDULER_CB_RESET_TIMEOUT: Reset timeout in seconds (float, 1.0-3600.0)
            Default: 60.0
            Example: "120.0"
            Warning: Logs warning if not a valid float, uses default

    Metrics Settings:
        ONEX_RUNTIME_SCHEDULER_ENABLE_METRICS: Enable metrics collection (boolean)
            Default: true
            True values: "true", "1", "yes", "on" (case-insensitive)
            False values: "false", "0", "no", "off" (case-insensitive)

        ONEX_RUNTIME_SCHEDULER_METRICS_PREFIX: Prefix for metrics names
            Default: "runtime_scheduler"
            Example: "prod_runtime_scheduler"

    Valkey (Redis-compatible) Settings:
        REDIS_HOST: Valkey host for sequence number persistence
            Default: "localhost"
            Example: "omninode-bridge-valkey" or use REDIS_HOST env var

        REDIS_PORT: Valkey port (integer, 1-65535)
            Default: 6379

        REDIS_PASSWORD: Valkey password (optional)
            Default: None (no authentication)

        ONEX_RUNTIME_SCHEDULER_VALKEY_TIMEOUT: Timeout for Valkey ops (float)
            Default: 5.0 seconds
            Range: 0.1-60.0

        ONEX_RUNTIME_SCHEDULER_VALKEY_RETRIES: Connection retries (integer)
            Default: 3
            Range: 0-10

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
import re
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.topics import SUFFIX_RUNTIME_TICK

logger = logging.getLogger(__name__)


class ModelRuntimeSchedulerConfig(BaseModel):
    """Configuration model for the runtime tick scheduler.

    Controls how frequently RuntimeTick events are emitted and provides
    restart-safe behavior through sequence number persistence.

    Attributes:
        tick_interval_ms: Interval between ticks in milliseconds (10-60000)
        scheduler_id: Unique identifier for this scheduler instance
        tick_topic: Kafka topic for publishing tick events
        persist_sequence_number: Whether to persist sequence numbers for restart safety
        sequence_number_key: Key used for sequence number storage
        max_tick_jitter_ms: Maximum jitter to prevent thundering herd (0-10000)
        circuit_breaker_threshold: Failures before circuit opens (1-100)
        circuit_breaker_reset_timeout_seconds: Reset timeout in seconds (1.0-3600.0)
        enable_metrics: Whether to collect scheduler metrics
        metrics_prefix: Prefix for metrics names
        valkey_host: Valkey host for sequence number persistence
        valkey_port: Valkey port for sequence number persistence (1-65535)
        valkey_password: Valkey password (optional)
        valkey_timeout_seconds: Timeout for Valkey operations (0.1-60.0)
        valkey_connection_retries: Connection retries before fallback (0-10)

    Example:
        ```python
        # Using defaults with environment overrides
        config = ModelRuntimeSchedulerConfig.default()

        # Manual construction with custom values
        config = ModelRuntimeSchedulerConfig(
            tick_interval_ms=5000,  # 5 seconds
            scheduler_id="prod-scheduler-1",
            tick_topic="prod.runtime.tick.v1",
        )
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Core scheduling configuration
    tick_interval_ms: int = Field(
        default=1000,
        description="Interval between ticks in milliseconds",
        ge=10,
        le=60000,
    )
    scheduler_id: str = Field(
        default="runtime-scheduler-default",
        description="Unique identifier for this scheduler instance",
        min_length=1,
        max_length=255,
    )
    tick_topic: str = Field(
        default=SUFFIX_RUNTIME_TICK,
        description="Kafka topic for publishing tick events",
        min_length=1,
        max_length=255,
    )

    # Restart-safety configuration
    persist_sequence_number: bool = Field(
        default=True,
        description="Whether to persist sequence numbers for restart safety",
    )
    sequence_number_key: str = Field(
        default="runtime_scheduler_sequence",
        description="Key used for sequence number storage",
        min_length=1,
        max_length=255,
    )

    # Performance tuning
    max_tick_jitter_ms: int = Field(
        default=100,
        description="Maximum jitter in milliseconds to prevent thundering herd",
        ge=0,
        le=10000,
    )

    # Circuit breaker configuration
    circuit_breaker_threshold: int = Field(
        default=5,
        description="Number of consecutive failures before circuit opens",
        ge=1,
        le=100,
    )
    circuit_breaker_reset_timeout_seconds: float = Field(
        default=60.0,
        description="Seconds before circuit breaker resets to half-open state",
        ge=1.0,
        le=3600.0,
    )

    # Metrics configuration
    enable_metrics: bool = Field(
        default=True,
        description="Whether to collect scheduler metrics",
    )
    metrics_prefix: str = Field(
        default="runtime_scheduler",
        description="Prefix for metrics names",
        min_length=1,
        max_length=255,
    )

    # Valkey (Redis-compatible) configuration for sequence number persistence
    valkey_host: str = Field(
        default="localhost",
        description="Valkey host for sequence number persistence",
        min_length=1,
        max_length=255,
    )
    valkey_port: int = Field(
        default=6379,
        description="Valkey port for sequence number persistence",
        ge=1,
        le=65535,
    )
    valkey_password: str | None = Field(
        default=None,
        description="Valkey password (optional, from REDIS_PASSWORD env var)",
    )
    valkey_timeout_seconds: float = Field(
        default=5.0,
        description="Timeout for Valkey operations in seconds",
        ge=0.1,
        le=60.0,
    )
    valkey_connection_retries: int = Field(
        default=3,
        description="Number of connection retries before fallback",
        ge=0,
        le=10,
    )

    @field_validator("scheduler_id", mode="before")
    @classmethod
    def validate_scheduler_id(cls, v: object) -> str:
        """Validate scheduler identifier.

        Args:
            v: Scheduler ID value (any type before Pydantic conversion)

        Returns:
            Validated scheduler ID string

        Raises:
            ProtocolConfigurationError: If scheduler ID is empty, invalid type,
                or contains invalid characters
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="validate_config",
            target_name="runtime_scheduler_config",
            correlation_id=uuid4(),
        )

        if v is None:
            raise ProtocolConfigurationError(
                "scheduler_id cannot be None",
                context=context,
                parameter="scheduler_id",
                value=None,
            )
        if not isinstance(v, str):
            raise ProtocolConfigurationError(
                f"scheduler_id must be a string, got {type(v).__name__}",
                context=context,
                parameter="scheduler_id",
                value=type(v).__name__,
            )

        scheduler_id = v.strip()
        if not scheduler_id:
            raise ProtocolConfigurationError(
                "scheduler_id cannot be empty",
                context=context,
                parameter="scheduler_id",
                value=v,
            )

        # Check for invalid characters (control characters, null bytes)
        for char in scheduler_id:
            if ord(char) < 32 or char == "\x7f":
                raise ProtocolConfigurationError(
                    f"scheduler_id '{scheduler_id}' contains invalid control character",
                    context=context,
                    parameter="scheduler_id",
                    value=scheduler_id,
                )

        return scheduler_id

    @field_validator("tick_topic", mode="before")
    @classmethod
    def validate_tick_topic(cls, v: object) -> str:
        """Validate tick topic name.

        Args:
            v: Topic name value (any type before Pydantic conversion)

        Returns:
            Validated topic name string

        Raises:
            ProtocolConfigurationError: If topic name is empty, invalid type,
                or contains invalid characters
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="validate_config",
            target_name="runtime_scheduler_config",
            correlation_id=uuid4(),
        )

        if v is None:
            raise ProtocolConfigurationError(
                "tick_topic cannot be None",
                context=context,
                parameter="tick_topic",
                value=None,
            )
        if not isinstance(v, str):
            raise ProtocolConfigurationError(
                f"tick_topic must be a string, got {type(v).__name__}",
                context=context,
                parameter="tick_topic",
                value=type(v).__name__,
            )

        topic_name = v.strip()
        if not topic_name:
            raise ProtocolConfigurationError(
                "tick_topic cannot be empty",
                context=context,
                parameter="tick_topic",
                value=v,
            )

        # Kafka topic naming rules: alphanumeric, dots, underscores, hyphens
        if not re.match(r"^[a-zA-Z0-9._-]+$", topic_name):
            raise ProtocolConfigurationError(
                f"tick_topic '{topic_name}' contains invalid characters. "
                "Only alphanumeric characters, dots, underscores, and hyphens are allowed",
                context=context,
                parameter="tick_topic",
                value=topic_name,
            )

        return topic_name

    @field_validator("sequence_number_key", mode="before")
    @classmethod
    def validate_sequence_number_key(cls, v: object) -> str:
        """Validate sequence number storage key.

        Args:
            v: Key value (any type before Pydantic conversion)

        Returns:
            Validated key string

        Raises:
            ProtocolConfigurationError: If key is empty or invalid type
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="validate_config",
            target_name="runtime_scheduler_config",
            correlation_id=uuid4(),
        )

        if v is None:
            raise ProtocolConfigurationError(
                "sequence_number_key cannot be None",
                context=context,
                parameter="sequence_number_key",
                value=None,
            )
        if not isinstance(v, str):
            raise ProtocolConfigurationError(
                f"sequence_number_key must be a string, got {type(v).__name__}",
                context=context,
                parameter="sequence_number_key",
                value=type(v).__name__,
            )

        key = v.strip()
        if not key:
            raise ProtocolConfigurationError(
                "sequence_number_key cannot be empty",
                context=context,
                parameter="sequence_number_key",
                value=v,
            )

        return key

    @field_validator("metrics_prefix", mode="before")
    @classmethod
    def validate_metrics_prefix(cls, v: object) -> str:
        """Validate metrics prefix.

        Args:
            v: Prefix value (any type before Pydantic conversion)

        Returns:
            Validated prefix string

        Raises:
            ProtocolConfigurationError: If prefix is empty, invalid type,
                or contains invalid characters
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="validate_config",
            target_name="runtime_scheduler_config",
            correlation_id=uuid4(),
        )

        if v is None:
            raise ProtocolConfigurationError(
                "metrics_prefix cannot be None",
                context=context,
                parameter="metrics_prefix",
                value=None,
            )
        if not isinstance(v, str):
            raise ProtocolConfigurationError(
                f"metrics_prefix must be a string, got {type(v).__name__}",
                context=context,
                parameter="metrics_prefix",
                value=type(v).__name__,
            )

        prefix = v.strip()
        if not prefix:
            raise ProtocolConfigurationError(
                "metrics_prefix cannot be empty",
                context=context,
                parameter="metrics_prefix",
                value=v,
            )

        # Metrics prefixes should be alphanumeric with underscores only
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", prefix):
            raise ProtocolConfigurationError(
                f"metrics_prefix '{prefix}' is invalid. "
                "Must start with a letter and contain only alphanumeric characters "
                "and underscores",
                context=context,
                parameter="metrics_prefix",
                value=prefix,
            )

        return prefix

    def apply_environment_overrides(self) -> ModelRuntimeSchedulerConfig:
        """Apply environment variable overrides to configuration.

        Environment variables are mapped as follows:
            - ONEX_RUNTIME_SCHEDULER_TICK_INTERVAL_MS -> tick_interval_ms
            - ONEX_RUNTIME_SCHEDULER_ID -> scheduler_id
            - ONEX_RUNTIME_SCHEDULER_TICK_TOPIC -> tick_topic
            - ONEX_RUNTIME_SCHEDULER_PERSIST_SEQUENCE -> persist_sequence_number
            - ONEX_RUNTIME_SCHEDULER_SEQUENCE_KEY -> sequence_number_key
            - ONEX_RUNTIME_SCHEDULER_MAX_JITTER_MS -> max_tick_jitter_ms
            - ONEX_RUNTIME_SCHEDULER_CB_THRESHOLD -> circuit_breaker_threshold
            - ONEX_RUNTIME_SCHEDULER_CB_RESET_TIMEOUT -> circuit_breaker_reset_timeout_seconds
            - ONEX_RUNTIME_SCHEDULER_ENABLE_METRICS -> enable_metrics
            - ONEX_RUNTIME_SCHEDULER_METRICS_PREFIX -> metrics_prefix

        Returns:
            New configuration instance with environment overrides applied
        """
        overrides: dict[str, object] = {}

        env_mappings: dict[str, str] = {
            "ONEX_RUNTIME_SCHEDULER_TICK_INTERVAL_MS": "tick_interval_ms",
            "ONEX_RUNTIME_SCHEDULER_ID": "scheduler_id",
            "ONEX_RUNTIME_SCHEDULER_TICK_TOPIC": "tick_topic",
            "ONEX_RUNTIME_SCHEDULER_PERSIST_SEQUENCE": "persist_sequence_number",
            "ONEX_RUNTIME_SCHEDULER_SEQUENCE_KEY": "sequence_number_key",
            "ONEX_RUNTIME_SCHEDULER_MAX_JITTER_MS": "max_tick_jitter_ms",
            "ONEX_RUNTIME_SCHEDULER_CB_THRESHOLD": "circuit_breaker_threshold",
            "ONEX_RUNTIME_SCHEDULER_CB_RESET_TIMEOUT": "circuit_breaker_reset_timeout_seconds",
            "ONEX_RUNTIME_SCHEDULER_ENABLE_METRICS": "enable_metrics",
            "ONEX_RUNTIME_SCHEDULER_METRICS_PREFIX": "metrics_prefix",
            # Valkey configuration from Redis environment variables
            "REDIS_HOST": "valkey_host",
            "REDIS_PORT": "valkey_port",
            "REDIS_PASSWORD": "valkey_password",
            "ONEX_RUNTIME_SCHEDULER_VALKEY_TIMEOUT": "valkey_timeout_seconds",
            "ONEX_RUNTIME_SCHEDULER_VALKEY_RETRIES": "valkey_connection_retries",
        }

        # Integer fields for type conversion
        int_fields = {
            "tick_interval_ms",
            "max_tick_jitter_ms",
            "circuit_breaker_threshold",
            "valkey_port",
            "valkey_connection_retries",
        }

        # Float fields for type conversion
        float_fields = {
            "circuit_breaker_reset_timeout_seconds",
            "valkey_timeout_seconds",
        }

        # Boolean fields for type conversion
        bool_fields = {
            "persist_sequence_number",
            "enable_metrics",
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
            return ModelRuntimeSchedulerConfig(**current_data)

        return self

    @classmethod
    def default(cls) -> ModelRuntimeSchedulerConfig:
        """Create default configuration with environment overrides.

        Returns a canonical default configuration for development, testing,
        and CLI fallback use, with environment variable overrides applied.

        Returns:
            Default configuration instance with environment overrides
        """
        base_config = cls(
            tick_interval_ms=1000,
            scheduler_id="runtime-scheduler-default",
            tick_topic=SUFFIX_RUNTIME_TICK,
            persist_sequence_number=True,
            sequence_number_key="runtime_scheduler_sequence",
            max_tick_jitter_ms=100,
            circuit_breaker_threshold=5,
            circuit_breaker_reset_timeout_seconds=60.0,
            enable_metrics=True,
            metrics_prefix="runtime_scheduler",
            valkey_host="localhost",
            valkey_port=6379,
            valkey_password=None,
            valkey_timeout_seconds=5.0,
            valkey_connection_retries=3,
        )
        return base_config.apply_environment_overrides()


__all__: list[str] = ["ModelRuntimeSchedulerConfig"]
