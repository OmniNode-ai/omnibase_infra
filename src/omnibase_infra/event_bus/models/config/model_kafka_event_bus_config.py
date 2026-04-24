# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
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
    - Reconnect backoff configuration to prevent thundering-herd storms
    - Warning logs for invalid environment variable values

Environment Variables:
    All environment variables are optional and fall back to defaults if not set
    or if parsing fails. Invalid values log warnings and use defaults.

    Connection Settings:
        KAFKA_BOOTSTRAP_SERVERS: Kafka broker addresses (comma-separated)
            Default: "localhost:19092"
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

    Instance Discriminator (OMN-2251):
        KAFKA_INSTANCE_ID: Instance discriminator for consumer group IDs (optional)
            Default: None (no discrimination, single-container behavior)
            Example: "container-1", "pod-abc123"
            When set, appended as '.__i.{instance_id}' to consumer group IDs
            so each container gets unique consumer group membership and
            proper Kafka partition assignment in multi-container dev environments.

    Reconnect Backoff Settings (OMN-2916):
        KAFKA_RECONNECT_BACKOFF_MS: Initial reconnect backoff in milliseconds (integer, >= 0)
            Default: 2000
            Example: "1000"
            Warning: Logs warning if not a valid integer, uses default
            Used to prevent thundering-herd reconnection storms after broker restarts.

        KAFKA_RECONNECT_BACKOFF_MAX_MS: Maximum reconnect backoff in milliseconds (integer, >= 0)
            Default: 30000
            Example: "60000"
            Warning: Logs warning if not a valid integer, uses default
            Must be >= reconnect_backoff_ms. Caps the exponential backoff growth.

    Authentication and TLS Settings (OMN-2793):
        KAFKA_SECURITY_PROTOCOL: Security protocol for Kafka connections
            Default: "PLAINTEXT"
            Options: "PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"

        KAFKA_SASL_MECHANISM: SASL authentication mechanism (optional)
            Default: None (no SASL)
            Options: "PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512", "OAUTHBEARER"
            Requires: security_protocol must be SASL_PLAINTEXT or SASL_SSL

        KAFKA_SASL_OAUTHBEARER_TOKEN_ENDPOINT_URL: Token endpoint for OAUTHBEARER (optional)
            Default: None
            Required when: KAFKA_SASL_MECHANISM=OAUTHBEARER

        KAFKA_SASL_OAUTHBEARER_CLIENT_ID: OAuth client ID (optional)
            Default: None
            Required when: KAFKA_SASL_MECHANISM=OAUTHBEARER

        KAFKA_SASL_OAUTHBEARER_CLIENT_SECRET: OAuth client secret (optional)
            Default: None
            Required when: KAFKA_SASL_MECHANISM=OAUTHBEARER

        KAFKA_SSL_CA_FILE: Path to CA certificate file for TLS verification (optional)
            Default: None
            Used when: security_protocol is SSL or SASL_SSL

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
from pathlib import Path
from uuid import uuid4

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from omnibase_infra.enums import (
    EnumInfraTransportType,
    EnumKafkaAcks,
)
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

logger = logging.getLogger(__name__)


class ModelKafkaEventBusConfig(BaseModel):
    """Configuration model for EventBusKafka.

    Defines all required configuration options for EventBusKafka including
    connection settings, resilience patterns (circuit breaker, retry),
    and Kafka producer/consumer options.

    Attributes:
        bootstrap_servers: Kafka bootstrap servers (host:port format)
        environment: Environment identifier for consumer groups and logging
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
        reconnect_backoff_ms: Initial reconnect backoff in milliseconds (OMN-2916)
        reconnect_backoff_max_ms: Maximum reconnect backoff in milliseconds (OMN-2916)

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
        default_factory=lambda: os.environ["KAFKA_BOOTSTRAP_SERVERS"],
        description="Kafka bootstrap servers (host:port format, comma-separated for multiple)",
        min_length=1,
    )
    environment: str = Field(
        default="local",
        description=(
            "Environment identifier for consumer groups and logging. "
            "Not used for topic naming (topics are realm-agnostic)."
        ),
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
    max_request_size: int = Field(
        default=4 * 1024 * 1024,  # 4 MB
        description=(
            "Maximum size in bytes for a Kafka produce request. "
            "Passed to AIOKafkaProducer(max_request_size=...). "
            "Must also align with broker-side max.message.bytes. "
            "Override via KAFKA_MAX_REQUEST_SIZE."
        ),
        ge=1024,  # 1 KB minimum
        le=52428800,  # 50 MB maximum
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
    session_timeout_ms: int = Field(
        default=45000,
        ge=6000,
        le=300000,
        description=(
            "Timeout in ms for consumer group session. If the broker receives no "
            "heartbeat within this window, the consumer is evicted and a rebalance "
            "is triggered. Default 45s (aiokafka default is 10s which causes "
            "rebalance storms during brief processing delays)."
        ),
    )
    heartbeat_interval_ms: int = Field(
        default=15000,
        ge=1000,
        le=60000,
        description=(
            "Interval in ms between heartbeats to the consumer group coordinator. "
            "Must be less than session_timeout_ms (typically 1/3). "
            "Default 15s (aiokafka default is 3s)."
        ),
    )
    max_poll_interval_ms: int = Field(
        default=300000,
        ge=10000,
        le=600000,
        description=(
            "Maximum time in ms between poll() calls before the consumer is "
            "considered failed. Default 300s (5 min). Set high enough to "
            "accommodate slow batch processing without triggering rebalances."
        ),
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

    # Authentication and TLS configuration (OMN-2793)
    security_protocol: str = Field(
        default="PLAINTEXT",
        description=(
            "Security protocol for Kafka connections. "
            "Valid values: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL"
        ),
        pattern=r"^(PLAINTEXT|SSL|SASL_PLAINTEXT|SASL_SSL)$",
    )
    sasl_mechanism: str | None = Field(
        default=None,
        description=(
            "SASL mechanism for authentication. "
            "Valid values: PLAIN, SCRAM-SHA-256, SCRAM-SHA-512, OAUTHBEARER. "
            "Requires security_protocol to be SASL_PLAINTEXT or SASL_SSL."
        ),
        pattern=r"^(PLAIN|SCRAM-SHA-256|SCRAM-SHA-512|OAUTHBEARER)$",
    )
    sasl_oauthbearer_token_endpoint_url: str | None = Field(
        default=None,
        description=(
            "Token endpoint URL for OAUTHBEARER SASL mechanism. "
            "Required when sasl_mechanism is OAUTHBEARER."
        ),
    )
    sasl_oauthbearer_client_id: str | None = Field(
        default=None,
        description=(
            "Client ID for OAUTHBEARER token requests. "
            "Required when sasl_mechanism is OAUTHBEARER."
        ),
    )
    sasl_oauthbearer_client_secret: str | None = Field(
        default=None,
        description=(
            "Client secret for OAUTHBEARER token requests. "
            "Required when sasl_mechanism is OAUTHBEARER."
        ),
    )
    ssl_ca_file: str | None = Field(
        default=None,
        description=(
            "Path to CA certificate file for SSL/TLS verification. "
            "Used when security_protocol is SSL or SASL_SSL."
        ),
    )

    # Reconnect backoff configuration (OMN-2916)
    reconnect_backoff_ms: int = Field(
        default=2000,
        description=(
            "Initial reconnect backoff in milliseconds. Used by aiokafka to space out "
            "reconnection attempts after a broker disconnect, preventing thundering-herd "
            "storms after broker restarts. Override via KAFKA_RECONNECT_BACKOFF_MS."
        ),
        ge=0,
    )
    reconnect_backoff_max_ms: int = Field(
        default=30000,
        description=(
            "Maximum reconnect backoff in milliseconds. Caps the exponential growth of "
            "reconnect delays. Must be >= reconnect_backoff_ms. "
            "Override via KAFKA_RECONNECT_BACKOFF_MAX_MS. "
            "NOTE: Not used by aiokafka 0.11.0 (no reconnect_backoff_max_ms parameter). "
            "Retained in config model for forward-compatibility with future aiokafka versions."
        ),
        ge=0,
    )

    @model_validator(mode="after")
    def validate_reconnect_backoff(self) -> ModelKafkaEventBusConfig:
        """Validate that reconnect_backoff_max_ms >= reconnect_backoff_ms.

        Returns:
            Self after validation

        Raises:
            ProtocolConfigurationError: If max backoff is less than base backoff
        """
        if self.reconnect_backoff_max_ms < self.reconnect_backoff_ms:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="validate_reconnect_backoff",
                target_name="kafka_config",
            )
            raise ProtocolConfigurationError(
                f"reconnect_backoff_max_ms ({self.reconnect_backoff_max_ms}) must be "
                f">= reconnect_backoff_ms ({self.reconnect_backoff_ms})",
                context=context,
                parameter="reconnect_backoff_max_ms",
                value=self.reconnect_backoff_max_ms,
            )
        return self

    @model_validator(mode="after")
    def validate_session_timeout_ratio(self) -> ModelKafkaEventBusConfig:
        """Validate session timeout / heartbeat / max-poll-interval relationships.

        Ensures:
        - heartbeat_interval_ms < session_timeout_ms (heartbeats must fit within session)
        - max_poll_interval_ms >= session_timeout_ms (poll interval must not be shorter)

        Returns:
            Self after validation

        Raises:
            ProtocolConfigurationError: If relationships are violated
        """
        if self.heartbeat_interval_ms >= self.session_timeout_ms:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="validate_session_timeout_ratio",
                target_name="kafka_config",
            )
            raise ProtocolConfigurationError(
                f"heartbeat_interval_ms ({self.heartbeat_interval_ms}) must be "
                f"< session_timeout_ms ({self.session_timeout_ms})",
                context=context,
                parameter="heartbeat_interval_ms",
                value=self.heartbeat_interval_ms,
            )
        if self.max_poll_interval_ms < self.session_timeout_ms:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="validate_session_timeout_ratio",
                target_name="kafka_config",
            )
            raise ProtocolConfigurationError(
                f"max_poll_interval_ms ({self.max_poll_interval_ms}) must be "
                f">= session_timeout_ms ({self.session_timeout_ms})",
                context=context,
                parameter="max_poll_interval_ms",
                value=self.max_poll_interval_ms,
            )
        return self

    @model_validator(mode="after")
    def validate_heartbeat_session_ratio(self) -> ModelKafkaEventBusConfig:
        """Advisory: warn if heartbeat_interval_ms > session_timeout_ms / 3.

        Kafka recommends heartbeat <= session_timeout / 3 to allow enough
        missed heartbeats before the session expires. This is advisory only
        (logs a warning) — it does not raise.

        Returns:
            Self after validation
        """
        if self.heartbeat_interval_ms > self.session_timeout_ms / 3:
            logger.warning(
                "heartbeat_interval_ms (%d) exceeds session_timeout_ms / 3 (%d). "
                "Kafka recommends heartbeat_interval_ms <= session_timeout_ms / 3 "
                "to prevent unnecessary rebalances. Current session_timeout_ms=%d.",
                self.heartbeat_interval_ms,
                self.session_timeout_ms // 3,
                self.session_timeout_ms,
            )
        return self

    @model_validator(mode="after")
    def validate_auth_config(self) -> ModelKafkaEventBusConfig:
        """Validate authentication configuration consistency.

        Enforces:
        - If security_protocol is SASL_PLAINTEXT or SASL_SSL, sasl_mechanism must be set
        - If sasl_mechanism is set, security_protocol must be SASL_PLAINTEXT or SASL_SSL
        - If sasl_mechanism is OAUTHBEARER, all three OAuth fields must be non-empty

        Returns:
            Self after validation

        Raises:
            ProtocolConfigurationError: If auth configuration is inconsistent
        """
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="validate_auth_config",
            target_name="kafka_config",
        )

        if (
            self.security_protocol in ("SASL_PLAINTEXT", "SASL_SSL")
            and self.sasl_mechanism is None
        ):
            raise ProtocolConfigurationError(
                "security_protocol requires sasl_mechanism when using SASL_*",
                context=context,
                parameter="sasl_mechanism",
                value=self.sasl_mechanism,
            )

        if self.sasl_mechanism is not None:
            if self.security_protocol not in ("SASL_PLAINTEXT", "SASL_SSL"):
                raise ProtocolConfigurationError(
                    f"sasl_mechanism={self.sasl_mechanism!r} requires security_protocol "
                    f"'SASL_PLAINTEXT' or 'SASL_SSL', got {self.security_protocol!r}",
                    context=context,
                    parameter="security_protocol",
                    value=self.security_protocol,
                )
            if self.sasl_mechanism == "OAUTHBEARER":
                missing = [
                    field
                    for field, val in (
                        (
                            "sasl_oauthbearer_token_endpoint_url",
                            self.sasl_oauthbearer_token_endpoint_url,
                        ),
                        ("sasl_oauthbearer_client_id", self.sasl_oauthbearer_client_id),
                        (
                            "sasl_oauthbearer_client_secret",
                            self.sasl_oauthbearer_client_secret,
                        ),
                    )
                    if val is None or (isinstance(val, str) and not val.strip())
                ]
                if missing:
                    raise ProtocolConfigurationError(
                        "sasl_mechanism='OAUTHBEARER' requires non-empty OAuth fields: "
                        + ", ".join(missing),
                        context=context,
                        parameter="sasl_mechanism",
                        value=self.sasl_mechanism,
                    )
        return self

    # Instance discriminator for multi-container dev environments (OMN-2251)
    instance_id: str | None = Field(
        default=None,
        description=(
            "Instance discriminator for consumer group IDs. When set, appended "
            "as '.__i.{instance_id}' to consumer group IDs so that each container "
            "in a multi-container dev environment gets unique consumer group "
            "membership and proper partition assignment. When None (default), "
            "consumer group IDs are unchanged (single-container behavior)."
        ),
    )

    @field_validator("instance_id", mode="before")
    @classmethod
    def validate_instance_id(cls, v: object) -> str | None:
        """Validate instance_id contains only Kafka-safe characters.

        Rejects values containing characters other than ``[a-zA-Z0-9._-]``,
        which is the same character set accepted by
        ``normalize_kafka_identifier``. This catches invalid characters
        (slashes, spaces, etc.) at config time rather than silently
        normalizing them at runtime.

        Args:
            v: Instance ID value (any type before Pydantic conversion).

        Returns:
            The validated instance_id string, or None if not provided.

        Raises:
            ValueError: If instance_id contains characters outside
                ``[a-zA-Z0-9._-]``.
        """
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(f"instance_id must be a string, got {type(v).__name__}")
        if not v.strip():
            return None
        if not re.match(r"^[a-zA-Z0-9._-]+$", v):
            raise ValueError(
                f"instance_id {v!r} contains invalid characters. "
                "Only alphanumeric characters, periods (.), underscores (_), "
                "and hyphens (-) are allowed."
            )
        return v

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

        Accepts any non-empty string as an environment identifier.
        The environment is used for consumer groups and logging, not
        for topic naming (topics are realm-agnostic).

        Args:
            v: Environment value (any type before Pydantic conversion)

        Returns:
            Normalized environment string

        Raises:
            ProtocolConfigurationError: If environment is None, not a string,
                or empty

        .. versionchanged:: 0.21.0
            OMN-5189: Simplified from EnumKafkaEnvironment coercion to plain
            string validation. Any non-empty string is accepted.
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
            # Coerce StrEnum and other string-like values
            v = str(v)
        v = v.strip()
        if not v:
            raise ProtocolConfigurationError(
                "environment cannot be empty",
                context=context,
                parameter="environment",
                value=v,
            )
        return v

    def apply_environment_overrides(self) -> ModelKafkaEventBusConfig:
        """Apply environment variable overrides to configuration.

        Environment variables are mapped as follows:
            - KAFKA_BOOTSTRAP_SERVERS -> bootstrap_servers
            - KAFKA_TIMEOUT_SECONDS -> timeout_seconds
            - KAFKA_ENVIRONMENT -> environment
            - KAFKA_MAX_RETRY_ATTEMPTS -> max_retry_attempts
            - KAFKA_CIRCUIT_BREAKER_THRESHOLD -> circuit_breaker_threshold
            - KAFKA_SECURITY_PROTOCOL -> security_protocol
            - KAFKA_SASL_MECHANISM -> sasl_mechanism
            - KAFKA_SASL_OAUTHBEARER_TOKEN_ENDPOINT_URL -> sasl_oauthbearer_token_endpoint_url
            - KAFKA_SASL_OAUTHBEARER_CLIENT_ID -> sasl_oauthbearer_client_id
            - KAFKA_SASL_OAUTHBEARER_CLIENT_SECRET -> sasl_oauthbearer_client_secret
            - KAFKA_SSL_CA_FILE -> ssl_ca_file
            - KAFKA_RECONNECT_BACKOFF_MS -> reconnect_backoff_ms
            - KAFKA_RECONNECT_BACKOFF_MAX_MS -> reconnect_backoff_max_ms
            - KAFKA_SESSION_TIMEOUT_MS -> session_timeout_ms
            - KAFKA_HEARTBEAT_INTERVAL_MS -> heartbeat_interval_ms

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
            "KAFKA_SESSION_TIMEOUT_MS": "session_timeout_ms",
            "KAFKA_HEARTBEAT_INTERVAL_MS": "heartbeat_interval_ms",
            "KAFKA_MAX_POLL_INTERVAL_MS": "max_poll_interval_ms",
            "KAFKA_DEAD_LETTER_TOPIC": "dead_letter_topic",
            "KAFKA_INSTANCE_ID": "instance_id",
            "KAFKA_SECURITY_PROTOCOL": "security_protocol",
            "KAFKA_SASL_MECHANISM": "sasl_mechanism",
            "KAFKA_SASL_OAUTHBEARER_TOKEN_ENDPOINT_URL": "sasl_oauthbearer_token_endpoint_url",
            "KAFKA_SASL_OAUTHBEARER_CLIENT_ID": "sasl_oauthbearer_client_id",
            "KAFKA_SASL_OAUTHBEARER_CLIENT_SECRET": "sasl_oauthbearer_client_secret",
            "KAFKA_SSL_CA_FILE": "ssl_ca_file",
            "KAFKA_RECONNECT_BACKOFF_MS": "reconnect_backoff_ms",
            "KAFKA_RECONNECT_BACKOFF_MAX_MS": "reconnect_backoff_max_ms",
            "KAFKA_MAX_REQUEST_SIZE": "max_request_size",
        }

        # Integer fields for type conversion
        int_fields = {
            "timeout_seconds",
            "max_retry_attempts",
            "circuit_breaker_threshold",
            "reconnect_backoff_ms",
            "reconnect_backoff_max_ms",
            "session_timeout_ms",
            "heartbeat_interval_ms",
            "max_poll_interval_ms",
            "max_request_size",
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
            current_data = self.model_dump(exclude={"acks_aiokafka"})  # noqa: model-dump-bare
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
            bootstrap_servers="localhost:19092",
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
            instance_id=None,
            reconnect_backoff_ms=2000,
            reconnect_backoff_max_ms=30000,
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
        Otherwise, builds a realm-agnostic DLQ topic name following
        ONEX conventions.

        DLQ Topic Naming Convention:
            Format: onex.dlq.<category>.v1
            Examples:
                - onex.dlq.intents.v1 (for permanently failed intents)
                - onex.dlq.events.v1 (for permanently failed events)
                - onex.dlq.commands.v1 (for permanently failed commands)

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
            >>> config = ModelKafkaEventBusConfig(environment="local")
            >>> config.get_dlq_topic()
            'onex.dlq.intents.v1'  # onex-topic-allow: pending contract auto-wiring
            >>> config.get_dlq_topic("events")
            'onex.dlq.events.v1'  # onex-topic-allow: pending contract auto-wiring
            >>> # Explicit topic takes precedence
            >>> config = ModelKafkaEventBusConfig(
            ...     environment="local",
            ...     dead_letter_topic="custom-dlq"
            ... )
            >>> config.get_dlq_topic()
            'custom-dlq'

        .. versionchanged:: 0.21.0
            OMN-5189: DLQ topics are now realm-agnostic.
        """
        if self.dead_letter_topic:
            return self.dead_letter_topic

        # Import here to avoid circular imports
        from omnibase_infra.event_bus.topic_constants import build_dlq_topic

        return build_dlq_topic(category)


__all__: list[str] = ["ModelKafkaEventBusConfig"]
