# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Emit Daemon Configuration Model.

This module provides the Pydantic configuration model for the Hook Event Emit Daemon.
The daemon provides a Unix socket interface for Claude Code hooks to emit events
to Kafka without blocking hook execution.

Configuration includes:
    - Socket and PID file paths
    - Spool directory for message persistence during Kafka unavailability
    - Memory and disk limits for backpressure management
    - Kafka connection settings
    - Timeout configurations for graceful operations

Environment Variable Overrides:
    EMIT_DAEMON_SOCKET_PATH: Override socket_path
    EMIT_DAEMON_PID_PATH: Override pid_path
    EMIT_DAEMON_SPOOL_DIR: Override spool_dir
    EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS: Override kafka_bootstrap_servers
    EMIT_DAEMON_KAFKA_CLIENT_ID: Override kafka_client_id
    EMIT_DAEMON_ENVIRONMENT: Override environment
    EMIT_DAEMON_MAX_RETRY_ATTEMPTS: Override max_retry_attempts
    EMIT_DAEMON_BACKOFF_BASE_SECONDS: Override backoff_base_seconds
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelEmitDaemonConfig(BaseModel):
    """Configuration model for the Hook Event Emit Daemon.

    The emit daemon provides a non-blocking interface for Claude Code hooks
    to emit events to Kafka. This configuration controls all operational
    parameters including paths, limits, and timeouts.

    Attributes:
        socket_path: Unix domain socket path for client connections
        pid_path: PID file path for daemon process management
        spool_dir: Directory for spooling messages when Kafka is unavailable
        max_payload_bytes: Maximum allowed payload size per message
        max_memory_queue: Maximum messages to hold in memory queue
        max_spool_messages: Maximum messages to persist in spool directory
        max_spool_bytes: Maximum total bytes in spool directory
        kafka_bootstrap_servers: Kafka broker addresses (host:port format)
        kafka_client_id: Client identifier for Kafka producer
        environment: Deployment environment for topic naming
        socket_timeout_seconds: Timeout for socket read/write operations
        kafka_timeout_seconds: Timeout for Kafka produce operations
        shutdown_drain_seconds: Time to drain queues during graceful shutdown
        max_retry_attempts: Maximum retry attempts before dropping an event
        backoff_base_seconds: Base backoff delay in seconds for exponential backoff

    Example:
        >>> config = ModelEmitDaemonConfig(
        ...     kafka_bootstrap_servers="192.168.86.200:29092",
        ...     socket_path=Path("/tmp/my-emit.sock"),
        ... )
        >>> print(config.max_payload_bytes)
        1048576

        >>> # Load with environment overrides
        >>> config = ModelEmitDaemonConfig.with_env_overrides(
        ...     kafka_bootstrap_servers="localhost:9092"
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
        validate_default=True,
    )

    # Path configurations
    # NOTE: /tmp is standard for Unix domain sockets - not a security issue
    socket_path: Path = Field(
        default=Path("/tmp/omniclaude-emit.sock"),  # noqa: S108
        description="Unix domain socket path for client connections",
    )
    pid_path: Path = Field(
        default=Path("/tmp/omniclaude-emit.pid"),  # noqa: S108
        description="PID file path for daemon process management",
    )
    spool_dir: Path = Field(
        default_factory=lambda: Path.home() / ".omniclaude" / "emit-spool",
        description="Directory for spooling messages when Kafka is unavailable",
    )

    # Limit configurations
    max_payload_bytes: int = Field(
        default=1_048_576,  # 1MB
        ge=1024,  # Minimum 1KB
        le=10_485_760,  # Maximum 10MB
        description="Maximum allowed payload size per message in bytes",
    )
    max_memory_queue: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Maximum messages to hold in memory queue",
    )
    max_spool_messages: int = Field(
        default=1000,
        ge=0,  # 0 disables spooling
        le=100_000,
        description="Maximum messages to persist in spool directory",
    )
    max_spool_bytes: int = Field(
        default=10_485_760,  # 10MB
        ge=0,  # 0 disables spooling
        le=1_073_741_824,  # Maximum 1GB
        description="Maximum total bytes in spool directory",
    )

    # Kafka configurations
    kafka_bootstrap_servers: str = Field(
        ...,  # Required, no default
        min_length=1,
        description="Kafka broker addresses (host:port format, comma-separated for multiple)",
    )
    # ONEX_EXCLUDE: string_id - kafka_client_id is Kafka identifier, not UUID
    kafka_client_id: str = Field(
        default="emit-daemon",
        min_length=1,
        max_length=255,
        description="Client identifier for Kafka producer",
    )
    environment: str = Field(
        default="dev",
        pattern=r"^[a-z][a-z0-9-]*$",
        description="Deployment environment (e.g., 'dev', 'staging', 'prod'). Used in topic names.",
    )

    # Timeout configurations
    socket_timeout_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Timeout for socket read/write operations in seconds",
    )
    kafka_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for Kafka produce operations in seconds",
    )
    shutdown_drain_seconds: float = Field(
        default=10.0,
        ge=0.0,
        le=300.0,
        description="Time to drain queues during graceful shutdown in seconds",
    )

    # Retry configurations
    max_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts before dropping an event",
    )
    backoff_base_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Base backoff delay in seconds for exponential backoff",
    )

    @field_validator("socket_path", "pid_path", mode="after")
    @classmethod
    def validate_file_path_parent_exists_or_creatable(cls, v: Path) -> Path:
        """Validate that file path's parent directory exists or can be created.

        For socket and PID files, we validate that the parent directory either
        exists or can be created (i.e., its parent exists).

        Args:
            v: The path to validate

        Returns:
            The validated path

        Raises:
            ValueError: If the parent directory path is invalid
        """
        parent = v.parent
        if parent.exists():
            if not parent.is_dir():
                raise ValueError(f"Parent path exists but is not a directory: {parent}")
            return v

        # Check if grandparent exists (parent can be created)
        grandparent = parent.parent
        if grandparent.exists() and grandparent.is_dir():
            return v

        raise ValueError(
            f"Parent directory does not exist and cannot be created: {parent}"
        )

    @field_validator("spool_dir", mode="after")
    @classmethod
    def validate_spool_dir_creatable(cls, v: Path) -> Path:
        """Validate that spool directory exists or can be created.

        The spool directory may be nested (e.g., ~/.omniclaude/emit-spool),
        so we validate that at least one ancestor exists.

        Args:
            v: The spool directory path to validate

        Returns:
            The validated path

        Raises:
            ValueError: If no valid ancestor exists for directory creation
        """
        if v.exists():
            if not v.is_dir():
                raise ValueError(f"Spool path exists but is not a directory: {v}")
            return v

        # Walk up the path to find an existing ancestor
        current = v
        while current != current.parent:  # Stop at filesystem root
            current = current.parent
            if current.exists():
                if current.is_dir():
                    return v
                raise ValueError(
                    f"Ancestor path exists but is not a directory: {current}"
                )

        raise ValueError(f"No valid ancestor directory found for spool path: {v}")

    @field_validator("kafka_bootstrap_servers", mode="after")
    @classmethod
    def validate_bootstrap_servers_format(cls, v: str) -> str:
        """Validate Kafka bootstrap servers format.

        Each server must be in host:port format with a valid port number.

        Args:
            v: Bootstrap servers string

        Returns:
            The validated bootstrap servers string

        Raises:
            ValueError: If the format is invalid
        """
        servers = v.strip().split(",")
        for server in servers:
            server = server.strip()
            if not server:
                raise ValueError("Bootstrap servers cannot contain empty entries")
            if ":" not in server:
                raise ValueError(
                    f"Invalid bootstrap server format '{server}'. "
                    "Expected 'host:port' (e.g., 'localhost:9092')"
                )
            host, port_str = server.rsplit(":", 1)
            if not host:
                raise ValueError(
                    f"Invalid bootstrap server format '{server}'. Host cannot be empty"
                )
            try:
                port = int(port_str)
                if port < 1 or port > 65535:
                    raise ValueError(
                        f"Invalid port {port} in '{server}'. "
                        "Port must be between 1 and 65535"
                    )
            except ValueError as e:
                if "Invalid port" in str(e):
                    raise
                raise ValueError(
                    f"Invalid port '{port_str}' in '{server}'. "
                    "Port must be a valid integer"
                ) from e

        return v.strip()

    @model_validator(mode="after")
    def validate_spool_limits_consistency(self) -> ModelEmitDaemonConfig:
        """Validate that spool limits are consistent.

        If max_spool_messages is 0 (spooling disabled), max_spool_bytes
        should also be 0, and vice versa.

        Returns:
            The validated model instance

        Raises:
            ValueError: If spool limits are inconsistent
        """
        if self.max_spool_messages == 0 and self.max_spool_bytes > 0:
            raise ValueError(
                "Inconsistent spool limits: max_spool_messages is 0 (disabled) "
                "but max_spool_bytes is non-zero. Set both to 0 to disable spooling."
            )
        if self.max_spool_bytes == 0 and self.max_spool_messages > 0:
            raise ValueError(
                "Inconsistent spool limits: max_spool_bytes is 0 (disabled) "
                "but max_spool_messages is non-zero. Set both to 0 to disable spooling."
            )
        return self

    @classmethod
    def with_env_overrides(cls, **kwargs: object) -> ModelEmitDaemonConfig:
        """Create configuration with environment variable overrides.

        Environment variables take precedence over provided kwargs.
        If an environment variable is set, it overrides the corresponding
        kwarg value.

        Environment Variable Mapping:
            EMIT_DAEMON_SOCKET_PATH -> socket_path
            EMIT_DAEMON_PID_PATH -> pid_path
            EMIT_DAEMON_SPOOL_DIR -> spool_dir
            EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS -> kafka_bootstrap_servers
            EMIT_DAEMON_KAFKA_CLIENT_ID -> kafka_client_id
            EMIT_DAEMON_MAX_PAYLOAD_BYTES -> max_payload_bytes
            EMIT_DAEMON_MAX_MEMORY_QUEUE -> max_memory_queue
            EMIT_DAEMON_MAX_SPOOL_MESSAGES -> max_spool_messages
            EMIT_DAEMON_MAX_SPOOL_BYTES -> max_spool_bytes
            EMIT_DAEMON_SOCKET_TIMEOUT_SECONDS -> socket_timeout_seconds
            EMIT_DAEMON_KAFKA_TIMEOUT_SECONDS -> kafka_timeout_seconds
            EMIT_DAEMON_SHUTDOWN_DRAIN_SECONDS -> shutdown_drain_seconds
            EMIT_DAEMON_MAX_RETRY_ATTEMPTS -> max_retry_attempts
            EMIT_DAEMON_BACKOFF_BASE_SECONDS -> backoff_base_seconds

        Args:
            **kwargs: Base configuration values to use if env vars not set

        Returns:
            Configuration instance with environment overrides applied

        Example:
            >>> import os
            >>> os.environ["EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS"] = "kafka:9092"
            >>> config = ModelEmitDaemonConfig.with_env_overrides(
            ...     kafka_bootstrap_servers="localhost:9092"  # Overridden by env
            ... )
            >>> config.kafka_bootstrap_servers
            'kafka:9092'
        """
        env_mappings: dict[str, tuple[str, type]] = {
            "EMIT_DAEMON_SOCKET_PATH": ("socket_path", Path),
            "EMIT_DAEMON_PID_PATH": ("pid_path", Path),
            "EMIT_DAEMON_SPOOL_DIR": ("spool_dir", Path),
            "EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS": ("kafka_bootstrap_servers", str),
            "EMIT_DAEMON_KAFKA_CLIENT_ID": ("kafka_client_id", str),
            "EMIT_DAEMON_ENVIRONMENT": ("environment", str),
            "EMIT_DAEMON_MAX_PAYLOAD_BYTES": ("max_payload_bytes", int),
            "EMIT_DAEMON_MAX_MEMORY_QUEUE": ("max_memory_queue", int),
            "EMIT_DAEMON_MAX_SPOOL_MESSAGES": ("max_spool_messages", int),
            "EMIT_DAEMON_MAX_SPOOL_BYTES": ("max_spool_bytes", int),
            "EMIT_DAEMON_SOCKET_TIMEOUT_SECONDS": ("socket_timeout_seconds", float),
            "EMIT_DAEMON_KAFKA_TIMEOUT_SECONDS": ("kafka_timeout_seconds", float),
            "EMIT_DAEMON_SHUTDOWN_DRAIN_SECONDS": ("shutdown_drain_seconds", float),
            "EMIT_DAEMON_MAX_RETRY_ATTEMPTS": ("max_retry_attempts", int),
            "EMIT_DAEMON_BACKOFF_BASE_SECONDS": ("backoff_base_seconds", float),
        }

        # Build final configuration dict with proper types
        final_config: dict[str, Path | str | int | float] = {}

        # First, apply provided kwargs (convert to proper types)
        for key, value in kwargs.items():
            if value is not None:
                # Trust that kwargs are already properly typed by caller
                final_config[key] = value  # type: ignore[assignment]

        # Then, apply environment variable overrides
        for env_var, (field_name, field_type) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    if field_type is Path:
                        final_config[field_name] = Path(env_value)
                    elif field_type is int:
                        final_config[field_name] = int(env_value)
                    elif field_type is float:
                        final_config[field_name] = float(env_value)
                    else:
                        final_config[field_name] = env_value
                except ValueError:
                    # Skip invalid env values, let Pydantic validation handle
                    pass

        return cls.model_validate(final_config)

    @property
    def spooling_enabled(self) -> bool:
        """Check if message spooling is enabled.

        Returns:
            True if both max_spool_messages and max_spool_bytes are non-zero
        """
        return self.max_spool_messages > 0 and self.max_spool_bytes > 0


__all__: list[str] = ["ModelEmitDaemonConfig"]
