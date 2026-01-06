# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Handler Configuration Model.

This module provides the Pydantic configuration model for HashiCorp Consul
handler initialization and operation.

Security Note:
    The token field uses SecretStr to prevent accidental logging of
    sensitive ACL tokens. Tokens should come from environment variables,
    never from YAML configuration files.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from omnibase_infra.handlers.models.consul.model_consul_retry_config import (
    ModelConsulRetryConfig,
)


class ModelConsulHandlerConfig(BaseModel):
    """Configuration for HashiCorp Consul handler.

    Security Policy:
        - The token field uses SecretStr to prevent accidental logging
        - ACL tokens should be provided via environment variables, not config files
        - Never log or expose token values in error messages
        - Use scheme="https" in production environments

    Attributes:
        host: Consul server hostname (default "localhost")
        port: Consul server port (1-65535, default 8500)
        scheme: HTTP scheme for connection (default "http")
        token: Consul ACL token (SecretStr for security, optional)
        timeout_seconds: Operation timeout in seconds (1.0-300.0, default 30.0)
        connect_timeout_seconds: Connection timeout in seconds (1.0-60.0, default 10.0)
        retry: Retry configuration with exponential backoff
        health_check_interval_seconds: Interval for health checks (5.0-300.0, default 30.0)
        datacenter: Consul datacenter for multi-datacenter deployments (optional)

    Example:
        >>> from pydantic import SecretStr
        >>> config = ModelConsulHandlerConfig(
        ...     host="consul.example.com",
        ...     port=8500,
        ...     scheme="https",
        ...     token=SecretStr("acl-token-abc123"),
        ...     datacenter="dc1",
        ... )
        >>> # Token is protected from accidental logging
        >>> print(config.token)
        SecretStr('**********')
        >>> # Base URL property
        >>> print(config.base_url)
        https://consul.example.com:8500
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    host: str = Field(
        default="localhost",
        min_length=1,
        description="Consul server hostname",
    )
    port: int = Field(
        default=8500,
        ge=1,
        le=65535,
        description="Consul server port",
    )
    scheme: Literal["http", "https"] = Field(
        default="http",
        description="HTTP scheme for Consul connection",
    )
    token: SecretStr | None = Field(
        default=None,
        description="Consul ACL token protected from logging",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Maximum time allowed per operation",
    )
    connect_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Maximum time allowed for connection establishment",
    )
    retry: ModelConsulRetryConfig = Field(
        default_factory=ModelConsulRetryConfig,
        description="Retry behavior for failed operations",
    )
    health_check_interval_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Interval between health check executions",
    )
    datacenter: str | None = Field(
        default=None,
        description="Consul datacenter for multi-datacenter deployments",
    )
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable automatic failure detection and recovery via circuit breaker",
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Consecutive failures required to open circuit",
    )
    circuit_breaker_reset_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Wait time before testing recovery after circuit opens",
    )
    max_concurrent_operations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Thread pool size for concurrent operations",
    )
    max_queue_size_multiplier: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Multiplier to calculate queue capacity from thread pool size",
    )

    @property
    def base_url(self) -> str:
        """Construct the base URL for Consul API requests.

        Returns:
            Complete base URL in format scheme://host:port
        """
        return f"{self.scheme}://{self.host}:{self.port}"


__all__: list[str] = ["ModelConsulHandlerConfig"]
