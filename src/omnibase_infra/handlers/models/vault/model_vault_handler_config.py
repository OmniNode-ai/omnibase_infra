# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Handler Configuration Model.

This module provides the Pydantic configuration model for HashiCorp Vault
handler initialization and operation.

Security Note:
    The token field uses SecretStr to prevent accidental logging of
    sensitive credentials. Tokens should come from environment variables,
    never from YAML configuration files.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from omnibase_infra.handlers.models.vault.model_vault_retry_config import (
    ModelVaultRetryConfig,
)


class ModelVaultHandlerConfig(BaseModel):
    """Configuration for HashiCorp Vault handler.

    Security Policy:
        - The token field uses SecretStr to prevent accidental logging
        - Tokens should be provided via environment variables, not config files
        - Never log or expose token values in error messages
        - Use verify_ssl=True in production environments

    Attributes:
        url: Vault server URL (required, e.g., "https://vault.example.com:8200")
        token: Vault authentication token (SecretStr for security, optional)
        namespace: Vault namespace for Vault Enterprise (optional)
        timeout_seconds: Operation timeout in seconds (1.0-300.0, default 30.0)
        verify_ssl: Whether to verify SSL certificates (default True)
        token_renewal_threshold_seconds: Token renewal threshold in seconds (default 300.0)
        default_token_ttl: Default token TTL in seconds when not provided by Vault (default 3600)
        retry: Retry configuration with exponential backoff

    Example:
        >>> from pydantic import SecretStr
        >>> config = ModelVaultHandlerConfig(
        ...     url="https://vault.example.com:8200",
        ...     token=SecretStr("s.1234567890abcdefghijklmnopqrstuv"),
        ...     namespace="engineering",
        ...     timeout_seconds=30.0,
        ...     verify_ssl=True,
        ... )
        >>> # Token is protected from accidental logging
        >>> print(config.token)
        SecretStr('**********')
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    url: str = Field(
        description="Vault server URL (e.g., 'https://vault.example.com:8200')",
    )

    @field_validator("url")
    @classmethod
    def validate_url_format(cls, v: str) -> str:
        """Validate that URL has proper format for Vault server."""
        if not v:
            msg = "URL cannot be empty"
            raise ValueError(msg)
        # Check for valid URL scheme
        if not v.startswith(("http://", "https://")):
            msg = "URL must start with http:// or https://"
            raise ValueError(msg)
        # Basic structure validation (scheme + host at minimum)
        parts = v.split("://", 1)
        if len(parts) != 2 or not parts[1]:
            msg = "URL must have format: scheme://host[:port]"
            raise ValueError(msg)
        return v

    token: SecretStr | None = Field(
        default=None,
        description="Authentication token protected from logging",
    )
    namespace: str | None = Field(
        default=None,
        description="Vault namespace for multi-tenant isolation",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Maximum time allowed per operation",
    )
    verify_ssl: bool = Field(
        default=True,
        description="Enable SSL certificate verification",
    )
    token_renewal_threshold_seconds: float = Field(
        default=300.0,
        ge=0.0,
        description="Trigger renewal when remaining TTL falls below this value",
    )
    default_token_ttl: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Fallback TTL when Vault response does not provide TTL",
    )
    retry: ModelVaultRetryConfig = Field(
        default_factory=ModelVaultRetryConfig,
        description="Retry behavior for failed operations",
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
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable automatic failure detection and recovery",
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


__all__: list[str] = ["ModelVaultHandlerConfig"]
