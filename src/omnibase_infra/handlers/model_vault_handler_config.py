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

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from omnibase_infra.handlers.model_vault_retry_config import ModelVaultRetryConfig


class ModelVaultAdapterConfig(BaseModel):
    """Configuration for HashiCorp Vault adapter.

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
        >>> config = ModelVaultAdapterConfig(
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
    token: SecretStr | None = Field(
        default=None,
        description="Vault authentication token (use SecretStr for security)",
    )
    namespace: str | None = Field(
        default=None,
        description="Vault namespace for Vault Enterprise",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Operation timeout in seconds",
    )
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates",
    )
    token_renewal_threshold_seconds: float = Field(
        default=300.0,
        ge=0.0,
        description="Token renewal threshold in seconds (renew when TTL below this)",
    )
    default_token_ttl: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Default token TTL in seconds when not provided by Vault (minimum 300s)",
    )
    retry: ModelVaultRetryConfig = Field(
        default_factory=ModelVaultRetryConfig,
        description="Retry configuration with exponential backoff",
    )
    max_concurrent_operations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent Vault operations (thread pool size)",
    )
    max_queue_size_multiplier: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Queue size multiplier (queue_size = max_workers * multiplier)",
    )
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker pattern for error recovery",
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of consecutive failures before opening circuit",
    )
    circuit_breaker_reset_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Seconds to wait before attempting to close opened circuit",
    )


__all__: list[str] = ["ModelVaultAdapterConfig"]
