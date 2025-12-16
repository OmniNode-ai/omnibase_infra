# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Handler Configuration Models.

This module provides Pydantic configuration models for HashiCorp Vault
handler initialization and operation.

Security Note:
    The token field uses SecretStr to prevent accidental logging of
    sensitive credentials. Tokens should come from environment variables,
    never from YAML configuration files.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ModelVaultRetryConfig(BaseModel):
    """Configuration for Vault operation retry logic with exponential backoff.

    Attributes:
        max_attempts: Maximum number of retry attempts (1-10)
        initial_backoff_seconds: Initial backoff delay in seconds (0.01-10.0)
        max_backoff_seconds: Maximum backoff delay in seconds (1.0-60.0)
        exponential_base: Exponential backoff multiplier (1.5-4.0)

    Example:
        >>> retry_config = ModelVaultRetryConfig(
        ...     max_attempts=3,
        ...     initial_backoff_seconds=0.1,
        ...     max_backoff_seconds=10.0,
        ...     exponential_base=2.0,
        ... )
        >>> # Backoff sequence: 0.1s, 0.2s, 0.4s
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retry attempts",
    )
    initial_backoff_seconds: float = Field(
        default=0.1,
        ge=0.01,
        le=10.0,
        description="Initial backoff delay in seconds",
    )
    max_backoff_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Maximum backoff delay in seconds",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=4.0,
        description="Exponential backoff multiplier",
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
    retry: ModelVaultRetryConfig = Field(
        default_factory=ModelVaultRetryConfig,
        description="Retry configuration with exponential backoff",
    )


__all__: list[str] = [
    "ModelVaultRetryConfig",
    "ModelVaultHandlerConfig",
]
