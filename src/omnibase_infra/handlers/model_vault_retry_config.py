# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Retry Configuration Model.

This module provides the Pydantic configuration model for Vault operation
retry logic with exponential backoff.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
        from_attributes=True,
    )

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed operations",
    )
    initial_backoff_seconds: float = Field(
        default=0.1,
        ge=0.01,
        le=10.0,
        description="Initial delay before first retry",
    )
    max_backoff_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Maximum delay cap for exponential backoff",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=4.0,
        description="Base multiplier for exponential backoff calculation",
    )


__all__: list[str] = ["ModelVaultRetryConfig"]
