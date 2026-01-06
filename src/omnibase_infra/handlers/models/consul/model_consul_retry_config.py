# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Retry Configuration Model.

This module provides the Pydantic configuration model for Consul operation
retry logic with exponential backoff.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelConsulRetryConfig(BaseModel):
    """Configuration for Consul operation retry logic with exponential backoff.

    Attributes:
        max_attempts: Maximum number of retry attempts (1-10)
        initial_delay_seconds: Initial backoff delay in seconds (0.1-60.0)
        max_delay_seconds: Maximum backoff delay in seconds (1.0-300.0)
        exponential_base: Exponential backoff multiplier (1.5-4.0)

    Example:
        >>> retry_config = ModelConsulRetryConfig(
        ...     max_attempts=3,
        ...     initial_delay_seconds=1.0,
        ...     max_delay_seconds=30.0,
        ...     exponential_base=2.0,
        ... )
        >>> # Backoff sequence: 1.0s, 2.0s, 4.0s, 8.0s, 16.0s (capped at 30.0s)
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
    initial_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial delay before first retry",
    )
    max_delay_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay cap for exponential backoff",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=4.0,
        description="Base multiplier for exponential backoff calculation",
    )


__all__: list[str] = ["ModelConsulRetryConfig"]
