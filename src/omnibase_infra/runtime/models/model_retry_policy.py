# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Retry policy configuration model.

.. versionadded:: 0.8.0
    Initial implementation for OMN-765.

This module provides the retry policy model used by binding configurations.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class ModelRetryPolicy(BaseModel):
    """Retry policy configuration for handler operations.

    Defines how failed handler operations should be retried, including
    the maximum number of attempts and the backoff strategy.

    Attributes:
        max_retries: Maximum number of retry attempts (0 means no retries).
        backoff_strategy: Strategy for calculating delays between retries.
            - "fixed": Same delay between each retry.
            - "exponential": Delay doubles with each retry.
        base_delay_ms: Initial delay in milliseconds before first retry.
        max_delay_ms: Maximum delay in milliseconds (caps exponential growth).

    Example:
        >>> policy = ModelRetryPolicy(
        ...     max_retries=5,
        ...     backoff_strategy="exponential",
        ...     base_delay_ms=100,
        ...     max_delay_ms=10000,
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts (0-10). "
        "0 means no retries; operation fails immediately on first error.",
    )

    backoff_strategy: Literal["fixed", "exponential"] = Field(
        default="exponential",
        description="Backoff strategy: 'fixed' uses constant delay, "
        "'exponential' doubles delay on each retry.",
    )

    base_delay_ms: int = Field(
        default=100,
        ge=10,
        le=60000,
        description="Base delay in milliseconds before first retry (10-60000). "
        "For exponential backoff, subsequent delays are base_delay * 2^attempt.",
    )

    max_delay_ms: int = Field(
        default=5000,
        ge=100,
        le=300000,
        description="Maximum delay in milliseconds between retries (100-300000). "
        "Caps exponential backoff growth to prevent excessive waits.",
    )

    @field_validator("max_delay_ms")
    @classmethod
    def validate_max_delay_greater_than_base(cls, v: int, info: ValidationInfo) -> int:
        """Ensure max_delay_ms is at least as large as base_delay_ms.

        Args:
            v: The max_delay_ms value to validate.
            info: Pydantic validation info containing other field values.

        Returns:
            The validated max_delay_ms value.

        Raises:
            ValueError: If max_delay_ms is less than base_delay_ms.
        """
        # Access data from validation info
        data = info.data if info.data else {}
        base_delay = data.get("base_delay_ms", 100)
        if v < base_delay:
            raise ValueError(
                f"max_delay_ms ({v}) must be >= base_delay_ms ({base_delay})"
            )
        return v


__all__: list[str] = [
    "ModelRetryPolicy",
]
