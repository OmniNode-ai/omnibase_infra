# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consumer-side retry configuration model.

This module provides the configuration model for consumer-side retry behavior
when message handlers fail. It distinguishes between:

- **Content errors** (non-retryable): Malformed messages, validation failures,
  business logic errors. These will fail regardless of retry attempts.

- **Infrastructure errors** (retryable): Network timeouts, temporary service
  unavailability, rate limiting. These may succeed on retry.

The model uses exponential backoff with optional jitter to prevent thundering
herd problems in distributed systems.

Example:
    >>> config = ModelConsumerRetryConfig(
    ...     max_attempts=5,
    ...     backoff_ms=500,
    ...     backoff_multiplier=2.0,
    ...     jitter_enabled=True,
    ... )
    >>> config.calculate_delay_ms(attempt=3)  # Returns ~2000ms + jitter

See Also:
    - docs/patterns/error_recovery_patterns.md: Error recovery patterns
    - docs/patterns/dispatcher_resilience.md: Dispatcher resilience patterns
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConsumerRetryConfig(BaseModel):
    """Consumer-side retry configuration.

    Controls retry behavior when message handlers fail. Distinguishes between
    content errors (non-retryable) and infrastructure errors (retryable).

    Attributes:
        max_attempts: Maximum retry attempts before giving up. Includes the
            initial attempt, so max_attempts=3 means 1 initial + 2 retries.
        backoff_ms: Base backoff delay in milliseconds. For exponential
            backoff, subsequent delays are backoff_ms * (multiplier ^ attempt).
        backoff_multiplier: Exponential backoff multiplier. A value of 2.0
            doubles the delay with each retry.
        jitter_enabled: When True, adds random jitter (0-25% of delay) to
            prevent thundering herd when multiple consumers retry simultaneously.
        backoff_strategy: Strategy for calculating delays between retries.
            "exponential" doubles delay each retry, "fixed" uses constant delay.
        max_backoff_ms: Maximum backoff delay cap to prevent excessive waits.

    Example:
        ```python
        from omnibase_infra.models.event_bus import ModelConsumerRetryConfig

        # Standard configuration with exponential backoff
        config = ModelConsumerRetryConfig(
            max_attempts=3,
            backoff_ms=1000,
            backoff_multiplier=2.0,
            jitter_enabled=True,
        )

        # Conservative configuration for critical operations
        config = ModelConsumerRetryConfig.create_conservative()

        # Aggressive configuration for resilient operations
        config = ModelConsumerRetryConfig.create_aggressive()
        ```

    Configuration Guidelines:
        - Critical operations: Use lower max_attempts (2-3), higher backoff
        - Best-effort operations: Use higher max_attempts (5+), lower backoff
        - High-concurrency: Always enable jitter to prevent thundering herd
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "max_attempts": 3,
                    "backoff_ms": 1000,
                    "backoff_multiplier": 2.0,
                    "jitter_enabled": True,
                    "backoff_strategy": "exponential",
                    "max_backoff_ms": 30000,
                },
                {
                    "max_attempts": 5,
                    "backoff_ms": 500,
                    "backoff_multiplier": 1.5,
                    "jitter_enabled": True,
                    "backoff_strategy": "exponential",
                    "max_backoff_ms": 60000,
                },
            ]
        },
    )

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts before giving up (1-10). "
        "Includes initial attempt, so 3 means 1 initial + 2 retries.",
    )

    backoff_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Base backoff delay in milliseconds (100-60000). "
        "For exponential backoff, subsequent delays are backoff_ms * (multiplier ^ attempt).",
    )

    backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Exponential backoff multiplier (1.0-4.0). "
        "A value of 2.0 doubles the delay with each retry.",
    )

    jitter_enabled: bool = Field(
        default=True,
        description="Add random jitter (0-25% of delay) to backoff. "
        "Prevents thundering herd when multiple consumers retry simultaneously.",
    )

    backoff_strategy: Literal["exponential", "fixed"] = Field(
        default="exponential",
        description="Backoff strategy: 'exponential' multiplies delay each retry, "
        "'fixed' uses constant delay.",
    )

    max_backoff_ms: int = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Maximum backoff delay cap in milliseconds (1000-300000). "
        "Prevents excessive waits in exponential backoff.",
    )

    @field_validator("max_backoff_ms")
    @classmethod
    def validate_max_backoff_greater_than_base(cls, v: int, info: object) -> int:
        """Ensure max_backoff_ms is at least as large as backoff_ms.

        Args:
            v: The max_backoff_ms value to validate.
            info: Pydantic validation info containing other field values.

        Returns:
            The validated max_backoff_ms value.

        Raises:
            ValueError: If max_backoff_ms is less than backoff_ms.
        """
        # Access data from validation info
        # Use getattr for safer access pattern
        data = getattr(info, "data", None) or {}
        base_delay = data.get("backoff_ms", 1000)
        if v < base_delay:
            msg = f"max_backoff_ms ({v}) must be >= backoff_ms ({base_delay})"
            raise ValueError(msg)
        return v

    def calculate_delay_ms(
        self, attempt: int, include_jitter: bool | None = None
    ) -> int:
        """Calculate delay in milliseconds for a specific retry attempt.

        Args:
            attempt: The retry attempt number (1-based). Attempt 1 is the first
                retry after the initial failure.
            include_jitter: Whether to include jitter. If None, uses the
                jitter_enabled setting from configuration.

        Returns:
            Delay in milliseconds for the specified attempt, capped at max_backoff_ms.

        Raises:
            ValueError: If attempt is less than 1 or greater than max_attempts.

        Example:
            >>> config = ModelConsumerRetryConfig(backoff_ms=1000, backoff_multiplier=2.0)
            >>> config.calculate_delay_ms(1)  # ~1000ms
            >>> config.calculate_delay_ms(2)  # ~2000ms
            >>> config.calculate_delay_ms(3)  # ~4000ms
        """
        if attempt < 1:
            msg = f"Attempt must be >= 1, got {attempt}"
            raise ValueError(msg)

        if attempt > self.max_attempts:
            msg = f"Attempt {attempt} exceeds max_attempts {self.max_attempts}"
            raise ValueError(msg)

        if self.backoff_strategy == "exponential":
            # Exponential: backoff_ms * (multiplier ^ (attempt - 1))
            delay = self.backoff_ms * (self.backoff_multiplier ** (attempt - 1))
        else:
            # Fixed: constant delay
            delay = float(self.backoff_ms)

        # Cap at max_backoff_ms
        delay = min(delay, float(self.max_backoff_ms))

        # Add jitter if enabled
        use_jitter = (
            include_jitter if include_jitter is not None else self.jitter_enabled
        )
        if use_jitter:
            # Add 0-25% random jitter
            jitter_factor = random.random() * 0.25
            delay += delay * jitter_factor

        return int(delay)

    def get_all_delays_ms(self, include_jitter: bool = False) -> list[int]:
        """Get all delay times for the complete retry sequence.

        Args:
            include_jitter: Whether to include jitter in calculations.
                Defaults to False for predictable results.

        Returns:
            List of delays in milliseconds for each retry attempt.

        Example:
            >>> config = ModelConsumerRetryConfig(max_attempts=3, backoff_ms=1000)
            >>> config.get_all_delays_ms()
            [1000, 2000, 4000]
        """
        return [
            self.calculate_delay_ms(i, include_jitter=include_jitter)
            for i in range(1, self.max_attempts + 1)
        ]

    def get_total_retry_time_ms(self) -> int:
        """Get total time that all retries will take.

        Returns:
            Total time in milliseconds for all retry attempts (without jitter).

        Example:
            >>> config = ModelConsumerRetryConfig(max_attempts=3, backoff_ms=1000)
            >>> config.get_total_retry_time_ms()
            7000  # 1000 + 2000 + 4000
        """
        return sum(self.get_all_delays_ms(include_jitter=False))

    @classmethod
    def create_conservative(cls) -> ModelConsumerRetryConfig:
        """Create conservative retry configuration for critical operations.

        Conservative configuration minimizes retry attempts and uses longer
        delays to reduce load on failing services.

        Returns:
            ModelConsumerRetryConfig with conservative settings.

        Example:
            >>> config = ModelConsumerRetryConfig.create_conservative()
            >>> config.max_attempts
            2
        """
        return cls(
            max_attempts=2,
            backoff_ms=2000,
            backoff_multiplier=2.0,
            jitter_enabled=True,
            backoff_strategy="exponential",
            max_backoff_ms=30000,
        )

    @classmethod
    def create_standard(cls) -> ModelConsumerRetryConfig:
        """Create standard retry configuration for typical operations.

        Standard configuration balances reliability with reasonable latency.

        Returns:
            ModelConsumerRetryConfig with standard settings.

        Example:
            >>> config = ModelConsumerRetryConfig.create_standard()
            >>> config.max_attempts
            3
        """
        return cls(
            max_attempts=3,
            backoff_ms=1000,
            backoff_multiplier=2.0,
            jitter_enabled=True,
            backoff_strategy="exponential",
            max_backoff_ms=30000,
        )

    @classmethod
    def create_aggressive(cls) -> ModelConsumerRetryConfig:
        """Create aggressive retry configuration for resilient operations.

        Aggressive configuration maximizes retry attempts with shorter delays
        for operations that need high availability.

        Returns:
            ModelConsumerRetryConfig with aggressive settings.

        Example:
            >>> config = ModelConsumerRetryConfig.create_aggressive()
            >>> config.max_attempts
            5
        """
        return cls(
            max_attempts=5,
            backoff_ms=500,
            backoff_multiplier=1.5,
            jitter_enabled=True,
            backoff_strategy="exponential",
            max_backoff_ms=15000,
        )

    @classmethod
    def create_no_retry(cls) -> ModelConsumerRetryConfig:
        """Create configuration with no retries (fail-fast).

        Use for operations where retries are not appropriate, such as
        idempotency-sensitive operations or when circuit breaker is preferred.

        Returns:
            ModelConsumerRetryConfig with no retry attempts.

        Example:
            >>> config = ModelConsumerRetryConfig.create_no_retry()
            >>> config.max_attempts
            1
        """
        return cls(
            max_attempts=1,
            backoff_ms=100,
            backoff_multiplier=1.0,
            jitter_enabled=False,
            backoff_strategy="fixed",
            max_backoff_ms=1000,
        )


__all__: list[str] = [
    "ModelConsumerRetryConfig",
]
