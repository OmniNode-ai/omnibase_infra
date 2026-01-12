# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Retry State Model for handler operations.

This module provides a Pydantic model for encapsulating retry state during
handler operation execution with exponential backoff.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field


class ModelRetryState(BaseModel):
    """Encapsulates retry state for handler operations.

    This model tracks the runtime state during retry loops, providing
    a strongly-typed alternative to scattered local variables.

    Attributes:
        attempt: Current retry attempt number (0-based, starts at 0)
        max_attempts: Maximum number of retry attempts allowed
        delay_seconds: Current delay before next retry (calculated with backoff)
        backoff_multiplier: Exponential backoff multiplier for delay calculation
        last_error: Description of the last error encountered (sanitized)
        last_attempt_at: Unix timestamp of the last attempt (None if not yet attempted)

    Example:
        >>> state = ModelRetryState(max_attempts=3)
        >>> state.is_retriable()
        True
        >>> state = state.next_attempt("Connection failed")
        >>> state.attempt
        1

    Note:
        The last_error field should contain SANITIZED error descriptions only.
        Never include credentials, tokens, or other sensitive data in this field.
        See CLAUDE.md "Error Sanitization Guidelines" for the security policy.

    Design Rationale:
        This model stores error messages as strings rather than exception objects.
        This trade-off ensures: (1) immutability (frozen=True), (2) serialization
        safety for logging/persistence, and (3) model simplicity. The original
        exception should be handled by the caller if traceback preservation is
        needed (e.g., wrap in try/except and log before calling next_attempt()).
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    attempt: int = Field(
        default=0,
        ge=0,
        description="Current retry attempt number (0-based)",
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Maximum number of retry attempts allowed",
    )
    delay_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Current delay in seconds before next retry",
    )
    backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Exponential backoff multiplier for delay calculation",
    )
    last_error: str | None = Field(
        default=None,
        description="Sanitized description of the last error encountered",
    )
    last_attempt_at: float | None = Field(
        default=None,
        description="Unix timestamp of the last attempt",
    )

    def is_retriable(self) -> bool:
        """Check if more retry attempts are allowed.

        Returns:
            True if current attempt is less than max_attempts, False otherwise.
        """
        return self.attempt < self.max_attempts

    def is_final_attempt(self) -> bool:
        """Check if this is the final retry attempt.

        Returns:
            True if this is the last allowed attempt, False otherwise.
        """
        return self.attempt >= self.max_attempts - 1

    def next_attempt(
        self,
        error_message: str | None = None,
        timestamp: float | None = None,
        max_delay_seconds: float = 300.0,
    ) -> ModelRetryState:
        """Create a new state for the next retry attempt.

        Args:
            error_message: Sanitized description of the error (optional)
            timestamp: Unix timestamp of this attempt (optional, for testing)
            max_delay_seconds: Maximum delay cap for backoff calculation

        Returns:
            New ModelRetryState with incremented attempt and calculated delay.

        Example:
            >>> state = ModelRetryState(delay_seconds=1.0, backoff_multiplier=2.0)
            >>> next_state = state.next_attempt("Connection timeout")
            >>> next_state.attempt
            1
            >>> next_state.delay_seconds  # 1.0 * 2.0 = 2.0
            2.0
        """
        new_delay = min(
            self.delay_seconds * self.backoff_multiplier,
            max_delay_seconds,
        )

        return ModelRetryState(
            attempt=self.attempt + 1,
            max_attempts=self.max_attempts,
            delay_seconds=new_delay,
            backoff_multiplier=self.backoff_multiplier,
            last_error=error_message,
            last_attempt_at=timestamp if timestamp is not None else time.time(),
        )

    def with_initial_delay(self, initial_delay: float) -> ModelRetryState:
        """Create a copy with a new initial delay value.

        Args:
            initial_delay: New delay value in seconds

        Returns:
            New ModelRetryState with updated delay_seconds.
        """
        return ModelRetryState(
            attempt=self.attempt,
            max_attempts=self.max_attempts,
            delay_seconds=initial_delay,
            backoff_multiplier=self.backoff_multiplier,
            last_error=self.last_error,
            last_attempt_at=self.last_attempt_at,
        )


__all__: list[str] = ["ModelRetryState"]
