"""
ONEX Notification Retry Policy Model

Shared model for webhook retry configuration in the ONEX infrastructure.
Defines how failed notification attempts should be retried.
"""

from typing import List
from pydantic import BaseModel, Field, field_validator
from omnibase_core.enums.enum_backoff_strategy import EnumBackoffStrategy


class ModelNotificationRetryPolicy(BaseModel):
    """
    Retry policy configuration for webhook notifications.

    This model defines how failed notification deliveries should be retried,
    including backoff strategies and which errors are retryable.

    Attributes:
        max_attempts: Maximum number of delivery attempts (1-10)
        backoff_strategy: Strategy for calculating delay between retries
        delay_seconds: Initial delay between retries in seconds
        retryable_status_codes: HTTP status codes that should trigger a retry
    """

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of delivery attempts"
    )

    backoff_strategy: EnumBackoffStrategy = Field(
        default=EnumBackoffStrategy.EXPONENTIAL,
        description="Strategy for calculating delay between retries"
    )

    delay_seconds: float = Field(
        default=5.0,
        ge=1.0,
        description="Initial delay between retries in seconds"
    )

    retryable_status_codes: List[int] = Field(
        default_factory=lambda: [408, 429, 500, 502, 503, 504],
        description="HTTP status codes that should trigger a retry"
    )

    class Config:
        """Pydantic configuration."""
        frozen = True
        extra = "forbid"
        use_enum_values = True

    @field_validator("retryable_status_codes")
    @classmethod
    def validate_status_codes(cls, v):
        """Validate that status codes are in valid HTTP range."""
        if not v:
            return v

        for code in v:
            if not isinstance(code, int) or code < 400 or code > 599:
                raise ValueError(f"Invalid HTTP status code: {code}. Must be between 400-599")

        return v

    def calculate_delay(self, attempt_number: int) -> float:
        """
        Calculate the delay before a retry attempt.

        Args:
            attempt_number: The attempt number (1-based)

        Returns:
            float: Delay in seconds before the next attempt
        """
        if attempt_number <= 1:
            return self.delay_seconds

        if self.backoff_strategy == EnumBackoffStrategy.EXPONENTIAL:
            return self.delay_seconds * (2 ** (attempt_number - 1))

        elif self.backoff_strategy == EnumBackoffStrategy.LINEAR:
            return self.delay_seconds * attempt_number

        elif self.backoff_strategy == EnumBackoffStrategy.FIXED:
            return self.delay_seconds

        else:
            # Default to exponential if unknown strategy
            return self.delay_seconds * (2 ** (attempt_number - 1))

    def should_retry(self, status_code: int, attempt_number: int) -> bool:
        """
        Determine if a failed attempt should be retried.

        Args:
            status_code: HTTP status code from the failed attempt
            attempt_number: The current attempt number (1-based)

        Returns:
            bool: True if the attempt should be retried
        """
        # Don't retry if we've exceeded max attempts
        if attempt_number >= self.max_attempts:
            return False

        # Retry if status code is in the retryable list
        return status_code in self.retryable_status_codes

    @property
    def is_exponential_backoff(self) -> bool:
        """Check if using exponential backoff strategy."""
        return self.backoff_strategy == EnumBackoffStrategy.EXPONENTIAL

    @property
    def is_linear_backoff(self) -> bool:
        """Check if using linear backoff strategy."""
        return self.backoff_strategy == EnumBackoffStrategy.LINEAR

    @property
    def is_fixed_backoff(self) -> bool:
        """Check if using fixed backoff strategy."""
        return self.backoff_strategy == EnumBackoffStrategy.FIXED