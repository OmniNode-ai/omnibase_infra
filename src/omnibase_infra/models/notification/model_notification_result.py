"""
ONEX Notification Result Model

Shared model for the final result of webhook delivery attempts in the ONEX infrastructure.
Aggregates all attempts and provides the final delivery status.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from omnibase_infra.models.notification.model_notification_attempt import ModelNotificationAttempt


class ModelNotificationResult(BaseModel):
    """
    Final result of webhook notification delivery attempts.

    This model aggregates all delivery attempts and provides the final
    status of the notification delivery process.

    Attributes:
        final_status_code: The HTTP status code from the final attempt
        is_success: Whether the notification was ultimately successful
        attempts: List of all delivery attempts made
        total_attempts: Total number of attempts made
    """

    final_status_code: Optional[int] = Field(
        default=None,
        description="Final HTTP status code received (null if all attempts failed with network errors)"
    )

    is_success: bool = Field(
        ...,
        description="Whether the notification was ultimately successful"
    )

    attempts: List[ModelNotificationAttempt] = Field(
        ...,
        description="List of all delivery attempts made"
    )

    total_attempts: int = Field(
        ...,
        ge=0,
        description="Total number of attempts made"
    )

    class Config:
        """Pydantic configuration."""
        frozen = True
        extra = "forbid"

    @validator("total_attempts")
    def validate_total_attempts(cls, v, values):
        """Validate that total_attempts matches the length of attempts list."""
        attempts = values.get("attempts", [])
        if v != len(attempts):
            raise ValueError(f"total_attempts ({v}) must match the number of attempts ({len(attempts)})")
        return v

    @validator("is_success")
    def validate_success_against_attempts(cls, v, values):
        """Validate that success status is consistent with attempts."""
        attempts = values.get("attempts", [])
        if attempts:
            last_attempt = attempts[-1]
            actual_success = last_attempt.was_successful
            if v != actual_success:
                raise ValueError(f"is_success ({v}) must match the result of the last attempt ({actual_success})")
        return v

    @classmethod
    def from_attempts(cls, attempts: List[ModelNotificationAttempt]) -> "ModelNotificationResult":
        """
        Create a result from a list of attempts.

        Args:
            attempts: List of notification attempts

        Returns:
            ModelNotificationResult: Result summarizing all attempts
        """
        if not attempts:
            return cls(
                final_status_code=None,
                is_success=False,
                attempts=[],
                total_attempts=0
            )

        last_attempt = attempts[-1]
        final_status_code = last_attempt.status_code
        is_success = last_attempt.was_successful

        return cls(
            final_status_code=final_status_code,
            is_success=is_success,
            attempts=attempts,
            total_attempts=len(attempts)
        )

    @property
    def total_execution_time_ms(self) -> float:
        """Calculate total execution time across all attempts."""
        return sum(attempt.execution_time_ms for attempt in self.attempts)

    @property
    def total_execution_time_seconds(self) -> float:
        """Get total execution time in seconds."""
        return self.total_execution_time_ms / 1000.0

    @property
    def had_retries(self) -> bool:
        """Check if there were retry attempts."""
        return self.total_attempts > 1

    @property
    def successful_attempt_number(self) -> Optional[int]:
        """Get the attempt number that succeeded (if any)."""
        for attempt in self.attempts:
            if attempt.was_successful:
                return attempt.attempt_number
        return None

    @property
    def failure_summary(self) -> str:
        """Get a summary of failures (if notification failed)."""
        if self.is_success:
            return "Notification delivered successfully"

        if not self.attempts:
            return "No delivery attempts made"

        last_attempt = self.attempts[-1]
        if last_attempt.was_network_error:
            return f"Network error after {self.total_attempts} attempts: {last_attempt.error}"
        elif last_attempt.status_code:
            return f"HTTP {last_attempt.status_code} after {self.total_attempts} attempts"
        else:
            return f"Unknown error after {self.total_attempts} attempts"