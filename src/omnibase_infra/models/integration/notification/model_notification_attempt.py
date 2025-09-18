"""
ONEX Notification Attempt Model

Shared model for tracking individual webhook delivery attempts in the ONEX infrastructure.
Records the outcome and timing of each notification attempt.
"""

import time

from pydantic import BaseModel, Field


class ModelNotificationAttempt(BaseModel):
    """
    Record of a single notification delivery attempt.

    This model captures the details of each attempt to deliver a webhook
    notification, including timing, status, and error information.

    Attributes:
        attempt_number: The attempt number (1-based)
        timestamp: Unix timestamp when the attempt was made
        status_code: HTTP status code received (null if network error)
        error: Error message if the attempt failed
        execution_time_ms: Time taken for this attempt in milliseconds
    """

    attempt_number: int = Field(
        ...,
        ge=1,
        description="Attempt number (1-based)",
    )

    timestamp: float = Field(
        ...,
        description="Unix timestamp when the attempt was made",
    )

    status_code: int | None = Field(
        default=None,
        description="HTTP status code received (null if network error)",
    )

    error: str | None = Field(
        default=None,
        description="Error message if the attempt failed",
    )

    execution_time_ms: float = Field(
        ...,
        ge=0,
        description="Time taken for this attempt in milliseconds",
    )

    class Config:
        """Pydantic configuration."""

        frozen = True
        extra = "forbid"

    @classmethod
    def create_now(
        cls,
        attempt_number: int,
        execution_time_ms: float,
        status_code: int | None = None,
        error: str | None = None,
    ) -> "ModelNotificationAttempt":
        """
        Create a new attempt record with the current timestamp.

        Args:
            attempt_number: The attempt number (1-based)
            execution_time_ms: Time taken for this attempt in milliseconds
            status_code: HTTP status code received (null if network error)
            error: Error message if the attempt failed

        Returns:
            ModelNotificationAttempt: New attempt record
        """
        return cls(
            attempt_number=attempt_number,
            timestamp=time.time(),
            status_code=status_code,
            error=error,
            execution_time_ms=execution_time_ms,
        )

    @property
    def was_successful(self) -> bool:
        """Check if this attempt was successful (2xx status code)."""
        return (
            self.status_code is not None
            and 200 <= self.status_code < 300
            and self.error is None
        )

    @property
    def was_client_error(self) -> bool:
        """Check if this attempt failed due to client error (4xx status code)."""
        return self.status_code is not None and 400 <= self.status_code < 500

    @property
    def was_server_error(self) -> bool:
        """Check if this attempt failed due to server error (5xx status code)."""
        return self.status_code is not None and 500 <= self.status_code < 600

    @property
    def was_network_error(self) -> bool:
        """Check if this attempt failed due to network error (no status code)."""
        return self.status_code is None and self.error is not None

    @property
    def execution_time_seconds(self) -> float:
        """Get execution time in seconds."""
        return self.execution_time_ms / 1000.0
