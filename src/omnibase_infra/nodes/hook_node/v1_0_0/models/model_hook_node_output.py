"""
Hook Node Output Model

Node-specific output model for the Hook Node EFFECT adapter.
Returns notification delivery results in message bus envelope format.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from omnibase_infra.models.notification.model_notification_result import ModelNotificationResult


class ModelHookNodeOutput(BaseModel):
    """
    Output envelope from the Hook Node.

    This model wraps notification delivery results in the standardized
    message bus envelope format for EFFECT nodes.

    Attributes:
        notification_result: The notification delivery result
        success: Whether the notification was successfully delivered
        error_message: Error message if the operation failed completely
        correlation_id: Request correlation ID for tracing
        timestamp: Unix timestamp of the response
        total_execution_time_ms: Total operation execution time
        context: Additional response context
    """

    notification_result: ModelNotificationResult = Field(
        ...,
        description="Notification delivery result with all attempt details"
    )

    success: bool = Field(
        ...,
        description="Whether the final notification attempt was successful (2xx status code)"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if the operation failed completely"
    )

    correlation_id: str = Field(
        ...,
        description="Request correlation ID for distributed tracing"
    )

    timestamp: float = Field(
        ...,
        description="Unix timestamp when the response was created"
    )

    total_execution_time_ms: float = Field(
        ...,
        ge=0,
        description="Total operation execution time in milliseconds"
    )

    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional response context and metadata"
    )

    class Config:
        """Pydantic configuration."""
        frozen = True
        extra = "forbid"

    @classmethod
    def from_result(
        cls,
        notification_result: ModelNotificationResult,
        correlation_id: str,
        timestamp: float,
        total_execution_time_ms: float,
        error_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> "ModelHookNodeOutput":
        """
        Create output from a notification result.

        Args:
            notification_result: The notification delivery result
            correlation_id: Request correlation ID
            timestamp: Response timestamp
            total_execution_time_ms: Total execution time
            error_message: Optional error message
            context: Optional additional context

        Returns:
            ModelHookNodeOutput: Formatted output envelope
        """
        return cls(
            notification_result=notification_result,
            success=notification_result.is_success,
            error_message=error_message,
            correlation_id=correlation_id,
            timestamp=timestamp,
            total_execution_time_ms=total_execution_time_ms,
            context=context
        )

    @property
    def has_error(self) -> bool:
        """Check if this output contains an error."""
        return not self.success or self.error_message is not None

    @property
    def total_execution_time_seconds(self) -> float:
        """Get total execution time in seconds."""
        return self.total_execution_time_ms / 1000.0

    @property
    def final_status_code(self) -> Optional[int]:
        """Get the final HTTP status code from the notification result."""
        return self.notification_result.final_status_code

    @property
    def attempt_count(self) -> int:
        """Get the number of delivery attempts made."""
        return self.notification_result.total_attempts