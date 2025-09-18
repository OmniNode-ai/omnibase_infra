"""
Hook Node Input Model

Node-specific input model for the Hook Node EFFECT adapter.
Wraps notification requests for message bus integration.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.notification.model_notification_request import (
    ModelNotificationRequest,
)


class ModelHookNodeInput(BaseModel):
    """
    Input envelope for the Hook Node.

    This model wraps notification requests in the standardized
    message bus envelope format for EFFECT nodes.

    Attributes:
        notification_request: The notification request payload
        correlation_id: Request correlation ID for tracing
        timestamp: Unix timestamp of the request
        context: Additional request context
    """

    notification_request: ModelNotificationRequest = Field(
        ...,
        description="Notification request payload to process",
    )

    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID for distributed tracing",
    )

    timestamp: float = Field(
        ...,
        description="Unix timestamp when the request was created",
    )

    context: (
        dict[
            str,
            str
            | int
            | float
            | bool
            | list[str | int | float | bool]
            | dict[str, str | int | float | bool],
        ]
        | None
    ) = Field(
        default=None,
        description="Additional request context and metadata with strongly typed values",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    @property
    def has_context(self) -> bool:
        """Check if this input has additional context."""
        return self.context is not None and len(self.context) > 0

    @property
    def target_url(self) -> str:
        """Get the target URL from the notification request."""
        return str(self.notification_request.url)

    @property
    def http_method(self) -> str:
        """Get the HTTP method from the notification request."""
        return self.notification_request.method.value
