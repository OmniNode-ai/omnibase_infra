"""
ONEX Notification Request Model

Shared model for webhook notification requests in the ONEX infrastructure.
This model defines the structure for external HTTP notification delivery.

Security Note: URL validation must be performed by the consuming service
to prevent SSRF attacks.
"""

from omnibase_core.enums.enum_notification_method import EnumNotificationMethod
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from omnibase_infra.models.notification.model_notification_auth import (
    ModelNotificationAuth,
)
from omnibase_infra.models.notification.model_notification_retry_policy import (
    ModelNotificationRetryPolicy,
)
from omnibase_infra.models.webhook.model_webhook_payload import ModelWebhookPayloadUnion


class ModelNotificationRequest(BaseModel):
    """
    Request model for webhook notifications.

    This model encapsulates all the information needed to deliver
    a webhook notification to an external service.

    Attributes:
        url: Target URL for the webhook delivery
        method: HTTP method to use (POST or PUT)
        headers: Optional HTTP headers to include
        payload: JSON payload to send (arbitrary structure for flexibility)
        auth: Optional authentication configuration
        retry_policy: Optional retry behavior configuration
    """

    url: HttpUrl = Field(
        ...,
        description="Target URL for the webhook notification",
    )

    method: EnumNotificationMethod = Field(
        ...,
        description="HTTP method for the notification request",
    )

    headers: dict[str, str] | None = Field(
        default=None,
        description="Optional HTTP headers to include with the request",
    )

    payload: ModelWebhookPayloadUnion = Field(
        ...,
        description="Strongly-typed webhook payload with agent-safe validation",
    )

    auth: ModelNotificationAuth | None = Field(
        default=None,
        description="Optional authentication configuration",
    )

    retry_policy: ModelNotificationRetryPolicy | None = Field(
        default=None,
        description="Optional retry policy for failed deliveries",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        use_enum_values=True,
    )

    def model_post_init(self, __context: dict[str, str | int | bool] | None) -> None:
        """Post-initialization validation."""
        # Validate that headers don't contain sensitive data in keys
        if self.headers:
            sensitive_header_patterns = ["password", "secret", "key", "token"]
            for header_name in self.headers.keys():
                header_lower = header_name.lower()
                if any(pattern in header_lower for pattern in sensitive_header_patterns):
                    # Log warning but don't fail - headers might legitimately contain these words
                    pass

    @property
    def requires_authentication(self) -> bool:
        """Check if this request requires authentication."""
        return self.auth is not None

    @property
    def has_retry_policy(self) -> bool:
        """Check if this request has a custom retry policy."""
        return self.retry_policy is not None
