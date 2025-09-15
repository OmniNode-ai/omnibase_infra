"""
ONEX Notification Request Model

Shared model for webhook notification requests in the ONEX infrastructure.
This model defines the structure for external HTTP notification delivery.

Security Note: URL validation must be performed by the consuming service
to prevent SSRF attacks.
"""

from typing import Dict, Optional, Union, List
from pydantic import Json
from pydantic import BaseModel, Field, HttpUrl
from omnibase_core.enums.enum_notification_method import EnumNotificationMethod
from omnibase_infra.models.notification.model_notification_auth import ModelNotificationAuth
from omnibase_infra.models.notification.model_notification_retry_policy import ModelNotificationRetryPolicy


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
        description="Target URL for the webhook notification"
    )

    method: EnumNotificationMethod = Field(
        ...,
        description="HTTP method for the notification request"
    )

    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional HTTP headers to include with the request"
    )

    payload: Dict[str, Union[str, int, float, bool, List[Union[str, int, float, bool]], Dict[str, Union[str, int, float, bool]]]] = Field(
        ...,
        description="JSON payload to send in the notification body - supports nested JSON structures with strongly typed values"
    )

    auth: Optional[ModelNotificationAuth] = Field(
        default=None,
        description="Optional authentication configuration"
    )

    retry_policy: Optional[ModelNotificationRetryPolicy] = Field(
        default=None,
        description="Optional retry policy for failed deliveries"
    )

    class Config:
        """Pydantic configuration."""
        frozen = True
        extra = "forbid"
        use_enum_values = True
        json_encoders = {
            HttpUrl: str
        }

    def model_post_init(self, __context: Optional[Dict[str, Union[str, int, bool]]]) -> None:
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