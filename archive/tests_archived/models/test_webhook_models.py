"""
Test-specific models for webhook testing.

These models provide strong typing for test fixtures and webhook server responses,
eliminating the need for Dict[str, Any] in test files.
"""

from pydantic import BaseModel, Field


class MockWebhookRequestModel(BaseModel):
    """Model for mock webhook server request capture."""

    url: str = Field(description="Request URL")
    method: str = Field(description="HTTP method")
    headers: dict[str, str] = Field(default_factory=dict, description="Request headers")
    body: str = Field(description="Request body as string")
    timestamp: float = Field(description="Request timestamp")


class MockWebhookResponseConfigModel(BaseModel):
    """Configuration model for mock webhook server responses."""

    status_code: int = Field(default=200, description="HTTP status code to return")
    body: str = Field(default='{"status": "ok"}', description="Response body")
    headers: dict[str, str] = Field(
        default_factory=lambda: {"Content-Type": "application/json"},
        description="Response headers",
    )
    delay_ms: int = Field(default=100, description="Response delay in milliseconds")


class MockWebhookFailureConfigModel(BaseModel):
    """Configuration model for mock webhook server failure scenarios."""

    status_code: int = Field(description="HTTP error status code")
    body: str = Field(
        default='{"error": "server error"}', description="Error response body",
    )
    headers: dict[str, str] = Field(
        default_factory=lambda: {"Content-Type": "application/json"},
        description="Error response headers",
    )
    delay_ms: int = Field(
        default=100, description="Error response delay in milliseconds",
    )
    fail_count: int = Field(
        default=1, description="Number of requests to fail before succeeding",
    )


class SlackWebhookPayloadModel(BaseModel):
    """Model for Slack webhook payload structure."""

    text: str = Field(description="Message text")
    username: str | None = Field(default=None, description="Bot username")
    icon_emoji: str | None = Field(default=None, description="Bot emoji icon")
    channel: str | None = Field(default=None, description="Target channel")
    attachments: list[dict[str, str]] | None = Field(
        default=None, description="Message attachments",
    )


class DiscordWebhookPayloadModel(BaseModel):
    """Model for Discord webhook payload structure."""

    content: str = Field(description="Message content")
    username: str | None = Field(default=None, description="Bot username")
    avatar_url: str | None = Field(default=None, description="Bot avatar URL")
    embeds: list[dict[str, str]] | None = Field(
        default=None, description="Message embeds",
    )


class GenericWebhookPayloadModel(BaseModel):
    """Model for generic webhook payload structure."""

    event_type: str = Field(description="Event type identifier")
    data: dict[str, str] = Field(description="Event data payload")
    timestamp: str = Field(description="Event timestamp")
    source: str = Field(description="Event source identifier")


class IntegrationTestRequestModel(BaseModel):
    """Model for integration test request tracking."""

    url: str = Field(description="Request URL")
    method: str = Field(description="HTTP method")
    headers: dict[str, str] = Field(description="Request headers")
    payload: dict[str, str] = Field(description="Request payload")
    timestamp: float = Field(description="Request timestamp")
    correlation_id: str = Field(description="Request correlation ID")
