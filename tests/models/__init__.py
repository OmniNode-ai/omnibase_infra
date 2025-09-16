"""
Test models package.

This package contains strongly-typed Pydantic models for test fixtures,
eliminating the need for Dict[str, Any] in test files.
"""

from .test_webhook_models import (
    DiscordWebhookPayloadModel,
    GenericWebhookPayloadModel,
    IntegrationTestRequestModel,
    MockWebhookFailureConfigModel,
    MockWebhookRequestModel,
    MockWebhookResponseConfigModel,
    SlackWebhookPayloadModel,
)

__all__ = [
    "DiscordWebhookPayloadModel",
    "GenericWebhookPayloadModel",
    "IntegrationTestRequestModel",
    "MockWebhookFailureConfigModel",
    "MockWebhookRequestModel",
    "MockWebhookResponseConfigModel",
    "SlackWebhookPayloadModel",
]
