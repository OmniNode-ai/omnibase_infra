"""
Test models package.

This package contains strongly-typed Pydantic models for test fixtures,
eliminating the need for Dict[str, Any] in test files.
"""

from .test_webhook_models import (
    MockWebhookRequestModel,
    MockWebhookResponseConfigModel,
    MockWebhookFailureConfigModel,
    SlackWebhookPayloadModel,
    DiscordWebhookPayloadModel,
    GenericWebhookPayloadModel,
    IntegrationTestRequestModel,
)

__all__ = [
    "MockWebhookRequestModel",
    "MockWebhookResponseConfigModel",
    "MockWebhookFailureConfigModel",
    "SlackWebhookPayloadModel",
    "DiscordWebhookPayloadModel",
    "GenericWebhookPayloadModel",
    "IntegrationTestRequestModel",
]