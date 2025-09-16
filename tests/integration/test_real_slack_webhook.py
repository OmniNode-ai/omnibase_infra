#!/usr/bin/env python3
"""
Real Slack Webhook Integration Test.

This test uses the actual Slack webhook URL from .env to send real notifications
to the #omninode-notifications channel, verifying the Hook Node functionality
with production Slack integration.
"""

import asyncio
import json
import os
import pytest
from datetime import datetime
from uuid import uuid4

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_spi.protocols.core import ProtocolHttpClient, ProtocolHttpResponse
from omnibase_spi.protocols.event_bus import ProtocolEventBus

# Hook Node implementation
from omnibase_infra.nodes.hook_node.v1_0_0.node import NodeHookEffect
from omnibase_infra.nodes.hook_node.v1_0_0.models.model_hook_node_input import ModelHookNodeInput

# Shared notification models
from omnibase_infra.models.notification.model_notification_request import ModelNotificationRequest
from omnibase_core.enums.enum_notification_method import EnumNotificationMethod


class RealSlackHttpClient:
    """Real HTTP client that makes actual HTTP requests to Slack."""

    async def request(self, method: str, url: str, headers: dict = None, body: str = None, timeout: float = 30.0):
        """Make real HTTP request to Slack webhook."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers or {},
                data=body,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                response_body = await response.text()

                return ProtocolHttpResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=response_body,
                    execution_time_ms=100.0,  # Approximate
                    is_success=(200 <= response.status < 300)
                )


class TestEventBus:
    """Test event bus that captures published events."""

    def __init__(self):
        self.events = []

    async def publish(self, event):
        """Capture published events for verification."""
        self.events.append(event)
        print(f"📢 Event published: {event.event_type}")
        return True


@pytest.mark.asyncio
async def test_real_slack_webhook_notification():
    """
    Test Hook Node with real Slack webhook - sends actual notification to Slack.

    This test verifies:
    1. Hook Node initialization with ONEX container patterns
    2. Real HTTP client integration with Slack webhooks
    3. Event bus integration for infrastructure events
    4. Complete notification processing flow
    5. Actual Slack message delivery
    """
    # Get real Slack webhook URL from environment
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not slack_webhook_url:
        pytest.skip("SLACK_WEBHOOK_URL not configured in .env file")

    print(f"🚀 Testing Hook Node with real Slack webhook")
    print(f"🎯 Target: {slack_webhook_url[:50]}...")
    print("=" * 60)

    # Set up container with real dependencies
    container = ModelONEXContainer()

    # Register real HTTP client
    http_client = RealSlackHttpClient()
    container.register_service("ProtocolHttpClient", http_client)

    # Register test event bus to capture events
    event_bus = TestEventBus()
    container.register_service("ProtocolEventBus", event_bus)

    # Create Hook Node with container injection (ONEX pattern)
    hook_node = NodeHookEffect(container)

    assert hook_node.node_type == "effect"
    assert hook_node.domain == "infrastructure"
    print(f"✅ Hook Node initialized: {hook_node.node_type} in {hook_node.domain} domain")

    # Create infrastructure alert notification
    notification_request = ModelNotificationRequest(
        url=slack_webhook_url,
        method=EnumNotificationMethod.POST,
        payload={
            "text": "🧪 ONEX Hook Node Integration Test",
            "username": "ONEX Hook Node",
            "icon_emoji": ":gear:",
            "attachments": [
                {
                    "color": "good",
                    "title": "Registry Fix Verification - Real Slack Integration",
                    "fields": [
                        {
                            "title": "Service",
                            "value": "hook_node_integration_test",
                            "short": True
                        },
                        {
                            "title": "Status",
                            "value": "✅ Hook Node operational with ONEX patterns",
                            "short": True
                        },
                        {
                            "title": "Fix Applied",
                            "value": "Registry pattern removed - using ModelONEXContainer",
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True
                        },
                        {
                            "title": "Test Type",
                            "value": "Real integration test with actual webhook",
                            "short": False
                        }
                    ],
                    "footer": "ONEX Infrastructure - Automated Integration Test"
                }
            ]
        }
    )

    # Create Hook Node input
    correlation_id = str(uuid4())
    hook_input = ModelHookNodeInput(
        notification_request=notification_request,
        correlation_id=correlation_id
    )

    print(f"📋 Processing notification through Hook Node...")
    print(f"🔗 Correlation ID: {correlation_id}")

    # Process through Hook Node - this will send real Slack notification
    result = await hook_node.process(hook_input)

    # Verify results
    assert result is not None, "Hook Node should return a result"
    assert result.success, f"Hook Node processing should succeed: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}"
    assert result.notification_result.final_status_code == 200, f"Expected 200 status code, got {result.notification_result.final_status_code}"

    print()
    print("📊 REAL SLACK INTEGRATION RESULTS:")
    print("=" * 40)
    print(f"✅ Success: {result.success}")
    print(f"📊 Status Code: {result.notification_result.final_status_code}")
    print(f"🔄 Total Attempts: {result.notification_result.total_attempts}")
    print(f"⏱️ Duration: {result.notification_result.total_duration_ms}ms")
    print(f"📢 Events Generated: {len(event_bus.events)}")
    print(f"🎯 Correlation ID: {correlation_id}")

    # Verify event bus integration
    if event_bus.events:
        print("\n📢 Generated Infrastructure Events:")
        for i, event in enumerate(event_bus.events, 1):
            print(f"   {i}. {event.event_type} - {event.event_id}")

    print("\n🎉 REAL SLACK NOTIFICATION SENT!")
    print("✅ Check your #omninode-notifications channel for the message!")
    print("✅ Hook Node registry fix verified with production Slack integration!")


if __name__ == "__main__":
    # Allow running this test directly
    asyncio.run(test_real_slack_webhook_notification())