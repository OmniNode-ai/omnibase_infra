#!/usr/bin/env python3
"""
Real Hook Node Test with Actual Slack Webhook

Tests the actual Hook Node implementation with your production Slack webhook.
This is how infrastructure services will actually use the Hook Node.
"""

import asyncio
import os
from datetime import datetime
from uuid import uuid4


async def test_real_hook_node(slack_webhook_url: str):
    """Test the actual Hook Node with real Slack webhook."""

    print("🚀 Testing Real Hook Node")
    print(f"🎯 Target: {slack_webhook_url[:50]}...")
    print("=" * 60)

    try:
        # Set up container with real HTTP client
        from omnibase_core.core.onex_container import ModelONEXContainer

        container = ModelONEXContainer()

        # We need real protocol implementations
        # For now, let's use a simple HTTP client mock that actually makes requests
        import aiohttp

        class RealHttpClient:
            async def post(
                self,
                url: str,
                headers: dict = None,
                body: str = None,
                timeout: float = 30.0,
            ):
                from omnibase_spi.protocols.core import ProtocolHttpResponse

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, headers=headers or {}, data=body, timeout=timeout,
                    ) as response:
                        response_body = await response.text()
                        return ProtocolHttpResponse(
                            status_code=response.status,
                            headers=dict(response.headers),
                            body=response_body,
                            execution_time_ms=100.0,
                            is_success=(200 <= response.status < 300),
                        )

        class SimpleEventBus:
            def __init__(self):
                self.events = []

            async def publish(self, event):
                self.events.append(event)
                print(f"📢 Event published: {event.event_type}")
                return True

        # Register services
        container.register_service("ProtocolHttpClient", RealHttpClient())
        container.register_service("ProtocolEventBus", SimpleEventBus())

        # Create Hook Node
        from omnibase_infra.nodes.hook_node.v1_0_0.node import NodeHookEffect

        hook_node = NodeHookEffect(container)

        # Create real notification request
        from omnibase_core.enums.enum_notification_method import EnumNotificationMethod

        from omnibase_infra.models.notification.model_notification_request import (
            ModelNotificationRequest,
        )

        # Create infrastructure alert message
        notification_request = ModelNotificationRequest(
            url=slack_webhook_url,
            method=EnumNotificationMethod.POST,
            payload={
                "text": "🧪 ONEX Infrastructure Test Alert",
                "username": "ONEX Hook Node",
                "icon_emoji": ":gear:",
                "attachments": [
                    {
                        "color": "good",
                        "title": "Hook Node Integration Test",
                        "fields": [
                            {
                                "title": "Service",
                                "value": "hook_node_test",
                                "short": True,
                            },
                            {
                                "title": "Status",
                                "value": "✅ Hook Node is operational",
                                "short": True,
                            },
                            {
                                "title": "Timestamp",
                                "value": datetime.utcnow().isoformat(),
                                "short": True,
                            },
                        ],
                        "footer": "ONEX Infrastructure",
                    },
                ],
            },
        )

        # Create Hook Node input
        from omnibase_infra.nodes.hook_node.v1_0_0.models.model_hook_node_input import (
            ModelHookNodeInput,
        )

        hook_input = ModelHookNodeInput(
            notification_request=notification_request,
            correlation_id=str(uuid4()),
        )

        print("📤 Processing notification through Hook Node...")

        # Process through Hook Node
        result = await hook_node.process(hook_input)

        if result.success:
            print("✅ Hook Node processed successfully!")
            print(f"📊 Status Code: {result.notification_result.final_status_code}")
            print(f"🔄 Attempts: {result.notification_result.total_attempts}")
            print(f"⏳ Duration: {result.notification_result.total_duration_ms}ms")
            print("🎉 Check your Slack channel for the alert message!")

            # Show any events that were published
            event_bus = container.get_service("ProtocolEventBus")
            if event_bus.events:
                print(f"📢 Circuit breaker events: {len(event_bus.events)}")

        else:
            print("❌ Hook Node processing failed!")
            print(f"📊 Status Code: {result.notification_result.final_status_code}")
            print(f"❌ Error: {result.error_message}")

        return result.success

    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test runner."""

    # Get Slack webhook URL from environment variable for security
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

    if not SLACK_WEBHOOK_URL:
        print("❌ ERROR: SLACK_WEBHOOK_URL environment variable not set")
        print(
            "🔧 Set it with: export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...your-url...'",
        )
        print(
            "   Get your webhook URL from: Slack App → Incoming Webhooks → Copy webhook URL",
        )
        return False

    try:
        import aiohttp
    except ImportError:
        print("❌ Missing aiohttp dependency")
        print("   Install with: pip install aiohttp")
        return False

    success = await test_real_hook_node(SLACK_WEBHOOK_URL)

    if success:
        print("\n🎉 SUCCESS! Your Hook Node is working with Slack!")
        print("✅ Infrastructure alerts will now flow to your Slack channel")
    else:
        print("\n❌ Test failed - check the output above")

    return success


if __name__ == "__main__":
    asyncio.run(main())
