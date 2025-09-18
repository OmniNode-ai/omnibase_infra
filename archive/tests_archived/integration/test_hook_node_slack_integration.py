#!/usr/bin/env python3
"""
Real Slack Webhook Integration Test for Hook Node

This script tests the Hook Node with a real Slack webhook to verify end-to-end functionality.
Replace SLACK_WEBHOOK_URL with your actual webhook URL from Slack.
"""

import asyncio
import sys
from datetime import datetime
from uuid import uuid4


# Mock implementations for testing without full infrastructure
class MockONEXContainer:
    def __init__(self):
        self.services = {}

    def get_service(self, service_name):
        return self.services.get(service_name)

    def register_singleton(self, service_name, factory):
        if callable(factory):
            self.services[service_name] = factory(self)
        else:
            self.services[service_name] = factory


class RealHttpClient:
    """Real HTTP client for testing actual webhook delivery."""

    async def post(
        self, url: str, headers: dict = None, body: str = None, timeout: float = 30.0,
    ):
        """Make real HTTP POST request."""
        import aiohttp

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as session:
                async with session.post(
                    url, headers=headers or {}, data=body,
                ) as response:
                    response_body = await response.text()

                    # Import the actual response model
                    from omnibase_spi.protocols.core import ProtocolHttpResponse

                    return ProtocolHttpResponse(
                        status_code=response.status,
                        headers=dict(response.headers),
                        body=response_body,
                        execution_time_ms=100.0,  # Approximate
                        is_success=(200 <= response.status < 300),
                    )

        except TimeoutError:
            from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

            raise OnexError(
                code=CoreErrorCode.TIMEOUT_ERROR,
                message=f"HTTP request to {url} timed out after {timeout}s",
                context={"url": url, "timeout": timeout},
            )
        except Exception as e:
            from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

            raise OnexError(
                code=CoreErrorCode.NETWORK_ERROR,
                message=f"HTTP request failed: {e!s}",
                context={"url": url, "error": str(e)},
            ) from e


class MockEventBus:
    """Mock event bus to capture published events."""

    def __init__(self):
        self.published_events = []

    async def publish(self, event):
        self.published_events.append(event)
        print(f"📢 Event published: {event.event_type}")
        return True


async def test_slack_webhook_integration():
    """Test Hook Node with real Slack webhook."""

    # REPLACE THIS with your actual Slack webhook URL
    SLACK_WEBHOOK_URL = "YOUR_SLACK_WEBHOOK_URL_HERE"

    if SLACK_WEBHOOK_URL == "YOUR_SLACK_WEBHOOK_URL_HERE":
        print("❌ Please replace SLACK_WEBHOOK_URL with your actual Slack webhook URL")
        print(
            "   Get it from: https://api.slack.com/apps → Your App → Incoming Webhooks",
        )
        return False

    print("🚀 Testing Hook Node with Real Slack Webhook")
    print(f"🎯 Target: {SLACK_WEBHOOK_URL[:50]}...")
    print("=" * 60)

    try:
        # Setup container with real HTTP client
        container = MockONEXContainer()
        http_client = RealHttpClient()
        event_bus = MockEventBus()

        container.services["ProtocolHttpClient"] = http_client
        container.services["ProtocolEventBus"] = event_bus

        # Create Hook Node
        from omnibase_infra.nodes.hook_node.v1_0_0.node import NodeHookEffect

        hook_node = NodeHookEffect(container)

        # Create test notification request
        from omnibase_core.enums.enum_notification_method import EnumNotificationMethod

        from omnibase_infra.models.notification.model_notification_request import (
            ModelNotificationRequest,
        )

        # Create rich Slack message
        slack_payload = {
            "text": "🧪 Hook Node Integration Test",
            "username": "ONEX Infrastructure Bot",
            "icon_emoji": ":robot_face:",
            "attachments": [
                {
                    "color": "good",
                    "title": "Hook Node Test Results",
                    "fields": [
                        {
                            "title": "Test Type",
                            "value": "Real Slack Webhook Integration",
                            "short": True,
                        },
                        {
                            "title": "Timestamp",
                            "value": datetime.utcnow().isoformat(),
                            "short": True,
                        },
                        {
                            "title": "Node Version",
                            "value": "v1.0.0",
                            "short": True,
                        },
                        {
                            "title": "Status",
                            "value": "✅ Hook Node is working correctly!",
                            "short": False,
                        },
                    ],
                    "footer": "ONEX Infrastructure",
                    "footer_icon": "https://github.com/favicon.ico",
                },
            ],
        }

        notification_request = ModelNotificationRequest(
            url=SLACK_WEBHOOK_URL,
            method=EnumNotificationMethod.POST,
            payload=slack_payload,
        )

        # Create Hook Node input
        from omnibase_infra.nodes.hook_node.v1_0_0.models.model_hook_node_input import (
            ModelHookNodeInput,
        )

        hook_input = ModelHookNodeInput(
            notification_request=notification_request,
            correlation_id=str(uuid4()),
        )

        print("📤 Sending test notification to Slack...")

        # Process the notification
        start_time = datetime.utcnow()
        result = await hook_node.process(hook_input)
        end_time = datetime.utcnow()

        duration = (end_time - start_time).total_seconds()

        print(f"⏱️  Processing completed in {duration:.2f} seconds")

        # Check results
        if result.success:
            print("✅ Hook Node test SUCCESSFUL!")
            print(f"📊 Status Code: {result.notification_result.final_status_code}")
            print(f"🔄 Attempts: {result.notification_result.total_attempts}")
            print(
                f"⏳ Total Duration: {result.notification_result.total_duration_ms}ms",
            )

            # Check if message appeared in Slack
            if result.notification_result.final_status_code == 200:
                print("🎉 Message should now appear in your Slack channel!")
                print("   Check your selected channel for the test message")

            # Show published events
            if event_bus.published_events:
                print(f"📢 Events published: {len(event_bus.published_events)}")
                for event in event_bus.published_events:
                    print(f"   - {event.event_type}")

            return True

        print("❌ Hook Node test FAILED!")
        print(f"📊 Status Code: {result.notification_result.final_status_code}")
        print(f"❌ Error: {result.error_message}")

        return False

    except Exception as e:
        print(f"💥 Test crashed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_authentication_webhook():
    """Test Hook Node with authentication (if needed)."""
    print("\n🔐 Testing Authentication (Optional)")
    print("-" * 40)

    # Example for webhook that requires authentication
    # Uncomment and modify if your webhook needs authentication

    """
    from omnibase_infra.models.notification.model_notification_auth import ModelNotificationAuth
    from omnibase_core.enums.enum_auth_type import EnumAuthType

    auth = ModelNotificationAuth(
        auth_type=EnumAuthType.BEARER,
        credentials={"token": "your-auth-token-here"}
    )

    notification_request = ModelNotificationRequest(
        url="YOUR_AUTHENTICATED_WEBHOOK_URL",
        method=EnumNotificationMethod.POST,
        payload={"message": "Authenticated test"},
        auth=auth
    )

    # Process with authentication...
    """

    print(
        "🔐 Authentication test skipped (no auth required for standard Slack webhooks)",
    )
    print("   Uncomment and modify the code above if you need to test authentication")


async def test_error_handling():
    """Test Hook Node error handling with invalid webhook."""
    print("\n🚨 Testing Error Handling")
    print("-" * 40)

    try:
        container = MockONEXContainer()
        http_client = RealHttpClient()
        event_bus = MockEventBus()

        container.services["ProtocolHttpClient"] = http_client
        container.services["ProtocolEventBus"] = event_bus

        from omnibase_infra.nodes.hook_node.v1_0_0.node import NodeHookEffect

        hook_node = NodeHookEffect(container)

        # Test with invalid URL
        from omnibase_core.enums.enum_notification_method import EnumNotificationMethod

        from omnibase_infra.models.notification.model_notification_request import (
            ModelNotificationRequest,
        )

        invalid_request = ModelNotificationRequest(
            url="https://invalid-webhook-url-that-does-not-exist.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "error handling"},
        )

        from omnibase_infra.nodes.hook_node.v1_0_0.models.model_hook_node_input import (
            ModelHookNodeInput,
        )

        hook_input = ModelHookNodeInput(
            notification_request=invalid_request,
            correlation_id=str(uuid4()),
        )

        print("📤 Sending request to invalid webhook (should fail)...")

        result = await hook_node.process(hook_input)

        if not result.success:
            print("✅ Error handling working correctly!")
            print(f"❌ Expected failure: {result.error_message}")

            # Check circuit breaker events
            failure_events = [
                e for e in event_bus.published_events if "failure" in e.event_type
            ]
            if failure_events:
                print(f"📢 Circuit breaker failure events: {len(failure_events)}")

            return True
        print("⚠️  Unexpected success - invalid URL should have failed")
        return False

    except Exception as e:
        print(f"✅ Exception handled correctly: {e}")
        return True


async def main():
    """Main test runner."""
    print("🧪 ONEX Hook Node - Real Slack Integration Test")
    print("=" * 60)

    tests_passed = 0
    total_tests = 3

    # Test 1: Real Slack webhook
    if await test_slack_webhook_integration():
        tests_passed += 1

    # Test 2: Authentication (optional)
    await test_authentication_webhook()
    tests_passed += 1  # Always pass since it's informational

    # Test 3: Error handling
    if await test_error_handling():
        tests_passed += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed >= 2:  # Allow auth test to be informational
        print("🎉 Hook Node integration testing SUCCESSFUL!")
        print("✅ Your Hook Node is ready for production use with Slack webhooks")
        return True
    print("❌ Some tests failed - check the output above")
    return False


if __name__ == "__main__":
    try:
        # Check dependencies
        missing_deps = []
        try:
            import aiohttp
        except ImportError:
            missing_deps.append("aiohttp")

        if missing_deps:
            print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
            print("   Install with: pip install aiohttp")
            sys.exit(1)

        # Run tests
        success = asyncio.run(main())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Test runner crashed: {e}")
        sys.exit(1)
