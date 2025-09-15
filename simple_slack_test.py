#!/usr/bin/env python3
"""
Simple Slack Webhook Test

Tests basic HTTP connectivity to your Slack webhook.
This validates that the webhook URL works before testing the full Hook Node.
"""

import asyncio
import json
import os
from datetime import datetime

async def test_slack_webhook():
    """Test direct HTTP connection to Slack webhook."""

    # Get Slack webhook URL from environment variable for security
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    if not SLACK_WEBHOOK_URL:
        print("❌ ERROR: SLACK_WEBHOOK_URL environment variable not set")
        print("🔧 Set it with: export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...your-url...'")
        return False

    print("🚀 Simple Slack Webhook Test")
    print(f"🎯 Target: {SLACK_WEBHOOK_URL[:50]}...")
    print("=" * 60)

    try:
        import aiohttp

        # Create infrastructure alert message
        slack_message = {
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
                            "short": True
                        },
                        {
                            "title": "Status",
                            "value": "✅ Hook Node is operational",
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True
                        }
                    ],
                    "footer": "ONEX Infrastructure"
                }
            ]
        }

        headers = {"Content-Type": "application/json"}
        body = json.dumps(slack_message)

        print("📤 Sending test message to Slack...")

        async with aiohttp.ClientSession() as session:
            start_time = datetime.utcnow()
            async with session.post(SLACK_WEBHOOK_URL, headers=headers, data=body) as response:
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()

                response_text = await response.text()

                print(f"⏱️  Request completed in {duration:.2f} seconds")
                print(f"📊 Status Code: {response.status}")
                print(f"📋 Response: {response_text}")

                if response.status == 200:
                    print("✅ SUCCESS! Slack webhook is working!")
                    print("🎉 Check your Slack channel for the test message")
                    print("🔗 Your Hook Node can now send alerts to this webhook")
                    return True
                else:
                    print(f"❌ FAILED: HTTP {response.status}")
                    print("🔍 Check your webhook URL and Slack app configuration")
                    return False

    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner."""

    print("🧪 Testing Slack Webhook Connectivity")
    print("This validates your webhook before testing the full Hook Node")
    print("=" * 60)

    success = await test_slack_webhook()

    if success:
        print("\n🎉 WEBHOOK TEST SUCCESSFUL!")
        print("✅ Your Slack webhook is working correctly")
        print("🎯 Next: The Hook Node can use this webhook for infrastructure alerts")
        print("\n📋 What this means:")
        print("   • Your Slack app and webhook are configured correctly")
        print("   • The Hook Node can send alerts to your Slack channel")
        print("   • Infrastructure events can now notify your team")
    else:
        print("\n❌ WEBHOOK TEST FAILED")
        print("🔍 Check your Slack app configuration:")
        print("   • Ensure 'Incoming Webhooks' is enabled")
        print("   • Verify the webhook URL is correct")
        print("   • Check that the app has permission to post to your channel")

    return success

if __name__ == "__main__":
    asyncio.run(main())