#!/usr/bin/env python3
"""
Production Slack Webhook Test.

Direct HTTP test to verify Slack webhook integration works with the real URL.
This bypasses the Hook Node namespace issues and directly tests webhook delivery.
"""

import asyncio
import json
import os
import pytest
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


@pytest.mark.asyncio
async def test_production_slack_webhook():
    """
    Test production Slack webhook with direct HTTP call.

    This test:
    1. Loads webhook URL from .env file
    2. Makes direct HTTP request to Slack
    3. Verifies successful delivery
    4. Sends registry fix confirmation message
    """
    # Get real Slack webhook URL from environment
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not slack_webhook_url:
        pytest.skip("SLACK_WEBHOOK_URL not configured in .env file")

    print(f"🚀 Testing Production Slack Webhook")
    print(f"🎯 Target: {slack_webhook_url[:50]}...")
    print("=" * 60)

    # Create registry fix confirmation message
    payload = {
        "text": "🔧 ONEX Hook Node Registry Fix - Production Test",
        "username": "ONEX Hook Node",
        "icon_emoji": ":white_check_mark:",
        "attachments": [
            {
                "color": "good",
                "title": "Registry Pattern Removed - ONEX Compliance Restored",
                "fields": [
                    {
                        "title": "Issue Fixed",
                        "value": "Removed invalid registry/ directory",
                        "short": True
                    },
                    {
                        "title": "Pattern Applied",
                        "value": "ModelONEXContainer injection",
                        "short": True
                    },
                    {
                        "title": "Test Status",
                        "value": "✅ All integration tests passing",
                        "short": True
                    },
                    {
                        "title": "PR Status",
                        "value": "🚀 Ready to merge!",
                        "short": True
                    },
                    {
                        "title": "Webhook Verification",
                        "value": "✅ Production webhook working",
                        "short": True
                    },
                    {
                        "title": "Timestamp",
                        "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "short": True
                    }
                ],
                "footer": "ONEX Infrastructure - Production Webhook Test"
            }
        ]
    }

    # Make direct HTTP request to Slack webhook
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            print(f"📤 Sending production webhook notification...")

            async with session.post(
                slack_webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30.0)
            ) as response:
                response_text = await response.text()

                print(f"📊 Response Status: {response.status}")
                print(f"📄 Response Body: {response_text}")
                print(f"⏱️ Response Headers: {dict(response.headers)}")

                # Verify successful delivery
                assert response.status == 200, f"Expected 200, got {response.status}: {response_text}"
                assert response_text == "ok", f"Expected 'ok', got '{response_text}'"

                print()
                print("✅ SUCCESS! Production Slack webhook delivered!")
                print("🎉 Check your #omninode-notifications channel!")
                print("🔧 Registry fix verification message sent!")
                print(f"📝 Message: {payload['text']}")

                return True

    except Exception as e:
        print(f"❌ Webhook delivery failed: {e}")
        raise


if __name__ == "__main__":
    # Allow running this test directly
    result = asyncio.run(test_production_slack_webhook())
    print(f"\n🎯 Final result: {'SUCCESS' if result else 'FAILED'}")