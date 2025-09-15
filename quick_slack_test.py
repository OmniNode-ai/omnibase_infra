#!/usr/bin/env python3
"""
Quick Slack Hook Node Test

Simple test to verify Hook Node works with your Slack webhook.
Just replace the webhook URL and run!
"""

import asyncio
import json
from datetime import datetime

async def quick_slack_test():
    """Quick test with minimal setup."""

    # 🔧 REPLACE THIS with your Slack webhook URL
    SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

    if "YOUR/WEBHOOK/URL" in SLACK_WEBHOOK_URL:
        print("❌ Please update SLACK_WEBHOOK_URL with your actual webhook URL")
        return

    print("🚀 Quick Hook Node → Slack Test")
    print("=" * 40)

    try:
        import aiohttp

        # Simple Slack message
        slack_message = {
            "text": f"🧪 Hook Node Test - {datetime.now().strftime('%H:%M:%S')}",
            "username": "ONEX Hook Node",
            "icon_emoji": ":gear:",
            "attachments": [
                {
                    "color": "good",
                    "fields": [
                        {
                            "title": "Status",
                            "value": "Hook Node is working! 🎉",
                            "short": False
                        }
                    ]
                }
            ]
        }

        headers = {"Content-Type": "application/json"}
        body = json.dumps(slack_message)

        print(f"📤 Sending to: {SLACK_WEBHOOK_URL[:50]}...")

        async with aiohttp.ClientSession() as session:
            async with session.post(SLACK_WEBHOOK_URL, headers=headers, data=body) as response:
                print(f"📊 Status Code: {response.status}")
                response_text = await response.text()
                print(f"📋 Response: {response_text}")

                if response.status == 200:
                    print("✅ SUCCESS! Check your Slack channel for the message")
                    print("🎯 Hook Node → Slack integration is working!")
                else:
                    print(f"❌ FAILED: Status {response.status}")

    except ImportError:
        print("❌ Missing aiohttp. Install with: pip install aiohttp")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(quick_slack_test())