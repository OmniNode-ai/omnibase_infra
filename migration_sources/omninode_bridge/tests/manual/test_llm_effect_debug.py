#!/usr/bin/env python3
"""Debug test to see raw Z.ai API response."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import httpx
from dotenv import load_dotenv


async def test_raw_api():
    """Test raw Z.ai API to see response structure."""

    # Load credentials from omniclaude .env
    load_dotenv("/Volumes/PRO-G40/Code/omniclaude/.env")

    # Verify API key is loaded
    api_key = os.getenv("ZAI_API_KEY")
    api_url = os.getenv("ZAI_ENDPOINT", "https://api.z.ai/api/anthropic")

    if not api_key:
        print("❌ ERROR: ZAI_API_KEY not found in environment")
        return False

    print(f"✅ API key loaded (length: {len(api_key)})")
    print(f"✅ API URL: {api_url}")

    # Build request
    request_data = {
        "model": "glm-4.5",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful coding assistant specialized in generating high-quality code.",
            },
            {
                "role": "user",
                "content": "Write a Python function that calculates the factorial of a number. Include docstring and type hints.",
            },
        ],
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 1.0,
    }

    print("\n🔄 Calling Z.ai API...")
    print(f"Request: {json.dumps(request_data, indent=2)}\n")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{api_url}/chat/completions",
                json=request_data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://omninode.ai",
                    "X-Title": "OmniNode Bridge Debug",
                },
            )

            response.raise_for_status()
            data = response.json()

            print("✅ SUCCESS! Raw response:")
            print(json.dumps(data, indent=2))

            # Extract and display the generated text
            generated_text = (
                data.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            print(f"\n📝 Generated text (length: {len(generated_text)}):")
            print(generated_text)

            return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_raw_api())
    sys.exit(0 if success else 1)
