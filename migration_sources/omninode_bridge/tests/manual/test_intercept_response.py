#!/usr/bin/env python3
"""Intercept HTTP response to debug content extraction."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv
from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.llm_effect.v1_0_0 import NodeLLMEffect
from omninode_bridge.nodes.llm_effect.v1_0_0.models import EnumLLMTier


async def test_with_intercept():
    """Test with intercepted HTTP response."""

    # Load credentials
    load_dotenv("/Volumes/PRO-G40/Code/omniclaude/.env")

    # Initialize node
    container = ModelContainer(value={}, container_type="config")
    node = NodeLLMEffect(container)
    await node.initialize()

    print(f"✅ Node initialized with URL: {node.config.zai_base_url}")

    # Create test contract
    from uuid import uuid4

    from omninode_bridge.nodes.llm_effect.v1_0_0.models import ModelLLMRequest

    request = ModelLLMRequest(
        prompt="Write a Python function that calculates the factorial of a number. Include docstring and type hints.",
        tier=EnumLLMTier.CLOUD_FAST,
        max_tokens=500,
        temperature=0.7,
    )

    print("\n🔄 Making request...")

    # Save original post method
    original_post = node.http_client.post

    async def intercepting_post(*args, **kwargs):
        """Intercept and log the response."""
        print("\n📤 HTTP POST called:")
        print(f"  URL: {args[0]}")
        print(f"  Headers: {json.dumps(kwargs.get('headers', {}), indent=2)}")

        # Call original
        response = await original_post(*args, **kwargs)

        # Log response
        print("\n📥 HTTP Response:")
        print(f"  Status: {response.status_code}")
        print(f"  Headers: {dict(response.headers)}")

        # Parse JSON
        data = response.json()
        print("\n📋 Response JSON:")
        print(json.dumps(data, indent=2))

        # Extract content
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print("\n📝 Extracted content:")
        print(f"  Type: {type(content)}")
        print(f"  Length: {len(content) if content else 0}")
        print(f"  First 200 chars: {content[:200] if content else '[EMPTY]'}")

        return response

    # Monkey-patch
    node.http_client.post = intercepting_post

    try:
        # Call the internal method
        response = await node._generate_cloud_fast(request, uuid4())

        print("\n✅ Final ModelLLMResponse:")
        print(f"  generated_text length: {len(response.generated_text)}")
        print(
            f"  generated_text (first 200 chars): {response.generated_text[:200] if response.generated_text else '[EMPTY]'}"
        )
        print(f"  tokens: {response.tokens_total}")
        print(f"  cost: ${response.cost_usd:.6f}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        await node.cleanup()


if __name__ == "__main__":
    success = asyncio.run(test_with_intercept())
    sys.exit(0 if success else 1)
