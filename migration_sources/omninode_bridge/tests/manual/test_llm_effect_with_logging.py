#!/usr/bin/env python3
"""Test NodeLLMEffect with added logging to debug API response extraction."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.llm_effect.v1_0_0 import NodeLLMEffect

# Monkey-patch the _generate_cloud_fast method to add logging
original_generate = NodeLLMEffect._generate_cloud_fast


async def logged_generate(self, request, correlation_id):
    """Wrapper to log API request and response."""
    print("\n📤 API Request:")
    print(f"  URL: {self.config.zai_base_url}/chat/completions")
    print("  API Key: [REDACTED]")

    # Call original with logging
    model = (
        request.model_override
        or self.tier_models[self.tier_models.__class__.__dict__["CLOUD_FAST"]]
    )

    zai_request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": request.system_prompt
                or "You are a helpful coding assistant specialized in generating high-quality code.",
            },
            {"role": "user", "content": request.prompt},
        ],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
    }

    print("\n📋 Request body:")
    print(json.dumps(zai_request, indent=2))

    response_obj = await self.http_client.post(
        f"{self.config.zai_base_url}/chat/completions",
        json=zai_request,
        headers={
            "Authorization": f"Bearer {self.config.zai_api_key}",
            "HTTP-Referer": "https://omninode.ai",
            "X-Title": "OmniNode Bridge",
        },
    )

    response_obj.raise_for_status()
    data = response_obj.json()

    print("\n📥 API Response:")
    print(json.dumps(data, indent=2))

    # Extract content
    generated_text = data["choices"][0]["message"]["content"]
    print("\n📝 Extracted content:")
    print(f"  Type: {type(generated_text)}")
    print(f"  Length: {len(generated_text) if generated_text else 0}")
    print(
        f"  Value (first 200 chars): {generated_text[:200] if generated_text else '[None/Empty]'}"
    )

    # Call original method
    return await original_generate(self, request, correlation_id)


NodeLLMEffect._generate_cloud_fast = logged_generate


async def test_llm_effect():
    """Test NodeLLMEffect with real Z.ai API and logging."""

    # Load credentials from omniclaude .env
    load_dotenv("/Volumes/PRO-G40/Code/omniclaude/.env")

    # Verify API key is loaded
    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        print("❌ ERROR: ZAI_API_KEY not found in environment")
        return False

    print(f"✅ API key loaded (length: {len(api_key)})")
    print(f"✅ ZAI_ENDPOINT: {os.getenv('ZAI_ENDPOINT', 'default')}")

    # Initialize node (credentials read from environment)
    container = ModelContainer(value={}, container_type="config")
    node = NodeLLMEffect(container)
    print("✅ NodeLLMEffect initialized")
    print(f"  Base URL: {node.config.zai_base_url}")

    # Create test contract
    contract = ModelContractEffect(
        name="llm_test",
        version={"major": 1, "minor": 0, "patch": 0},
        input_data={
            "prompt": "Write a Python function that calculates the factorial of a number. Include docstring and type hints.",
            "tier": "CLOUD_FAST",
            "max_tokens": 500,
            "temperature": 0.7,
        },
    )

    print("\n🔄 Calling Z.ai API (GLM-4.5)...")
    try:
        response = await node.execute_effect(contract)

        print("\n✅ SUCCESS! Final response:")
        print(f"  - Model used: {response.model_used}")
        print(f"  - Tokens: {response.tokens_total}")
        print(f"  - Generated text length: {len(response.generated_text)}")

        if response.generated_text:
            print("\n  Generated code (first 500 chars):")
            print(f"  {response.generated_text[:500]}")
        else:
            print("\n  ❌ generated_text is EMPTY!")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_llm_effect())
    sys.exit(0 if success else 1)
