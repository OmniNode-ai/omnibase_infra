#!/usr/bin/env python3
"""Live test of NodeLLMEffect with real Z.ai API."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv
from omnibase_core.models.core import ModelContainer
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omninode_bridge.nodes.llm_effect.v1_0_0 import NodeLLMEffect


async def test_llm_effect():
    """Test NodeLLMEffect with real Z.ai API."""

    # Load credentials from omniclaude .env
    load_dotenv("/Volumes/PRO-G40/Code/omniclaude/.env")

    # Verify API key is loaded
    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        print("❌ ERROR: ZAI_API_KEY not found in environment")
        return False

    print(f"✅ API key loaded (length: {len(api_key)})")

    # Initialize node (credentials read from environment)
    container = ModelContainer(value={}, container_type="config")
    node = NodeLLMEffect(container)
    print("✅ NodeLLMEffect initialized")

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

        print("\n✅ SUCCESS! Response received:")
        print(f"  - Response type: {type(response)}")
        print(f"  - Model used: {response.model_used}")
        print(
            f"  - Tokens: {response.tokens_total} ({response.tokens_input} in, {response.tokens_output} out)"
        )
        print(f"  - Latency: {response.latency_ms:.2f}ms")
        print(f"  - Cost: ${response.cost_usd:.6f}")
        print(f"  - Finish reason: {response.finish_reason}")
        print(f"  - Truncated: {response.truncated}")
        print(f"  - Warnings: {response.warnings}")

        print(f"\n  Generated text info:")
        print(f"  - Type: {type(response.generated_text)}")
        print(f"  - Length: {len(response.generated_text)} characters")
        print(f"  - Repr: {repr(response.generated_text[:100])}")

        if response.generated_text:
            print(f"\n  Generated code preview (first 500 chars):")
            print(f"  {response.generated_text[:500]}")
        else:
            print(f"\n  ❌ WARNING: generated_text is empty!")
            print(f"  Response dict: {response.model_dump()}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_llm_effect())
    sys.exit(0 if success else 1)
