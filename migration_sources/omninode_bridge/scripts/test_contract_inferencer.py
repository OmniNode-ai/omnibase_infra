#!/usr/bin/env python3
"""
Test script for ContractInferencer.

Tests the automated contract generation from existing node.py files.

Usage:
    python scripts/test_contract_inferencer.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed, environment variables may not be loaded")

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omninode_bridge.codegen.contract_inferencer import ContractInferencer


async def main():
    """Test ContractInferencer with llm_effect node."""
    print("=" * 80)
    print("Testing ContractInferencer")
    print("=" * 80)

    # Check for ZAI_API_KEY
    if not os.getenv("ZAI_API_KEY"):
        print("\n⚠️  Warning: ZAI_API_KEY not set")
        print("Set ZAI_API_KEY environment variable to enable LLM inference")
        print("\nRunning with LLM disabled (will use default configs)")
        enable_llm = False
    else:
        print("\n✓ ZAI_API_KEY found - LLM inference enabled")
        enable_llm = True

    # Initialize inferencer
    print("\nInitializing ContractInferencer...")
    inferencer = ContractInferencer(enable_llm=enable_llm)

    # Test with llm_effect node
    node_path = (
        Path(__file__).parent.parent
        / "src"
        / "omninode_bridge"
        / "nodes"
        / "llm_effect"
        / "v1_0_0"
        / "node.py"
    )

    if not node_path.exists():
        print(f"\n❌ Error: Node file not found: {node_path}")
        return 1

    print(f"\nAnalyzing node: {node_path}")
    print("-" * 80)

    try:
        # Generate contract
        contract_yaml = await inferencer.infer_from_node(
            node_path=node_path,
            # Don't write to file for testing
        )

        print("\n✓ Contract generated successfully!")
        print("\n" + "=" * 80)
        print("Generated Contract:")
        print("=" * 80)
        print(contract_yaml)
        print("=" * 80)

        # Optionally write to test output
        output_path = (
            Path(__file__).parent.parent / "test_output" / "inferred_contract.yaml"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(contract_yaml)
        print(f"\n✓ Contract saved to: {output_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        await inferencer.cleanup()

    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
