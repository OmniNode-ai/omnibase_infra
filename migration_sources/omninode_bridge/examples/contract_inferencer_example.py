#!/usr/bin/env python3
"""
ContractInferencer Integration Example.

Demonstrates how to use ContractInferencer to automatically generate
ONEX v2.0 contracts from existing node implementations.

This example shows:
1. Basic usage with LLM inference
2. Batch processing multiple nodes
3. Integration with code generation workflow
4. Error handling patterns

Usage:
    # With LLM inference (requires ZAI_API_KEY)
    poetry run python examples/contract_inferencer_example.py

    # Without LLM (uses default configs)
    poetry run python examples/contract_inferencer_example.py --no-llm
"""

import argparse
import asyncio
from pathlib import Path

from omninode_bridge.codegen.contract_inferencer import ContractInferencer


async def example_basic_usage():
    """Example 1: Basic usage with a single node."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Initialize inferencer
    inferencer = ContractInferencer(enable_llm=True)

    # Path to node file
    node_path = (
        Path(__file__).parent.parent
        / "src"
        / "omninode_bridge"
        / "nodes"
        / "llm_effect"
        / "v1_0_0"
        / "node.py"
    )

    print(f"\nGenerating contract from: {node_path.name}")

    # Generate contract
    contract_yaml = await inferencer.infer_from_node(node_path)

    # Print first 20 lines
    lines = contract_yaml.split("\n")[:20]
    print("\nGenerated contract (first 20 lines):")
    print("\n".join(lines))
    print("...")

    # Cleanup
    await inferencer.cleanup()

    print("\n✓ Basic usage complete")


async def example_batch_processing():
    """Example 2: Batch process multiple nodes."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Processing")
    print("=" * 80)

    # Initialize inferencer (reuse for multiple nodes)
    inferencer = ContractInferencer(enable_llm=True)

    # Find all node.py files
    nodes_dir = Path(__file__).parent.parent / "src" / "omninode_bridge" / "nodes"
    node_files = list(nodes_dir.rglob("*/v1_0_0/node.py"))

    print(f"\nFound {len(node_files)} node files to process")

    results = []
    for node_file in node_files[:3]:  # Process first 3 for demo
        try:
            print(f"\nProcessing: {node_file.parent.parent.name}")

            # Generate contract
            contract_yaml = await inferencer.infer_from_node(node_file)

            # Save to output directory
            output_path = (
                Path(__file__).parent.parent
                / "test_output"
                / "contracts"
                / f"{node_file.parent.parent.name}_inferred.yaml"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(contract_yaml)

            results.append({"node": node_file.parent.parent.name, "status": "success"})
            print(f"✓ Contract saved to: {output_path}")

        except Exception as e:
            results.append(
                {
                    "node": node_file.parent.parent.name,
                    "status": "failed",
                    "error": str(e),
                }
            )
            print(f"✗ Failed: {e}")

    # Cleanup
    await inferencer.cleanup()

    # Summary
    print("\n" + "-" * 80)
    print("Batch Processing Summary:")
    for result in results:
        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"  {status_icon} {result['node']}: {result['status']}")

    print("\n✓ Batch processing complete")


async def example_error_handling():
    """Example 3: Error handling patterns."""
    print("\n" + "=" * 80)
    print("Example 3: Error Handling")
    print("=" * 80)

    inferencer = ContractInferencer(enable_llm=True)

    # Test with invalid file
    print("\nTest 1: Invalid file path")
    try:
        await inferencer.infer_from_node("nonexistent/file.py")
        print("✗ Should have raised an error")
    except Exception as e:
        print(f"✓ Caught expected error: {type(e).__name__}")

    # Test with empty file
    print("\nTest 2: Empty file")
    try:
        empty_file = Path(__file__).parent.parent / "test_output" / "empty.py"
        empty_file.parent.mkdir(parents=True, exist_ok=True)
        empty_file.write_text("# Empty file")

        await inferencer.infer_from_node(empty_file)
        print("✗ Should have raised an error")
    except Exception as e:
        print(f"✓ Caught expected error: {type(e).__name__}")

    await inferencer.cleanup()

    print("\n✓ Error handling examples complete")


async def example_integration_workflow():
    """Example 4: Integration with code generation workflow."""
    print("\n" + "=" * 80)
    print("Example 4: Integration Workflow")
    print("=" * 80)

    print("\nSimulated workflow:")
    print("1. User provides PRD")
    print("2. Generate node.py using TemplateEngine + BusinessLogicGenerator")
    print("3. Use ContractInferencer to generate contract.yaml")
    print("4. Validate contract against schema")
    print("5. Use contract for deployment")

    print("\nIn practice:")
    print("```python")
    print("# Step 1: Generate node (existing workflow)")
    print("artifacts = await template_engine.generate_artifacts(...)")
    print("enhanced = await business_logic_generator.enhance_artifacts(...)")
    print()
    print("# Step 2: Generate contract (NEW - automated)")
    print("inferencer = ContractInferencer(enable_llm=True)")
    print("contract_yaml = await inferencer.infer_from_node(")
    print("    node_path=enhanced.enhanced_node_file_path,")
    print('    output_path=output_dir / "contract.yaml"')
    print(")")
    print()
    print("# Step 3: Validate and deploy")
    print("parser = YAMLContractParser()")
    print("contract = parser.parse_contract_file(contract_yaml)")
    print("if contract.is_valid:")
    print('    print("✓ Ready for deployment!")')
    print("```")

    print("\n✓ Integration workflow explanation complete")


async def main():
    """Run all examples."""
    parser = argparse.ArgumentParser(description="ContractInferencer examples")
    parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM inference (use defaults)"
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific example only",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ContractInferencer Integration Examples")
    print("=" * 80)

    if args.no_llm:
        print("\n⚠️  Running with LLM disabled (default configs)")
        # Update global setting (for demo purposes)
        import os

        os.environ.pop("ZAI_API_KEY", None)

    try:
        if args.example:
            if args.example == 1:
                await example_basic_usage()
            elif args.example == 2:
                await example_batch_processing()
            elif args.example == 3:
                await example_error_handling()
            elif args.example == 4:
                await example_integration_workflow()
        else:
            # Run all examples
            await example_basic_usage()
            await example_batch_processing()
            await example_error_handling()
            await example_integration_workflow()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✓ All examples completed")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
