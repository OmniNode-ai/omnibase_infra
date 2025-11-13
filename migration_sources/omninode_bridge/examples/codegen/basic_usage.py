#!/usr/bin/env python3
"""
Basic Code Generation Usage Example.

Demonstrates the simplest way to use CodeGenerationService to generate
a node from requirements.

Usage:
    python examples/codegen/basic_usage.py
"""

import asyncio
from pathlib import Path
from uuid import uuid4

from omninode_bridge.codegen import CodeGenerationService, ModelPRDRequirements


async def basic_generation_example():
    """
    Basic example: Generate a simple CRUD node.

    This example shows:
    1. Initialize CodeGenerationService
    2. Define requirements
    3. Generate code with default settings
    4. Access results
    """
    print("🚀 Basic Code Generation Example")
    print("=" * 60)

    # Step 1: Initialize service (with defaults)
    service = CodeGenerationService()
    print("\n✅ Service initialized")

    # Step 2: Define requirements
    requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="user_crud",
        domain="database",
        business_description="User CRUD operations with PostgreSQL",
        operations=["create", "read", "update", "delete", "list"],
        features=[
            "connection pooling",
            "automatic retry",
            "structured logging",
            "metrics collection",
        ],
    )
    print("\n✅ Requirements defined:")
    print(f"   Node Type: {requirements.node_type}")
    print(f"   Service: {requirements.service_name}")
    print(f"   Domain: {requirements.domain}")
    print(f"   Operations: {', '.join(requirements.operations)}")

    # Step 3: Generate code
    print("\n⏳ Generating code...")
    output_dir = Path(f"./generated/user_crud_{uuid4().hex[:8]}")

    result = await service.generate_node(
        requirements=requirements,
        output_directory=output_dir,
        strategy="auto",  # Let service choose best strategy
        validation_level="standard",  # Standard validation
    )

    print("\n✅ Generation complete!")

    # Step 4: Display results
    print("\n📊 Results:")
    print(f"   Node Name: {result.artifacts.node_name}")
    print(f"   Strategy: {result.strategy_used.value}")
    print(f"   Time: {result.generation_time_ms:.0f}ms")
    print(f"   Validation: {'✅ PASSED' if result.validation_passed else '❌ FAILED'}")
    print(f"   LLM Used: {'Yes' if result.llm_used else 'No'}")

    # Step 5: List generated files
    all_files = result.artifacts.get_all_files()
    print(f"\n📦 Generated {len(all_files)} files:")
    for filename in sorted(all_files.keys()):
        file_size = len(all_files[filename])
        print(f"   - {filename} ({file_size:,} bytes)")

    # Step 6: Write files to disk
    print(f"\n💾 Writing files to {output_dir}")
    for filename, content in all_files.items():
        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    print(f"\n✅ Files written to: {output_dir.absolute()}")

    # Step 7: Show validation details (if any issues)
    if result.validation_errors:
        print(f"\n⚠️  Validation Errors ({len(result.validation_errors)}):")
        for error in result.validation_errors[:5]:
            print(f"   - {error}")

    return result


async def main():
    """Run the example."""
    try:
        result = await basic_generation_example()

        print("\n" + "=" * 60)
        print("✨ Example completed successfully!")
        print("=" * 60)

        print("\n🔍 Next Steps:")
        print("   1. Review generated code")
        print("   2. Run tests: pytest <output_directory>/tests/")
        print("   3. Integrate with your project")

        return 0

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
