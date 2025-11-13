#!/usr/bin/env python3
"""
Strategy Selection Example.

Demonstrates how to select and use different code generation strategies
based on requirements and use cases.

Usage:
    python examples/codegen/strategy_selection.py
"""

import asyncio
from pathlib import Path
from uuid import uuid4

from omninode_bridge.codegen import CodeGenerationService, ModelPRDRequirements


async def compare_strategies():
    """
    Compare different strategies for the same requirements.

    This example shows:
    1. Generate with Jinja2 (fast template-based)
    2. Generate with TemplateLoading (LLM-powered)
    3. Generate with Auto (automatic selection)
    4. Compare results
    """
    print("🔄 Strategy Comparison Example")
    print("=" * 60)

    service = CodeGenerationService()

    # Define requirements (same for all strategies)
    requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="cache_manager",
        domain="cache",
        business_description="Redis cache manager with TTL support",
        operations=["get", "set", "delete", "exists", "expire"],
        features=[
            "connection pooling",
            "automatic serialization",
            "TTL support",
            "metrics collection",
        ],
    )

    print("\n📋 Requirements:")
    print(f"   Service: {requirements.service_name}")
    print(f"   Domain: {requirements.domain}")

    # Strategy 1: Jinja2 (Fast template-based)
    print("\n" + "=" * 60)
    print("Strategy 1: Jinja2 (Template-based)")
    print("=" * 60)

    output_dir_jinja2 = Path(f"./generated/cache_manager_jinja2_{uuid4().hex[:8]}")
    result_jinja2 = await service.generate_node(
        requirements=requirements,
        output_directory=output_dir_jinja2,
        strategy="jinja2",  # Explicit Jinja2
        validation_level="standard",
    )

    print("\n✅ Jinja2 generation complete:")
    print(f"   Time: {result_jinja2.generation_time_ms:.0f}ms")
    print(f"   Files: {len(result_jinja2.artifacts.get_all_files())}")
    print(f"   Validation: {'PASSED' if result_jinja2.validation_passed else 'FAILED'}")

    # Strategy 2: Template Loading (LLM-powered) - only if LLM available
    print("\n" + "=" * 60)
    print("Strategy 2: Template Loading (LLM-powered)")
    print("=" * 60)

    output_dir_llm = Path(f"./generated/cache_manager_llm_{uuid4().hex[:8]}")
    try:
        result_llm = await service.generate_node(
            requirements=requirements,
            output_directory=output_dir_llm,
            strategy="template_loading",  # Explicit LLM
            enable_llm=True,
            validation_level="standard",
        )

        print("\n✅ LLM generation complete:")
        print(f"   Time: {result_llm.generation_time_ms:.0f}ms")
        print(f"   Files: {len(result_llm.artifacts.get_all_files())}")
        print(
            f"   Validation: {'PASSED' if result_llm.validation_passed else 'FAILED'}"
        )
        print(
            f"   Intelligence: {', '.join(result_llm.intelligence_sources) or 'None'}"
        )

    except Exception as e:
        print(f"\n⚠️  LLM generation skipped: {e}")
        result_llm = None

    # Strategy 3: Auto (Automatic selection)
    print("\n" + "=" * 60)
    print("Strategy 3: Auto (Automatic selection)")
    print("=" * 60)

    output_dir_auto = Path(f"./generated/cache_manager_auto_{uuid4().hex[:8]}")
    result_auto = await service.generate_node(
        requirements=requirements,
        output_directory=output_dir_auto,
        strategy="auto",  # Let service decide
        enable_llm=True,  # Allow LLM if beneficial
        validation_level="standard",
    )

    print("\n✅ Auto generation complete:")
    print(f"   Selected Strategy: {result_auto.strategy_used.value}")
    print(f"   Time: {result_auto.generation_time_ms:.0f}ms")
    print(f"   Files: {len(result_auto.artifacts.get_all_files())}")
    print(f"   Validation: {'PASSED' if result_auto.validation_passed else 'FAILED'}")

    # Comparison
    print("\n" + "=" * 60)
    print("📊 Strategy Comparison")
    print("=" * 60)

    print("\n| Strategy | Time | Files | Validation |")
    print("|----------|------|-------|------------|")
    print(
        f"| Jinja2   | {result_jinja2.generation_time_ms:>4.0f}ms | {len(result_jinja2.artifacts.get_all_files()):>5} | {'PASSED' if result_jinja2.validation_passed else 'FAILED':<10} |"
    )
    if result_llm:
        print(
            f"| LLM      | {result_llm.generation_time_ms:>4.0f}ms | {len(result_llm.artifacts.get_all_files()):>5} | {'PASSED' if result_llm.validation_passed else 'FAILED':<10} |"
        )
    print(
        f"| Auto     | {result_auto.generation_time_ms:>4.0f}ms | {len(result_auto.artifacts.get_all_files()):>5} | {'PASSED' if result_auto.validation_passed else 'FAILED':<10} |"
    )

    # Recommendations
    print("\n💡 Recommendations:")
    print("   - Use Jinja2: Simple CRUD, speed critical")
    print("   - Use LLM: Complex logic, high quality needed")
    print("   - Use Auto: General use, let service optimize")

    return {
        "jinja2": result_jinja2,
        "llm": result_llm,
        "auto": result_auto,
    }


async def use_case_strategy_selection():
    """
    Demonstrate strategy selection for different use cases.

    Shows when to use each strategy based on specific requirements.
    """
    print("\n" + "=" * 60)
    print("📚 Use Case Strategy Selection")
    print("=" * 60)

    service = CodeGenerationService()

    # Use Case 1: Simple CRUD (Use Jinja2)
    print("\n🔹 Use Case 1: Simple CRUD → Jinja2")
    print("   When: Well-defined CRUD, speed critical")
    print("   Why: Fast generation, proven patterns")

    simple_requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="product_crud",
        domain="database",
        business_description="Simple product CRUD operations",
        operations=["create", "read", "update", "delete"],
    )

    result1 = await service.generate_node(
        requirements=simple_requirements,
        output_directory=Path(f"./generated/product_crud_{uuid4().hex[:8]}"),
        strategy="jinja2",  # Fast template-based
    )
    print(f"   ✅ Generated in {result1.generation_time_ms:.0f}ms")

    # Use Case 2: Complex Algorithm (Use LLM)
    print("\n🔹 Use Case 2: Complex Algorithm → Template Loading")
    print("   When: Novel logic, complex business rules")
    print("   Why: LLM can generate sophisticated implementations")

    complex_requirements = ModelPRDRequirements(
        node_type="compute",
        service_name="recommendation_engine",
        domain="machine_learning",
        business_description="Collaborative filtering recommendation engine",
        operations=["compute_similarity", "generate_recommendations", "filter_results"],
        features=["user-based filtering", "item-based filtering", "hybrid approach"],
    )

    try:
        result2 = await service.generate_node(
            requirements=complex_requirements,
            output_directory=Path(f"./generated/recommendation_{uuid4().hex[:8]}"),
            strategy="template_loading",  # LLM-powered
            enable_llm=True,
        )
        print(f"   ✅ Generated in {result2.generation_time_ms:.0f}ms")
    except Exception as e:
        print(f"   ⚠️  LLM generation skipped: {e}")

    # Use Case 3: Production Critical (Use Auto or Hybrid)
    print("\n🔹 Use Case 3: Production Critical → Auto/Hybrid")
    print("   When: High visibility, must be high quality")
    print("   Why: Service optimizes for both speed and quality")

    critical_requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="payment_processor",
        domain="payment",
        business_description="Payment processing with multiple providers",
        operations=["process_payment", "refund", "verify", "reconcile"],
        features=[
            "PCI compliance",
            "multi-provider support",
            "automatic retry",
            "fraud detection",
        ],
    )

    result3 = await service.generate_node(
        requirements=critical_requirements,
        output_directory=Path(f"./generated/payment_{uuid4().hex[:8]}"),
        strategy="auto",  # Let service decide
        enable_llm=True,
        validation_level="strict",  # Strict validation
    )
    print(f"   ✅ Generated in {result3.generation_time_ms:.0f}ms")
    print(f"   📊 Selected strategy: {result3.strategy_used.value}")


async def list_available_strategies():
    """List all available strategies with their capabilities."""
    print("\n" + "=" * 60)
    print("📋 Available Strategies")
    print("=" * 60)

    service = CodeGenerationService()
    strategies = service.list_strategies()

    for strategy in strategies:
        print(f"\n🔹 {strategy['name']}")
        print(f"   Type: {strategy['type']}")
        print(f"   Default: {'Yes' if strategy['is_default'] else 'No'}")

        # Get detailed info
        try:
            info = service.get_strategy_info(strategy["type"])
            print(
                f"   Supported Types: {', '.join(info.get('supported_node_types', []))}"
            )
            print(f"   Requires LLM: {info.get('requires_llm', False)}")
            print(f"   Performance: {info.get('performance_profile', 'N/A')}")
        except Exception:
            pass


async def main():
    """Run all examples."""
    try:
        # List available strategies
        await list_available_strategies()

        # Compare strategies
        results = await compare_strategies()

        # Use case examples
        await use_case_strategy_selection()

        print("\n" + "=" * 60)
        print("✨ All examples completed successfully!")
        print("=" * 60)

        print("\n🔍 Key Takeaways:")
        print("   1. Jinja2: Fast, template-based, best for simple CRUD")
        print("   2. Template Loading: LLM-powered, best for complex logic")
        print("   3. Auto: Service decides, best for general use")
        print("   4. Match strategy to use case for optimal results")

        return 0

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
