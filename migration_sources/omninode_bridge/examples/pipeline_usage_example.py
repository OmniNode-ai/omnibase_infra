#!/usr/bin/env python3
"""
Example: Code Generation Pipeline Usage

Demonstrates how to use the unified CodeGenerationPipeline to:
1. Load pre-written templates
2. Enhance stubs with LLM-generated implementations
3. Generate production-ready nodes

IMPORTANT: Set ZAI_API_KEY environment variable before running with LLM enabled.
"""

import asyncio
import os
from pathlib import Path

from omninode_bridge.codegen import CodeGenerationPipeline


async def example_template_only():
    """Example: Load template without LLM enhancement."""
    print("=" * 80)
    print("Example 1: Template Loading (No LLM)")
    print("=" * 80)

    # Initialize pipeline with LLM disabled
    pipeline = CodeGenerationPipeline(
        template_dir=Path("templates/node_templates"),
        enable_llm=False,  # Disable LLM for template-only
    )

    # Define requirements
    requirements = {
        "service_name": "postgres_crud",
        "business_description": "PostgreSQL CRUD operations with connection pooling",
        "operations": ["create", "read", "update", "delete"],
        "domain": "database",
        "features": [
            "Async operations with asyncpg",
            "Connection pooling",
            "Transaction support",
        ],
    }

    try:
        # Generate node (template only, no LLM)
        result = await pipeline.generate_node(
            node_type="effect",
            version="v1_0_0",
            requirements=requirements,
        )

        print(f"\n✅ Generated: {result.node_name}")
        print(f"   Stubs detected: {len(result.methods_generated)}")
        print(f"   Node file size: {len(result.enhanced_node_file)} chars")
        print("\n   Note: Stubs not enhanced (LLM disabled)")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        await pipeline.cleanup()


async def example_with_llm():
    """Example: Load template and enhance with LLM."""
    print("\n")
    print("=" * 80)
    print("Example 2: Template Loading + LLM Enhancement")
    print("=" * 80)

    # Check for API key
    if not os.getenv("ZAI_API_KEY"):
        print("\n⚠️  ZAI_API_KEY not set - skipping LLM example")
        print("   Set environment variable to enable: export ZAI_API_KEY=your_key")
        return

    # Initialize pipeline with LLM enabled
    pipeline = CodeGenerationPipeline(
        template_dir=Path("templates/node_templates"),
        enable_llm=True,  # Enable LLM enhancement
    )

    # Define requirements
    requirements = {
        "service_name": "postgres_crud",
        "business_description": "PostgreSQL CRUD operations with connection pooling",
        "operations": ["create", "read", "update", "delete"],
        "domain": "database",
        "features": [
            "Async operations with asyncpg",
            "Connection pooling",
            "Transaction support",
            "Query result caching",
        ],
        "performance_requirements": {
            "max_latency_ms": 100,
            "target_throughput": 1000,
        },
    }

    # Additional context for LLM
    context_data = {
        "patterns": [
            "Use asyncpg for high-performance PostgreSQL access",
            "Implement connection pool with 10-20 connections",
            "Use prepared statements for performance",
        ],
        "best_practices": [
            "Always handle connection errors gracefully",
            "Log all database operations",
            "Use transactions for multi-operation workflows",
            "Implement proper connection lifecycle management",
        ],
    }

    try:
        # Generate node with LLM enhancement
        result = await pipeline.generate_node(
            node_type="effect",
            version="v1_0_0",
            requirements=requirements,
            context_data=context_data,
        )

        print(f"\n✅ Generated: {result.node_name}")
        print("\n📊 LLM Generation Metrics:")
        print(f"   Methods generated: {len(result.methods_generated)}")
        print(f"   Total tokens: {result.total_tokens_used:,}")
        print(f"   Total cost: ${result.total_cost_usd:.4f}")
        print(f"   Total latency: {result.total_latency_ms:.1f}ms")
        print(f"   Success rate: {result.generation_success_rate:.1%}")

        # Show details for each method
        if result.methods_generated:
            print("\n📝 Generated Methods:")
            for method in result.methods_generated:
                status = "✅" if method.syntax_valid else "❌"
                print(f"   {status} {method.method_name}")
                print(
                    f"      Tokens: {method.tokens_used}, Cost: ${method.cost_usd:.4f}"
                )
                print(f"      ONEX compliant: {method.onex_compliant}")
                if method.security_issues:
                    print(f"      ⚠️  Security issues: {len(method.security_issues)}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        await pipeline.cleanup()


async def example_discover_templates():
    """Example: Discover available templates."""
    print("\n")
    print("=" * 80)
    print("Example 3: Template Discovery")
    print("=" * 80)

    pipeline = CodeGenerationPipeline(
        template_dir=Path("templates/node_templates"),
        enable_llm=False,
    )

    try:
        templates = pipeline.discover_templates()

        print(f"\n📁 Found {len(templates)} templates:\n")

        for template in templates:
            print(
                f"   {template.node_type}/{template.version}/{template.template_name}.py"
            )
            if template.metadata.description:
                print(f"      {template.metadata.description[:60]}...")
            if template.metadata.tags:
                print(f"      Tags: {', '.join(template.metadata.tags)}")
            print()

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        await pipeline.cleanup()


async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Code Generation Pipeline Examples" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")

    # Run examples
    await example_discover_templates()
    await example_template_only()
    await example_with_llm()

    print("\n")
    print("=" * 80)
    print("Examples Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
