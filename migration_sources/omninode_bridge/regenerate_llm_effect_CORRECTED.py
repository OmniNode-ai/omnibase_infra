#!/usr/bin/env python3
"""
Regenerate llm_effect node with FULL automation - CORRECTED VERSION.

Wave 4 Phase 1 (Corrected): Business logic generation ENABLED.
"""

import asyncio
import logging
import os
from pathlib import Path
from uuid import uuid4

# Ensure credentials are loaded
from dotenv import load_dotenv

load_dotenv()

from src.omninode_bridge.codegen import CodeGenerationService
from src.omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def regenerate_llm_effect_node():
    """
    Regenerate llm_effect node with FULL automation including business logic.

    CORRECTED: enable_llm=True (credentials exist in .env!)
    """
    print("=" * 80)
    print("🚀 Regenerating llm_effect Node - CORRECTED (Full Automation)")
    print("=" * 80)

    # Verify credentials
    zai_key = os.getenv("ZAI_API_KEY")
    zai_endpoint = os.getenv("ZAI_ENDPOINT")

    if not zai_key:
        print("❌ ERROR: ZAI_API_KEY not found in .env")
        return

    print("\n✅ Z.ai credentials loaded:")
    print("   API Key: [REDACTED]")
    print(f"   Endpoint: {zai_endpoint}")

    # Paths
    node_dir = Path("src/omninode_bridge/nodes/llm_effect/v1_0_0")
    output_dir = node_dir / "llm_effect_full_automation"

    print(f"\n📦 Output: {output_dir}")

    # Requirements (from original regeneration script)
    requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="llm_effect",
        domain="ai_services",
        business_description=(
            "LLM Effect Node for multi-tier LLM generation with Z.ai integration. "
            "Supports CLOUD_FAST (GLM-4.5) tier with 128K context window. "
            "Includes circuit breaker, retry logic, token tracking, and cost management."
        ),
        operations=[
            "generate_text",
            "calculate_cost",
            "track_usage",
        ],
        features=[
            "Multi-tier LLM support (LOCAL, CLOUD_FAST, CLOUD_PREMIUM)",
            "Z.ai API integration (Anthropic-compatible endpoint)",
            "GLM-4.5 model support (PRIMARY tier)",
            "Circuit breaker pattern for fault tolerance",
            "Retry logic with exponential backoff",
            "HTTP client with connection pooling",
            "Token usage tracking (input/output/total)",
            "Cost tracking with sub-cent accuracy",
            "Comprehensive metrics collection via MixinMetrics",
            "Performance monitoring (latency, throughput)",
            "Health checks via MixinHealthCheck",
            "Z.ai API health monitoring",
            "Structured logging with correlation tracking",
            "OpenTelemetry tracing support",
            "Environment-based credentials (ZAI_API_KEY)",
            "Configurable timeouts and thresholds",
            "Per-tier pricing configuration",
        ],
        integrations=[
            "Z.ai API (https://api.z.ai/api/anthropic)",
            "omnibase_core ModelCircuitBreaker",
            "httpx AsyncClient for HTTP operations",
            "PostgreSQL for metrics persistence (via MixinMetrics)",
        ],
        performance_requirements={
            "latency_p95_ms": 3000,
            "throughput_rps": 10,
            "max_memory_mb": 512,
            "max_cpu_percent": 25,
        },
        quality_requirements={
            "test_coverage_percent": 80,
            "code_complexity_max": 10,
            "documentation_required": True,
        },
        extraction_confidence=0.95,
    )

    print("\n✅ Requirements created:")
    print(f"   Node Type: {requirements.node_type}")
    print(f"   Service: {requirements.service_name}")
    print(f"   Operations: {', '.join(requirements.operations)}")
    print(f"   Features: {len(requirements.features)}")

    # Initialize service
    print("\n" + "=" * 80)
    print("Step 1: Generate Node with FULL Automation (LLM Enabled)")
    print("=" * 80)

    service = CodeGenerationService()
    correlation_id = uuid4()

    print("⏳ Generating code...")
    print("   Strategy: hybrid (Jinja2 + LLM business logic)")
    print("   Mixins: enabled")
    print("   LLM: ✅ ENABLED (credentials verified)")
    print(f"   Correlation ID: {correlation_id}")

    result = await service.generate_node(
        requirements=requirements,
        output_directory=output_dir,
        strategy="hybrid",  # Use HybridStrategy for business logic generation
        enable_llm=True,  # ✅ CORRECTED: Enable LLM generation!
        enable_mixins=True,
        validation_level="basic",
        run_tests=False,
        correlation_id=correlation_id,
    )

    # Display results
    print("\n" + "=" * 80)
    print("Step 2: Generation Results")
    print("=" * 80)

    print("\n✅ Generation complete!")
    print(f"   Node Name: {result.artifacts.node_name}")
    print(f"   Strategy: {result.strategy_used.value}")
    print(f"   Time: {result.generation_time_ms:.0f}ms")
    print(f"   Validation: {'✅ PASSED' if result.validation_passed else '❌ FAILED'}")
    print(f"   LLM Used: {'✅ YES' if result.llm_used else '❌ NO'}")

    if result.mixins_applied:
        print(f"\n✅ Mixins Applied ({len(result.mixins_applied)}):")
        for mixin in result.mixins_applied:
            print(f"   - {mixin}")

    # List generated files
    all_files = result.artifacts.get_all_files()
    print(f"\n📦 Generated {len(all_files)} files:")
    for filename in sorted(all_files.keys()):
        file_size = len(all_files[filename])
        print(f"   - {filename} ({file_size:,} bytes)")

    # Write files
    print(f"\n💾 Writing files to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in all_files.items():
        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        print(f"   ✅ {filename}")

    print(f"\n✅ Files written to: {output_dir.absolute()}")

    # Validation details
    if result.validation_errors:
        print(f"\n⚠️  Validation Errors ({len(result.validation_errors)}):")
        for error in result.validation_errors[:10]:
            print(f"   - {error}")

    if result.validation_warnings:
        print(f"\n⚠️  Validation Warnings ({len(result.validation_warnings)}):")
        for warning in result.validation_warnings[:10]:
            print(f"   - {warning}")

    # Check if business logic was generated
    node_file = output_dir / "node.py"
    if node_file.exists():
        content = node_file.read_text()
        has_stubs = "IMPLEMENTATION REQUIRED" in content or "# Stub" in content

        print("\n" + "=" * 80)
        print("Step 3: Business Logic Verification")
        print("=" * 80)

        if has_stubs:
            print("⚠️  Node contains stubs - business logic NOT fully generated")
        else:
            print("✅ Business logic appears to be generated (no stub markers found)")

        # Count lines
        lines = len(
            [
                line
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
        )
        print(f"   Total code lines: {lines}")

    print("\n" + "=" * 80)
    print("🎉 Generation Complete!")
    print("=" * 80)
    print(f"\n✅ Check {output_dir} for generated files")
    print(
        f"✅ LLM business logic generation: {'ENABLED' if result.llm_used else 'FAILED'}"
    )


if __name__ == "__main__":
    asyncio.run(regenerate_llm_effect_node())
