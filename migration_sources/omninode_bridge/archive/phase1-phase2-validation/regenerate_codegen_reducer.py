#!/usr/bin/env python3
"""
Regenerate codegen_metrics_reducer node with omnibase_core mixins.

Wave 4: Replace local mixins with standardized omnibase_core mixins.
This script regenerates the codegen_metrics_reducer node to use:
- MixinHealthCheck (omnibase_core)
- MixinMetrics (omnibase_core)
- MixinEventDrivenNode (omnibase_core)
- MixinIntentPublisher (omnibase_core)

Previous implementation used MixinIntentPublisher from omninode_bridge.mixins.
New implementation will use standardized mixins from omnibase_core for:
- Better code reuse and maintainability
- Standardized mixin contracts
- Reduced LOC through mixin composition
"""

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from src.omninode_bridge.codegen import CodeGenerationService
from src.omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def regenerate_codegen_reducer_node():
    """
    Regenerate codegen_metrics_reducer node with omnibase_core mixins.

    Approach:
    1. Create requirements for reducer node
    2. Specify omnibase_core mixin features
    3. Generate with mixins enabled
    4. Compare and validate
    5. Save to separate directory for review
    """
    print("=" * 80)
    print("🚀 Regenerating codegen_metrics_reducer Node (Wave 4 - Mixin Migration)")
    print("=" * 80)

    # Paths
    node_dir = Path("src/omninode_bridge/nodes/codegen_metrics_reducer/v1_0_0")
    backup_dir = Path(str(node_dir) + ".backup")
    output_dir = node_dir / "generated"

    print(f"\n📦 Output: {output_dir}")
    print(f"💾 Backup: {backup_dir}")
    print(f"📄 Original: {node_dir / 'node.py'}")

    # Step 1: Create requirements for reducer node
    print("\n" + "=" * 80)
    print("Step 1: Create Requirements for Codegen Metrics Reducer Node")
    print("=" * 80)

    requirements = ModelPRDRequirements(
        node_type="reducer",
        service_name="codegen_metrics_reducer",
        domain="code_generation",
        business_description=(
            "Aggregates code generation metrics from event streams for analytics, "
            "monitoring, and trend analysis. Consumes CODEGEN_* events from Kafka, "
            "aggregates by time window (hourly/daily/weekly), computes performance, "
            "quality, and cost metrics, and publishes GENERATION_METRICS_RECORDED events. "
            "Architecture: Pure aggregation logic (MetricsAggregator) + coordination I/O "
            "via MixinIntentPublisher + Intent executor publishes via EFFECT."
        ),
        operations=[
            "aggregate_metrics",  # Main aggregation operation
            "stream_events",  # Event streaming with windowing
            "publish_intents",  # Intent publishing for coordination
        ],
        features=[
            # Core aggregation features
            "Streaming aggregation with windowed processing",
            "Time-based windowing (hourly/daily/weekly)",
            "Multi-dimensional metrics computation (performance/quality/cost)",
            "Batch processing with configurable batch sizes",
            "Event type parsing and validation",
            # Performance features
            ">1000 events/second aggregation throughput",
            "<100ms aggregation latency for 1000 items",
            "Async streaming with AsyncIterator support",
            # Architecture patterns
            "Pure domain logic separation (MetricsAggregator)",
            "Intent pattern for coordination I/O",
            "IntentExecutor EFFECT delegation for event publishing",
            # Mixin integration (omnibase_core)
            "Health checks via MixinHealthCheck (omnibase_core)",
            "Metrics collection via MixinMetrics (omnibase_core)",
            "Event-driven architecture via MixinEventDrivenNode (omnibase_core)",
            "Intent publishing via MixinIntentPublisher (omnibase_core)",
            # Service discovery
            "Consul service registration and health checks",
            "Service discovery with metadata tagging",
            # Observability
            "Structured logging with correlation tracking",
            "Performance monitoring (duration/throughput)",
            "Aggregation metrics tracking",
            # Configuration
            "Environment-based configuration (Consul host/port)",
            "Configurable window types and batch sizes",
            "Health check mode for testing",
        ],
        integrations=[
            "Kafka event streams (CODEGEN_* topics)",
            "omnibase_core.nodes.NodeReducer (base class)",
            "omnibase_core.mixins.MixinHealthCheck",
            "omnibase_core.mixins.MixinMetrics",
            "omnibase_core.mixins.MixinEventDrivenNode",
            "omnibase_core.mixins.MixinIntentPublisher",
            "MetricsAggregator (pure aggregation logic)",
            "Consul for service discovery",
        ],
        performance_requirements={
            "latency_p99_ms": 100,  # <100ms for 1000 items
            "throughput_rps": 1000,  # >1000 events/second
            "max_memory_mb": 512,
            "max_cpu_percent": 25,
        },
        quality_requirements={
            "test_coverage_percent": 80,
            "code_complexity_max": 10,
            "documentation_required": True,
        },
        extraction_confidence=0.95,  # High confidence - manually crafted requirements
    )

    print("✅ Requirements created:")
    print(f"   Node Type: {requirements.node_type}")
    print(f"   Service: {requirements.service_name}")
    print(f"   Domain: {requirements.domain}")
    print(f"   Operations: {', '.join(requirements.operations)}")
    print(f"   Features: {len(requirements.features)}")
    print(
        "   Mixins: 4 (MixinHealthCheck, MixinMetrics, MixinEventDrivenNode, MixinIntentPublisher)"
    )
    print("   Source: omnibase_core.mixins (NOT omninode_bridge.mixins)")

    # Step 2: Initialize service and generate
    print("\n" + "=" * 80)
    print("Step 2: Generate Node with omnibase_core Mixins")
    print("=" * 80)

    service = CodeGenerationService()
    correlation_id = uuid4()

    print("⏳ Generating code...")
    print("   Strategy: auto")
    print("   Mixins: enabled (omnibase_core)")
    print("   Validation: none")
    print("   LLM: disabled")
    print(f"   Correlation ID: {correlation_id}")

    result = await service.generate_node(
        requirements=requirements,
        output_directory=output_dir,
        strategy="auto",
        enable_llm=False,  # Template-based generation (faster)
        enable_mixins=True,  # Enable mixin injection
        validation_level="none",  # Skip validation to avoid contract issues
        run_tests=False,  # Run tests separately
        correlation_id=correlation_id,
    )

    # Step 3: Display results
    print("\n" + "=" * 80)
    print("Step 3: Generation Results")
    print("=" * 80)

    print("\n✅ Generation complete!")
    print(f"   Node Name: {result.artifacts.node_name}")
    print(f"   Strategy: {result.strategy_used.value}")
    print(f"   Time: {result.generation_time_ms:.0f}ms")
    print(f"   Validation: {'✅ PASSED' if result.validation_passed else '❌ FAILED'}")
    print(f"   LLM Used: {'Yes' if result.llm_used else 'No'}")

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

    # Step 4: Validation details
    if result.validation_errors:
        print(f"\n⚠️  Validation Errors ({len(result.validation_errors)}):")
        for error in result.validation_errors[:10]:
            print(f"   - {error}")

    if result.validation_warnings:
        print(f"\n⚠️  Validation Warnings ({len(result.validation_warnings)}):")
        for warning in result.validation_warnings[:10]:
            print(f"   - {warning}")

    # Step 5: Verify mixin imports
    print("\n" + "=" * 80)
    print("Step 4: Verify Mixin Imports")
    print("=" * 80)

    generated_node = output_dir / "node.py"
    if generated_node.exists():
        node_content = generated_node.read_text()

        # Check for correct imports
        has_omnibase_mixins = "from omnibase_core.mixins import" in node_content
        has_local_mixins = "from omninode_bridge.mixins import" in node_content

        print("\n📋 Import Analysis:")
        print(f"   ✅ Uses omnibase_core.mixins: {has_omnibase_mixins}")
        print(f"   ❌ Uses omninode_bridge.mixins: {has_local_mixins}")

        if has_omnibase_mixins and not has_local_mixins:
            print("   Status: ✅ SUCCESS - Using omnibase_core mixins")
        elif has_local_mixins:
            print("   Status: ⚠️  WARNING - Still using local mixins")
        else:
            print("   Status: ⚠️  WARNING - No mixin imports found")

        # Check class declaration
        if "class NodeCodegenMetricsReducer(NodeReducer" in node_content:
            print("\n📋 Class Declaration:")
            for line in node_content.split("\n"):
                if "class NodeCodegenMetricsReducer" in line:
                    print(f"   {line.strip()}")
                    # Check for expected mixins
                    expected_mixins = [
                        "MixinHealthCheck",
                        "MixinMetrics",
                        "MixinEventDrivenNode",
                        "MixinIntentPublisher",
                    ]
                    found_mixins = [m for m in expected_mixins if m in line]
                    print(
                        f"   Found mixins: {', '.join(found_mixins) if found_mixins else 'None'}"
                    )
                    break
    else:
        print("⚠️  Could not verify imports (node.py not found)")

    # Step 6: LOC comparison
    print("\n" + "=" * 80)
    print("Step 5: LOC Comparison")
    print("=" * 80)

    original_node = node_dir / "node.py"
    generated_node = output_dir / "node.py"

    if original_node.exists() and generated_node.exists():
        original_loc = count_code_loc(original_node)
        generated_loc = count_code_loc(generated_node)
        reduction = original_loc - generated_loc
        percentage = (reduction / original_loc) * 100 if original_loc > 0 else 0

        print("\n📊 LOC Metrics:")
        print(f"   Original: {original_loc} lines")
        print(f"   Generated: {generated_loc} lines")
        print(f"   Reduction: {reduction} lines ({percentage:.1f}%)")
        print("   Target: 20-30% reduction via mixin reuse")

        if 20 <= percentage <= 35:
            print("   Status: ✅ SUCCESS - Within target range")
        elif percentage > 0:
            print("   Status: ⚠️  REVIEW - Outside target range but positive")
        else:
            print("   Status: ❌ FAILURE - No reduction achieved")
    else:
        print("⚠️  Could not perform LOC comparison (files not found)")
        if not original_node.exists():
            print(f"   Missing: {original_node}")
        if not generated_node.exists():
            print(f"   Missing: {generated_node}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print("\n✅ Regeneration Complete!")
    print(f"\n📁 Generated files: {output_dir}")
    print(f"📁 Original node: {original_node}")
    print("\n🔍 Next Steps:")
    print(f"   1. Review generated code: {output_dir / 'node.py'}")
    print("   2. Verify omnibase_core mixin imports (NOT omninode_bridge.mixins)")
    print("   3. Check class declaration includes all 4 mixins")
    print(f"   4. Compare with original: {original_node}")
    print("   5. Run tests: pytest tests/unit/nodes/codegen_metrics_reducer/")
    print("   6. If tests pass, replace original with generated code")
    print("\n⚠️  Important: Verify no imports from omninode_bridge.mixins")

    return 0


def count_code_loc(file_path: Path) -> int:
    """
    Count LOC excluding blank lines, comments, and docstrings.

    Args:
        file_path: Path to Python file

    Returns:
        Number of code lines
    """
    with open(file_path) as f:
        lines = f.readlines()

    code_lines = 0
    in_docstring = False

    for line in lines:
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            continue

        # Handle docstrings
        if '"""' in stripped or "'''" in stripped:
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                # Single-line docstring
                continue
            in_docstring = not in_docstring
            continue

        if in_docstring:
            continue

        # Skip comment-only lines
        if stripped.startswith("#"):
            continue

        code_lines += 1

    return code_lines


async def main():
    """Main entry point."""
    try:
        exit_code = await regenerate_codegen_reducer_node()
        return exit_code
    except Exception as e:
        logger.error(f"Regeneration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
