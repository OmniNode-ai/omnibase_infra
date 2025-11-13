#!/usr/bin/env python3
"""
Code Generation Workflow Integration Example - Phase 4 Complete.

Demonstrates the complete integration of all 7 Phase 4 workflow components:

**Core Workflow (Weeks 1-4):**
1. StagedParallelExecutor - 6-phase orchestration pipeline
2. TemplateManager - LRU-cached template loading (85-95% hit rate)
3. ValidationPipeline - Multi-stage validation (completeness, quality, ONEX)
4. AIQuorum - 4-model consensus validation

**Optimization (Weeks 7-8):**
5. ErrorRecoveryOrchestrator - 5 recovery strategies (90%+ success)
6. PerformanceOptimizer - Automatic optimization (2-3x speedup)
7. ProductionMonitor - SLA tracking and alerting

This example shows:
- Contract → code generation workflow
- Template-based code generation
- Multi-stage validation
- AI quorum consensus (optional)
- Automatic error recovery
- Performance optimization
- Production monitoring and SLA tracking
- Performance metrics and statistics

Performance targets:
- Full workflow: <5s for typical contract (with optimization: 2-3x faster)
- Template hit rate: 85-95% (optimized: 95%+)
- Validation: <800ms (pipeline) + 2-10s (quorum)
- Overall speedup: 2.25x-4.17x vs sequential (optimized: 3-4x)
- Error recovery success: 90%+ for recoverable errors
- Monitoring overhead: <5ms

Usage:
    # Without optimization (baseline)
    python code_generation_workflow_example.py --no-optimization --no-monitoring

    # With optimization (default)
    python code_generation_workflow_example.py

    # With optimization and AI Quorum
    python code_generation_workflow_example.py --enable-quorum

    # Custom template directory
    python code_generation_workflow_example.py --template-dir /path/to/templates

    # Performance benchmark
    python code_generation_workflow_example.py --benchmark --iterations 10
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.workflows import (
    CodeGenerationWorkflow,
    EnumStageStatus,
    WorkflowResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def create_sample_contracts(output_dir: Path) -> list[str]:
    """
    Create sample contract files for testing.

    Args:
        output_dir: Directory to create contracts in

    Returns:
        List of contract file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    contracts = []

    # Contract 1: Simple Effect Node
    contract1 = output_dir / "contract_effect_simple.yaml"
    contract1.write_text(
        """
node_type: effect
node_name: SimpleEffect
version: 1.0.0
description: A simple effect node for testing
"""
    )
    contracts.append(str(contract1))

    # Contract 2: Complex Effect Node
    contract2 = output_dir / "contract_effect_complex.yaml"
    contract2.write_text(
        """
node_type: effect
node_name: ComplexEffect
version: 1.0.0
description: A complex effect node with multiple features
"""
    )
    contracts.append(str(contract2))

    logger.info(f"Created {len(contracts)} sample contracts in {output_dir}")
    return contracts


async def run_workflow_example(
    contracts: list[str],
    enable_quorum: bool = False,
    enable_optimization: bool = True,
    enable_monitoring: bool = True,
    template_dir: str | None = None,
) -> WorkflowResult:
    """
    Run code generation workflow with all 7 components.

    Args:
        contracts: List of contract file paths
        enable_quorum: Enable AI quorum validation
        enable_optimization: Enable performance optimization (default: True)
        enable_monitoring: Enable production monitoring (default: True)
        template_dir: Optional template directory

    Returns:
        WorkflowResult with execution summary
    """
    logger.info("=" * 80)
    logger.info("CODE GENERATION WORKFLOW EXAMPLE - PHASE 4 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Contracts: {len(contracts)}")
    logger.info(f"AI Quorum: {'ENABLED' if enable_quorum else 'DISABLED'}")
    logger.info(f"Optimization: {'ENABLED' if enable_optimization else 'DISABLED'}")
    logger.info(f"Monitoring: {'ENABLED' if enable_monitoring else 'DISABLED'}")
    logger.info(f"Template Dir: {template_dir or 'DEFAULT'}")
    logger.info("=" * 80)

    # Initialize metrics collector
    metrics = MetricsCollector()

    # Create workflow with all 7 components
    workflow = CodeGenerationWorkflow(
        template_dir=template_dir,
        metrics_collector=metrics,
        quality_threshold=0.8,
        enable_ai_quorum=enable_quorum,
        quorum_threshold=0.6,
        cache_size=100,
        enable_optimization=enable_optimization,
        enable_monitoring=enable_monitoring,
    )

    try:
        # Initialize workflow
        logger.info("\n[1/3] Initializing workflow components...")
        start_init = time.perf_counter()
        await workflow.initialize()
        init_duration = (time.perf_counter() - start_init) * 1000
        logger.info(f"✓ Initialization complete ({init_duration:.2f}ms)")

        # Run code generation
        logger.info("\n[2/3] Running code generation workflow...")
        logger.info("  Phase 1: Parse contracts")
        logger.info("  Phase 2: Generate models (with templates)")
        logger.info("  Phase 3: Generate validators (with templates)")
        logger.info("  Phase 4: Generate tests (with templates)")
        logger.info("  Phase 5: Validate code (pipeline + quorum)")
        logger.info("  Phase 6: Package nodes")

        start_workflow = time.perf_counter()
        result = await workflow.generate_code(
            contracts=contracts,
            workflow_id="example-workflow-1",
            output_dir=str(Path.cwd() / "output"),
        )
        workflow_duration = (time.perf_counter() - start_workflow) * 1000

        logger.info(f"✓ Workflow complete ({workflow_duration:.2f}ms)")

        # Display results
        logger.info("\n[3/3] Workflow Results")
        logger.info("=" * 80)
        display_workflow_results(result, workflow_duration)

        # Display component statistics
        logger.info("\n" + "=" * 80)
        logger.info("COMPONENT STATISTICS")
        logger.info("=" * 80)
        display_component_stats(workflow)

        return result

    finally:
        # Cleanup
        logger.info("\nCleaning up workflow...")
        await workflow.shutdown()
        logger.info("✓ Cleanup complete")


def display_workflow_results(result: WorkflowResult, duration_ms: float) -> None:
    """Display workflow execution results."""
    # Overall status
    status_symbol = "✓" if result.status == EnumStageStatus.COMPLETED else "✗"
    logger.info(f"Status: {status_symbol} {result.status.value.upper()}")
    logger.info(f"Workflow ID: {result.workflow_id}")
    logger.info(f"Duration: {duration_ms:.2f}ms")
    logger.info(f"Speedup: {result.overall_speedup:.2f}x")
    logger.info("")

    # Stage summary
    logger.info("Stages:")
    logger.info(f"  Total: {result.total_stages}")
    logger.info(f"  Successful: {result.successful_stages}")
    logger.info(f"  Failed: {result.failed_stages}")
    logger.info("")

    # Step summary
    logger.info("Steps:")
    logger.info(f"  Total: {result.total_steps}")
    logger.info(f"  Successful: {result.successful_steps}")
    logger.info(f"  Failed: {result.failed_steps}")
    logger.info("")

    # Stage details
    logger.info("Stage Details:")
    for stage_result in result.stage_results:
        status_symbol = "✓" if stage_result.status == EnumStageStatus.COMPLETED else "✗"
        logger.info(
            f"  {status_symbol} Stage {stage_result.stage_number}: {stage_result.stage_id}"
        )
        logger.info(f"      Duration: {stage_result.duration_ms:.2f}ms")
        logger.info(
            f"      Steps: {stage_result.successful_steps}/{stage_result.total_steps}"
        )
        logger.info(f"      Speedup: {stage_result.speedup_ratio:.2f}x")


def display_component_stats(workflow: CodeGenerationWorkflow) -> None:
    """Display statistics for all workflow components."""
    stats = workflow.get_statistics()

    # Workflow stats
    logger.info("Workflow:")
    logger.info(f"  Total Generations: {stats['generation_count']}")
    logger.info(f"  Avg Duration: {stats['avg_duration_ms']:.2f}ms")
    logger.info("")

    # Template Manager stats
    logger.info("Template Manager:")
    logger.info(f"  Cache Hit Rate: {stats['template_hit_rate']:.2%}")
    logger.info(f"  Cache Size: {stats['template_cache_size']}")
    logger.info(f"  Avg Load Time: {stats['template_avg_load_ms']:.2f}ms")
    logger.info("")

    # AI Quorum stats (if enabled)
    if "quorum_validations" in stats:
        logger.info("AI Quorum:")
        logger.info(f"  Total Validations: {stats['quorum_validations']}")
        logger.info(f"  Pass Rate: {stats['quorum_pass_rate']:.2%}")
        logger.info("")

    # Error Recovery stats (if enabled)
    if stats.get("optimization_enabled") and "error_recovery" in stats:
        recovery = stats["error_recovery"]
        logger.info("Error Recovery:")
        logger.info(f"  Total Attempts: {recovery['total_attempts']}")
        logger.info(f"  Successful Recoveries: {recovery['successful_recoveries']}")
        logger.info(f"  Success Rate: {recovery['success_rate']:.2%}")
        logger.info("")

    # Performance Optimization stats (if enabled)
    if stats.get("optimization_enabled") and "performance_optimization" in stats:
        perf_opt = stats["performance_optimization"]
        logger.info("Performance Optimization:")
        logger.info(f"  Total Optimizations: {perf_opt['total_optimizations']}")
        logger.info(f"  Avg Speedup: {perf_opt['avg_speedup']:.2f}x")
        logger.info("")

    # Monitoring stats (if enabled)
    if "monitoring" in stats:
        monitoring = stats["monitoring"]
        logger.info("Production Monitoring:")
        logger.info(
            f"  Status: {'ACTIVE' if monitoring['is_monitoring'] else 'INACTIVE'}"
        )
        logger.info(f"  Overhead: {monitoring['overhead_ms']:.2f}ms")
        logger.info("")

    # Display optimization summaries
    recovery_stats = workflow.get_recovery_stats()
    if recovery_stats.get("enabled"):
        logger.info("Error Recovery Details:")
        logger.info(f"  Strategies Used: {recovery_stats.get('strategies_used', {})}")
        logger.info(f"  Error Types Seen: {recovery_stats.get('error_types_seen', {})}")
        logger.info("")

    sla_compliance = workflow.get_sla_compliance()
    if sla_compliance.get("enabled", True):  # Default to enabled for display
        logger.info("SLA Compliance:")
        logger.info(
            f"  Overall Compliant: {sla_compliance.get('overall_compliant', 'N/A')}"
        )
        if "metrics" in sla_compliance:
            compliant_count = sum(
                1
                for m in sla_compliance["metrics"].values()
                if m.get("status") == "compliant"
            )
            total_count = len(sla_compliance["metrics"])
            logger.info(f"  Metrics Compliant: {compliant_count}/{total_count}")
        logger.info("")


async def run_performance_benchmark(iterations: int = 5) -> None:
    """
    Run performance benchmark with multiple iterations.

    Args:
        iterations: Number of iterations to run
    """
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Running {iterations} iterations...")

    # Create sample contracts
    output_dir = Path.cwd() / "tmp_contracts"
    contracts = await create_sample_contracts(output_dir)

    # Initialize workflow once
    workflow = CodeGenerationWorkflow(
        metrics_collector=MetricsCollector(),
        quality_threshold=0.8,
        enable_ai_quorum=False,  # Disable for faster benchmarking
        cache_size=100,
    )
    await workflow.initialize()

    durations: list[float] = []

    try:
        for i in range(iterations):
            logger.info(f"\nIteration {i + 1}/{iterations}...")
            start = time.perf_counter()

            result = await workflow.generate_code(
                contracts=contracts,
                workflow_id=f"benchmark-{i}",
            )

            duration = (time.perf_counter() - start) * 1000
            durations.append(duration)

            logger.info(
                f"  Duration: {duration:.2f}ms, "
                f"Speedup: {result.overall_speedup:.2f}x, "
                f"Status: {result.status.value}"
            )

        # Calculate statistics
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        logger.info("\n" + "-" * 80)
        logger.info("Benchmark Results:")
        logger.info(f"  Iterations: {iterations}")
        logger.info(f"  Avg Duration: {avg_duration:.2f}ms")
        logger.info(f"  Min Duration: {min_duration:.2f}ms")
        logger.info(f"  Max Duration: {max_duration:.2f}ms")
        logger.info(f"  Target: <5000ms ({'PASS' if avg_duration < 5000 else 'FAIL'})")

        # Component stats
        stats = workflow.get_statistics()
        logger.info(f"\n  Template Hit Rate: {stats['template_hit_rate']:.2%}")
        logger.info(
            f"  Target: >85% ({'PASS' if stats['template_hit_rate'] > 0.85 else 'FAIL'})"
        )

    finally:
        await workflow.shutdown()
        # Cleanup temp contracts
        import shutil

        if output_dir.exists():
            shutil.rmtree(output_dir)


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Code Generation Workflow Integration Example - Phase 4 Complete"
    )
    parser.add_argument(
        "--enable-quorum",
        action="store_true",
        help="Enable AI quorum validation (slower but higher quality)",
    )
    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="Disable performance optimization (enabled by default)",
    )
    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable production monitoring (enabled by default)",
    )
    parser.add_argument(
        "--template-dir",
        type=str,
        help="Custom template directory path",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)",
    )

    args = parser.parse_args()

    try:
        if args.benchmark:
            # Run performance benchmark
            await run_performance_benchmark(iterations=args.iterations)
        else:
            # Create sample contracts
            output_dir = Path.cwd() / "tmp_contracts"
            contracts = await create_sample_contracts(output_dir)

            # Run workflow example with optimization flags
            result = await run_workflow_example(
                contracts=contracts,
                enable_quorum=args.enable_quorum,
                enable_optimization=not args.no_optimization,
                enable_monitoring=not args.no_monitoring,
                template_dir=args.template_dir,
            )

            # Cleanup
            import shutil

            if output_dir.exists():
                shutil.rmtree(output_dir)

            # Exit with appropriate code
            if result.status == EnumStageStatus.COMPLETED:
                logger.info("\n✓ Example completed successfully!")
                sys.exit(0)
            else:
                logger.error("\n✗ Example failed!")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nExample failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
