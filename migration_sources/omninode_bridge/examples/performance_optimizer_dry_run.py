"""
Performance optimizer dry-run mode usage example.

Demonstrates how to use the dry-run mode to simulate optimizations
without actually applying changes to the system.

This is useful for:
- Testing optimization recommendations
- Understanding what would change before applying
- Generating reports for review before production changes
- Validating optimization logic
"""

import asyncio

from omninode_bridge.agents.workflows.performance_optimizer import PerformanceOptimizer
from omninode_bridge.agents.workflows.profiling import PerformanceProfiler


async def example_workflow():
    """Sample workflow to optimize."""
    # Simulate some work with profiling points
    profiler = PerformanceProfiler()

    async with profiler.profile_operation("parse_input"):
        await asyncio.sleep(0.02)  # Simulate parsing

    async with profiler.profile_operation("generate_code"):
        await asyncio.sleep(0.05)  # Simulate code generation

    async with profiler.profile_operation("validate_output"):
        await asyncio.sleep(0.01)  # Simulate validation


async def main():
    """Run dry-run optimization example."""
    print("=" * 60)
    print("Performance Optimizer - Dry-Run Mode Example")
    print("=" * 60)

    # Create profiler and optimizer with dry_run=True
    profiler = PerformanceProfiler()
    optimizer = PerformanceOptimizer(
        profiler=profiler,
        dry_run=True,  # Enable dry-run mode
        auto_optimize=True,  # Auto-apply optimizations (but in dry-run mode)
    )

    print("\n1. Running workflow optimization in DRY-RUN mode...")
    print("-" * 60)

    # Run optimization - will simulate all changes
    report = await optimizer.optimize_workflow(
        workflow_id="example-session",
        workflow_func=example_workflow,
        iterations=3,
    )

    print("\nPerformance Report:")
    print(f"  - Total duration: {report.total_duration_ms:.2f}ms")
    print(f"  - Profiled operations: {len(report.profile_results)}")
    print(f"  - Recommendations: {len(report.recommendations)}")
    print(f"  - Estimated speedup: {report.estimated_speedup:.2f}x")

    # Get dry-run report
    dry_report = optimizer.get_dry_run_report()

    if dry_report:
        print("\n2. Dry-Run Report:")
        print("-" * 60)
        print(dry_report.get_summary())

        print("\n3. Export to JSON:")
        print("-" * 60)
        json_data = dry_report.export_json()
        print(f"  - Workflow: {json_data['workflow_id']}")
        print(f"  - Total changes: {json_data['total_changes']}")
        print(f"  - Affected components: {len(json_data['affected_components'])}")
        print("  - Changes by area:")
        for area, count in json_data["changes_by_area"].items():
            print(f"    * {area}: {count} changes")

    print("\n4. Comparison: Dry-Run vs Actual")
    print("-" * 60)

    # Create optimizer without dry-run to show difference
    optimizer_real = PerformanceOptimizer(
        profiler=profiler,
        dry_run=False,
        auto_optimize=False,  # Don't auto-apply for this example
    )

    print("\nWith dry_run=True:")
    print("  ✓ No actual changes applied")
    print("  ✓ Generates detailed report of proposed changes")
    print("  ✓ Safe to test on production systems")
    print("  ✓ Shows current vs proposed values")
    print("  ✓ Estimates performance impact")

    print("\nWith dry_run=False:")
    print("  • Actually modifies system configuration")
    print("  • Preloads templates into cache")
    print("  • Updates parallel execution settings")
    print("  • Should be used after reviewing dry-run report")

    print("\n5. Method-Level Override Example:")
    print("-" * 60)

    # You can override instance-level dry_run in individual methods
    optimizer_override = PerformanceOptimizer(
        profiler=profiler,
        dry_run=False,  # Default is not dry-run
    )

    # But call with dry_run=True to simulate just this operation
    await optimizer_override.optimize_template_cache(
        target_hit_rate=0.95, dry_run=True  # Override to dry-run for this call
    )

    print("✓ Used dry_run=True on single method call")
    print("✓ Overrode instance setting (dry_run=False)")

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
