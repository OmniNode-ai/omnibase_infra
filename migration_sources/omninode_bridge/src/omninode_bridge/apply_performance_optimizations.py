#!/usr/bin/env python3
"""
Apply Performance Optimizations to OmniNode Bridge

This script demonstrates how to integrate the performance optimizations
identified in the container restart loop fix analysis.
"""

import asyncio
import logging
import time
from typing import Optional

from performance_optimizations import (
    ConnectionPoolOptimizer,
    MemoryOptimizer,
    OptimizedConfigLoader,
    StartupOptimizer,
    optimize_postgres_client,
)
from workflow.coordinator import WorkflowCoordinator
from workflow.performance_coordinator_optimizations import (
    WorkflowPerformanceConfig,
    integrate_performance_optimizations,
    start_optimized_background_tasks,
    stop_optimized_background_tasks,
)

logger = logging.getLogger(__name__)


class OptimizedWorkflowCoordinator(WorkflowCoordinator):
    """Enhanced WorkflowCoordinator with integrated performance optimizations."""

    def __init__(self, config: Optional[dict] = None):
        # Record startup time
        self._startup_optimizer = StartupOptimizer()
        self._startup_optimizer.record_startup_milestone("init_start")

        # Use optimized config loading
        self._config_loader = OptimizedConfigLoader()

        # Initialize base coordinator
        super().__init__(config)

        # Integrate performance optimizations
        perf_config = WorkflowPerformanceConfig(
            max_concurrent_workflows=50,
            max_concurrent_tasks_per_workflow=10,
            workflow_execution_timeout=300.0,
            slow_workflow_threshold_ms=5000.0,
        )

        integrate_performance_optimizations(self, perf_config)

        # Initialize memory optimizer
        self._memory_optimizer = MemoryOptimizer()

        self._startup_optimizer.record_startup_milestone("init_complete")

    async def initialize(self):
        """Enhanced initialization with performance optimizations."""
        self._startup_optimizer.record_startup_milestone("enhanced_init_start")

        # Run base initialization
        await super().initialize()

        # Apply database optimizations
        if self.postgres_client:
            optimization_results = await optimize_postgres_client(self.postgres_client)
            logger.info("Database optimizations applied", extra=optimization_results)

        # Start optimized background tasks
        await start_optimized_background_tasks(self)

        # Take initial memory snapshot
        initial_snapshot = self._memory_optimizer.take_memory_snapshot(
            "post_initialization"
        )
        logger.info(
            "Initial memory snapshot taken",
            extra={
                "memory_mb": initial_snapshot.get("memory_info", {}).get("rss_mb", 0)
            },
        )

        self._startup_optimizer.record_startup_milestone("enhanced_init_complete")

    async def execute_workflow_optimized(self, request):
        """Execute workflow with performance optimizations."""
        workflow_id = request.workflow_definition.workflow_id

        # Use optimized execution with resource limits
        if hasattr(self, "_workflow_executor"):
            async with self._workflow_executor.execute_workflow_with_limits(
                workflow_id
            ):
                return await super().execute_workflow(request)
        else:
            # Fallback to standard execution
            return await super().execute_workflow(request)

    async def cleanup_optimized(self):
        """Enhanced cleanup with performance optimization cleanup."""
        # Stop optimized background tasks
        await stop_optimized_background_tasks(self)

        # Force memory cleanup
        if hasattr(self, "_memory_optimizer"):
            cleanup_results = await self._memory_optimizer.force_cleanup()
            logger.info("Memory cleanup completed", extra=cleanup_results)

        # Run base cleanup
        await super().cleanup()

    def get_performance_metrics(self):
        """Get comprehensive performance metrics."""
        base_metrics = super().get_performance_metrics()

        # Add optimized metrics if available
        if hasattr(self, "_performance_monitor"):
            optimized_metrics = self._performance_monitor.get_performance_summary()
            base_metrics.update(optimized_metrics)

        return base_metrics


async def demonstrate_performance_optimizations():
    """Demonstrate the performance optimizations in action."""
    print("🚀 OmniNode Bridge Performance Optimization Demo")
    print("=" * 50)

    # Create optimized coordinator
    print("📊 Creating optimized workflow coordinator...")
    coordinator = OptimizedWorkflowCoordinator()

    try:
        # Initialize with performance monitoring
        start_time = time.time()
        await coordinator.initialize()
        init_time = (time.time() - start_time) * 1000

        print(f"✅ Initialization completed in {init_time:.1f}ms")

        # Demonstrate configuration optimization
        print("\n🔧 Configuration Optimization:")
        config_loader = OptimizedConfigLoader()

        # Test cached environment variable access
        cache_start = time.time()
        for _ in range(100):
            config_loader.get_cached_env("POSTGRES_HOST", "localhost")
        cache_time = (time.time() - cache_start) * 1000
        print(f"   - 100 cached env var access: {cache_time:.2f}ms")

        # Demonstrate database optimization
        if coordinator.postgres_client:
            print("\n💾 Database Optimization:")
            pool_optimizer = ConnectionPoolOptimizer(coordinator.postgres_client)
            health_report = await pool_optimizer.monitor_connection_health()
            print(f"   - Connection health: {health_report['status']}")
            print(f"   - Issues found: {len(health_report.get('issues', []))}")

        # Demonstrate memory optimization
        print("\n🧠 Memory Optimization:")
        memory_optimizer = MemoryOptimizer()
        snapshot = memory_optimizer.take_memory_snapshot("demo")
        memory_mb = snapshot.get("memory_info", {}).get("rss_mb", 0)
        print(f"   - Current memory usage: {memory_mb:.1f}MB")

        # Get performance metrics
        print("\n📈 Performance Metrics:")
        metrics = coordinator.get_performance_metrics()
        for key, value in metrics.items():
            if isinstance(value, int | float):
                print(f"   - {key}: {value}")

        print("\n✨ Performance optimizations successfully demonstrated!")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        logger.error(f"Performance optimization demo failed: {e}")

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            await coordinator.cleanup_optimized()
            print("✅ Cleanup completed successfully")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


def create_performance_optimized_coordinator(
    config: Optional[dict] = None,
) -> OptimizedWorkflowCoordinator:
    """Factory function to create performance-optimized workflow coordinator.

    This is the main entry point for using the performance optimizations
    in production code.

    Args:
        config: Optional configuration dictionary

    Returns:
        OptimizedWorkflowCoordinator instance with all performance optimizations applied
    """
    return OptimizedWorkflowCoordinator(config)


async def benchmark_optimization_impact():
    """Benchmark the impact of performance optimizations."""
    print("🏁 Performance Optimization Benchmark")
    print("=" * 40)

    # Benchmark configuration loading
    print("Testing configuration loading performance...")

    # Standard approach
    import os

    standard_start = time.time()
    for _ in range(1000):
        _ = os.getenv("POSTGRES_HOST", "localhost")
    standard_time = (time.time() - standard_start) * 1000

    # Optimized approach
    config_loader = OptimizedConfigLoader()
    optimized_start = time.time()
    for _ in range(1000):
        _ = config_loader.get_cached_env("POSTGRES_HOST", "localhost")
    optimized_time = (time.time() - optimized_start) * 1000

    improvement = ((standard_time - optimized_time) / standard_time) * 100

    print(f"Standard config loading (1000 calls): {standard_time:.2f}ms")
    print(f"Optimized config loading (1000 calls): {optimized_time:.2f}ms")
    print(f"Performance improvement: {improvement:.1f}%")

    # Benchmark memory management
    print("\nTesting memory management...")
    memory_optimizer = MemoryOptimizer()

    gc_start = time.time()
    cleanup_results = await memory_optimizer.force_cleanup()
    gc_time = (time.time() - gc_start) * 1000

    print(f"Garbage collection completed in: {gc_time:.1f}ms")
    print(f"Objects collected: {cleanup_results.get('total_collected', 0)}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🔧 OmniNode Bridge Performance Optimization Integration")
    print("Choose an option:")
    print("1. Demonstrate optimizations")
    print("2. Run benchmark")
    print("3. Both")

    choice = input("Enter choice (1-3): ").strip()

    async def run_demo():
        if choice in ["1", "3"]:
            await demonstrate_performance_optimizations()
            print()

        if choice in ["2", "3"]:
            await benchmark_optimization_impact()

    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo execution failed: {e}")
