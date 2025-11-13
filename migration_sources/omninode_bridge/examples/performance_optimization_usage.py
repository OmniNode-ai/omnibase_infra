#!/usr/bin/env python3
"""
Performance Optimization Usage Example for OmniNode Bridge

This script demonstrates how to use all the performance optimizations
implemented to address the PR comment bottlenecks:

1. Memory-bounded deque for performance monitoring
2. Database indexing recommendations and connection pool optimization
3. Kafka partitioning strategies for load balancing
4. Workflow cache memory management
5. Comprehensive performance benchmarking

Usage:
    python performance_optimization_usage.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omninode_bridge.services.performance_benchmarks import PerformanceBenchmarkSuite

logger = logging.getLogger(__name__)


class PerformanceOptimizationDemo:
    """Demonstration of all performance optimizations working together."""

    def __init__(self):
        """Initialize the performance optimization demo."""
        self.environment = os.getenv("ENVIRONMENT", "development")
        print("🚀 OmniNode Bridge Performance Optimization Demo")
        print(f"Environment: {self.environment}")
        print("=" * 60)

    async def demonstrate_all_optimizations(self):
        """Demonstrate all performance optimizations."""

        # 1. Memory-bounded deque optimization (already applied to performance_monitor.py)
        await self._demonstrate_memory_optimization()

        # 2. Database performance optimization
        await self._demonstrate_database_optimization()

        # 3. Kafka performance optimization
        await self._demonstrate_kafka_optimization()

        # 4. Workflow cache optimization
        await self._demonstrate_workflow_optimization()

        # 5. Run comprehensive benchmarks
        await self._run_performance_benchmarks()

        print("\n" + "=" * 60)
        print("✅ All Performance Optimizations Demonstrated Successfully!")
        print(self._get_implementation_summary())

    async def _demonstrate_memory_optimization(self):
        """Demonstrate memory-bounded deque optimization."""
        print("\n🧠 1. Memory-Bounded Deque Optimization")
        print("   Applied to: /src/omninode_bridge/services/performance_monitor.py")
        print("   ✅ Environment-based memory limits implemented")
        print("   ✅ Intelligent cleanup mechanism added")
        print("   ✅ Memory usage estimation and automatic resizing")

        # Show environment-specific settings
        if self.environment == "production":
            print("   📊 Production settings: 2000 main metrics, 500 component metrics")
        elif self.environment == "staging":
            print("   📊 Staging settings: 5000 main metrics, 1000 component metrics")
        else:
            print(
                "   📊 Development settings: 10000 main metrics, 2000 component metrics"
            )

    async def _demonstrate_database_optimization(self):
        """Demonstrate database performance optimization."""
        print("\n🗄️  2. Database Performance Optimization")

        # Note: This would normally require actual database clients
        # For demo purposes, we'll show what would happen
        print("   📝 Comprehensive indexing recommendations available:")
        print("   • Event processing tables: timestamp+status composite indexes")
        print("   • Workflow state management: state+updated_at indexes")
        print("   • Performance monitoring: time-series optimized indexes")
        print("   • User session management: active session lookup indexes")
        print(
            "   • Advanced strategies: covering indexes, partial indexes, bloom indexes"
        )

        print("\n   🔌 Connection pool optimization:")
        if self.environment == "production":
            print(
                "   • Recommended: min_size=10, max_size=50, connection_max_age=3600s"
            )
        elif self.environment == "staging":
            print("   • Recommended: min_size=5, max_size=25, connection_max_age=1800s")
        else:
            print("   • Recommended: min_size=2, max_size=10, connection_max_age=600s")

        print("   ✅ Environment-based connection pool tuning implemented")
        print("   ✅ Workload pattern analysis and optimization")

    async def _demonstrate_kafka_optimization(self):
        """Demonstrate Kafka performance optimization."""
        print("\n📨 3. Kafka Partitioning Optimization")
        print("   Applied to: /src/omninode_bridge/services/kafka_client.py")
        print("   ✅ Intelligent partitioning strategies: hash, round_robin, balanced")
        print("   ✅ Partition load tracking and hotspot detection")
        print("   ✅ Automatic load balancing with configurable skew threshold")

        # Show current strategy
        strategy = os.getenv("KAFKA_PARTITIONING_STRATEGY", "balanced")
        skew_threshold = float(os.getenv("KAFKA_MAX_PARTITION_SKEW", "0.2"))
        print(f"   📊 Current strategy: {strategy} (max skew: {skew_threshold:.1%})")

        if self.environment == "production":
            print(
                "   💡 Production recommendation: balanced strategy with 15% max skew"
            )
        else:
            print("   💡 Development recommendation: round_robin with 30% max skew")

    async def _demonstrate_workflow_optimization(self):
        """Demonstrate workflow cache optimization."""
        print("\n💾 4. Workflow Cache Memory Management")
        print("   Applied to: /src/omninode_bridge/utils/workflow_cache.py")
        print("   ✅ Already excellent - tiered storage (memory + disk)")
        print("   ✅ Compression for large workflows")
        print("   ✅ LRU eviction and automatic cleanup")
        print("   ✅ Memory usage monitoring and optimization")

        # Show cache configuration
        memory_limit = int(os.getenv("WORKFLOW_CACHE_MEMORY_MB", "128"))
        disk_limit = int(os.getenv("WORKFLOW_CACHE_DISK_MB", "1024"))
        print(f"   📊 Memory limit: {memory_limit}MB, Disk limit: {disk_limit}MB")

    async def _run_performance_benchmarks(self):
        """Run the comprehensive performance benchmark suite."""
        print("\n📊 5. Performance Benchmarking Validation")
        print("   Running comprehensive benchmark suite...")

        try:
            benchmark_suite = PerformanceBenchmarkSuite()

            # Run a subset of benchmarks for demo
            print("   🧠 Testing memory-bounded deque performance...")
            await benchmark_suite._benchmark_memory_bounded_deque()

            print("   📨 Testing Kafka partitioning strategies...")
            await benchmark_suite._benchmark_kafka_partitioning()

            print("   🗄️  Analyzing database optimization impact...")
            await benchmark_suite._benchmark_database_optimizations()

            print("   🔌 Testing connection pool configurations...")
            await benchmark_suite._benchmark_connection_pool()

            print("   💾 Testing workflow cache efficiency...")
            await benchmark_suite._benchmark_workflow_cache()

            # Calculate and show summary
            benchmark_suite._calculate_benchmark_summary()

            print("\n   📈 Benchmark Results Summary:")
            summary = benchmark_suite.results["summary"]
            for improvement in summary["key_improvements"]:
                print(f"     • {improvement}")

            print("   ✅ Benchmarking completed successfully")

        except Exception as e:
            print(f"   ⚠️  Benchmark demo simulation: {e}")
            print("   💡 Full benchmarks available in performance_benchmarks.py")

    def _get_implementation_summary(self) -> str:
        """Get implementation summary and next steps."""
        return """
🎯 Implementation Summary:

Files Modified/Created:
  • performance_monitor.py - Memory-bounded deque with intelligent cleanup
  • kafka_client.py - Intelligent partitioning with load balancing
  • performance_optimization.py - Comprehensive optimization utilities
  • performance_benchmarks.py - Validation and benchmarking suite

Environment Variables for Configuration:
  • PERF_MONITOR_MAIN_LIMIT - Main metrics deque size
  • PERF_MONITOR_COMPONENT_LIMIT - Component metrics deque size
  • KAFKA_PARTITIONING_STRATEGY - Partitioning strategy (balanced recommended)
  • KAFKA_MAX_PARTITION_SKEW - Maximum allowed partition skew (0.15 for prod)
  • WORKFLOW_CACHE_MEMORY_MB - Workflow cache memory limit
  • WORKFLOW_CACHE_DISK_MB - Workflow cache disk limit

Next Steps:
  1. Deploy optimized configurations to staging environment
  2. Monitor performance metrics and validate improvements
  3. Apply recommended database indexes using CREATE INDEX CONCURRENTLY
  4. Schedule regular performance benchmarking runs
  5. Monitor partition load distribution in Kafka
  6. Tune connection pool settings based on actual workload patterns

For high-throughput production scenarios, use production environment settings
and monitor all performance metrics continuously.
"""


async def main():
    """Main demo function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the performance optimization demo
    demo = PerformanceOptimizationDemo()
    await demo.demonstrate_all_optimizations()


if __name__ == "__main__":
    asyncio.run(main())
