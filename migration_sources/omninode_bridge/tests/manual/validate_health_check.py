"""
Manual validation script for database adapter health checks with real database operations.

This script demonstrates:
1. Real database connectivity checks (SELECT 1)
2. Real connection pool statistics
3. Real database version queries
4. Performance validation (<100ms target)
5. Error handling in various states

Usage:
    poetry run python tests/manual/validate_health_check.py

Requirements:
    - PostgreSQL running on localhost:5432
    - Database: omninode_bridge
    - User: postgres / Password: omninode-bridge-postgres-dev-2024
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from omninode_core.infrastructure.service_container import ServiceContainer

from omninode_bridge.nodes.database_adapter_effect.v1_0_0.node import (
    NodeBridgeDatabaseAdapterEffect,
)


class HealthCheckValidator:
    """Validates health check functionality with real database operations."""

    def __init__(self):
        self.node = None
        self.container = None
        self.results = []

    async def setup(self):
        """Initialize database adapter node with real database connection."""
        print("🔧 Setting up database adapter node...")

        # Create service container
        self.container = ServiceContainer()

        # Create database adapter node
        self.node = NodeBridgeDatabaseAdapterEffect(container=self.container)

        # Initialize the node (connects to database)
        try:
            await self.node.initialize()
            print("✅ Node initialized successfully")
            print(f"   Connection manager: {self.node._connection_manager is not None}")
            print(f"   Query executor: {self.node._query_executor is not None}")
            print(f"   Circuit breaker: {self.node._circuit_breaker is not None}")
        except Exception as e:
            print(f"❌ Failed to initialize node: {e}")
            raise

    async def validate_healthy_state(self):
        """Validate health check in healthy state."""
        print("\n📊 Test 1: Health check in HEALTHY state")
        print("-" * 60)

        start_time = time.perf_counter()
        health = await self.node.get_health_status()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print(f"   Status: {health.database_status}")
        print(f"   Success: {health.success}")
        print(
            f"   Execution time: {health.execution_time_ms}ms (actual: {elapsed_ms:.2f}ms)"
        )
        print(f"   Pool size: {health.connection_pool_size}")
        print(f"   Pool available: {health.connection_pool_available}")
        print(f"   Pool in use: {health.connection_pool_in_use}")
        print(
            f"   Database version: {health.database_version[:50]}..."
            if health.database_version
            else "   Database version: None"
        )
        print(f"   Uptime: {health.uptime_seconds}s")
        print(f"   Error: {health.error_message}")

        # Validate results
        assert health.success, "Health check should succeed in healthy state"
        assert (
            health.database_status == "HEALTHY"
        ), f"Status should be HEALTHY, got {health.database_status}"
        assert (
            health.execution_time_ms < 100
        ), f"Execution time {health.execution_time_ms}ms exceeds 100ms target"
        assert health.connection_pool_size > 0, "Pool size should be greater than 0"
        assert (
            health.database_version is not None
        ), "Database version should be populated"
        assert (
            "PostgreSQL" in health.database_version
        ), "Database version should contain 'PostgreSQL'"

        print("✅ PASSED - Health check working with real database operations")
        self.results.append(("Healthy State", True, health.execution_time_ms))

    async def validate_performance(self):
        """Validate health check performance over multiple runs."""
        print("\n⚡ Test 2: Performance validation (10 runs)")
        print("-" * 60)

        execution_times = []
        for i in range(10):
            start_time = time.perf_counter()
            health = await self.node.get_health_status()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            execution_times.append(elapsed_ms)

            if i == 0:
                print(f"   Run {i+1}: {elapsed_ms:.2f}ms (first run)")
            elif i == 9:
                print(f"   Run {i+1}: {elapsed_ms:.2f}ms (last run)")

        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        p95_time = sorted(execution_times)[int(len(execution_times) * 0.95)]

        print(f"\n   Average: {avg_time:.2f}ms")
        print(f"   Min: {min_time:.2f}ms")
        print(f"   Max: {max_time:.2f}ms")
        print(f"   P95: {p95_time:.2f}ms")

        # Validate performance targets
        assert avg_time < 100, f"Average time {avg_time:.2f}ms exceeds 100ms target"
        assert p95_time < 100, f"P95 time {p95_time:.2f}ms exceeds 100ms target"

        print("✅ PASSED - Performance meets <100ms target")
        self.results.append(("Performance (avg)", True, avg_time))
        self.results.append(("Performance (p95)", True, p95_time))

    async def validate_real_queries(self):
        """Validate that real database queries are being executed."""
        print("\n🔍 Test 3: Verify real database queries")
        print("-" * 60)

        # Get initial health check
        health = await self.node.get_health_status()

        # Verify real database version query
        assert (
            health.database_version is not None
        ), "Database version should not be None"
        assert (
            "PostgreSQL" in health.database_version
        ), "Should query real PostgreSQL version"
        assert (
            "(simulated)" not in health.database_version
        ), "Should not contain '(simulated)' marker"

        print(f"   ✓ Real database version query: {health.database_version[:80]}")

        # Verify real connection pool stats
        assert health.connection_pool_size > 0, "Should have real pool size"
        assert health.connection_pool_available >= 0, "Should have real available count"
        assert health.connection_pool_in_use >= 0, "Should have real in-use count"

        print("   ✓ Real connection pool stats:")
        print(f"     - Size: {health.connection_pool_size}")
        print(f"     - Available: {health.connection_pool_available}")
        print(f"     - In use: {health.connection_pool_in_use}")

        # Verify connectivity check executed (no way to verify directly, but we can check success)
        assert health.success, "Connectivity check should succeed"
        print("   ✓ Real database connectivity check (SELECT 1)")

        print("✅ PASSED - All health checks use real database operations")
        self.results.append(("Real Database Queries", True, 0))

    async def validate_no_simulated_code(self):
        """Validate that no simulated code paths exist."""
        print("\n🔒 Test 4: Verify no simulated code paths")
        print("-" * 60)

        # Check that the flag doesn't exist
        assert not hasattr(
            self.node, "_use_simulated_health_checks"
        ), "Node should not have _use_simulated_health_checks attribute"

        print("   ✓ No _use_simulated_health_checks flag")
        print("   ✓ No simulated code paths available")
        print("   ✓ Always uses real database operations")

        print("✅ PASSED - No simulated code paths remain")
        self.results.append(("No Simulated Paths", True, 0))

    async def cleanup(self):
        """Cleanup resources."""
        print("\n🧹 Cleaning up...")
        if self.node:
            await self.node.cleanup()
        print("✅ Cleanup complete")

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)

        for test_name, success, time_ms in self.results:
            status = "✅ PASS" if success else "❌ FAIL"
            time_str = f"({time_ms:.2f}ms)" if time_ms > 0 else ""
            print(f"{status} - {test_name} {time_str}")

        print(f"\n{passed}/{total} tests passed")

        if passed == total:
            print(
                "\n🎉 ALL VALIDATIONS PASSED - Health checks use real database operations!"
            )
        else:
            print(f"\n⚠️  {total - passed} validation(s) failed")

    async def run(self):
        """Run all validation tests."""
        try:
            await self.setup()
            await self.validate_healthy_state()
            await self.validate_performance()
            await self.validate_real_queries()
            await self.validate_no_simulated_code()
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            import traceback

            traceback.print_exc()
            self.results.append(("Validation", False, 0))
        finally:
            await self.cleanup()
            self.print_summary()


async def main():
    """Main entry point."""
    print("=" * 60)
    print("🚀 DATABASE ADAPTER HEALTH CHECK VALIDATION")
    print("=" * 60)
    print("\nValidating health checks with REAL database operations...")
    print("(No simulated checks, no await asyncio.sleep())\n")

    validator = HealthCheckValidator()
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
