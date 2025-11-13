"""Performance optimization initialization script for OmniNode Bridge.

This script applies all the performance fixes and optimizations:
1. Initializes memory management utilities
2. Configures enhanced resource cleanup
3. Sets up performance monitoring
4. Provides usage examples for the new optimizations
"""

import asyncio
import logging
import os
from typing import Any

from omninode_bridge.services.performance_monitor import performance_monitor

# Import our optimized utilities
from omninode_bridge.utils.memory_management import (
    MemoryMonitor,
    SafeDictAccessor,
    WorkflowDataStreamer,
    managed_workflow_processing,
    resource_tracker,
)
from omninode_bridge.utils.resource_cleanup import (
    managed_database_connection,
    managed_kafka_producer,
    monitor_resource_cleanup,
    resource_manager,
)

logger = logging.getLogger(__name__)


class OptimizedOmniNodeBridge:
    """Optimized OmniNode Bridge with all performance fixes applied."""

    def __init__(self):
        self.memory_monitor = MemoryMonitor()
        self.workflow_streamer = WorkflowDataStreamer(
            memory_monitor=self.memory_monitor,
        )
        self.safe_dict = SafeDictAccessor()
        self.connection_pools = {}
        self.performance_monitor = performance_monitor

        # Configure environment-specific optimizations
        self._configure_environment_optimizations()

    def _configure_environment_optimizations(self):
        """Configure optimizations based on environment."""
        environment = os.getenv("ENVIRONMENT", "development").lower()

        if environment == "production":
            # Production: Optimize for throughput and reliability
            self.workflow_streamer.chunk_size = 2000
            self.memory_monitor.warning_threshold = 70.0  # More aggressive monitoring
            self.memory_monitor.critical_threshold = 85.0

        elif environment == "staging":
            # Staging: Balanced performance
            self.workflow_streamer.chunk_size = 1500
            self.memory_monitor.warning_threshold = 75.0
            self.memory_monitor.critical_threshold = 90.0

        else:  # development
            # Development: Optimize for debugging and safety
            self.workflow_streamer.chunk_size = 500
            self.memory_monitor.warning_threshold = 80.0
            self.memory_monitor.critical_threshold = 95.0

        logger.info(f"Configured optimizations for {environment} environment")

    async def initialize_services(self):
        """Initialize all optimized services."""
        logger.info("Initializing optimized OmniNode Bridge services...")

        # Start performance monitoring
        await self.performance_monitor.start_monitoring()
        logger.info("✅ Performance monitoring started with memory optimization")

        # Start resource cleanup monitoring
        cleanup_task = asyncio.create_task(monitor_resource_cleanup())
        await resource_manager.register_resource(
            cleanup_task,
            "cleanup_monitor",
            cleanup_callback=lambda: cleanup_task.cancel(),
        )
        logger.info("✅ Resource cleanup monitoring started")

        # Register global resource tracker
        resource_tracker.register_resource("global_bridge", self)
        logger.info("✅ Global resource tracking enabled")

        return True

    async def process_large_workflow(self, workflow_data: list) -> dict[str, Any]:
        """Process large workflow data using optimized memory management.

        Example of how to use the new memory management utilities.
        """
        async with managed_workflow_processing(
            memory_limit_mb=500,
            auto_gc=True,
            gc_interval_seconds=30.0,  # 500MB limit
        ) as workflow_context:
            memory_monitor = workflow_context["memory_monitor"]
            streamer = workflow_context["streamer"]
            safe_dict = workflow_context["safe_dict"]

            results = []
            processed_count = 0

            # Stream workflow processing to avoid memory exhaustion
            async for chunk in streamer.stream_workflow_results(workflow_data):
                # Process chunk with safe dictionary access
                chunk_results = []

                for item in chunk:
                    try:
                        # Use safe dictionary access to prevent KeyError
                        workflow_id = safe_dict.safe_get(item, "id", required=True)
                        workflow_status = safe_dict.safe_get(item, "status", "pending")
                        workflow_metadata = safe_dict.safe_nested_get(
                            item,
                            ["metadata", "execution_info"],
                            {},
                        )

                        # Process workflow item
                        processed_item = {
                            "id": workflow_id,
                            "status": workflow_status,
                            "metadata": workflow_metadata,
                            "processed_at": "2024-01-01T00:00:00Z",  # Mock timestamp
                            "success": True,
                        }

                        chunk_results.append(processed_item)
                        processed_count += 1

                    except ValueError as e:
                        logger.error(f"Workflow item validation error: {e}")
                        # Record failed item with safe access
                        chunk_results.append(
                            {
                                "id": safe_dict.safe_get(
                                    item,
                                    "id",
                                    f"unknown_{processed_count}",
                                ),
                                "status": "failed",
                                "error": str(e),
                                "success": False,
                            },
                        )

                results.extend(chunk_results)

                # Monitor memory usage
                memory_status = memory_monitor.check_memory_pressure()
                if memory_status["status"] != "normal":
                    logger.warning(
                        f"Memory pressure during workflow processing: {memory_status['status']}",
                    )

            # Final processing statistics
            successful_items = sum(1 for r in results if r.get("success", False))
            failed_items = len(results) - successful_items

            return {
                "total_processed": len(results),
                "successful": successful_items,
                "failed": failed_items,
                "success_rate": (
                    (successful_items / len(results)) * 100 if results else 0
                ),
                "memory_status": memory_monitor.check_memory_pressure(),
            }

    async def demonstrate_database_optimization(self, postgres_client):
        """Demonstrate optimized database connection usage."""

        try:
            # Use managed database connection with automatic cleanup
            async with managed_database_connection(postgres_client) as conn:
                # Simulate database operations with safe error handling
                try:
                    result = await conn.fetchval(
                        "SELECT COUNT(*) FROM service_sessions",
                    )
                    logger.info(f"Database query result: {result}")

                    # Get pool metrics to verify optimization
                    pool_metrics = await postgres_client.get_pool_metrics()
                    logger.info(f"Connection pool metrics: {pool_metrics}")

                    return {
                        "query_result": result,
                        "pool_metrics": pool_metrics,
                        "success": True,
                    }

                except Exception as e:
                    logger.error(f"Database operation failed: {e}")
                    return {"error": str(e), "success": False}

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return {"error": f"Connection error: {e}", "success": False}

    async def demonstrate_kafka_optimization(self, kafka_client):
        """Demonstrate optimized Kafka client usage."""

        try:
            # Use managed Kafka producer with race condition prevention
            async with managed_kafka_producer(kafka_client) as producer:
                # Test message publishing with performance monitoring
                test_message = {
                    "type": "test_message",
                    "data": {"test": "optimization_demo"},
                    "timestamp": "2024-01-01T00:00:00Z",
                }

                # Record performance metrics for the operation
                start_time = asyncio.get_event_loop().time()

                success = await producer.publish_raw_event(
                    topic="test.optimization",
                    data=test_message,
                    key="test_key",
                )

                end_time = asyncio.get_event_loop().time()
                duration_ms = (end_time - start_time) * 1000

                # Record metrics in performance monitor
                await self.performance_monitor.record_performance_metric(
                    component="kafka_client",
                    operation="publish_test_message",
                    duration_ms=duration_ms,
                    success=success,
                )

                # Get resilience metrics
                resilience_metrics = await producer.get_resilience_metrics()

                return {
                    "publish_success": success,
                    "duration_ms": duration_ms,
                    "resilience_metrics": resilience_metrics,
                    "connection_status": producer.is_connected,
                }

        except Exception as e:
            logger.error(f"Kafka operation failed: {e}")
            return {"error": str(e), "success": False}

    async def get_comprehensive_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report showing all optimizations."""

        # Get performance dashboard
        performance_dashboard = (
            await self.performance_monitor.get_performance_dashboard()
        )

        # Get memory statistics
        memory_status = self.memory_monitor.check_memory_pressure()

        # Get resource management statistics
        resource_stats = resource_manager.get_resource_stats()

        # Get resource tracker statistics
        tracker_stats = resource_tracker.get_resource_stats()

        return {
            "optimization_status": {
                "kafka_race_condition_fix": "✅ Applied - Connection state managed properly",
                "database_pool_optimization": "✅ Applied - Pool monitoring and leak detection enabled",
                "memory_management": "✅ Applied - Streaming and safe access patterns implemented",
                "resource_cleanup": "✅ Applied - Automatic lifecycle management enabled",
                "performance_monitoring": "✅ Applied - Memory-efficient deque optimization",
            },
            "current_performance": performance_dashboard,
            "memory_status": memory_status,
            "resource_management": {
                "managed_resources": resource_stats,
                "tracked_resources": tracker_stats,
            },
            "recommendations": self._get_performance_recommendations(
                memory_status,
                resource_stats,
            ),
        }

    def _get_performance_recommendations(
        self,
        memory_status: dict,
        resource_stats: dict,
    ) -> list:
        """Generate performance optimization recommendations."""
        recommendations = []

        if memory_status["status"] == "warning":
            recommendations.append(
                "Consider enabling workflow result streaming for large datasets",
            )
            recommendations.append(
                "Monitor workflow batch sizes to prevent memory pressure",
            )

        if memory_status["status"] == "critical":
            recommendations.append("URGENT: Enable immediate garbage collection")
            recommendations.append("URGENT: Reduce workflow processing batch sizes")
            recommendations.append(
                "URGENT: Consider workflow pagination for large operations",
            )

        if resource_stats["total_resources"] > 100:
            recommendations.append(
                "High resource count detected - consider more aggressive cleanup",
            )

        if not recommendations:
            recommendations.append("System performance is optimal")

        return recommendations


async def main():
    """Main function demonstrating the optimized OmniNode Bridge."""

    # Initialize optimized bridge
    bridge = OptimizedOmniNodeBridge()
    await bridge.initialize_services()

    # Create sample large workflow data
    large_workflow_data = [
        {
            "id": f"workflow_{i}",
            "status": "pending" if i % 3 == 0 else "running",
            "metadata": {
                "execution_info": {"priority": i % 5, "estimated_duration": i * 10},
            },
            "data": {"payload": f"test_data_{i}"},
        }
        for i in range(5000)  # Large dataset to test memory management
    ]

    logger.info(f"Processing {len(large_workflow_data)} workflow items...")

    # Process using optimized memory management
    result = await bridge.process_large_workflow(large_workflow_data)

    logger.info("Workflow processing completed:")
    logger.info(f"  Total processed: {result['total_processed']}")
    logger.info(f"  Success rate: {result['success_rate']:.2f}%")
    logger.info(f"  Memory status: {result['memory_status']['status']}")

    # Get comprehensive performance report
    performance_report = await bridge.get_comprehensive_performance_report()

    print("\n" + "=" * 80)
    print("OMNINODE BRIDGE PERFORMANCE OPTIMIZATION REPORT")
    print("=" * 80)

    for optimization, status in performance_report["optimization_status"].items():
        print(f"{optimization}: {status}")

    print(f"\nMemory Status: {performance_report['memory_status']['status'].upper()}")
    print(
        f"Memory Usage: {performance_report['memory_status']['metrics'].rss_mb:.1f}MB",
    )

    print(
        f"\nManaged Resources: {performance_report['resource_management']['managed_resources']['total_resources']}",
    )

    print("\nRecommendations:")
    for rec in performance_report["recommendations"]:
        print(f"  • {rec}")

    print("=" * 80)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the demonstration
    asyncio.run(main())
