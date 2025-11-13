#!/usr/bin/env python3
"""
Usage examples for NodeIntrospectionMixin integration.

Demonstrates how to integrate introspection capabilities into:
1. NodeBridgeOrchestrator (with health checks and FSM states)
2. NodeBridgeReducer (with aggregation metrics)
3. Custom node implementations

ONEX v2.0 Compliance:
- OnexEnvelopeV1 integration for event publishing
- Kafka topic routing with namespace support
- Correlation ID tracking for related events
"""

import asyncio
from typing import Any
from uuid import uuid4

# Example 1: NodeBridgeOrchestrator with Introspection
# =====================================================


async def example_orchestrator_with_introspection():
    """
    Example: Integrating NodeIntrospectionMixin into NodeBridgeOrchestrator.

    This example demonstrates:
    1. Mixin integration with multiple inheritance
    2. Initialization sequence
    3. Startup introspection broadcasting
    4. Periodic heartbeat broadcasting
    5. Manual introspection triggers
    """
    # Import required components - omnibase_core is required
    from omnibase_core.models.core import ModelContainer

    # Note: This is a conceptual example - actual integration would modify node.py
    class NodeBridgeOrchestratorWithIntrospection:
        """
        Example orchestrator with introspection capabilities.

        Integrates:
        - NodeOrchestrator (base class)
        - HealthCheckMixin (component health monitoring)
        - NodeIntrospectionMixin (capability broadcasting)
        """

        def __init__(self, container: ModelContainer):
            """Initialize orchestrator with all capabilities."""
            # Initialize base classes (would use super() in real implementation)
            self.container = container
            self.node_id = str(uuid4())

            # Configure services
            self.metadata_stamping_service_url = container.config.get(
                "metadata_stamping_service_url", "http://metadata-stamping:8053"
            )
            self.onextree_service_url = container.config.get(
                "onextree_service_url", "http://onextree:8080"
            )
            self.kafka_broker_url = container.config.get(
                "kafka_broker_url", "localhost:9092"
            )

            # FSM state tracking for introspection
            self.workflow_fsm_states = {}
            self.stamping_metrics = {}

            # Initialize health checks (from HealthCheckMixin)
            # self.initialize_health_checks()

            # Initialize introspection system
            # self.initialize_introspection()

            print(f"Orchestrator initialized: {self.node_id}")

        async def startup(self):
            """
            Node startup sequence with introspection.

            1. Broadcast initial introspection (reason: startup)
            2. Start periodic heartbeat broadcasting (every 30s)
            3. Start registry request listener
            4. Begin accepting workflow requests
            """
            print("\n=== Orchestrator Startup Sequence ===")

            # Step 1: Broadcast startup introspection
            # await self.publish_introspection(reason="startup")
            print("✓ Published startup introspection to event bus")

            # Step 2: Start background introspection tasks
            # await self.start_introspection_tasks(
            #     enable_heartbeat=True,
            #     heartbeat_interval_seconds=30,
            #     enable_registry_listener=True
            # )
            print("✓ Started heartbeat broadcasting (30s interval)")
            print("✓ Started registry request listener")

            # Step 3: Begin accepting requests
            print("✓ Node ready to accept orchestration requests")

        async def handle_registry_request(self, correlation_id):
            """
            Handle registry request by re-broadcasting introspection.

            Args:
                correlation_id: Correlation ID from registry request
            """
            print(
                f"\n=== Handling Registry Request (correlation: {correlation_id}) ==="
            )

            # Re-broadcast introspection with registry correlation_id
            # await self.publish_introspection(
            #     reason="registry_request",
            #     correlation_id=correlation_id
            # )
            print("✓ Responded to registry request with introspection data")

        async def shutdown(self):
            """
            Node shutdown sequence.

            1. Stop accepting new workflows
            2. Complete in-flight workflows
            3. Stop introspection tasks (heartbeat, registry listener)
            4. Broadcast final status (optional)
            """
            print("\n=== Orchestrator Shutdown Sequence ===")

            # Step 1: Stop introspection tasks
            # await self.stop_introspection_tasks()
            print("✓ Stopped heartbeat broadcasting")
            print("✓ Stopped registry request listener")

            # Step 2: Final introspection broadcast (optional)
            # await self.publish_introspection(reason="shutdown")
            print("✓ Published final introspection (shutdown)")

            print("✓ Node shutdown complete")

    # Run example
    container = ModelContainer(
        config={
            "metadata_stamping_service_url": "http://metadata-stamping:8053",
            "onextree_service_url": "http://onextree:8080",
            "kafka_broker_url": "localhost:9092",
            "environment": "development",
        }
    )

    node = NodeBridgeOrchestratorWithIntrospection(container)
    await node.startup()

    # Simulate registry request
    await node.handle_registry_request(correlation_id=uuid4())

    # Simulate shutdown
    await asyncio.sleep(1)
    await node.shutdown()


# Example 2: NodeBridgeReducer with Introspection
# ================================================


async def example_reducer_with_introspection():
    """
    Example: Integrating NodeIntrospectionMixin into NodeBridgeReducer.

    This example demonstrates:
    1. Reducer-specific capability extraction (aggregation metrics)
    2. Namespace-based introspection
    3. State management introspection
    """
    # Import required components - omnibase_core is required
    from omnibase_core.models.core import ModelContainer

    class NodeBridgeReducerWithIntrospection:
        """
        Example reducer with introspection capabilities.

        Integrates:
        - NodeReducer (base class)
        - HealthCheckMixin (component health monitoring)
        - NodeIntrospectionMixin (capability broadcasting)
        """

        def __init__(self, container: ModelContainer):
            """Initialize reducer with all capabilities."""
            self.container = container
            self.node_id = str(uuid4())

            # Reducer-specific state
            self.aggregation_state = {}
            self.namespace_groups = {}

            # Performance metrics for introspection
            self.aggregation_metrics = {
                "total_items_aggregated": 0,
                "aggregation_throughput_per_second": 1000,
                "avg_aggregation_time_ms": 0.5,
            }

            # Initialize introspection
            # self.initialize_introspection()

            print(f"Reducer initialized: {self.node_id}")

        async def startup(self):
            """Reducer startup with introspection broadcasting."""
            print("\n=== Reducer Startup Sequence ===")

            # Broadcast startup introspection
            # await self.publish_introspection(reason="startup")
            print("✓ Published startup introspection to event bus")

            # Start heartbeat broadcasting
            # await self.start_introspection_tasks(
            #     enable_heartbeat=True,
            #     heartbeat_interval_seconds=30
            # )
            print("✓ Started heartbeat broadcasting (30s interval)")

        async def get_reducer_capabilities(self) -> dict[str, Any]:
            """
            Get reducer-specific capabilities for introspection.

            Returns:
                Dictionary of reducer capabilities
            """
            # This would be called by extract_capabilities() from the mixin
            return {
                "streaming_aggregation": {
                    "throughput_items_per_second": self.aggregation_metrics[
                        "aggregation_throughput_per_second"
                    ],
                    "window_size_ms": 5000,
                    "batch_size": 100,
                },
                "namespace_grouping": {
                    "enabled": True,
                    "active_namespaces": len(self.namespace_groups),
                },
                "state_persistence": {
                    "backend": "postgresql",
                    "state_size_bytes": len(str(self.aggregation_state)),
                },
                "performance_metrics": self.aggregation_metrics,
            }

    # Run example
    container = ModelContainer(
        config={
            "environment": "development",
            "kafka_broker_url": "localhost:9092",
        }
    )

    node = NodeBridgeReducerWithIntrospection(container)
    await node.startup()

    # Get capabilities
    capabilities = await node.get_reducer_capabilities()
    print(f"\nReducer capabilities: {capabilities}")


# Example 3: Manual Introspection Triggers
# =========================================


async def example_manual_introspection_triggers():
    """
    Example: Manual introspection broadcasting scenarios.

    Demonstrates when to manually trigger introspection:
    1. After major capability changes
    2. After configuration updates
    3. On-demand for debugging
    4. In response to registry requests
    """
    print("\n=== Manual Introspection Triggers ===")

    # Scenario 1: After capability change
    print("\n1. After capability change (new feature enabled):")
    # await node.publish_introspection(reason="capability_change")
    print("   ✓ Broadcasted updated capabilities")

    # Scenario 2: After configuration update
    print("\n2. After configuration update (resource limits changed):")
    # await node.publish_introspection(reason="config_update")
    print("   ✓ Broadcasted updated configuration")

    # Scenario 3: On-demand for debugging
    print("\n3. On-demand introspection for debugging:")
    # await node.publish_introspection(reason="debug")
    print("   ✓ Broadcasted current state for debugging")

    # Scenario 4: Registry request response
    print("\n4. Registry request response:")
    registry_correlation_id = uuid4()
    # await node.publish_introspection(
    #     reason="registry_request",
    #     correlation_id=registry_correlation_id
    # )
    print(f"   ✓ Responded to registry (correlation: {registry_correlation_id})")


# Example 4: Capability Extraction Customization
# ==============================================


async def example_custom_capability_extraction():
    """
    Example: Customizing capability extraction for specific nodes.

    Demonstrates how to extend extract_capabilities() for custom needs.
    """

    class CustomNodeWithIntrospection:
        """Custom node with specialized capability extraction."""

        def __init__(self):
            self.node_id = str(uuid4())
            self.custom_features = {
                "advanced_caching": True,
                "ml_inference": False,
                "batch_processing": True,
            }
            self.performance_tier = "high"

        async def extract_capabilities(self) -> dict[str, Any]:
            """
            Override capability extraction to include custom features.

            Returns:
                Dictionary with both standard and custom capabilities
            """
            # Get base capabilities from mixin
            # capabilities = await super().extract_capabilities()

            # Simulate base capabilities
            capabilities = {
                "node_type": "custom",
                "node_version": "1.0.0",
            }

            # Add custom features
            capabilities["custom_features"] = self.custom_features
            capabilities["performance_tier"] = self.performance_tier

            # Add business-specific metrics
            capabilities["business_metrics"] = {
                "uptime_percentage": 99.9,
                "requests_processed_today": 150000,
                "average_response_time_ms": 45,
            }

            return capabilities

    print("\n=== Custom Capability Extraction ===")
    node = CustomNodeWithIntrospection()
    capabilities = await node.extract_capabilities()
    print(f"Custom capabilities: {capabilities}")


# Example 5: Integration with Health Checks
# =========================================


async def example_introspection_with_health_integration():
    """
    Example: Integration between introspection and health checks.

    Demonstrates:
    1. Including health status in introspection
    2. Broadcasting introspection on health changes
    3. Coordinating health checks with heartbeat
    """
    print("\n=== Introspection with Health Integration ===")

    # Scenario 1: Healthy node broadcasting
    print("\n1. Healthy node introspection:")
    print("   Status: HEALTHY")
    print("   Components: all healthy")
    # await node.publish_introspection(reason="health_check")
    print("   ✓ Broadcasted healthy status")

    # Scenario 2: Degraded node (non-critical component down)
    print("\n2. Degraded node introspection:")
    print("   Status: DEGRADED")
    print("   Components: onextree unavailable (non-critical)")
    # await node.publish_introspection(reason="health_degraded")
    print("   ✓ Broadcasted degraded status")

    # Scenario 3: Recovery to healthy
    print("\n3. Node recovery introspection:")
    print("   Status: HEALTHY (recovered)")
    print("   Components: all healthy")
    # await node.publish_introspection(reason="health_recovered")
    print("   ✓ Broadcasted recovery status")


# Example 6: Complete Integration Pattern
# =======================================


def example_complete_integration_code():
    """
    Example: Complete code pattern for integrating NodeIntrospectionMixin.

    This is the actual code structure to use in node.py files.
    """
    integration_code = '''
# In src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py

from ...mixins.health_mixin import HealthCheckMixin
from ...mixins.introspection_mixin import NodeIntrospectionMixin

class NodeBridgeOrchestrator(NodeOrchestrator, HealthCheckMixin, NodeIntrospectionMixin):
    """
    Bridge Orchestrator with health checks and introspection.
    """

    def __init__(self, container: ModelContainer) -> None:
        """Initialize with all capabilities."""
        super().__init__(container)

        # ... existing initialization code ...

        # Initialize health check system
        self.initialize_health_checks()

        # Initialize introspection system
        self.initialize_introspection()

    async def startup(self) -> None:
        """Node startup sequence."""
        # Broadcast startup introspection
        await self.publish_introspection(reason="startup")

        # Start background tasks
        await self.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=30,
            enable_registry_listener=True
        )

    async def shutdown(self) -> None:
        """Node shutdown sequence."""
        # Stop background tasks
        await self.stop_introspection_tasks()

        # Optional: broadcast shutdown introspection
        await self.publish_introspection(reason="shutdown")
'''

    print("\n=== Complete Integration Pattern ===")
    print(integration_code)


# Main execution
# ==============


async def main():
    """Run all examples."""
    print("=" * 70)
    print("NodeIntrospectionMixin Usage Examples")
    print("=" * 70)

    # Example 1: Orchestrator
    await example_orchestrator_with_introspection()

    # Example 2: Reducer
    await example_reducer_with_introspection()

    # Example 3: Manual triggers
    await example_manual_introspection_triggers()

    # Example 4: Custom capabilities
    await example_custom_capability_extraction()

    # Example 5: Health integration
    await example_introspection_with_health_integration()

    # Example 6: Complete integration
    example_complete_integration_code()

    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
