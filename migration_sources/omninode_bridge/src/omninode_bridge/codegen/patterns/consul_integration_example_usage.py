#!/usr/bin/env python3
"""
Example usage of Consul integration patterns in code generation.

This demonstrates how the Consul patterns from Workstream 2 integrate with
the template engine to generate complete ONEX v2.0 nodes with service discovery.
"""

from omninode_bridge.codegen.patterns.consul_integration import (
    ConsulPatternGenerator,
    ConsulRegistrationConfig,
    generate_consul_deregistration,
    generate_consul_discovery,
    generate_consul_registration,
)


def example_1_quick_generation():
    """Example 1: Quick pattern generation with convenience functions."""
    print("=" * 80)
    print("EXAMPLE 1: Quick Pattern Generation")
    print("=" * 80)

    # Generate registration code
    registration = generate_consul_registration(
        node_type="effect", service_name="postgres_crud", port=8000
    )

    print("\n📝 Generated Registration Method:")
    print(registration[:500] + "...")

    # Generate discovery code
    discovery = generate_consul_discovery()

    print("\n📝 Generated Discovery Method:")
    print(discovery[:500] + "...")

    # Generate deregistration code
    deregistration = generate_consul_deregistration()

    print("\n📝 Generated Deregistration Method:")
    print(deregistration[:300] + "...")


def example_2_full_generator():
    """Example 2: Using ConsulPatternGenerator for all patterns."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Full Pattern Generator")
    print("=" * 80)

    generator = ConsulPatternGenerator()

    # Generate all patterns at once
    patterns = generator.generate_all_patterns(
        node_type="orchestrator",
        service_name="workflow_orchestrator",
        port=8001,
        version="2.0.0",
        domain="bridge",
    )

    print("\n✅ Generated 3 patterns:")
    print(f"  - Registration: {len(patterns['registration'])} chars")
    print(f"  - Discovery: {len(patterns['discovery'])} chars")
    print(f"  - Deregistration: {len(patterns['deregistration'])} chars")

    # Get required imports
    imports = generator.get_required_imports()
    print(f"\n📦 Required imports ({len(imports)}):")
    for imp in imports:
        print(f"  {imp}")

    # Get generated patterns metadata
    metadata = generator.get_generated_patterns()
    print(f"\n📊 Generated patterns metadata: {metadata}")


def example_3_node_class_generation():
    """Example 3: Generate complete node class with Consul integration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Complete Node Class Generation")
    print("=" * 80)

    generator = ConsulPatternGenerator()

    # Configuration
    node_config = {
        "class_name": "NodePostgresCRUDEffect",
        "node_type": "effect",
        "service_name": "postgres_crud",
        "port": 8000,
        "operations": ["read", "write", "update", "delete"],
    }

    # Generate Consul patterns
    patterns = generator.generate_all_patterns(
        node_type=node_config["node_type"],
        service_name=node_config["service_name"],
        port=node_config["port"],
    )

    # Build complete node class
    node_class = f'''
"""
{node_config["class_name"]} - Generated ONEX v2.0 Effect Node
"""

from uuid import uuid4, UUID
from typing import Optional
from datetime import UTC, datetime

from omnibase_core.container.model_container import ModelContainer
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event


class {node_config["class_name"]}:
    """
    PostgreSQL CRUD Effect Node with Consul service discovery.

    Operations: {", ".join(node_config["operations"])}
    Service: {node_config["service_name"]}
    Port: {node_config["port"]}
    """

    def __init__(self, container: ModelContainer):
        """Initialize node with ModelContainer."""
        self.container = container
        self.node_id: UUID = uuid4()

        # Consul-specific state
        self._consul_service_id: Optional[str] = None
        self._service_cache: dict = {{}}

        # Node state
        self._is_shutting_down: bool = False

    async def startup(self) -> None:
        """
        Node startup sequence.

        Steps:
        1. Initialize database connection
        2. Register with Consul
        3. Start health checks
        """
        emit_log_event(
            LogLevel.INFO,
            f"Starting {{self.__class__.__name__}}",
            {{"node_id": str(self.node_id)}}
        )

        # Initialize components
        await self._initialize_database()

        # Register with Consul (generated code)
        await self._register_with_consul()

        # Start health monitoring
        await self._start_health_checks()

        emit_log_event(
            LogLevel.INFO,
            f"{{self.__class__.__name__}} started successfully",
            {{"node_id": str(self.node_id), "service": "{node_config["service_name"]}"}}
        )

    async def shutdown(self) -> None:
        """
        Node shutdown sequence.

        Steps:
        1. Stop accepting new requests
        2. Deregister from Consul
        3. Close database connections
        """
        emit_log_event(
            LogLevel.INFO,
            f"Shutting down {{self.__class__.__name__}}",
            {{"node_id": str(self.node_id)}}
        )

        # Mark as shutting down
        self._is_shutting_down = True

        # Deregister from Consul (generated code)
        await self._deregister_from_consul()

        # Cleanup resources
        await self._close_database()

        emit_log_event(
            LogLevel.INFO,
            f"{{self.__class__.__name__}} shutdown complete",
            {{"node_id": str(self.node_id)}}
        )

    # === GENERATED CONSUL METHODS ===
{patterns["registration"]}
{patterns["discovery"]}
{patterns["deregistration"]}

    # === BUSINESS LOGIC METHODS ===

    async def _initialize_database(self) -> None:
        """Initialize database connection."""
        # Implementation placeholder
        pass

    async def _start_health_checks(self) -> None:
        """Start health check monitoring."""
        # Implementation placeholder
        pass

    async def _close_database(self) -> None:
        """Close database connections."""
        # Implementation placeholder
        pass
'''

    print("\n📄 Generated Node Class Structure:")
    print(f"  Class: {node_config['class_name']}")
    print(f"  Service: {node_config['service_name']}")
    print(f"  Port: {node_config['port']}")
    print(f"  Total size: {len(node_class)} characters")
    print("\n✅ Complete node class with Consul integration generated!")


def example_4_multi_service_discovery():
    """Example 4: Node that discovers multiple services."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Multi-Service Discovery")
    print("=" * 80)

    generator = ConsulPatternGenerator()

    # Generate discovery for orchestrator that needs multiple services
    patterns = generator.generate_all_patterns(
        node_type="orchestrator", service_name="multi_service_orchestrator", port=8002
    )

    # Example orchestration method that discovers multiple services
    orchestration_method = '''
    async def orchestrate_workflow(self, workflow_id: str) -> dict:
        """
        Orchestrate workflow across multiple services.

        Discovers and coordinates:
        - postgres_crud (database operations)
        - metrics_aggregator (metrics collection)
        - notification_service (alerts)
        """
        emit_log_event(
            LogLevel.INFO,
            f"Starting workflow orchestration: {workflow_id}",
            {"node_id": str(self.node_id), "workflow_id": workflow_id}
        )

        # Discover required services (uses generated discovery method)
        postgres_url = await self._discover_service("postgres_crud")
        metrics_url = await self._discover_service("metrics_aggregator")
        notification_url = await self._discover_service("notification_service")

        # Check all services are available
        if not all([postgres_url, metrics_url, notification_url]):
            emit_log_event(
                LogLevel.ERROR,
                "Not all required services available",
                {
                    "postgres": bool(postgres_url),
                    "metrics": bool(metrics_url),
                    "notifications": bool(notification_url)
                }
            )
            return {"status": "failed", "reason": "services_unavailable"}

        # Execute workflow steps
        results = {}

        # Step 1: Database operations
        results["database"] = await self._call_service(
            postgres_url,
            "/execute",
            {"operation": "read", "workflow_id": workflow_id}
        )

        # Step 2: Record metrics
        results["metrics"] = await self._call_service(
            metrics_url,
            "/record",
            {"workflow_id": workflow_id, "step": "database_complete"}
        )

        # Step 3: Send notifications
        results["notifications"] = await self._call_service(
            notification_url,
            "/notify",
            {"workflow_id": workflow_id, "status": "complete"}
        )

        emit_log_event(
            LogLevel.INFO,
            f"Workflow orchestration complete: {workflow_id}",
            {"node_id": str(self.node_id), "results": results}
        )

        return {"status": "success", "workflow_id": workflow_id, "results": results}
'''

    print("\n📝 Generated orchestration method with multi-service discovery:")
    print(orchestration_method[:800] + "...")
    print(
        "\n✅ Orchestrator can discover 3 services: postgres_crud, metrics_aggregator, notification_service"
    )


def example_5_config_dataclass():
    """Example 5: Using ConsulRegistrationConfig dataclass."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Configuration Dataclass")
    print("=" * 80)

    # Create configuration
    config = ConsulRegistrationConfig(
        node_type="reducer",
        service_name="metrics_aggregator",
        port=8003,
        health_endpoint="/health",
        version="2.1.0",
        domain="analytics",
    )

    print("\n📋 Configuration:")
    print(f"  Node Type: {config.node_type}")
    print(f"  Service Name: {config.service_name}")
    print(f"  Port: {config.port}")
    print(f"  Health Endpoint: {config.health_endpoint}")
    print(f"  Version: {config.version}")
    print(f"  Domain: {config.domain}")

    # Generate using config
    generator = ConsulPatternGenerator()
    registration = generator.generate_registration(
        node_type=config.node_type,
        service_name=config.service_name,
        port=config.port,
        health_endpoint=config.health_endpoint,
        version=config.version,
        domain=config.domain,
    )

    print(f"\n✅ Generated registration code from config ({len(registration)} chars)")


def example_6_integration_with_lifecycle():
    """Example 6: Integration with lifecycle patterns."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Lifecycle Integration")
    print("=" * 80)

    generator = ConsulPatternGenerator()

    # Generate Consul patterns
    consul_patterns = generator.generate_all_patterns(
        node_type="compute", service_name="data_transformer", port=8004
    )

    # Demonstrate lifecycle integration
    lifecycle_example = f'''
class NodeDataTransformerCompute:
    """Data transformer with Consul and lifecycle integration."""

    def __init__(self, container):
        self.container = container
        self.node_id = uuid4()
        self._consul_service_id: Optional[str] = None
        self._service_cache: dict = {{}}

    # === LIFECYCLE METHODS (from lifecycle.py) ===

    async def startup(self):
        """Startup sequence with Consul registration."""
        # 1. Initialize components
        await self._initialize_components()

        # 2. Register with Consul (from consul_integration.py)
        await self._register_with_consul()

        # 3. Start monitoring
        await self._start_monitoring()

    async def shutdown(self):
        """Shutdown sequence with Consul deregistration."""
        # 1. Stop new requests
        self._is_shutting_down = True

        # 2. Deregister from Consul (from consul_integration.py)
        await self._deregister_from_consul()

        # 3. Cleanup
        await self._cleanup_resources()

    # === CONSUL METHODS (from consul_integration.py) ===
{consul_patterns["registration"][:300]}
    # ... (full registration method)
{consul_patterns["discovery"][:300]}
    # ... (full discovery method)
{consul_patterns["deregistration"][:300]}
    # ... (full deregistration method)
'''

    print("\n📝 Lifecycle + Consul Integration Example:")
    print(lifecycle_example[:1500] + "...")
    print(
        "\n✅ Shows how Consul registration/deregistration fits into startup/shutdown"
    )


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("CONSUL INTEGRATION PATTERNS - USAGE EXAMPLES")
    print("Workstream 2: Service Discovery Integration")
    print("=" * 80)

    # Run examples
    example_1_quick_generation()
    example_2_full_generator()
    example_3_node_class_generation()
    example_4_multi_service_discovery()
    example_5_config_dataclass()
    example_6_integration_with_lifecycle()

    print("\n" + "=" * 80)
    print("✅ ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Quick convenience functions for simple use cases")
    print("  2. ConsulPatternGenerator for complete pattern generation")
    print("  3. Graceful degradation - nodes work without Consul")
    print("  4. Health-aware service discovery with caching")
    print("  5. Seamless integration with lifecycle patterns")
    print("  6. Production-ready code with structured logging")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
