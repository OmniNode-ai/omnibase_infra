#!/usr/bin/env python3
"""
Lifecycle Management Pattern Generator - Example Usage.

Demonstrates how to use the lifecycle pattern generator to create production-ready
lifecycle management code for ONEX v2.0 nodes.

This example shows:
1. Basic lifecycle generation (init, startup, shutdown)
2. Runtime monitoring generation
3. Helper methods generation
4. Complete node class generation with lifecycle
5. Integration with other workstreams (health, consul, events, metrics)

Run this script to see generated code examples:
    python lifecycle_example_usage.py
"""

from omninode_bridge.codegen.patterns.lifecycle import (
    LifecyclePatternGenerator,
    generate_helper_methods,
    generate_init_method,
    generate_runtime_monitoring,
    generate_shutdown_method,
    generate_startup_method,
)


def example_1_basic_init():
    """Example 1: Basic initialization code generation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Initialization")
    print("=" * 80)

    code = generate_init_method(
        node_type="effect",
        operations=["database_query", "cache_read", "api_call"],
        enable_health_checks=True,
        enable_introspection=True,
        enable_metrics=True,
    )

    print(code)


def example_2_startup_with_dependencies():
    """Example 2: Startup with multiple dependencies."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Startup with Dependencies")
    print("=" * 80)

    code = generate_startup_method(
        node_type="orchestrator",
        dependencies=["consul", "kafka", "postgres"],
        enable_consul=True,
        enable_kafka=True,
        background_tasks=["workflow_monitor", "cache_refresher"],
    )

    print(code)


def example_3_graceful_shutdown():
    """Example 3: Graceful shutdown with cleanup."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Graceful Shutdown")
    print("=" * 80)

    code = generate_shutdown_method(
        dependencies=["kafka", "postgres", "consul"],
        enable_consul=True,
        enable_kafka=True,
        enable_metrics=True,
        background_tasks=["workflow_monitor", "cache_refresher"],
    )

    print(code)


def example_4_runtime_monitoring():
    """Example 4: Runtime health and metrics monitoring."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Runtime Monitoring")
    print("=" * 80)

    code = generate_runtime_monitoring(
        monitor_health=True,
        monitor_metrics=True,
        monitor_resources=True,
        interval_seconds=60,
    )

    print(code)


def example_5_helper_methods():
    """Example 5: Helper methods for lifecycle management."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Helper Methods")
    print("=" * 80)

    code = generate_helper_methods(dependencies=["consul", "kafka", "postgres"])

    print(code)


def example_6_complete_node_lifecycle():
    """Example 6: Complete node class with full lifecycle."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Complete Node with Lifecycle")
    print("=" * 80)

    generator = LifecyclePatternGenerator()

    # Generate all lifecycle components
    init_code = generator.generate_init_method(
        node_type="reducer",
        operations=["stream_aggregation", "state_update"],
        enable_health_checks=True,
        enable_introspection=True,
        enable_metrics=True,
    )

    startup_code = generator.generate_startup_method(
        node_type="reducer",
        dependencies=["consul", "kafka", "postgres"],
        enable_consul=True,
        enable_kafka=True,
        enable_health_checks=True,
        enable_metrics=True,
        enable_introspection=True,
        background_tasks=["aggregation_monitor"],
    )

    shutdown_code = generator.generate_shutdown_method(
        dependencies=["kafka", "postgres", "consul"],
        enable_consul=True,
        enable_kafka=True,
        enable_metrics=True,
        enable_introspection=True,
        background_tasks=["aggregation_monitor"],
    )

    monitoring_code = generator.generate_runtime_monitoring(
        monitor_health=True,
        monitor_metrics=True,
        monitor_resources=True,
        interval_seconds=60,
    )

    helpers_code = generator.generate_helper_methods(
        dependencies=["consul", "kafka", "postgres"]
    )

    # Combine into complete node class
    complete_node = f'''
"""
Generated Node with Complete Lifecycle Management.

This node demonstrates full lifecycle integration with:
- Initialization: Container setup, health checks, metrics
- Startup: Consul registration, Kafka connection, database connection
- Runtime: Health monitoring, metrics publication, resource tracking
- Shutdown: Graceful cleanup, deregistration, resource release
"""

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

try:
    from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
    from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
    from omnibase_core.models.model_container import ModelContainer
    MIXINS_AVAILABLE = True
except ImportError:
    MIXINS_AVAILABLE = False
    # Fallback implementations for testing
    from enum import Enum

    class LogLevel(str, Enum):
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"

    def emit_log_event(level: LogLevel, message: str, metadata: dict[str, Any]) -> None:
        print(f"[{{level.value}}] {{message}}: {{metadata}}")

    class _FallbackConfig:
        """Fallback config when omnibase_core not available."""
        def __init__(self, values: dict[str, Any] | None = None):
            self._values = values or {{}}

        def get(self, key: str, default: Any = None) -> Any:
            return self._values.get(key, default)

    class ModelContainer:
        """Fallback container when omnibase_core not available."""
        def __init__(self, value: Any = None):
            self.value = value
            self.consul_client = None
            self.kafka_client = None
            self.postgres_client = None
            self._services: dict[str, Any] = {{}}
            self.config = _FallbackConfig()

        def get_service(self, name: str) -> Any | None:
            """Get a registered service by name."""
            return self._services.get(name)

        def register_service(self, name: str, service: Any) -> None:
            """Register a service by name."""
            self._services[name] = service


class NodeBridgeReducerGenerated:
    """
    Generated reducer node with complete lifecycle management.

    Demonstrates integration with all workstreams:
    - Health checks (Workstream 1)
    - Consul service discovery (Workstream 2)
    - Event publishing (Workstream 3)
    - Metrics collection (Workstream 4)
    - Lifecycle management (Workstream 5)
    """

{init_code}

{startup_code}

{shutdown_code}

{monitoring_code}

{helpers_code}


# Example usage
async def main():
    """Example node lifecycle execution."""
    # Create container with configuration
    container = ModelContainer(value={{
        "service_address": "localhost",
        "service_port": 8061,
        "max_concurrent_workflows": 100,
    }})

    # Initialize node
    node = NodeBridgeReducerGenerated(container)
    print(f"Node initialized: {{node.node_id}}")

    # Start node
    await node.startup()
    print("Node started successfully")

    # Simulate runtime operation
    print("Node running... (monitoring active)")
    await asyncio.sleep(5)

    # Shutdown node
    await node.shutdown()
    print("Node shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
'''

    print(complete_node)


def example_7_custom_configuration():
    """Example 7: Initialization with custom configuration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Custom Configuration")
    print("=" * 80)

    generator = LifecyclePatternGenerator()

    code = generator.generate_init_method(
        node_type="orchestrator",
        operations=["workflow_orchestration", "task_routing"],
        enable_health_checks=True,
        enable_introspection=True,
        enable_metrics=True,
        custom_config={
            "max_concurrent_workflows": 100,
            "workflow_timeout_seconds": 300,
            "result_cache_ttl": 3600,
            "enable_result_caching": True,
        },
    )

    print(code)


def example_8_minimal_lifecycle():
    """Example 8: Minimal lifecycle (no integrations)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Minimal Lifecycle")
    print("=" * 80)

    generator = LifecyclePatternGenerator()

    init_code = generator.generate_init_method(
        node_type="compute",
        operations=["calculation"],
        enable_health_checks=False,
        enable_introspection=False,
        enable_metrics=False,
    )

    startup_code = generator.generate_startup_method(
        node_type="compute",
        dependencies=[],
        enable_consul=False,
        enable_kafka=False,
        enable_health_checks=False,
        enable_metrics=False,
        enable_introspection=False,
    )

    shutdown_code = generator.generate_shutdown_method(
        dependencies=[],
        enable_consul=False,
        enable_kafka=False,
        enable_metrics=False,
        enable_introspection=False,
    )

    print("=== MINIMAL INIT ===")
    print(init_code)
    print("\n=== MINIMAL STARTUP ===")
    print(startup_code)
    print("\n=== MINIMAL SHUTDOWN ===")
    print(shutdown_code)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LIFECYCLE MANAGEMENT PATTERN GENERATOR - EXAMPLES")
    print("=" * 80)
    print(
        "\nThis script demonstrates lifecycle pattern generation for ONEX v2.0 nodes."
    )
    print("Generated code includes:")
    print("  - Initialization with container setup and correlation tracking")
    print("  - Startup with dependency initialization and service registration")
    print("  - Shutdown with graceful cleanup and resource release")
    print("  - Runtime monitoring with health checks and metrics")
    print("  - Helper methods for all lifecycle operations")

    # Run all examples
    example_1_basic_init()
    example_2_startup_with_dependencies()
    example_3_graceful_shutdown()
    example_4_runtime_monitoring()
    example_5_helper_methods()
    example_6_complete_node_lifecycle()
    example_7_custom_configuration()
    example_8_minimal_lifecycle()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review generated code above")
    print("  2. Integrate patterns into your node generator")
    print("  3. Test generated lifecycle with your container configuration")
    print("  4. Customize error handling for your production needs")


if __name__ == "__main__":
    main()
