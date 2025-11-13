"""Example usage of protocol adapters.

This module demonstrates how to use the adapter pattern for type-safe
duck typing with omnibase_core classes.

Author: OmniNode Bridge Team
Created: 2025-10-30
"""

import asyncio
from typing import Any

# Import adapters for implementation
from .container_adapter import AdapterModelContainer
from .node_adapters import AdapterNodeOrchestrator

# Import protocols for type hints
from .protocols import ProtocolContainer, ProtocolNode


async def example_container_usage() -> None:
    """Demonstrate protocol-compliant container usage."""
    print("=" * 60)
    print("Container Adapter Usage Example")
    print("=" * 60)

    # Create container with protocol typing
    container: ProtocolContainer = AdapterModelContainer.create(
        config={
            "metadata_stamping_service_url": "http://localhost:8053",
            "kafka_broker_url": "localhost:9092",
            "default_namespace": "omninode.bridge",
        }
    )

    print(f"Container created: {container}")
    print(f"Container config: {container.config}")

    # Register services using protocol interface
    mock_service = {"name": "mock_kafka_client", "status": "connected"}
    container.register_service("kafka_client", mock_service)
    print("Registered service: kafka_client")

    # Retrieve services using protocol interface
    retrieved_service = container.get_service("kafka_client")
    print(f"Retrieved service: {retrieved_service}")

    # Type checking works!
    assert isinstance(container, ProtocolContainer)
    print("✓ Container is protocol-compliant")


async def example_node_usage() -> None:
    """Demonstrate protocol-compliant node usage."""
    print("\n" + "=" * 60)
    print("Node Adapter Usage Example")
    print("=" * 60)

    # Create container
    container: ProtocolContainer = AdapterModelContainer.create(
        config={
            "metadata_stamping_service_url": "http://localhost:8053",
            "health_check_mode": True,  # Skip actual service initialization
        }
    )

    # Create node with protocol typing
    # Note: This would normally be a concrete implementation like NodeBridgeOrchestrator
    # but we're using the adapter base class for demonstration
    node: ProtocolNode = AdapterNodeOrchestrator(container)

    print(f"Node created: {node}")
    print(f"Node ID: {node.node_id}")

    # Type checking works!
    assert isinstance(node, ProtocolNode)
    print("✓ Node is protocol-compliant")


def example_type_hints() -> None:
    """Demonstrate type hint usage with protocols."""
    print("\n" + "=" * 60)
    print("Type Hints with Protocols")
    print("=" * 60)

    def process_with_any_container(container: ProtocolContainer) -> dict[str, Any]:
        """Function accepting any protocol-compliant container.

        This function uses duck typing - it works with ANY container
        that implements ProtocolContainer, not just specific classes.

        Args:
            container: Any container implementing ProtocolContainer

        Returns:
            Configuration dictionary
        """
        # Access methods defined in protocol
        config = container.config
        return {"config_keys": list(config.keys()), "source": "protocol"}

    def process_with_any_node(node: ProtocolNode) -> str:
        """Function accepting any protocol-compliant node.

        This function uses duck typing - it works with ANY node
        that implements ProtocolNode, regardless of node type
        (Effect, Compute, Reducer, Orchestrator).

        Args:
            node: Any node implementing ProtocolNode

        Returns:
            Node identifier
        """
        # Access methods defined in protocol
        return node.node_id

    # Create instances
    container = AdapterModelContainer.create(config={"test_key": "test_value"})
    node = AdapterNodeOrchestrator(container)

    # Use with type-safe functions
    config_info = process_with_any_container(container)
    node_id = process_with_any_node(node)

    print(f"Config info: {config_info}")
    print(f"Node ID: {node_id}")
    print("✓ Type hints work with protocol duck typing")


def example_migration_path() -> None:
    """Show migration path from adapters to native omnibase_core."""
    print("\n" + "=" * 60)
    print("Migration Path (Future)")
    print("=" * 60)

    print("BEFORE (with adapters):")
    print("  from omninode_bridge.adapters import AdapterModelContainer")
    print("  container = AdapterModelContainer.create(config={...})")
    print()

    print("AFTER (native omnibase_core v0.2.0+):")
    print("  from omnibase_core.models.container import ModelONEXContainer")
    print("  container = ModelONEXContainer(config={...})")
    print()

    print("Key Changes:")
    print("  1. Update omnibase_core and omnibase_spi to v0.2.0+")
    print("  2. Search/replace adapter imports")
    print("  3. Remove adapters module")
    print("  4. Same protocol types work!")
    print()
    print("See UPSTREAM_CHANGES.md for detailed migration guide")


async def main() -> None:
    """Run all examples."""
    print("Protocol Adapter Examples")
    print("=" * 60)

    # Run examples
    await example_container_usage()
    await example_node_usage()
    example_type_hints()
    example_migration_path()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
