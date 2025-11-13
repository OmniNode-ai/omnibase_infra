"""Tests for protocol adapters.

Validates that adapters provide protocol-compliant interfaces for
omnibase_core classes.

Author: OmniNode Bridge Team
Created: 2025-10-30
"""

import pytest

from omninode_bridge.adapters import (
    AdapterModelContainer,
    AdapterNodeOrchestrator,
    ProtocolContainer,
    ProtocolNode,
)


class TestContainerAdapter:
    """Tests for AdapterModelContainer."""

    def test_create_container(self):
        """Test container creation with config."""
        container = AdapterModelContainer.create(
            config={
                "test_key": "test_value",
                "metadata_stamping_service_url": "http://localhost:8053",
            }
        )

        assert container is not None
        assert hasattr(container, "config")
        assert container.config["test_key"] == "test_value"

    def test_container_is_protocol_compliant(self):
        """Test that container implements ProtocolContainer."""
        container = AdapterModelContainer.create(config={"test": "value"})

        # Runtime checkable protocol validation
        assert isinstance(container, ProtocolContainer)

        # Check protocol methods exist
        assert hasattr(container, "get_service")
        assert hasattr(container, "register_service")
        assert hasattr(container, "config")

    def test_service_registration_and_retrieval(self):
        """Test service registration and retrieval through protocol interface."""
        container = AdapterModelContainer.create(config={})

        # Register service
        test_service = {"name": "test_service", "status": "active"}
        container.register_service("test_service", test_service)

        # Retrieve service
        retrieved = container.get_service("test_service")
        assert retrieved == test_service

    def test_get_nonexistent_service(self):
        """Test getting non-existent service returns None."""
        container = AdapterModelContainer.create(config={})

        # Should return None for non-existent service
        service = container.get_service("nonexistent")
        assert service is None


class TestNodeAdapter:
    """Tests for node adapters."""

    def test_create_orchestrator_node(self):
        """Test orchestrator node creation."""
        container = AdapterModelContainer.create(config={"health_check_mode": True})

        node = AdapterNodeOrchestrator(container)

        assert node is not None
        assert hasattr(node, "node_id")
        assert node.node_id is not None

    def test_node_is_protocol_compliant(self):
        """Test that node implements ProtocolNode."""
        container = AdapterModelContainer.create(config={"health_check_mode": True})

        node = AdapterNodeOrchestrator(container)

        # Runtime checkable protocol validation
        assert isinstance(node, ProtocolNode)

        # Check protocol methods exist
        assert hasattr(node, "process")
        assert hasattr(node, "get_contract")

    def test_node_has_unified_process_method(self):
        """Test that node has unified process() method."""
        container = AdapterModelContainer.create(config={"health_check_mode": True})

        node = AdapterNodeOrchestrator(container)

        # Should have process method
        assert hasattr(node, "process")
        assert callable(node.process)

    def test_node_get_contract_raises_not_implemented(self):
        """Test that get_contract() raises NotImplementedError for now."""
        container = AdapterModelContainer.create(config={"health_check_mode": True})

        node = AdapterNodeOrchestrator(container)

        # Should raise NotImplementedError until upstream implements
        with pytest.raises(NotImplementedError):
            node.get_contract()


class TestProtocolDuckTyping:
    """Tests for protocol-based duck typing."""

    def test_function_accepts_protocol_container(self):
        """Test that functions can accept ProtocolContainer."""

        def process_container(container: ProtocolContainer) -> str:
            """Function using protocol type hint."""
            return container.config.get("test_key", "default")

        container = AdapterModelContainer.create(config={"test_key": "success"})

        result = process_container(container)
        assert result == "success"

    def test_function_accepts_protocol_node(self):
        """Test that functions can accept ProtocolNode."""

        def get_node_id(node: ProtocolNode) -> str:
            """Function using protocol type hint."""
            # Convert UUID to string for consistent interface
            return str(node.node_id)

        container = AdapterModelContainer.create(config={"health_check_mode": True})
        node = AdapterNodeOrchestrator(container)

        result = get_node_id(node)
        assert result is not None
        assert isinstance(result, str)


class TestAdapterPassthrough:
    """Tests for adapter passthrough behavior."""

    def test_container_delegates_to_wrapped(self):
        """Test that adapter delegates to wrapped container."""
        container = AdapterModelContainer.create(config={"test": "value"})

        # Access wrapped container attributes
        assert hasattr(container, "_container")

        # Config should be accessible
        assert container.config["test"] == "value"

    def test_node_delegates_to_base_class(self):
        """Test that node adapter delegates to base class."""
        container = AdapterModelContainer.create(config={"health_check_mode": True})
        node = AdapterNodeOrchestrator(container)

        # Should have base class attributes
        assert hasattr(node, "container")
        # Note: Node gets the adapter container passed to __init__,
        # which then passes the wrapped container to the base class
        # So node.container is actually the adapter itself
        assert node.container == container


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
