"""Unit tests for @register_agent decorator."""

import pytest

from omninode_bridge.agents.registry.decorators import (
    agent_is_registered,
    clear_registry,
    find_agents_by_capability,
    get_agent_instance,
    get_agent_metadata,
    get_registered_agents,
    register_agent,
)
from omninode_bridge.agents.registry.models import AgentType


# Clear registry before each test
@pytest.fixture(autouse=True)
def clear_decorator_registry():
    """Clear decorator registry before each test."""
    clear_registry()
    yield
    clear_registry()


class TestRegisterAgentDecorator:
    """Tests for @register_agent decorator."""

    def test_register_class(self):
        """Test decorator registers class successfully."""

        @register_agent(
            agent_id="test_agent",
            capabilities=["test"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test agent",
        )
        class TestAgent:
            async def execute(self, task):
                pass

        # Verify agent is registered
        assert agent_is_registered("test_agent")

        # Verify metadata stored correctly
        metadata = get_agent_metadata("test_agent")
        assert metadata is not None
        assert metadata["agent_id"] == "test_agent"
        assert metadata["capabilities"] == ["test"]
        assert metadata["description"] == "Test agent"

    def test_register_function(self):
        """Test decorator registers function successfully."""

        @register_agent(
            agent_id="test_func",
            capabilities=["test"],
            agent_type=AgentType.VALIDATOR,
            version="1.0.0",
            description="Test function",
        )
        async def test_function(task):
            pass

        # Verify function is registered
        assert agent_is_registered("test_func")

        # Verify it's recognized as a function
        metadata = get_agent_metadata("test_func")
        assert metadata["is_class"] is False

    def test_register_duplicate_fails(self):
        """Test registering duplicate agent_id fails."""

        @register_agent(
            agent_id="duplicate",
            capabilities=["test"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="First",
        )
        class FirstAgent:
            async def execute(self, task):
                pass

        # Try to register again with same ID
        with pytest.raises(ValueError) as exc_info:

            @register_agent(
                agent_id="duplicate",
                capabilities=["test"],
                agent_type=AgentType.CONTRACT_INFERENCER,
                version="1.0.0",
                description="Second",
            )
            class SecondAgent:
                async def execute(self, task):
                    pass

        assert "already registered" in str(exc_info.value).lower()

    def test_register_invalid_class(self):
        """Test registering class without execute method fails."""
        with pytest.raises(TypeError) as exc_info:

            @register_agent(
                agent_id="invalid",
                capabilities=["test"],
                agent_type=AgentType.CONTRACT_INFERENCER,
                version="1.0.0",
                description="Invalid",
            )
            class InvalidAgent:
                pass  # No execute method

        assert "execute" in str(exc_info.value).lower()

    def test_register_non_callable(self):
        """Test registering non-callable fails."""
        with pytest.raises(TypeError):

            @register_agent(
                agent_id="not_callable",
                capabilities=["test"],
                agent_type=AgentType.CONTRACT_INFERENCER,
                version="1.0.0",
                description="Not callable",
            )
            class NotCallable:
                pass

    def test_decorator_attaches_metadata(self):
        """Test decorator attaches metadata to class/function."""

        @register_agent(
            agent_id="metadata_test",
            capabilities=["test"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test",
        )
        class MetadataAgent:
            async def execute(self, task):
                pass

        # Check metadata attached to class
        assert hasattr(MetadataAgent, "_agent_id")
        assert hasattr(MetadataAgent, "_capabilities")
        assert hasattr(MetadataAgent, "_metadata")

        assert MetadataAgent._agent_id == "metadata_test"
        assert MetadataAgent._capabilities == ["test"]

    def test_register_with_additional_metadata(self):
        """Test registering with additional metadata kwargs."""

        @register_agent(
            agent_id="extra_metadata",
            capabilities=["test"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test",
            priority=90,
            max_concurrent_tasks=20,
        )
        class ExtraMetadataAgent:
            async def execute(self, task):
                pass

        metadata = get_agent_metadata("extra_metadata")
        assert metadata["metadata"].priority == 90
        assert metadata["metadata"].max_concurrent_tasks == 20


class TestAgentDiscovery:
    """Tests for agent discovery functions."""

    def test_get_registered_agents(self):
        """Test getting all registered agents."""

        @register_agent(
            agent_id="agent1",
            capabilities=["cap1"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Agent 1",
        )
        class Agent1:
            async def execute(self, task):
                pass

        @register_agent(
            agent_id="agent2",
            capabilities=["cap2"],
            agent_type=AgentType.VALIDATOR,
            version="1.0.0",
            description="Agent 2",
        )
        class Agent2:
            async def execute(self, task):
                pass

        agents = get_registered_agents()
        assert len(agents) == 2
        assert "agent1" in agents
        assert "agent2" in agents

    def test_agent_is_registered(self):
        """Test checking if agent is registered."""

        @register_agent(
            agent_id="exists",
            capabilities=["test"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Exists",
        )
        class ExistsAgent:
            async def execute(self, task):
                pass

        assert agent_is_registered("exists") is True
        assert agent_is_registered("not_exists") is False

    def test_get_agent_metadata(self):
        """Test getting agent metadata."""

        @register_agent(
            agent_id="metadata_agent",
            capabilities=["cap1", "cap2"],
            agent_type=AgentType.TEMPLATE_SELECTOR,
            version="2.0.0",
            description="Metadata test",
        )
        class MetadataAgent:
            async def execute(self, task):
                pass

        metadata = get_agent_metadata("metadata_agent")
        assert metadata is not None
        assert metadata["agent_id"] == "metadata_agent"
        assert metadata["capabilities"] == ["cap1", "cap2"]
        assert metadata["description"] == "Metadata test"

    def test_get_metadata_nonexistent(self):
        """Test getting metadata for nonexistent agent returns None."""
        metadata = get_agent_metadata("nonexistent")
        assert metadata is None

    def test_find_agents_by_capability(self):
        """Test finding agents by capability."""

        @register_agent(
            agent_id="agent1",
            capabilities=["cap1", "cap2"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Agent 1",
        )
        class Agent1:
            async def execute(self, task):
                pass

        @register_agent(
            agent_id="agent2",
            capabilities=["cap1", "cap3"],
            agent_type=AgentType.VALIDATOR,
            version="1.0.0",
            description="Agent 2",
        )
        class Agent2:
            async def execute(self, task):
                pass

        @register_agent(
            agent_id="agent3",
            capabilities=["cap2", "cap3"],
            agent_type=AgentType.ANALYZER,
            version="1.0.0",
            description="Agent 3",
        )
        class Agent3:
            async def execute(self, task):
                pass

        # Find agents with cap1
        cap1_agents = find_agents_by_capability("cap1")
        assert len(cap1_agents) == 2
        agent_ids = {agent["agent_id"] for agent in cap1_agents}
        assert agent_ids == {"agent1", "agent2"}

        # Find agents with cap2
        cap2_agents = find_agents_by_capability("cap2")
        assert len(cap2_agents) == 2
        agent_ids = {agent["agent_id"] for agent in cap2_agents}
        assert agent_ids == {"agent1", "agent3"}

        # Find agents with nonexistent capability
        none_agents = find_agents_by_capability("nonexistent")
        assert len(none_agents) == 0


class TestAgentInstantiation:
    """Tests for agent instantiation."""

    def test_get_class_instance(self):
        """Test getting instance of registered class."""

        @register_agent(
            agent_id="class_agent",
            capabilities=["test"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test",
        )
        class ClassAgent:
            def __init__(self, config=None):
                self.config = config

            async def execute(self, task):
                pass

        # Get instance with kwargs
        instance = get_agent_instance("class_agent", config={"key": "value"})
        assert isinstance(instance, ClassAgent)
        assert instance.config == {"key": "value"}

    def test_get_function_instance(self):
        """Test getting registered function."""

        @register_agent(
            agent_id="func_agent",
            capabilities=["test"],
            agent_type=AgentType.VALIDATOR,
            version="1.0.0",
            description="Test",
        )
        async def func_agent(task):
            return "result"

        # Get function
        instance = get_agent_instance("func_agent")
        assert callable(instance)

    def test_get_instance_nonexistent(self):
        """Test getting instance of nonexistent agent raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_instance("nonexistent")

        assert "not registered" in str(exc_info.value).lower()

    def test_get_function_with_kwargs_fails(self):
        """Test getting function with init_kwargs raises error."""

        @register_agent(
            agent_id="func",
            capabilities=["test"],
            agent_type=AgentType.VALIDATOR,
            version="1.0.0",
            description="Test",
        )
        async def func_agent(task):
            pass

        with pytest.raises(TypeError) as exc_info:
            get_agent_instance("func", config="value")

        assert "function" in str(exc_info.value).lower()


class TestClearRegistry:
    """Tests for clearing registry."""

    def test_clear_registry(self):
        """Test clearing registry removes all agents."""

        @register_agent(
            agent_id="agent1",
            capabilities=["test"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test",
        )
        class Agent1:
            async def execute(self, task):
                pass

        assert agent_is_registered("agent1")

        clear_registry()

        assert not agent_is_registered("agent1")
        assert len(get_registered_agents()) == 0


class TestModuleTracking:
    """Tests for module tracking."""

    def test_module_name_recorded(self):
        """Test decorator records module name."""

        @register_agent(
            agent_id="module_test",
            capabilities=["test"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test",
        )
        class ModuleAgent:
            async def execute(self, task):
                pass

        metadata = get_agent_metadata("module_test")
        assert "module_name" in metadata
        assert metadata["module_name"] == __name__
