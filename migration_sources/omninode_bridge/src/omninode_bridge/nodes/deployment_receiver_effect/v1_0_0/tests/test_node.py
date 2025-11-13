#!/usr/bin/env python3
"""Unit tests for NodeDeploymentReceiverEffect."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect

from ..node import NodeDeploymentReceiverEffect


@pytest.fixture
def mock_event_bus():
    """Create mock EventBus."""
    event_bus = AsyncMock()
    event_bus.is_initialized = True
    event_bus.publish_action_event = AsyncMock(return_value=True)
    return event_bus


@pytest.fixture
def container(mock_event_bus):
    """Create test container with mock EventBus."""
    container = MagicMock(spec=ModelONEXContainer)
    container.config = {}

    # Mock get_service to return our mock EventBus
    def get_service_mock(service_name):
        if service_name == "event_bus":
            return mock_event_bus
        return None

    container.get_service = MagicMock(side_effect=get_service_mock)
    container.register_service = MagicMock()
    return container


@pytest.fixture
def node(container):
    """Create test node instance."""
    return NodeDeploymentReceiverEffect(container)


@pytest.mark.asyncio
async def test_node_initialization(node):
    """Test node initializes correctly."""
    assert node is not None
    assert node.node_id is not None
    assert node.event_bus is not None


@pytest.mark.asyncio
async def test_node_initialization_without_event_bus():
    """Test node initialization when EventBus is unavailable."""
    container = MagicMock(spec=ModelONEXContainer)
    container.config = {}

    # Mock container.get_service to return None (no EventBus available)
    container.get_service = MagicMock(return_value=None)
    container.register_service = MagicMock()

    node = NodeDeploymentReceiverEffect(container)
    assert node is not None
    assert node.node_id is not None
    assert node.event_bus is None  # Should handle gracefully


@pytest.mark.asyncio
async def test_publish_deployment_event_with_event_bus(node, mock_event_bus):
    """Test deployment event publishing via EventBus."""
    from ..models import ModelDeploymentConfig

    # Create test deployment config
    deployment_config = ModelDeploymentConfig(
        image_name="test-repo/test-image:v1.0",
        container_name="test-container",
        ports={"8080": 8080},
        environment_vars={},
        volumes=[],
        networks=[],
    )

    correlation_id = uuid4()
    contract = MagicMock(spec=ModelContractEffect)
    contract.correlation_id = correlation_id
    contract.input_state = {
        "operation_type": "publish_deployment_event",
        "event_type": "DEPLOYMENT_STARTED",
        "deployment_config": deployment_config.model_dump(),
        "correlation_id": str(correlation_id),
    }

    result = await node._handle_publish_event(contract, correlation_id)

    assert result.success is True
    assert result.event_id is not None
    assert result.topic is not None

    # Verify EventBus was called
    mock_event_bus.publish_action_event.assert_called_once()
    call_args = mock_event_bus.publish_action_event.call_args
    assert call_args.kwargs["correlation_id"] == correlation_id
    assert call_args.kwargs["action_type"] == "DEPLOYMENT_DEPLOYMENT_STARTED"


@pytest.mark.asyncio
async def test_publish_deployment_event_without_event_bus():
    """Test deployment event publishing falls back gracefully when EventBus unavailable."""
    from ..models import ModelDeploymentConfig

    # Create container without EventBus
    container = MagicMock(spec=ModelONEXContainer)
    container.config = {}

    container.get_service = MagicMock(return_value=None)
    container.register_service = MagicMock()

    node = NodeDeploymentReceiverEffect(container)

    # Create test deployment config
    deployment_config = ModelDeploymentConfig(
        image_name="test-repo/test-image:v1.0",
        container_name="test-container",
        ports={"8080": 8080},
        environment_vars={},
        volumes=[],
        networks=[],
    )

    correlation_id = uuid4()
    contract = MagicMock(spec=ModelContractEffect)
    contract.correlation_id = correlation_id
    contract.input_state = {
        "operation_type": "publish_deployment_event",
        "event_type": "DEPLOYMENT_STARTED",
        "deployment_config": deployment_config.model_dump(),
        "correlation_id": str(correlation_id),
    }

    result = await node._handle_publish_event(contract, correlation_id)

    # Should still succeed (fallback mode - logs only)
    assert result.success is True
    assert result.event_id is not None
    assert result.topic is not None


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Requires full contract with deployment archive and orchestration setup"
)
async def test_execute_effect(node):
    """Test effect execution.

    Implementation pending (Phase 2):
    Requires proper ModelContractEffect with:
    - name, version, description, node_type (EFFECT)
    - input_model (deployment archive path, target host)
    - output_model (deployment status, service endpoints)
    - io_operations (Docker API, service restart, health checks)
    """
    pass
