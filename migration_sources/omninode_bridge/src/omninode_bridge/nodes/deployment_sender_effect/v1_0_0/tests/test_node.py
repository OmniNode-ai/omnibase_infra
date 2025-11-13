#!/usr/bin/env python3
"""
Unit tests for NodeDeploymentSenderEffect.

Tests cover:
- Node initialization
- Docker image building and packaging
- BLAKE3 checksum generation
- HTTP package transfer
- Kafka event publishing
- Error handling
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from omnibase_core.enums.enum_node_type import EnumNodeType
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

from ..models import (
    ModelContainerPackageInput,
    ModelKafkaPublishInput,
    ModelPackageTransferInput,
)
from ..node import NodeDeploymentSenderEffect


@pytest.fixture
def container():
    """Create test container with configuration."""
    return ModelContainer(
        value={
            "package_dir": "/tmp/test_deployment_packages",
            "environment": "test",
            "health_check_mode": True,  # Skip Kafka/EventBus initialization in tests
        },
        container_type="config",
    )


@pytest.fixture
def node(container):
    """Create test node instance."""
    return NodeDeploymentSenderEffect(container)


def create_test_contract(input_state: dict) -> ModelContractEffect:
    """
    Create a minimal valid ModelContractEffect for testing.

    Args:
        input_state: The input_state dict containing operation and parameters

    Returns:
        Valid ModelContractEffect instance
    """
    return ModelContractEffect(
        name="deployment_sender_effect",
        version={"major": 1, "minor": 0, "patch": 0},
        description="Test deployment sender effect contract",
        node_type=EnumNodeType.EFFECT,
        input_model="Any",
        output_model="Any",
        io_operations=[{"operation_type": "read", "description": "Test operation"}],
        input_state=input_state,
        correlation_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_node_initialization(node):
    """Test node initializes correctly with configuration."""
    assert node is not None
    assert node.node_id is not None
    assert node.package_dir == Path("/tmp/test_deployment_packages")
    assert node._metrics["total_builds"] == 0
    assert node._metrics["successful_builds"] == 0


@pytest.mark.asyncio
@patch("docker.from_env")
async def test_package_container_success(mock_docker, node, tmp_path):
    """Test successful Docker image building and packaging."""
    # Setup mock Docker client
    mock_image = MagicMock()
    mock_image.id = (
        "sha256:a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"
    )
    mock_image.save.return_value = iter([b"mock_image_data"])

    mock_docker_client = MagicMock()
    mock_docker_client.ping.return_value = True
    mock_docker_client.images.build.return_value = (mock_image, [])
    mock_docker_client.images.get.return_value = mock_image
    mock_docker.return_value = mock_docker_client

    # Create input
    input_data = ModelContainerPackageInput(
        container_name="test-container",
        image_tag="1.0.0",
        dockerfile_path="Dockerfile",
        build_context=str(tmp_path),
        compression="none",
    )

    # Execute
    result = await node.package_container(input_data)

    # Assertions
    assert result.success is True
    assert result.package_id is not None
    assert (
        result.image_id
        == "sha256:a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"
    )
    assert result.package_checksum is not None
    assert len(result.package_checksum) == 64  # BLAKE3 hex digest
    assert result.build_duration_ms >= 0
    assert node._metrics["successful_builds"] == 1


@pytest.mark.asyncio
@patch("docker.from_env")
async def test_package_container_docker_failure(mock_docker, node, tmp_path):
    """Test Docker build failure handling."""
    # Setup mock Docker client to fail
    mock_docker_client = MagicMock()
    mock_docker_client.ping.return_value = True
    mock_docker_client.images.build.side_effect = Exception("Docker build failed")
    mock_docker.return_value = mock_docker_client

    # Create input
    input_data = ModelContainerPackageInput(
        container_name="test-container",
        image_tag="1.0.0",
        dockerfile_path="Dockerfile",
        build_context=str(tmp_path),
    )

    # Execute
    result = await node.package_container(input_data)

    # Assertions
    assert result.success is False
    assert result.error_code == "BUILD_FAILED"
    assert result.error_message is not None
    assert node._metrics["failed_builds"] == 1


@pytest.mark.asyncio
async def test_generate_blake3_checksum(node, tmp_path):
    """Test BLAKE3 checksum generation."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_data = b"Hello, World!" * 1000
    test_file.write_bytes(test_data)

    # Generate checksum
    checksum = await node._generate_blake3_checksum(test_file)

    # Assertions
    assert checksum is not None
    assert len(checksum) == 64
    assert isinstance(checksum, str)
    assert all(c in "0123456789abcdef" for c in checksum)


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_transfer_package_success(mock_post, node, tmp_path):
    """Test successful package transfer via HTTP."""
    # Create test package
    package_path = tmp_path / "test-package.tar.gz"
    package_path.write_bytes(b"mock_package_data")

    # Setup mock HTTP response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(
        return_value={
            "deployment_id": "deploy-123",
            "checksum_verified": True,
        }
    )
    mock_post.return_value = mock_response

    # Create input
    input_data = ModelPackageTransferInput(
        package_id=uuid4(),
        package_path=str(package_path),
        package_checksum="a" * 64,
        remote_receiver_url="http://example.com/deploy",
        transfer_method="http",
    )

    # Execute
    result = await node.transfer_package(input_data)

    # Assertions
    assert result.success is True
    assert result.transfer_success is True
    assert result.remote_deployment_id == "deploy-123"
    assert result.checksum_verified is True
    assert result.transfer_duration_ms >= 0
    assert result.bytes_transferred == len(b"mock_package_data")
    assert node._metrics["successful_transfers"] == 1


@pytest.mark.asyncio
async def test_transfer_package_file_not_found(node):
    """Test transfer failure when package file doesn't exist."""
    # Create input with non-existent file
    input_data = ModelPackageTransferInput(
        package_id=uuid4(),
        package_path="/non/existent/package.tar.gz",
        package_checksum="a" * 64,
        remote_receiver_url="http://example.com/deploy",
    )

    # Execute
    result = await node.transfer_package(input_data)

    # Assertions
    assert result.success is False
    assert result.transfer_success is False
    assert result.error_code == "TRANSFER_FAILED"
    assert node._metrics["failed_transfers"] == 1


@pytest.mark.asyncio
async def test_publish_transfer_event_success(node):
    """Test successful Kafka event publishing."""
    # Mock KafkaClient
    mock_kafka_client = AsyncMock()
    mock_kafka_client.is_connected = True
    mock_kafka_client.publish_with_envelope = AsyncMock(return_value=True)
    node.kafka_client = mock_kafka_client

    # Create input
    input_data = ModelKafkaPublishInput(
        event_type="BUILD_COMPLETED",
        event_payload={
            "image_id": "sha256:abc123def456789abc123def456789abc123def456789abc123def456789abcd",
            "build_duration_ms": 5000,
        },
        correlation_id=uuid4(),
        package_id=uuid4(),
        container_name="test-container",
        image_tag="1.0.0",
    )

    # Execute
    result = await node.publish_transfer_event(input_data)

    # Assertions
    assert result.success is True
    assert result.event_published is True
    assert result.topic is not None
    assert "build-completed" in result.topic
    assert result.publish_duration_ms >= 0
    assert node._metrics["total_events_published"] == 1

    # Verify KafkaClient was called correctly
    mock_kafka_client.publish_with_envelope.assert_called_once()
    call_kwargs = mock_kafka_client.publish_with_envelope.call_args.kwargs
    assert call_kwargs["event_type"] == "BUILD_COMPLETED"
    assert call_kwargs["source_node_id"] == str(node.node_id)
    assert "build-completed" in call_kwargs["topic"]


@pytest.mark.asyncio
async def test_execute_effect_package_container(node, tmp_path):
    """Test execute_effect routing to package_container operation."""
    with patch("docker.from_env") as mock_docker:
        # Setup mock
        mock_image = MagicMock()
        mock_image.id = (
            "sha256:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        )
        mock_image.save.return_value = iter([b"data"])

        mock_docker_client = MagicMock()
        mock_docker_client.ping.return_value = True
        mock_docker_client.images.build.return_value = (mock_image, [])
        mock_docker_client.images.get.return_value = mock_image
        mock_docker.return_value = mock_docker_client

        # Create contract using helper
        contract = create_test_contract(
            input_state={
                "operation": "package_container",
                "input": {
                    "container_name": "test",
                    "image_tag": "1.0",
                    "build_context": str(tmp_path),
                    "compression": "none",
                },
            }
        )

        # Execute
        result = await node.execute_effect(contract)

        # Assertions
        assert hasattr(result, "success")
        assert hasattr(result, "package_id")


@pytest.mark.asyncio
async def test_execute_effect_get_metrics(node):
    """Test execute_effect routing to get_metrics operation."""
    # Create contract using helper
    contract = create_test_contract(input_state={"operation": "get_metrics"})

    # Execute
    result = await node.execute_effect(contract)

    # Assertions
    assert isinstance(result, dict)
    assert "total_builds" in result
    assert "successful_builds" in result
    assert "failed_builds" in result
    assert "total_transfers" in result


@pytest.mark.asyncio
async def test_execute_effect_unknown_operation(node):
    """Test execute_effect with unknown operation."""
    from omnibase_core import ModelOnexError

    # Create contract with unknown operation using helper
    contract = create_test_contract(input_state={"operation": "unknown_operation"})

    # Execute and expect error
    with pytest.raises(ModelOnexError) as exc_info:
        await node.execute_effect(contract)

    assert "Unknown operation" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_cleanup(node):
    """Test resource cleanup."""
    # Initialize clients
    node._http_client = AsyncMock()
    node._docker_client = MagicMock()

    # Execute cleanup
    await node.cleanup()

    # Assertions
    node._http_client.aclose.assert_called_once()
    node._docker_client.close.assert_called_once()


@pytest.mark.asyncio
@patch("docker.from_env")
async def test_docker_client_lazy_initialization(mock_docker, node):
    """Test Docker client is lazily initialized."""
    mock_docker_client = MagicMock()
    mock_docker_client.ping.return_value = True
    mock_docker.return_value = mock_docker_client

    # Client should be None initially
    assert node._docker_client is None

    # Get client
    client = node._get_docker_client()

    # Client should now be initialized
    assert client is not None
    assert node._docker_client is not None
    mock_docker.assert_called_once()

    # Subsequent calls should return same instance
    client2 = node._get_docker_client()
    assert client2 is client
    assert mock_docker.call_count == 1  # Still only called once


@pytest.mark.asyncio
async def test_http_client_lazy_initialization(node):
    """Test HTTP client is lazily initialized."""
    # Client should be None initially
    assert node._http_client is None

    # Get client
    client = node._get_http_client()

    # Client should now be initialized
    assert client is not None
    assert node._http_client is not None

    # Subsequent calls should return same instance
    client2 = node._get_http_client()
    assert client2 is client


@pytest.mark.asyncio
async def test_metrics_tracking(node, tmp_path):
    """Test metrics are properly tracked across operations."""
    with patch("docker.from_env") as mock_docker:
        # Setup mock
        mock_image = MagicMock()
        mock_image.id = (
            "sha256:fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"
        )
        mock_image.save.return_value = iter([b"data"])

        mock_docker_client = MagicMock()
        mock_docker_client.ping.return_value = True
        mock_docker_client.images.build.return_value = (mock_image, [])
        mock_docker_client.images.get.return_value = mock_image
        mock_docker.return_value = mock_docker_client

        # Initial state
        assert node._metrics["total_builds"] == 0
        assert node._metrics["successful_builds"] == 0

        # Execute successful build
        input_data = ModelContainerPackageInput(
            container_name="test",
            image_tag="1.0",
            build_context=str(tmp_path),
            compression="none",
        )
        await node.package_container(input_data)

        # Verify metrics updated
        assert node._metrics["total_builds"] == 1
        assert node._metrics["successful_builds"] == 1
        assert node._metrics["failed_builds"] == 0


@pytest.mark.asyncio
async def test_model_validation():
    """Test Pydantic model validation."""
    from pydantic import ValidationError

    # Test invalid container name (uppercase not allowed)
    with pytest.raises(ValidationError):
        ModelContainerPackageInput(
            container_name="INVALID-NAME",  # Should be lowercase
            image_tag="1.0.0",
        )

    # Test invalid compression type
    with pytest.raises(ValidationError):
        ModelContainerPackageInput(
            container_name="valid-name",
            image_tag="1.0.0",
            compression="invalid",  # Not in allowed values
        )

    # Test directory traversal protection
    with pytest.raises(ValidationError):
        ModelContainerPackageInput(
            container_name="test",
            image_tag="1.0",
            build_context="../../../etc",  # Contains ..
        )


@pytest.mark.asyncio
async def test_package_transfer_input_validation():
    """Test package transfer input validation."""
    from pydantic import ValidationError

    # Test invalid transfer method
    with pytest.raises(ValidationError):
        ModelPackageTransferInput(
            package_id=uuid4(),
            package_path="/tmp/package.tar.gz",
            package_checksum="a" * 64,
            remote_receiver_url="http://example.com",
            transfer_method="ftp",  # Not allowed
        )

    # Test invalid checksum format
    with pytest.raises(ValidationError):
        ModelPackageTransferInput(
            package_id=uuid4(),
            package_path="/tmp/package.tar.gz",
            package_checksum="invalid",  # Not 64 hex chars
            remote_receiver_url="http://example.com",
        )


@pytest.mark.asyncio
async def test_kafka_publish_input_validation():
    """Test Kafka publish input validation."""
    from pydantic import ValidationError

    # Test invalid event type
    with pytest.raises(ValidationError):
        ModelKafkaPublishInput(
            event_type="INVALID_EVENT",  # Not in allowed values
            event_payload={},
            correlation_id=uuid4(),
        )

    # Test valid event types
    for event_type in [
        "BUILD_STARTED",
        "BUILD_COMPLETED",
        "TRANSFER_STARTED",
        "TRANSFER_COMPLETED",
        "DEPLOYMENT_FAILED",
    ]:
        model = ModelKafkaPublishInput(
            event_type=event_type,
            event_payload={"test": "data"},
            correlation_id=uuid4(),
        )
        assert model.event_type == event_type


@pytest.mark.asyncio
async def test_publish_transfer_event_kafka_unavailable(node):
    """Test event publishing when Kafka is unavailable."""
    # Ensure kafka_client is None
    node.kafka_client = None

    # Create input
    input_data = ModelKafkaPublishInput(
        event_type="BUILD_STARTED",
        event_payload={"test": "data"},
        correlation_id=uuid4(),
        package_id=uuid4(),
        container_name="test-container",
        image_tag="1.0.0",
    )

    # Execute
    result = await node.publish_transfer_event(input_data)

    # Assertions - should succeed but not publish to Kafka
    assert result.success is True
    assert result.event_published is False  # Not published to Kafka
    assert result.topic is not None
    assert "build-started" in result.topic
    assert result.publish_duration_ms >= 0


@pytest.mark.asyncio
async def test_publish_transfer_event_kafka_publish_fails(node):
    """Test event publishing when Kafka publish operation fails."""
    # Mock KafkaClient to return False
    mock_kafka_client = AsyncMock()
    mock_kafka_client.is_connected = True
    mock_kafka_client.publish_with_envelope = AsyncMock(return_value=False)
    node.kafka_client = mock_kafka_client

    # Create input
    input_data = ModelKafkaPublishInput(
        event_type="TRANSFER_COMPLETED",
        event_payload={"test": "data"},
        correlation_id=uuid4(),
    )

    # Execute
    result = await node.publish_transfer_event(input_data)

    # Assertions
    assert result.success is False
    assert result.event_published is False
    assert result.error_code == "KAFKA_PUBLISH_FAILED"
    assert result.error_message == "Kafka publish returned False"
