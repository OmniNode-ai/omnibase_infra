"""Unit tests for AsyncMetadataStampingClient."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from omninode_bridge.clients.circuit_breaker import CircuitState
from omninode_bridge.clients.metadata_stamping_client import AsyncMetadataStampingClient


@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient."""
    mock = AsyncMock(spec=httpx.AsyncClient)
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture
async def metadata_client(mock_httpx_client):
    """Provide AsyncMetadataStampingClient instance for testing."""
    client = AsyncMetadataStampingClient(
        base_url="http://localhost:8053",
        timeout=5.0,
        max_retries=2,
    )

    # Mock HTTP client
    with patch.object(
        client,
        "_http_client",
        mock_httpx_client,
    ):
        yield client

    # Cleanup
    await client.close()


class TestMetadataStampingClientInitialization:
    """Test client initialization and configuration."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initializes with correct defaults."""
        client = AsyncMetadataStampingClient()

        assert (
            client.base_url == "http://192.168.86.200:8057"
        )  # Remote infrastructure default
        assert client.service_name == "MetadataStampingService"
        assert client.timeout == 30.0
        assert client.max_retries == 3

        await client.close()

    @pytest.mark.asyncio
    async def test_client_custom_configuration(self):
        """Test client accepts custom configuration."""
        client = AsyncMetadataStampingClient(
            base_url="http://custom:9999",
            timeout=10.0,
            max_retries=5,
        )

        assert client.base_url == "http://custom:9999"
        assert client.timeout == 10.0
        assert client.max_retries == 5

        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client works as async context manager."""
        async with AsyncMetadataStampingClient() as client:
            assert client._http_client is not None

        # Should be closed after context
        assert client._http_client is None


class TestHashGeneration:
    """Test hash generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_hash_success(self, metadata_client, mock_httpx_client):
        """Test successful hash generation."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "hash": "blake3_hash_123",
                "execution_time_ms": 1.5,
                "file_size_bytes": 1024,
                "performance_grade": "A",
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # Generate hash
        file_data = b"test content"
        correlation_id = uuid4()

        result = await metadata_client.generate_hash(
            file_data=file_data,
            namespace="test",
            correlation_id=correlation_id,
        )

        # Verify result
        assert result["hash"] == "blake3_hash_123"
        assert result["execution_time_ms"] == 1.5
        assert result["performance_grade"] == "A"

        # Verify request was made correctly
        mock_httpx_client.request.assert_called_once()
        call_kwargs = mock_httpx_client.request.call_args.kwargs
        assert call_kwargs["method"] == "POST"
        assert "/hash" in call_kwargs["url"]
        assert call_kwargs["params"]["namespace"] == "test"
        assert call_kwargs["headers"]["X-Correlation-ID"] == str(correlation_id)

    @pytest.mark.asyncio
    async def test_generate_hash_failure(self, metadata_client, mock_httpx_client):
        """Test hash generation failure handling."""
        # Mock error response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "error",
            "error": "Invalid file data",
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # Should raise ClientError
        from omninode_bridge.clients.base_client import ClientError

        with pytest.raises(ClientError, match="Hash generation failed"):
            await metadata_client.generate_hash(file_data=b"test")


class TestStampCreation:
    """Test metadata stamp creation."""

    @pytest.mark.asyncio
    async def test_create_stamp_success(self, metadata_client, mock_httpx_client):
        """Test successful stamp creation."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "stamp_id": str(uuid4()),
                "file_hash": "blake3_hash_123",
                "stamped_content": "content with stamp",
                "stamp": "stamp_data",
                "created_at": "2025-10-02T12:00:00Z",
                "op_id": str(uuid4()),
                "namespace": "omninode.services.metadata",
                "version": 1,
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # Create stamp
        result = await metadata_client.create_stamp(
            file_hash="blake3_hash_123",
            file_path="/path/to/file",
            file_size=1024,
            stamp_data={"key": "value"},
            content_type="text/plain",
            namespace="omninode.services.metadata",
            correlation_id=uuid4(),
        )

        # Verify result
        assert result["file_hash"] == "blake3_hash_123"
        assert result["namespace"] == "omninode.services.metadata"
        assert result["version"] == 1

        # Verify request
        mock_httpx_client.request.assert_called_once()
        call_kwargs = mock_httpx_client.request.call_args.kwargs
        assert call_kwargs["method"] == "POST"
        assert "/stamp" in call_kwargs["url"]

    @pytest.mark.asyncio
    async def test_create_stamp_with_defaults(self, metadata_client, mock_httpx_client):
        """Test stamp creation with default namespace."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"stamp_id": str(uuid4())},
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        await metadata_client.create_stamp(
            file_hash="hash",
            file_path="/path",
            file_size=100,
            stamp_data={},
        )

        # Verify default namespace was used
        call_kwargs = mock_httpx_client.request.call_args.kwargs
        request_body = call_kwargs["json"]
        assert request_body["namespace"] == "omninode.services.metadata"


class TestStampValidation:
    """Test stamp validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_stamp_success(self, metadata_client, mock_httpx_client):
        """Test successful stamp validation."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "validation_result": True,
                "stamps_found": 5,
                "valid_stamps": 5,
                "invalid_stamps": 0,
                "details": [],
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # Validate stamps
        result = await metadata_client.validate_stamp(
            content="content with stamps",
            namespace="test",
        )

        # Verify result
        assert result["validation_result"] is True
        assert result["stamps_found"] == 5
        assert result["valid_stamps"] == 5

    @pytest.mark.asyncio
    async def test_validate_stamp_partial_success(
        self, metadata_client, mock_httpx_client
    ):
        """Test partial validation success."""
        # Mock partial response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "partial",
            "data": {
                "validation_result": False,
                "stamps_found": 5,
                "valid_stamps": 3,
                "invalid_stamps": 2,
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # Should not raise error for partial status
        result = await metadata_client.validate_stamp(content="content")
        assert result["invalid_stamps"] == 2


class TestStampRetrieval:
    """Test stamp retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_stamp_success(self, metadata_client, mock_httpx_client):
        """Test successful stamp retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "stamp_id": str(uuid4()),
                "file_hash": "blake3_hash_123",
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        result = await metadata_client.get_stamp(
            file_hash="blake3_hash_123",
            namespace="test",
        )

        assert result is not None
        assert result["file_hash"] == "blake3_hash_123"

    @pytest.mark.asyncio
    async def test_get_stamp_not_found(self, metadata_client, mock_httpx_client):
        """Test stamp not found handling."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "error",
            "error": "Not found",
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        result = await metadata_client.get_stamp(file_hash="nonexistent")

        # Should return None for not found
        assert result is None


class TestBatchOperations:
    """Test batch stamping operations."""

    @pytest.mark.asyncio
    async def test_batch_create_stamps_success(
        self, metadata_client, mock_httpx_client
    ):
        """Test successful batch stamp creation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "total": 3,
                "successful": 3,
                "failed": 0,
                "results": [{"stamp_id": str(uuid4())} for _ in range(3)],
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        items = [
            {
                "file_hash": f"hash_{i}",
                "file_path": f"/path/{i}",
                "file_size": 100 * i,
                "stamp_data": {},
            }
            for i in range(3)
        ]

        result = await metadata_client.batch_create_stamps(items=items)

        assert result["total"] == 3
        assert result["successful"] == 3
        assert result["failed"] == 0


class TestClientMetrics:
    """Test client metrics collection."""

    @pytest.mark.asyncio
    async def test_get_service_metrics(self, metadata_client, mock_httpx_client):
        """Test service metrics retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hash_generation": {"avg_time_ms": 1.5},
            "database": {"connection_pool_size": 20},
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        metrics = await metadata_client.get_service_metrics()

        assert "hash_generation" in metrics
        assert "database" in metrics

    @pytest.mark.asyncio
    async def test_client_metrics_tracking(self, metadata_client, mock_httpx_client):
        """Test client tracks its own metrics."""
        # Make a successful request
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"hash": "test"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        await metadata_client.generate_hash(b"test")

        # Check metrics
        metrics = metadata_client.get_metrics()
        assert metrics["service_name"] == "MetadataStampingService"
        assert metrics["request_count"] >= 1


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_state(self, metadata_client):
        """Test circuit breaker is properly initialized."""
        assert metadata_client.circuit_breaker is not None
        assert metadata_client.circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_in_metrics(self, metadata_client):
        """Test circuit breaker metrics are included."""
        metrics = metadata_client.get_metrics()
        assert "circuit_breaker" in metrics
        assert metrics["circuit_breaker"]["state"] == CircuitState.CLOSED.value


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, metadata_client, mock_httpx_client):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        health = await metadata_client.health_check()

        assert health["status"] == "healthy"
        assert "version" in health

    @pytest.mark.asyncio
    async def test_validate_connection(self, metadata_client, mock_httpx_client):
        """Test connection validation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        is_valid = await metadata_client._validate_connection()
        assert is_valid is True
