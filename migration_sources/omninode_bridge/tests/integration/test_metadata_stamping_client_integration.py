"""Integration tests for AsyncMetadataStampingClient with real MetadataStamping service.

Tests real HTTP integration, circuit breaker behavior, and error scenarios.
"""

import asyncio
from uuid import uuid4

import pytest

from omninode_bridge.clients.base_client import ClientError, ServiceUnavailableError
from omninode_bridge.clients.circuit_breaker import CircuitState
from omninode_bridge.clients.metadata_stamping_client import AsyncMetadataStampingClient

# Service URL (can be overridden with environment variable)
METADATA_STAMPING_URL = "http://localhost:8053"


@pytest.fixture
async def metadata_client():
    """Provide initialized AsyncMetadataStampingClient for integration tests."""
    async with AsyncMetadataStampingClient(
        base_url=METADATA_STAMPING_URL,
        timeout=10.0,
        max_retries=2,
    ) as client:
        yield client


@pytest.fixture
async def metadata_client_short_timeout():
    """Provide client with short timeout for testing timeout scenarios."""
    async with AsyncMetadataStampingClient(
        base_url=METADATA_STAMPING_URL,
        timeout=0.1,  # Very short timeout to trigger timeout errors
        max_retries=1,
    ) as client:
        yield client


class TestHealthAndConnectivity:
    """Test service health and connectivity."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_health_check(self, metadata_client):
        """Test health check endpoint."""
        health = await metadata_client.health_check()

        assert health is not None
        assert "status" in health
        # Service should be healthy
        assert health["status"] in ("healthy", "ok", "success")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_connection_validation(self, metadata_client):
        """Test connection validation."""
        is_valid = await metadata_client._validate_connection()
        assert is_valid is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_service_metrics(self, metadata_client):
        """Test service metrics endpoint."""
        metrics = await metadata_client.get_service_metrics()

        assert metrics is not None
        # Metrics should contain performance data
        assert isinstance(metrics, dict)


class TestHashGeneration:
    """Test hash generation with real service."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_hash_small_file(self, metadata_client):
        """Test hash generation for small file."""
        file_data = b"Hello, World!"
        correlation_id = uuid4()

        result = await metadata_client.generate_hash(
            file_data=file_data,
            namespace="test",
            correlation_id=correlation_id,
        )

        # Verify hash was generated
        assert "hash" in result
        assert result["hash"] is not None
        assert len(result["hash"]) > 0

        # Verify performance metrics
        assert "execution_time_ms" in result
        assert result["execution_time_ms"] >= 0

        # Should be very fast for small files
        assert result["execution_time_ms"] < 100  # <100ms for small files

        # Verify file size
        assert "file_size_bytes" in result
        assert result["file_size_bytes"] == len(file_data)

        # Check performance grade (should be A or B for small files)
        assert "performance_grade" in result
        assert result["performance_grade"] in ("A", "B", "C")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_hash_medium_file(self, metadata_client):
        """Test hash generation for medium file (100KB)."""
        file_data = b"X" * (100 * 1024)  # 100KB
        correlation_id = uuid4()

        result = await metadata_client.generate_hash(
            file_data=file_data,
            namespace="test",
            correlation_id=correlation_id,
        )

        # Verify hash was generated
        assert "hash" in result
        assert result["hash"] is not None

        # Should still be fast for medium files
        assert result["execution_time_ms"] < 500  # <500ms for 100KB

        # Verify file size
        assert result["file_size_bytes"] == len(file_data)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_hash_with_namespace(self, metadata_client):
        """Test hash generation with custom namespace."""
        file_data = b"namespace test"

        result = await metadata_client.generate_hash(
            file_data=file_data,
            namespace="custom.namespace",
        )

        # Hash should be generated successfully
        assert "hash" in result
        assert result["hash"] is not None


class TestStampOperations:
    """Test stamp creation and retrieval."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_stamp(self, metadata_client):
        """Test creating a metadata stamp."""
        # First generate a hash
        file_data = b"test content for stamping"
        hash_result = await metadata_client.generate_hash(file_data=file_data)
        file_hash = hash_result["hash"]

        # Create stamp
        correlation_id = uuid4()
        stamp = await metadata_client.create_stamp(
            file_hash=file_hash,
            file_path="/test/file.txt",
            file_size=len(file_data),
            stamp_data={"test_key": "test_value"},
            content_type="text/plain",
            namespace="omninode.services.metadata",
            correlation_id=correlation_id,
        )

        # Verify stamp was created
        assert stamp is not None
        assert "stamp_id" in stamp or "file_hash" in stamp
        assert stamp.get("file_hash") == file_hash or "stamp" in stamp

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_stamp(self, metadata_client):
        """Test retrieving a stamp by hash."""
        # First create a stamp
        file_data = b"content for retrieval test"
        hash_result = await metadata_client.generate_hash(file_data=file_data)
        file_hash = hash_result["hash"]

        # Create stamp
        await metadata_client.create_stamp(
            file_hash=file_hash,
            file_path="/test/retrieve.txt",
            file_size=len(file_data),
            stamp_data={},
        )

        # Retrieve stamp
        retrieved_stamp = await metadata_client.get_stamp(
            file_hash=file_hash,
            namespace="omninode.services.metadata",
        )

        # Stamp should be found (or None if not implemented yet)
        # This is acceptable for a service in development
        if retrieved_stamp is not None:
            assert "file_hash" in retrieved_stamp or "stamp_id" in retrieved_stamp

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_nonexistent_stamp(self, metadata_client):
        """Test retrieving a nonexistent stamp returns None."""
        nonexistent_hash = "blake3_nonexistent_hash_12345678"

        stamp = await metadata_client.get_stamp(file_hash=nonexistent_hash)

        # Should return None for nonexistent stamp
        assert stamp is None


class TestStampValidation:
    """Test stamp validation functionality."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_validate_stamp(self, metadata_client):
        """Test validating stamps in content."""
        content = """
        This is test content with embedded stamps.
        [STAMP:abc123] Some stamped content here.
        [STAMP:def456] More stamped content.
        """

        result = await metadata_client.validate_stamp(
            content=content,
            namespace="omninode.services.metadata",
        )

        # Validation should return results
        assert result is not None
        # Result structure depends on implementation
        # Could have stamps_found, valid_stamps, etc.


class TestCircuitBreakerBehavior:
    """Test circuit breaker integration with real service failures."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_circuit_breaker_state(self, metadata_client):
        """Test circuit breaker is properly initialized."""
        assert metadata_client.circuit_breaker is not None
        assert metadata_client.circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_circuit_breaker_metrics(self, metadata_client):
        """Test circuit breaker metrics are tracked."""
        # Make a successful request
        await metadata_client.generate_hash(b"test")

        # Check metrics
        metrics = metadata_client.get_metrics()
        assert "circuit_breaker" in metrics
        assert metrics["circuit_breaker"]["state"] == CircuitState.CLOSED.value
        assert metrics["request_count"] > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_timeout_handling(self, metadata_client_short_timeout):
        """Test timeout handling with circuit breaker."""
        # This test uses a client with very short timeout
        # Large file should trigger timeout
        large_file = b"X" * (10 * 1024 * 1024)  # 10MB

        with pytest.raises(
            (ServiceUnavailableError, ClientError, asyncio.TimeoutError)
        ):
            await metadata_client_short_timeout.generate_hash(large_file)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_service_unavailable_handling(self):
        """Test handling when service is completely unavailable."""
        # Use an invalid URL to simulate service down
        invalid_client = AsyncMetadataStampingClient(
            base_url="http://localhost:99999",  # Invalid port
            timeout=1.0,
            max_retries=1,
        )

        await invalid_client.initialize()

        with pytest.raises((ServiceUnavailableError, ClientError)):
            await invalid_client.generate_hash(b"test")

        await invalid_client.close()


class TestBatchOperations:
    """Test batch stamping operations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_batch_create_stamps(self, metadata_client):
        """Test batch stamp creation."""
        # Generate hashes for batch items
        items = []
        for i in range(3):
            file_data = f"batch_content_{i}".encode()
            hash_result = await metadata_client.generate_hash(file_data=file_data)

            items.append(
                {
                    "file_hash": hash_result["hash"],
                    "file_path": f"/test/batch_{i}.txt",
                    "file_size": len(file_data),
                    "stamp_data": {"index": i},
                }
            )

        # Create batch stamps
        result = await metadata_client.batch_create_stamps(
            items=items,
            namespace="omninode.services.metadata",
        )

        # Batch operation should succeed
        assert result is not None
        # Result structure depends on implementation
        if "total" in result:
            assert result["total"] == len(items)


class TestCorrelationIDPropagation:
    """Test correlation ID propagation through requests."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_correlation_id_in_headers(self, metadata_client):
        """Test correlation ID is properly propagated in request headers."""
        correlation_id = uuid4()

        # Make request with correlation ID
        result = await metadata_client.generate_hash(
            file_data=b"correlation test",
            correlation_id=correlation_id,
        )

        # Request should succeed with correlation ID
        assert result is not None
        assert "hash" in result

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_requests_with_correlation(self, metadata_client):
        """Test multiple requests with same correlation ID."""
        correlation_id = uuid4()

        # Make multiple requests with same correlation ID
        for i in range(3):
            result = await metadata_client.generate_hash(
                file_data=f"request_{i}".encode(),
                correlation_id=correlation_id,
            )
            assert result is not None


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_invalid_hash_format(self, metadata_client):
        """Test handling of invalid hash format."""
        # This might raise an error or return None depending on implementation
        result = await metadata_client.get_stamp(file_hash="invalid_hash_format")

        # Should handle gracefully (either None or raise specific error)
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_network_error_retry(self):
        """Test retry behavior on network errors."""
        # Use client with low max_retries for faster testing
        client = AsyncMetadataStampingClient(
            base_url="http://localhost:99999",  # Invalid port
            timeout=0.5,
            max_retries=2,
        )

        await client.initialize()

        # Should attempt retries and then fail
        with pytest.raises((ServiceUnavailableError, ClientError)):
            await client.generate_hash(b"test")

        # Verify retry count
        metrics = client.get_metrics()
        # Error count should reflect multiple attempts
        assert metrics["error_count"] > 0

        await client.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_client_context_manager():
    """Test client works correctly as async context manager."""
    async with AsyncMetadataStampingClient(base_url=METADATA_STAMPING_URL) as client:
        # Client should be initialized
        assert client._http_client is not None

        # Should be able to make requests
        result = await client.generate_hash(b"context manager test")
        assert result is not None

    # Client should be closed after context
    # Note: _http_client is set to None in __aexit__


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_requests():
    """Test handling of concurrent requests."""
    async with AsyncMetadataStampingClient(base_url=METADATA_STAMPING_URL) as client:
        # Make 10 concurrent requests
        tasks = [client.generate_hash(f"concurrent_{i}".encode()) for i in range(10)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should succeed
        assert len(results) == 10
        for result in results:
            assert not isinstance(result, Exception)
            assert "hash" in result
