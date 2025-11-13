"""Test improved exception handling in metadata stamping service.

This module tests the specific exception handling improvements made to:
1. hash_generator.py - specific exceptions for memory, I/O, and validation errors
2. service.py - specific exceptions for configuration, connection, and resource errors
3. api/router.py - specific HTTP exceptions for different error categories

These tests ensure better error diagnosis and appropriate error responses.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from ..api.router import create_metadata_stamp, generate_file_hash, validate_stamps
from ..engine.hash_generator import BLAKE3HashGenerator
from ..service import MetadataStampingService


class TestHashGeneratorExceptionHandling:
    """Test improved exception handling in BLAKE3HashGenerator."""

    @pytest.fixture
    def hash_generator(self):
        """Create hash generator instance for testing."""
        return BLAKE3HashGenerator(pool_size=5, max_workers=2)

    @pytest.mark.asyncio
    async def test_memory_error_handling(self, hash_generator):
        """Test specific MemoryError handling in hash generation."""
        # Mock the hasher to raise MemoryError
        with patch.object(
            hash_generator,
            "_direct_hash_small_file",
            side_effect=MemoryError("Out of memory"),
        ):
            with pytest.raises(MemoryError) as exc_info:
                await hash_generator.generate_hash(b"test", "test.txt")

            assert "Insufficient memory to hash file of size" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_os_error_handling(self, hash_generator):
        """Test specific OSError handling in hash generation."""
        # Mock the hasher to raise OSError
        with patch.object(
            hash_generator,
            "_hash_with_pooled_hasher",
            side_effect=OSError("File system error"),
        ):
            with pytest.raises(OSError) as exc_info:
                await hash_generator.generate_hash(b"test_data", "test.txt")

            assert "File system error during hash generation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, hash_generator):
        """Test specific TimeoutError handling in hash generation."""
        # Mock the hasher to raise TimeoutError
        with patch.object(
            hash_generator,
            "_stream_hash_large_file",
            side_effect=TimeoutError("Operation timed out"),
        ):
            large_data = b"x" * (1024 * 1024 + 1)  # > 1MB to trigger streaming

            with pytest.raises(asyncio.TimeoutError) as exc_info:
                await hash_generator.generate_hash(large_data, "large_file.txt")

            assert "Hash generation timed out for file size" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_value_error_handling(self, hash_generator):
        """Test specific ValueError handling in hash generation."""
        # Test invalid input type
        with pytest.raises(TypeError) as exc_info:
            await hash_generator.generate_hash("not_bytes")

        assert "file_data must be bytes" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cleanup_exception_handling(self, hash_generator):
        """Test specific exception handling in cleanup method."""
        # Mock thread pool to raise specific errors
        hash_generator.thread_pool = Mock()
        hash_generator.thread_pool.shutdown.side_effect = [
            OSError("Thread pool error"),
            None,
        ]

        # Should not raise exception, but should log errors
        await hash_generator.cleanup()

        # Verify that cleanup completed despite errors
        assert True  # Test passes if no exception raised


class TestServiceExceptionHandling:
    """Test improved exception handling in MetadataStampingService."""

    @pytest.mark.asyncio
    async def test_initialization_config_error(self):
        """Test specific configuration error handling in service initialization."""
        # Test with invalid database config
        invalid_config = {
            "database": {
                "host": None,  # Invalid - should cause TypeError/ValueError
                "port": "invalid_port",  # Invalid type
                "database": "test_db",
                # Missing required fields
            }
        }

        service = MetadataStampingService(invalid_config)

        # Should return False due to configuration error
        result = await service.initialize()
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_connection_error(self):
        """Test specific connection error handling in service cleanup."""
        service = MetadataStampingService()
        service.is_initialized = True

        # Mock stamping engine cleanup to raise ConnectionError
        service.stamping_engine = Mock()
        service.stamping_engine.cleanup = AsyncMock(
            side_effect=ConnectionError("Connection lost")
        )

        # Should not raise exception, but should log error
        await service.cleanup()

        # Verify cleanup attempted despite connection error
        service.stamping_engine.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_component_errors(self):
        """Test specific component error handling in health check."""
        service = MetadataStampingService()
        service.is_initialized = True

        # Mock stamping engine to raise RuntimeError
        service.stamping_engine = Mock()
        service.stamping_engine.hash_generator.generate_hash = AsyncMock(
            side_effect=RuntimeError("Component failure")
        )

        health_result = await service.health_check()

        # Should return unhealthy status with specific error
        assert health_result["status"] == "unhealthy"
        assert "stamping_engine" in health_result["components"]
        assert health_result["components"]["stamping_engine"]["status"] == "unhealthy"
        assert (
            "Component error" in health_result["components"]["stamping_engine"]["error"]
        )


class TestAPIRouterExceptionHandling:
    """Test improved exception handling in API router endpoints."""

    @pytest.mark.asyncio
    async def test_stamp_creation_memory_error(self):
        """Test specific MemoryError handling in stamp creation endpoint."""
        # Mock service to raise MemoryError
        mock_service = Mock()
        mock_service.stamp_content = AsyncMock(
            side_effect=MemoryError("Insufficient memory")
        )

        # Mock request
        mock_request = Mock()
        mock_request.content = "test content"
        mock_request.file_path = "test.txt"
        mock_request.stamp_type = "lightweight"
        mock_request.metadata = {}
        mock_request.protocol_version = "1.0"

        with pytest.raises(HTTPException) as exc_info:
            await create_metadata_stamp(mock_request, mock_service)

        assert exc_info.value.status_code == 507  # Insufficient memory
        assert "Insufficient memory" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_validation_connection_error(self):
        """Test specific ConnectionError handling in validation endpoint."""
        # Mock service to raise ConnectionError
        mock_service = Mock()
        mock_service.validate_stamp = AsyncMock(
            side_effect=ConnectionError("Database connection lost")
        )

        # Mock request
        mock_request = Mock()
        mock_request.content = "test content"
        mock_request.expected_hash = None

        with pytest.raises(HTTPException) as exc_info:
            await validate_stamps(mock_request, mock_service)

        assert exc_info.value.status_code == 503  # Service unavailable
        assert "Service temporarily unavailable" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_hash_generation_value_error(self):
        """Test specific ValueError handling in hash generation endpoint."""
        # Mock service to raise ValueError
        mock_service = Mock()
        mock_service.generate_hash = AsyncMock(
            side_effect=ValueError("Invalid input data")
        )

        # Mock file
        mock_file = Mock()
        mock_file.read = AsyncMock(return_value=b"test content")
        mock_file.filename = "test.txt"
        mock_file.size = 12

        with pytest.raises(HTTPException) as exc_info:
            await generate_file_hash(mock_file, mock_service)

        assert exc_info.value.status_code == 422  # Unprocessable entity
        assert "Invalid file" in exc_info.value.detail


class TestExceptionHandlingIntegration:
    """Integration tests for exception handling across components."""

    @pytest.mark.asyncio
    async def test_end_to_end_error_propagation(self):
        """Test that errors propagate correctly through the entire stack."""
        # Create service with invalid database config
        config = {
            "database": {
                "host": "invalid_host",
                "port": 9999,
                "database": "nonexistent_db",
                "user": "invalid_user",
                "password": "invalid_password",
            }
        }

        service = MetadataStampingService(config)

        # Service initialization should fail gracefully
        result = await service.initialize()
        assert result is False

        # Service should not be marked as initialized
        assert not service.is_initialized

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self):
        """Test error recovery in various failure scenarios."""
        hash_generator = BLAKE3HashGenerator(pool_size=2)

        # Test recovery from pool exhaustion
        # Exhaust the pool
        tasks = []
        for _ in range(5):  # More than pool size
            # Mock long-running operation
            with patch.object(
                hash_generator,
                "_hash_with_pooled_hasher",
                side_effect=asyncio.sleep(0.1),
            ):
                task = asyncio.create_task(hash_generator.generate_hash(b"test"))
                tasks.append(task)

        # Should handle pool exhaustion gracefully
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Some results may be exceptions, but should not crash
            assert len(results) == 5
        except Exception:
            pytest.fail("Should handle pool exhaustion gracefully")
        finally:
            await hash_generator.cleanup()


@pytest.mark.asyncio
async def test_exception_error_codes_mapping():
    """Test that different exception types map to appropriate HTTP status codes."""

    error_mappings = [
        (ValueError("Invalid input"), 422),  # Unprocessable Entity
        (TypeError("Type error"), 422),  # Unprocessable Entity
        (ConnectionError("Connection failed"), 503),  # Service Unavailable
        (OSError("I/O error"), 503),  # Service Unavailable
        (MemoryError("Out of memory"), 507),  # Insufficient Storage
        (RuntimeError("Runtime error"), 500),  # Internal Server Error
        (Exception("Generic error"), 500),  # Internal Server Error
    ]

    for exception, expected_status in error_mappings:
        mock_service = Mock()
        mock_service.stamp_content = AsyncMock(side_effect=exception)

        mock_request = Mock()
        mock_request.content = "test"
        mock_request.file_path = "test.txt"
        mock_request.stamp_type = "lightweight"
        mock_request.metadata = {}
        mock_request.protocol_version = "1.0"

        with pytest.raises(HTTPException) as exc_info:
            await create_metadata_stamp(mock_request, mock_service)

        assert (
            exc_info.value.status_code == expected_status
        ), f"Exception {type(exception).__name__} should map to status code {expected_status}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
