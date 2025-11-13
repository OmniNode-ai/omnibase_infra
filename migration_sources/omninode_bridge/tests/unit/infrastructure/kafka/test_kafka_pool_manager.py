"""
Unit tests for KafkaConnectionPool.

This test suite verifies:
1. Pool initialization and shutdown
2. Producer acquisition and release
3. Timeout handling (UnboundLocalError fix)
4. Health checks and metrics
5. Error recovery and producer recycling
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from omninode_bridge.infrastructure.kafka.kafka_pool_manager import KafkaConnectionPool

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_producer():
    """Create a mock AIOKafkaProducer."""
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer._closed = False
    return producer


@pytest.fixture
async def pool(mock_producer):
    """Create a KafkaConnectionPool with mocked producer creation."""
    pool = KafkaConnectionPool(
        bootstrap_servers="localhost:29092",
        pool_size=2,
        max_wait_ms=1000,
        health_check_interval=60,
    )

    # Mock producer creation
    with patch.object(pool, "_create_producer", return_value=mock_producer):
        await pool.initialize()

    yield pool

    # Cleanup
    await pool.shutdown()


# ============================================================================
# Test Suite 1: Pool Initialization
# ============================================================================


@pytest.mark.asyncio
async def test_pool_initialization(mock_producer):
    """Test that pool initializes with correct number of producers."""
    pool = KafkaConnectionPool(pool_size=3)

    with patch.object(pool, "_create_producer", return_value=mock_producer):
        await pool.initialize()

    try:
        assert pool.is_initialized
        assert len(pool._pool) == 3
        assert pool._available.qsize() == 3
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_double_initialization(mock_producer):
    """Test that pool can't be initialized twice."""
    pool = KafkaConnectionPool(pool_size=2)

    with patch.object(pool, "_create_producer", return_value=mock_producer):
        await pool.initialize()
        await pool.initialize()  # Should be no-op

    try:
        assert len(pool._pool) == 2  # Still only 2 producers
    finally:
        await pool.shutdown()


# ============================================================================
# Test Suite 2: Producer Acquisition (Bug Fix Validation)
# ============================================================================


@pytest.mark.asyncio
async def test_acquire_timeout_no_unbound_error():
    """
    Test that timeout during producer acquisition doesn't cause UnboundLocalError.

    This test validates the fix for the critical bug where wrapper was referenced
    in the finally block but never assigned if a timeout occurred.
    """
    pool = KafkaConnectionPool(pool_size=1, max_wait_ms=100)

    # Mock _create_producer to create a valid producer
    mock_producer = AsyncMock()
    mock_producer.start = AsyncMock()
    mock_producer.stop = AsyncMock()
    mock_producer._closed = False

    with patch.object(pool, "_create_producer", return_value=mock_producer):
        await pool.initialize()

    try:
        # Acquire the only producer
        async with pool.acquire():
            # Now try to acquire another producer with short timeout
            # This should timeout since pool size is 1 and producer is in use
            with pytest.raises(TimeoutError) as exc_info:
                async with pool.acquire(timeout_ms=50):
                    pass  # This should never execute

            # Verify the timeout error message
            assert "No producer available" in str(exc_info.value)

            # Key validation: No UnboundLocalError should occur
            # The fix ensures wrapper is initialized to None before try block
            # and checked for None in finally block

        # After first producer is released, we should be able to acquire again
        async with pool.acquire():
            pass  # Should succeed now

    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_acquire_and_release_updates_metrics(pool):
    """Test that acquire and release update metrics correctly."""
    initial_acquisitions = pool.metrics.total_acquisitions
    initial_releases = pool.metrics.total_releases

    async with pool.acquire():
        # During acquisition
        assert pool.metrics.total_acquisitions == initial_acquisitions + 1
        assert pool.metrics.current_utilization > 0

    # After release
    assert pool.metrics.total_releases == initial_releases + 1
    assert pool.metrics.current_utilization == 0


@pytest.mark.asyncio
async def test_acquire_unhealthy_producer_recreates(mock_producer):
    """Test that unhealthy producers are recreated on acquisition."""
    pool = KafkaConnectionPool(pool_size=1)

    new_producer = AsyncMock()
    new_producer.start = AsyncMock()
    new_producer.stop = AsyncMock()

    with patch.object(pool, "_create_producer", return_value=mock_producer):
        await pool.initialize()

        # Mark producer as unhealthy
        pool._pool[0].is_healthy = False

        # Mock _create_producer for recreation
        with patch.object(pool, "_create_producer", return_value=new_producer):
            async with pool.acquire():
                pass

        # Verify producer was recreated
        assert pool._pool[0].is_healthy
        assert pool._pool[0].error_count == 0
        assert pool.metrics.total_reconnections == 1

    await pool.shutdown()


# ============================================================================
# Test Suite 3: Metrics and Health
# ============================================================================


@pytest.mark.asyncio
async def test_get_metrics(pool):
    """Test that get_metrics returns correct structure."""
    metrics = pool.get_metrics()

    assert "pool_size" in metrics
    assert "current_utilization" in metrics
    assert "utilization_percentage" in metrics
    assert "peak_utilization" in metrics
    assert "total_acquisitions" in metrics
    assert "total_releases" in metrics
    assert "average_wait_time_ms" in metrics
    assert "total_errors" in metrics
    assert "total_reconnections" in metrics
    assert "uptime_seconds" in metrics
    assert "is_initialized" in metrics

    assert metrics["pool_size"] == 2
    assert metrics["is_initialized"] is True


@pytest.mark.asyncio
async def test_timeout_increments_error_count():
    """Test that timeout errors increment the error count metric."""
    pool = KafkaConnectionPool(pool_size=1, max_wait_ms=100)

    mock_producer = AsyncMock()
    mock_producer.start = AsyncMock()
    mock_producer.stop = AsyncMock()

    with patch.object(pool, "_create_producer", return_value=mock_producer):
        await pool.initialize()

    try:
        initial_errors = pool.metrics.total_errors

        # Acquire the only producer
        async with pool.acquire():
            # Try to acquire another (should timeout)
            with pytest.raises(TimeoutError):
                async with pool.acquire(timeout_ms=50):
                    pass

        # Verify error count increased
        assert pool.metrics.total_errors == initial_errors + 1

    finally:
        await pool.shutdown()


# ============================================================================
# Test Suite 4: Pool Lifecycle
# ============================================================================


@pytest.mark.asyncio
async def test_pool_shutdown_closes_all_producers(mock_producer):
    """Test that shutdown closes all producers."""
    pool = KafkaConnectionPool(pool_size=3)

    producers = [AsyncMock() for _ in range(3)]
    for p in producers:
        p.start = AsyncMock()
        p.stop = AsyncMock()
        p._closed = False

    with patch.object(pool, "_create_producer", side_effect=producers):
        await pool.initialize()

    await pool.shutdown()

    # Verify all producers were stopped
    for p in producers:
        p.stop.assert_called_once()

    assert not pool.is_initialized
    assert len(pool._pool) == 0


@pytest.mark.asyncio
async def test_pool_context_manager():
    """Test that pool works as async context manager."""
    mock_producer = AsyncMock()
    mock_producer.start = AsyncMock()
    mock_producer.stop = AsyncMock()
    mock_producer._closed = False

    pool = KafkaConnectionPool(pool_size=2)

    with patch.object(pool, "_create_producer", return_value=mock_producer):
        async with pool:
            # Pool should be initialized
            assert pool.is_initialized

            async with pool.acquire():
                pass

    # Pool should be shutdown after context exit
    # Note: The __aexit__ calls shutdown, but we can't easily verify
    # state after context exit in the same scope


# ============================================================================
# Test Suite 5: Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_acquire_before_initialization():
    """Test that acquire fails before initialization."""
    pool = KafkaConnectionPool()

    with pytest.raises(RuntimeError, match="Pool not initialized"):
        async with pool.acquire():
            pass


@pytest.mark.asyncio
async def test_multiple_concurrent_acquisitions(pool):
    """Test that multiple concurrent acquisitions work correctly."""

    async def acquire_and_hold(duration_ms: int):
        async with pool.acquire():
            await asyncio.sleep(duration_ms / 1000.0)

    # Run 5 concurrent acquisitions (pool size is 2)
    # Some will wait for producers to become available
    tasks = [acquire_and_hold(10) for _ in range(5)]
    await asyncio.gather(*tasks)

    # All should complete successfully
    assert pool.metrics.total_acquisitions == 5
    assert pool.metrics.total_releases == 5
    assert pool.metrics.current_utilization == 0


@pytest.mark.asyncio
async def test_peak_utilization_tracking(pool):
    """Test that peak utilization is tracked correctly."""

    async def acquire_and_hold():
        async with pool.acquire():
            await asyncio.sleep(0.1)

    # Acquire both producers simultaneously
    await asyncio.gather(acquire_and_hold(), acquire_and_hold())

    # Peak utilization should be 2 (pool size)
    assert pool.metrics.peak_utilization == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
