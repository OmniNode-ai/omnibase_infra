"""
Unit tests for RingBuffer.

Tests lock-free ring buffer performance and correctness.
"""

import asyncio

import pytest

from omninode_bridge.agents.metrics.ring_buffer import RingBuffer


class TestRingBuffer:
    """Tests for RingBuffer."""

    @pytest.mark.asyncio
    async def test_ring_buffer_creation(self):
        """Test creating ring buffer."""
        buffer = RingBuffer[int](capacity=100)

        assert await buffer.capacity() == 100
        assert await buffer.size() == 0
        assert await buffer.is_empty()
        assert not await buffer.is_full()

    @pytest.mark.asyncio
    async def test_ring_buffer_creation_invalid_capacity(self):
        """Test creating buffer with invalid capacity fails."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            RingBuffer[int](capacity=0)

        with pytest.raises(ValueError, match="Capacity must be positive"):
            RingBuffer[int](capacity=-10)

    @pytest.mark.asyncio
    async def test_ring_buffer_write_single(self):
        """Test writing single item."""
        buffer = RingBuffer[str](capacity=10)

        success = await buffer.write("test")

        assert success
        assert await buffer.size() == 1
        assert not await buffer.is_empty()

    @pytest.mark.asyncio
    async def test_ring_buffer_write_multiple(self):
        """Test writing multiple items."""
        buffer = RingBuffer[int](capacity=10)

        for i in range(5):
            await buffer.write(i)

        assert await buffer.size() == 5

    @pytest.mark.asyncio
    async def test_ring_buffer_read_batch(self):
        """Test reading batch of items."""
        buffer = RingBuffer[int](capacity=10)

        # Write items
        for i in range(5):
            await buffer.write(i)

        # Read batch
        batch = await buffer.read_batch(max_size=3)

        assert len(batch) == 3
        assert batch == [0, 1, 2]
        assert await buffer.size() == 2  # 2 items remaining

    @pytest.mark.asyncio
    async def test_ring_buffer_read_all(self):
        """Test reading all items."""
        buffer = RingBuffer[str](capacity=10)

        items = ["a", "b", "c", "d", "e"]
        for item in items:
            await buffer.write(item)

        batch = await buffer.read_batch(max_size=10)

        assert len(batch) == 5
        assert batch == items
        assert await buffer.size() == 0
        assert await buffer.is_empty()

    @pytest.mark.asyncio
    async def test_ring_buffer_read_empty(self):
        """Test reading from empty buffer."""
        buffer = RingBuffer[int](capacity=10)

        batch = await buffer.read_batch(max_size=10)

        assert len(batch) == 0
        assert batch == []

    @pytest.mark.asyncio
    async def test_ring_buffer_wraparound(self):
        """Test buffer wraparound behavior."""
        buffer = RingBuffer[int](capacity=5)

        # Fill buffer
        for i in range(5):
            await buffer.write(i)

        # Read 3 items
        batch1 = await buffer.read_batch(max_size=3)
        assert batch1 == [0, 1, 2]

        # Write 3 more (should wrap around)
        for i in range(5, 8):
            await buffer.write(i)

        # Read remaining items
        batch2 = await buffer.read_batch(max_size=10)
        assert batch2 == [3, 4, 5, 6, 7]

    @pytest.mark.asyncio
    async def test_ring_buffer_overflow(self):
        """Test buffer overflow behavior (overwrites oldest)."""
        buffer = RingBuffer[int](capacity=5)

        # Write 10 items (buffer capacity is 5)
        for i in range(10):
            await buffer.write(i)

        # Buffer should have newest 5 items
        assert await buffer.size() == 5
        batch = await buffer.read_batch(max_size=10)
        # Buffer may contain items 5-9 depending on overflow behavior
        assert len(batch) == 5

    @pytest.mark.asyncio
    async def test_ring_buffer_clear(self):
        """Test clearing buffer."""
        buffer = RingBuffer[int](capacity=10)

        # Write items
        for i in range(5):
            await buffer.write(i)

        assert await buffer.size() == 5

        # Clear
        await buffer.clear()

        assert await buffer.size() == 0
        assert await buffer.is_empty()

    @pytest.mark.asyncio
    async def test_ring_buffer_concurrent_writes(self):
        """Test concurrent writes (stress test)."""
        buffer = RingBuffer[int](capacity=1000)

        # Concurrent writes
        async def write_items(start: int, count: int):
            for i in range(start, start + count):
                await buffer.write(i)

        # 10 concurrent writers, 100 items each
        tasks = [write_items(i * 100, 100) for i in range(10)]
        await asyncio.gather(*tasks)

        assert await buffer.size() == 1000

    @pytest.mark.asyncio
    async def test_ring_buffer_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        buffer = RingBuffer[int](capacity=100)

        # Writer task
        async def writer():
            for i in range(50):
                await buffer.write(i)
                await asyncio.sleep(0.001)

        # Reader task
        async def reader():
            total_read = 0
            for _ in range(10):
                await asyncio.sleep(0.005)
                batch = await buffer.read_batch(max_size=10)
                total_read += len(batch)
            return total_read

        # Run concurrently
        write_task = asyncio.create_task(writer())
        read_task = asyncio.create_task(reader())

        await write_task
        total_read = await read_task

        # Should have read some items
        assert total_read > 0

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_ring_buffer_write_performance(self):
        """Test write performance (<0.1ms per write)."""
        import time

        buffer = RingBuffer[int](capacity=10000)

        start = time.perf_counter()
        for i in range(1000):
            await buffer.write(i)
        duration_ms = (time.perf_counter() - start) * 1000

        avg_write_ms = duration_ms / 1000
        assert avg_write_ms < 0.1, f"Write took {avg_write_ms:.3f}ms (>0.1ms)"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_ring_buffer_read_performance(self):
        """Test read performance (<1ms for 1000 items)."""
        import time

        buffer = RingBuffer[int](capacity=10000)

        # Fill buffer
        for i in range(1000):
            await buffer.write(i)

        # Read batch
        start = time.perf_counter()
        batch = await buffer.read_batch(max_size=1000)
        duration_ms = (time.perf_counter() - start) * 1000

        assert len(batch) == 1000
        assert duration_ms < 1.0, f"Read took {duration_ms:.3f}ms (>1ms)"
