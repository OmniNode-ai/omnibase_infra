"""
Unit tests for decorators.

Tests @timed, @counted decorators and timing() context manager.
"""

import asyncio
import time

import pytest

from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.metrics.decorators import (
    counted,
    get_metrics_collector,
    set_metrics_collector,
    timed,
    timing,
)


class TestDecorators:
    """Tests for metric decorators."""

    @pytest.fixture
    async def collector(self):
        """Create metrics collector for testing."""
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)
        set_metrics_collector(collector)
        yield collector
        set_metrics_collector(None)

    @pytest.mark.asyncio
    async def test_get_set_metrics_collector(self):
        """Test getting and setting global collector."""
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)

        set_metrics_collector(collector)
        assert get_metrics_collector() == collector

        set_metrics_collector(None)
        assert get_metrics_collector() is None

    @pytest.mark.asyncio
    async def test_timed_decorator_async(self, collector):
        """Test @timed decorator on async function."""

        @timed("test_operation_time_ms")
        async def test_operation():
            await asyncio.sleep(0.01)
            return "result"

        result = await test_operation()

        assert result == "result"

        # Give metric recording time to execute
        await asyncio.sleep(0.1)

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_timed_decorator_with_tags(self, collector):
        """Test @timed decorator with tags."""

        @timed("test_operation_time_ms", tags={"env": "test", "type": "unit"})
        async def test_operation():
            await asyncio.sleep(0.01)

        await test_operation()
        await asyncio.sleep(0.1)

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_timed_decorator_sync(self, collector):
        """Test @timed decorator on sync function."""

        @timed("sync_operation_time_ms")
        def sync_operation():
            time.sleep(0.01)
            return "sync_result"

        result = sync_operation()

        assert result == "sync_result"

        # Give metric recording time to execute
        await asyncio.sleep(0.1)

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_timed_decorator_multiple_calls(self, collector):
        """Test @timed decorator records multiple calls."""

        @timed("operation_time_ms")
        async def operation():
            await asyncio.sleep(0.005)

        # Call 3 times
        await operation()
        await operation()
        await operation()

        await asyncio.sleep(0.1)

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 3

    @pytest.mark.asyncio
    async def test_counted_decorator_async(self, collector):
        """Test @counted decorator on async function."""

        @counted("test_operation_count")
        async def test_operation():
            return "result"

        result = await test_operation()

        assert result == "result"

        await asyncio.sleep(0.1)

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_counted_decorator_with_tags(self, collector):
        """Test @counted decorator with tags."""

        @counted("cache_hit_count", tags={"cache": "template"})
        async def get_from_cache():
            return "cached_value"

        await get_from_cache()
        await asyncio.sleep(0.1)

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_counted_decorator_sync(self, collector):
        """Test @counted decorator on sync function."""

        @counted("sync_operation_count")
        def sync_operation():
            return "sync_result"

        result = sync_operation()

        assert result == "sync_result"

        await asyncio.sleep(0.1)

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_timing_context_manager(self, collector):
        """Test timing() context manager."""

        async with timing("block_time_ms"):
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.1)

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_timing_context_manager_with_tags(self, collector):
        """Test timing() context manager with tags."""

        async with timing("parse_time_ms", tags={"type": "yaml"}):
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.1)

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_timing_context_manager_with_exception(self, collector):
        """Test timing() context manager records time even on exception."""

        with pytest.raises(ValueError):
            async with timing("error_operation_time_ms"):
                await asyncio.sleep(0.01)
                raise ValueError("Test error")

        await asyncio.sleep(0.1)

        # Metric should still be recorded
        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_decorators_without_collector(self):
        """Test decorators work without collector (no-op)."""
        set_metrics_collector(None)

        @timed("test_time_ms")
        async def test_operation():
            return "result"

        @counted("test_count")
        async def test_count():
            return "counted"

        # Should not raise errors
        result1 = await test_operation()
        result2 = await test_count()

        assert result1 == "result"
        assert result2 == "counted"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_timed_decorator_overhead(self, collector):
        """Test @timed decorator overhead is minimal."""

        @timed("test_time_ms")
        async def fast_operation():
            pass

        # Measure overhead
        start = time.perf_counter()
        await fast_operation()
        duration_ms = (time.perf_counter() - start) * 1000

        # Overhead should be < 1ms
        assert duration_ms < 1.0, f"Decorator overhead: {duration_ms:.3f}ms (>1ms)"
