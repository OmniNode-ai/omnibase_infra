"""Unit tests for CacheManager."""

import asyncio

import pytest

from omninode_bridge.caching import CacheManager


@pytest.mark.asyncio
async def test_cache_manager_initialization():
    """Test cache manager initializes correctly."""
    cache = CacheManager(backend="memory")
    await cache.initialize()

    assert cache._initialized
    assert cache.backend == "memory"

    await cache.shutdown()


@pytest.mark.asyncio
async def test_cache_set_and_get_intelligence():
    """Test caching intelligence query results."""
    cache = CacheManager(backend="memory")
    await cache.initialize()

    # Set intelligence result
    success = await cache.set_intelligence_result(
        query="test query",
        context={"domain": "test"},
        result={"data": "test result"},
        ttl=3600,
    )
    assert success

    # Get intelligence result
    result = await cache.get_intelligence_result(
        query="test query", context={"domain": "test"}
    )
    assert result is not None
    assert result["data"] == "test result"

    # Verify metrics
    metrics = cache.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 0
    assert metrics["hit_rate"] == 1.0

    await cache.shutdown()


@pytest.mark.asyncio
async def test_cache_miss():
    """Test cache miss returns None."""
    cache = CacheManager(backend="memory")
    await cache.initialize()

    # Get non-existent key
    result = await cache.get_intelligence_result(
        query="missing query", context={"domain": "test"}
    )
    assert result is None

    # Verify metrics
    metrics = cache.get_metrics()
    assert metrics["misses"] == 1
    assert metrics["hit_rate"] == 0.0

    await cache.shutdown()


@pytest.mark.asyncio
async def test_cache_ttl_expiration():
    """Test cache TTL expiration."""
    cache = CacheManager(backend="memory")
    await cache.initialize()

    # Set with very short TTL
    await cache.set_intelligence_result(
        query="expiring query",
        context={"domain": "test"},
        result={"data": "expiring"},
        ttl=1,  # 1 second
    )

    # Should exist immediately
    result = await cache.get_intelligence_result(
        query="expiring query", context={"domain": "test"}
    )
    assert result is not None

    # Wait for expiration
    await asyncio.sleep(1.5)

    # Should be expired
    result = await cache.get_intelligence_result(
        query="expiring query", context={"domain": "test"}
    )
    assert result is None

    await cache.shutdown()


@pytest.mark.asyncio
async def test_cache_pattern_invalidation():
    """Test pattern-based cache invalidation."""
    cache = CacheManager(backend="memory")
    await cache.initialize()

    # Set multiple entries
    await cache.set_intelligence_result(
        "query1", {"domain": "test"}, {"data": "1"}, ttl=3600
    )
    await cache.set_intelligence_result(
        "query2", {"domain": "test"}, {"data": "2"}, ttl=3600
    )
    await cache.set_contract_template(
        "effect", "my_effect", {"template": "data"}, ttl=3600
    )

    # Invalidate intelligence entries
    count = await cache.invalidate_pattern("intelligence:*")
    assert count == 2

    # Verify intelligence entries are gone
    result = await cache.get_intelligence_result("query1", {"domain": "test"})
    assert result is None

    # Contract should still exist
    result = await cache.get_contract_template("effect", "my_effect")
    assert result is not None

    await cache.shutdown()


@pytest.mark.asyncio
async def test_cache_multiple_types():
    """Test caching multiple data types."""
    cache = CacheManager(backend="memory")
    await cache.initialize()

    # Cache intelligence
    await cache.set_intelligence_result(
        "query", {"domain": "test"}, {"data": "intelligence"}
    )

    # Cache contract
    await cache.set_contract_template("effect", "my_effect", {"template": "contract"})

    # Cache pattern
    await cache.set_validated_pattern("singleton", "python", {"valid": True})

    # Cache code snippet
    await cache.set_code_snippet("abc123", {"code": "print('hello')"})

    # Verify all types
    assert (
        await cache.get_intelligence_result("query", {"domain": "test"})
    ) is not None
    assert await cache.get_contract_template("effect", "my_effect") is not None
    assert await cache.get_validated_pattern("singleton", "python") is not None
    assert await cache.get_code_snippet("abc123") is not None

    metrics = cache.get_metrics()
    assert metrics["hit_rate"] == 1.0

    await cache.shutdown()


@pytest.mark.asyncio
async def test_cache_metrics():
    """Test cache metrics tracking."""
    cache = CacheManager(backend="memory")
    await cache.initialize()

    # Generate some cache activity
    await cache.set_intelligence_result("q1", {}, {"data": "1"})
    await cache.get_intelligence_result("q1", {})  # hit
    await cache.get_intelligence_result("q2", {})  # miss
    await cache.get_intelligence_result("q3", {})  # miss

    metrics = cache.get_metrics()

    # Verify metrics structure
    assert "hits" in metrics
    assert "misses" in metrics
    assert "hit_rate" in metrics
    assert "invalidations" in metrics
    assert "errors" in metrics
    assert "backend" in metrics
    assert "uptime_seconds" in metrics

    # Verify values
    assert metrics["hits"] == 1
    assert metrics["misses"] == 2
    assert abs(metrics["hit_rate"] - 0.3333) < 0.01  # 1/3

    await cache.shutdown()


@pytest.mark.asyncio
async def test_cache_context_manager():
    """Test cache as async context manager."""
    async with CacheManager(backend="memory") as cache:
        await cache.set_intelligence_result("test", {}, {"data": "value"})
        result = await cache.get_intelligence_result("test", {})
        assert result is not None

    # Cache should be shutdown after context
    assert not cache._initialized
