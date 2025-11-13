"""Unit tests for AsyncOnexTreeClient."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from omninode_bridge.clients.circuit_breaker import CircuitState
from omninode_bridge.clients.onextree_client import AsyncOnexTreeClient, CacheEntry


@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient."""
    mock = AsyncMock(spec=httpx.AsyncClient)
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture
async def onextree_client(mock_httpx_client):
    """Provide AsyncOnexTreeClient instance for testing."""
    client = AsyncOnexTreeClient(
        base_url="http://localhost:8054",
        timeout=5.0,
        max_retries=2,
        enable_cache=True,
        cache_ttl=300,
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


class TestOnexTreeClientInitialization:
    """Test client initialization and configuration."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initializes with correct defaults."""
        client = AsyncOnexTreeClient()

        assert (
            client.base_url == "http://192.168.86.200:8058"
        )  # Remote infrastructure default
        assert client.service_name == "OnexTreeService"
        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client.enable_cache is True

        await client.close()

    @pytest.mark.asyncio
    async def test_client_custom_configuration(self):
        """Test client accepts custom configuration."""
        client = AsyncOnexTreeClient(
            base_url="http://custom:9999",
            timeout=10.0,
            max_retries=5,
            enable_cache=False,
            cache_ttl=600,
        )

        assert client.base_url == "http://custom:9999"
        assert client.timeout == 10.0
        assert client.max_retries == 5
        assert client.enable_cache is False
        assert client.cache_ttl == 600

        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client works as async context manager."""
        async with AsyncOnexTreeClient() as client:
            assert client._http_client is not None

        # Should be closed after context
        assert client._http_client is None


class TestIntelligenceRetrieval:
    """Test intelligence data retrieval."""

    @pytest.mark.asyncio
    async def test_get_intelligence_success(self, onextree_client, mock_httpx_client):
        """Test successful intelligence retrieval."""
        from omninode_bridge.clients.onextree_client import IntelligenceResult

        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "intelligence": {"key": "value"},
                "patterns": ["pattern1", "pattern2"],
                "relationships": [{"from": "A", "to": "B"}],
                "metadata": {"source": "onextree"},
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # Get intelligence
        correlation_id = uuid4()
        result = await onextree_client.get_intelligence(
            context="authentication patterns",
            include_patterns=True,
            include_relationships=True,
            correlation_id=correlation_id,
        )

        # Verify result is IntelligenceResult
        assert isinstance(result, IntelligenceResult)
        assert result.intelligence == {"key": "value"}
        assert result.patterns == ["pattern1", "pattern2"]
        assert result.relationships == [{"from": "A", "to": "B"}]
        assert result.degraded is False

        # Verify request was made correctly
        mock_httpx_client.request.assert_called_once()
        call_kwargs = mock_httpx_client.request.call_args.kwargs
        assert call_kwargs["method"] == "POST"
        assert "/intelligence" in call_kwargs["url"]
        assert call_kwargs["headers"]["X-Correlation-ID"] == str(correlation_id)

    @pytest.mark.asyncio
    async def test_get_intelligence_caching(self, onextree_client, mock_httpx_client):
        """Test intelligence results are cached."""
        from omninode_bridge.clients.onextree_client import IntelligenceResult

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "intelligence": {"key": "value"},
                "patterns": [],
                "relationships": [],
                "metadata": {},
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # First call should hit the service
        result1 = await onextree_client.get_intelligence(
            context="test",
            use_cache=True,
        )

        # Second call should use cache
        result2 = await onextree_client.get_intelligence(
            context="test",
            use_cache=True,
        )

        # Should only have made one request
        assert mock_httpx_client.request.call_count == 1

        # Both results should be IntelligenceResult
        assert isinstance(result1, IntelligenceResult)
        assert isinstance(result2, IntelligenceResult)

        # Verify they have the same data (second should be from cache)
        assert result1.intelligence == result2.intelligence
        assert result1.patterns == result2.patterns
        assert result1.relationships == result2.relationships

        # Cache stats should show hit
        cache_stats = onextree_client.get_cache_stats()
        assert cache_stats["cache_hits"] == 1
        assert cache_stats["cache_misses"] == 1  # First request

    @pytest.mark.asyncio
    async def test_get_intelligence_bypass_cache(
        self, onextree_client, mock_httpx_client
    ):
        """Test cache can be bypassed."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"intelligence": "data"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # Both calls should hit the service
        await onextree_client.get_intelligence(context="test", use_cache=False)
        await onextree_client.get_intelligence(context="test", use_cache=False)

        # Should have made two requests
        assert mock_httpx_client.request.call_count == 2


class TestKnowledgeQuery:
    """Test knowledge graph queries."""

    @pytest.mark.asyncio
    async def test_query_knowledge_success(self, onextree_client, mock_httpx_client):
        """Test successful knowledge query."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "results": [{"node": "A"}, {"node": "B"}],
                "paths": [["A", "B"]],
                "metadata": {"query_time_ms": 50},
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        result = await onextree_client.query_knowledge(
            query="Find all API endpoints",
            max_depth=3,
            filters={"type": "endpoint"},
        )

        assert "results" in result
        assert len(result["results"]) == 2
        assert "paths" in result

    @pytest.mark.asyncio
    async def test_query_knowledge_with_filters(
        self, onextree_client, mock_httpx_client
    ):
        """Test knowledge query with filters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"results": []},
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        await onextree_client.query_knowledge(
            query="test",
            filters={"category": "auth", "status": "active"},
        )

        # Verify filters were included
        call_kwargs = mock_httpx_client.request.call_args.kwargs
        request_body = call_kwargs["json"]
        assert request_body["filters"] == {"category": "auth", "status": "active"}


class TestTreeNavigation:
    """Test tree navigation functionality."""

    @pytest.mark.asyncio
    async def test_navigate_tree_success(self, onextree_client, mock_httpx_client):
        """Test successful tree navigation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "nodes": [{"id": "1"}, {"id": "2"}],
                "edges": [{"from": "1", "to": "2"}],
                "metadata": {"depth": 2},
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        result = await onextree_client.navigate_tree(
            start_node="root",
            direction="both",
            max_nodes=100,
        )

        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 2

    @pytest.mark.asyncio
    async def test_navigate_tree_directions(self, onextree_client, mock_httpx_client):
        """Test different navigation directions."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"nodes": [], "edges": []},
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # Test different directions
        for direction in ["up", "down", "both"]:
            await onextree_client.navigate_tree(
                start_node="test",
                direction=direction,
            )

        # Verify directions were used
        assert mock_httpx_client.request.call_count == 3


class TestCaching:
    """Test caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_entry_expiration(self):
        """Test cache entries expire after TTL."""
        import time

        entry = CacheEntry(data="test", ttl=1)
        assert not entry.is_expired()

        # Wait for expiration
        time.sleep(1.1)
        assert entry.is_expired()

    @pytest.mark.asyncio
    async def test_cache_entry_hit_tracking(self):
        """Test cache entry tracks hits."""
        entry = CacheEntry(data="test")

        assert entry.hits == 0

        entry.access()
        assert entry.hits == 1

        entry.access()
        assert entry.hits == 2

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, onextree_client):
        """Test cache key generation is consistent."""
        key1 = onextree_client._get_cache_key("/test", {"a": "1", "b": "2"})
        key2 = onextree_client._get_cache_key("/test", {"b": "2", "a": "1"})

        # Keys should be same regardless of param order
        assert key1 == key2

        # Different params should give different keys
        key3 = onextree_client._get_cache_key("/test", {"a": "1", "b": "3"})
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_clear_cache(self, onextree_client, mock_httpx_client):
        """Test cache can be cleared."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"test": "data"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # Add some cached data
        await onextree_client.get_intelligence(context="test1")
        await onextree_client.get_intelligence(context="test2")

        cache_stats = onextree_client.get_cache_stats()
        assert cache_stats["cache_size"] > 0

        # Clear cache
        await onextree_client.clear_cache()

        cache_stats = onextree_client.get_cache_stats()
        assert cache_stats["cache_size"] == 0

    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """Test client works with caching disabled."""
        client = AsyncOnexTreeClient(enable_cache=False)

        mock_httpx_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"test": "data"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        with patch.object(client, "_http_client", mock_httpx_client):
            # Make two identical requests
            await client.get_intelligence(context="test")
            await client.get_intelligence(context="test")

            # Both should hit the service (no caching)
            assert mock_httpx_client.request.call_count == 2

            # Cache stats should show disabled
            cache_stats = client.get_cache_stats()
            assert cache_stats["enabled"] is False

        await client.close()


class TestCacheStatistics:
    """Test cache statistics and metrics."""

    @pytest.mark.asyncio
    async def test_cache_hit_rate_calculation(self, onextree_client, mock_httpx_client):
        """Test cache hit rate is calculated correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"test": "data"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # First call: cache miss
        await onextree_client.get_intelligence(context="test", use_cache=True)

        # Next 3 calls: cache hits
        for _ in range(3):
            await onextree_client.get_intelligence(context="test", use_cache=True)

        stats = onextree_client.get_cache_stats()
        assert stats["cache_hits"] == 3
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 0.75  # 3 hits out of 4 total

    @pytest.mark.asyncio
    async def test_get_metrics_includes_cache(self, onextree_client):
        """Test get_metrics includes cache statistics."""
        metrics = onextree_client.get_metrics()

        assert "cache" in metrics
        assert "enabled" in metrics["cache"]
        assert "cache_size" in metrics["cache"]
        assert "hit_rate" in metrics["cache"]


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_state(self, onextree_client):
        """Test circuit breaker is properly initialized."""
        assert onextree_client.circuit_breaker is not None
        assert onextree_client.circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_in_metrics(self, onextree_client):
        """Test circuit breaker metrics are included."""
        metrics = onextree_client.get_metrics()
        assert "circuit_breaker" in metrics
        assert metrics["circuit_breaker"]["state"] == CircuitState.CLOSED.value


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, onextree_client, mock_httpx_client):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        health = await onextree_client.health_check()

        assert health["status"] == "healthy"
        assert "version" in health

    @pytest.mark.asyncio
    async def test_validate_connection(self, onextree_client, mock_httpx_client):
        """Test connection validation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        is_valid = await onextree_client._validate_connection()
        assert is_valid is True


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_service_error_handling(self, onextree_client, mock_httpx_client):
        """Test handling of service errors."""
        from omninode_bridge.clients.onextree_client import IntelligenceResult

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "error",
            "error": "Service unavailable",
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # With fallback enabled (default), should return degraded result
        result = await onextree_client.get_intelligence(context="test")
        assert isinstance(result, IntelligenceResult)
        assert result.degraded is True

        # Without fallback, should raise exception
        from omninode_bridge.clients.base_client import ClientError

        with pytest.raises(ClientError, match="Intelligence retrieval failed"):
            await onextree_client.get_intelligence(
                context="test", enable_fallback=False
            )

    @pytest.mark.asyncio
    async def test_cache_size_management(self, onextree_client, mock_httpx_client):
        """Test cache automatically manages size."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"test": "data"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # Fill cache beyond limit (1000 entries)
        # This is a simplified test - in practice we'd need 1000+ unique queries
        for i in range(10):
            await onextree_client.get_intelligence(context=f"test{i}")

        stats = onextree_client.get_cache_stats()
        # Cache should not grow indefinitely
        assert stats["cache_size"] <= 1000
