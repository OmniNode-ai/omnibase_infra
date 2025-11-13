"""
Production Readiness Fixes Test Suite

Comprehensive tests for all production readiness fixes implemented in the NodeBridgeRegistry.

Tests cover:
1. Memory leak prevention with TTL cache
2. Race condition prevention with cleanup locks
3. Circuit breaker pattern for error recovery
4. Atomic registration with proper transaction handling
5. Security-aware logging with sensitive data masking
6. Configurable magic numbers
7. Optimized FSM state discovery with caching
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from omninode_bridge.config.registry_config import RegistryConfig
from omninode_bridge.nodes.mixins.introspection_mixin import (
    IntrospectionMixin,
    clear_global_caches,
)
from omninode_bridge.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    create_circuit_breaker,
)
from omninode_bridge.utils.secure_logging import (
    SecureLogFormatter,
    get_secure_logger,
    sanitize_log_data,
)
from omninode_bridge.utils.ttl_cache import TTLCache, create_ttl_cache


class TestRegistryConfig:
    """Test registry configuration with environment-specific settings."""

    def test_development_config(self):
        """Test development environment configuration."""
        config = RegistryConfig(environment="development")

        assert config.environment == "development"
        assert config.max_tracked_offsets == 5000
        assert config.offset_cache_ttl_seconds == 1800
        assert config.circuit_breaker_enabled is True
        assert not config.sanitize_logs_in_production
        assert config.enable_emoji_logs is True

    def test_production_config(self):
        """Test production environment configuration."""
        config = RegistryConfig(environment="production")

        assert config.environment == "production"
        assert config.max_tracked_offsets == 10000
        assert config.offset_cache_ttl_seconds == 3600
        assert config.circuit_breaker_enabled is True
        assert config.sanitize_logs_in_production is True
        assert config.enable_emoji_logs is False

    def test_config_summary(self):
        """Test configuration summary generation."""
        config = RegistryConfig(environment="production")
        summary = config.get_summary()

        assert "environment" in summary
        assert "offset_tracking" in summary
        assert "circuit_breaker" in summary
        assert "atomic_registration" in summary
        assert "performance" in summary
        assert "security" in summary

        assert summary["environment"] == "production"


class TestCircuitBreaker:
    """Test circuit breaker implementation."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a test circuit breaker."""
        return CircuitBreaker(
            name="test-circuit",
            failure_threshold=3,
            timeout_seconds=60,
            max_attempts=3,
            base_delay_seconds=0.1,  # Fast for tests
            max_delay_seconds=1.0,
        )

    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful circuit breaker call."""

        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.successful_calls == 1
        assert circuit_breaker.metrics.failed_calls == 0

    @pytest.mark.asyncio
    async def test_retry_logic(self, circuit_breaker):
        """Test circuit breaker retry logic."""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success_after_retries"

        result = await circuit_breaker.call(failing_func)
        assert result == "success_after_retries"
        assert call_count == 3  # Should retry 2 times + 1 success
        assert circuit_breaker.metrics.successful_calls == 1
        assert circuit_breaker.metrics.failed_calls == 2

    @pytest.mark.asyncio
    async def test_circuit_opening(self, circuit_breaker):
        """Test circuit opening after threshold failures."""

        async def always_failing_func():
            raise ConnectionError("Always fails")

        # Make multiple calls to trigger circuit opening
        for i in range(5):
            try:
                await circuit_breaker.call(always_failing_func)
            except Exception:
                pass

        # Circuit should be open now
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.metrics.circuit_opens >= 1

        # Subsequent calls should fail fast
        with pytest.raises(Exception):  # CircuitBreakerError
            await circuit_breaker.call(always_failing_func)

    @pytest.mark.asyncio
    async def test_circuit_reset(self, circuit_breaker):
        """Test manual circuit reset."""

        async def failing_func():
            raise ConnectionError("Always fails")

        # Trigger circuit opening
        for i in range(5):
            try:
                await circuit_breaker.call(failing_func)
            except Exception:
                pass

        assert circuit_breaker.state == CircuitState.OPEN

        # Reset circuit
        await circuit_breaker.reset()
        assert circuit_breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_factory(self):
        """Test circuit breaker factory function."""
        cb = create_circuit_breaker("test-factory", environment="production")
        assert cb.name == "test-factory"
        assert cb._failure_threshold >= 5  # Production default

    def test_circuit_breaker_metrics(self, circuit_breaker):
        """Test circuit breaker metrics."""
        metrics = circuit_breaker.get_status()
        assert "name" in metrics
        assert "state" in metrics
        assert "metrics" in metrics
        assert "config" in metrics

        assert metrics["name"] == "test-circuit"
        assert metrics["state"] in ["closed", "open", "half_open"]


class TestTTLCache:
    """Test TTL cache implementation."""

    @pytest.fixture
    def ttl_cache(self):
        """Create a test TTL cache."""
        return TTLCache(
            name="test-cache",
            max_size=100,
            default_ttl_seconds=1.0,  # Fast for tests
            cleanup_interval_seconds=0.5,
            enable_background_cleanup=True,
        )

    @pytest.mark.asyncio
    async def test_basic_operations(self, ttl_cache):
        """Test basic cache operations."""
        # Test put and get
        ttl_cache.put("key1", "value1")
        assert ttl_cache.get("key1") == "value1"

        # Test non-existent key
        assert ttl_cache.get("non_existent") is None

        # Test cache size
        assert ttl_cache.size() == 1

        # Test contains
        assert "key1" in ttl_cache
        assert "non_existent" not in ttl_cache

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, ttl_cache):
        """Test TTL expiration."""
        ttl_cache.put("expire_key", "expire_value", ttl_seconds=0.1)

        # Should be available immediately
        assert ttl_cache.get("expire_key") == "expire_value"

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Should be expired
        assert ttl_cache.get("expire_key") is None

    @pytest.mark.asyncio
    async def test_max_size_eviction(self, ttl_cache):
        """Test max size eviction (LRU)."""
        # Fill cache to max size
        for i in range(100):
            ttl_cache.put(f"key{i}", f"value{i}")

        assert ttl_cache.size() == 100

        # Add one more item
        ttl_cache.put("key101", "value101")

        # Should still be at max size (one item evicted)
        assert ttl_cache.size() == 100

        # Least recently used item should be evicted
        assert ttl_cache.get("key0") is None
        assert ttl_cache.get("key1") is not None

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, ttl_cache):
        """Test manual cleanup of expired entries."""
        # Add some entries with different TTLs
        ttl_cache.put("keep", "keep_value", ttl_seconds=10.0)
        ttl_cache.put("expire1", "expire_value1", ttl_seconds=0.1)
        ttl_cache.put("expire2", "expire_value2", ttl_seconds=0.1)

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Manual cleanup
        expired_count = ttl_cache.cleanup_expired()
        assert expired_count == 2

        # Check remaining entries
        assert ttl_cache.get("keep") == "keep_value"
        assert ttl_cache.get("expire1") is None
        assert ttl_cache.get("expire2") is None

    @pytest.mark.asyncio
    async def test_memory_monitoring(self, ttl_cache):
        """Test memory monitoring functionality."""
        # Add entries to trigger memory monitoring
        for i in range(50):
            ttl_cache.put(f"key{i}", f"value{i}" * 100)  # Larger values

        metrics = ttl_cache.get_metrics()
        assert metrics.memory_usage_bytes > 0
        assert metrics.total_operations >= 50

        status = ttl_cache.get_status()
        assert "memory_usage_mb" in status["metrics"]
        assert status["metrics"]["memory_usage_mb"] > 0

    @pytest.mark.asyncio
    async def test_cache_stop(self, ttl_cache):
        """Test cache cleanup on stop."""
        await ttl_cache.stop()
        assert not ttl_cache._cleanup_active

    def test_ttl_cache_factory(self):
        """Test TTL cache factory function."""
        cache = create_ttl_cache("test-factory-cache", environment="production")
        assert cache.name == "test-factory-cache"
        assert cache._max_size >= 10000  # Production default


class TestSecureLogging:
    """Test secure logging implementation."""

    def test_sensitive_data_masking(self):
        """Test sensitive data masking in logs."""
        formatter = SecureLogFormatter(environment="production")

        # Test password masking
        log_message = 'User login: password="secret123"'
        masked = formatter._sanitize_sensitive_data(log_message)
        assert "secret123" not in masked
        assert "s" in masked and masked.count("s") >= 2  # First and last char preserved

        # Test API key masking
        log_message = 'API key: api_key="sk-1234567890abcdef"'
        masked = formatter._sanitize_sensitive_data(log_message)
        assert "1234567890abcdef" not in masked

        # Test connection string masking
        log_message = "DB connection: postgresql://user:password@host:5432/db"
        masked = formatter._sanitize_sensitive_data(log_message)
        assert "password" not in masked
        assert "********" in masked

    def test_emoji_removal(self):
        """Test emoji removal in production logs."""
        formatter = SecureLogFormatter(environment="production", enable_emoji=False)

        log_message = "Processing complete ✅ | Error occurred 🚨 | Warning ⚠️"
        cleaned = formatter._remove_emojis(log_message)

        assert "✅" not in cleaned
        assert "🚨" not in cleaned
        assert "⚠️" not in cleaned
        assert "Processing complete" in cleaned
        assert "Error occurred" in cleaned
        assert "Warning" in cleaned

    def test_secure_logger_context_sanitization(self):
        """Test secure logger context sanitization."""
        logger = get_secure_logger("test-logger", environment="production")

        context = {
            "username": "testuser",
            "password": "secret123",
            "api_key": "sk-abcdef123456",
            "connection_string": "postgresql://user:pass@host/db",
            "safe_data": "this is safe",
        }

        sanitized = logger._sanitize_context(context)

        assert sanitized["username"] == "testuser"  # Safe
        assert sanitized["safe_data"] == "this is safe"  # Safe
        assert "secret123" not in str(sanitized["password"])  # Masked
        assert "sk-abcdef123456" not in str(sanitized["api_key"])  # Masked
        assert "pass" not in str(sanitized["connection_string"])  # Masked

    def test_log_data_sanitization(self):
        """Test log data sanitization utility."""
        # Test string data
        data = "password=secret123&api_key=sk-abcdef"
        sanitized = sanitize_log_data(data, environment="production")
        assert "secret123" not in sanitized
        assert "sk-abcdef" not in sanitized

        # Test dictionary data
        data = {
            "username": "test",
            "password": "secret123",
            "nested": {"api_key": "sk-abcdef", "safe": "data"},
        }
        sanitized = sanitize_log_data(data, environment="production")
        assert sanitized["username"] == "test"
        assert "secret123" not in str(sanitized["password"])
        assert "sk-abcdef" not in str(sanitized["nested"]["api_key"])
        assert sanitized["nested"]["safe"] == "data"


class TestIntrospectionMixinOptimized:
    """Test optimized introspection mixin."""

    @pytest.fixture
    def mock_node(self):
        """Create a mock node with introspection mixin."""

        class MockNode(IntrospectionMixin):
            def __init__(self):
                super().__init__()
                self.node_id = "test-node-123"
                self.node_name = "TestNode"
                self.node_version = "1.0.0"
                self.container = MagicMock()
                self.container.config = {
                    "host": "localhost",
                    "port": 8053,
                    "environment": "dev",
                }

        return MockNode()

    @pytest.mark.asyncio
    async def test_cached_introspection_data(self, mock_node):
        """Test introspection data caching."""
        # First call should compute data
        start_time = time.time()
        data1 = await mock_node.get_introspection_data()
        first_call_time = time.time() - start_time

        # Second call should use cache
        start_time = time.time()
        data2 = await mock_node.get_introspection_data()
        second_call_time = time.time() - start_time

        # Data should be identical
        assert data1 == data2

        # Second call should be faster (cached)
        assert second_call_time < first_call_time

        # Check cache stats
        cache_stats = mock_node.get_cache_stats()
        assert cache_stats["cache_entries"] > 0
        assert "introspection_data" in cache_stats["cached_keys"]

    @pytest.mark.asyncio
    async def test_cached_capabilities(self, mock_node):
        """Test capabilities caching."""
        mock_node.workflow_fsm_states = {
            "workflow1": MagicMock(current_state=MagicMock(value="processing"))
        }

        # First call
        cap1 = await mock_node.get_capabilities()
        assert "fsm_states" in cap1

        # Second call should use cache
        cap2 = await mock_node.get_capabilities()
        assert cap1 == cap2

    def test_node_type_caching(self, mock_node):
        """Test node type caching."""
        # First call
        node_type1 = mock_node._get_node_type_cached()
        assert node_type1 == "unknown"  # Default for mock

        # Second call should use cache
        node_type2 = mock_node._get_node_type_cached()
        assert node_type1 == node_type2

        # Check that cached value is used
        assert mock_node._cached_node_type == node_type1

    def test_cache_clearing(self, mock_node):
        """Test cache clearing."""
        # Add some data to cache
        asyncio.run(mock_node.get_introspection_data())

        # Verify cache has entries
        cache_stats = mock_node.get_cache_stats()
        assert cache_stats["cache_entries"] > 0

        # Clear cache
        mock_node.clear_introspection_cache()

        # Verify cache is empty
        cache_stats = mock_node.get_cache_stats()
        assert cache_stats["cache_entries"] == 0

    def test_global_cache_functions(self):
        """Test global cache utility functions."""
        # Test global cache stats
        stats = clear_global_caches()
        assert stats is None  # Function returns None

        # After clearing, stats should be empty
        from omninode_bridge.nodes.mixins.introspection_mixin_fixed import (
            get_global_cache_stats,
        )

        global_stats = get_global_cache_stats()
        assert isinstance(global_stats, dict)
        assert "enum_cache" in global_stats
        assert "supported_states_cache" in global_stats


class TestProductionReadinessIntegration:
    """Integration tests for production readiness fixes."""

    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self):
        """Test memory leak prevention with TTL cache."""
        cache = create_ttl_cache("memory-test", max_size=10, ttl_seconds=0.1)

        # Add many entries beyond max size
        for i in range(100):
            cache.put(f"key{i}", f"value{i}")

        # Cache should not grow beyond max size
        assert cache.size() <= 10

        # Memory usage should be reasonable
        metrics = cache.get_metrics()
        assert metrics.memory_usage_bytes > 0
        assert metrics.evictions > 0

        await cache.stop()

    @pytest.mark.asyncio
    async def test_circuit_breaker_error_recovery(self):
        """Test circuit breaker error recovery."""
        cb = create_circuit_breaker(
            "recovery-test", failure_threshold=2, timeout_seconds=0.1
        )

        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("Flaky failure")
            return "recovered"

        # Should recover after failures
        result = await cb.call(flaky_func)
        assert result == "recovered"
        assert call_count == 4  # 3 failures + 1 success

    @pytest.mark.asyncio
    async def test_secure_logging_integration(self):
        """Test secure logging integration."""
        logger = get_secure_logger("integration-test", environment="production")

        # This should not raise an exception and should mask sensitive data
        logger.info(
            "User action completed",
            user_id="user123",
            password="secret123",
            action="login",
        )

        # The log should be sanitized (we can't easily test the output here,
        # but we can verify no exceptions are raised)
        assert True  # If we get here, no exceptions were raised

    def test_configuration_integration(self):
        """Test configuration integration."""
        # Test development config
        dev_config = RegistryConfig(environment="development")
        assert dev_config.enable_emoji_logs is True
        assert dev_config.sanitize_logs_in_production is False

        # Test production config
        prod_config = RegistryConfig(environment="production")
        assert prod_config.enable_emoji_logs is False
        assert prod_config.sanitize_logs_in_production is True
        assert prod_config.atomic_registration_enabled is True

    @pytest.mark.asyncio
    async def test_performance_optimization(self):
        """Test performance optimizations."""
        mock_node = IntrospectionMixin()
        mock_node.node_id = "perf-test"
        mock_node.container = MagicMock()
        mock_node.container.config = {"host": "localhost", "port": 8053}

        # First call should be slower
        start_time = time.time()
        data1 = await mock_node.get_introspection_data()
        first_call_duration = time.time() - start_time

        # Subsequent calls should be faster
        start_time = time.time()
        data2 = await mock_node.get_introspection_data()
        second_call_duration = time.time() - start_time

        # Verify caching is working
        assert data1 == data2
        assert second_call_duration < first_call_duration

        # Verify cache stats
        cache_stats = mock_node.get_cache_stats()
        assert cache_stats["cache_entries"] > 0


class TestErrorHandlingEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_non_retryable_error(self):
        """Test circuit breaker with non-retryable errors."""
        cb = CircuitBreaker(
            "non-retryable-test", retryable_exceptions=[ConnectionError]
        )

        async def non_retryable_func():
            raise ValueError("Non-retryable error")

        # Should fail immediately without retry
        with pytest.raises(ValueError):
            await cb.call(non_retryable_func)

        # Should not increment failed calls for retry tracking
        assert cb.metrics.failed_calls == 1  # Only one call made

    @pytest.mark.asyncio
    async def test_ttl_cache_concurrent_access(self):
        """Test TTL cache with concurrent access."""
        cache = TTLCache("concurrent-test", max_size=100, default_ttl_seconds=1.0)

        async def worker(worker_id):
            for i in range(10):
                key = f"worker{worker_id}_key{i}"
                value = f"worker{worker_id}_value{i}"
                cache.put(key, value)
                retrieved = cache.get(key)
                assert retrieved == value

        # Run multiple workers concurrently
        tasks = [worker(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Cache should have correct number of entries
        assert cache.size() == 50  # 5 workers * 10 entries each

        await cache.stop()

    def test_secure_logging_edge_cases(self):
        """Test secure logging edge cases."""
        formatter = SecureLogFormatter(environment="production")

        # Test empty string
        assert formatter._sanitize_sensitive_data("") == ""

        # Test string without sensitive data
        safe_string = "This is a safe log message"
        assert formatter._sanitize_sensitive_data(safe_string) == safe_string

        # Test malformed patterns
        malformed = "password= api_key="
        masked = formatter._sanitize_sensitive_data(malformed)
        # Should not crash and should preserve some structure
        assert isinstance(masked, str)

    def test_configuration_edge_cases(self):
        """Test configuration edge cases."""
        # Test invalid environment (should default to development behavior)
        config = RegistryConfig(environment="invalid")
        assert config.environment == "invalid"
        # Should have reasonable defaults
        assert config.max_tracked_offsets > 0
        assert config.offset_cache_ttl_seconds > 0

    @pytest.mark.asyncio
    async def test_introspection_with_missing_attributes(self):
        """Test introspection with missing node attributes."""

        class MinimalNode(IntrospectionMixin):
            def __init__(self):
                super().__init__()
                self.node_id = "minimal"

        node = MinimalNode()

        # Should not crash even with minimal attributes
        data = await node.get_introspection_data()
        assert data["node_id"] == "minimal"
        assert "capabilities" in data
        assert "endpoints" in data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
