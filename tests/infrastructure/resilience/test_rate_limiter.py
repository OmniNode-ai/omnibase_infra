"""
Comprehensive tests for Rate Limiter.

Tests token bucket algorithm, rate limiting enforcement,
burst capacity, and OnexError integration.
"""

import time

import pytest
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

from omnibase_infra.infrastructure.resilience.rate_limiter import (
    TokenBucketLimiter,
    create_api_rate_limiter,
    create_database_rate_limiter,
)


class TestTokenBucketLimiterInit:
    """Test token bucket limiter initialization."""

    def test_init_default(self):
        """Test initialization with required parameters."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=100)

        assert limiter.rate == 10.0
        assert limiter.capacity == 100
        assert limiter.tokens == 100.0
        assert limiter.name == "rate_limiter"

    def test_init_with_name(self):
        """Test initialization with custom name."""
        limiter = TokenBucketLimiter(
            rate=5.0,
            capacity=50,
            name="custom_limiter",
        )

        assert limiter.name == "custom_limiter"
        assert limiter.rate == 5.0
        assert limiter.capacity == 50


class TestTokenBucketLimiterAcquire:
    """Test token acquisition."""

    def test_acquire_single_token(self):
        """Test acquiring a single token."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=100)

        result = limiter.acquire()

        assert result is True
        assert limiter.tokens == 99.0

    def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=100)

        result = limiter.acquire(tokens=10)

        assert result is True
        assert limiter.tokens == 90.0

    def test_acquire_insufficient_tokens(self):
        """Test acquire fails when insufficient tokens."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=10)
        limiter.tokens = 5.0

        result = limiter.acquire(tokens=10)

        assert result is False
        assert limiter.tokens == 5.0  # Tokens unchanged

    def test_acquire_exact_capacity(self):
        """Test acquiring exact capacity."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=100)

        result = limiter.acquire(tokens=100)

        assert result is True
        assert limiter.tokens == 0.0

    def test_acquire_refills_tokens(self):
        """Test that tokens refill over time."""
        limiter = TokenBucketLimiter(rate=100.0, capacity=100)
        limiter.tokens = 0.0

        # Wait for tokens to refill
        time.sleep(0.5)

        result = limiter.acquire(tokens=10)

        # Should have refilled approximately 50 tokens (100/s * 0.5s)
        assert result is True

    def test_acquire_respects_capacity(self):
        """Test that token count doesn't exceed capacity."""
        limiter = TokenBucketLimiter(rate=1000.0, capacity=50)
        limiter.tokens = 10.0

        # Wait long enough to exceed capacity if not limited
        time.sleep(0.1)

        # Try to acquire - should work but tokens shouldn't exceed capacity
        limiter.acquire(tokens=1)
        assert limiter.tokens <= 50.0


class TestTokenBucketLimiterCheckLimit:
    """Test rate limit checking."""

    def test_check_limit_passes(self):
        """Test check_limit passes when tokens available."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=100)

        # Should not raise exception
        limiter.check_limit()

        assert limiter.tokens == 99.0

    def test_check_limit_raises_onex_error(self):
        """Test check_limit raises OnexError when rate exceeded."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=10, name="test_limiter")
        limiter.tokens = 0.0

        with pytest.raises(OnexError) as exc_info:
            limiter.check_limit()

        assert exc_info.value.code == CoreErrorCode.RATE_LIMIT_EXCEEDED
        assert "test_limiter" in exc_info.value.message

    def test_check_limit_multiple_calls(self):
        """Test multiple check_limit calls."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=5)

        # Should succeed 5 times
        for _ in range(5):
            limiter.check_limit()

        # 6th call should fail
        with pytest.raises(OnexError):
            limiter.check_limit()


class TestTokenBucketLimiterDecorator:
    """Test rate limiter as decorator."""

    def test_decorator_successful_calls(self):
        """Test decorator allows calls within rate limit."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=10)

        @limiter
        def limited_function():
            return "success"

        # Should succeed within capacity
        for _ in range(10):
            result = limited_function()
            assert result == "success"

    def test_decorator_blocks_excess_calls(self):
        """Test decorator blocks calls exceeding rate limit."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=5)

        @limiter
        def limited_function():
            return "success"

        # First 5 should succeed
        for _ in range(5):
            limited_function()

        # 6th should raise OnexError
        with pytest.raises(OnexError):
            limited_function()

    def test_decorator_with_arguments(self):
        """Test decorator works with function arguments."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=10)

        @limiter
        def limited_function(x, y):
            return x + y

        result = limited_function(2, 3)
        assert result == 5

    def test_decorator_with_kwargs(self):
        """Test decorator works with keyword arguments."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=10)

        @limiter
        def limited_function(*, value):
            return value * 2

        result = limited_function(value=5)
        assert result == 10

    def test_decorator_preserves_function_metadata(self):
        """Test decorator preserves function metadata."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=10)

        @limiter
        def documented_function():
            """This function has documentation."""
            return "result"

        assert documented_function.__name__ == "documented_function"
        assert "documentation" in documented_function.__doc__


class TestDatabaseRateLimiter:
    """Test database-specific rate limiter."""

    def test_create_database_rate_limiter_defaults(self):
        """Test database rate limiter with default parameters."""
        limiter = create_database_rate_limiter()

        assert isinstance(limiter, TokenBucketLimiter)
        assert limiter.name == "database"

    def test_create_database_rate_limiter_custom(self):
        """Test database rate limiter with custom parameters."""
        limiter = create_database_rate_limiter(
            rate=50.0,
            capacity=200,
            name="postgres_db",
        )

        assert limiter.rate == 50.0
        assert limiter.capacity == 200
        assert limiter.name == "postgres_db"

    def test_database_rate_limiter_usage(self):
        """Test database rate limiter in realistic scenario."""
        limiter = create_database_rate_limiter(rate=10.0, capacity=5)

        @limiter
        def query_database():
            return "query_result"

        # Should succeed within capacity
        for _ in range(5):
            query_database()

        # Should fail when exceeded
        with pytest.raises(OnexError):
            query_database()


class TestApiRateLimiter:
    """Test API-specific rate limiter."""

    def test_create_api_rate_limiter_defaults(self):
        """Test API rate limiter with default parameters."""
        limiter = create_api_rate_limiter()

        assert isinstance(limiter, TokenBucketLimiter)
        assert limiter.name == "api"

    def test_create_api_rate_limiter_custom(self):
        """Test API rate limiter with custom parameters."""
        limiter = create_api_rate_limiter(
            rate=100.0,
            capacity=500,
            name="external_api",
        )

        assert limiter.rate == 100.0
        assert limiter.capacity == 500
        assert limiter.name == "external_api"

    def test_api_rate_limiter_usage(self):
        """Test API rate limiter in realistic scenario."""
        limiter = create_api_rate_limiter(rate=20.0, capacity=10)

        @limiter
        def call_external_api():
            return {"status": "ok"}

        # Should succeed within capacity
        for _ in range(10):
            result = call_external_api()
            assert result["status"] == "ok"

        # Should fail when exceeded
        with pytest.raises(OnexError):
            call_external_api()


class TestRateLimiterIntegration:
    """Integration tests for rate limiting."""

    def test_multiple_limiters_independent(self):
        """Test that different limiters operate independently."""
        db_limiter = create_database_rate_limiter(rate=10.0, capacity=5)
        api_limiter = create_api_rate_limiter(rate=10.0, capacity=5)

        @db_limiter
        def db_operation():
            return "db"

        @api_limiter
        def api_operation():
            return "api"

        # Exhaust database limiter
        for _ in range(5):
            db_operation()

        # API limiter should still work
        result = api_operation()
        assert result == "api"

        # Database limiter should be exhausted
        with pytest.raises(OnexError):
            db_operation()

    def test_rate_limiter_recovery_over_time(self):
        """Test that rate limiter recovers tokens over time."""
        limiter = TokenBucketLimiter(rate=100.0, capacity=10)

        # Exhaust tokens
        for _ in range(10):
            limiter.acquire()

        assert limiter.tokens < 1.0

        # Wait for refill
        time.sleep(0.2)

        # Should be able to acquire again
        result = limiter.acquire()
        assert result is True

    def test_burst_capacity_handling(self):
        """Test handling of burst traffic within capacity."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=100)

        @limiter
        def burst_operation():
            return "success"

        # Burst of 50 requests should succeed (within capacity)
        for _ in range(50):
            result = burst_operation()
            assert result == "success"

        # Still have tokens left
        assert limiter.tokens > 0

    def test_rate_limiter_with_concurrent_access(self):
        """Test rate limiter behavior with concurrent access pattern."""
        limiter = TokenBucketLimiter(rate=50.0, capacity=20)

        call_count = [0]

        @limiter
        def concurrent_operation():
            call_count[0] += 1
            return call_count[0]

        # Simulate concurrent requests
        results = []
        for _ in range(20):
            try:
                results.append(concurrent_operation())
            except OnexError:
                break

        # Should process requests up to capacity
        assert len(results) <= 20

    def test_rate_limiter_error_details(self):
        """Test that rate limit errors include helpful details."""
        limiter = TokenBucketLimiter(
            rate=10.0,
            capacity=1,
            name="test_service",
        )

        # Exhaust capacity
        limiter.acquire()

        with pytest.raises(OnexError) as exc_info:
            limiter.check_limit()

        error = exc_info.value
        assert error.code == CoreErrorCode.RATE_LIMIT_EXCEEDED
        assert "test_service" in error.message
        assert "Rate limit exceeded" in error.message

    def test_combined_rate_limiting_and_retry(self):
        """Test rate limiter combined with retry logic."""
        from omnibase_infra.infrastructure.resilience.retry_policy import (
            create_network_retry_policy,
        )

        limiter = TokenBucketLimiter(rate=100.0, capacity=5)
        retry_decorator = create_network_retry_policy(max_attempts=2)

        call_count = [0]

        @retry_decorator
        @limiter
        def rate_limited_with_retry():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        # Should retry on connection error but also consume tokens
        result = rate_limited_with_retry()
        assert result == "success"
        # Tokens consumed on each attempt
        assert limiter.tokens < 5.0
