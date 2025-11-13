"""Unit tests for rate limiting middleware."""

import asyncio
import time

import pytest

from omninode_bridge.middleware import TokenBucketRateLimiter


@pytest.mark.asyncio
async def test_token_bucket_allows_requests():
    """Test token bucket allows requests under limit."""
    limiter = TokenBucketRateLimiter(rate=10, burst=20, window=60)

    # Should allow first 20 requests (burst capacity)
    for _ in range(20):
        assert await limiter.check_rate_limit("user1")

    # Should deny 21st request
    assert not await limiter.check_rate_limit("user1")


@pytest.mark.asyncio
async def test_token_bucket_per_user_isolation():
    """Test rate limits are isolated per user."""
    limiter = TokenBucketRateLimiter(rate=10, burst=20, window=60)

    # Exhaust user1's tokens
    for _ in range(20):
        assert await limiter.check_rate_limit("user1")

    # user1 should be rate limited
    assert not await limiter.check_rate_limit("user1")

    # user2 should still have tokens
    assert await limiter.check_rate_limit("user2")


@pytest.mark.asyncio
async def test_token_bucket_refill():
    """Test token bucket refills over time."""
    limiter = TokenBucketRateLimiter(rate=60, burst=10, window=60)

    # Exhaust tokens
    for _ in range(10):
        assert await limiter.check_rate_limit("user1")

    # Should be rate limited
    assert not await limiter.check_rate_limit("user1")

    # Wait for refill (1 second should add 1 token at 60 tokens/60 seconds)
    await asyncio.sleep(1.1)

    # Should have 1 new token
    assert await limiter.check_rate_limit("user1")


@pytest.mark.asyncio
async def test_token_bucket_get_remaining():
    """Test getting remaining tokens."""
    limiter = TokenBucketRateLimiter(rate=10, burst=20, window=60)

    # Initial tokens
    remaining = limiter.get_remaining_tokens("user1")
    assert remaining == 20

    # Use 5 tokens
    for _ in range(5):
        await limiter.check_rate_limit("user1")

    remaining = limiter.get_remaining_tokens("user1")
    assert remaining == 15


@pytest.mark.asyncio
async def test_token_bucket_retry_after():
    """Test getting retry-after time."""
    limiter = TokenBucketRateLimiter(rate=60, burst=10, window=60)

    # Exhaust tokens
    for _ in range(10):
        await limiter.check_rate_limit("user1")

    # Should be rate limited
    assert not await limiter.check_rate_limit("user1")

    # Get retry-after time
    retry_after = limiter.get_retry_after("user1")
    assert retry_after >= 1  # At least 1 second


@pytest.mark.asyncio
async def test_token_bucket_concurrent_requests():
    """Test token bucket under concurrent load."""
    limiter = TokenBucketRateLimiter(rate=100, burst=200, window=60)

    async def make_request(user_id: str) -> bool:
        return await limiter.check_rate_limit(user_id)

    # Make 100 concurrent requests for same user
    results = await asyncio.gather(*[make_request("user1") for _ in range(100)])

    # All should succeed (under burst limit)
    assert all(results)
    assert sum(results) == 100


@pytest.mark.asyncio
async def test_token_bucket_reset_time():
    """Test getting bucket reset time."""
    limiter = TokenBucketRateLimiter(rate=10, burst=20, window=60)

    # Use some tokens
    for _ in range(10):
        await limiter.check_rate_limit("user1")

    # Get reset time
    reset_time = limiter.get_reset_time("user1")
    assert reset_time.timestamp() > time.time()


@pytest.mark.asyncio
async def test_token_bucket_burst_allowance():
    """Test burst allowance allows traffic spikes."""
    limiter = TokenBucketRateLimiter(rate=10, burst=50, window=60)

    # Should handle burst of 50 requests
    for i in range(50):
        assert await limiter.check_rate_limit("user1"), f"Request {i+1} failed"

    # 51st should fail
    assert not await limiter.check_rate_limit("user1")


@pytest.mark.asyncio
async def test_token_bucket_zero_tokens():
    """Test behavior when bucket has zero tokens."""
    limiter = TokenBucketRateLimiter(rate=10, burst=1, window=60)

    # Use the single token
    assert await limiter.check_rate_limit("user1")

    # Should be rate limited with 0 tokens
    assert not await limiter.check_rate_limit("user1")

    # Get remaining should show 0
    remaining = limiter.get_remaining_tokens("user1")
    assert remaining == 0


def test_token_bucket_zero_rate():
    """Test that zero rate raises ValueError."""
    with pytest.raises(ValueError, match="rate must be a positive integer"):
        TokenBucketRateLimiter(rate=0, burst=20, window=60)


def test_token_bucket_negative_rate():
    """Test that negative rate raises ValueError."""
    with pytest.raises(ValueError, match="rate must be a positive integer"):
        TokenBucketRateLimiter(rate=-10, burst=20, window=60)


def test_token_bucket_zero_burst():
    """Test that zero burst raises ValueError."""
    with pytest.raises(ValueError, match="burst must be a positive integer"):
        TokenBucketRateLimiter(rate=10, burst=0, window=60)


def test_token_bucket_negative_burst():
    """Test that negative burst raises ValueError."""
    with pytest.raises(ValueError, match="burst must be a positive integer"):
        TokenBucketRateLimiter(rate=10, burst=-20, window=60)


def test_token_bucket_zero_window():
    """Test that zero window raises ValueError."""
    with pytest.raises(ValueError, match="window must be a positive integer"):
        TokenBucketRateLimiter(rate=10, burst=20, window=0)


def test_token_bucket_negative_window():
    """Test that negative window raises ValueError."""
    with pytest.raises(ValueError, match="window must be a positive integer"):
        TokenBucketRateLimiter(rate=10, burst=20, window=-60)


def test_token_bucket_all_invalid_params():
    """Test that all invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match="rate must be a positive integer"):
        TokenBucketRateLimiter(rate=0, burst=0, window=0)
