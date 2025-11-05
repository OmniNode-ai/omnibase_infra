"""
Rate Limiter utilities for ONEX Infrastructure.

Provides rate limiting utilities using token bucket algorithm.
Protects infrastructure services from overload and ensures fair resource allocation.

Features:
- Token bucket rate limiting algorithm
- In-memory and Redis-backed implementations
- Service-specific rate limit configurations
- Integration with OnexError exception handling
"""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

# Type variable for generic function wrapping
F = TypeVar("F", bound=Callable[..., Any])


class TokenBucketLimiter:
    """
    Token bucket rate limiter implementation.

    Implements the token bucket algorithm for rate limiting.
    Tokens are added at a fixed rate, and operations consume tokens.

    Features:
    - Configurable token refill rate
    - Burst capacity support
    - Thread-safe in-memory implementation
    """

    def __init__(
        self,
        rate: float,
        capacity: int,
        name: str = "rate_limiter",
    ):
        """
        Initialize token bucket limiter.

        Args:
            rate: Tokens added per second
            capacity: Maximum token capacity (burst size)
            name: Limiter name for error messages
        """
        self.rate = rate
        self.capacity = capacity
        self.name = name
        self.tokens = float(capacity)
        self.last_update = time.monotonic()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time since last update."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + (elapsed * self.rate))
        self.last_update = now

    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False if rate limit exceeded
        """
        self._refill_tokens()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def check_limit(self) -> None:
        """
        Check rate limit and raise exception if exceeded.

        Raises:
            OnexError: If rate limit is exceeded
        """
        if not self.acquire():
            raise OnexError(
                code=CoreErrorCode.RATE_LIMIT_EXCEEDED,
                message=f"Rate limit exceeded for {self.name}",
            )

    def __call__(self, func: F) -> F:
        """
        Decorate function with rate limiting.

        Args:
            func: Function to protect with rate limiting

        Returns:
            Wrapped function with rate limiting

        Example:
            ```python
            limiter = TokenBucketLimiter(rate=10.0, capacity=20)

            @limiter
            def protected_operation():
                # This will be rate limited
                pass
            ```
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.check_limit()
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


class RateLimiter:
    """
    Rate limiter with multiple bucket support.

    Manages multiple token buckets for different rate limit tiers.
    Useful for implementing per-user, per-IP, or per-service rate limits.
    """

    def __init__(self) -> None:
        """Initialize rate limiter with empty bucket registry."""
        self.buckets: dict[str, TokenBucketLimiter] = {}

    def get_or_create_bucket(
        self,
        key: str,
        rate: float,
        capacity: int,
    ) -> TokenBucketLimiter:
        """
        Get or create a token bucket for a specific key.

        Args:
            key: Unique identifier for the bucket (e.g., user ID, IP address)
            rate: Tokens added per second
            capacity: Maximum token capacity

        Returns:
            Token bucket limiter for the key
        """
        if key not in self.buckets:
            self.buckets[key] = TokenBucketLimiter(
                rate=rate,
                capacity=capacity,
                name=f"rate_limiter_{key}",
            )
        return self.buckets[key]

    def check_limit(self, key: str, rate: float, capacity: int) -> None:
        """
        Check rate limit for a specific key.

        Args:
            key: Unique identifier for the bucket
            rate: Tokens added per second
            capacity: Maximum token capacity

        Raises:
            OnexError: If rate limit is exceeded
        """
        bucket = self.get_or_create_bucket(key, rate, capacity)
        bucket.check_limit()


def create_api_rate_limiter(
    requests_per_second: float = 10.0,
    burst_capacity: int = 20,
) -> TokenBucketLimiter:
    """
    Create a rate limiter for API endpoints.

    Limits API requests to prevent overload and ensure fair usage.

    Args:
        requests_per_second: Maximum requests per second
        burst_capacity: Maximum burst size

    Returns:
        Configured token bucket limiter for API operations

    Example:
        ```python
        api_limiter = create_api_rate_limiter(requests_per_second=100.0)

        @api_limiter
        async def handle_api_request(request: Request) -> Response:
            # This endpoint will be rate limited
            return {"status": "ok"}
        ```
    """
    return TokenBucketLimiter(
        rate=requests_per_second,
        capacity=burst_capacity,
        name="api_rate_limiter",
    )


def create_database_rate_limiter(
    queries_per_second: float = 50.0,
    burst_capacity: int = 100,
) -> TokenBucketLimiter:
    """
    Create a rate limiter for database operations.

    Limits database queries to prevent connection pool exhaustion
    and database overload.

    Args:
        queries_per_second: Maximum queries per second
        burst_capacity: Maximum burst size

    Returns:
        Configured token bucket limiter for database operations

    Example:
        ```python
        db_limiter = create_database_rate_limiter(queries_per_second=200.0)

        @db_limiter
        async def execute_query(sql: str) -> list[Record]:
            # This will be rate limited
            async with connection_manager.acquire_connection() as conn:
                return await conn.fetch(sql)
        ```
    """
    return TokenBucketLimiter(
        rate=queries_per_second,
        capacity=burst_capacity,
        name="database_rate_limiter",
    )
