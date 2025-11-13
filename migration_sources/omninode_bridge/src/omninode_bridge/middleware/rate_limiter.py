"""Rate limiting middleware using token bucket algorithm.

This module provides rate limiting for API endpoints to prevent abuse and ensure
fair resource allocation across users.

Performance Targets:
- <1ms overhead per request
- Support for per-user and global rate limits
- Burst allowance for traffic spikes
- Automatic token refill

Key Features:
- Token bucket algorithm for smooth rate limiting
- Per-user and global limits
- Configurable burst allowance
- Rate limit headers in responses
- Optional request queueing

Environment Configuration:
- RATE_LIMIT_PER_USER: Requests per minute per user (default: 10)
- RATE_LIMIT_GLOBAL: Requests per minute globally (default: 100)
- RATE_LIMIT_BURST: Burst allowance (default: 20)
- RATE_LIMIT_WINDOW_SECONDS: Time window in seconds (default: 60)
- RATE_LIMIT_ENABLED: Enable rate limiting (default: true)
"""

import asyncio
import logging
import os
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Token bucket rate limiter for smooth rate limiting.

    The token bucket algorithm allows for burst traffic while maintaining
    an average rate limit. Tokens are added to the bucket at a constant rate,
    and each request consumes one token.

    Example:
        ```python
        limiter = TokenBucketRateLimiter(
            rate=10,  # 10 requests per minute
            burst=20,  # Allow burst of 20 requests
            window=60  # 60 second window
        )

        # Check if request is allowed
        if await limiter.check_rate_limit("user123"):
            # Process request
            pass
        else:
            # Reject with 429 Too Many Requests
            retry_after = limiter.get_retry_after("user123")
            raise HTTPException(429, headers={"Retry-After": str(retry_after)})
        ```
    """

    def __init__(
        self,
        rate: int = 10,  # tokens per window
        burst: int = 20,  # maximum tokens
        window: int = 60,  # seconds
    ):
        """Initialize token bucket rate limiter.

        Args:
            rate: Number of tokens added per window (requests per minute)
            burst: Maximum bucket capacity (burst allowance)
            window: Time window in seconds

        Raises:
            ValueError: If rate, burst, or window are not positive values
        """
        # Validate parameters to prevent ZeroDivisionError
        if rate <= 0:
            raise ValueError(
                f"rate must be a positive integer, got {rate}. "
                "Rate determines the number of tokens added per time window."
            )
        if burst <= 0:
            raise ValueError(
                f"burst must be a positive integer, got {burst}. "
                "Burst determines the maximum bucket capacity."
            )
        if window <= 0:
            raise ValueError(
                f"window must be a positive integer, got {window}. "
                "Window determines the time period in seconds."
            )

        self.rate = rate
        self.burst = burst
        self.window = window

        # Token buckets per user
        self.buckets: dict[str, dict[str, float | datetime]] = defaultdict(
            lambda: {"tokens": float(burst), "last_update": datetime.now(UTC)}
        )

        # Lock for thread-safe bucket updates
        self._lock = asyncio.Lock()

    def _refill_bucket(self, bucket: dict[str, float | datetime]) -> None:
        """Refill tokens based on elapsed time.

        Args:
            bucket: Token bucket dictionary with 'tokens' and 'last_update'
        """
        now = datetime.now(UTC)
        elapsed = (now - bucket["last_update"]).total_seconds()

        # Calculate tokens to add based on elapsed time
        tokens_to_add = (elapsed / self.window) * self.rate

        # Update bucket
        bucket["tokens"] = min(self.burst, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now

    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if request is allowed under rate limit.

        Args:
            user_id: User identifier (IP, API key, user ID, etc.)

        Returns:
            True if request allowed, False if rate limited
        """
        async with self._lock:
            bucket = self.buckets[user_id]
            self._refill_bucket(bucket)

            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True

            return False

    def get_retry_after(self, user_id: str) -> int:
        """Get seconds until next token is available.

        Args:
            user_id: User identifier

        Returns:
            Seconds until request will be allowed
        """
        bucket = self.buckets[user_id]

        # Calculate time for next token
        tokens_needed = 1.0 - bucket["tokens"]
        seconds_per_token = self.window / self.rate
        retry_after_seconds = tokens_needed * seconds_per_token

        return max(1, int(retry_after_seconds))

    def get_remaining_tokens(self, user_id: str) -> int:
        """Get remaining tokens for user.

        Args:
            user_id: User identifier

        Returns:
            Number of remaining tokens
        """
        bucket = self.buckets[user_id]
        self._refill_bucket(bucket)
        return int(bucket["tokens"])

    def get_reset_time(self, user_id: str) -> datetime:
        """Get time when bucket will be fully refilled.

        Args:
            user_id: User identifier

        Returns:
            Datetime when bucket is full
        """
        bucket = self.buckets[user_id]
        tokens_to_refill = self.burst - bucket["tokens"]
        seconds_to_refill = (tokens_to_refill / self.rate) * self.window

        return bucket["last_update"] + timedelta(seconds=seconds_to_refill)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting with token bucket algorithm.

    This middleware applies rate limiting to all API endpoints, providing:
    - Per-user rate limiting based on X-User-ID header or IP address
    - Global rate limiting across all users
    - Rate limit headers in responses (X-RateLimit-Limit, X-RateLimit-Remaining, etc.)
    - 429 Too Many Requests response with Retry-After header

    Example:
        ```python
        app = FastAPI()

        # Add rate limiting middleware
        app.add_middleware(
            RateLimitMiddleware,
            per_user_rate=10,
            global_rate=100,
            burst=20,
            enabled=True
        )
        ```
    """

    def __init__(
        self,
        app: FastAPI,
        per_user_rate: int | None = None,
        global_rate: int | None = None,
        burst: int | None = None,
        window: int | None = None,
        enabled: bool | None = None,
    ):
        """Initialize rate limit middleware.

        Args:
            app: FastAPI application instance
            per_user_rate: Requests per minute per user (default: 10)
            global_rate: Requests per minute globally (default: 100)
            burst: Burst allowance (default: 20)
            window: Time window in seconds (default: 60)
            enabled: Enable rate limiting (default: true)
        """
        super().__init__(app)

        # Configuration
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        )
        self.per_user_rate = per_user_rate or int(
            os.getenv("RATE_LIMIT_PER_USER", "10")
        )
        self.global_rate = global_rate or int(os.getenv("RATE_LIMIT_GLOBAL", "100"))
        self.burst = burst or int(os.getenv("RATE_LIMIT_BURST", "20"))
        self.window = window or int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

        # Rate limiters
        self.per_user_limiter = TokenBucketRateLimiter(
            rate=self.per_user_rate, burst=self.burst, window=self.window
        )
        self.global_limiter = TokenBucketRateLimiter(
            rate=self.global_rate, burst=self.burst * 5, window=self.window
        )

        logger.info(
            f"Rate limiting {'enabled' if self.enabled else 'disabled'}: "
            f"{self.per_user_rate} req/min per user, {self.global_rate} req/min global"
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting.

        Args:
            request: Incoming request
            call_next: Next middleware or endpoint handler

        Returns:
            Response with rate limit headers

        Raises:
            HTTPException: 429 Too Many Requests if rate limited
        """
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip rate limiting for health check endpoints
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Extract user identifier
        user_id = self._get_user_id(request)

        # Check global rate limit first
        if not await self.global_limiter.check_rate_limit("global"):
            retry_after = self.global_limiter.get_retry_after("global")
            logger.warning(
                f"Global rate limit exceeded. Retry after {retry_after}s",
                extra={"user_id": user_id, "path": request.url.path},
            )
            raise HTTPException(
                status_code=429,
                detail="Global rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.global_rate),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(
                        int(self.global_limiter.get_reset_time("global").timestamp())
                    ),
                },
            )

        # Check per-user rate limit
        if not await self.per_user_limiter.check_rate_limit(user_id):
            retry_after = self.per_user_limiter.get_retry_after(user_id)
            logger.warning(
                f"User rate limit exceeded for {user_id}. Retry after {retry_after}s",
                extra={"user_id": user_id, "path": request.url.path},
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.per_user_rate),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(
                        int(self.per_user_limiter.get_reset_time(user_id).timestamp())
                    ),
                },
            )

        # Request allowed, process it
        try:
            response = await call_next(request)

            # Add rate limit headers to response
            remaining_tokens = self.per_user_limiter.get_remaining_tokens(user_id)
            reset_time = int(self.per_user_limiter.get_reset_time(user_id).timestamp())

            response.headers["X-RateLimit-Limit"] = str(self.per_user_rate)
            response.headers["X-RateLimit-Remaining"] = str(remaining_tokens)
            response.headers["X-RateLimit-Reset"] = str(reset_time)

            return response

        except Exception as e:
            logger.error(f"Error processing rate-limited request: {e}")
            raise

    def _get_user_id(self, request: Request) -> str:
        """Extract user identifier from request.

        Priority:
        1. X-User-ID header (for authenticated users)
        2. X-API-Key header (for API keys)
        3. Client IP address (fallback)

        Args:
            request: Incoming request

        Returns:
            User identifier string
        """
        # Check for user ID header
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return f"user:{user_id}"

        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Use hash for rate limiting key to avoid API key disclosure
            import hashlib

            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return f"api:{key_hash}"

        # Fallback to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"


def add_rate_limit_middleware(
    app: FastAPI,
    per_user_rate: int | None = None,
    global_rate: int | None = None,
    burst: int | None = None,
    window: int | None = None,
    enabled: bool | None = None,
) -> None:
    """Add rate limiting middleware to FastAPI app.

    Args:
        app: FastAPI application instance
        per_user_rate: Requests per minute per user (default: 10)
        global_rate: Requests per minute globally (default: 100)
        burst: Burst allowance (default: 20)
        window: Time window in seconds (default: 60)
        enabled: Enable rate limiting (default: true)
    """
    app.add_middleware(
        RateLimitMiddleware,
        per_user_rate=per_user_rate,
        global_rate=global_rate,
        burst=burst,
        window=window,
        enabled=enabled,
    )

    logger.info("Rate limiting middleware added to FastAPI app")
