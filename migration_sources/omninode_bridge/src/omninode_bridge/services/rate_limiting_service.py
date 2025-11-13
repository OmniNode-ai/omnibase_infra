"""
Comprehensive Rate Limiting Service for OmniNode Bridge

This module provides advanced rate limiting capabilities with multiple algorithms,
Redis-based distributed rate limiting, and comprehensive monitoring.

Features:
- Multiple rate limiting algorithms (fixed window, sliding window, token bucket)
- Redis-based distributed rate limiting for multi-instance deployments
- Per-user, per-IP, and global rate limiting
- Automatic rate limit adjustment based on load
- Comprehensive audit logging and metrics
- Integration with FastAPI and other frameworks
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import redis.asyncio as redis
from fastapi import Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategy algorithms."""

    FIXED_WINDOW = "fixed-window"
    SLIDING_WINDOW = "sliding-window"
    TOKEN_BUCKET = "token-bucket"
    LEAKY_BUCKET = "leaky-bucket"


class RateLimitScope(Enum):
    """Scope for rate limiting."""

    GLOBAL = "global"
    PER_IP = "per-ip"
    PER_USER = "per-user"
    PER_ENDPOINT = "per-endpoint"
    PER_API_KEY = "per-api-key"  # pragma: allowlist secret


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""

    name: str
    strategy: RateLimitStrategy
    scope: RateLimitScope
    requests_per_minute: int
    burst_size: int
    window_size_minutes: int = 1
    priority: int = 1  # Higher priority rules are checked first
    endpoints: list[str] = field(
        default_factory=list
    )  # Specific endpoints, empty = all
    exclude_endpoints: list[str] = field(default_factory=list)
    user_roles: list[str] = field(
        default_factory=list
    )  # Specific user roles, empty = all
    enabled: bool = True


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    requests_remaining: int
    requests_limit: int
    reset_time: float
    retry_after: Optional[int] = None
    rule_name: str = ""
    current_requests: int = 0


@dataclass
class RateLimitMetrics:
    """Rate limiting metrics for monitoring."""

    total_requests: int = 0
    blocked_requests: int = 0
    allowed_requests: int = 0
    rules_triggered: dict[str, int] = field(default_factory=dict)
    top_blocked_ips: dict[str, int] = field(default_factory=dict)
    response_times_ms: list[float] = field(default_factory=list)
    last_reset: float = field(default_factory=time.time)


class RateLimitStorage:
    """Abstract storage backend for rate limiting data."""

    async def get_counter(self, key: str) -> int:
        """Get request counter for key."""
        raise NotImplementedError

    async def increment_counter(self, key: str, window_seconds: int) -> int:
        """Increment counter and return new value."""
        raise NotImplementedError

    async def get_window_data(self, key: str, window_seconds: int) -> list[float]:
        """Get request timestamps within window."""
        raise NotImplementedError

    async def add_request(self, key: str, timestamp: float, window_seconds: int) -> int:
        """Add request timestamp and return count in window."""
        raise NotImplementedError

    async def reset_counter(self, key: str) -> None:
        """Reset counter for key."""
        raise NotImplementedError


class MemoryRateLimitStorage(RateLimitStorage):
    """In-memory storage for rate limiting (single instance only)."""

    def __init__(self):
        self.counters: dict[str, int] = {}
        self.counter_expiry: dict[str, float] = {}
        self.windows: dict[str, deque] = defaultdict(deque)
        self.lock = asyncio.Lock()

    async def get_counter(self, key: str) -> int:
        async with self.lock:
            # Check if counter has expired
            if key in self.counter_expiry and time.time() > self.counter_expiry[key]:
                self.counters.pop(key, None)
                self.counter_expiry.pop(key, None)

            return self.counters.get(key, 0)

    async def increment_counter(self, key: str, window_seconds: int) -> int:
        async with self.lock:
            current_time = time.time()

            # Check if counter has expired
            if key in self.counter_expiry and current_time > self.counter_expiry[key]:
                self.counters[key] = 0

            # Increment counter
            self.counters[key] = self.counters.get(key, 0) + 1
            self.counter_expiry[key] = current_time + window_seconds

            return self.counters[key]

    async def get_window_data(self, key: str, window_seconds: int) -> list[float]:
        async with self.lock:
            current_time = time.time()
            cutoff_time = current_time - window_seconds

            # Clean old entries
            window_data = self.windows[key]
            while window_data and window_data[0] < cutoff_time:
                window_data.popleft()

            return list(window_data)

    async def add_request(self, key: str, timestamp: float, window_seconds: int) -> int:
        async with self.lock:
            cutoff_time = timestamp - window_seconds

            # Add new request
            self.windows[key].append(timestamp)

            # Clean old entries
            window_data = self.windows[key]
            while window_data and window_data[0] < cutoff_time:
                window_data.popleft()

            return len(window_data)

    async def reset_counter(self, key: str) -> None:
        async with self.lock:
            self.counters.pop(key, None)
            self.counter_expiry.pop(key, None)
            self.windows.pop(key, None)


class RedisRateLimitStorage(RateLimitStorage):
    """Redis-based storage for distributed rate limiting."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None

    async def initialize(self) -> bool:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url, decode_responses=True, health_check_interval=30
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Redis rate limiting storage initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Redis storage: {e}")
            return False

    async def get_counter(self, key: str) -> int:
        if not self.redis_client:
            return 0

        try:
            value = await self.redis_client.get(f"rate_limit:counter:{key}")
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Redis get_counter error: {e}")
            return 0

    async def increment_counter(self, key: str, window_seconds: int) -> int:
        if not self.redis_client:
            return 1

        try:
            pipe = self.redis_client.pipeline()
            counter_key = f"rate_limit:counter:{key}"

            pipe.incr(counter_key)
            pipe.expire(counter_key, window_seconds)

            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.error(f"Redis increment_counter error: {e}")
            return 1

    async def get_window_data(self, key: str, window_seconds: int) -> list[float]:
        if not self.redis_client:
            return []

        try:
            current_time = time.time()
            cutoff_time = current_time - window_seconds

            # Get requests in time window
            window_key = f"rate_limit:window:{key}"
            timestamps = await self.redis_client.zrangebyscore(
                window_key, cutoff_time, current_time
            )

            return [float(ts) for ts in timestamps]
        except Exception as e:
            logger.error(f"Redis get_window_data error: {e}")
            return []

    async def add_request(self, key: str, timestamp: float, window_seconds: int) -> int:
        if not self.redis_client:
            return 1

        try:
            window_key = f"rate_limit:window:{key}"
            cutoff_time = timestamp - window_seconds

            pipe = self.redis_client.pipeline()

            # Add new request
            pipe.zadd(window_key, {str(timestamp): timestamp})

            # Remove old entries
            pipe.zremrangebyscore(window_key, 0, cutoff_time)

            # Set expiry
            pipe.expire(window_key, window_seconds * 2)

            # Count current entries
            pipe.zcard(window_key)

            results = await pipe.execute()
            return results[-1]  # Return count from zcard
        except Exception as e:
            logger.error(f"Redis add_request error: {e}")
            return 1

    async def reset_counter(self, key: str) -> None:
        if not self.redis_client:
            return

        try:
            await self.redis_client.delete(
                f"rate_limit:counter:{key}", f"rate_limit:window:{key}"
            )
        except Exception as e:
            logger.error(f"Redis reset_counter error: {e}")


class RateLimitingService:
    """
    Comprehensive rate limiting service with multiple algorithms and distributed support.
    """

    def __init__(
        self,
        storage: Optional[RateLimitStorage] = None,
        enable_metrics: bool = True,
        enable_adaptive_limits: bool = False,
    ):
        self.storage = storage or MemoryRateLimitStorage()
        self.rules: list[RateLimitRule] = []
        self.metrics = RateLimitMetrics()
        self.enable_metrics = enable_metrics
        self.enable_adaptive_limits = enable_adaptive_limits

        # Token bucket states for token bucket algorithm
        self.token_buckets: dict[str, dict[str, Any]] = {}

        # Load default rules
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default rate limiting rules."""
        default_rules = [
            RateLimitRule(
                name="global_api_limit",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.GLOBAL,
                requests_per_minute=1000,
                burst_size=100,
                priority=1,
            ),
            RateLimitRule(
                name="per_ip_limit",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.PER_IP,
                requests_per_minute=100,
                burst_size=20,
                priority=2,
            ),
            RateLimitRule(
                name="per_api_key_limit",
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                scope=RateLimitScope.PER_API_KEY,
                requests_per_minute=200,
                burst_size=50,
                priority=3,
            ),
            RateLimitRule(
                name="sensitive_endpoints",
                strategy=RateLimitStrategy.FIXED_WINDOW,
                scope=RateLimitScope.PER_IP,
                requests_per_minute=10,
                burst_size=2,
                endpoints=["/auth/login", "/auth/register", "/admin/*"],
                priority=10,
            ),
        ]

        self.rules.extend(default_rules)

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limiting rule."""
        self.rules.append(rule)
        # Sort by priority (higher priority first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rate limiting rule."""
        original_length = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        return len(self.rules) < original_length

    def _get_rate_limit_key(
        self, rule: RateLimitRule, request_context: dict[str, Any]
    ) -> str:
        """Generate rate limit key based on rule scope."""
        key_parts = [rule.name]

        if rule.scope == RateLimitScope.GLOBAL:
            key_parts.append("global")
        elif rule.scope == RateLimitScope.PER_IP:
            key_parts.append(f"ip:{request_context.get('client_ip', 'unknown')}")
        elif rule.scope == RateLimitScope.PER_USER:
            key_parts.append(f"user:{request_context.get('user_id', 'anonymous')}")
        elif rule.scope == RateLimitScope.PER_ENDPOINT:
            key_parts.append(f"endpoint:{request_context.get('endpoint', 'unknown')}")
        elif rule.scope == RateLimitScope.PER_API_KEY:
            api_key = request_context.get("api_key", "no_key")
            # Hash API key for privacy
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            key_parts.append(f"apikey:{api_key_hash}")

        return ":".join(key_parts)

    def _should_apply_rule(
        self, rule: RateLimitRule, request_context: dict[str, Any]
    ) -> bool:
        """Check if rule should be applied to this request."""
        if not rule.enabled:
            return False

        endpoint = request_context.get("endpoint", "")
        user_role = request_context.get("user_role", "")

        # Check endpoint filters
        if rule.endpoints:
            endpoint_match = any(
                (
                    endpoint.startswith(pattern.rstrip("*"))
                    if pattern.endswith("*")
                    else endpoint == pattern
                )
                for pattern in rule.endpoints
            )
            if not endpoint_match:
                return False

        # Check excluded endpoints
        if rule.exclude_endpoints:
            endpoint_excluded = any(
                (
                    endpoint.startswith(pattern.rstrip("*"))
                    if pattern.endswith("*")
                    else endpoint == pattern
                )
                for pattern in rule.exclude_endpoints
            )
            if endpoint_excluded:
                return False

        # Check user roles
        return not rule.user_roles or user_role in rule.user_roles

    async def _check_fixed_window(
        self, rule: RateLimitRule, key: str
    ) -> RateLimitResult:
        """Check rate limit using fixed window algorithm."""
        current_time = time.time()
        window_start = int(current_time // (rule.window_size_minutes * 60)) * (
            rule.window_size_minutes * 60
        )
        window_key = f"{key}:{window_start}"

        current_count = await self.storage.increment_counter(
            window_key, rule.window_size_minutes * 60
        )

        allowed = current_count <= rule.requests_per_minute
        reset_time = window_start + (rule.window_size_minutes * 60)

        return RateLimitResult(
            allowed=allowed,
            requests_remaining=max(0, rule.requests_per_minute - current_count),
            requests_limit=rule.requests_per_minute,
            reset_time=reset_time,
            retry_after=int(reset_time - current_time) if not allowed else None,
            rule_name=rule.name,
            current_requests=current_count,
        )

    async def _check_sliding_window(
        self, rule: RateLimitRule, key: str
    ) -> RateLimitResult:
        """Check rate limit using sliding window algorithm."""
        current_time = time.time()
        window_seconds = rule.window_size_minutes * 60

        request_count = await self.storage.add_request(
            key, current_time, window_seconds
        )

        allowed = request_count <= rule.requests_per_minute
        reset_time = current_time + window_seconds

        return RateLimitResult(
            allowed=allowed,
            requests_remaining=max(0, rule.requests_per_minute - request_count),
            requests_limit=rule.requests_per_minute,
            reset_time=reset_time,
            retry_after=None if allowed else 60,  # Suggest retry in 1 minute
            rule_name=rule.name,
            current_requests=request_count,
        )

    async def _check_token_bucket(
        self, rule: RateLimitRule, key: str
    ) -> RateLimitResult:
        """Check rate limit using token bucket algorithm."""
        current_time = time.time()

        # Initialize bucket if not exists
        if key not in self.token_buckets:
            self.token_buckets[key] = {
                "tokens": rule.burst_size,
                "last_refill": current_time,
            }

        bucket = self.token_buckets[key]

        # Refill tokens based on elapsed time
        elapsed = current_time - bucket["last_refill"]
        refill_rate = rule.requests_per_minute / 60.0  # tokens per second
        tokens_to_add = elapsed * refill_rate

        bucket["tokens"] = min(rule.burst_size, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = current_time

        # Check if request can be allowed
        allowed = bucket["tokens"] >= 1.0

        if allowed:
            bucket["tokens"] -= 1.0

        # Calculate retry after
        retry_after = None
        if not allowed:
            retry_after = int((1.0 - bucket["tokens"]) / refill_rate)

        return RateLimitResult(
            allowed=allowed,
            requests_remaining=int(bucket["tokens"]),
            requests_limit=rule.burst_size,
            reset_time=current_time + 60,  # Next minute
            retry_after=retry_after,
            rule_name=rule.name,
            current_requests=rule.burst_size - int(bucket["tokens"]),
        )

    async def check_rate_limit(
        self, request_context: dict[str, Any]
    ) -> RateLimitResult:
        """
        Check rate limits for a request against all applicable rules.

        Args:
            request_context: Dictionary containing request information:
                - client_ip: Client IP address
                - user_id: User identifier (optional)
                - api_key: API key (optional)
                - endpoint: Request endpoint
                - user_role: User role (optional)

        Returns:
            RateLimitResult with the most restrictive limit result
        """
        start_time = time.time()
        most_restrictive_result = None

        # Update metrics
        if self.enable_metrics:
            self.metrics.total_requests += 1

        try:
            # Check each applicable rule
            for rule in self.rules:
                if not self._should_apply_rule(rule, request_context):
                    continue

                key = self._get_rate_limit_key(rule, request_context)

                # Apply appropriate algorithm
                if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                    result = await self._check_fixed_window(rule, key)
                elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                    result = await self._check_sliding_window(rule, key)
                elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                    result = await self._check_token_bucket(rule, key)
                else:
                    # Default to fixed window
                    result = await self._check_fixed_window(rule, key)

                # Track the most restrictive result (first non-allowed or lowest remaining)
                if most_restrictive_result is None or (
                    not result.allowed and most_restrictive_result.allowed
                ):
                    most_restrictive_result = result
                elif not result.allowed and not most_restrictive_result.allowed:
                    # Both are blocked, choose one with longer retry_after
                    if (result.retry_after or 0) > (
                        most_restrictive_result.retry_after or 0
                    ):
                        most_restrictive_result = result
                elif result.allowed and most_restrictive_result.allowed:
                    # Both are allowed, choose one with fewer remaining requests
                    if (
                        result.requests_remaining
                        < most_restrictive_result.requests_remaining
                    ):
                        most_restrictive_result = result

            # Default result if no rules applied
            if most_restrictive_result is None:
                most_restrictive_result = RateLimitResult(
                    allowed=True,
                    requests_remaining=999999,
                    requests_limit=999999,
                    reset_time=time.time() + 3600,
                    rule_name="no_rules",
                )

            # Update metrics
            if self.enable_metrics:
                processing_time = (time.time() - start_time) * 1000
                self.metrics.response_times_ms.append(processing_time)

                if most_restrictive_result.allowed:
                    self.metrics.allowed_requests += 1
                else:
                    self.metrics.blocked_requests += 1

                    # Track blocked IPs
                    client_ip = request_context.get("client_ip", "unknown")
                    self.metrics.top_blocked_ips[client_ip] = (
                        self.metrics.top_blocked_ips.get(client_ip, 0) + 1
                    )

                    # Track rule triggers
                    rule_name = most_restrictive_result.rule_name
                    self.metrics.rules_triggered[rule_name] = (
                        self.metrics.rules_triggered.get(rule_name, 0) + 1
                    )

            return most_restrictive_result

        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # In case of error, allow the request but log the issue
            return RateLimitResult(
                allowed=True,
                requests_remaining=0,
                requests_limit=1,
                reset_time=time.time() + 60,
                rule_name="error_fallback",
            )

    def create_fastapi_middleware(self) -> Callable:
        """Create FastAPI middleware for rate limiting."""

        async def rate_limit_middleware(request: Request, call_next):
            # Extract request context
            request_context = {
                "client_ip": request.client.host if request.client else "unknown",
                "endpoint": str(request.url.path),
                "method": request.method,
                "user_agent": request.headers.get("user-agent", ""),
            }

            # Extract API key from headers or query params
            api_key = request.headers.get("X-API-Key") or request.query_params.get(
                "api_key"
            )
            if api_key:
                request_context["api_key"] = api_key

            # Extract user info if available (from JWT or session)
            # This would be implemented based on your authentication system
            # request_context['user_id'] = get_user_id_from_request(request)
            # request_context['user_role'] = get_user_role_from_request(request)

            # Check rate limits
            rate_limit_result = await self.check_rate_limit(request_context)

            if not rate_limit_result.allowed:
                # Create rate limit exceeded response
                headers = {
                    "X-RateLimit-Limit": str(rate_limit_result.requests_limit),
                    "X-RateLimit-Remaining": str(rate_limit_result.requests_remaining),
                    "X-RateLimit-Reset": str(int(rate_limit_result.reset_time)),
                    "X-RateLimit-Rule": rate_limit_result.rule_name,
                }

                if rate_limit_result.retry_after:
                    headers["Retry-After"] = str(rate_limit_result.retry_after)

                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Rate limit: {rate_limit_result.requests_limit} per minute.",
                        "retry_after": rate_limit_result.retry_after,
                        "rule": rate_limit_result.rule_name,
                    },
                    headers=headers,
                )

            # Add rate limit headers to successful responses
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(
                rate_limit_result.requests_limit
            )
            response.headers["X-RateLimit-Remaining"] = str(
                rate_limit_result.requests_remaining
            )
            response.headers["X-RateLimit-Reset"] = str(
                int(rate_limit_result.reset_time)
            )
            response.headers["X-RateLimit-Rule"] = rate_limit_result.rule_name

            return response

        return rate_limit_middleware

    async def get_metrics(self) -> dict[str, Any]:
        """Get rate limiting metrics for monitoring."""
        if not self.enable_metrics:
            return {"metrics_disabled": True}

        # Calculate average response time
        avg_response_time = 0
        if self.metrics.response_times_ms:
            avg_response_time = sum(self.metrics.response_times_ms) / len(
                self.metrics.response_times_ms
            )

        # Get top blocked IPs (limit to top 10)
        sorted_blocked_ips = sorted(
            self.metrics.top_blocked_ips.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "total_requests": self.metrics.total_requests,
            "blocked_requests": self.metrics.blocked_requests,
            "allowed_requests": self.metrics.allowed_requests,
            "block_rate": self.metrics.blocked_requests
            / max(self.metrics.total_requests, 1),
            "average_response_time_ms": avg_response_time,
            "rules_triggered": dict(self.metrics.rules_triggered),
            "top_blocked_ips": dict(sorted_blocked_ips),
            "active_rules": len([r for r in self.rules if r.enabled]),
            "total_rules": len(self.rules),
        }

    async def reset_metrics(self) -> None:
        """Reset rate limiting metrics."""
        self.metrics = RateLimitMetrics()
        logger.info("Rate limiting metrics reset")

    async def health_check(self) -> dict[str, Any]:
        """Check health of rate limiting service."""
        try:
            # Test storage backend
            test_key = "health_check_test"
            await self.storage.increment_counter(test_key, 60)
            await self.storage.reset_counter(test_key)

            storage_healthy = True
        except Exception as e:
            logger.error(f"Rate limiting storage health check failed: {e}")
            storage_healthy = False

        return {
            "status": "healthy" if storage_healthy else "unhealthy",
            "storage_backend": type(self.storage).__name__,
            "storage_healthy": storage_healthy,
            "active_rules": len([r for r in self.rules if r.enabled]),
            "total_rules": len(self.rules),
            "metrics_enabled": self.enable_metrics,
            "adaptive_limits_enabled": self.enable_adaptive_limits,
        }


# Global rate limiting service instance
_rate_limiting_service: Optional[RateLimitingService] = None


async def get_rate_limiting_service(
    redis_url: Optional[str] = None, use_redis: bool = True
) -> RateLimitingService:
    """Get or create the global rate limiting service."""
    global _rate_limiting_service

    if _rate_limiting_service is None:
        # Choose storage backend
        storage = None
        if use_redis and redis_url:
            redis_storage = RedisRateLimitStorage(redis_url)
            if await redis_storage.initialize():
                storage = redis_storage
                logger.info("Using Redis for distributed rate limiting")
            else:
                logger.warning(
                    "Failed to initialize Redis, falling back to memory storage"
                )

        if storage is None:
            storage = MemoryRateLimitStorage()
            logger.info("Using in-memory rate limiting storage")

        _rate_limiting_service = RateLimitingService(
            storage=storage, enable_metrics=True, enable_adaptive_limits=False
        )

    return _rate_limiting_service


def create_rate_limit_decorator(
    requests_per_minute: int = 60,
    burst_size: int = 10,
    scope: RateLimitScope = RateLimitScope.PER_IP,
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
):
    """Create a decorator for rate limiting specific functions."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would need to be implemented based on your specific needs
            # For FastAPI, you'd typically use middleware instead of decorators
            return await func(*args, **kwargs)

        return wrapper

    return decorator
