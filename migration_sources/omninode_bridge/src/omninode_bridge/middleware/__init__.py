"""Middleware components for OmniNode Bridge services."""

from .rate_limiter import (
    RateLimitMiddleware,
    TokenBucketRateLimiter,
    add_rate_limit_middleware,
)
from .request_correlation import (
    RequestCorrelationMiddleware,
    add_correlation_headers,
    add_request_correlation_middleware,
    configure_request_correlation_logging,
    get_correlation_context,
    get_parent_span_id,
    get_request_id,
    get_trace_id,
)

__all__ = [
    # Request correlation
    "RequestCorrelationMiddleware",
    "add_correlation_headers",
    "add_request_correlation_middleware",
    "configure_request_correlation_logging",
    "get_correlation_context",
    "get_parent_span_id",
    "get_request_id",
    "get_trace_id",
    # Rate limiting
    "RateLimitMiddleware",
    "TokenBucketRateLimiter",
    "add_rate_limit_middleware",
]
