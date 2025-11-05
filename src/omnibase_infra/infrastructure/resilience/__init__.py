"""
Resilience utilities for ONEX Infrastructure.

Provides circuit breakers, retry policies, and rate limiting utilities
for building resilient infrastructure services.
"""

from omnibase_infra.infrastructure.resilience.circuit_breaker_factory import (
    create_database_circuit_breaker,
    create_kafka_circuit_breaker,
    create_network_circuit_breaker,
    create_vault_circuit_breaker,
)
from omnibase_infra.infrastructure.resilience.rate_limiter import (
    RateLimiter,
    TokenBucketLimiter,
    create_api_rate_limiter,
    create_database_rate_limiter,
)
from omnibase_infra.infrastructure.resilience.retry_policy import (
    create_database_retry_policy,
    create_kafka_retry_policy,
    create_network_retry_policy,
    create_vault_retry_policy,
)

__all__ = [  # noqa: RUF022 - Grouped by category for clarity
    # Circuit breakers
    "create_database_circuit_breaker",
    "create_kafka_circuit_breaker",
    "create_network_circuit_breaker",
    "create_vault_circuit_breaker",
    # Retry policies
    "create_database_retry_policy",
    "create_kafka_retry_policy",
    "create_network_retry_policy",
    "create_vault_retry_policy",
    # Rate limiters
    "RateLimiter",
    "TokenBucketLimiter",
    "create_api_rate_limiter",
    "create_database_rate_limiter",
]
