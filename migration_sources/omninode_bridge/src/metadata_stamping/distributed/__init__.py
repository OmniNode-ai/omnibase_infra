"""
Distributed architecture components for MetadataStampingService.

Provides database sharding, circuit breaker patterns, and distributed
system resilience capabilities for high-scale deployments.
"""

from .circuit_breaker import CircuitBreakerManager, DistributedCircuitBreaker
from .sharding import DatabaseShardManager, ShardHealthMonitor, ShardRouter

__all__ = [
    "DatabaseShardManager",
    "ShardRouter",
    "ShardHealthMonitor",
    "DistributedCircuitBreaker",
    "CircuitBreakerManager",
]
