# Registry Effect Node

## Overview

`NodeRegistryEffect` is an ONEX Effect node responsible for dual-backend node registration. It executes I/O operations against both Consul (service discovery) and PostgreSQL (registration persistence), with support for partial failure handling.

> **Important**: Effect nodes are **single-shot operations** - they do NOT implement retries.
> Retry logic is the exclusive responsibility of the orchestrator layer.
> See [Retry Strategy: Orchestrator-Owned Retries](../../../../docs/patterns/retry_backoff_compensation_strategy.md#architectural-responsibility-orchestrator-owned-retries) for details.

### Architecture

```
                                    +-------------------+
                                    |  RegistryReducer  |
                                    | (emits intents)   |
                                    +---------+---------+
                                              |
                                              v
+-----------------------------+     +-------------------+
|  IdempotencyStore          |<--->| NodeRegistryEffect|
| (tracks completed backends) |     +--------+----------+
+-----------------------------+              |
                                    +--------+--------+
                                    |                 |
                              +-----v-----+     +-----v-----+
                              |  Consul   |     | PostgreSQL|
                              |  Client   |     |  Adapter  |
                              +-----------+     +-----------+
```

The Effect node:
- Receives registration requests from the Reducer
- Checks idempotency store for already-completed backends
- Executes I/O operations against external backends
- Returns structured responses with per-backend results
- Supports partial failure handling and targeted retries (via idempotency tracking)

> **Note**: "Targeted retries" refers to the idempotency store tracking which backends have completed,
> allowing the orchestrator to retry only failed backends. The effect itself is single-shot - the
> orchestrator initiates each retry attempt.

## Memory Characteristics

### Idempotency Store Memory Usage

The default `InMemoryEffectIdempotencyStore` uses bounded memory with LRU eviction.

| Configuration | Entries | Memory Usage (Estimated) |
|---------------|---------|--------------------------|
| Default       | 10,000  | ~1 MB                    |
| Low memory    | 1,000   | ~100 KB                  |
| High volume   | 100,000 | ~10 MB                   |
| Maximum       | 1,000,000 | ~100 MB                |

### Per-Entry Memory Breakdown

Each cache entry consists of:

| Component | Size | Description |
|-----------|------|-------------|
| UUID key (correlation_id) | 16 bytes | Unique operation identifier |
| Backend set | ~40 bytes | Set of completed backend strings ("consul", "postgres") |
| Timestamp (created_at) | 8 bytes | Monotonic timestamp for TTL |
| Timestamp (accessed_at) | 8 bytes | Monotonic timestamp for LRU |
| Python overhead | ~28 bytes | Object header, slots overhead |
| **Total** | **~100 bytes** | Per correlation_id entry |

### Memory Formula

```
Total Memory = max_cache_size * 100 bytes + OrderedDict overhead (~50KB base)
```

For practical planning:
- **10K entries**: 10,000 * 100B + 50KB = ~1.05 MB
- **100K entries**: 100,000 * 100B + 50KB = ~10.05 MB

## Performance Characteristics

### Time Complexity

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| `mark_completed()` | O(1) amortized | Hash lookup + set add + OrderedDict move_to_end |
| `is_completed()` | O(1) | Hash lookup + set contains |
| `get_completed_backends()` | O(k) | Where k = number of backends (typically 2) |
| `clear()` | O(1) | Hash delete |
| LRU eviction | O(1) | OrderedDict popitem(last=False) |
| TTL cleanup | O(n) | Full scan, but runs lazily on interval |

### Throughput Benchmarks

Based on performance tests (with mock backends):

| Scenario | Operations | Duration | Throughput |
|----------|------------|----------|------------|
| Sequential writes | 1,000 | < 2s | > 500 ops/sec |
| Concurrent writes (100 workers) | 100 | < 1s | > 100 ops/sec |
| Concurrent writes (500 workers) | 500 | < 3s | > 166 ops/sec |
| Sustained throughput | N/A | 1 second | > 5,000 ops/sec |
| Concurrent throughput (10 workers) | 10,000 | N/A | > 10,000 ops/sec |

### Latency Expectations

| Percentile | Target (with mocks) | Notes |
|------------|---------------------|-------|
| p50 | < 10ms | Median operation |
| p95 | < 50ms | 95th percentile |
| p99 | < 100ms | 99th percentile, outliers |

**Note**: Actual latency depends heavily on backend performance (Consul, PostgreSQL network latency). Mock benchmarks isolate store overhead.

## Configuration Guide

### ModelEffectIdempotencyConfig

```python
from omnibase_infra.nodes.effects.models import ModelEffectIdempotencyConfig

config = ModelEffectIdempotencyConfig(
    max_cache_size=10000,           # Default: 10,000 entries
    cache_ttl_seconds=3600.0,       # Default: 1 hour
    cleanup_interval_seconds=300.0, # Default: 5 minutes
)
```

### Configuration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_cache_size` | 10,000 | 1 - 1,000,000 | Maximum entries before LRU eviction |
| `cache_ttl_seconds` | 3600.0 (1 hour) | 1 - 86,400 | Entry time-to-live |
| `cleanup_interval_seconds` | 300.0 (5 min) | 1 - 3,600 | Minimum interval between TTL cleanup passes |

### Recommended Values by Deployment Size

| Deployment | max_cache_size | cache_ttl_seconds | Memory |
|------------|----------------|-------------------|--------|
| Development/Testing | 1,000 | 300 (5 min) | ~100 KB |
| Small (< 100 nodes) | 5,000 | 1,800 (30 min) | ~500 KB |
| Medium (100-1000 nodes) | 10,000 | 3,600 (1 hour) | ~1 MB |
| Large (1000+ nodes) | 50,000 | 3,600 (1 hour) | ~5 MB |

### Selecting TTL

Choose TTL based on:
1. **Expected retry window**: TTL should exceed the maximum time a registration retry might occur
2. **Backend failure recovery time**: If backends take 10 minutes to recover, TTL should be > 10 minutes
3. **Memory constraints**: Shorter TTL = faster eviction = lower memory

## Production Considerations

### In-Memory Store Limitations

The default `InMemoryEffectIdempotencyStore` has critical limitations:

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Not persistent** | Data lost on restart | Use persistent backend in production |
| **Not distributed** | Multi-instance deployments will have inconsistent state | Use shared backend (Redis, PostgreSQL) |
| **Single-process only** | Cannot share across processes | Use IPC or shared backend |

### Multi-Instance Deployment Warning

**The in-memory store does NOT work correctly in multi-instance deployments:**

```
Instance A: mark_completed(uuid1, "consul") -> SUCCESS
Instance B: is_completed(uuid1, "consul") -> FALSE (!)  # Different memory space
```

Each instance maintains its own cache, leading to:
- Duplicate backend calls
- Inconsistent idempotency behavior
- Potential data corruption in backends

### Recommended Persistent Backends

For production distributed deployments, implement `ProtocolEffectIdempotencyStore` with:

| Backend | Pros | Cons | Use Case |
|---------|------|------|----------|
| **Redis/Valkey** | Fast, native TTL, distributed | Requires infrastructure | High-volume, low-latency |
| **PostgreSQL** | Durable, existing infrastructure | Higher latency | Consistency-critical |
| **Consul KV** | Already in stack | Limited querying | Simple deployments |

### Implementing a Persistent Store

```python
from omnibase_infra.nodes.effects.protocol_effect_idempotency_store import (
    ProtocolEffectIdempotencyStore,
)
from uuid import UUID


class RedisEffectIdempotencyStore(ProtocolEffectIdempotencyStore):
    """Redis-backed idempotency store for production use."""

    async def mark_completed(self, correlation_id: UUID, backend: str) -> None:
        key = f"idempotency:{correlation_id}"
        await self._redis.sadd(key, backend)
        await self._redis.expire(key, self._ttl_seconds)

    async def is_completed(self, correlation_id: UUID, backend: str) -> bool:
        key = f"idempotency:{correlation_id}"
        return await self._redis.sismember(key, backend)

    async def get_completed_backends(self, correlation_id: UUID) -> set[str]:
        key = f"idempotency:{correlation_id}"
        members = await self._redis.smembers(key)
        return {m.decode() for m in members}

    async def clear(self, correlation_id: UUID) -> None:
        key = f"idempotency:{correlation_id}"
        await self._redis.delete(key)
```

## Usage Examples

### Basic Usage

```python
from unittest.mock import AsyncMock

from omnibase_infra.nodes.effects import NodeRegistryEffect
from omnibase_infra.nodes.effects.models import ModelRegistryRequest

# Create mock clients
consul_client = AsyncMock()
consul_client.register_service.return_value = {"success": True}

postgres_adapter = AsyncMock()
postgres_adapter.upsert.return_value = {"success": True}

# Create effect node with default config
effect = NodeRegistryEffect(
    consul_client=consul_client,
    postgres_adapter=postgres_adapter,
)

# Execute registration
request = ModelRegistryRequest(
    node_id="my-node-001",
    node_type="effect",
    node_version="1.0.0",
    correlation_id=uuid4(),
)

response = await effect.register_node(request)
print(f"Status: {response.status}")  # "success" or "partial" or "failure"
```

### Custom Configuration (Smaller Cache)

```python
from omnibase_infra.nodes.effects.models import ModelEffectIdempotencyConfig

# For development/testing - smaller cache, shorter TTL
config = ModelEffectIdempotencyConfig(
    max_cache_size=1000,
    cache_ttl_seconds=300.0,  # 5 minutes
)

effect = NodeRegistryEffect(
    consul_client=consul_client,
    postgres_adapter=postgres_adapter,
    idempotency_config=config,
)
```

### Production Configuration (Persistent Store)

```python
# Use a persistent store for production
redis_store = RedisEffectIdempotencyStore(
    redis_client=redis_client,
    ttl_seconds=3600,
)

effect = NodeRegistryEffect(
    consul_client=consul_client,
    postgres_adapter=postgres_adapter,
    idempotency_store=redis_store,  # Custom store takes precedence
)
```

### Retry After Partial Failure

```python
# First attempt - Consul succeeds, PostgreSQL fails
response = await effect.register_node(request)
# response.status == "partial"
# response.consul.success == True
# response.postgres.success == False

# Retry only PostgreSQL (Consul already completed, stored in idempotency cache)
response = await effect.register_node(request)
# response.status == "success"
# Both backends now completed
```

## Thread Safety

### Async Safety Guarantees

The `InMemoryEffectIdempotencyStore` provides the following guarantees:

1. **All operations are atomic**: Protected by `asyncio.Lock`
2. **No race conditions**: LRU updates and evictions are synchronized
3. **Safe for concurrent access**: Multiple coroutines can call store methods safely

### Lock Scope

```python
async def mark_completed(self, correlation_id: UUID, backend: str) -> None:
    async with self._lock:  # Lock held for entire operation
        await self._maybe_cleanup_expired()
        # ... rest of operation
```

### Performance Under Contention

With 10+ concurrent writers:
- Lock contention increases latency slightly
- Throughput remains > 10,000 ops/sec
- No deadlocks or data corruption

## Related Components

| Component | Description |
|-----------|-------------|
| `ProtocolEffectIdempotencyStore` | Protocol interface for pluggable backends |
| `ModelEffectIdempotencyConfig` | Configuration model for in-memory store |
| `ModelRegistryRequest` | Input request model |
| `ModelRegistryResponse` | Output response model |
| `ModelBackendResult` | Per-backend result model |
| `RegistrationReducer` | Emits intents consumed by this Effect |

## Related Tickets

- **OMN-954**: Registry Effect Node idempotency and retry behavior testing
- **OMN-944**: Registration projection schema

## Testing

Run effect tests:

```bash
# Unit tests
poetry run pytest tests/unit/registration/effect/ -v

# Performance tests
poetry run pytest tests/performance/registration/effect/ -v -m performance

# Skip performance tests in CI
poetry run pytest tests/ -m "not performance"
```
