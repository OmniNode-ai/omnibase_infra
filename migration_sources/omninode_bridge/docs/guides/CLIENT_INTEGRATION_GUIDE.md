# Client Integration Layer - Implementation Guide

## Overview

This document describes the service client integration layer for OnexTree and MetadataStampingService, providing clean abstractions for the NodeBridgeOrchestrator.

## Architecture

### Components Created

1. **Base Infrastructure**
   - `BaseServiceClient` - Abstract base class with common patterns
   - `CircuitBreaker` - Resilience pattern implementation
   - Retry logic with exponential backoff
   - Correlation ID propagation
   - Health check interface

2. **Service Clients**
   - `AsyncMetadataStampingClient` - MetadataStampingService integration
   - `AsyncOnexTreeClient` - OnexTree Agent Intelligence integration

3. **Test Suite**
   - Comprehensive unit tests with >80% coverage target
   - Circuit breaker state machine tests
   - Service client functionality tests
   - Caching and performance tests

## File Structure

```
src/omninode_bridge/clients/
├── __init__.py                      # Package exports
├── base_client.py                   # BaseServiceClient abstract class
├── circuit_breaker.py               # CircuitBreaker implementation
├── metadata_stamping_client.py      # MetadataStamping client
└── onextree_client.py              # OnexTree client

tests/unit/clients/
├── __init__.py
├── test_circuit_breaker.py         # Circuit breaker tests
├── test_metadata_stamping_client.py # MetadataStamping client tests
└── test_onextree_client.py         # OnexTree client tests
```

## Usage Examples

### MetadataStamping Client

```python
from uuid import uuid4
from omninode_bridge.clients import AsyncMetadataStampingClient

# Initialize client
async with AsyncMetadataStampingClient("http://localhost:8053") as client:
    correlation_id = uuid4()

    # Generate hash
    file_data = b"content to hash"
    hash_result = await client.generate_hash(
        file_data=file_data,
        namespace="default",
        correlation_id=correlation_id
    )

    # Create metadata stamp
    stamp = await client.create_stamp(
        file_hash=hash_result["hash"],
        file_path="/path/to/file",
        file_size=len(file_data),
        stamp_data={"key": "value"},
        content_type="text/plain",
        namespace="omninode.services.metadata",
        correlation_id=correlation_id
    )

    # Validate stamps
    validation = await client.validate_stamp(
        content="content with stamps",
        namespace="omninode.services.metadata",
        correlation_id=correlation_id
    )

    # Batch operations
    batch_items = [
        {
            "file_hash": f"hash_{i}",
            "file_path": f"/path/{i}",
            "file_size": 100 * i,
            "stamp_data": {}
        }
        for i in range(10)
    ]

    batch_result = await client.batch_create_stamps(
        items=batch_items,
        namespace="omninode.services.metadata",
        correlation_id=correlation_id
    )
```

### OnexTree Client

```python
from uuid import uuid4
from omninode_bridge.clients import AsyncOnexTreeClient

# Initialize client with caching
async with AsyncOnexTreeClient(
    base_url="http://localhost:8054",
    enable_cache=True,
    cache_ttl=300
) as client:
    correlation_id = uuid4()

    # Get intelligence data
    intelligence = await client.get_intelligence(
        context="authentication patterns",
        include_patterns=True,
        include_relationships=True,
        correlation_id=correlation_id
    )

    # Query knowledge graph
    knowledge = await client.query_knowledge(
        query="Find all API endpoints",
        max_depth=3,
        filters={"type": "endpoint"},
        correlation_id=correlation_id
    )

    # Navigate tree
    tree_data = await client.navigate_tree(
        start_node="root",
        direction="both",
        max_nodes=100,
        correlation_id=correlation_id
    )

    # Get cache statistics
    cache_stats = client.get_cache_stats()
    print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
```

## Features

### Circuit Breaker Pattern

The circuit breaker protects against cascading failures:

```python
from omninode_bridge.clients.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState
)

# Create custom circuit breaker
config = CircuitBreakerConfig(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60,      # Wait 60s before retry
    half_open_max_calls=3,    # Max calls in half-open
    success_threshold=2,      # Close after 2 successes
    timeout=30.0             # Operation timeout
)

cb = CircuitBreaker("my-service", config)

# Use with async operations
async def risky_operation():
    return await external_service_call()

try:
    result = await cb.call(risky_operation)
except CircuitBreakerError:
    # Circuit is open, fail fast
    handle_service_unavailable()
```

**States:**
- **CLOSED**: Normal operation, requests allowed
- **OPEN**: Service failing, requests rejected immediately
- **HALF_OPEN**: Testing recovery, limited requests allowed

### Retry Logic

Exponential backoff retry with configurable attempts:

```python
client = AsyncMetadataStampingClient(
    base_url="http://localhost:8053",
    timeout=30.0,
    max_retries=3  # Retry up to 3 times
)

# Retries automatically on:
# - Network errors
# - Timeout errors
# - Temporary service unavailability
```

### Correlation ID Propagation

All requests include correlation IDs for distributed tracing:

```python
correlation_id = uuid4()

# Correlation ID is propagated through headers
result = await client.generate_hash(
    file_data=data,
    correlation_id=correlation_id
)

# All logs include correlation ID for tracing
```

### Health Checks

Built-in health check support:

```python
# Check service health
health = await client.health_check()
if health["status"] == "healthy":
    # Service is ready
    pass

# Validate connection
is_connected = await client._validate_connection()
```

### Metrics Collection

Clients track performance metrics:

```python
# Get client metrics
metrics = client.get_metrics()
print(f"Requests: {metrics['request_count']}")
print(f"Errors: {metrics['error_count']}")
print(f"Error rate: {metrics['error_rate']:.2%}")
print(f"Circuit state: {metrics['circuit_breaker']['state']}")

# For OnexTree: includes cache metrics
cache_stats = onextree_client.get_cache_stats()
print(f"Cache size: {cache_stats['cache_size']}")
print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
```

## Caching (OnexTree Client)

The OnexTree client includes intelligent caching:

```python
# Enable caching (default)
client = AsyncOnexTreeClient(
    enable_cache=True,
    cache_ttl=300  # 5 minutes
)

# Cached automatically
result1 = await client.get_intelligence(context="test")
result2 = await client.get_intelligence(context="test")  # From cache

# Bypass cache when needed
result3 = await client.get_intelligence(
    context="test",
    use_cache=False
)

# Clear cache manually
await client.clear_cache()

# Check cache performance
stats = client.get_cache_stats()
# {
#     "enabled": True,
#     "cache_size": 42,
#     "cache_hits": 150,
#     "cache_misses": 50,
#     "hit_rate": 0.75,
#     "ttl_seconds": 300
# }
```

**Cache Features:**
- Automatic expiration with TTL
- LRU-style eviction when size limit reached
- Thread-safe with async locks
- Hit tracking for performance monitoring
- Consistent key generation from parameters

## Error Handling

Clients provide structured error handling:

```python
from omninode_bridge.clients import (
    ClientError,
    ServiceUnavailableError,
    CircuitBreakerError
)

try:
    result = await client.generate_hash(file_data=data)

except CircuitBreakerError:
    # Circuit breaker is open - fail fast
    logger.error("Service circuit breaker is open")

except ServiceUnavailableError:
    # Service is down or unreachable
    logger.error("Service unavailable")

except ClientError as e:
    # General client error
    logger.error(f"Client error: {e}")

except Exception as e:
    # Unexpected error
    logger.exception("Unexpected error")
```

## Integration with NodeBridgeOrchestrator

### Example Integration

```python
from omninode_bridge.clients import (
    AsyncMetadataStampingClient,
    AsyncOnexTreeClient
)

class NodeBridgeOrchestrator:
    """Orchestrator with service client integration."""

    def __init__(
        self,
        metadata_service_url: str = "http://localhost:8053",
        onextree_service_url: str = "http://localhost:8054"
    ):
        # Initialize service clients
        self.metadata_client = AsyncMetadataStampingClient(
            base_url=metadata_service_url,
            timeout=30.0,
            max_retries=3
        )

        self.onextree_client = AsyncOnexTreeClient(
            base_url=onextree_service_url,
            timeout=30.0,
            max_retries=3,
            enable_cache=True,
            cache_ttl=300
        )

    async def initialize(self):
        """Initialize all service clients."""
        await self.metadata_client.initialize()
        await self.onextree_client.initialize()

        # Validate connections
        metadata_ok = await self.metadata_client._validate_connection()
        onextree_ok = await self.onextree_client._validate_connection()

        if not (metadata_ok and onextree_ok):
            raise RuntimeError("Service connection validation failed")

    async def close(self):
        """Close all service clients."""
        await self.metadata_client.close()
        await self.onextree_client.close()

    async def process_with_intelligence(
        self,
        file_data: bytes,
        file_path: str,
        context: str,
        correlation_id: UUID
    ) -> dict:
        """Process file with metadata stamping and intelligence."""

        # Generate hash with MetadataStamping
        hash_result = await self.metadata_client.generate_hash(
            file_data=file_data,
            correlation_id=correlation_id
        )

        # Get intelligence from OnexTree
        intelligence = await self.onextree_client.get_intelligence(
            context=context,
            correlation_id=correlation_id
        )

        # Create stamp with intelligence data
        stamp = await self.metadata_client.create_stamp(
            file_hash=hash_result["hash"],
            file_path=file_path,
            file_size=len(file_data),
            stamp_data={
                "intelligence": intelligence,
                "context": context
            },
            correlation_id=correlation_id
        )

        return {
            "hash": hash_result["hash"],
            "stamp": stamp,
            "intelligence": intelligence
        }
```

## Testing

### Running Tests

```bash
# Run all client tests
poetry run pytest tests/unit/clients/ -v

# Run specific test class
poetry run pytest tests/unit/clients/test_circuit_breaker.py::TestCircuitBreakerStates -v

# Run with coverage
poetry run pytest tests/unit/clients/ --cov=src/omninode_bridge/clients --cov-report=term-missing

# Run single test
poetry run pytest tests/unit/clients/test_circuit_breaker.py::TestCircuitBreakerStates::test_initial_state_is_closed -v
```

### Test Coverage

The test suite includes:

**Circuit Breaker Tests:**
- State transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
- Failure threshold detection
- Recovery timeout behavior
- Success threshold in HALF_OPEN state
- Metrics collection
- Manual control (reset, force open)
- Timeout handling
- Context manager usage

**MetadataStamping Client Tests:**
- Hash generation
- Stamp creation
- Stamp validation
- Stamp retrieval
- Batch operations
- Service metrics
- Health checks
- Error handling

**OnexTree Client Tests:**
- Intelligence retrieval
- Knowledge queries
- Tree navigation
- Cache operations
- Cache statistics
- Health checks
- Error handling

## Performance Considerations

### Connection Pooling

Clients use httpx connection pooling:

```python
# httpx.AsyncClient configuration
limits=httpx.Limits(
    max_connections=100,      # Total connections
    max_keepalive_connections=20  # Keep-alive pool
)
```

### Timeouts

Configurable timeouts at multiple levels:

```python
# Client-level default timeout
client = AsyncMetadataStampingClient(timeout=30.0)

# Per-operation timeout override
result = await circuit_breaker.call(
    operation,
    timeout=10.0  # Override for this call
)
```

### Caching Strategy

OnexTree client implements intelligent caching:

```python
# Cache size management
MAX_CACHE_SIZE = 1000  # entries

# Automatic eviction
if cache_size > MAX_CACHE_SIZE:
    # Remove oldest 20% of entries
    evict_least_recently_used(0.2)
```

## Security Considerations

### Correlation ID Propagation

All requests include correlation IDs in headers:

```http
X-Correlation-ID: 550e8400-e29b-41d4-a716-446655440000
```

### Error Message Sanitization

Error messages avoid leaking sensitive information:

```python
# Good: Generic error message
raise ClientError("Service request failed")

# Bad: Leaking internal details
# raise ClientError(f"Database connection failed: {db_connection_string}")
```

### Timeout Protection

Prevents hung requests from consuming resources:

```python
# All operations have timeouts
await asyncio.wait_for(
    operation(),
    timeout=30.0
)
```

## Monitoring and Observability

### Structured Logging

All clients use structured logging:

```python
logger.info(
    "Hash generation completed",
    extra={
        "correlation_id": str(correlation_id),
        "file_size": len(file_data),
        "execution_time_ms": 1.5
    }
)
```

### Metrics Export

Clients expose metrics for monitoring:

```python
metrics = client.get_metrics()
# Export to Prometheus, StatsD, etc.
prometheus_registry.register_metrics(metrics)
```

### Health Endpoints

Integration with service health checks:

```python
@app.get("/health")
async def health():
    metadata_health = await metadata_client.health_check()
    onextree_health = await onextree_client.health_check()

    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": {
            "metadata": metadata_health,
            "onextree": onextree_health
        }
    }
```

## Troubleshooting

### Circuit Breaker Issues

**Problem:** Circuit breaker opens too frequently

**Solution:** Adjust failure threshold or recovery timeout:

```python
config = CircuitBreakerConfig(
    failure_threshold=10,  # Increase threshold
    recovery_timeout=120,  # Longer recovery period
)
```

**Problem:** Circuit breaker doesn't open when it should

**Solution:** Check minimum throughput requirement:

```python
config = CircuitBreakerConfig(
    failure_threshold=5,
    minimum_throughput=10  # Needs at least 10 calls
)
```

### Cache Issues

**Problem:** Stale data in cache

**Solution:** Reduce TTL or clear cache explicitly:

```python
# Shorter TTL
client = AsyncOnexTreeClient(cache_ttl=60)

# Or clear cache
await client.clear_cache()
```

**Problem:** Memory usage from cache

**Solution:** Cache size is auto-managed, but you can clear it:

```python
# Periodic cache cleanup
async def cleanup_task():
    while True:
        await asyncio.sleep(3600)  # Every hour
        await client.clear_cache()
```

### Connection Issues

**Problem:** Service unavailable errors

**Solution:** Check service health and retry configuration:

```python
# Verify service is running
health = await client.health_check()

# Adjust retry attempts
client = AsyncMetadataStampingClient(
    max_retries=5  # More retries
)
```

## Future Enhancements

1. **Distributed Circuit Breaker**
   - Coordinate state across multiple instances
   - Use Redis for shared state

2. **Advanced Caching**
   - Distributed cache with Redis
   - Cache invalidation strategies
   - Cache warming on startup

3. **Rate Limiting**
   - Client-side rate limiting
   - Token bucket algorithm
   - Adaptive rate limiting

4. **Metrics Enrichment**
   - OpenTelemetry integration
   - Distributed tracing
   - Custom metrics exporters

5. **Connection Management**
   - DNS-based service discovery
   - Load balancing across replicas
   - Automatic failover

## References

- Circuit Breaker Pattern: https://martinfowler.com/bliki/CircuitBreaker.html
- httpx Documentation: https://www.python-httpx.org/
- Tenacity (Retry): https://tenacity.readthedocs.io/
- OpenTelemetry: https://opentelemetry.io/
