> **Navigation**: [Home](../index.md) > [Operations](README.md) > Thread Pool Tuning

# Thread Pool Tuning Operational Runbook

Operational guide for tuning thread pool configurations in omnibase_infra components.

## Overview

The omnibase_infra project uses `ThreadPoolExecutor` from `concurrent.futures` to handle synchronous blocking operations within async contexts. This pattern is essential for integrating synchronous client libraries with asyncio-based infrastructure.

> **Note**: The secret management backend migrated from HashiCorp Vault (hvac) to Infisical in OMN-2288. The Infisical SDK is async-native and does **not** require thread pool wrapping. The thread pool configuration documented below applies to legacy synchronous adapters only.

## Thread Pool Locations

### 1. Secret Management (Infisical - Async Native)

**Location**: `src/omnibase_infra/adapters/_internal/adapter_infisical.py`

**Purpose**: Infisical uses an async-native Python SDK (`infisicalsdk`). No thread pool is required. Concurrency is handled directly by asyncio.

**Configuration Model**: `ModelInfisicalHandlerConfig`

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_concurrent_operations` | 10 | 1-100 | Maximum worker threads in pool |
| `max_queue_size_multiplier` | 3 | 1-10 | Queue size = workers * multiplier |

**Calculated Values**:
- Thread Pool Size: `max_concurrent_operations`
- Max Queue Size: `max_concurrent_operations * max_queue_size_multiplier`
- Default: 10 workers, 30 queue capacity

### 2. Kafka Event Bus (Async Native)

**Location**: `src/omnibase_infra/event_bus/kafka_event_bus.py`

The Kafka Event Bus uses aiokafka which is natively async and does not use thread pools for I/O operations. It uses `asyncio.Lock` for thread-safe state management.

## Configuration Methods

### Method 1: Direct Configuration (Infisical)

```python
from omnibase_infra.adapters._internal.adapter_infisical import AdapterInfisical

adapter = AdapterInfisical()
await adapter.initialize({
    "infisical_addr": "http://192.168.86.200:8200",
    "client_id": "...",
    "client_secret": "...",
    "timeout_seconds": 30.0,
})
# No thread pool configuration needed - Infisical adapter is async-native
```

### Method 2: Configuration Model (Infisical)

```python
from pydantic import SecretStr
from omnibase_infra.handlers.models.infisical.model_infisical_handler_config import ModelInfisicalHandlerConfig

config = ModelInfisicalHandlerConfig(
    infisical_addr="http://192.168.86.200:8200",
    client_id="...",
    client_secret=SecretStr("..."),
    timeout_seconds=30.0,
)
# No thread pool configuration - async-native SDK
```

### Method 3: Environment Variables

Environment variables are not directly supported for thread pool configuration. Use configuration files or direct initialization.

## Tuning Guidelines

### High Throughput Scenarios

**Use Case**: High volume of concurrent Infisical operations (secret reads)

**Recommended Settings**:
```python
{
    "max_concurrent_operations": 50,    # Increased workers
    "max_queue_size_multiplier": 5,     # Larger queue buffer
    "timeout_seconds": 60.0,            # Longer timeout for load
    "circuit_breaker_failure_threshold": 10,  # More tolerance
}
```

**Rationale** (applies to legacy synchronous adapters):
- 50 workers allow 50 simultaneous API calls
- Queue capacity of 250 (50 * 5) handles burst traffic
- Longer timeout accounts for load
- Higher circuit breaker threshold prevents premature tripping

**Note**: Infisical uses async-native concurrency. Concurrency limits are controlled by asyncio semaphores, not thread pool size. No thread pool tuning is needed for Infisical.

### Low Latency Scenarios

**Use Case**: Real-time systems requiring fast secret access

**Recommended Settings**:
```python
{
    "max_concurrent_operations": 5,     # Small pool
    "max_queue_size_multiplier": 2,     # Minimal queuing
    "timeout_seconds": 5.0,             # Fast fail
    "circuit_breaker_failure_threshold": 3,  # Quick protection
}
```

**Rationale**:
- Small pool reduces context switching
- Minimal queue ensures fast feedback on overload
- Short timeout fails fast rather than waiting
- Lower circuit breaker threshold protects quickly

**Resource Impact**:
- Memory: ~5MB additional
- CPU: Minimal thread overhead
- Network: Up to 5 concurrent connections

### Resource-Constrained Scenarios

**Use Case**: Container/pod with limited CPU and memory

**Recommended Settings**:
```python
{
    "max_concurrent_operations": 3,     # Minimal workers
    "max_queue_size_multiplier": 10,    # Large queue instead of threads
    "timeout_seconds": 30.0,            # Standard timeout
    "circuit_breaker_failure_threshold": 5,  # Standard protection
}
```

**Rationale**:
- 3 workers minimize memory and CPU usage
- Large queue (30) buffers requests instead of more threads
- Trades latency for resource efficiency
- Standard circuit breaker prevents cascading failures

**Resource Impact**:
- Memory: ~3MB additional
- CPU: Minimal thread overhead
- Network: Up to 3 concurrent connections

### Kubernetes Pod Sizing

| Container Memory | Container CPU | Recommended Workers | Queue Multiplier |
|-----------------|---------------|---------------------|------------------|
| 128Mi           | 100m          | 2-3                 | 5                |
| 256Mi           | 250m          | 5-10                | 3                |
| 512Mi           | 500m          | 10-20               | 3                |
| 1Gi             | 1000m         | 20-50               | 5                |

## Monitoring Thread Pool Health

### Health Check Metrics

The Infisical adapter `health_check()` method returns connection metrics (no thread pool metrics since it is async-native):

```python
health = await adapter.health_check()
print(health)
# {
#     "healthy": True,
#     "initialized": True,
#     "handler_type": "infisical",
#     "timeout_seconds": 30.0,
#     "token_ttl_remaining_seconds": 3600,
#     "circuit_breaker_state": "closed",
#     "circuit_breaker_failure_count": 0,
#     "thread_pool_active_workers": 3,    # Current active threads
#     "thread_pool_max_workers": 10,      # Configured max
# }
```

### Key Metrics to Monitor

| Metric | Healthy Range | Warning | Critical |
|--------|---------------|---------|----------|
| `thread_pool_active_workers` | 0-50% of max | 50-80% of max | >80% of max |
| `circuit_breaker_state` | "closed" | "half_open" | "open" |
| `circuit_breaker_failure_count` | 0 | 1-threshold/2 | >threshold/2 |

### Prometheus Metrics (Custom Implementation)

Example custom metrics collector:

```python
from prometheus_client import Gauge

infisical_requests_active = Gauge(
    'infisical_adapter_active_requests',
    'Number of active Infisical requests',
    ['service']
)

circuit_breaker_state = Gauge(
    'infisical_adapter_circuit_breaker_open',
    'Circuit breaker open state (1=open, 0=closed)',
    ['service']
)

async def collect_metrics(adapter: object, service_name: str):
    health = await adapter.health_check()
    circuit_breaker_state.labels(service=service_name).set(
        1 if health.get('circuit_breaker_state') == 'open' else 0
    )
```

### Log-Based Monitoring

Enable debug logging to monitor thread pool behavior:

```python
import logging
logging.getLogger('omnibase_infra.adapters._internal.adapter_infisical').setLevel(logging.DEBUG)
```

Key log patterns to monitor:
- `"HandlerInfisical initialized"` - Shows configuration values
- `"Retrying Infisical operation"` - Indicates transient failures
- `"Circuit breaker opened"` - Service may be overwhelmed

## Troubleshooting

### Issue: Thread Starvation

**Symptoms**:
- Operations timing out despite Infisical being healthy
- High concurrency causing latency spikes

**Diagnosis**:
```python
health = await adapter.health_check()
print(f"Circuit breaker state: {health.get('circuit_breaker_state')}")
print(f"Failure count: {health.get('circuit_breaker_failure_count')}")
```

**Resolution** (Infisical - async-native):
1. Check network latency to Infisical
2. Verify Infisical server health at `http://192.168.86.200:8200`
3. Review circuit breaker failure threshold
4. Increase `timeout_seconds` if Infisical is under load

### Issue: Queue Buildup

**Symptoms**:
- Operations accepted but slow to complete
- Memory usage increasing
- Eventual timeouts

**Diagnosis**:
Monitor queue depth (requires custom instrumentation):
```python
import concurrent.futures

# Access internal queue (implementation detail)
if adapter._executor is not None:
    work_queue = adapter._executor._work_queue
    queue_size = work_queue.qsize()
    print(f"Queue depth: {queue_size}")
```

**Resolution**:
1. Reduce `max_queue_size_multiplier` for faster feedback
2. Increase `max_concurrent_operations` if resources allow
3. Implement request rate limiting upstream
4. Add backpressure mechanisms

### Issue: Circuit Breaker Tripping Frequently

**Symptoms**:
- `InfraUnavailableError` raised frequently
- Circuit alternating between open and closed
- Intermittent service availability

**Diagnosis**:
```python
health = await adapter.health_check()
if health['circuit_breaker_state'] == 'open':
    print(f"Circuit open! Failures: {health['circuit_breaker_failure_count']}")
```

**Resolution**:
1. Increase `circuit_breaker_failure_threshold`
2. Increase `circuit_breaker_reset_timeout_seconds`
3. Investigate root cause of failures
4. Check Infisical server health and network

### Issue: Memory Pressure

**Symptoms**:
- OOM kills in containers
- Increasing memory usage over time
- Slow garbage collection

**Diagnosis**:
```bash
# Check thread count in container
ps -eLf | wc -l
cat /proc/<pid>/status | grep Threads
```

**Resolution**:
1. Reduce `max_concurrent_operations`
2. Reduce `max_queue_size_multiplier`
3. Increase container memory limits
4. Add memory monitoring and alerting

## Circuit Breaker Integration

Thread pools work with the circuit breaker pattern for resilience:

### State Interactions

| Thread Pool State | Circuit State | Behavior |
|------------------|---------------|----------|
| Available threads | CLOSED | Normal operation |
| Saturated | CLOSED | Queue buildup |
| Available threads | OPEN | All requests blocked |
| Any | HALF_OPEN | Single test request allowed |

### Configuration Coordination

```python
# Coordinated configuration for resilience (Infisical - async-native)
config = ModelInfisicalHandlerConfig(
    infisical_addr="http://192.168.86.200:8200",
    client_id="...",
    client_secret=SecretStr("..."),
    # No thread pool - async-native SDK
    # Circuit breaker
    circuit_breaker_enabled=True,
    circuit_breaker_failure_threshold=5,
    circuit_breaker_reset_timeout_seconds=30.0,
    # Timeouts
    timeout_seconds=30.0,
)
```

### Retry Integration

The Infisical adapter uses exponential backoff with circuit breaker:

```python
# Retry configuration (ModelInfisicalHandlerConfig)
{
    "max_attempts": 3,              # Total attempts
    "initial_backoff_seconds": 0.1, # First retry delay
    "max_backoff_seconds": 10.0,    # Max retry delay
    "exponential_base": 2.0,        # Backoff multiplier
}
```

**Formula**: `delay = min(initial * (base ** attempt), max_backoff)`

Example delays: 0.1s, 0.2s, 0.4s, ... capped at 10s

## Production Deployment Checklist

### Pre-Deployment

- [ ] Thread pool size matches expected concurrent load
- [ ] Queue multiplier provides sufficient buffer
- [ ] Timeout values appropriate for Infisical latency
- [ ] Circuit breaker thresholds tuned for environment
- [ ] Container resources allocated for thread count
- [ ] Monitoring and alerting configured

### Post-Deployment

- [ ] Verify `thread_pool_active_workers` in expected range
- [ ] Confirm circuit breaker remains closed under load
- [ ] Check for timeout errors in logs
- [ ] Monitor memory and CPU usage
- [ ] Validate end-to-end latency requirements

### Rollback Criteria

- Circuit breaker opens within 5 minutes of deployment
- Error rate exceeds 5%
- P99 latency exceeds SLA
- Memory usage exceeds container limits

## Quick Reference

### Default Configuration

```python
{
    "max_concurrent_operations": 10,    # Thread pool workers
    "max_queue_size_multiplier": 3,     # Queue = workers * 3
    "timeout_seconds": 30.0,            # Per-operation timeout
    "circuit_breaker_failure_threshold": 5,  # Failures to open
    "circuit_breaker_reset_timeout_seconds": 30.0,  # Reset delay
}
```

### Sizing Formula

```
Recommended Workers = (Expected Concurrent Operations) * 1.5
Queue Size = Workers * 3 (default) to 10 (high burst)
```

### Memory Estimation

```
Thread Memory = Workers * 1MB (approximate stack size)
Queue Memory = Queue Size * Average Request Size
Total = Thread Memory + Queue Memory + Base Memory
```

## Related Documentation

- [Circuit Breaker Thread Safety](../architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md)
- [Validation Performance Notes](../validation/performance_notes.md)
- [HandlerInfisical Source](../../src/omnibase_infra/handlers/handler_infisical.py)
- [Configuration Model](../../src/omnibase_infra/handlers/models/infisical/model_infisical_handler_config.py)
