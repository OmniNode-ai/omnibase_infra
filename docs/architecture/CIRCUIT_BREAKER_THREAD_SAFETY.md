# Circuit Breaker Thread Safety Documentation

## Overview

This document describes the thread safety guarantees and lock ordering policy for the circuit breaker implementation in omnibase_infra.

## Thread Safety Model

### Lock-Based Protection

The circuit breaker uses `asyncio.Lock` (coroutine-safe, not thread-safe) to protect shared state:

- `_circuit_breaker_open`: Circuit open/closed state
- `_circuit_breaker_failures`: Failure counter
- `_circuit_breaker_open_until`: Auto-reset timestamp

**All circuit breaker methods require the caller to hold `_circuit_breaker_lock` before invocation.**

### Caller Responsibility

The mixin delegates lock acquisition to the caller to allow flexible integration patterns:

```python
# Correct usage - lock held by caller
async with self._circuit_breaker_lock:
    await self._check_circuit_breaker("operation", correlation_id)

# INCORRECT - race condition!
await self._check_circuit_breaker("operation", correlation_id)
```

### Debug Assertions

The mixin now includes debug assertions to detect lock protocol violations:

```python
if not self._circuit_breaker_lock.locked():
    logger.error("Circuit breaker lock not held during state check")
    # Continues execution but logs violation
```

These assertions help identify incorrect usage during development and testing.

## Lock Ordering Policy (CRITICAL)

### The Deadlock Problem

When using multiple locks, inconsistent lock ordering causes deadlocks:

```
Thread A: acquires lock1 → tries to acquire lock2 (blocks)
Thread B: acquires lock2 → tries to acquire lock1 (blocks)
Result: DEADLOCK
```

### Mandatory Lock Ordering

To prevent deadlocks, **ALWAYS** acquire locks in this order:

1. **`self._lock`** - Main resource lock (if present)
2. **`self._circuit_breaker_lock`** - Circuit breaker state
3. **Other component locks** - Component-specific locks (e.g., `_producer_lock`, `_consumer_lock`)

### Example: Correct Lock Ordering

```python
class KafkaEventBus(MixinAsyncCircuitBreaker):
    async def start(self):
        async with self._lock:  # 1. Main lock first
            if self._started:
                return

            # 2. Circuit breaker lock second
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker("start", correlation_id)

            # 3. Perform operation
            self._producer = await create_producer()
            self._started = True
```

### Example: INCORRECT Lock Ordering (Deadlock Risk)

```python
# ❌ WRONG - reversed lock order causes deadlocks!
async with self._circuit_breaker_lock:
    await self._check_circuit_breaker("start")
    async with self._lock:  # Deadlock risk!
        self._producer = await create_producer()
```

### Why This Matters

Consider two concurrent operations:

**Operation A (start):**
```python
async with self._lock:
    async with self._circuit_breaker_lock:
        # ...
```

**Operation B (publish):**
```python
async with self._circuit_breaker_lock:  # ❌ Acquires in wrong order!
    await self._check_circuit_breaker()
    async with self._lock:  # Deadlock!
        # ...
```

**Result:** Operation A holds `_lock` and waits for `_circuit_breaker_lock`. Operation B holds `_circuit_breaker_lock` and waits for `_lock`. **DEADLOCK.**

## Implementation in KafkaEventBus

### Correct Lock Usage Examples

#### Start Method (Multiple Locks)

```python
async def start(self):
    correlation_id = uuid4()

    async with self._lock:  # 1. Main lock
        if self._started:
            return

        async with self._circuit_breaker_lock:  # 2. Circuit breaker lock
            await self._check_circuit_breaker("start", correlation_id)

        try:
            self._producer = AIOKafkaProducer(...)
            await self._producer.start()
            self._started = True

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()
        except Exception:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("start", correlation_id)
            raise
```

#### Publish Method (Circuit Breaker Only)

```python
async def publish(self, topic, key, value, headers):
    # No main lock needed for circuit breaker check
    async with self._circuit_breaker_lock:
        await self._check_circuit_breaker("publish", headers.correlation_id)

    # Publish with retry
    await self._publish_with_retry(topic, key, value, headers)
```

## State Transition Safety

### Atomic State Transitions

All state transitions are protected by the circuit breaker lock:

1. **CLOSED → OPEN** (on failure threshold)
   ```python
   async with self._circuit_breaker_lock:
       self._circuit_breaker_failures += 1
       if self._circuit_breaker_failures >= threshold:
           self._circuit_breaker_open = True
           self._circuit_breaker_open_until = time.time() + reset_timeout
   ```

2. **OPEN → HALF_OPEN** (on timeout)
   ```python
   async with self._circuit_breaker_lock:
       if current_time >= self._circuit_breaker_open_until:
           self._circuit_breaker_open = False
           self._circuit_breaker_failures = 0
   ```

3. **HALF_OPEN → CLOSED** (on success)
   ```python
   async with self._circuit_breaker_lock:
       self._circuit_breaker_open = False
       self._circuit_breaker_failures = 0
       self._circuit_breaker_open_until = 0.0
   ```

### Race Condition Prevention

Without locks, these race conditions could occur:

#### Race 1: Failure Counter

```python
# Thread A reads: failures = 4
# Thread B reads: failures = 4
# Thread A writes: failures = 5 (opens circuit)
# Thread B writes: failures = 5 (opens circuit again, loses 1 failure)
```

**Solution:** Lock protects read-modify-write sequence.

#### Race 2: State Transition

```python
# Thread A: checks open=True, timeout elapsed, sets open=False
# Thread B: checks open=False (wrong!), allows operation
# Thread A: operation fails, sets open=True
```

**Solution:** Lock ensures atomic state transition.

## Testing Thread Safety

### Unit Tests

The mixin includes comprehensive thread safety tests:

```python
async def test_concurrent_failure_recording(self):
    """Test concurrent failure recording doesn't lose counts."""
    tasks = [
        self.service._record_circuit_failure("op")
        for _ in range(10)
    ]
    await asyncio.gather(*tasks)

    # All 10 failures should be recorded
    assert self.service._circuit_breaker_failures == 10
```

### Integration Tests

KafkaEventBus integration tests verify concurrent operations:

```python
async def test_concurrent_publish_with_circuit_breaker(self):
    """Test concurrent publishes don't corrupt circuit breaker state."""
    tasks = [
        event_bus.publish(topic, None, b"data")
        for _ in range(100)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Circuit breaker state should be consistent
    assert circuit_breaker_state in ["open", "closed", "half_open"]
```

## Verification Checklist

When implementing circuit breaker integration:

- [ ] Circuit breaker initialized with `_init_circuit_breaker()`
- [ ] All circuit breaker calls wrapped with `async with self._circuit_breaker_lock:`
- [ ] Lock ordering follows policy: `_lock` → `_circuit_breaker_lock` → component locks
- [ ] Lock ordering verified (no reversed order in any code path)
- [ ] Thread safety tests added for concurrent operations
- [ ] Debug log assertions enabled during testing

## References

- **Implementation:** `src/omnibase_infra/mixins/mixin_async_circuit_breaker.py`
- **Usage Example:** `src/omnibase_infra/event_bus/kafka_event_bus.py`
- **Unit Tests:** `tests/unit/mixins/test_mixin_async_circuit_breaker.py`
- **Integration Tests:** `tests/unit/event_bus/test_kafka_event_bus.py`
- **CLAUDE.md:** Infrastructure Circuit Breaker Pattern section

## Summary

**Key Principles:**

1. **Lock Required:** All circuit breaker methods require caller to hold lock
2. **Lock Ordering:** Always acquire locks in documented order to prevent deadlocks
3. **Debug Assertions:** Lock violations logged for debugging
4. **Atomic Transitions:** All state changes protected by lock
5. **Comprehensive Testing:** Thread safety verified through unit and integration tests

**Bottom Line:** The circuit breaker provides thread-safe state management when used correctly. Follow the lock ordering policy and hold the lock when calling circuit breaker methods.
