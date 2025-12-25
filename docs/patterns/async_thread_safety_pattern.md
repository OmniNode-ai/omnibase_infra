# Async Thread Safety Pattern

## Overview

This document describes thread safety patterns for asyncio-based concurrency in ONEX infrastructure components. These patterns apply when multiple coroutines may access shared state within a single event loop.

**Key Distinction**: asyncio uses cooperative concurrency (single-threaded with coroutines), not multi-threading. The patterns here protect against race conditions during `await` points, not CPU-level parallelism.

### When to Use

- Components with mutable shared state accessed by multiple coroutines
- Counter updates that must be atomic
- State transitions that must be consistent
- Operations with pre-check/execute/post-update patterns

### When NOT Needed

- Pure functions with no shared state
- Read-only operations on immutable data
- Single-coroutine-at-a-time usage (caller serializes)
- Dataclasses used as value objects (create new instances instead of mutating)

## Lock Scope Patterns

The critical insight is to **minimize lock scope** - only protect the specific state mutations, not entire operations.

### Pattern: Counter-Only Locking

```python
import asyncio
from uuid import UUID

class OperationTracker:
    """Tracks operations with atomic counter updates.

    Thread Safety:
        Designed for single asyncio event loop usage. The asyncio.Lock
        protects counter updates ONLY, not the entire operation.

    Lock Scope:
        - PROTECTED: Counter increments (execution_count, failed_count)
        - NOT PROTECTED: Pre-checks, operation execution, callbacks
    """

    def __init__(self) -> None:
        self.execution_count = 0
        self.failed_count = 0
        self._lock = asyncio.Lock()

    async def execute_tracked_operation(
        self,
        operation_id: UUID,
        backend_operation: ...,
    ) -> bool:
        """Execute operation with atomic counter tracking.

        Multiple concurrent calls are supported. Counters are eventually
        accurate after all operations complete.
        """
        # --- PHASE 1: Pre-execution (NO LOCK HELD) ---
        # Pre-checks run concurrently, no lock needed
        await self._validate_preconditions(operation_id)

        # --- PHASE 2: Execution (NO LOCK HELD) ---
        # Main operation runs without lock to allow concurrency
        try:
            result = await backend_operation()

            # --- COUNTER UPDATE: Lock acquired briefly ---
            async with self._lock:
                self.execution_count += 1
            # --- Lock released immediately ---

            return result

        except Exception:
            # --- FAILURE COUNTER: Lock acquired briefly ---
            async with self._lock:
                self.failed_count += 1
            # --- Lock released ---
            raise
```

### Pattern: State Transition Locking

```python
@dataclass
class StateManager:
    """Manages state with atomic transitions.

    Thread Safety:
        Lock protects state mutations only. Callbacks execute outside
        lock to avoid deadlocks from callback-initiated lock acquisition.

    Lock Scope:
        - PROTECTED: is_active, state_data modifications
        - NOT PROTECTED: Transition callbacks (may perform I/O)
    """

    is_active: bool = False
    state_data: dict = field(default_factory=dict)
    transition_callbacks: list = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def transition_to_active(self) -> None:
        """Transition to active state.

        Acquires lock for state modification, releases before callbacks.
        """
        # Acquire lock for atomic state transition
        async with self._lock:
            if self.is_active:
                return  # Already active, early exit
            self.is_active = True
            self.state_data["activated_at"] = time.time()

        # Callbacks execute OUTSIDE lock (may perform I/O, acquire other locks)
        for callback in self.transition_callbacks:
            await callback()
```

## Counter Accuracy Pattern

Counters are **eventually accurate** - they correctly reflect totals after all concurrent operations complete, but may show intermediate values during active execution.

### The Problem

```python
# INCORRECT: Asserting during active execution
task = asyncio.create_task(executor.execute(...))
assert executor.execution_count == 0  # Race condition! May be 0 or 1
await task
```

### The Solution

```python
# CORRECT: Wait for all operations before asserting
results = await asyncio.gather(
    executor.execute(uuid4(), "op1"),
    executor.execute(uuid4(), "op2"),
    executor.execute(uuid4(), "op3"),
    return_exceptions=True,
)
# Now counters are stable and accurate
assert executor.execution_count + executor.failed_count == 3
```

### Test Utility

```python
async def gather_with_error_collection(
    coroutines: list,
    *,
    return_exceptions: bool = True,
) -> tuple[list[object], list[Exception]]:
    """Execute coroutines concurrently and separate successes from failures.

    Simplifies the common pattern of running multiple async operations
    and classifying their outcomes.

    Args:
        coroutines: List of coroutines to execute concurrently.
        return_exceptions: If True, exceptions are captured instead of raised.

    Returns:
        Tuple of (successful_results, exceptions).

    Example:
        >>> successes, failures = await gather_with_error_collection(
        ...     [operation(i) for i in range(10)]
        ... )
        >>> # Both lists are complete - safe to check counters now
        >>> assert len(successes) + len(failures) == 10
    """
    results = await asyncio.gather(*coroutines, return_exceptions=return_exceptions)

    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]

    return successes, failures
```

## Callback Execution Outside Locks

**Critical Rule**: Callbacks that may perform I/O or acquire locks MUST execute outside the lock that triggers them.

### Why?

1. **Deadlock Prevention**: If callback acquires the same lock, deadlock occurs
2. **I/O Blocking**: Holding lock during I/O blocks all waiters unnecessarily
3. **Cascading Delays**: Long callbacks delay unrelated state updates

### Pattern: Capture-Then-Execute

```python
async def simulate_recovery_with_callbacks(
    self,
    duration_ms: int = 100,
) -> None:
    """Recover from failure state with callbacks.

    Thread Safety:
        Acquires lock for state transition only. Callbacks execute
        outside lock to allow concurrent I/O operations.
    """
    await asyncio.sleep(duration_ms / 1000.0)

    # Step 1: Acquire lock and update state atomically
    async with self._lock:
        self.is_recovering = False
        self.recovered_at = time.time()
        # Capture callbacks to execute (could also copy list if mutable)
        callbacks_to_run = self.recovery_callbacks

    # Step 2: Execute callbacks OUTSIDE lock
    for callback in callbacks_to_run:
        await callback()  # May perform I/O, acquire other locks safely
```

### Anti-Pattern: Lock Held During Callbacks

```python
# WRONG - Holds lock during potentially slow callbacks
async with self._lock:
    self.is_recovering = False
    for callback in self.recovery_callbacks:
        await callback()  # If this acquires _lock -> DEADLOCK
                          # If this is slow -> all waiters blocked
```

## Test Isolation Recommendations

### Fresh Instances Per Test

Use pytest fixtures that provide fresh instances:

```python
@pytest.fixture
def effect_executor(
    idempotency_store: InMemoryIdempotencyStore,
    failure_injector: FailureInjector,
    backend_client: MagicMock,
) -> EffectExecutor:
    """Create fresh executor per test.

    Returns:
        EffectExecutor with zero counters and clean state.
    """
    return EffectExecutor(
        idempotency_store=idempotency_store,
        failure_injector=failure_injector,
        backend_client=backend_client,
    )
```

### Reset Methods for Shared State

When fresh instances are expensive, provide reset methods:

```python
def reset_counts(self) -> None:
    """Reset execution counters.

    Warning:
        NOT thread-safe. Only call when no concurrent operations
        are in progress. Typically called at test setup/teardown.
    """
    # No lock - caller ensures no concurrent operations
    self.execution_count = 0
    self.failed_count = 0
```

### Gather-Then-Assert Pattern

For concurrent test scenarios:

```python
async def test_concurrent_operations(effect_executor):
    """Test concurrent execution with proper counter verification."""
    # Launch multiple concurrent operations
    results = await asyncio.gather(
        effect_executor.execute(uuid4(), "op1"),
        effect_executor.execute(uuid4(), "op2"),
        effect_executor.execute(uuid4(), "op3"),
        effect_executor.execute(uuid4(), "op4"),
        effect_executor.execute(uuid4(), "op5"),
        return_exceptions=True,
    )

    # NOW counters are stable - safe to assert
    total = effect_executor.execution_count + effect_executor.failed_count
    assert total == 5, f"Expected 5 operations tracked, got {total}"

    # Classify results
    successes = [r for r in results if r is True]
    failures = [r for r in results if isinstance(r, Exception)]

    # Counters should match result classification
    assert effect_executor.execution_count == len(successes)
    assert effect_executor.failed_count == len(failures)
```

## Complete Example: ChaosEffectExecutor

This example from the chaos testing infrastructure demonstrates all patterns:

```python
class ChaosEffectExecutor:
    """Effect executor with chaos injection capability.

    Thread Safety:
        Designed for single asyncio event loop usage. NOT thread-safe
        for multi-threaded access.

        The asyncio.Lock provides atomicity for counter updates ONLY.
        Lock is NOT held during the entire execute operation.

    Lock Scope:
        - PROTECTED: execution_count and failed_count increments
        - NOT PROTECTED: Idempotency checks, chaos injection, backend execution

    Concurrency Considerations:
        Multiple concurrent calls to execute_with_chaos are supported.
        Each call will:
        1. Independently check idempotency (no lock held)
        2. Execute backend operations concurrently (no lock held)
        3. Atomically update counters (lock acquired/released per update)

    Counter Accuracy:
        Counters are eventually accurate - they reflect total successes
        and failures, but assertions during active execution may observe
        intermediate states.

    Test Isolation:
        For deterministic testing:
        1. Sequential execution: Await each operation individually
        2. Gather then assert: Use asyncio.gather() then check counters
        3. Fresh executor per test: Use fixture for clean state
    """

    def __init__(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        failure_injector: FailureInjector,
        backend_client: MagicMock,
    ) -> None:
        self.idempotency_store = idempotency_store
        self.failure_injector = failure_injector
        self.backend_client = backend_client
        self.execution_count = 0
        self.failed_count = 0
        # Lock for atomic counter updates only
        self._lock = asyncio.Lock()

    async def execute_with_chaos(
        self,
        intent_id: UUID,
        operation: str,
        domain: str = "chaos",
        correlation_id: UUID | None = None,
        fail_point: str | None = None,
    ) -> bool:
        """Execute operation with chaos injection.

        Args:
            intent_id: Unique identifier for this intent.
            operation: Name of the operation.
            domain: Idempotency domain.
            correlation_id: Optional correlation ID.
            fail_point: Specific point to inject failure.

        Returns:
            True if operation succeeded.
        """
        # --- PHASE 1: Pre-execution chaos (NO LOCK) ---
        if fail_point == "pre":
            await self.failure_injector.maybe_inject_failure(
                f"{operation}:pre", correlation_id
            )

        # --- PHASE 2: Idempotency check (NO LOCK) ---
        # Store has its own synchronization
        is_new = await self.idempotency_store.check_and_record(
            message_id=intent_id,
            domain=domain,
            correlation_id=correlation_id,
        )

        if not is_new:
            return True  # Duplicate - skip execution

        # --- PHASE 3: Execution (NO LOCK) ---
        try:
            if fail_point == "mid":
                await self.failure_injector.maybe_inject_failure(
                    f"{operation}:mid", correlation_id
                )

            await self.failure_injector.maybe_inject_latency()
            await self.backend_client.execute(operation, intent_id)

            # --- Counter update: Lock briefly ---
            async with self._lock:
                self.execution_count += 1

            if fail_point == "post":
                await self.failure_injector.maybe_inject_failure(
                    f"{operation}:post", correlation_id
                )

            return True

        except Exception:
            # --- Failure counter: Lock briefly ---
            async with self._lock:
                self.failed_count += 1
            raise
```

## Best Practices Summary

### DO

- Minimize lock scope to specific state mutations
- Execute callbacks outside locks
- Use `asyncio.gather()` before asserting on counters
- Provide fresh fixtures per test
- Document thread safety in class docstrings
- Clearly specify what is protected vs not protected

### DON'T

- Hold locks during I/O operations
- Assert on counters during active concurrent execution
- Share mutable instances across tests without reset
- Use same lock for unrelated state (creates contention)
- Call lock-acquiring methods from within callbacks

## Multi-Threading Warning

These patterns are for **asyncio cooperative concurrency only**. If your code uses `concurrent.futures.ThreadPoolExecutor` or multiple threads:

1. Use `threading.Lock` instead of `asyncio.Lock`
2. Consider `threading.RLock` if callbacks may re-enter
3. Use `asyncio.run_coroutine_threadsafe()` for cross-thread async calls
4. See `docs/architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md` for multi-threaded patterns

## Related Patterns

- [Circuit Breaker Implementation](./circuit_breaker_implementation.md) - Circuit breaker with lock patterns
- [Error Recovery Patterns](./error_recovery_patterns.md) - Retry and backoff strategies
- [Correlation ID Tracking](./correlation_id_tracking.md) - Request tracing across async operations
