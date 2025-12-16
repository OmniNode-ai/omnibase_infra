# Test Isolation and Cleanup Improvements

## Summary

Fixed test isolation issues in the KafkaEventBus threading safety test suite to prevent resource leaks and ensure reliable test execution in parallel environments (pytest-xdist).

## Changes Made

### 1. Fixed Cleanup Order in test_kafka_threading_safety.py

**Problem**: `await bus.close()` was called outside the patch context, causing potential AttributeError during cleanup when mocks were no longer active.

**Solution**: Moved all `await bus.close()` calls inside the patch context to ensure mocks remain active during cleanup.

**Locations Fixed**:
- `test_concurrent_publish_operations_thread_safe` (line 61)
- `test_initialize_start_race_condition_fixed` (line 98)
- `test_producer_access_during_retry_thread_safe` (line 140)
- `test_circuit_breaker_concurrent_access_thread_safe` (line 251)

### 2. Added Comprehensive Documentation

Added module-level docstring explaining test isolation patterns:

```python
"""Threading safety tests for KafkaEventBus race condition fixes.

Test Isolation and Cleanup Patterns
====================================

This test suite demonstrates proper test isolation patterns for async tests:

1. **Cleanup Order**: Always call `await bus.close()` inside the patch context
   to ensure mocks are still active during cleanup. This prevents AttributeError
   during producer.stop() calls.

2. **Exception Handling**: Tests that simulate failures properly catch and
   suppress expected exceptions using try/except blocks with pass statements.

3. **Timing-Sensitive Tests**: Tests using asyncio.sleep() include documentation
   noting that timing may vary under system load. These tests verify thread
   safety, not strict timing behavior.

4. **Resource Management**: Each test creates its own bus instance and ensures
   cleanup within the test method, preventing resource leaks between tests.

5. **Idempotency Testing**: Tests like concurrent_close_operations verify that
   operations can be safely called multiple times without additional cleanup.

6. **Documentation**: Inline comments explain cleanup patterns and expected
   behaviors to aid future maintenance and parallel test execution.

These patterns ensure tests can run reliably in parallel (pytest-xdist) without
interference or resource contention.
"""
```

### 3. Enhanced Test Documentation

Added inline comments to clarify:
- Cleanup order and rationale
- Expected exception behavior (failures are intentional)
- Timing sensitivity disclaimers
- Idempotency testing patterns

### 4. Test Coverage

All 6 threading safety tests pass:
- ✅ test_concurrent_publish_operations_thread_safe
- ✅ test_initialize_start_race_condition_fixed
- ✅ test_producer_access_during_retry_thread_safe
- ✅ test_concurrent_close_operations_thread_safe
- ✅ test_health_check_during_shutdown_thread_safe
- ✅ test_circuit_breaker_concurrent_access_thread_safe

All 130 event bus unit tests pass with no flakiness.

## Best Practices Established

### Async Test Cleanup Pattern

```python
async def test_example(self):
    """Test example with proper cleanup."""
    bus = KafkaEventBus.default()

    with patch("module.AIOKafkaProducer") as MockProducer:
        mock_producer = AsyncMock()
        MockProducer.return_value = mock_producer

        await bus.start()

        # Test operations...

        # Cleanup: Close bus within patch context
        await bus.close()  # ✅ CORRECT - inside patch

    # ❌ WRONG - outside patch, mocks no longer active
    # await bus.close()
```

### Exception Handling in Tests

```python
async def test_with_expected_failures(self):
    """Test that properly handles expected failures."""
    async def failing_operation():
        """Operation that catches and suppresses expected failures."""
        try:
            await bus.publish(...)
        except Exception:
            pass  # Expected - circuit breaker or producer failure

    # Tests verify behavior despite expected exceptions
    await asyncio.gather(*[failing_operation() for _ in range(5)])
```

### Timing-Sensitive Tests

```python
async def test_timing_dependent(self):
    """Test with timing coordination.

    Note: Uses sleep() for timing coordination which may be affected by
    system load. The test verifies thread safety, not strict timing behavior.
    """
    await asyncio.sleep(0.05)  # Timing coordination only
    # Assertions verify thread safety, not timing precision
```

## pytest-xdist Compatibility

These improvements ensure tests can run in parallel using pytest-xdist:

```bash
# Run tests in parallel (4 workers)
poetry run pytest tests/unit/event_bus/ -n 4

# All tests should pass without interference
```

## Related Work

- PR #37 review identified these isolation issues
- KafkaEventBus uses MixinAsyncCircuitBreaker for fault tolerance
- All fixtures in test_kafka_event_bus.py already follow proper cleanup patterns

## Verification

Run the test suite:

```bash
# Single test file
poetry run pytest tests/unit/event_bus/test_kafka_threading_safety.py -v

# All event bus tests
poetry run pytest tests/unit/event_bus/ -v

# Parallel execution
poetry run pytest tests/unit/event_bus/ -n auto
```

All tests should pass reliably without flakiness.
