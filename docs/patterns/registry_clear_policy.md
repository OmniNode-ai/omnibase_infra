> **Navigation**: [Home](../index.md) > [Patterns](README.md) > Registry clear() Policy

# Registry clear() Policy

This document defines the policy for implementing `clear()` methods on registry classes, balancing test isolation requirements with production safety.

## Overview

Registry classes often need a way to reset their state for testing purposes, but exposing this capability in production can lead to accidental data loss or system instability. This document establishes guidelines for when and how to implement clearing functionality.

## When to Implement clear()

Registries SHOULD provide a `clear()` method when:

### 1. Test Isolation Required

When registries maintain state that persists between test cases, clearing enables proper test isolation:

```python
@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry state between tests."""
    yield
    PolicyRegistry.clear()  # Ensure clean slate for next test
```

### 2. Development/Debugging Scenarios

When developers need to reset state during local development without restarting the application:

```python
# Development REPL usage
>>> registry.register("my_policy", MyPolicy)
>>> registry.list_keys()
['my_policy']
>>> registry.clear()  # Reset to try different configuration
>>> registry.list_keys()
[]
```

### 3. Explicit Reset is Part of Contract

When the registry's design explicitly supports runtime reconfiguration:

```python
class DynamicConfigRegistry:
    """Registry that supports hot-reloading configuration.

    This registry explicitly supports clearing as part of its contract
    for configuration hot-reload scenarios.
    """

    def reload_configuration(self, new_config: Config) -> None:
        """Reload configuration, clearing existing registrations."""
        self.clear()
        self._load_from_config(new_config)
```

### 4. In-Memory/Mock Implementations

Test doubles and in-memory implementations typically need clear() for test setup:

```python
class InMemoryEventStore:
    """In-memory event store for testing."""

    async def clear(self) -> None:
        """Clear all events. Test utility method."""
        async with self._lock:
            self._events.clear()
```

## When NOT to Implement clear()

Registries SHOULD NOT provide a public `clear()` method when:

### 1. Accidental Clearing Would Break Production

If clearing the registry would cause immediate system failure:

```python
class CriticalServiceRegistry:
    """Registry for core infrastructure services.

    No clear() method: Accidentally clearing this registry would
    cause all service resolution to fail, breaking the entire system.
    Services are registered once at startup and must remain available.
    """
    pass  # No clear() method
```

### 2. Append-Only Design

When the registry is designed to be immutable after initial population:

```python
class ContractRegistry:
    """Registry for ONEX contracts.

    Contracts are loaded at startup and never modified. The registry
    is designed as append-only with no runtime modification support.
    """
    pass  # No clear() method - immutable after startup
```

### 3. No Legitimate Runtime Use Case

If the only reason to clear would be a bug or mistake:

```python
class HandlerRegistry:
    """Registry for message handlers.

    Handlers are registered during application bootstrap. There is no
    legitimate runtime scenario where clearing would be appropriate.
    Clearing would leave the system unable to process messages.
    """
    pass  # No clear() - would only happen by mistake
```

### 4. External Persistence

When the registry is backed by external storage that shouldn't be cleared:

```python
class PersistentSchemaRegistry:
    """Registry backed by PostgreSQL schema table.

    No clear() method. The underlying table contains production data
    that should never be cleared through the registry interface.
    Use database migrations for schema changes.
    """
    pass  # No clear() - database is source of truth
```

## Implementation Patterns

### Standard Pattern: Public clear() with Warning

For registries that need clearing capability but primarily in tests:

```python
import warnings
import logging
import threading

logger = logging.getLogger(__name__)


class PolicyRegistry:
    """Registry with clear() for testing."""

    def __init__(self) -> None:
        self._registry: dict[str, type] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        """Clear all policy registrations.

        Removes all registered policies from the registry.

        Warning:
            This method is intended for **testing purposes only**.
            Calling it in production code will emit a warning.
            It breaks the immutability guarantee after startup.

        Thread Safety:
            This method is protected by the instance lock to ensure
            thread-safe clearing of the registry.

        Example:
            >>> registry = PolicyRegistry()
            >>> registry.register("retry", RetryPolicy)
            >>> registry.clear()
            >>> registry.list_keys()
            []
        """
        # Emit warning to catch accidental production usage
        warnings.warn(
            "PolicyRegistry.clear() is intended for testing only. "
            "Do not use in production code.",
            UserWarning,
            stacklevel=2,
        )

        # Log for debugging/audit trail
        logger.warning(
            "Registry cleared - this should only happen in tests",
            extra={"registry_type": self.__class__.__name__},
        )

        # Thread-safe clearing
        with self._lock:
            self._registry.clear()
```

### Alternative Pattern: Private Test-Only Method

For stricter separation, use underscore prefix to indicate internal use:

```python
class StrictRegistry:
    """Registry with private test-only clearing."""

    def _clear_for_testing(self) -> None:
        """Clear registry for test isolation. NOT for production use.

        This method uses underscore prefix to indicate it is internal
        and should only be called from test fixtures.

        Warning:
            Calling this method in production code is a bug.
        """
        with self._lock:
            self._registry.clear()
            logger.warning(
                "Registry cleared via _clear_for_testing()",
                extra={"registry_type": self.__class__.__name__},
            )
```

Usage in tests:

```python
@pytest.fixture(autouse=True)
def reset_strict_registry(strict_registry):
    """Reset registry between tests."""
    yield
    # Access private method explicitly for test isolation
    strict_registry._clear_for_testing()
```

### Async Pattern: For Async Registries

For registries with async operations:

```python
import asyncio


class AsyncEventStore:
    """Async event store with clear() for testing."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = asyncio.Lock()

    async def clear(self) -> None:
        """Clear all stored events.

        Test utility method to reset store state between tests.

        Thread Safety:
            Protected by async lock for concurrent access safety.
        """
        async with self._lock:
            self._events.clear()
```

### Class-Level Pattern: For Class Variable Registries

For registries using class-level storage:

```python
class RegistryPayloadHttp:
    """Payload type registry using class variables."""

    _types: ClassVar[dict[str, type]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered types.

        Warning:
            This method is intended for **testing purposes only**.
            Calling it in production code will emit a warning.
            It breaks the immutability guarantee after startup.

        Thread Safety:
            This method is protected by the class-level lock to ensure
            thread-safe clearing of the registry.
        """
        warnings.warn(
            f"{cls.__name__}.clear() is intended for testing only. "
            "Do not use in production code.",
            UserWarning,
            stacklevel=2,
        )
        with cls._lock:
            cls._types.clear()
```

## Thread Safety Requirements

**ALWAYS acquire the lock before clearing:**

```python
# CORRECT: Thread-safe clearing
def clear(self) -> None:
    with self._lock:
        self._registry.clear()

# INCORRECT: Race condition possible
def clear(self) -> None:
    self._registry.clear()  # Other thread might be reading/writing
```

For async code:

```python
# CORRECT: Async-safe clearing
async def clear(self) -> None:
    async with self._lock:
        self._registry.clear()
```

## Logging and Debugging

**ALWAYS log when clear() is called:**

This creates an audit trail for debugging unexpected behavior:

```python
import logging

logger = logging.getLogger(__name__)


def clear(self) -> None:
    """Clear registry with audit logging."""
    # Log at WARNING level - this should be rare
    logger.warning(
        "Registry cleared",
        extra={
            "registry_type": self.__class__.__name__,
            "previous_count": len(self._registry),
        },
    )
    with self._lock:
        self._registry.clear()
```

In tests, you can assert the warning was logged:

```python
def test_clear_logs_warning(caplog):
    """Verify clear() logs for debugging."""
    registry = PolicyRegistry()
    registry.register("test", TestPolicy)

    with caplog.at_level(logging.WARNING):
        registry.clear()

    assert "Registry cleared" in caplog.text
```

## Warning Emission

Use `warnings.warn()` to catch accidental production usage:

```python
import warnings


def clear(self) -> None:
    """Clear with warning for production detection."""
    warnings.warn(
        f"{self.__class__.__name__}.clear() is intended for testing only. "
        "Do not use in production code.",
        UserWarning,
        stacklevel=2,  # Point to caller, not this method
    )
    with self._lock:
        self._registry.clear()
```

In tests, suppress the expected warning:

```python
def test_clear_empties_registry():
    """Test that clear() removes all entries."""
    registry = PolicyRegistry()
    registry.register("test", TestPolicy)

    # Suppress expected warning in tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        registry.clear()

    assert len(registry) == 0
```

Or use `pytest.warns()` to verify the warning:

```python
def test_clear_emits_warning():
    """Verify clear() warns about test-only usage."""
    registry = PolicyRegistry()

    with pytest.warns(UserWarning, match="intended for testing only"):
        registry.clear()
```

## Documentation Requirements

Every `clear()` method MUST include:

1. **Warning section** explaining test-only nature
2. **Thread Safety section** describing synchronization
3. **Example** showing expected usage

```python
def clear(self) -> None:
    """Clear all registrations.

    Removes all entries from the registry.

    Warning:
        This method is intended for **testing purposes only**.
        Calling it in production code will emit a warning.
        It breaks the immutability guarantee after startup.

    Thread Safety:
        This method is protected by the instance lock to ensure
        thread-safe clearing of the registry.

    Example:
        >>> registry = MyRegistry()
        >>> registry.register("key", value)
        >>> registry.clear()
        >>> len(registry)
        0
    """
```

## Checklist for Implementation

When adding `clear()` to a registry:

- [ ] Confirm legitimate test isolation need exists
- [ ] Acquire lock before clearing (thread safety)
- [ ] Add `warnings.warn()` call with descriptive message
- [ ] Add logging at WARNING level
- [ ] Document Warning, Thread Safety, and Example sections
- [ ] Clear all internal data structures (main registry, indexes, caches)
- [ ] Add test verifying warning is emitted
- [ ] Add test verifying thread safety under concurrent access

## Existing Implementations

These registries in omnibase_infra implement `clear()` following this policy:

| Registry | Location | Pattern |
|----------|----------|---------|
| `PolicyRegistry` | `runtime/policy_registry.py` | Public with warning |
| `RegistryProtocolBinding` | `runtime/registry/registry_protocol_binding.py` | Public with warning |
| `RegistryCompute` | `runtime/registry_compute.py` | Public with warning |
| `RegistryPayloadHttp` | `handlers/models/http/registry_payload_http.py` | Class-level with warning |
| `RegistryPayloadConsul` | `handlers/models/consul/registry_payload_consul.py` | Class-level with warning |
| `RegistryPayloadInfisical` | `handlers/models/infisical/` | Removed — Infisical uses `HandlerInfisical` directly |
| `InMemoryEventBus` | `event_bus/inmemory_event_bus.py` | Async clear_event_history() |
| `InMemoryIdempotencyStore` | `idempotency/store_inmemory.py` | Async test utility |

## Related Patterns

- [Container Dependency Injection](./container_dependency_injection.md) - How registries are resolved
- [Policy Registry Trust Model](./policy_registry_trust_model.md) - Security for policy registries
- [Testing Patterns](./testing_patterns.md) - Test fixture patterns using clear()
- [Async Thread Safety Pattern](./async_thread_safety_pattern.md) - Lock patterns for registries

## See Also

- `pytest.warns()` documentation for warning assertions
- `warnings.warn()` with `stacklevel` parameter
- `threading.Lock` and `asyncio.Lock` for synchronization
