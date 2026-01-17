# Protocol Patterns

## Overview

Protocols in this codebase use `typing.Protocol` (PEP 544) to enable structural typing with full type checker support. Unlike abstract base classes (ABCs) that require explicit inheritance, protocols use structural subtyping - any class with matching method signatures automatically satisfies the protocol.

This approach enables three primary use cases in ONEX infrastructure:

1. **Mixin Composition** - Type-safe cross-mixin method access without inheritance conflicts (e.g., `ProtocolCircuitBreakerAware`)
2. **Plugin Systems** - Behavioral contracts for deterministic compute operations (e.g., `ProtocolPluginCompute`)
3. **Validation Frameworks** - Self-describing plugin interfaces for extensible validators (e.g., `ProtocolArchitectureRule`)

Protocols are the preferred abstraction boundary for interfaces where the consumer does not need to modify or extend behavior - only call it. When implementation sharing or runtime guarantees are required, use ABCs or concrete base classes instead.

## Protocol vs ABC vs Concrete Base

### Decision Matrix

| Criterion | Protocol | ABC | Concrete Base |
|-----------|----------|-----|---------------|
| Structural typing (duck typing) | Yes | No | No |
| Shared implementation | No | Yes | Yes |
| Multiple inheritance friendly | Yes | Careful | No |
| Runtime `isinstance()` reliable | Method names only | Yes | Yes |
| Template method pattern | No | Yes | Yes |
| Protected helpers (`_method`) | No | Yes | Yes |
| State management | No | Yes | Yes |

### When to Use Protocol

- **Structural typing needed** - Consumer should work with any class having matching signatures
- **Cross-mixin method access** - Type-safe access to methods from cooperating mixins
- **Adapter/DI boundaries** - Loose coupling between components
- **Optional capabilities with guard flags** - Check capability before use (e.g., `_circuit_breaker_initialized`)

### When to Use ABC

- **Shared behavior/implementation needed** - Common logic that subclasses inherit
- **Invariants must be enforced** - Constructor validation, state initialization
- **Protected helper methods required** - Internal API for subclass use
- **Runtime inheritance checks needed** - Reliable `isinstance()` checks

### When to Use Concrete Base

- **Default implementation with override points** - Sensible defaults, optional customization
- **Template method pattern** - Algorithm skeleton with customizable steps
- **Common initialization logic** - Constructor patterns shared by all subclasses

### When NOT to Use Protocol

- **You need state** - Protocols cannot enforce instance attributes at runtime
- **Runtime guarantees required** - `isinstance()` only checks method names, not signatures
- **Private hooks needed** - Protocols cannot define protected/private API contracts

## Exemplar: ProtocolCircuitBreakerAware

**Location**: `src/omnibase_infra/mixins/protocol_circuit_breaker_aware.py`

### Purpose

Enables type-safe access to circuit breaker methods across mixin boundaries without inheritance. When `MixinRetryExecution` needs to interact with `MixinAsyncCircuitBreaker` methods, the protocol provides the contract without requiring a common base class.

### Interface

```python
from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class ProtocolCircuitBreakerAware(Protocol):
    """Protocol for components with circuit breaker capability.

    Concurrency Safety:
        All circuit breaker methods MUST be called while holding
        ``_circuit_breaker_lock``. This is documented in each method's
        docstring using: "REQUIRES: self._circuit_breaker_lock must be
        held by caller."
    """

    _circuit_breaker_lock: asyncio.Lock

    async def _check_circuit_breaker(
        self, operation: str, correlation_id: UUID | None = None
    ) -> None:
        """Check if circuit breaker allows operation.

        REQUIRES: self._circuit_breaker_lock must be held by caller.

        Raises:
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def _record_circuit_failure(
        self, operation: str, correlation_id: UUID | None = None
    ) -> None:
        """Record a failure and potentially open the circuit.

        REQUIRES: self._circuit_breaker_lock must be held by caller.
        """
        ...

    async def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state.

        REQUIRES: self._circuit_breaker_lock must be held by caller.
        """
        ...
```

### Why Protocol Over ABC

1. **Multiple mixins compose without inheritance conflicts** - A handler can inherit from both `MixinAsyncCircuitBreaker` and `MixinRetryExecution` without MRO issues
2. **`MixinAsyncCircuitBreaker` satisfies protocol through structural typing** - No explicit inheritance declaration needed
3. **`MixinRetryExecution` consumes via `cast()` pattern** - Type-safe access without runtime overhead
4. **No runtime overhead, full type safety** - Type checker validates at compile time

### Consumer Pattern

The `MixinRetryExecution` class demonstrates the canonical pattern for consuming a protocol from a cooperating mixin:

```python
from typing import cast
from omnibase_infra.mixins.protocol_circuit_breaker_aware import (
    ProtocolCircuitBreakerAware,
)


class MixinRetryExecution:
    """Mixin providing retry logic with optional circuit breaker integration."""

    # Guard flag - set by MixinAsyncCircuitBreaker when initialized
    _circuit_breaker_initialized: bool = False

    def _as_circuit_breaker(self) -> ProtocolCircuitBreakerAware:
        """Cast self to ProtocolCircuitBreakerAware for type-safe access.

        Returns:
            Self cast as ProtocolCircuitBreakerAware for type checker.

        Note:
            Only call when _circuit_breaker_initialized is True.
        """
        return cast(ProtocolCircuitBreakerAware, self)

    async def _record_circuit_failure_if_enabled(
        self, operation: str, correlation_id: UUID
    ) -> None:
        """Record circuit breaker failure if enabled."""
        if self._circuit_breaker_initialized:
            cb = self._as_circuit_breaker()
            async with cb._circuit_breaker_lock:
                await cb._record_circuit_failure(operation, correlation_id)

    async def _check_circuit_if_enabled(
        self, operation: str, correlation_id: UUID
    ) -> None:
        """Check circuit breaker state if enabled."""
        if self._circuit_breaker_initialized:
            cb = self._as_circuit_breaker()
            async with cb._circuit_breaker_lock:
                await cb._check_circuit_breaker(operation, correlation_id)
```

### Best Practices from This Exemplar

- **Document concurrency requirements** - Each method specifies lock requirements
- **Use `...` (Ellipsis) for method bodies** - Per PEP 544 convention, not `pass` or `raise NotImplementedError`
- **Include version annotations** - `.. versionadded:: 0.4.1` for API stability tracking
- **Private prefix (`_`) for internal interfaces** - Signals this is not public API
- **Guard flag pattern** - Check `_circuit_breaker_initialized` before accessing protocol methods

## Mini-Exemplars

### ProtocolPluginCompute

**Location**: `src/omnibase_infra/protocols/protocol_plugin_compute.py`

**Purpose**: Behavioral contract for deterministic compute operations. Plugins implementing this protocol perform pure data transformations without side effects.

**Surface**: 1 method (`execute`)

```python
from typing import Protocol, runtime_checkable
from omnibase_infra.plugins.models import (
    ModelPluginInputData,
    ModelPluginOutputData,
    ModelPluginContext,
)


@runtime_checkable
class ProtocolPluginCompute(Protocol):
    """Protocol for deterministic compute plugins.

    Implementations must guarantee deterministic behavior where the
    same inputs always produce the same outputs.
    """

    def execute(
        self,
        input_data: ModelPluginInputData,
        context: ModelPluginContext,
    ) -> ModelPluginOutputData:
        """Execute deterministic computation on input data.

        Must not perform I/O operations or access external state.
        """
        ...
```

**Key Lesson**: Protocols enforce behavioral guarantees through documentation - the contract specifies what implementations **CAN'T** do (I/O, external state, randomness) as much as what they must provide. This documentation-as-contract approach works because violations manifest as test failures or runtime errors, not type errors.

### ProtocolArchitectureRule

**Location**: `src/omnibase_infra/nodes/architecture_validator/protocols/protocol_architecture_rule.py`

**Purpose**: Self-describing plugin interface for validation rules. Each rule provides metadata (ID, name, severity) alongside its validation logic.

**Surface**: 4 properties + 1 method

```python
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_infra.enums import EnumValidationSeverity
    from omnibase_infra.nodes.architecture_validator.models import (
        ModelRuleCheckResult,
    )


@runtime_checkable
class ProtocolArchitectureRule(Protocol):
    """Contract for architecture validation rules."""

    @property
    def rule_id(self) -> str:
        """Unique identifier (e.g., 'NO_HANDLER_PUBLISHING')."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name (e.g., 'No Handler Publishing')."""
        ...

    @property
    def description(self) -> str:
        """Detailed description of what this rule checks."""
        ...

    @property
    def severity(self) -> EnumValidationSeverity:
        """Severity level for violations (ERROR, WARNING, INFO)."""
        ...

    def check(self, target: object) -> ModelRuleCheckResult:
        """Check the target against this rule."""
        ...
```

**Key Lesson**: Properties in protocols enable metadata-driven systems. The validator can introspect rules to build reports, filter by severity, or display human-readable names - all without coupling to specific rule implementations.

## Common Mistakes

### Using `isinstance()` for Signature Checking

`isinstance()` with `@runtime_checkable` protocols only verifies method **names** exist, not signatures:

```python
# WRONG - This passes even if signatures don't match!
class BadImplementation:
    def execute(self) -> None:  # Missing required parameters
        pass

# isinstance check passes (method name exists)
assert isinstance(BadImplementation(), ProtocolPluginCompute)  # True!

# But actual call fails
plugin.execute(input_data, context)  # TypeError at runtime
```

**Fix**: Use duck typing verification for critical interfaces:

```python
# Verify method exists and is callable
if hasattr(obj, "execute") and callable(obj.execute):
    # Type checker handles signature validation
    result = obj.execute(input_data, context)
```

### Forgetting `@runtime_checkable`

Without `@runtime_checkable`, `isinstance()` raises `TypeError`:

```python
# WRONG - Missing decorator
class MyProtocol(Protocol):
    def method(self) -> None: ...

isinstance(obj, MyProtocol)  # TypeError: Protocols without @runtime_checkable cannot be used with isinstance()
```

**Fix**: Add `@runtime_checkable` if you need runtime type checks.

### Putting Implementation in Protocol

Protocols define interfaces, not behavior:

```python
# WRONG - Protocol with implementation
class BadProtocol(Protocol):
    def process(self, data: str) -> str:
        return data.upper()  # Implementation in protocol!
```

**Fix**: Use ABC if you need shared implementation:

```python
from abc import ABC, abstractmethod

class ProcessorBase(ABC):
    @abstractmethod
    def process(self, data: str) -> str:
        """Subclasses must implement."""
        ...

    def preprocess(self, data: str) -> str:
        """Shared preprocessing logic."""
        return data.strip()
```

### Over-Engineering with Protocols

Not every interface needs a protocol:

```python
# WRONG - Protocol for simple direct dependency
class ProtocolLogger(Protocol):
    def log(self, message: str) -> None: ...

# When there's only one implementation and no testing benefit
class MyService:
    def __init__(self, logger: ProtocolLogger): ...
```

**Fix**: Use protocols when you need structural typing or multiple implementations. For simple DI, direct class references often suffice.

## Review Checklist

Use this checklist when reviewing PRs that add or modify protocols:

```markdown
### Protocol Review Checklist
- [ ] Protocol uses `typing.Protocol` base
- [ ] `@runtime_checkable` added if `isinstance()` checks needed
- [ ] Method bodies use `...` (Ellipsis), not `pass` or `raise NotImplementedError`
- [ ] All method signatures have complete type hints
- [ ] Docstrings document behavioral contracts and concurrency requirements
- [ ] Protocol lives in appropriate location (`protocols/` or with coupled implementation)
- [ ] At least one implementation exists and satisfies the protocol
- [ ] No implementation code in protocol (use ABC if shared behavior needed)
- [ ] Version annotation included (`.. versionadded::`)
```

## Related Patterns

- **[Circuit Breaker Implementation](./circuit_breaker_implementation.md)** - Uses `ProtocolCircuitBreakerAware` for mixin composition
- **[Error Handling Patterns](./error_handling_patterns.md)** - Error types raised by protocol implementations
- **[Container Dependency Injection](./container_dependency_injection.md)** - Protocols as DI boundaries for loose coupling

## See Also

### Source Files

- `src/omnibase_infra/mixins/protocol_circuit_breaker_aware.py` - Cross-mixin protocol exemplar
- `src/omnibase_infra/protocols/protocol_plugin_compute.py` - Behavioral contract exemplar
- `src/omnibase_infra/nodes/architecture_validator/protocols/protocol_architecture_rule.py` - Metadata-driven protocol exemplar
- `src/omnibase_infra/mixins/mixin_retry_execution.py` - Protocol consumer pattern (`_as_circuit_breaker`)

### CLAUDE.md Sections

- **File & Class Naming** - `protocol_<name>.py` naming convention
- **Protocol File Naming** - Single protocol vs domain-grouped `protocols.py`
- **Strong Typing & Models** - No `Any` types in protocol signatures

### External References

- [PEP 544 - Protocols: Structural subtyping](https://peps.python.org/pep-0544/)
- [typing.Protocol documentation](https://docs.python.org/3/library/typing.html#typing.Protocol)
