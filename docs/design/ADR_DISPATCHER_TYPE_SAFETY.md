# ADR: Dispatcher Type Safety

## Status

Accepted

## Context

The `MessageDispatchEngine` in `omnibase_infra` currently uses runtime introspection via `inspect.signature()` to determine whether a registered dispatcher callable accepts a context parameter. This occurs in the `_dispatcher_accepts_context()` method:

```python
def _dispatcher_accepts_context(self, dispatcher: DispatcherFunc) -> bool:
    try:
        sig = inspect.signature(dispatcher)
        params = list(sig.parameters.values())
        # Dispatcher with context has 2 parameters: (envelope, context)
        # Dispatcher without context has 1 parameter: (envelope)
        return len(params) >= 2
    except (ValueError, TypeError):
        # If we can't inspect the signature, assume no context
        return False
```

### Current Type Definitions

The engine defines two dispatcher type aliases:

```python
# Basic dispatcher (envelope only)
DispatcherFunc = Callable[
    [ModelEventEnvelope[object]], DispatcherOutput | Awaitable[DispatcherOutput]
]

# Context-aware dispatcher (envelope + context)
ContextAwareDispatcherFunc = Callable[
    [ModelEventEnvelope[object], ModelDispatchContext],
    DispatcherOutput | Awaitable[DispatcherOutput],
]
```

### Problem Statement

1. **Runtime inspection overhead**: `inspect.signature()` is called during dispatch for every dispatcher invocation that has `node_kind` set
2. **Type erasure at registration**: Both dispatcher types are stored as `DispatcherFunc`, losing the type information
3. **No static type safety**: Static type checkers cannot verify that a dispatcher's signature matches how it will be called
4. **Introspection fallback**: When signature inspection fails, the engine silently assumes no context is needed, which could lead to runtime errors

### PR Review Feedback

A PR review suggested exploring Python `Protocol` from `typing` to define dispatcher signatures, which could reduce reliance on runtime inspection.

## Decision Drivers

- **Type safety at registration time**: Catch signature mismatches early
- **Runtime performance**: Minimize overhead during message dispatch
- **Code simplicity**: Solution should not add significant complexity
- **Static analysis**: Enable mypy/pyright to catch type errors

## Options Considered

### Option 1: Protocol + runtime_checkable

Define two protocols for the dispatcher signatures and use `@runtime_checkable` for `isinstance()` checks.

```python
from typing import Protocol, runtime_checkable
from collections.abc import Awaitable

@runtime_checkable
class ProtocolBasicDispatcher(Protocol):
    """Protocol for dispatchers that only accept an envelope."""

    def __call__(
        self, envelope: ModelEventEnvelope[object]
    ) -> DispatcherOutput | Awaitable[DispatcherOutput]: ...

@runtime_checkable
class ProtocolContextAwareDispatcher(Protocol):
    """Protocol for dispatchers that accept envelope and context."""

    def __call__(
        self, envelope: ModelEventEnvelope[object], context: ModelDispatchContext
    ) -> DispatcherOutput | Awaitable[DispatcherOutput]: ...
```

**Usage at Registration**:
```python
def register_dispatcher(
    self,
    dispatcher_id: str,
    dispatcher: ProtocolBasicDispatcher | ProtocolContextAwareDispatcher,
    ...
) -> None:
    # Check type at registration
    accepts_context = isinstance(dispatcher, ProtocolContextAwareDispatcher)
    # Store the boolean alongside the dispatcher
```

**Pros**:
- Enables `isinstance()` checks at registration time
- Protocols are the Pythonic way to define structural interfaces
- Static type checkers can validate dispatcher implementations
- Self-documenting: Protocol names describe the expected interface

**Cons**:
- **`runtime_checkable` for callables has significant limitations**:
  - Only checks for `__call__` method existence, NOT signature
  - `isinstance(func, ProtocolContextAwareDispatcher)` returns `True` for ANY callable
  - Cannot distinguish between 1-param and 2-param functions at runtime
- Still requires `inspect.signature()` or similar for actual signature validation
- Two protocol definitions add conceptual overhead
- Does not actually solve the runtime inspection problem for callables

**Critical Issue**: Python's `runtime_checkable` protocols only verify that structural members (methods, attributes) exist. For a `Protocol` with just `__call__`, any callable will match because all callables have `__call__`. The protocol cannot verify the _signature_ of `__call__` at runtime.

**Verdict**: **Not viable** as a standalone solution for callable type discrimination.

### Option 2: Separate Registration Methods

Provide two distinct registration methods that encode the dispatcher type in the API:

```python
def register_dispatcher(
    self,
    dispatcher_id: str,
    dispatcher: DispatcherFunc,  # (envelope) -> output
    category: EnumMessageCategory,
    message_types: set[str] | None = None,
) -> None:
    """Register a basic dispatcher (no context injection)."""
    entry = DispatchEntryInternal(
        ...,
        accepts_context=False,  # Known at registration
    )
    ...

def register_context_aware_dispatcher(
    self,
    dispatcher_id: str,
    dispatcher: ContextAwareDispatcherFunc,  # (envelope, context) -> output
    category: EnumMessageCategory,
    message_types: set[str] | None = None,
    node_kind: EnumNodeKind = ...,  # Required for context injection
) -> None:
    """Register a context-aware dispatcher (receives ModelDispatchContext)."""
    entry = DispatchEntryInternal(
        ...,
        accepts_context=True,  # Known at registration
    )
    ...
```

**Pros**:
- **Type safety at registration**: Caller explicitly chooses the correct method
- **No runtime introspection needed**: The method choice encodes the type
- **Static type checkers work perfectly**: Each method has a precise signature
- **Clear API semantics**: Method names document the expected dispatcher type
- **Zero dispatch-time overhead**: The `accepts_context` flag is pre-computed

**Cons**:
- **API surface doubles**: Two methods instead of one
- **User burden**: Caller must choose the correct method
- **Potential for misuse**: User could call wrong method (though static type checker would catch it)

**Migration Path**:
1. Add `register_context_aware_dispatcher()` as the new recommended API
2. Deprecate `register_dispatcher(..., node_kind=...)` pattern
3. Update documentation to guide users to the appropriate method
4. Eventually remove introspection fallback

**Verdict**: **Viable** and provides the strongest type safety guarantees.

### Option 3: Registration-Time Introspection Caching (Current PR Approach)

Continue using `inspect.signature()` but cache the result at registration time instead of dispatch time:

```python
class DispatchEntryInternal:
    __slots__ = (
        "category",
        "dispatcher",
        "dispatcher_id",
        "message_types",
        "node_kind",
        "accepts_context",  # NEW: Cached at registration
    )

    def __init__(
        self,
        dispatcher_id: str,
        dispatcher: DispatcherFunc,
        category: EnumMessageCategory,
        message_types: set[str] | None,
        node_kind: EnumNodeKind | None = None,
    ) -> None:
        self.dispatcher_id = dispatcher_id
        self.dispatcher = dispatcher
        self.category = category
        self.message_types = message_types
        self.node_kind = node_kind
        # Cache signature check at registration time
        self.accepts_context = self._check_accepts_context(dispatcher)

    @staticmethod
    def _check_accepts_context(dispatcher: DispatcherFunc) -> bool:
        try:
            sig = inspect.signature(dispatcher)
            params = list(sig.parameters.values())
            return len(params) >= 2
        except (ValueError, TypeError):
            return False
```

**Changes to dispatch**:
```python
async def _execute_dispatcher(
    self,
    entry: DispatchEntryInternal,
    envelope: ModelEventEnvelope[object],
) -> DispatcherOutput:
    # Use cached value instead of re-inspecting
    if entry.node_kind is not None and entry.accepts_context:
        context = self._create_context_for_entry(entry, envelope)
        if inspect.iscoroutinefunction(entry.dispatcher):
            return await entry.dispatcher(envelope, context)
        ...
```

**Pros**:
- **Minimal API change**: Existing registration API preserved
- **Backwards compatible**: No changes for existing callers
- **Dispatch-time performance**: No `inspect.signature()` call during dispatch
- **Simple implementation**: Just moves the check from dispatch to registration

**Cons**:
- **Still relies on introspection**: `inspect.signature()` is still used
- **No compile-time type safety**: Static type checkers cannot verify signatures
- **Introspection can fail**: Some callables (C extensions, some wrappers) may not be inspectable
- **Deferred error detection**: Signature mismatch only caught when dispatcher is invoked

**Verdict**: **Good incremental improvement** with minimal risk.

### Option 4: TypeVar + Overload Pattern

Use `@overload` to provide precise type signatures for different registration patterns:

```python
from typing import TypeVar, overload

T = TypeVar("T", DispatcherFunc, ContextAwareDispatcherFunc)

@overload
def register_dispatcher(
    self,
    dispatcher_id: str,
    dispatcher: DispatcherFunc,
    category: EnumMessageCategory,
    message_types: set[str] | None = None,
    node_kind: None = None,  # No node_kind -> basic dispatcher
) -> None: ...

@overload
def register_dispatcher(
    self,
    dispatcher_id: str,
    dispatcher: ContextAwareDispatcherFunc,
    category: EnumMessageCategory,
    message_types: set[str] | None = None,
    node_kind: EnumNodeKind = ...,  # With node_kind -> context-aware
) -> None: ...

def register_dispatcher(
    self,
    dispatcher_id: str,
    dispatcher: DispatcherFunc | ContextAwareDispatcherFunc,
    category: EnumMessageCategory,
    message_types: set[str] | None = None,
    node_kind: EnumNodeKind | None = None,
) -> None:
    # Implementation still needs runtime check
    ...
```

**Pros**:
- **Static type safety**: Type checkers can validate based on `node_kind` presence
- **Single API**: Maintains unified registration method
- **Self-documenting**: Overloads show the valid combinations

**Cons**:
- **Runtime still needs introspection**: Overloads are purely for static analysis
- **Complex type annotations**: Multiple overloads can be confusing
- **Doesn't solve the runtime problem**: Implementation still checks at runtime

**Verdict**: **Complementary enhancement** but does not eliminate runtime introspection.

### Option 5: Descriptor-Based Registration Object

Use a registration descriptor that carries type information:

```python
@dataclass
class DispatcherRegistration:
    """Registration object that carries dispatcher type information."""
    dispatcher_id: str
    category: EnumMessageCategory
    message_types: set[str] = field(default_factory=set)

class BasicDispatcherRegistration(DispatcherRegistration):
    """Registration for basic dispatchers."""
    dispatcher: DispatcherFunc

class ContextAwareDispatcherRegistration(DispatcherRegistration):
    """Registration for context-aware dispatchers."""
    dispatcher: ContextAwareDispatcherFunc
    node_kind: EnumNodeKind

# Registration
engine.register(BasicDispatcherRegistration(
    dispatcher_id="my-dispatcher",
    dispatcher=process_event,
    category=EnumMessageCategory.EVENT,
))
```

**Pros**:
- **Type encoded in class**: No introspection needed
- **Extensible**: Can add more registration types
- **Validation at construction**: Pydantic can validate fields

**Cons**:
- **Significant API change**: Complete redesign of registration
- **Verbose**: More code to register a dispatcher
- **Breaking change**: Existing code must be rewritten

**Verdict**: **Over-engineered** for the problem at hand.

## Analysis: Protocol Limitations for Callables

Python's `typing.Protocol` with `@runtime_checkable` has fundamental limitations when applied to callables:

### What `runtime_checkable` Actually Checks

```python
@runtime_checkable
class MyCallableProtocol(Protocol):
    def __call__(self, x: int) -> str: ...

def func_one_arg(x: int) -> str:
    return str(x)

def func_two_args(x: int, y: int) -> str:
    return str(x + y)

# BOTH return True! Protocol only checks __call__ existence
isinstance(func_one_arg, MyCallableProtocol)   # True
isinstance(func_two_args, MyCallableProtocol)  # True
isinstance(lambda: None, MyCallableProtocol)   # True
isinstance(42, MyCallableProtocol)             # False (no __call__)
```

### Why This Happens

From [PEP 544](https://peps.python.org/pep-0544/):

> For protocol classes, `isinstance()` checks are limited to attributes and methods defined in the protocol class body. The checks do not verify method signatures or return types.

This is a fundamental limitation of Python's runtime type system. Signature validation would require:
1. Inspecting `__code__` or `inspect.signature()`
2. Comparing parameter counts, types, and annotations
3. Handling edge cases (varargs, kwargs, defaults)

### Implications for Our Problem

`@runtime_checkable` **cannot** distinguish between:
- `(envelope) -> output` (basic dispatcher)
- `(envelope, context) -> output` (context-aware dispatcher)

Both are callables with a `__call__` method, so both pass the protocol check. This means **Option 1 cannot work as described**.

## Decision

**Recommended Approach: Option 3 (Registration-Time Caching) with Option 2 (Separate Methods) as Future Enhancement**

### Phase 1: Immediate (Low Risk)

Implement **Option 3** - cache the signature check at registration time:

1. Add `accepts_context: bool` field to `DispatchEntryInternal`
2. Move `_dispatcher_accepts_context()` logic to registration time
3. Store the cached value in the entry
4. Use the cached value during dispatch instead of re-inspecting

This provides:
- **Immediate performance improvement**: No `inspect.signature()` during dispatch
- **No API changes**: Fully backwards compatible
- **Minimal implementation risk**: Simple refactoring

### Phase 2: Future Enhancement (Optional)

Consider **Option 2** - separate registration methods:

1. Add `register_context_aware_dispatcher()` method
2. Document it as the preferred API when using `node_kind`
3. Remove `register_dispatcher()` introspection fallback (per no-backwards-compatibility policy)
4. Use `@overload` (Option 4) to improve static type checking on existing API

This provides:
- **Type safety**: Static verification for all dispatchers
- **Clear documentation**: API names describe expected usage

## Implementation Notes

### Phase 1 Changes

**File**: `src/omnibase_infra/runtime/message_dispatch_engine.py`

1. **Update `DispatchEntryInternal`**:
```python
class DispatchEntryInternal:
    __slots__ = (
        "accepts_context",  # NEW
        "category",
        "dispatcher",
        "dispatcher_id",
        "message_types",
        "node_kind",
    )

    def __init__(
        self,
        dispatcher_id: str,
        dispatcher: DispatcherFunc,
        category: EnumMessageCategory,
        message_types: set[str] | None,
        node_kind: EnumNodeKind | None = None,
    ) -> None:
        self.dispatcher_id = dispatcher_id
        self.dispatcher = dispatcher
        self.category = category
        self.message_types = message_types
        self.node_kind = node_kind
        self.accepts_context = self._compute_accepts_context(dispatcher)

    @staticmethod
    def _compute_accepts_context(dispatcher: DispatcherFunc) -> bool:
        """Compute whether dispatcher accepts context parameter."""
        try:
            sig = inspect.signature(dispatcher)
            params = list(sig.parameters.values())
            return len(params) >= 2
        except (ValueError, TypeError):
            return False
```

2. **Update `_execute_dispatcher`**:
```python
async def _execute_dispatcher(
    self,
    entry: DispatchEntryInternal,
    envelope: ModelEventEnvelope[object],
) -> DispatcherOutput:
    dispatcher = entry.dispatcher
    context: ModelDispatchContext | None = None

    # Use cached value instead of _dispatcher_accepts_context()
    if entry.node_kind is not None and entry.accepts_context:
        context = self._create_context_for_entry(entry, envelope)

    # ... rest of execution logic
```

3. **Deprecate `_dispatcher_accepts_context`**:
```python
def _dispatcher_accepts_context(self, dispatcher: DispatcherFunc) -> bool:
    """
    .. removed:: 0.6.0
        This method has been removed. Signature checking is now performed
        at registration time and cached in DispatchEntryInternal.
        Per CLAUDE.md policy, no backwards compatibility is maintained.
    """
    raise NotImplementedError(
        "_check_dispatcher_accepts_context has been removed. "
        "Signature is now cached at registration time."
    )
```

### Test Coverage

Add tests for:
- Registration with context-aware dispatcher (2+ params) correctly sets `accepts_context=True`
- Registration with basic dispatcher (1 param) correctly sets `accepts_context=False`
- Dispatch correctly uses cached value instead of re-inspecting
- Edge cases: lambdas, partial functions, methods, classmethods

## Consequences

### Positive

1. **Performance**: Eliminates `inspect.signature()` calls during dispatch hot path
2. **Backwards Compatible**: No API changes required
3. **Deterministic**: `accepts_context` is computed once and cached
4. **Simple**: Minimal code changes with low risk

### Negative

1. **Still uses introspection**: `inspect.signature()` still called at registration
2. **No static type safety**: Type checkers cannot verify dispatcher signatures
3. **Deferred discovery**: Signature issues only found at registration time, not compile time

### Neutral

1. **Introspection at registration is acceptable**: Registration happens once during startup
2. **Phase 2 optional**: Can add separate methods later if static safety is needed

## References

- [PEP 544 - Protocols: Structural subtyping](https://peps.python.org/pep-0544/)
- [Python `typing.Protocol` documentation](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [Python `inspect.signature()` documentation](https://docs.python.org/3/library/inspect.html#inspect.signature)
- PR #61 Review Comment (internal reference - original document no longer available)
- `src/omnibase_infra/runtime/message_dispatch_engine.py` - Current implementation
- `src/omnibase_infra/runtime/dispatcher_registry.py` - `ProtocolMessageDispatcher` example

## Appendix: Protocol vs ABC for Dispatchers

The codebase already uses `ProtocolMessageDispatcher` in `dispatcher_registry.py` with `@runtime_checkable`. This works well because it checks for **property and method existence**, not callable signatures:

```python
@runtime_checkable
class ProtocolMessageDispatcher(Protocol):
    @property
    def dispatcher_id(self) -> str: ...

    @property
    def category(self) -> EnumMessageCategory: ...

    async def handle(self, envelope: ModelEventEnvelope[object]) -> ModelDispatchResult: ...
```

This pattern works because:
- Properties (`dispatcher_id`, `category`) have their **existence** checked
- Method `handle` has its **existence** checked
- The class structure provides enough type information

For function-based dispatchers in `MessageDispatchEngine`, we only have a callable with no additional structure, making `runtime_checkable` insufficient.

## Changelog

| Date | Author | Description |
|------|--------|-------------|
| 2025-12-21 | Claude | Initial ADR created |
| 2025-12-21 | Claude | Status changed to Accepted. Phase 1 implementation complete in PR: `DispatchEntryInternal.accepts_context` caching at registration time, `_dispatcher_accepts_context()` called once at registration, `_execute_dispatcher()` uses cached value without runtime inspection |
