# Handler to Dispatcher Migration Guide

**Version**: 0.4.0
**Status**: Active
**Ticket**: OMN-934

## Overview

This guide documents the terminology migration from "Handler" to "Dispatcher" in the ONEX message routing infrastructure. The rename clarifies the architectural distinction between:

- **Protocol Handlers** (`handlers/`): Infrastructure adapters for external protocols (HTTP, Database, Kafka, Vault, Consul)
- **Dispatchers** (`dispatch/`): Message routing units in the dispatch engine that process event envelopes

## Scope of Changes

### Terminology Mapping

| Old Term | New Term | Context |
|----------|----------|---------|
| `handler` | `dispatcher` | Message routing and event processing |
| `handler_id` | `dispatcher_id` | Unique identifier for routing unit |
| `handler_count` | `dispatcher_count` | Count of registered routing units |
| `no_handler` | `no_dispatcher` | Status when no routing unit found |
| `handler_error` | `dispatcher_error` | Execution error in routing unit |
| `HANDLER_ERROR` | `HANDLER_ERROR` | **Unchanged** - EnumDispatchStatus value |

### What Changes

#### 1. MessageDispatchEngine API

The `MessageDispatchEngine` now uses "dispatcher" terminology:

```python
# OLD (deprecated, still works via legacy aliases)
engine.register_handler(
    handler_id="user-handler",
    handler=process_user_event,
    category=EnumMessageCategory.EVENT,
)
metrics = engine.get_handler_metrics("user-handler")

# NEW (preferred)
engine.register_dispatcher(
    dispatcher_id="user-dispatcher",
    dispatcher=process_user_event,
    category=EnumMessageCategory.EVENT,
)
metrics = engine.get_dispatcher_metrics("user-dispatcher")
```

#### 2. Model Classes

New model names in `omnibase_infra.models.dispatch`:

| Old (if existed) | New |
|------------------|-----|
| N/A | `ModelDispatcherRegistration` |
| N/A | `ModelDispatcherMetrics` |
| N/A | `ModelDispatchRoute` |
| N/A | `ModelDispatchResult` |
| N/A | `ModelDispatchMetrics` |

#### 3. Metrics Fields

```python
# OLD metrics field names
{
    "no_handler_count": 5,
    "handler_execution_count": 100,
    "handler_error_count": 2,
}

# NEW metrics field names
{
    "no_handler_count": 5,  # Note: unchanged in EnumDispatchStatus
    "dispatcher_execution_count": 100,
    "dispatcher_error_count": 2,
}
```

#### 4. DispatcherRegistry and ProtocolMessageDispatcher (New Components)

Version 0.4.0 introduces new infrastructure for class-based dispatchers. Note that `DispatcherRegistry` is a **new class** - there was no prior `HandlerRegistry` for message dispatching (the existing `ProtocolBindingRegistry` handles protocol handlers and remains unchanged):

```python
from omnibase_infra.runtime import (
    DispatcherRegistry,
    ProtocolMessageDispatcher,
)
from omnibase_infra.enums import EnumMessageCategory
from omnibase_core.enums.enum_node_kind import EnumNodeKind

# Implement the ProtocolMessageDispatcher protocol
class UserEventDispatcher:
    """Class-based dispatcher for user events."""

    @property
    def dispatcher_id(self) -> str:
        return "user-event-dispatcher"

    @property
    def category(self) -> EnumMessageCategory:
        return EnumMessageCategory.EVENT

    @property
    def message_types(self) -> set[str]:
        return {"UserCreated", "UserUpdated", "UserDeleted"}

    @property
    def node_kind(self) -> EnumNodeKind:
        return EnumNodeKind.REDUCER

    async def handle(
        self, envelope: ModelEventEnvelope[Any]
    ) -> ModelDispatchResult:
        # Process the event
        return ModelDispatchResult(
            status=EnumDispatchStatus.SUCCESS,
            topic="user.events.processed",
            dispatcher_id=self.dispatcher_id,
        )

# Register with DispatcherRegistry
registry = DispatcherRegistry()
registry.register_dispatcher(UserEventDispatcher())
registry.freeze()

# Lookup dispatchers
dispatchers = registry.get_dispatchers(
    category=EnumMessageCategory.EVENT,
    message_type="UserCreated",
)
```

**Key differences from MessageDispatchEngine:**
- `DispatcherRegistry` manages class-based dispatchers implementing `ProtocolMessageDispatcher`
- `MessageDispatchEngine` accepts function-based dispatchers (callables)
- Both use the "freeze after init" pattern for thread safety
- Both support execution shape validation

### What Does NOT Change

1. **Protocol Handlers** (`handlers/` directory):
   - `HttpHandler`, `DbHandler`, `VaultHandler`, `ConsulHandler`
   - These remain "handlers" as they handle protocol interactions
   - `HANDLER_TYPE_*` constants remain unchanged
   - `ProtocolBindingRegistry` continues to manage handlers

2. **EnumDispatchStatus Values**:
   - `NO_HANDLER` - Status enum value unchanged
   - `HANDLER_ERROR` - Status enum value unchanged

3. **Error Classes**:
   - Infrastructure errors remain unchanged
   - `HANDLER_EXECUTION_ERROR` error code unchanged

## Migration Steps

### Step 1: Update Method Calls

Replace deprecated method calls with new equivalents:

```python
# Before
engine.register_handler(handler_id="...", handler=fn, category=cat)
engine.get_handler_metrics("...")
engine.handler_count

# After
engine.register_dispatcher(dispatcher_id="...", dispatcher=fn, category=cat)
engine.get_dispatcher_metrics("...")
engine.dispatcher_count
```

### Step 2: Update Variable Names

Rename variables to match new terminology:

```python
# Before
def create_user_handler(envelope: ModelEventEnvelope) -> dict:
    ...

handler_id = "user-event-handler"
handler_func = create_user_handler

# After
def create_user_dispatcher(envelope: ModelEventEnvelope) -> dict:
    ...

dispatcher_id = "user-event-dispatcher"
dispatcher_func = create_user_dispatcher
```

### Step 3: Update Model References

Use the new dispatch model names:

```python
# Before (hypothetical)
from omnibase_infra.models.handlers import ModelHandlerRegistration

# After
from omnibase_infra.models.dispatch import ModelDispatcherRegistration
```

### Step 4: Update Metrics Access

```python
# Before
metrics = engine.get_handler_metrics("my-handler")
if metrics:
    print(f"Handler errors: {metrics.error_count}")

# After
metrics = engine.get_dispatcher_metrics("my-dispatcher")
if metrics:
    print(f"Dispatcher errors: {metrics.error_count}")
```

## Backward Compatibility

### Legacy Aliases (Deprecated)

The following legacy aliases are provided for backward compatibility but are deprecated and will be removed in a future version:

```python
class MessageDispatchEngine:
    # Legacy property alias
    @property
    def handler_count(self) -> int:
        """Get the number of registered dispatchers (legacy alias)."""
        return len(self._dispatchers)

    # Legacy method alias
    def register_handler(
        self,
        handler_id: str,
        handler: DispatcherFunc,
        category: EnumMessageCategory,
        message_types: set[str] | None = None,
    ) -> None:
        """Register a message handler (legacy alias for register_dispatcher)."""
        return self.register_dispatcher(
            dispatcher_id=handler_id,
            dispatcher=handler,
            category=category,
            message_types=message_types,
        )

    # Legacy method alias
    def get_handler_metrics(self, handler_id: str) -> ModelDispatcherMetrics | None:
        """Get metrics for a specific handler (legacy alias)."""
        return self.get_dispatcher_metrics(handler_id)
```

### Deprecation Timeline

| Version | Status |
|---------|--------|
| 0.4.0 | Legacy aliases added, deprecation warnings |
| 0.5.0 | Deprecation warnings become errors in dev mode |
| 0.6.0 | Legacy aliases removed |

## Automated Migration Script

A migration script is provided to help update your codebase:

```bash
#!/usr/bin/env bash
# migrate_handler_to_dispatcher.sh
# Run from project root

# Detect OS for sed compatibility
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS requires '' after -i
    SED_INPLACE="sed -i ''"
else
    # Linux/GNU sed
    SED_INPLACE="sed -i"
fi

# Rename in Python files (excluding handlers/ directory and enum values)
find . -name "*.py" \
    -not -path "*/handlers/*" \
    -not -path "*/.venv/*" \
    -not -path "*/node_modules/*" \
    -exec $SED_INPLACE \
        -e 's/handler_id/dispatcher_id/g' \
        -e 's/register_handler/register_dispatcher/g' \
        -e 's/get_handler_metrics/get_dispatcher_metrics/g' \
        -e 's/handler_count/dispatcher_count/g' \
        {} \;

echo "Migration complete. Please review changes and run tests."
```

**Important**: Review all changes manually, as the script may affect:
- Comments and docstrings
- String literals
- Unrelated variables with "handler" in the name

## Verification Checklist

After migration, verify:

- [ ] All `register_handler` calls replaced with `register_dispatcher`
- [ ] All `handler_id` parameters renamed to `dispatcher_id`
- [ ] All `get_handler_metrics` calls replaced with `get_dispatcher_metrics`
- [ ] Tests pass with new terminology
- [ ] No deprecation warnings in logs

## FAQ

### Q: Why rename Handler to Dispatcher?

**A**: To clarify architectural roles:
- **Handlers** (`handlers/`): Protocol adapters that handle external I/O (HTTP requests, DB queries, etc.)
- **Dispatchers** (`dispatch/`): Message routing units that dispatch events to processing functions

### Q: Do I need to rename my `handlers/` directory?

**A**: No. Protocol handlers (HTTP, DB, Kafka, Vault, Consul) remain "handlers" as they handle protocol-level concerns. Only message routing units in the dispatch engine are "dispatchers".

### Q: Will the EnumDispatchStatus.HANDLER_ERROR value change?

**A**: No. Enum values are part of the serialized contract and remain unchanged to maintain backward compatibility with existing events and logs.

### Q: What about HANDLER_TYPE_* constants?

**A**: These remain unchanged. They identify protocol handler types (HTTP, DATABASE, etc.), not dispatchers.

## Architecture Overview

### Handler vs Dispatcher Comparison

```
+-------------------------------------------------------------------+
|                       ONEX Infrastructure                          |
+-------------------------------------------------------------------+
|                                                                   |
|  Protocol Handlers (handlers/)         Dispatchers (dispatch/)    |
|  -------------------------         -------------------------      |
|  External I/O adapters             Message routing units          |
|  Protocol-specific logic           Category-based processing      |
|  HANDLER_TYPE_* constants          dispatcher_id identifiers      |
|  ProtocolBindingRegistry           DispatcherRegistry             |
|                                    MessageDispatchEngine           |
|                                                                   |
|  Examples:                         Examples:                       |
|  - HttpHandler                     - UserEventDispatcher          |
|  - DbHandler                       - OrderCommandDispatcher       |
|  - VaultHandler                    - ProvisionIntentDispatcher    |
|  - ConsulHandler                                                  |
|  - KafkaHandler                                                   |
|                                                                   |
+-------------------------------------------------------------------+
```

### Two Dispatch Approaches

The ONEX infrastructure provides two complementary approaches for message dispatching:

**1. MessageDispatchEngine (Function-Based)**
- Accepts callable functions as dispatchers
- Simpler for lightweight event handlers
- Uses routes to match topics to dispatchers

```python
engine = MessageDispatchEngine()
engine.register_dispatcher(
    dispatcher_id="user-dispatcher",
    dispatcher=lambda envelope: process_user(envelope),
    category=EnumMessageCategory.EVENT,
)
engine.freeze()
result = await engine.dispatch("dev.user.events.v1", envelope)
```

**2. DispatcherRegistry (Class-Based)**
- Requires ProtocolMessageDispatcher protocol implementation
- Validates execution shapes (category -> node_kind)
- Better for complex dispatchers with state or dependencies

```python
registry = DispatcherRegistry()
registry.register_dispatcher(MyDispatcherClass())
registry.freeze()
dispatchers = registry.get_dispatchers(EnumMessageCategory.EVENT, "UserCreated")
```

## Import Reference

### Runtime Components

```python
# Primary imports for message dispatching
from omnibase_infra.runtime import (
    MessageDispatchEngine,      # Function-based message dispatch engine
    DispatcherRegistry,         # Class-based dispatcher registry
    ProtocolMessageDispatcher,  # Protocol for implementing dispatchers
)
```

### Dispatch Models

```python
# Dispatch models
from omnibase_infra.models.dispatch import (
    ModelDispatchRoute,           # Route definition model
    ModelDispatchResult,          # Dispatch result with status and metrics
    ModelDispatcherRegistration,  # Dispatcher registration metadata
    ModelDispatcherMetrics,       # Per-dispatcher metrics
    ModelDispatchMetrics,         # Aggregate dispatch metrics
    EnumDispatchStatus,           # Status enum (SUCCESS, NO_HANDLER, HANDLER_ERROR, etc.)
)

# Message category enum
from omnibase_infra.enums import EnumMessageCategory
```

### Protocol Handlers (Unchanged)

These components remain as "handlers" - they handle protocol-level concerns:

```python
# Protocol handler registry (unchanged)
from omnibase_infra.runtime import (
    ProtocolBindingRegistry,    # Registry for protocol handlers
    HANDLER_TYPE_HTTP,          # Handler type constants
    HANDLER_TYPE_DATABASE,
    HANDLER_TYPE_KAFKA,
    HANDLER_TYPE_VAULT,
    HANDLER_TYPE_CONSUL,
)
```

## Related Documentation

- [MessageDispatchEngine API](../architecture/RUNTIME_HOST_IMPLEMENTATION_PLAN.md)
- [Dispatch Models](../../src/omnibase_infra/models/dispatch/__init__.py)
- [Protocol Handler Registry](../../src/omnibase_infra/runtime/handler_registry.py)
- [Dispatcher Registry](../../src/omnibase_infra/runtime/dispatcher_registry.py)

## Change History

| Date | Version | Description |
|------|---------|-------------|
| 2025-12-19 | 0.4.0 | Initial migration guide created |
| 2025-12-19 | 0.4.0 | Enhanced: Added Import Reference section, fixed cross-platform script, clarified DispatcherRegistry is new (not a rename) |
