# Handoff Document: Contract-Driven Handler Routing

**Date**: 2026-01-11
**Ticket**: [OMN-1293](https://linear.app/omninode/issue/OMN-1293)
**Branch**: `jonah/omn-1293-omnibase_core-mixinhandlerrouting-for-contract-driven`

---

> **Scope Note**: This document covers OMN-1293 (MixinHandlerRouting integration into
> omnibase_core). For the OMN-1102 declarative refactor of NodeRegistrationOrchestrator,
> see the `node.py` module docstring in
> `src/omnibase_infra/nodes/node_registration_orchestrator/node.py`.

---

## Executive Summary

**UPDATE (2026-01-11)**: **OMN-1293 is NOW COMPLETE**. The `MixinHandlerRouting` mixin is implemented in omnibase_core and integrated into `NodeOrchestrator`. The `RegistryInfraNodeRegistrationOrchestrator` registry is fully implemented with handler adapters. Contract-driven handler routing is now functional.

---

## Context

### Original Problem

PR review for `NodeRegistrationOrchestrator` refactor identified gaps:

1. **Registry Implementation Missing** - Docstring referenced non-existent `registry_infra_node_registration_orchestrator.py` - **NOW RESOLVED**
2. **No Integration Tests** - No tests verifying contract-driven handler routing
3. **Documentation Gap** - No usage examples for declarative routing - **NOW RESOLVED** (see registry docstrings)

### Investigation Findings (UPDATED 2026-01-11)

| Question | Answer |
|----------|--------|
| Does `NodeOrchestrator` base class support handler routing? | **YES** - Composes `MixinHandlerRouting` with `_init_handler_routing()`, `route_to_handlers()`, `is_routing_initialized` |
| Does `MixinHandlerRouting` exist? | **YES** - Implemented in omnibase_core and integrated into base classes |
| Is `ServiceHandlerRegistry` ready? | **YES** - Fully implemented with `register_handler()`, `get_handler_by_id()`, `get_handlers()`, `freeze()`, `unregister_handler()`, `handler_count`, `is_frozen` |
| Are routing models ready? | **YES** - `ModelHandlerRoutingSubcontract` and `ModelHandlerRoutingEntry` are implemented and in use |

---

## What Exists in omnibase_core

### MixinHandlerRouting (COMPLETE)

**Status**: Implemented and integrated into `NodeOrchestrator` base class.

```python
# This import WORKS:
from omnibase_core.nodes import NodeOrchestrator
# NodeOrchestrator now composes MixinHandlerRouting
```

**Available Methods** (from MixinHandlerRouting):
```python
class MixinHandlerRouting:
    def _init_handler_routing(
        self,
        handler_routing: ModelHandlerRoutingSubcontract | None,
        registry: ServiceHandlerRegistry,
    ) -> None: ...

    def route_to_handlers(
        self,
        routing_key: str,
        category: EnumMessageCategory,
    ) -> list[ProtocolMessageHandler]: ...

    def validate_handler_routing(self) -> list[str]: ...

    @property
    def is_routing_initialized(self) -> bool: ...
```

**Implemented Routing Strategies**:
- `payload_type_match` - Route by event model class name
- `operation_match` - Route by operation field value
- `topic_pattern` - Glob pattern matching

### Node Integration (COMPLETE)

**Status**: `NodeOrchestrator` NOW composes `MixinHandlerRouting`. Handler routing methods are available on the base class.

```python
# CURRENT (handler routing integrated):
class NodeOrchestrator(NodeCoreBase, MixinWorkflowExecution, MixinHandlerRouting):
    ...  # Handler routing methods available

# Available methods:
# - _init_handler_routing(handler_routing, registry)
# - route_to_handlers(routing_key, category)
# - validate_handler_routing()
# - is_routing_initialized (property)
```

### ServiceHandlerRegistry (COMPLETE)

**Status**: Fully implemented and functional.

**Available Methods**:
- `register_handler(handler)` - Register a handler
- `get_handler_by_id(handler_id)` - Lookup by ID
- `get_handlers()` - Get all handlers
- `freeze()` - Make registry immutable (thread-safe)
- `unregister_handler(handler_id)` - Remove handler (before freeze)

**Properties**:
- `handler_count` - Number of registered handlers
- `is_frozen` - Whether registry is frozen

### Models (COMPLETE)

Both routing models are implemented and in use:
- `ModelHandlerRoutingSubcontract` - Contract configuration (version, routing_strategy, handlers, default_handler)
- `ModelHandlerRoutingEntry` - Individual routing entries (routing_key, handler_key)

```python
# Import pattern:
from omnibase_core.models.contracts.subcontracts.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_core.models.contracts.subcontracts.model_handler_routing_subcontract import (
    ModelHandlerRoutingSubcontract,
)
```

---

## What Has Been Implemented in omnibase_infra

### 1. NodeRegistrationOrchestrator Integration (COMPLETE)

The orchestrator now:
1. Calls `_init_handler_routing()` via `_initialize_handler_routing()` method
2. Registers handlers via `RegistryInfraNodeRegistrationOrchestrator.create_registry()`
3. Uses `route_to_handlers()` for event processing (inherited from base class)

```python
class NodeRegistrationOrchestrator(NodeOrchestrator):
    def __init__(
        self,
        container: ModelONEXContainer,
        projection_reader: ProjectionReaderRegistration | None = None,
    ) -> None:
        super().__init__(container)
        self._projection_reader = projection_reader

        # Initialize handler routing if projection_reader is available
        if projection_reader is not None:
            self._initialize_handler_routing(projection_reader)

    def _initialize_handler_routing(
        self, projection_reader: ProjectionReaderRegistration
    ) -> None:
        handler_routing = _create_handler_routing_subcontract()
        registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
            projection_reader=projection_reader,
        )
        self._init_handler_routing(handler_routing, registry)
```

### 2. Handler Registration (COMPLETE)

Handler adapters and registration implemented via `RegistryInfraNodeRegistrationOrchestrator`:

```python
# In registry/registry_infra_node_registration_orchestrator.py
registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
    projection_reader=reader,
    projector=projector,
    consul_handler=consul_handler,
)
# Registry is automatically frozen and thread-safe
```

Handler adapters bridge existing handlers to `ProtocolMessageHandler` interface:
- `AdapterNodeIntrospected` - Wraps `HandlerNodeIntrospected`
- `AdapterRuntimeTick` - Wraps `HandlerRuntimeTick`
- `AdapterNodeRegistrationAcked` - Wraps `HandlerNodeRegistrationAcked`
- `AdapterNodeHeartbeat` - Wraps `HandlerNodeHeartbeat`

### 3. Integration Tests

**Remaining work**: Add tests that verify:
- Contract `handler_routing` section is parsed correctly
- Events route to correct handlers
- Unknown handlers fail fast at init

### 4. Docstring (COMPLETE)

The registry file now exists and the docstring accurately references it:
- `registry/registry_infra_node_registration_orchestrator.py`: Handler wiring - **NOW EXISTS**

---

## Ticket Status

### OMN-1293 Current State (COMPLETE)

The ticket work is now **COMPLETE**:
- **MixinHandlerRouting** mixin - **IMPLEMENTED** in omnibase_core
- **ModelHandlerRoutingSubcontract** model - **IMPLEMENTED** and in use
- **ModelHandlerRoutingEntry** model - **IMPLEMENTED** and in use
- Multiple routing strategies - **IMPLEMENTED** (payload_type_match, operation_match, topic_pattern)
- **NodeOrchestrator** composes mixin - **COMPLETE** (has `_init_handler_routing()`, `route_to_handlers()`, `is_routing_initialized`)
- **ServiceHandlerRegistry** - **COMPLETE** (fully functional)
- **RegistryInfraNodeRegistrationOrchestrator** - **COMPLETE** (handler adapters and factory methods)
- **NodeRegistrationOrchestrator** integration - **COMPLETE** (uses handler routing from base class)

### Remaining Work

**Minor items**:
1. **Integration tests** - Add tests verifying end-to-end handler routing
2. **NodeEffect mixin integration** - Verify if needed (may be orchestrator-only pattern)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     omnibase_core                            │
├─────────────────────────────────────────────────────────────┤
│  MixinHandlerRouting                      ← COMPLETE ✓       │
│  ├── _init_handler_routing(contract, registry)    ← DONE    │
│  ├── route_to_handlers(routing_key, category)     ← DONE    │
│  ├── validate_handler_routing()                   ← DONE    │
│  └── is_routing_initialized (property)            ← DONE    │
│                                                              │
│  ServiceHandlerRegistry                   ← COMPLETE ✓       │
│  ├── register_handler(handler)                              │
│  ├── get_handler_by_id(handler_id)                          │
│  ├── get_handlers()                                         │
│  ├── freeze()                                               │
│  ├── unregister_handler(handler_id)                         │
│  ├── handler_count (property)                               │
│  └── is_frozen (property)                                   │
│                                                              │
│  ModelHandlerRoutingSubcontract           ← COMPLETE ✓       │
│  ModelHandlerRoutingEntry                 ← COMPLETE ✓       │
│                                                              │
│  NodeOrchestrator(NodeCoreBase, MixinWorkflow, MixinHandler)│
│                   ↑ NOW COMPOSES MixinHandlerRouting ✓      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ extends
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     omnibase_infra                           │
├─────────────────────────────────────────────────────────────┤
│  RegistryInfraNodeRegistrationOrchestrator    ← COMPLETE ✓  │
│  ├── create_registry() - static factory                     │
│  ├── AdapterNodeIntrospected                                │
│  ├── AdapterRuntimeTick                                     │
│  ├── AdapterNodeRegistrationAcked                           │
│  └── AdapterNodeHeartbeat                                   │
│                                                              │
│  NodeRegistrationOrchestrator(NodeOrchestrator) ← COMPLETE ✓│
│  ├── contract.yaml defines handler_routing                  │
│  ├── __init__: calls _initialize_handler_routing()    ✓     │
│  └── process(): uses route_to_handlers()              ✓     │
│                                                              │
│  Handlers (registered via adapters):            ← COMPLETE ✓│
│  ├── HandlerNodeIntrospected                                │
│  ├── HandlerRuntimeTick                                     │
│  ├── HandlerNodeRegistrationAcked                           │
│  └── HandlerNodeHeartbeat                                   │
└─────────────────────────────────────────────────────────────┘

LEGEND: ✓ = Complete, DONE = Implemented
```

---

## Key Files

### omnibase_core

| File | Status | Notes |
|------|--------|-------|
| `mixins/mixin_handler_routing.py` | **COMPLETE** | Mixin implemented and integrated |
| `services/service_handler_registry.py` | **COMPLETE** | Fully functional |
| `models/contracts/subcontracts/model_handler_routing_subcontract.py` | **COMPLETE** | Implemented and in use |
| `models/contracts/subcontracts/model_handler_routing_entry.py` | **COMPLETE** | Implemented and in use |
| `nodes/node_orchestrator.py` | **COMPLETE** | Composes MixinHandlerRouting |

### omnibase_infra

| File | Status | Notes |
|------|--------|-------|
| `nodes/node_registration_orchestrator/node.py` | **COMPLETE** | Calls `_init_handler_routing()` via `_initialize_handler_routing()` |
| `nodes/node_registration_orchestrator/contract.yaml` | **COMPLETE** | Handler routing config defined |
| `nodes/node_registration_orchestrator/handlers/` | **COMPLETE** | Handler implementations ready |
| `nodes/node_registration_orchestrator/registry/registry_infra_node_registration_orchestrator.py` | **COMPLETE** | Handler adapters and factory (NEW) |

---

## Next Steps

### Phase 1: omnibase_core Implementation (OMN-1293) - COMPLETE

All items completed:
1. **Routing models** - `ModelHandlerRoutingSubcontract` and `ModelHandlerRoutingEntry` implemented
2. **MixinHandlerRouting** - Implemented in `omnibase_core/mixins/mixin_handler_routing.py`
   - `_init_handler_routing(contract, registry)` - implemented
   - `route_to_handlers(routing_key, category)` - implemented
   - `validate_handler_routing()` - implemented
   - `is_routing_initialized` property - implemented
3. **Routing strategies** - `payload_type_match`, `operation_match`, `topic_pattern` - implemented
4. **Base class integration** - `NodeOrchestrator` now composes `MixinHandlerRouting`

### Phase 2: omnibase_infra Integration - COMPLETE

All items completed:
6. **Initialize routing** - `NodeRegistrationOrchestrator` calls `_init_handler_routing()` via `_initialize_handler_routing()`
7. **Register handlers** - `RegistryInfraNodeRegistrationOrchestrator.create_registry()` registers all handlers with adapters
8. **Docstring** - Registry file now exists, docstring accurate
9. **PR** - Original review concerns addressed

### Remaining Work

10. **Integration tests** - Add tests verifying end-to-end handler routing behavior
11. **Documentation** - Update any remaining outdated references

---

## Investigation Findings (Updated - 2026-01-11)

This section documents the updated status after implementation.

### Verification Method

```python
# Verified imports that confirm implementation:

# 1. MixinHandlerRouting - EXISTS (via NodeOrchestrator)
from omnibase_core.nodes import NodeOrchestrator
# Result: Success - NodeOrchestrator composes MixinHandlerRouting

# 2. ServiceHandlerRegistry - EXISTS
from omnibase_core.services import ServiceHandlerRegistry
# Result: Success - class is available

# 3. NodeOrchestrator inspection - HAS handler routing methods
from omnibase_core.nodes import NodeOrchestrator
# Result: Has _init_handler_routing, route_to_handlers, validate_handler_routing, is_routing_initialized

# 4. Routing models - EXIST
from omnibase_core.models.contracts.subcontracts.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_core.models.contracts.subcontracts.model_handler_routing_subcontract import (
    ModelHandlerRoutingSubcontract,
)
# Result: Success - both models available
```

### Component Status Summary

| Component | Expected | Actual | Status |
|-----------|----------|--------|--------|
| `MixinHandlerRouting` | Exists in `omnibase_core.mixins` | **Exists and integrated** | **COMPLETE** |
| `ServiceHandlerRegistry` | Handler registration service | **Exists and works** | **COMPLETE** |
| `NodeOrchestrator` | Composes `MixinHandlerRouting` | **Composes mixin** | **COMPLETE** |
| Routing models | `ModelHandlerRoutingSubcontract`, `ModelHandlerRoutingEntry` | **Both exist** | **COMPLETE** |
| `RegistryInfraNodeRegistrationOrchestrator` | Handler registry factory | **Exists with adapters** | **COMPLETE** |
| `NodeRegistrationOrchestrator` | Uses handler routing | **Integrated** | **COMPLETE** |

### ServiceHandlerRegistry Confirmed Methods

The following methods were verified to exist on `ServiceHandlerRegistry`:

```python
# Methods
register_handler(handler) -> None
get_handler_by_id(handler_id: str) -> Handler | None
get_handlers() -> list[Handler]
freeze() -> None
unregister_handler(handler_id: str) -> bool

# Properties
handler_count: int
is_frozen: bool
```

### Implementation Summary

**OMN-1293 is now COMPLETE**. All core components have been implemented:

- **MixinHandlerRouting** mixin - implemented in omnibase_core
- **Three routing strategies** - payload_type_match, operation_match, topic_pattern
- **NodeOrchestrator integration** - base class composes mixin
- **RegistryInfraNodeRegistrationOrchestrator** - handler adapters and factory methods
- **NodeRegistrationOrchestrator** - integrated with handler routing

**Remaining work**: Integration tests to verify end-to-end handler routing behavior.

---

## References

- [OMN-1293 Linear Ticket](https://linear.app/omninode/issue/OMN-1293)
- `omnibase_core` version: Check `pyproject.toml` for current version
- PR under review: `jonah/omn-1102-refactor-noderegistrationorchestrator-to-be-fully`
