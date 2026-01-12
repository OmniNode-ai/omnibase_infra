# Handoff Document: Contract-Driven Handler Routing

**Date**: 2026-01-11 (Updated)
**Tickets**:
- [OMN-1293](https://linear.app/omninode/issue/OMN-1293) - MixinHandlerRouting integration
- [OMN-1102](https://linear.app/omninode/issue/OMN-1102) - Declarative orchestrator refactor
**Branch**: `jonah/omn-1102-refactor-noderegistrationorchestrator-to-be-fully`

---

## Scope Clarification

This document covers **two related but distinct tickets**:

| Ticket | Scope | Status |
|--------|-------|--------|
| **OMN-1293** | `MixinHandlerRouting` mixin implementation in `omnibase_core`. Provides contract-driven handler routing infrastructure for all orchestrators. | **COMPLETE** |
| **OMN-1102** | Declarative refactor of `NodeRegistrationOrchestrator` in `omnibase_infra`. Makes the orchestrator pure declarative with runtime-driven initialization. | **COMPLETE** (in PR #141) |

**Key Distinction**:
- **OMN-1293** is about the _infrastructure_ (the mixin, registry service, and routing models)
- **OMN-1102** is about _applying_ that infrastructure to make `NodeRegistrationOrchestrator` fully declarative

---

## Executive Summary

**UPDATE (2026-01-11)**: **OMN-1293 is NOW COMPLETE**. The `MixinHandlerRouting` mixin is implemented in omnibase_core and integrated into `NodeOrchestrator`. The `RegistryInfraNodeRegistrationOrchestrator` registry is fully implemented. Handlers implement `ProtocolMessageHandler` directly (no adapter classes needed). Contract-driven handler routing is now functional.

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

The orchestrator is now **pure declarative**:
1. **No custom initialization logic** - just extends `NodeOrchestrator`
2. **Runtime-driven routing** - `RuntimeHostProcess` initializes handler routing externally
3. **Registry factory** - `RegistryInfraNodeRegistrationOrchestrator.create_registry()` creates handler instances
4. **Base class routing** - `route_to_handlers()` is inherited from `MixinHandlerRouting`

```python
# PURE DECLARATIVE - node.py contains ONLY the class definition
class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Declarative orchestrator for node registration workflow.

    All behavior is defined in contract.yaml - no custom logic here.
    Handler routing is driven entirely by the contract and initialized
    by the runtime via MixinHandlerRouting from the base class.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)
        # No custom logic - runtime handles handler routing initialization
```

**Runtime Initialization Pattern** (performed by `RuntimeHostProcess`, not the orchestrator):
```python
# The runtime (not the orchestrator) performs this initialization:
handler_routing = _create_handler_routing_subcontract()
registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
    projection_reader=reader,
    projector=projector,
    consul_handler=consul_handler,
)
orchestrator._init_handler_routing(handler_routing, registry)
```

This separation ensures the orchestrator remains purely declarative with zero custom initialization logic.

### 2. Handler Registration (COMPLETE)

Handler registration implemented via `RegistryInfraNodeRegistrationOrchestrator`:

```python
# In registry/registry_infra_node_registration_orchestrator.py
registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
    projection_reader=reader,
    projector=projector,
    consul_handler=consul_handler,
)
# Registry is automatically frozen and thread-safe
```

Handlers implement `ProtocolMessageHandler` directly (no adapter classes needed):
- `HandlerNodeIntrospected` - Processes `ModelNodeIntrospectionEvent`
- `HandlerRuntimeTick` - Processes `ModelRuntimeTick`
- `HandlerNodeRegistrationAcked` - Processes `ModelNodeRegistrationAcked`
- `HandlerNodeHeartbeat` - Processes `ModelNodeHeartbeatEvent` (requires projector)

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

The `MixinHandlerRouting` infrastructure work is now **COMPLETE**:
- **MixinHandlerRouting** mixin - **IMPLEMENTED** in omnibase_core
- **ModelHandlerRoutingSubcontract** model - **IMPLEMENTED** and in use
- **ModelHandlerRoutingEntry** model - **IMPLEMENTED** and in use
- Multiple routing strategies - **IMPLEMENTED** (payload_type_match, operation_match, topic_pattern)
- **NodeOrchestrator** composes mixin - **COMPLETE** (has `_init_handler_routing()`, `route_to_handlers()`, `is_routing_initialized`)
- **ServiceHandlerRegistry** - **COMPLETE** (fully functional)

### OMN-1102 Current State (COMPLETE in PR #141)

The declarative refactor of `NodeRegistrationOrchestrator` is now **COMPLETE**:
- **RegistryInfraNodeRegistrationOrchestrator** - **COMPLETE** (handler factory and `create_registry()` method)
- **NodeRegistrationOrchestrator** - **COMPLETE** (pure declarative, runtime-driven initialization)
- **Handler registration** - **COMPLETE** (4 handlers implement ProtocolMessageHandler directly)
- **Registry file exists** - `registry/registry_infra_node_registration_orchestrator.py`

### Remaining Work

**All core implementation is COMPLETE**. Only remaining items:

1. **Integration tests** - Add tests verifying end-to-end handler routing behavior
   - Contract `handler_routing` section parsing
   - Events routing to correct handlers
   - Unknown handler fail-fast validation

**Not needed** (clarified during PR #141 review):
- ~~NodeEffect mixin integration~~ - Handler routing is orchestrator-only pattern

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
│  └── get_handler_map() - instance method                    │
│                                                              │
│  NodeRegistrationOrchestrator(NodeOrchestrator) ← COMPLETE ✓│
│  ├── contract.yaml defines handler_routing                  │
│  ├── __init__: PURE DECLARATIVE (no custom logic)     ✓     │
│  └── Runtime calls _init_handler_routing()            ✓     │
│                                                              │
│  Handlers (implement ProtocolMessageHandler):   ← COMPLETE ✓│
│  ├── HandlerNodeIntrospected                                │
│  ├── HandlerRuntimeTick                                     │
│  ├── HandlerNodeRegistrationAcked                           │
│  └── HandlerNodeHeartbeat (requires projector)              │
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
| `nodes/node_registration_orchestrator/node.py` | **COMPLETE** | Pure declarative (no custom init logic, runtime-driven) |
| `nodes/node_registration_orchestrator/contract.yaml` | **COMPLETE** | Handler routing config defined |
| `nodes/node_registration_orchestrator/handlers/` | **COMPLETE** | Handler implementations ready |
| `nodes/node_registration_orchestrator/registry/registry_infra_node_registration_orchestrator.py` | **COMPLETE** | Handler factory with `create_registry()` method |

**Registry File Details** (`registry/registry_infra_node_registration_orchestrator.py`):
- **Path**: `src/omnibase_infra/nodes/node_registration_orchestrator/registry/registry_infra_node_registration_orchestrator.py`
- **Class**: `RegistryInfraNodeRegistrationOrchestrator`
- **Factory**: `create_registry(projection_reader, projector?, consul_handler?)` -> `ServiceHandlerRegistry`
- **Handlers**: Registers `HandlerNodeIntrospected`, `HandlerRuntimeTick`, `HandlerNodeRegistrationAcked`, `HandlerNodeHeartbeat` directly (no adapter classes)

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

### Phase 2: omnibase_infra Integration (OMN-1102) - COMPLETE

All items completed:
6. **Pure declarative orchestrator** - `NodeRegistrationOrchestrator` extends `NodeOrchestrator` with no custom logic
7. **Runtime-driven initialization** - Handler routing initialized by `RuntimeHostProcess`, not by the orchestrator
8. **Registry file implemented** - `RegistryInfraNodeRegistrationOrchestrator` with `create_registry()` factory
9. **PR #141** - Original review concerns addressed, registry file now exists

### Remaining Work

10. **Integration tests** - Add tests verifying end-to-end handler routing behavior
    - Verify contract parsing
    - Verify event-to-handler routing
    - Verify fail-fast on unknown handlers

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

| Component | Expected | Actual | Status | Ticket |
|-----------|----------|--------|--------|--------|
| `MixinHandlerRouting` | Exists in `omnibase_core.mixins` | **Exists and integrated** | **COMPLETE** | OMN-1293 |
| `ServiceHandlerRegistry` | Handler registration service | **Exists and works** | **COMPLETE** | OMN-1293 |
| `NodeOrchestrator` | Composes `MixinHandlerRouting` | **Composes mixin** | **COMPLETE** | OMN-1293 |
| Routing models | `ModelHandlerRoutingSubcontract`, `ModelHandlerRoutingEntry` | **Both exist** | **COMPLETE** | OMN-1293 |
| `RegistryInfraNodeRegistrationOrchestrator` | Handler registry factory | **Exists with create_registry()** | **COMPLETE** | OMN-1102 |
| `NodeRegistrationOrchestrator` | Pure declarative | **No custom init logic** | **COMPLETE** | OMN-1102 |
| Registry file | Exists at documented path | **File exists** | **COMPLETE** | OMN-1102 |

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

**Both OMN-1293 and OMN-1102 are now COMPLETE**:

**OMN-1293 (omnibase_core infrastructure)**:
- **MixinHandlerRouting** mixin - implemented in omnibase_core
- **Three routing strategies** - payload_type_match, operation_match, topic_pattern
- **NodeOrchestrator integration** - base class composes mixin
- **ServiceHandlerRegistry** - handler registration and lookup service

**OMN-1102 (omnibase_infra declarative refactor)**:
- **NodeRegistrationOrchestrator** - pure declarative with runtime-driven initialization
- **RegistryInfraNodeRegistrationOrchestrator** - `create_registry()` factory method
- **Four handlers** - HandlerNodeIntrospected, HandlerRuntimeTick, HandlerNodeRegistrationAcked, HandlerNodeHeartbeat (implement ProtocolMessageHandler directly, no adapter classes)

**Remaining work**: Integration tests to verify end-to-end handler routing behavior.

---

## References

- [OMN-1293 Linear Ticket](https://linear.app/omninode/issue/OMN-1293) - MixinHandlerRouting infrastructure
- [OMN-1102 Linear Ticket](https://linear.app/omninode/issue/OMN-1102) - Declarative orchestrator refactor
- `omnibase_core` version: Check `pyproject.toml` for current version (^0.6.4)
- PR: [#141](https://github.com/omninode/omnibase_infra/pull/141) - Addresses PR review concerns, adds registry file
