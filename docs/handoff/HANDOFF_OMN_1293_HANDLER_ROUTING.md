# Handoff Document: Contract-Driven Handler Routing

**Date**: 2026-01-11
**Ticket**: [OMN-1293](https://linear.app/omninode/issue/OMN-1293)
**Branch**: `jonah/omn-1293-omnibase_core-mixinhandlerrouting-for-contract-driven`

---

## Executive Summary

**CORRECTION (2026-01-11)**: Initial investigation was inaccurate. Re-investigation revealed that **OMN-1293 is NOT complete**. The `MixinHandlerRouting` mixin does NOT exist in omnibase_core. Only `ServiceHandlerRegistry` is implemented. Significant work remains to implement the full contract-driven handler routing capability.

---

## Context

### Original Problem

PR review for `NodeRegistrationOrchestrator` refactor identified gaps:

1. **Registry Implementation Missing** - Docstring referenced non-existent `registry_infra_node_registration_orchestrator.py`
2. **No Integration Tests** - No tests verifying contract-driven handler routing
3. **Documentation Gap** - No usage examples for declarative routing

### Investigation Findings (CORRECTED 2026-01-11)

| Question | Answer |
|----------|--------|
| Does `NodeOrchestrator` base class support handler routing? | **NO** - No handler routing methods found |
| Does `MixinHandlerRouting` exist? | **NO** - `ModuleNotFoundError` when importing |
| Is `ServiceHandlerRegistry` ready? | **YES** - Fully implemented with `register_handler()`, `get_handler_by_id()`, `get_handlers()`, `freeze()`, `unregister_handler()`, `handler_count`, `is_frozen` |
| Are routing models ready? | **UNKNOWN** - Need to verify `ModelHandlerRoutingSubcontract`, `ModelHandlerRoutingEntry` |

---

## What Exists in omnibase_core

### MixinHandlerRouting (NOT IMPLEMENTED)

**Status**: Does NOT exist. Attempting to import raises `ModuleNotFoundError`.

```python
# This import FAILS:
# from omnibase_core.mixins import MixinHandlerRouting
# ModuleNotFoundError: No module named 'omnibase_core.mixins.mixin_handler_routing'
```

**Required Implementation**:
```python
class MixinHandlerRouting:
    def _init_handler_routing(
        self,
        handler_routing: ModelHandlerRoutingSubcontract | None,
        registry: ProtocolHandlerRegistry,
    ) -> None: ...

    def route_to_handlers(
        self,
        routing_key: str,
        category: EnumMessageCategory,
    ) -> list[ProtocolMessageHandler]: ...

    def validate_handler_routing(self) -> list[str]: ...
```

**Routing Strategies** (to be implemented):
- `payload_type_match` - Route by event model class name
- `operation_match` - Route by operation field value
- `topic_pattern` - Glob pattern matching

### Node Integration (NOT COMPLETE)

**Status**: `NodeOrchestrator` does NOT compose `MixinHandlerRouting`. No handler routing methods exist on the base class.

```python
# ACTUAL (no handler routing):
class NodeOrchestrator(NodeCoreBase, MixinWorkflowExecution):
    ...  # NO handler routing methods

# REQUIRED (when mixin is implemented):
class NodeOrchestrator(NodeCoreBase, MixinWorkflowExecution, MixinHandlerRouting):
    ...
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

### Models (STATUS UNKNOWN)

Need to verify existence of:
- `ModelHandlerRoutingSubcontract` - Contract configuration
- `ModelHandlerRoutingEntry` - Individual routing entries

---

## What Needs to Happen in omnibase_infra

### 1. NodeRegistrationOrchestrator Integration

The orchestrator needs to:
1. Call `_init_handler_routing()` in `__init__`
2. Register handlers with `ServiceHandlerRegistry`
3. Use `route_to_handlers()` in event processing

```python
class NodeRegistrationOrchestrator(NodeOrchestrator):
    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

        # Initialize handler routing from contract
        self._init_handler_routing(
            handler_routing=self._load_handler_routing_from_contract(),
            registry=container.get_service(ServiceHandlerRegistry),
        )
```

### 2. Handler Registration

Handlers must be registered before routing:

```python
# In registry setup (startup code)
registry = ServiceHandlerRegistry()
registry.register_handler(HandlerNodeIntrospected(...))
registry.register_handler(HandlerRuntimeTick(...))
registry.register_handler(HandlerNodeHeartbeat(...))
registry.freeze()  # Thread-safe after freeze
```

### 3. Integration Tests

Add tests that verify:
- Contract `handler_routing` section is parsed correctly
- Events route to correct handlers
- Unknown handlers fail fast at init

### 4. Fix Docstring

Remove reference to non-existent registry file:
```python
# REMOVE this from docstring:
# - registry/registry_infra_node_registration_orchestrator.py: Handler wiring
```

---

## Ticket Status

### OMN-1293 Current State (CORRECTED)

The ticket describes work that is **NOT complete**:
- ❌ `MixinHandlerRouting` mixin - **NOT IMPLEMENTED** (ModuleNotFoundError)
- ❓ `ModelHandlerRoutingSubcontract` model - Status unknown, needs verification
- ❌ Multiple routing strategies - **NOT IMPLEMENTED** (depends on mixin)
- ❌ `NodeOrchestrator` composes mixin - **NOT COMPLETE** (no handler routing methods)
- ❌ `NodeEffect` composes mixin - **NOT COMPLETE** (no handler routing methods)
- ❓ Tests exist - Status unknown, mixin doesn't exist

**Only complete component**: `ServiceHandlerRegistry` (fully functional)

### Recommended Action

**OMN-1293 must remain open** - Core implementation work is required:

1. **Create `MixinHandlerRouting`** in omnibase_core
2. **Verify/create routing models** (`ModelHandlerRoutingSubcontract`, `ModelHandlerRoutingEntry`)
3. **Integrate mixin into `NodeOrchestrator`** and `NodeEffect` base classes
4. **Add unit tests** for the mixin
5. **Then** proceed with omnibase_infra integration

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     omnibase_core                            │
├─────────────────────────────────────────────────────────────┤
│  MixinHandlerRouting                      ← NOT IMPLEMENTED  │
│  ├── _init_handler_routing(contract, registry)    ← TODO    │
│  ├── route_to_handlers(routing_key, category)     ← TODO    │
│  └── validate_handler_routing()                   ← TODO    │
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
│  NodeOrchestrator(NodeCoreBase, MixinWorkflowExecution)     │
│                   ↑ NO MixinHandlerRouting ← NOT COMPLETE   │
│                                                              │
│  NodeEffect(NodeCoreBase, MixinEffectExecution)             │
│             ↑ NO MixinHandlerRouting       ← NOT COMPLETE   │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ extends
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     omnibase_infra                           │
├─────────────────────────────────────────────────────────────┤
│  NodeRegistrationOrchestrator(NodeOrchestrator)             │
│  ├── contract.yaml defines handler_routing                  │
│  ├── __init__: calls _init_handler_routing()   ← BLOCKED    │
│  └── process(): uses route_to_handlers()       ← BLOCKED    │
│                                                              │
│  Handlers (to register when mixin ready):                    │
│  ├── HandlerNodeIntrospected                                │
│  ├── HandlerRuntimeTick                                     │
│  └── HandlerNodeHeartbeat                                   │
└─────────────────────────────────────────────────────────────┘

LEGEND: ✓ = Complete, TODO = Needs implementation, BLOCKED = Waiting on dependency
```

---

## Key Files

### omnibase_core

| File | Status | Notes |
|------|--------|-------|
| `mixins/mixin_handler_routing.py` | **NOT EXISTS** | Must be created |
| `services/service_handler_registry.py` | **COMPLETE** | Fully functional |
| `models/contracts/subcontracts/model_handler_routing_subcontract.py` | **UNKNOWN** | Needs verification |
| `models/contracts/subcontracts/model_handler_routing_entry.py` | **UNKNOWN** | Needs verification |
| `nodes/node_orchestrator.py` | **INCOMPLETE** | No mixin composition |
| `tests/unit/mixins/test_mixin_handler_routing.py` | **NOT EXISTS** | Mixin doesn't exist |

### omnibase_infra (BLOCKED - waiting on omnibase_core)

| File | Status | Notes |
|------|--------|-------|
| `nodes/node_registration_orchestrator/node.py` | **BLOCKED** | Cannot call `_init_handler_routing()` |
| `nodes/node_registration_orchestrator/contract.yaml` | **EXISTS** | Handler routing config defined |
| `nodes/node_registration_orchestrator/handlers/` | **EXISTS** | Handler implementations ready |

---

## Next Steps

### Phase 1: omnibase_core Implementation (OMN-1293)

1. **Verify routing models** - Check if `ModelHandlerRoutingSubcontract` and `ModelHandlerRoutingEntry` exist
2. **Create `MixinHandlerRouting`** - Implement the mixin in `omnibase_core/mixins/mixin_handler_routing.py`
   - `_init_handler_routing(contract, registry)`
   - `route_to_handlers(routing_key, category)`
   - `validate_handler_routing()`
3. **Implement routing strategies** - `payload_type_match`, `operation_match`, `topic_pattern`
4. **Integrate into base classes** - Add mixin to `NodeOrchestrator` and `NodeEffect`
5. **Add unit tests** - Create `tests/unit/mixins/test_mixin_handler_routing.py`

### Phase 2: omnibase_infra Integration (After Phase 1)

6. **Initialize routing** - Call `_init_handler_routing()` in `NodeRegistrationOrchestrator`
7. **Register handlers** - Add handler registration at startup
8. **Add integration tests** - Verify contract-driven routing works
9. **Fix docstring** - Remove reference to non-existent registry file
10. **Update PR** - Address original review concerns

---

## Investigation Findings (Corrected - 2026-01-11)

This section documents the corrected investigation findings after re-verification.

### Verification Method

```python
# Attempted imports that revealed the true status:

# 1. MixinHandlerRouting - DOES NOT EXIST
from omnibase_core.mixins import MixinHandlerRouting
# Result: ModuleNotFoundError: No module named 'omnibase_core.mixins.mixin_handler_routing'

# 2. ServiceHandlerRegistry - EXISTS
from omnibase_core.services import ServiceHandlerRegistry
# Result: Success - class is available

# 3. NodeOrchestrator inspection - NO handler routing methods
from omnibase_core.nodes import NodeOrchestrator
dir(NodeOrchestrator)
# Result: No _init_handler_routing, route_to_handlers, or validate_handler_routing methods
```

### Component Status Summary

| Component | Expected | Actual | Gap |
|-----------|----------|--------|-----|
| `MixinHandlerRouting` | Exists in `omnibase_core.mixins` | Does NOT exist | **Full implementation needed** |
| `ServiceHandlerRegistry` | Handler registration service | **Exists and works** | None |
| `NodeOrchestrator` | Composes `MixinHandlerRouting` | Does NOT compose mixin | **Mixin integration needed** |
| `NodeEffect` | Composes `MixinHandlerRouting` | Does NOT compose mixin | **Mixin integration needed** |
| Routing models | `ModelHandlerRoutingSubcontract`, `ModelHandlerRoutingEntry` | Unknown | **Verification needed** |

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

### Impact Assessment

**Original document claimed**: OMN-1293 is complete, only omnibase_infra integration remains.

**Reality**: OMN-1293 requires significant core implementation work:
- Create the entire `MixinHandlerRouting` mixin from scratch
- Implement three routing strategies
- Integrate mixin into two base classes
- Write comprehensive unit tests

**Estimated effort**: The scope is approximately 3-5x larger than the original document suggested.

---

## References

- [OMN-1293 Linear Ticket](https://linear.app/omninode/issue/OMN-1293)
- `omnibase_core` version: Check `pyproject.toml` for current version
- PR under review: `jonah/omn-1102-refactor-noderegistrationorchestrator-to-be-fully`
