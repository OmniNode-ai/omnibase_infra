# Handoff: Two-Way Registration Refactoring

**Created**: 2025-12-19
**Ticket**: OMN-889
**Branch**: `jonah/omn-889-dual-registration-reducer-fsm`
**PR**: #52 (open - contains wrong architecture)
**Status**: BLOCKED - Requires complete architectural refactor

---

## 1. Executive Summary

The current `NodeRegistryEffect` implementation (3,065 lines) completely violates the ONEX 4-node architecture. It combines orchestration, reducer, and effect responsibilities into a single monolithic file, making it untestable, unmaintainable, and architecturally unsound.

**Current State**: A 3,000+ line Effect node that:
- Directly calls handlers (should be via runtime)
- Mixes I/O with business logic (violates purity)
- Has no proper command/event separation
- Combines orchestrator and reducer logic
- Manages state ad-hoc instead of via event projections

**Target State**: Three distinct components following ONEX architecture:
1. **NodeDualRegistrationReducer** (PURE) - Emits typed intents from events
2. **NodeRegistrationOrchestrator** (PURE) - Consumes commands, coordinates workflow
3. **NodeRegistryEffect** (I/O) - Executes intents via handlers registered with runtime

**Impact**: This refactor affects the core registration pattern used by all ONEX nodes for service discovery.

---

## 2. Current State Analysis

### 2.1 What Exists Today

```
nodes/node_registry_effect/v1_0_0/
    node.py                    # 3,065 lines - THE PROBLEM
    contract.yaml              # 92 lines
    models/                    # Various request/response models
    protocols.py               # Handler protocols

nodes/reducers/
    node_dual_registration_reducer.py  # 887 lines - Correct reducer (partially)
    enums/                     # FSM enums
    models/                    # FSM models
```

### 2.2 What's Wrong with node.py

| Violation | Description | Lines Affected |
|-----------|-------------|----------------|
| **Mixed I/O + Business Logic** | `_register_consul()` and `_register_postgres()` directly call handlers | 1990-2240 |
| **Orchestration in Effect** | `execute()` routes operations, manages FSM-like flow | 1698-1907 |
| **Ad-hoc State Management** | Circuit breaker state, operation tracking without proper FSM | Throughout |
| **No Command/Event Separation** | `ModelRegistryRequest` conflates commands with data | Models |
| **Direct Handler Calls** | `self.consul_handler.execute()` bypasses runtime | 2013-2019 |
| **Validation Mixed with Execution** | Security validation interleaved with business logic | 1093-1304 |
| **Massive Method Count** | 35+ methods in a single class | Full file |

### 2.3 Existing Reducer (Partially Correct)

The `NodeDualRegistrationReducer` at `nodes/reducers/node_dual_registration_reducer.py` is mostly correct:

**Correct:**
- Pure reducer architecture (no I/O)
- Emits typed intents (`ModelConsulRegisterIntent`, `ModelPostgresUpsertRegistrationIntent`)
- FSM-driven state machine from YAML contract
- Deterministic - same inputs produce same outputs

**Missing:**
- Not integrated with the Effect node
- Effect node doesn't consume its intents
- No orchestrator to coordinate the workflow

---

## 3. Target Architecture

### 3.1 Correct ONEX 4-Node Pattern

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                      Kafka Event Bus                        │
                    └─────────────────────────────────────────────────────────────┘
                           │                                        ▲
                           │ *.introspection.* events               │ *.registration.* events
                           ▼                                        │
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                        NodeRegistrationOrchestrator (PURE)                        │
    │  - Consumes commands (RegisterNodeCommand)                                        │
    │  - Validates workflow preconditions                                               │
    │  - Coordinates Reducer invocation                                                 │
    │  - Tracks workflow state via FSM                                                  │
    │  - Emits workflow events (RegistrationStarted, RegistrationCompleted)             │
    │  - Handles timeouts and retries at workflow level                                 │
    └───────────────────────────────────────┬──────────────────────────────────────────┘
                                            │
                                            │ Invokes
                                            ▼
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                       NodeDualRegistrationReducer (PURE)                          │
    │  - Receives NODE_INTROSPECTION events                                             │
    │  - Validates event payload                                                        │
    │  - Emits typed intents:                                                           │
    │    * ModelConsulRegisterIntent                                                    │
    │    * ModelPostgresUpsertRegistrationIntent                                        │
    │  - FSM-driven state machine (contracts/fsm/dual_registration_reducer_fsm.yaml)    │
    │  - NO I/O - purely computes and emits                                             │
    └───────────────────────────────────────┬──────────────────────────────────────────┘
                                            │
                                            │ Intents published to event bus
                                            ▼
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                          NodeRegistryEffect (I/O)                                 │
    │  - Subscribes to intent topics                                                    │
    │  - Executes intents via handlers registered with NodeRuntime:                     │
    │    * ConsulHandler (for ModelConsulRegisterIntent)                                │
    │    * PostgresHandler (for ModelPostgresUpsertRegistrationIntent)                  │
    │  - Circuit breaker protection                                                     │
    │  - Emits result events back to event bus                                          │
    │  - NO business logic - pure I/O execution                                         │
    └──────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            │ Handler calls via NodeRuntime
                                            ▼
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                              NodeRuntime (omnibase_core)                          │
    │  - Single event loop per host                                                     │
    │  - Manages handler lifecycle                                                      │
    │  - Routes intents to appropriate handlers                                         │
    │  - Provides circuit breaker, retry, timeout at infrastructure level               │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Architectural Invariants

From `docs/architecture/DECLARATIVE_EFFECT_NODES_PLAN.md`:

1. **NodeRuntime is the only executable event loop** - Effect nodes don't run their own loops
2. **Node logic is pure: no I/O, no mixins, no inheritance** (except base classes)
3. **Core never depends on SPI or infra** - Dependency: infra -> spi -> core
4. **SPI only defines protocols, never implementations**
5. **Infra owns all I/O and real system integrations** - Handlers live in infra
6. **Contract-driven behavior** - YAML contracts define everything

### 3.3 Intent-Based Communication

Instead of direct handler calls:

```python
# WRONG (current)
result = await self.consul_handler.execute({...})

# CORRECT (target)
# Reducer emits intent:
intent = ModelConsulRegisterIntent(
    correlation_id=correlation_id,
    service_id=service_id,
    service_name=service_name,
    tags=tags,
    health_check=health_check,
)

# Effect node receives intent from event bus and executes:
async def handle_consul_register(self, intent: ModelConsulRegisterIntent) -> None:
    result = await self.runtime.execute_handler(
        handler_type="consul",
        operation="register",
        payload=intent.model_dump(),
    )
```

---

## 4. Files to Delete

| File | Reason |
|------|--------|
| `nodes/node_registry_effect/v1_0_0/node.py` | Complete rewrite required |
| `nodes/node_registry_effect/v1_0_0/registry/__init__.py` | Already deleted per git status |

**Note**: Keep the `models/` directory - most models can be reused.

---

## 5. Files to Create

### 5.1 New Directory Structure

```
nodes/
├── node_registry_effect/v1_0_0/              # EFFECT - I/O only
│   ├── __init__.py
│   ├── node.py                               # ~200-300 lines (lean)
│   ├── contract.yaml                         # Updated for intent consumption
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── handler_consul_intent.py          # Consul intent executor
│   │   └── handler_postgres_intent.py        # Postgres intent executor
│   └── models/                               # Existing models, mostly reusable
│
├── node_registration_orchestrator/v1_0_0/    # NEW - ORCHESTRATOR
│   ├── __init__.py
│   ├── node.py                               # Workflow coordination
│   ├── contract.yaml                         # Orchestrator contract
│   └── models/
│       ├── __init__.py
│       ├── model_register_node_command.py    # Command model
│       └── model_registration_workflow_state.py
│
└── reducers/                                 # EXISTING - enhance
    ├── node_dual_registration_reducer.py     # Already correct
    └── ...
```

### 5.2 File Specifications

#### `nodes/node_registration_orchestrator/v1_0_0/node.py`

```python
"""Registration Orchestrator for two-way node registration workflow.

Coordinates the registration workflow by:
1. Receiving RegisterNodeCommand from event bus
2. Validating workflow preconditions
3. Invoking NodeDualRegistrationReducer
4. Publishing intents to event bus for Effect execution
5. Tracking workflow state and handling failures
"""

class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Pure orchestrator - no I/O, coordinates workflow via events."""

    async def process_command(
        self,
        command: ModelRegisterNodeCommand,
    ) -> ModelRegistrationWorkflowState:
        # 1. Validate preconditions
        # 2. Convert command to introspection event
        # 3. Call reducer.execute(event)
        # 4. Publish intents from reducer output
        # 5. Return workflow state
        pass
```

#### `nodes/node_registry_effect/v1_0_0/node.py` (Rewritten)

```python
"""Registry Effect Node - Pure I/O execution for registration intents.

This effect node:
1. Subscribes to intent topics (consul.register, postgres.upsert)
2. Executes intents via handlers registered with NodeRuntime
3. Emits result events to event bus
4. NO business logic - pure intent execution
"""

class NodeRegistryEffect(NodeEffect):
    """Lean effect node - executes intents via runtime handlers."""

    async def handle_consul_register(
        self,
        intent: ModelConsulRegisterIntent,
    ) -> ModelIntentResult:
        # Execute via runtime-registered handler
        return await self.runtime.execute_handler(
            "consul", "register", intent.model_dump()
        )

    async def handle_postgres_upsert(
        self,
        intent: ModelPostgresUpsertRegistrationIntent,
    ) -> ModelIntentResult:
        # Execute via runtime-registered handler
        return await self.runtime.execute_handler(
            "postgres", "upsert", intent.model_dump()
        )
```

---

## 6. Migration Steps

### Phase 1: Create Orchestrator (3-4 days)

1. Create `nodes/node_registration_orchestrator/v1_0_0/` directory structure
2. Implement `ModelRegisterNodeCommand` and workflow state models
3. Implement orchestrator contract.yaml
4. Implement `NodeRegistrationOrchestrator`:
   - Command consumption from event bus
   - Reducer invocation
   - Intent publishing
   - Workflow state tracking
5. Write unit tests for orchestrator (no I/O mocking needed - it's pure)

### Phase 2: Rewrite Effect Node (2-3 days)

1. Create new lean `node.py` (~200-300 lines)
2. Implement intent handlers:
   - `handle_consul_register()`
   - `handle_postgres_upsert()`
   - `handle_deregister()` (if needed)
3. Integrate with NodeRuntime for handler execution
4. Keep circuit breaker as infrastructure concern
5. Update `contract.yaml` for intent consumption

### Phase 3: Wire Integration (2 days)

1. Update `container_wiring.py` to register new components
2. Configure Kafka topics for command/intent/result flow
3. Integrate orchestrator with existing introspection event flow
4. Update tests to use new architecture

### Phase 4: Validation and Cleanup (2 days)

1. Run full integration tests
2. Validate FSM transitions
3. Remove old code paths
4. Update documentation

---

## 7. Dependencies

### Required Before Starting

| Dependency | Status | Notes |
|------------|--------|-------|
| `omnibase_core.nodes.NodeOrchestrator` | Available | Base class for orchestrator |
| `omnibase_core.nodes.NodeEffect` | Available | Base class for effect |
| `omnibase_core.nodes.NodeReducer` | Available | Base class for reducer |
| `NodeRuntime` | Available | Handler execution runtime |
| Intent models in `omnibase_core.models.intents` | Available | `ModelConsulRegisterIntent`, etc. |
| FSM contract loader | Available | Already in reducer |

### Import Paths

```python
from omnibase_core.nodes import NodeOrchestrator, NodeEffect, NodeReducer
from omnibase_core.models.intent import ModelIntent
```

### External Services

- Kafka event bus configured for command/intent/result topics
- Consul running for registration target
- PostgreSQL with `node_registrations` table

---

## 8. Testing Requirements

### 8.1 Unit Tests (No I/O)

| Component | Test Focus |
|-----------|------------|
| `NodeRegistrationOrchestrator` | Command validation, reducer invocation, workflow state |
| `NodeDualRegistrationReducer` | Intent emission, FSM transitions, validation |
| Intent handlers | Intent parsing, handler invocation |

### 8.2 Integration Tests

| Test | Description |
|------|-------------|
| Full registration flow | Command -> Orchestrator -> Reducer -> Effect -> Handlers |
| Partial failure handling | One backend fails, other succeeds |
| Circuit breaker | Effect-level circuit breaker behavior |
| Idempotency | Re-registration produces consistent state |

### 8.3 Contract Tests

- FSM contract (`dual_registration_reducer_fsm.yaml`) validation
- Orchestrator contract validation
- Effect contract validation

---

## 9. Open Questions

### Decisions Required Before Phase 1

These decisions block Phase 1 implementation and must be resolved first:

1. **Command Source**: Where do `RegisterNodeCommand`s originate?
   - Option A: API endpoint creates commands
   - Option B: Introspection events auto-generate commands
   - Option C: Both (API for manual, introspection for automatic)
   - **Blocking**: Needed for orchestrator contract design

2. **Intent Topics**: Topic naming convention for intents?
   - Proposed: `dev.omnibase-infra.intent.<backend>.<operation>.v1`
   - Example: `dev.omnibase-infra.intent.consul.register.v1`
   - **Blocking**: Needed for reducer integration

3. **Reducer Invocation**: Does orchestrator call reducer directly or via event?
   - Direct call is simpler for MVP
   - Event-based is more decoupled
   - **Blocking**: Affects orchestrator implementation

### Can Be Deferred

These questions can be answered during implementation:

4. **Result Events**: How do result events flow back to orchestrator?
   - Option A: Orchestrator subscribes to result topics
   - Option B: Effect publishes completion events, separate aggregator handles

5. **Timeout Handling**: Where does timeout tracking live?
   - Current: Effect node has `_slow_operation_threshold_ms`
   - Proposed: Orchestrator tracks workflow-level timeouts

6. **Existing Tests**: What to do with `tests/unit/nodes/test_node_registry_effect.py`?
   - Rewrite for new architecture
   - Many tests are testing wrong patterns

7. **Handler Registration**: When are handlers registered with NodeRuntime?
   - Container wiring phase?
   - Effect node initialization?

---

## 10. Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Create Orchestrator | 3-4 days | None |
| Phase 2: Rewrite Effect Node | 2-3 days | Phase 1 |
| Phase 3: Wire Integration | 2 days | Phase 1, 2 |
| Phase 4: Validation/Cleanup | 2 days | Phase 3 |
| **Total** | **9-11 days** | |

**Risk Factors**:
- Open questions may add 1-2 days if decisions require discussion
- Integration issues with NodeRuntime may surface
- Existing test refactoring may take longer than estimated

---

## 11. PR #52 Disposition

**Current PR #52** contains the wrong architecture and should be:

1. **Close PR #52** - The current implementation violates ONEX architecture
2. **Create new branch** - Start fresh with correct architecture
3. **Reference this handoff** - New PR should link to this document

**Do NOT merge PR #52** - it will create technical debt that requires immediate rework.

---

## 12. Related Documents

- `docs/architecture/DECLARATIVE_EFFECT_NODES_PLAN.md`
- `docs/architecture/CURRENT_NODE_ARCHITECTURE.md`
- `contracts/fsm/dual_registration_reducer_fsm.yaml`
- `omnibase_core` documentation for NodeRuntime, NodeOrchestrator, NodeEffect

---

## 13. Code References

### Current Implementation (Wrong)
- `src/omnibase_infra/nodes/node_registry_effect/v1_0_0/node.py` - 3,065 lines

### Correct Reducer (Reusable)
- `src/omnibase_infra/nodes/reducers/node_dual_registration_reducer.py` - 887 lines

### Intent Models (Reusable)
- `omnibase_core/models/intents/model_consul_register_intent.py`
- `omnibase_core/models/intents/model_postgres_upsert_registration_intent.py`

---

## Appendix A: Current node.py Method Inventory

The current `node.py` has 35+ methods that need to be distributed:

**Move to Orchestrator:**
- `execute()` routing logic
- Operation dispatch (`_register_node`, `_deregister_node`, `_discover_nodes`)
- Workflow coordination

**Move to Effect (Intent Handlers):**
- `_register_consul()` -> `handle_consul_register()`
- `_register_postgres()` -> `handle_postgres_upsert()`
- `_deregister_consul()` -> `handle_consul_deregister()`
- `_deregister_postgres()` -> `handle_postgres_delete()`

**Keep in Effect (Infrastructure):**
- Circuit breaker methods
- Error sanitization (`_sanitize_error()`, `_redact_sensitive_patterns()`)
- JSON serialization helpers

**Keep in Reducer (Already Correct):**
- `_build_consul_intent()`
- `_build_postgres_intent()`
- FSM transitions
- Payload validation

**Delete (Obsolete):**
- Direct handler invocation code
- Mixed validation/execution methods
- Ad-hoc state management
