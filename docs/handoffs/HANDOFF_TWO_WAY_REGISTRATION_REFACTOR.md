# Handoff: Two-Way Registration Refactoring

**Created**: 2025-12-19
**Ticket**: OMN-889
**Branch**: `jonah/omn-889-dual-registration-reducer-fsm`
**PR**: #52 (open - contains wrong architecture)
**Status**: BLOCKED - Requires complete architectural refactor

---

> **BLOCKER**: This refactor requires `omnibase_core >= 0.5.0` which is NOT YET RELEASED.
>
> **Current Status**: `omnibase_core` 0.5.x release is pending (see [PR #216](https://github.com/OmniNode-ai/omnibase_core/pull/216)).
>
> **What is BLOCKED**:
> - Phase 1-4 implementation (requires `NodeOrchestrator`, `NodeEffect`, `NodeReducer` base classes)
> - Integration with `NodeRuntime` for handler execution
> - Container wiring with actual base classes
>
> **What CAN proceed** (see Phase 0 Contingency Plan):
> - Directory structure and contract.yaml creation
> - Pydantic model definitions
> - Unit tests with mocked base classes
> - Documentation and ADRs
>
> See also: `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md` (OMN-959 tracks dependency update)

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

> **Note on Directory Structure**: The existing `v1_0_0/` directories shown below are LEGACY patterns
> from earlier architectural decisions. Per CLAUDE.md policy "NO VERSIONED DIRECTORIES", new
> components use flat structure. See ticket H1 in `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md`
> for migration plan of existing versioned directories.

```
nodes/node_registry_effect/v1_0_0/    # LEGACY structure (will be migrated per H1)
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

> **State-Reading Invariant**: Orchestrators read current state using projections ONLY
> (via `ProtocolProjectionReader`). Orchestrators NEVER scan topics for state - all state
> decisions are projection-backed.

> **CRITICAL Terminology - Projection "Publishing" vs Event Publishing**:
> Projections are **PERSISTED to storage** (PostgreSQL), NOT **published to Kafka**.
> This distinction is architecturally critical:
> - **Projections**: Written synchronously to storage by the Projector before any Kafka publishing
> - **Events/Intents**: Published to Kafka topics after projection persistence completes
>
> See ticket F0 in `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md` for the authoritative
> definition of projection execution model and the sequence diagram showing this ordering.

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
| `nodes/node_registry_effect/v1_0_0/node.py` | Complete rewrite required (LEGACY path, will migrate to flat structure) |
| `nodes/node_registry_effect/v1_0_0/registry/__init__.py` | Already deleted per git status |

**Note**: Keep the `models/` directory - most models can be reused.

**Migration Note**: The rewritten components will use FLAT directory structure (no `v1_0_0/`).
Version is tracked via `contract_version` field in `contract.yaml`, not directory hierarchy.

---

## 5. Files to Create

### 5.1 New Directory Structure

> **IMPORTANT**: Per CLAUDE.md policy "NO VERSIONED DIRECTORIES", new components use FLAT structure.
> Version is tracked via `contract_version` field in `contract.yaml`, not directory hierarchy.
> See Global Constraint #6 in `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md`.

```
nodes/
├── node_registry_effect/                     # EFFECT - I/O only (FLAT structure)
│   ├── __init__.py
│   ├── node.py                               # ~200-300 lines (lean)
│   ├── contract.yaml                         # Updated for intent consumption
│   │                                         # (includes contract_version: "1.0.0")
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── handler_consul_intent.py          # Consul intent executor
│   │   └── handler_postgres_intent.py        # Postgres intent executor
│   └── models/                               # Existing models, mostly reusable
│
├── node_registration_orchestrator/           # NEW - ORCHESTRATOR (FLAT structure)
│   ├── __init__.py
│   ├── node.py                               # Workflow coordination
│   ├── contract.yaml                         # Orchestrator contract
│   │                                         # (includes contract_version: "1.0.0")
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

#### `nodes/node_registration_orchestrator/node.py`

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

#### `nodes/node_registry_effect/node.py` (Rewritten)

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

### Phase 0: Dependency Verification (Before Phase 1)

**Task 0: Verify omnibase_core 0.5.x Availability**

Before starting any implementation work, verify that `omnibase_core >= 0.5.0` is available and
compatible with this project:

1. **Check Release Status**: Verify [omnibase_core PR #216](https://github.com/OmniNode-ai/omnibase_core/pull/216) is merged and 0.5.x is released
2. **Update pyproject.toml**: Change `omnibase-core = "^0.4.0"` to `omnibase-core = "^0.5.0"`
3. **Install and Verify**: Run `poetry update omnibase-core` and verify installation succeeds
4. **Validate Imports**: Confirm required imports are available:
   ```python
   from omnibase_core.nodes import NodeOrchestrator, NodeEffect, NodeReducer
   from omnibase_core.models.intent import ModelIntent
   from omnibase_core.runtime import NodeRuntime
   ```
5. **Run Existing Tests**: Ensure no regressions with `pytest tests/`

**Exit Criteria**: All imports resolve, existing tests pass, no breaking changes detected.

#### Contingency Plan: If omnibase_core 0.5.x is Delayed

If `omnibase_core 0.5.x` is not available when Phase 1 is scheduled to begin:

**Work That CAN Proceed in Parallel (No 0.5.x Dependency)**:
- [ ] Create orchestrator directory structure and contract.yaml
- [ ] Define command and workflow state models (Pydantic models are independent)
- [ ] Write orchestrator unit tests with mocked base classes
- [ ] Draft intent topic naming conventions and Kafka configuration
- [ ] Write integration test scaffolding with mock handlers
- [ ] Document architectural decisions in ADRs

**Work That is BLOCKED (Requires 0.5.x)**:
- [ ] Implement `NodeRegistrationOrchestrator` class (requires `NodeOrchestrator` base class)
- [ ] Implement rewritten `NodeRegistryEffect` class (requires `NodeEffect` base class)
- [ ] Integrate with `NodeRuntime` for handler execution
- [ ] Wire components in `container_wiring.py` with actual base classes

**Escalation Timeline**:
- **Day 0**: Phase 1 start date arrives, omnibase_core 0.5.x not available
- **Day 1-2**: Begin parallel work (directory structure, models, mocked tests)
- **Day 3**: If still unavailable, escalate to Tech Lead with status report
- **Day 5**: If still unavailable, schedule meeting with omnibase_core team
- **Day 7+**: If delay exceeds 1 week, activate fallback option below

**Mitigation Actions**:
1. **Escalate**: If 0.5.x is not available within 3 days of Phase 1 start date, escalate to Tech Lead
   with written status report including: current blocker, parallel work completed, revised timeline
2. **Coordinate**: Work with omnibase_core team to understand blockers and get firm timeline
3. **Fallback Option**: Consider using stub base classes with TODO markers for development (not production)
   - Create minimal `NodeOrchestrator`, `NodeEffect` protocol stubs in `src/omnibase_infra/stubs/`
   - Mark with `# TODO: Replace with omnibase_core >= 0.5.0 imports when available`
   - Allows development to continue while waiting for upstream release
   - **WARNING**: Stub implementations must NOT be deployed to production

---

### Phase 1: Create Orchestrator (3-4 days)

> **GATE**: Task 1 below is a BLOCKING gate. Do not proceed to Task 2+ until Task 1 passes.

1. **[GATE] Verify omnibase_core 0.5.x availability and run integration tests**
   - Confirm Phase 0 exit criteria met: imports resolve, tests pass, no breaking changes
   - Run `poetry update omnibase-core && pytest tests/`
   - If this task fails, Phase 1 is BLOCKED - see Phase 0 contingency plan
2. Create `nodes/node_registration_orchestrator/` directory structure (FLAT, no v1_0_0)
3. Implement `ModelRegisterNodeCommand` and workflow state models
4. Implement orchestrator contract.yaml
5. Implement `NodeRegistrationOrchestrator`:
   - Command consumption from event bus
   - Reducer invocation
   - Intent publishing
   - Workflow state tracking
6. Write unit tests for orchestrator (no I/O mocking needed - it's pure)

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

### Version Requirements

> **BLOCKER**: This refactor requires `omnibase_core >= 0.5.0` which is **NOT YET RELEASED**.
>
> **Current State**:
> - Project currently uses: `omnibase-core = "^0.4.0"` (in pyproject.toml)
> - Required version: `omnibase-core = "^0.5.0"`
>
> **Release Status**: `omnibase_core` 0.5.x release is pending
> (see [PR #216](https://github.com/OmniNode-ai/omnibase_core/pull/216)).
>
> **Action Required**: Complete Phase 0 (Dependency Verification) before starting Phase 1.
> See the BLOCKER notice at the top of this document for details on what can proceed vs what is blocked.

| Package | Required Version | Current Project Version | Action Required |
|---------|------------------|------------------------|-----------------|
| `omnibase_core` | `>= 0.5.0` | `^0.4.0` | Update to `^0.5.0` when released |
| `omnibase_spi` | `>= 0.4.0` | `^0.4.0` | Already compatible |

### Required Before Starting

| Dependency | Status | Notes |
|------------|--------|-------|
| `omnibase_core.nodes.NodeOrchestrator` | Requires 0.5.x | Base class for orchestrator |
| `omnibase_core.nodes.NodeEffect` | Requires 0.5.x | Base class for effect |
| `omnibase_core.nodes.NodeReducer` | Requires 0.5.x | Base class for reducer |
| `omnibase_core.runtime.NodeRuntime` | Requires 0.5.x | Handler execution runtime |
| Intent models in `omnibase_core.models.intents` | Requires 0.5.x | `ModelConsulRegisterIntent`, etc. |
| FSM contract loader | Available (0.4.x) | Already in reducer |

### Target Import Paths

The following import paths are the **target** paths once `omnibase_core >= 0.5.0` is available:

```python
# Base Classes (available in omnibase_core >= 0.5.0)
from omnibase_core.nodes import NodeOrchestrator, NodeEffect, NodeReducer

# Intent Models (available in omnibase_core >= 0.5.0)
from omnibase_core.models.intent import ModelIntent
from omnibase_core.models.intents import (
    ModelConsulRegisterIntent,
    ModelPostgresUpsertRegistrationIntent,
)

# Runtime (available in omnibase_core >= 0.5.0)
from omnibase_core.runtime import NodeRuntime

# Protocols (from SPI, available in omnibase_spi >= 0.4.0)
from omnibase_spi.protocols import ProtocolIntentHandler, ProtocolNodeRuntime
```

> **Note**: In the current 0.4.x codebase, legacy node classes exist as `NodeEffectLegacy`,
> `NodeReducerLegacy`, etc. These will be renamed/refactored in 0.5.x per
> `docs/architecture/DECLARATIVE_EFFECT_NODES_PLAN.md`. Do not build on the legacy classes.

### Expected Method Signatures

For protocol documentation, see `omnibase_spi/protocols/`. Key signatures:

```python
# NodeRuntime handler execution
async def execute_handler(
    self,
    handler_type: str,
    operation: str,
    payload: dict[str, Any],
) -> ModelIntentResult: ...

# Intent handler protocol
async def handle(
    self,
    intent: ModelIntent,
    context: ModelExecutionContext,
) -> ModelIntentResult: ...
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

## 9. Open Questions and Decision Process

### 9.1 Decisions Required Before Phase 1

These decisions block Phase 1 implementation and must be resolved first.

#### Blocking Questions

> **BLOCKING**: The following questions MUST be resolved before Phase 1 implementation can begin.
> No development work on Phase 1 tasks should start until these decisions are documented as ADRs.

1. **[BLOCKING] Command Source**: Where do `RegisterNodeCommand`s originate?
   - Option A: API endpoint creates commands
   - Option B: Introspection events auto-generate commands
   - Option C: Both (API for manual, introspection for automatic)
   - **Blocking**: Needed for orchestrator contract design

2. **[BLOCKING] Intent Topics**: Topic naming convention for intents?
   - Proposed: `dev.omnibase-infra.intent.<backend>.<operation>.v1`
   - Example: `dev.omnibase-infra.intent.consul.register.v1`
   - **Blocking**: Needed for reducer integration

3. **[BLOCKING] Reducer Invocation**: Does orchestrator call reducer directly or via event?
   - Direct call is simpler for MVP
   - Event-based is more decoupled
   - **Blocking**: Affects orchestrator implementation

#### Decision RACI Matrix

| Decision | Responsible | Accountable | Consulted | Informed | Target Date |
|----------|-------------|-------------|-----------|----------|-------------|
| Command Source | [To be assigned by Tech Lead] | [To be assigned by Tech Lead] | Platform Team | All Devs | [To be assigned by Tech Lead] |
| Intent Topics | [To be assigned by Tech Lead] | [To be assigned by Tech Lead] | Platform Team | All Devs | [To be assigned by Tech Lead] |
| Reducer Invocation | [To be assigned by Tech Lead] | [To be assigned by Tech Lead] | Platform Team | All Devs | [To be assigned by Tech Lead] |

> **Note**: Specific names and dates must be assigned before Phase 1 starts. The Tech Lead
> is responsible for populating the "Responsible", "Accountable", and "Target Date" columns
> during the pre-implementation planning meeting.

#### Pre-Implementation Meeting Requirements

A decision-review meeting with Tech Lead and Platform Team MUST be scheduled BEFORE Phase 1 begins.

**Meeting Scheduling**:
- Schedule meeting at least 3 business days before planned Phase 1 start date
- Attendees: Tech Lead (required), Platform Team representative (required), Lead Developer (required)
- Duration: 1-2 hours

**Meeting Agenda**:
1. Review all three BLOCKING questions (Command Source, Intent Topics, Reducer Invocation)
2. Select final options for each decision with documented rationale
3. Validate omnibase_core 0.5.x availability status (Phase 0 verification complete)
4. Document decisions as ADRs under `docs/adr/`
5. Assign implementation owners for each Phase 1 task
6. Populate RACI matrix with specific names and target dates
7. Confirm timeline and resource availability

**Meeting Output Requirements**:
- All three BLOCKING decisions documented in ADRs
- RACI matrix fully populated with names and dates
- Phase 0 verification confirmed complete
- Phase 1 kickoff formally approved
- Implementation owners assigned and acknowledged

**Exit Criteria**: Phase 1 may NOT begin until all meeting output requirements are satisfied.

### 9.2 Deferrable Questions

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
| Phase 0: Dependency Verification | 1 day | omnibase_core 0.5.x release |
| Phase 1: Create Orchestrator | 3-4 days | Phase 0 complete |
| Phase 2: Rewrite Effect Node | 2-3 days | Phase 1 |
| Phase 3: Wire Integration | 2 days | Phase 1, 2 |
| Phase 4: Validation/Cleanup | 2 days | Phase 3 |
| **Total** | **10-12 days** | |

**Risk Factors**:
- **omnibase_core 0.5.x delay**: See Phase 0 contingency plan for mitigation
- **Decision resolution delay**: Open questions (Section 9.1) may add 1-2 days if decisions require discussion
- **Timeline assumption**: Timeline assumes blocking decisions (Section 9.1) are resolved before Phase 1 start
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

### Current Implementation (Wrong - LEGACY path)
- `src/omnibase_infra/nodes/node_registry_effect/v1_0_0/node.py` - 3,065 lines
  (This is a LEGACY versioned directory path. The rewritten component will use flat structure:
  `src/omnibase_infra/nodes/node_registry_effect/node.py`)

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
