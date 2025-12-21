# Handoff: Two-Way Registration Refactoring

**Created**: 2025-12-19
**Ticket**: OMN-889
**Branch**: `jonah/omn-889-dual-registration-reducer-fsm`
**PR**: #52 (open - contains wrong architecture)
**Status**: BLOCKED - Requires complete architectural refactor

---

> ## BLOCKER - READ THIS FIRST
>
> **This refactor CANNOT proceed until `omnibase_core >= 0.5.0` is released.**
>
> | Aspect | Details |
> |--------|---------|
> | **Required Version** | `omnibase_core >= 0.5.0` |
> | **Current Version** | `omnibase_core ^0.4.0` |
> | **Release Status** | PENDING - see [PR #216](https://github.com/OmniNode-ai/omnibase_core/pull/216) |
> | **Tracking Ticket** | OMN-959 |
>
> **What is BLOCKED** (requires 0.5.x base classes):
> - Phase 1-4 implementation (`NodeOrchestrator`, `NodeEffect`, `NodeReducer` base classes)
> - Integration with `NodeRuntime` for handler execution
> - Container wiring with actual base classes
>
> **What CAN proceed** (see Phase 0 Contingency Plan in Section 6):
> - Directory structure and contract.yaml creation
> - Pydantic model definitions
> - Unit tests with mocked base classes
> - Documentation and ADRs
>
> **Action Required**: Verify 0.5.x availability BEFORE starting Phase 1 (see Phase 0 Task 0 gate).
>
> See also:
> - `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md` (Document Version 1.1.0)
> - `docs/design/DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md` (Version 2.1.2)

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

> ## ARCHITECTURAL DISTINCTION: Projection Persistence vs Event Publishing
>
> This document uses two distinct concepts that must not be confused:
>
> | Term | Target | Timing | Purpose |
> |------|--------|--------|---------|
> | **Projection Persistence** | PostgreSQL (storage) | SYNCHRONOUS - before Kafka publish | State materialization for queries |
> | **Event/Intent Publishing** | Kafka (event bus) | AFTER projection persistence | Downstream workflow triggers |
>
> **Critical Invariant**: The Runtime ALWAYS persists projections to storage BEFORE publishing
> events/intents to Kafka. This ordering guarantee is enforced by the F0 Projector execution model.
>
> See Section 3.1 for detailed architectural diagrams and ticket F0 in the design doc for
> the authoritative projection execution model.

---

## 2. Current State Analysis

### 2.1 What Exists Today

> **Note on Directory Structure**: The existing `v1_0_0/` directories shown below are LEGACY patterns
> from earlier architectural decisions. Per CLAUDE.md policy "NO VERSIONED DIRECTORIES", new
> components use flat structure. See ticket H1 (OMN-956) in `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md`
> for migration plan of existing versioned directories. This aligns with Global Constraint #6 in the ticket plan.

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
>
> **Implementation Note**: The orchestrator resolves `ProtocolProjectionReader` from the
> container at initialization. State queries (e.g., "is this node already registered?")
> are synchronous reads against the projection store (PostgreSQL), not Kafka topic scans.
> See `omnibase_spi/protocols/protocol_projection_reader.py` for the protocol definition.
>
> **Orchestrator State-Reading Path**: The state-reading path is:
> 1. Orchestrator receives event or RuntimeTick
> 2. Orchestrator invokes `projection_reader.get_projection(entity_id)`
> 3. Projection reader queries PostgreSQL (or cache layer) synchronously
> 4. Orchestrator uses projection state to make workflow decisions
> 5. Orchestrator emits decision events based on projection state
>
> Orchestrators NEVER scan Kafka topics for state. All state is materialized in projections
> and queried via the projection reader protocol.

> **CRITICAL Terminology - Projection "Persistence" vs Event Publishing**:
> Projections are **PERSISTED to storage** (PostgreSQL), NOT **published to Kafka**.
> This distinction is architecturally critical:
> - **Projections**: Written synchronously to storage by the Projector BEFORE any Kafka publishing
> - **Events/Intents**: Published to Kafka topics AFTER projection persistence completes
> - **Ordering Guarantee**: Runtime invokes Projector.persist() and waits for acknowledgment
>   before publishing intents to Kafka (see F0 ↔ B2 interaction sequence diagram in design doc)
>
> See ticket F0 in `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md` for the authoritative
> definition of projection execution model and the F0 ↔ B2 interaction sequence diagram showing
> the synchronization between Handler Output Model (B2) and Projector Execution (F0).
>
> See also ticket A2a in the design doc for canonical envelope usage patterns across all
> architectural planes (Ingestion, Decision, State, Execution).

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

> **Terminology Note** (per Global Constraint #7 in `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md`):
> - **Node**: The deployable/addressable unit that hosts one or more handlers
> - **Handler**: A pure function that processes a specific message type within a node
> - **Runtime**: Infrastructure that dispatches messages to handlers and publishes outputs
>
> Usage rules:
> - Use "handler" when referring to message processing logic
> - Use "node" when referring to deployment, lifecycle, or identity
> - **Never say "handler publishes"** - only runtime publishes
> - **Never say "node processes messages"** - handlers process, nodes host

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
> Migration of existing versioned directories is tracked by ticket H1 (OMN-956).

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

> **GATE TASKS**: Tasks 0 and 1 below are BLOCKING gates. Do not proceed to Task 2+ until both gates pass.
> If either gate fails, Phase 1 is BLOCKED - see Phase 0 contingency plan.
>
> **Task 0 is the CRITICAL DEPENDENCY GATE**: This task verifies that omnibase_core 0.5.x is available.
> Without 0.5.x, Phase 1 cannot proceed. See Phase 0 "Contingency Plan: If omnibase_core 0.5.x is Delayed"
> for mitigation options if this gate fails.

0. **[GATE - DEPENDENCY] Verify omnibase_core 0.5.x is available and installable**
   - Check that [PR #216](https://github.com/OmniNode-ai/omnibase_core/pull/216) is merged
   - Verify 0.5.x version is published to PyPI or internal registry
   - Update `pyproject.toml`: change `omnibase-core = "^0.4.0"` to `omnibase-core = "^0.5.0"`
   - Run `poetry update omnibase-core` - must succeed without errors
   - Validate imports:
     ```python
     from omnibase_core.nodes import NodeOrchestrator, NodeEffect, NodeReducer
     from omnibase_core.runtime import NodeRuntime
     ```
   - **Exit Criteria**: Imports resolve, poetry install succeeds
   - **If FAILED**: Phase 1 is BLOCKED - activate Phase 0 contingency plan (see "Contingency Plan: If omnibase_core 0.5.x is Delayed" in Phase 0 section above)

1. **[GATE - VALIDATION] Run integration tests with new dependency**
   - Run `pytest tests/` - all existing tests must pass
   - Confirm no breaking changes from 0.4.x to 0.5.x
   - Document any required migration steps
   - **Exit Criteria**: All tests pass, no regressions detected
   - **If FAILED**: Phase 1 is BLOCKED - coordinate with omnibase_core team
   - **Note**: This gate verifies v0.5.0 compatibility and must pass before proceeding to Task 2+

2. Create `nodes/node_registration_orchestrator/` directory structure (FLAT, no v1_0_0)
3. Implement `ModelRegisterNodeCommand` and workflow state models
4. Implement orchestrator contract.yaml
5. Implement `NodeRegistrationOrchestrator`:
   - Command consumption from event bus
   - Reducer invocation
   - Intent publishing
   - Workflow state tracking
6. Write unit tests for orchestrator (no I/O mocking needed - it's pure)

> **Note**: Task numbering starts at 0 to emphasize the dependency verification gate.
> Total Phase 1 tasks: 7 (Tasks 0-6).

### Phase 2: Rewrite Effect Node (2-3 days)

1. Create new lean `node.py` (~200-300 lines)
2. Implement intent handlers:
   - `handle_consul_register()`
   - `handle_postgres_upsert()`
   - `handle_deregister()` (if needed)
3. Integrate with NodeRuntime for handler execution
4. Keep circuit breaker as infrastructure concern (see CLAUDE.md "Circuit Breaker Pattern" section)
5. Update `contract.yaml` for intent consumption
6. **Error Handling Requirements** (see CLAUDE.md "Error Sanitization Guidelines"):
   - All errors MUST be sanitized before logging (no secrets, credentials, PII)
   - Use transport-aware error codes from `EnumCoreErrorCode`
   - Propagate `correlation_id` from intent envelope to all error contexts
   - Safe to include: service names, operation names, correlation IDs, error codes
   - NEVER include: passwords, API keys, tokens, connection strings with credentials

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

### Base Class Version Requirements

The following base classes are introduced in `omnibase_core >= 0.5.0` and are REQUIRED
for this refactor. These classes do not exist in 0.4.x:

| Base Class | Minimum Version | Purpose |
|------------|-----------------|---------|
| `NodeOrchestrator` | 0.5.0 | Base class for workflow coordination nodes |
| `NodeEffect` | 0.5.0 | Base class for I/O execution nodes |
| `NodeReducer` | 0.5.0 | Base class for pure transformation nodes |
| `NodeRuntime` | 0.5.0 | Handler execution and lifecycle management |
| `ModelIntent` | 0.5.0 | Base model for typed intent messages |

> **Note**: The 0.4.x codebase contains `NodeEffectLegacy`, `NodeReducerLegacy` classes.
> These are NOT compatible with this refactor and will be deprecated in 0.5.x.
> Do not extend legacy classes.

**Dependency Version Table** (Minimum Required):

| Component | Package | Version | Purpose |
|-----------|---------|---------|---------|
| `NodeOrchestrator` | `omnibase_core` | 0.5.0 | Workflow coordination base class |
| `NodeEffect` | `omnibase_core` | 0.5.0 | I/O execution base class |
| `NodeReducer` | `omnibase_core` | 0.5.0 | Pure transformation base class |
| `NodeRuntime` | `omnibase_core` | 0.5.0 | Handler lifecycle management |
| `ModelIntent` | `omnibase_core` | 0.5.0 | Typed intent base model |

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

| Component | Test Focus | Acceptance Criteria |
|-----------|------------|---------------------|
| `NodeRegistrationOrchestrator` | Command validation, reducer invocation, workflow state | Acceptance Criteria: Orchestrator tests verify command validation without mocking I/O. All workflow decisions are deterministic given projection state and injected time. |
| `NodeDualRegistrationReducer` | Intent emission, FSM transitions, validation | Acceptance Criteria: Reducer tests verify deterministic intent emission for identical inputs. Same event sequence produces identical projections and intents. |
| Intent handlers | Intent parsing, handler invocation | Acceptance Criteria: Intent handler tests verify correct payload parsing and handler dispatch. Handler invocation matches runtime contract. |

### 8.2 Integration Tests

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| Full registration flow | Command -> Orchestrator -> Reducer -> Effect -> Handlers | Acceptance Criteria: Full registration flow completes with node visible in both Consul and PostgreSQL. End-to-end latency is within performance thresholds. |
| Partial failure handling | One backend fails, other succeeds | Acceptance Criteria: Partial failure test verifies error handling and state consistency. System maintains consistency when Consul succeeds but PostgreSQL fails (or vice versa). |
| Circuit breaker | Effect-level circuit breaker behavior | Acceptance Criteria: Circuit breaker test verifies OPEN state after threshold failures. Subsequent requests fail fast with InfraUnavailableError until reset timeout. |
| Idempotency | Re-registration produces consistent state | Acceptance Criteria: Idempotency test verifies re-registration produces identical state. Duplicate message_id or natural key causes no additional side effects. |

### 8.3 Contract Tests

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| FSM contract validation | `dual_registration_reducer_fsm.yaml` state machine validation | Acceptance Criteria: FSM contract validation passes with all valid state transitions. Invalid transitions are rejected deterministically. |
| Orchestrator contract | Command/event shape validation | Acceptance Criteria: Orchestrator contract validates command/event shapes. All required envelope fields are present and typed correctly. |
| Effect contract | Intent consumption and result emission validation | Acceptance Criteria: Effect contract validates intent consumption and result emission. Intent handlers produce optional result events matching contract. |

---

## 9. Open Questions and Decision Process

### 9.1 Decisions Required Before Phase 1

> **CRITICAL BLOCKERS**: The three decisions below are MANDATORY PREREQUISITES for Phase 1.
> Phase 1 implementation MUST NOT begin until all three decisions are documented as ADRs.
> These are not optional refinements - they are architectural decisions that fundamentally
> shape the implementation. Proceeding without resolution will result in rework.

These decisions block Phase 1 implementation and must be resolved first.

#### Blocking Questions

> **BLOCKING**: The following questions MUST be resolved before Phase 1 implementation can begin.
> No development work on Phase 1 tasks should start until these decisions are documented as ADRs.
>
> **Target Dates Required**: Each decision below MUST have a target completion date assigned
> in the RACI matrix. Target dates MUST be before the Phase 1 start date to avoid blocking
> Wave 1 execution. See "Decision Timeline Guidance" below for recommended scheduling.

1. **[BLOCKING] Command Source**: Where do `RegisterNodeCommand`s originate?
   - Option A: API endpoint creates commands
   - Option B: Introspection events auto-generate commands
   - Option C: Both (API for manual, introspection for automatic)
   - **Blocking**: Needed for orchestrator contract design
   - **Target Date**: [Tech Lead: Assign in RACI matrix - MUST be before Phase 1]
   - **ADR**: Decision will be documented in `docs/adr/ADR-XXX-command-source.md`

2. **[BLOCKING] Intent Topics**: Topic naming convention for intents?
   - Proposed: `dev.omnibase-infra.intent.<backend>.<operation>.v1`
   - Example: `dev/omnibase-infra.intent.consul.register.v1`
   - **Blocking**: Needed for reducer integration
   - **Target Date**: [Tech Lead: Assign in RACI matrix - MUST be before Phase 1]
   - **ADR**: Decision will be documented in `docs/adr/ADR-XXX-intent-topic-naming.md`

3. **[BLOCKING] Reducer Invocation**: Does orchestrator call reducer directly or via event?
   - Direct call is simpler for MVP
   - Event-based is more decoupled
   - **Blocking**: Affects orchestrator implementation
   - **Target Date**: [Tech Lead: Assign in RACI matrix - MUST be before Phase 1]
   - **ADR**: Decision will be documented in `docs/adr/ADR-XXX-reducer-invocation-pattern.md`

#### Decision RACI Matrix

> **TEMPLATE - REQUIRES POPULATION BEFORE EXECUTION**
>
> This RACI matrix is intentionally provided as a TEMPLATE with placeholder values.
> The Tech Lead MUST populate the "Responsible", "Accountable", and "Target Date"
> columns BEFORE the pre-implementation meeting. Phase 1 cannot begin until this
> matrix is complete with specific names and dates.
>
> **Placeholder Format**: Values shown as `[Tech Lead: Assign ...]` indicate fields
> that MUST be replaced with actual names and dates during project planning.
>
> **CRITICAL**: Decision owners MUST be assigned before Wave 1 starts. See "Pre-Implementation
> Meeting Requirements" below for the formal assignment process.

| Decision | Responsible | Accountable | Consulted | Informed | Target Date |
|----------|-------------|-------------|-----------|----------|-------------|
| Command Source | [Tech Lead: Assign name] | [Tech Lead: Assign name] | Platform Team | All Devs | [Tech Lead: Assign date - BEFORE Phase 1] |
| Intent Topics | [Tech Lead: Assign name] | [Tech Lead: Assign name] | Platform Team | All Devs | [Tech Lead: Assign date - BEFORE Phase 1] |
| Reducer Invocation | [Tech Lead: Assign name] | [Tech Lead: Assign name] | Platform Team | All Devs | [Tech Lead: Assign date - BEFORE Phase 1] |

**RACI Matrix Completion Checklist**:
- [ ] Tech Lead has assigned "Responsible" person for each decision
- [ ] Tech Lead has assigned "Accountable" person for each decision
- [ ] Tech Lead has set "Target Date" for each decision (MUST be before Phase 1 start)
- [ ] All assigned persons have acknowledged their roles
- [ ] All target dates are realistic and account for review cycles
- [ ] Escalation contacts identified for each decision area

> **Note**: Specific names and dates must be assigned before Phase 1 starts. The Tech Lead
> is responsible for populating this matrix during the pre-implementation planning meeting.
> Placeholder values `[Tech Lead: Assign ...]` indicate incomplete assignments.
>
> **Decision Owner Assignment Requirement**: All decision owners (Responsible and Accountable)
> MUST be formally assigned and acknowledged before Wave 1 (Phase 1) execution begins. The
> pre-implementation meeting is the mandatory forum for completing these assignments.

#### Escalation Path for Decision Blockers

If blocking decisions (Command Source, Intent Topics, Reducer Invocation) are not
resolved by their target dates, follow this escalation path:

| Days Overdue | Action | Owner | Deliverable |
|--------------|--------|-------|-------------|
| 1-2 days | Daily standup reminder + async Slack ping | Lead Developer | Written status in standup notes |
| 3-4 days | Schedule dedicated decision meeting (30-60 min) | Tech Lead | Meeting invite with agenda |
| 5+ days | Escalate to Engineering Manager, propose default decisions | Tech Lead | Written escalation with options |
| 7+ days | Emergency decision meeting with EM approval for defaults | Engineering Manager | Decision memo + ADR drafts |

**Escalation Communication Templates**:

*Day 3-4 Slack message (to decision owner)*:
> Reminder: Decision for [Command Source/Intent Topics/Reducer Invocation] is now [X] days
> overdue from target date [DATE]. This is blocking Phase 1 of the registration refactor.
> Please confirm if you can provide a decision by EOD tomorrow, or let me know if you need
> a synchronous meeting to discuss options.

*Day 5+ escalation email (to Engineering Manager)*:
> Subject: ESCALATION: Blocking decisions for OMN-889 Registration Refactor
>
> Three architectural decisions are blocking Phase 1 implementation:
> 1. Command Source - [X] days overdue
> 2. Intent Topics - [X] days overdue
> 3. Reducer Invocation - [X] days overdue
>
> Proposed action: Apply default decisions (documented in handoff) and proceed.
> This will allow Phase 1 to start while decisions are refined in parallel.
>
> Please confirm approval to use defaults, or schedule an emergency decision meeting.

**Default Decisions** (if escalation reaches 5+ days):
- Command Source: Option C (Both API + Introspection)
- Intent Topics: Use proposed naming (`dev.omnibase-infra.intent.<backend>.<operation>.v1`)
- Reducer Invocation: Direct call for MVP, migrate to event-based post-MVP

> **Note**: If default decisions are used due to escalation, ADRs should still be created
> documenting the decision rationale as "escalation default - subject to review". This
> ensures traceability and allows for future reconsideration if needed.

**Wave 2 Impact Mitigation**: If decisions block Wave 2 start by more than 1 week, Tech Lead
should evaluate partial Wave 2 execution using default decisions with documented technical
debt tickets for potential refactor. This allows Wave 2 to proceed while decisions are refined
in parallel, preventing cascading schedule impacts.

#### Decision Timeline Guidance

**Recommended Timeline** (relative to planned Phase 1 start date):

| Milestone | Target | Notes |
|-----------|--------|-------|
| RACI Matrix Populated | T-10 days | Tech Lead assigns owners and dates |
| Decision Owners Acknowledged | T-8 days | All assignees confirm availability |
| First Decision Draft | T-5 days | Initial proposals for each decision |
| Pre-Implementation Meeting | T-3 days | Finalize decisions, create ADRs |
| ADRs Committed | T-1 day | All three ADRs merged to main |
| Phase 1 Gate Check | T-0 | Verify all prerequisites met |

**Timeline Rationale**: This schedule provides:
- 5 days for decision drafting and review
- 3 days buffer for meeting scheduling conflicts
- 1 day final verification before Phase 1 kickoff

**If timeline is compressed**: Minimum viable timeline is T-3 days for all milestones,
but this increases risk of incomplete decisions requiring mid-phase corrections.

#### Pre-Implementation Meeting Requirements

> **MANDATORY**: A decision-review meeting MUST be scheduled and completed BEFORE Phase 1 begins.
> This is not optional - Phase 1 kickoff is gated on meeting completion.

**Meeting Scheduling**:
- Schedule meeting at least 3 business days before planned Phase 1 start date
- **Scheduling Owner**: Tech Lead is responsible for scheduling this meeting
- **Calendar Invite**: Must include agenda and pre-read materials (this handoff document)
- Attendees: Tech Lead (required), Platform Team representative (required), Lead Developer (required)
- Duration: 1-2 hours
- **Fallback**: If primary attendees unavailable, reschedule - do not proceed without meeting

**Meeting Agenda**:
1. Review all three BLOCKING questions (Command Source, Intent Topics, Reducer Invocation)
2. Select final options for each decision with documented rationale
3. **Validate omnibase_core 0.5.x availability status** (Phase 0 Task 0 gate requirement - see Phase 1 Task 1)
4. Document decisions as ADRs under `docs/adr/`
5. **Assign decision owners**: Populate RACI matrix with specific names for Responsible/Accountable roles
6. **Set target dates**: Assign completion dates for each decision (MUST be before Phase 1 start)
7. Assign implementation owners for each Phase 1 task
8. Confirm timeline and resource availability
9. Identify escalation contacts for decision blockers

**Meeting Output Requirements**:
- All three BLOCKING decisions documented in ADRs
- **RACI matrix fully populated** with specific names and target dates (no placeholders remaining)
- **Decision owners formally assigned** and acknowledged for all three decisions
- **Phase 0 verification confirmed complete** (omnibase_core 0.5.x available - gate for Phase 1 Task 1)
- Phase 1 kickoff formally approved
- Implementation owners assigned and acknowledged
- Escalation path validated and contacts confirmed

**Exit Criteria**: Phase 1 may NOT begin until all meeting output requirements are satisfied.

> **CRITICAL GATE**: The pre-implementation meeting MUST verify that Phase 0 Task 0
> (omnibase_core 0.5.x availability) is complete. If 0.5.x is not available, Phase 1 Task 1
> gate will fail. See Phase 0 contingency plan for mitigation options.

### 9.2 Deferrable Questions

These questions can be answered during implementation:

4. **Result Events**: How do result events flow back to orchestrator?
   - Option A: Orchestrator subscribes to result topics
   - Option B: Effect publishes completion events, separate aggregator handles

5. **Timeout Handling**: Where does timeout tracking live?
   - Current: Effect node has `_slow_operation_threshold_ms`
   - Proposed: Orchestrator tracks workflow-level timeouts
   - **Note**: Detailed timeout handling patterns are documented in:
     - `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md` (tickets B6, C2)
       - B6: Runtime scheduler configuration and tick interval bounds
       - C2: Durable timeout handling with projection-backed deadlines
     - `docs/design/DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md` (section 8.2)
     This handoff does not prescribe specific timeout values - those are implementation details.

6. **Existing Tests**: What to do with `tests/unit/nodes/test_node_registry_effect.py`?
   - Rewrite for new architecture
   - Many tests are testing wrong patterns

7. **Handler Registration**: When are handlers registered with NodeRuntime?
   - Container wiring phase?
   - Effect node initialization?

---

## 10. Timeline Estimate

> **Decision Escalation**: If blocking decisions are not resolved, see Section 9.1
> "Escalation Path for Decision Blockers" for mitigation steps.

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

### Stakeholder Communication

When closing PR #52, include a clear explanation referencing this handoff document:

**Suggested close comment**:
> This PR is being closed because the implementation violates the ONEX 4-node architecture.
> The current implementation combines orchestrator, reducer, and effect responsibilities
> into a single monolithic file (3,065 lines), making it untestable and unmaintainable.
>
> **Architectural issues documented in**:
> - `docs/handoffs/HANDOFF_TWO_WAY_REGISTRATION_REFACTOR.md` (Section 2: Current State Analysis)
> - `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md` (canonical execution model)
>
> **Key violations**:
> - Mixed I/O with business logic (violates Effect purity)
> - Direct handler calls instead of intent-based communication
> - No proper command/event separation
> - Orchestration logic embedded in Effect node
>
> A new implementation following the correct architecture will be created in a separate branch.
> See the handoff document for the target architecture and migration plan.

### Stakeholder Notification Plan

**Stakeholders to Notify** (before closing PR #52):

| Stakeholder | Role | Notification Method | Timing |
|-------------|------|---------------------|--------|
| PR Author | Original implementer | Direct message + PR comment | Before close |
| Tech Lead | Architecture approval | Slack/Email | Before close |
| Platform Team | Downstream consumers | Team channel | Day of close |
| All Devs | Awareness | Engineering channel | After close |

**Notification Template** (for direct message to PR author):

> Hi [Author],
>
> I wanted to give you a heads-up that we'll be closing PR #52. This is NOT a reflection
> on code quality - the implementation itself is solid. However, during architecture review,
> we identified that the approach violates the ONEX 4-node pattern we need to follow.
>
> The key issue is that the current implementation mixes orchestrator, reducer, and effect
> responsibilities in a single node. Our architecture requires these to be separate components
> for testability and maintainability.
>
> Full details are in the handoff document:
> `docs/handoffs/HANDOFF_TWO_WAY_REGISTRATION_REFACTOR.md`
>
> The work you did is valuable and will inform the new implementation. Would you like to
> be involved in the refactored approach?
>
> Thanks for your understanding.

**Post-Close Actions**:
1. Link handoff document in PR close comment
2. Create new tracking ticket for refactored implementation
3. Archive (don't delete) PR #52 branch for reference

---

## 12. Documentation Deliverables

The following documentation artifacts must be created as part of this refactoring effort.

### Architecture Decision Records (ADRs)

| ADR | Status | Description | Related Ticket Plan Reference |
|-----|--------|-------------|-------------------------------|
| `docs/adr/ADR-XXX-command-source.md` | Pending | Command Source decision (Option A/B/C) | A3 (OMN-943) |
| `docs/adr/ADR-XXX-intent-topic-naming.md` | Pending | Intent topic naming convention | A2b (OMN-939) |
| `docs/adr/ADR-XXX-reducer-invocation-pattern.md` | Pending | Reducer invocation (direct vs event-based) | D1 (OMN-889) |

> **Note**: ADR numbers (XXX) will be assigned during the pre-implementation meeting.
> ADR template: `docs/adr/ADR-TEMPLATE.md` (if available) or follow standard ADR format.
> Related ticket references map to `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md`.
>
> **ADR Requirements from Ticket Plan Section 9.1**:
> All three blocking decisions (Command Source, Intent Topics, Reducer Invocation) MUST be
> documented as ADRs before Phase 1 implementation begins. These are mandatory prerequisites,
> not optional refinements. See Section 9.1 in this document for decision requirements,
> RACI matrix, and escalation paths.

### Operational Documentation

| Document | Status | Description |
|----------|--------|-------------|
| `docs/runbooks/registration-workflow.md` | To be created | Operator runbook for registration workflow |

**Runbook content requirements**:
- Troubleshooting registration failures
- Circuit breaker state recovery procedures
- Kafka topic health verification
- Consul/PostgreSQL registration state inspection
- Common failure scenarios and resolution steps
- Monitoring dashboards and alert response
- Feature flag rollback procedures (H1 migration)
- DLQ monitoring and recovery workflows

> **Operator Runbook Reference**:
> The registration workflow runbook is REQUIRED for production readiness. This runbook must
> be created and validated before deploying the refactored registration system. Operators
> should be trained on the runbook procedures during the H1 migration phase.

### Developer Documentation

| Document | Status | Description |
|----------|--------|-------------|
| `docs/guides/v1_0_0-migration-guide.md` | To be created | Migration guide for v1_0_0 directory structure |
| `docs/architecture/RUNTIME_EXECUTION_MODEL.md` | To be created | Runtime execution model documentation |

**Migration guide content requirements** (see H1 ticket):
- Import path changes (versioned to flat structure)
- Contract.yaml version field usage
- Backwards compatibility shim removal timeline
- Testing migration completeness

> **Developer Migration Guide Reference**:
> The v1_0_0 migration guide is REQUIRED for ticket H1 (Legacy Component Refactor Plan).
> This guide documents how to migrate from versioned directory structure (nodes/<name>/v1_0_0/)
> to flat structure (nodes/<name>/), including import path changes, contract.yaml version
> field usage, and testing migration completeness. See H1 ticket in the Ticket Plan for
> detailed migration requirements.

### Documentation Checklist

- [ ] ADR: Command Source Decision
- [ ] ADR: Intent Topic Naming
- [ ] ADR: Reducer Invocation Pattern
- [ ] Operator Runbook: Registration Workflow
- [ ] Developer Guide: v1_0_0 Migration
- [ ] Architecture Doc: Runtime Execution Model

---

## 13. Related Documents

**Canonical References** (authoritative for this refactor):
- `docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md` (Document Version 1.1.0)
  - Defines ticket structure, global constraints, and execution model
- `docs/design/DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md` (Version 2.1.2)
  - Defines workflow architecture patterns and terminology

**Architecture References**:
- `docs/architecture/DECLARATIVE_EFFECT_NODES_PLAN.md`
- `docs/architecture/CURRENT_NODE_ARCHITECTURE.md`

**Contract References**:
- `contracts/fsm/dual_registration_reducer_fsm.yaml`

**External Dependencies**:
- `omnibase_core` documentation for NodeRuntime, NodeOrchestrator, NodeEffect (requires 0.5.x)

---

## 14. Code References

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
