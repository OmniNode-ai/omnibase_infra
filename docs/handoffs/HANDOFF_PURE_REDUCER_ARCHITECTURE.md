# Handoff: Pure Reducer Architecture for MVP Registration Workflow

**Date**: 2025-12-18
**Author**: Architecture Planning Session
**Status**: Architecture Locked

---

## Executive Summary

This document captures the architectural decisions for the MVP registration workflow. The core principle is **reducer purity**: reducers perform no I/O operations, emit only typed intents, and all serialization happens at the I/O boundary (Effects).

**Key Decision**: `ModelIntent.payload` requires `BaseModel` only - no `dict[str, Any]` escape hatch. This enforces type safety at the architecture level.

---

## Architecture Decision

### The Locked Architecture Rules

1. **Reducers are PURE**
   - No I/O operations (no database, network, file system)
   - No side effects
   - Deterministic: same input always produces same output
   - Emit only typed `ModelIntent` objects

2. **Orchestrators Own Workflows**
   - Coordinate between reducers and effects
   - Manage workflow state transitions
   - Handle retry logic and error recovery
   - Route intents to appropriate effects

3. **Effects Own All I/O**
   - Database operations (Consul, PostgreSQL)
   - Network calls (Kafka publish/consume)
   - File system operations
   - External service integrations
   - Call `serialize_payload()` at I/O boundary

4. **ModelIntent Requires Typed BaseModel Payloads**
   - `payload: BaseModel` (NOT `BaseModel | dict[str, Any]`)
   - No dict escape hatch permitted
   - Type safety enforced at compile time
   - Serialization only at I/O boundary via `serialize_payload()`

5. **Serialization at I/O Boundary Only**
   - Reducers emit typed models
   - Effects call `intent.serialize_payload()` before I/O
   - No early serialization in domain layer
   - Type information preserved until the last moment

---

## Ticket Summary

| Ticket | Title | Repo | Priority | Status |
|--------|-------|------|----------|--------|
| [OMN-912](https://linear.app/omninode/issue/OMN-912) | Update ModelIntent to Accept Typed Payloads | omnibase_core | Urgent | Backlog |
| [OMN-913](https://linear.app/omninode/issue/OMN-913) | Registration Payload and Outcome Models | omnibase_core | High | Backlog |
| [OMN-914](https://linear.app/omninode/issue/OMN-914) | Reducer Purity Enforcement Gates | omnibase_infra | High | Backlog |
| [OMN-915](https://linear.app/omninode/issue/OMN-915) | Mocked E2E Registration Workflow Test (A0-A6) | omnibase_infra | High | Backlog |
| [OMN-889](https://linear.app/omninode/issue/OMN-889) | Dual Registration Reducer FSM | omnibase_infra | High | In Progress (PR #52) |
| [OMN-888](https://linear.app/omninode/issue/OMN-888) | Node Registration Orchestrator | omnibase_infra | Urgent | Backlog |
| [OMN-890](https://linear.app/omninode/issue/OMN-890) | Registry Effect Node | omnibase_infra | High | Merged |
| [OMN-892](https://linear.app/omninode/issue/OMN-892) | 2-Way Registration E2E Test | omnibase_infra | Medium | Backlog |

---

## Dependency Graph

```
                                    OMN-912 (ModelIntent typed payloads)
                                              |
                    +-------------------------+-------------------------+
                    |                                                   |
                    v                                                   v
            OMN-913 (Payload Models)                            OMN-914 (Purity Gates)
                    |                                                   |
                    +-------------------------+-------------------------+
                                              |
                                              v
                                    OMN-889 (Dual Registration Reducer)
                                         [REFACTOR REQUIRED]
                                              |
                    +-------------------------+-------------------------+
                    |                         |                         |
                    v                         v                         v
            OMN-890 (Registry Effect)   OMN-888 (Orchestrator)   OMN-915 (Mocked E2E)
                 [MERGED]                     |                         |
                                              |                         |
                                              v                         |
                                    OMN-892 (2-Way E2E Test) <----------+
```

### Dependency Explanation

1. **OMN-912** is foundational - defines the typed payload contract
2. **OMN-913** and **OMN-914** depend on OMN-912 for the payload type definitions
3. **OMN-889** must be refactored after OMN-913 to use typed payloads
4. **OMN-890** is already merged (Registry Effect Node)
5. **OMN-888** (Orchestrator) can proceed once reducer refactor is complete
6. **OMN-915** provides mocked E2E testing during development
7. **OMN-892** is the final integration test with real infrastructure

---

## Key Design Decisions

### 1. Reducers Are Pure (No I/O)

**Decision**: Reducers MUST NOT perform any I/O operations.

**Rationale**:
- Enables replay/recovery: given same inputs, replay produces identical outputs
- Simplifies testing: no mocking required for I/O
- Guarantees determinism: essential for FSM-based state machines
- Separates concerns: business logic (reducer) vs infrastructure (effect)

**Implementation**:
```python
class NodeDualRegistrationReducer(NodeReducer):
    async def execute_reduction(self, input_data: ModelReducerInput) -> ModelReducerOutput:
        # Pure computation only
        decision = self._compute_registration_decision(input_data)

        # Emit typed intent - NO serialization here
        intent = ModelIntent(
            intent_type="registration.execute",
            target="registry_effect",
            payload=ModelRegistrationPayload(  # Typed model, not dict
                node_id=input_data.node_id,
                registration_type=decision.registration_type,
            )
        )

        return ModelReducerOutput(
            intents=[intent],
            state_updates=[...],
        )
```

### 2. Orchestrators Own Workflows

**Decision**: Orchestrators coordinate the workflow, manage state, and route intents.

**Responsibilities**:
- Receive events from Kafka
- Invoke reducers with typed inputs
- Route emitted intents to appropriate effects
- Handle failures and retry logic
- Manage workflow state transitions

**Implementation**:
```python
class NodeRegistrationOrchestrator(NodeOrchestrator):
    async def execute_orchestration(self, input_data: ModelOrchestratorInput) -> ModelOrchestratorOutput:
        # 1. Invoke reducer (pure computation)
        reducer_output = await self.reducer.execute_reduction(reducer_input)

        # 2. Route intents to effects
        for intent in reducer_output.intents:
            await self.route_intent_to_effect(intent)

        # 3. Return orchestration result
        return ModelOrchestratorOutput(...)
```

### 3. Effects Own All I/O

**Decision**: Effects are the only components that perform I/O operations.

**Responsibilities**:
- Database reads/writes (Consul KV, PostgreSQL)
- Message publishing/consuming (Kafka)
- External API calls
- Serialization at I/O boundary

**Implementation**:
```python
class NodeRegistryEffect(NodeEffect):
    async def execute_effect(self, input_data: ModelEffectInput) -> ModelEffectOutput:
        intent = input_data.intent

        # Serialize at I/O boundary - ONLY place this happens
        payload_dict = intent.serialize_payload()

        # Perform I/O
        await self.consul_adapter.kv_put(
            key=f"nodes/{payload_dict['node_id']}",
            value=payload_dict,
        )

        return ModelEffectOutput(success=True)
```

### 4. ModelIntent Requires Typed BaseModel Payloads

**Decision**: `payload: BaseModel` only, no `dict[str, Any]` escape hatch.

**Rationale**:
- Once a dict is emitted, type safety is permanently lost
- Dict escape hatch creates two code paths to maintain
- Encourages lazy typing ("I'll just use a dict for now")
- Makes serialization logic conditional and harder to reason about

**The Dict Escape Hatch Was Explicitly Rejected**:
```python
# REJECTED - This allows type safety to leak
payload: BaseModel | dict[str, Any]  # NO!

# APPROVED - Strict type safety
payload: BaseModel  # YES!
```

**Migration Strategy**:
1. Identify all dict payload usages
2. Create typed `BaseModel` for each payload shape
3. Update code to use typed models
4. Remove dict fallback code paths

### 5. Serialization at I/O Boundary

**Decision**: Serialization happens ONLY when data crosses the I/O boundary.

**Implementation**:
```python
class ModelIntent(BaseModel):
    payload: BaseModel  # Typed model, not dict

    def serialize_payload(self) -> dict[str, Any]:
        """Called by Effects at I/O boundary only."""
        return self.payload.model_dump(mode="json")
```

**Where Serialization Occurs**:
- Effect writes to Consul: `intent.serialize_payload()`
- Effect publishes to Kafka: `intent.serialize_payload()`
- Effect writes to database: `intent.serialize_payload()`

**Where Serialization Does NOT Occur**:
- Reducer computation
- Orchestrator routing
- Intent creation
- State transitions

---

## Implementation Order

### Phase 1: Foundation (omnibase_core)

1. **OMN-912**: Update ModelIntent to BaseModel-only payload
   - Remove dict fallback
   - Add `serialize_payload()` method
   - Update tests

2. **OMN-913**: Create Registration Payload Models
   - `ModelRegistrationPayload`
   - `ModelRegistrationOutcome`
   - Other workflow-specific models

### Phase 2: Enforcement (omnibase_infra)

3. **OMN-914**: Reducer Purity Enforcement Gates
   - AST-based I/O detection
   - Test fixtures for purity validation
   - CI integration

4. **OMN-889**: Refactor Dual Registration Reducer
   - Update PR #52 to use typed payloads
   - Remove any dict usage
   - Ensure pure reducer implementation

### Phase 3: Integration (omnibase_infra)

5. **OMN-888**: Node Registration Orchestrator
   - Implement workflow coordination
   - Intent routing to effects
   - Error handling

6. **OMN-915**: Mocked E2E Registration Test
   - Test A0-A6 registration phases
   - Mock infrastructure dependencies
   - Validate workflow correctness

### Phase 4: Validation (omnibase_infra)

7. **OMN-892**: 2-Way Registration E2E Test
   - Real Consul integration
   - Real Kafka integration
   - Full workflow validation

---

## Breaking Changes

### PR #52 (OMN-889) Refactor Required

The current PR #52 for OMN-889 must be updated to comply with the pure reducer architecture:

**Required Changes**:

1. **Remove dict payloads**: Replace all `dict[str, Any]` payloads with typed models
2. **Remove I/O operations**: Move any I/O to the effect layer
3. **Use typed ModelIntent**: Ensure all intents use `BaseModel` payloads
4. **Update tests**: Use typed payload models in all tests

**Example Migration**:
```python
# BEFORE (dict payload)
intent = ModelIntent(
    intent_type="registration.execute",
    target="registry_effect",
    payload={"node_id": node_id, "action": "register"},  # dict - BAD
)

# AFTER (typed payload)
intent = ModelIntent(
    intent_type="registration.execute",
    target="registry_effect",
    payload=ModelRegistrationPayload(  # typed - GOOD
        node_id=node_id,
        action=EnumRegistrationAction.REGISTER,
    ),
)
```

---

## Final Invariant: The Replay/Determinism Guarantee

The architecture guarantees the following invariant:

> **Given the same sequence of inputs to a reducer, replay will always produce the same sequence of outputs (intents and state updates).**

This invariant enables:
- **Debugging**: Replay production issues locally
- **Recovery**: Reconstruct state from event log
- **Testing**: Deterministic test execution
- **Auditing**: Verify decision correctness

**How the Architecture Ensures This**:

1. **No I/O in reducers**: Eliminates non-determinism from external state
2. **Typed payloads**: Prevents schema drift that could affect replay
3. **FSM-based transitions**: State machine guarantees valid transitions
4. **Immutable inputs**: Inputs are not modified during processing

---

## Related Documents

- [MVP Execution Plan](/workspace/omnibase_infra3/docs/MVP_EXECUTION_PLAN.md)
- [MVP Plan](/workspace/omnibase_infra3/docs/MVP_PLAN.md)
- [Container Dependency Injection](/workspace/omnibase_infra3/docs/patterns/container_dependency_injection.md)
- [Error Handling Patterns](/workspace/omnibase_infra3/docs/patterns/error_handling_patterns.md)

---

## Questions or Concerns

If any aspect of this architecture is unclear or needs refinement, please raise it before implementation begins. The architecture is **locked** but clarifications are welcome.
