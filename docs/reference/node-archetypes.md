> **Navigation**: [Home](../index.md) > Reference > Node Archetypes

# Node Archetypes Reference

ONEX organizes all processing into four node archetypes. This reference provides complete documentation for each type.

> **Note**: For authoritative coding rules and standards, see [CLAUDE.md](../../CLAUDE.md). This documentation provides explanations and examples that supplement those rules.

## Quick Reference

| Archetype | Contract Type | Base Class | Purpose | Side Effects |
|-----------|---------------|------------|---------|--------------|
| **EFFECT** | `EFFECT_GENERIC` | `NodeEffect` | External I/O operations | Yes |
| **COMPUTE** | `COMPUTE_GENERIC` | `NodeCompute` | Pure transformations | None |
| **REDUCER** | `REDUCER_GENERIC` | `NodeReducer` | State + FSM + intents | None |
| **ORCHESTRATOR** | `ORCHESTRATOR_GENERIC` | `NodeOrchestrator` | Workflow coordination | Publishes events |

## Directory Structure

Every node follows this structure:

```
nodes/<node_name>/
├── contract.yaml     # ONEX contract (required)
├── node.py          # Declarative node class (required)
├── models/          # Node-specific Pydantic models
│   ├── __init__.py
│   └── model_*.py
├── handlers/        # Handler implementations (orchestrators)
│   └── handler_*.py
└── registry/        # Optional DI registry
    └── registry_infra_<name>.py
```

---

## EFFECT Archetype

### Purpose

EFFECT nodes handle **external I/O operations** - any interaction with systems outside the ONEX runtime boundary.

### When to Use

- Database operations (PostgreSQL, Qdrant)
- Service discovery (Consul, Kubernetes, Etcd)
- Secret management (Vault)
- HTTP/API calls to external services
- File system operations
- Message queue interactions

### Base Class

```python
from omnibase_core.nodes.node_effect import NodeEffect
```

### Example Implementation

**`contract.yaml`**:
```yaml
contract_version:
  major: 1
  minor: 0
  patch: 0
node_version: "1.0.0"
name: "node_registration_storage_effect"
node_type: "EFFECT_GENERIC"
description: "Storage operations for node registration records."

input_model:
  name: "ModelStorageRequest"
  module: "omnibase_infra.nodes.node_registration_storage_effect.models"

output_model:
  name: "ModelStorageResult"
  module: "omnibase_infra.nodes.node_registration_storage_effect.models"

# Capability-oriented naming (what it does, not what it uses)
capabilities:
  - name: "registration.storage"
    description: "Store, query, update, and delete registration records"
  - name: "registration.storage.query"
  - name: "registration.storage.upsert"
  - name: "registration.storage.delete"

# External I/O operations
io_operations:
  - operation: "store_registration"
    description: "Persist a registration record"
    input_fields:
      - record: "ModelRegistrationRecord"
      - correlation_id: "UUID | None"
    output_fields:
      - result: "ModelUpsertResult"
    idempotent: true

  - operation: "query_registration"
    description: "Query registration by node_id"
    input_fields:
      - node_id: "str"
    output_fields:
      - record: "ModelRegistrationRecord | None"
    idempotent: true

# Pluggable handler routing
handler_routing:
  routing_strategy: "handler_type_match"
  default_handler: "postgresql"
  handlers:
    - handler_type: "postgresql"
      handler:
        name: "HandlerRegistrationStoragePostgres"
        module: "omnibase_infra.handlers.registration_storage.handler_postgres"
    - handler_type: "mock"
      handler:
        name: "HandlerRegistrationStorageMock"
        module: "omnibase_infra.handlers.registration_storage.handler_mock"

# Error handling configuration
error_handling:
  retry_policy:
    max_retries: 3
    exponential_base: 2
    retry_on:
      - "InfraConnectionError"
      - "InfraTimeoutError"
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    reset_timeout_seconds: 60
```

**`node.py`**:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeRegistrationStorageEffect(NodeEffect):
    """Effect node for registration storage operations.

    Capability: registration.storage

    This node is declarative - all behavior is defined in contract.yaml
    and implemented through pluggable handlers. No custom storage logic
    exists in this class.

    Supported backends:
        - PostgreSQL (default)
        - Mock (for testing)
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the storage effect node."""
        super().__init__(container)


__all__ = ["NodeRegistrationStorageEffect"]
```

### Key Characteristics

1. **Capability-Oriented Naming**: Name by what it does ("registration.storage"), not technology ("postgres"). See [Capability Naming Convention](contracts.md#capability-naming-convention) for full guide.
2. **Pluggable Handlers**: Multiple backend implementations selectable via `handler_type`
3. **Idempotent Operations**: Mark operations as idempotent for safe retries
4. **Error Handling**: Configure circuit breakers and retry policies in contract

### Real Examples in Codebase

| Node | Path | Capability |
|------|------|------------|
| `NodeRegistrationStorageEffect` | `nodes/node_registration_storage_effect/` | `registration.storage` |
| `NodeServiceDiscoveryEffect` | `nodes/node_service_discovery_effect/` | `service.discovery` |
| `NodeRegistryEffect` | `nodes/node_registry_effect/` | Dual-backend registry |

---

## COMPUTE Archetype

### Purpose

COMPUTE nodes perform **pure transformations** - deterministic computations with no side effects.

### When to Use

- Data validation and rule enforcement
- Format conversion and mapping
- Business rule evaluation
- Parsing and serialization
- Aggregation calculations

### Base Class

```python
from omnibase_core.nodes.node_compute import NodeCompute
```

### Example Implementation

**`contract.yaml`**:
```yaml
contract_version:
  major: 1
  minor: 0
  patch: 0
node_version: "1.0.0"
name: "node_architecture_validator"
node_type: "COMPUTE_GENERIC"
description: "Validates ONEX architecture patterns in source code."

input_model:
  name: "ModelArchitectureValidationRequest"
  module: "omnibase_infra.nodes.architecture_validator.models"

output_model:
  name: "ModelArchitectureValidationResult"
  module: "omnibase_infra.nodes.architecture_validator.models"

capabilities:
  - name: "architecture_validation"
  - name: "violation_detection"
  - name: "fix_suggestions"

# Validation rules (contract-driven)
validation_rules:
  - rule_id: "ARCH-001"
    name: "No Direct Handler Dispatch"
    severity: "WARNING"
    description: "Handlers must not be called directly; route through RuntimeHost"
    detection_strategy:
      type: "ast_pattern"
      patterns:
        - "direct_handler_call"
        - "handler.handle("
    suggested_fix: "Route events through RuntimeHost.process_event()"

  - rule_id: "ARCH-002"
    name: "No Handler Publishing Events"
    severity: "ERROR"
    description: "Only orchestrators may publish events"
    detection_strategy:
      type: "signature_analysis"
      checks:
        - no_bus_in_init_params
        - no_bus_instance_attributes
        - no_publish_method_calls
    suggested_fix: "Return events in ModelHandlerOutput instead"

  - rule_id: "ARCH-003"
    name: "No Workflow FSM in Orchestrators"
    severity: "ERROR"
    description: "FSM logic belongs in reducers, not orchestrators"
    detection_strategy:
      type: "ast_pattern"
      patterns:
        - "state_machine"
        - "fsm_transition"
    suggested_fix: "Move FSM logic to a reducer node"
```

**`node.py`**:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeArchitectureValidator(NodeCompute):
    """Architecture validator - COMPUTE_GENERIC node.

    Validates ONEX architecture patterns:
    - ARCH-001: No direct handler dispatch
    - ARCH-002: No handler publishing events
    - ARCH-003: No workflow FSM in orchestrators

    Pure computation - no side effects.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodeArchitectureValidator"]
```

### Key Characteristics

1. **Referential Transparency**: Same input always produces same output
2. **No I/O**: Cannot access databases, APIs, or file system
3. **Deterministic**: Results are predictable and testable
4. **Validation-Focused**: Often used for rule enforcement

### Real Examples in Codebase

| Node | Path | Purpose |
|------|------|---------|
| `NodeArchitectureValidator` | `nodes/architecture_validator/` | Validate architecture patterns |

---

## REDUCER Archetype

### Purpose

REDUCER nodes manage **state aggregation** via FSM (Finite State Machine) and emit intents for effect execution.

### When to Use

- Workflow state management
- Event sourcing patterns
- Multi-step process tracking
- Decision making based on accumulated state

### Base Class

```python
from omnibase_core.nodes.node_reducer import NodeReducer
```

### Example Implementation

**`contract.yaml`**:
```yaml
contract_version:
  major: 1
  minor: 0
  patch: 0
node_version: "1.0.0"
name: "node_registration_reducer"
node_type: "REDUCER_GENERIC"
description: "FSM-driven reducer for node registration state transitions."

input_model:
  name: "ModelReducerInput"
  module: "omnibase_core.models.reducer.model_reducer_input"

output_model:
  name: "ModelReducerOutput"
  module: "omnibase_core.models.reducer.model_reducer_output"

# FSM State Machine (Single Source of Truth)
state_machine:
  state_machine_name: "registration_fsm"
  initial_state: "idle"

  states:
    - state_name: "idle"
      description: "Waiting for introspection event"
      entry_actions: []
      exit_actions: []

    - state_name: "pending"
      description: "Registration initiated, awaiting backend confirmations"
      entry_actions:
        - "emit_consul_intent"
        - "emit_postgres_intent"

    - state_name: "partial"
      description: "One backend confirmed, waiting for second"

    - state_name: "complete"
      description: "Both backends confirmed - registration successful"

    - state_name: "failed"
      description: "Registration failed - requires intervention"

  transitions:
    - from_state: "idle"
      to_state: "pending"
      trigger: "introspection_received"
      conditions:
        - expression: "node_id is_present"
          required: true
      actions:
        - action_name: "build_consul_intent"
          action_type: "intent_emission"
        - action_name: "build_postgres_intent"
          action_type: "intent_emission"

    - from_state: "pending"
      to_state: "partial"
      trigger: "consul_confirmed"

    - from_state: "pending"
      to_state: "partial"
      trigger: "postgres_confirmed"

    - from_state: "partial"
      to_state: "complete"
      trigger: "consul_confirmed"
      conditions:
        - expression: "postgres_confirmed is_true"
          required: true

    - from_state: "partial"
      to_state: "complete"
      trigger: "postgres_confirmed"
      conditions:
        - expression: "consul_confirmed is_true"
          required: true

    - from_state: "*"
      to_state: "failed"
      trigger: "error_received"

    - from_state: "failed"
      to_state: "idle"
      trigger: "reset_requested"

    - from_state: "complete"
      to_state: "idle"
      trigger: "reset_requested"

# Intent emission configuration
intent_emission:
  enabled: true
  intents:
    - intent_type: "consul.register"
      target_pattern: "consul://service/{service_name}"
      payload_model: "ModelPayloadConsulRegister"
      payload_module: "omnibase_infra.nodes.reducers.models"

    - intent_type: "postgres.upsert_registration"
      target_pattern: "postgres://node_registrations/{node_id}"
      payload_model: "ModelPayloadPostgresUpsertRegistration"
      payload_module: "omnibase_infra.nodes.reducers.models"

# Idempotency configuration
idempotency:
  enabled: true
  strategy: "event_id_tracking"
  description: "Track processed event_ids to prevent duplicate processing"

# Validation rules
validation:
  event_validation:
    - field: "node_id"
      required: true
      type: "str"
    - field: "node_type"
      required: true
      type: "EnumNodeKind"
```

**`node.py`**:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_reducer import NodeReducer

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer
    from omnibase_infra.nodes.node_registration_reducer.models import (
        ModelRegistrationState,
    )


class NodeRegistrationReducer(
    NodeReducer["ModelRegistrationState", "ModelRegistrationState"]
):
    """Registration reducer - FSM state transitions driven by contract.yaml.

    All state transition logic, intent emission, and validation are driven
    entirely by the contract.yaml FSM configuration.

    Pattern:
        reduce(state, event) → (new_state, intents)

    FSM Flow:
        1. Receive introspection event (trigger: introspection_received)
        2. FSM transitions idle → pending (emits registration intents)
        3. Receive confirmation events from backends
        4. FSM transitions pending → partial → complete
        5. On errors, FSM transitions to failed state
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the reducer."""
        super().__init__(container)


__all__ = ["NodeRegistrationReducer"]
```

### State Model Pattern

Reducer state models must be **immutable** with `with_*` transition methods:

```python
from pydantic import BaseModel, ConfigDict
from uuid import UUID

class ModelRegistrationState(BaseModel):
    """Immutable state for registration FSM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = "idle"
    consul_confirmed: bool = False
    postgres_confirmed: bool = False
    last_processed_event_id: UUID | None = None

    def with_consul_confirmed(self, event_id: UUID) -> "ModelRegistrationState":
        """Return NEW instance with consul_confirmed=True."""
        new_status = "complete" if self.postgres_confirmed else "partial"
        return ModelRegistrationState(
            status=new_status,
            consul_confirmed=True,
            postgres_confirmed=self.postgres_confirmed,
            last_processed_event_id=event_id,
        )

    def with_postgres_confirmed(self, event_id: UUID) -> "ModelRegistrationState":
        """Return NEW instance with postgres_confirmed=True."""
        new_status = "complete" if self.consul_confirmed else "partial"
        return ModelRegistrationState(
            status=new_status,
            consul_confirmed=self.consul_confirmed,
            postgres_confirmed=True,
            last_processed_event_id=event_id,
        )

    def is_duplicate_event(self, event_id: UUID) -> bool:
        """Check if event was already processed (idempotency)."""
        return self.last_processed_event_id == event_id
```

### Intent Payload Pattern

Intents use a two-layer structure:

```python
from pydantic import BaseModel, Field
from typing import Literal
from uuid import UUID

class ModelPayloadConsulRegister(BaseModel):
    """Typed payload for Consul registration intents."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Routing key at payload level
    intent_type: Literal["consul.register"] = "consul.register"

    # Business data
    correlation_id: UUID
    service_id: str
    service_name: str
    tags: list[str]
    health_check: dict[str, str] | None = None
```

The outer `ModelIntent` uses `intent_type="extension"`, and routing happens via `payload.intent_type`.

### Key Characteristics

1. **Pure Function**: `reduce(state, event) → (new_state, intents)` - no side effects
2. **FSM-Driven**: All transitions defined in contract, not Python
3. **Immutable State**: State models are frozen, use `with_*` methods
4. **Intent Emission**: Emit intents for Effect layer to execute
5. **Idempotency**: Track event_ids to prevent duplicate processing

### Real Examples in Codebase

| Node | Path | FSM |
|------|------|-----|
| `NodeRegistrationReducer` | `nodes/node_registration_reducer/` | `registration_fsm` |

---

## ORCHESTRATOR Archetype

### Purpose

ORCHESTRATOR nodes coordinate **workflows** by routing events to handlers and managing multi-step execution.

### When to Use

- Multi-handler workflows
- Event coordination
- Publishing events (ONLY orchestrators can publish)
- Routing intents to effects

### Base Class

```python
from omnibase_core.nodes.node_orchestrator import NodeOrchestrator
```

### Example Implementation

**`contract.yaml`**:
```yaml
contract_version:
  major: 1
  minor: 0
  patch: 0
node_version: "1.0.0"
name: "node_registration_orchestrator"
node_type: "ORCHESTRATOR_GENERIC"
description: "Coordinates the node registration workflow."

input_model:
  name: "ModelOrchestratorInput"
  module: "omnibase_infra.nodes.node_registration_orchestrator.models"

output_model:
  name: "ModelOrchestratorOutput"
  module: "omnibase_infra.nodes.node_registration_orchestrator.models"

# Workflow definition with execution graph
workflow_coordination:
  workflow_definition:
    workflow_metadata:
      workflow_name: "node_registration_workflow"
      workflow_version: { major: 1, minor: 0, patch: 0 }
      description: "Two-way registration with Consul and PostgreSQL"

    execution_graph:
      nodes:
        - node_id: "receive_introspection"
          node_type: EFFECT_GENERIC
          description: "Receive node introspection event"
          step_config:
            event_pattern: ["node-introspection.*", "node-registration-acked.*"]

        - node_id: "read_projection"
          node_type: EFFECT_GENERIC
          depends_on: ["receive_introspection"]
          description: "Query current registration state"
          step_config:
            protocol: "ProtocolProjectionReader"

        - node_id: "compute_intents"
          node_type: REDUCER_GENERIC
          depends_on: ["read_projection"]
          description: "FSM transition and intent computation"

        - node_id: "execute_consul_registration"
          node_type: EFFECT_GENERIC
          depends_on: ["compute_intents"]
          description: "Register with Consul"
          step_config:
            intent_filter: "consul.*"

        - node_id: "execute_postgres_registration"
          node_type: EFFECT_GENERIC
          depends_on: ["compute_intents"]
          description: "Persist to PostgreSQL"
          step_config:
            intent_filter: "postgres.*"

        - node_id: "aggregate_results"
          node_type: COMPUTE_GENERIC
          depends_on: ["execute_consul_registration", "execute_postgres_registration"]
          description: "Aggregate backend results"

    coordination_rules:
      execution_mode: parallel
      parallel_execution_allowed: true
      max_parallel_branches: 2
      failure_recovery_strategy: retry
      max_retries: 3
      timeout_ms: 30000
      checkpointing:
        enabled: true
        checkpoint_after_steps: ["compute_intents"]

# Handler routing (event → handler mapping)
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "ModelNodeIntrospectionEvent"
        module: "omnibase_infra.models.registration.model_node_introspection_event"
      handler:
        name: "HandlerNodeIntrospected"
        module: "omnibase_infra.nodes.node_registration_orchestrator.handlers"
      output_events:
        - "ModelNodeRegistrationInitiated"
      state_decision_matrix:
        - current_state: null
          action: "emit_registration_initiated"
          description: "New node - initiate registration"
        - current_state: "LIVENESS_EXPIRED"
          action: "emit_registration_initiated"
          description: "Expired node - re-register"
        - current_state: "ACTIVE"
          action: "no_op"
          description: "Already active - skip"

    - event_model:
        name: "ModelNodeRegistrationAcked"
        module: "omnibase_infra.models.registration.model_node_registration_acked"
      handler:
        name: "HandlerNodeRegistrationAcked"
        module: "omnibase_infra.nodes.node_registration_orchestrator.handlers"
      output_events:
        - "ModelNodeRegistrationAckReceived"
        - "ModelNodeBecameActive"

    - event_model:
        name: "ModelRuntimeTick"
        module: "omnibase_infra.runtime.models.model_runtime_tick"
      handler:
        name: "HandlerRuntimeTick"
        module: "omnibase_infra.nodes.node_registration_orchestrator.handlers"
      timeout_evaluation:
        uses_injected_now: true
        queries_pending_entities: true

# Events consumed from Kafka
consumed_events:
  - topic: "{env}.{namespace}.onex.evt.node-introspection.v1"
    event_type: "NodeIntrospectionEvent"
    description: "Node startup introspection"
  - topic: "{env}.{namespace}.onex.cmd.node-registration-acked.v1"
    event_type: "NodeRegistrationAcked"
    description: "Node acknowledges registration"
  - topic: "{env}.{namespace}.onex.internal.runtime-tick.v1"
    event_type: "RuntimeTick"
    description: "Periodic tick for timeout evaluation"

# Events published to Kafka
published_events:
  - topic: "{env}.{namespace}.onex.evt.node-registration-initiated.v1"
    event_type: "NodeRegistrationInitiated"
  - topic: "{env}.{namespace}.onex.evt.node-became-active.v1"
    event_type: "NodeBecameActive"
  - topic: "{env}.{namespace}.onex.evt.node-liveness-expired.v1"
    event_type: "NodeLivenessExpired"

# Intent routing to effect nodes
intent_consumption:
  intent_routing_table:
    "consul.register": "node_registry_effect"
    "consul.deregister": "node_registry_effect"
    "postgres.upsert_registration": "node_registry_effect"
    "postgres.deactivate_registration": "node_registry_effect"
```

**`node.py`**:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Declarative orchestrator for node registration workflow.

    All behavior is defined in contract.yaml - no custom logic here.
    Handler routing is driven entirely by the contract.

    Workflow:
        1. Receive introspection event
        2. Query current projection state
        3. Call reducer to compute intents
        4. Execute intents via effect nodes (parallel)
        5. Aggregate results
        6. Publish result events

    Only orchestrators may publish events to Kafka.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeRegistrationOrchestrator"]
```

### Key Characteristics

1. **ONLY Node Type That Publishes Events**: Handlers return events, orchestrators publish
2. **Workflow Coordination**: Define execution graphs with dependencies
3. **Handler Routing**: Map event types to handler implementations
4. **Intent Routing**: Route intents to appropriate effect nodes
5. **No FSM Logic**: Orchestrators coordinate, reducers own state

### Real Examples in Codebase

| Node | Path | Workflow |
|------|------|----------|
| `NodeRegistrationOrchestrator` | `nodes/node_registration_orchestrator/` | Node registration |

---

## Summary: When to Use Each Archetype

| Scenario | Archetype | Reason |
|----------|-----------|--------|
| Call external API | EFFECT | External I/O |
| Store data in database | EFFECT | External I/O |
| Validate input data | COMPUTE | Pure transformation |
| Transform data format | COMPUTE | Pure transformation |
| Track workflow state | REDUCER | FSM-driven state |
| Emit intents for backends | REDUCER | Intent emission |
| Coordinate multi-step workflow | ORCHESTRATOR | Workflow coordination |
| Publish events to Kafka | ORCHESTRATOR | Only type that can publish |

## Related Documentation

| Topic | Document |
|-------|----------|
| **Coding standards** | [CLAUDE.md](../../CLAUDE.md) - **authoritative source** for all rules |
| Quick start | [Quick Start Guide](../getting-started/quickstart.md) |
| Architecture overview | [Architecture Overview](../architecture/overview.md) |
| Contract format | [Contract.yaml Reference](contracts.md) |
| Registration example | [2-Way Registration Walkthrough](../guides/registration-example.md) |
| All patterns | [Pattern Documentation](../patterns/README.md) |
| Handler patterns | [Handler Plugin Loader](../patterns/handler_plugin_loader.md) |
| Error handling | [Error Handling Patterns](../patterns/error_handling_patterns.md) |
