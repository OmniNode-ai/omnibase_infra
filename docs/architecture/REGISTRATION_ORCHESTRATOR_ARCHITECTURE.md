# Registration Orchestrator Architecture

**Document Version**: 1.0.0
**Node Version**: 1.0.0
**Contract Version**: 1.0.0
**Ticket**: OMN-888

## Overview

The Node Registration Orchestrator is an ONEX **ORCHESTRATOR** node that coordinates node lifecycle registration workflows. It uses a **declarative pattern** where 100% of workflow behavior is driven by `contract.yaml`, not Python code.

**Location**: `src/omnibase_infra/nodes/node_registration_orchestrator/`

**Purpose**: Coordinate the dual-registration of ONEX nodes to both Consul (service discovery) and PostgreSQL (persistent registry) in response to introspection events.

## Architectural Position

```
                           ONEX 4-Node Architecture

    ┌─────────────────────────────────────────────────────────────────┐
    │                         Event Flow                               │
    │                                                                  │
    │   ┌────────────┐    Events    ┌────────────┐    Intents         │
    │   │   EFFECT   │─────────────►│  REDUCER   │──────────────┐     │
    │   │ (Adapters) │              │  (State)   │              │     │
    │   └────────────┘              └────────────┘              │     │
    │        ▲                                                  ▼     │
    │        │                                          ┌────────────┐│
    │        │         Workflow Coordination            │ORCHESTRATOR││
    │        └─────────────────────────────────────────►│ (Workflow) ││
    │                                                   └────────────┘│
    │                                  ▲                      │       │
    │                                  │                      │       │
    │                           ┌────────────┐                │       │
    │                           │  COMPUTE   │◄───────────────┘       │
    │                           │  (Pure)    │                        │
    │                           └────────────┘                        │
    └─────────────────────────────────────────────────────────────────┘

    The Registration Orchestrator is an ORCHESTRATOR node that:
    1. Receives introspection events
    2. Delegates to reducer to compute intents
    3. Executes intents via effect nodes
    4. Publishes outcome events
```

## 4-Node Architecture Integration

The Registration Orchestrator implements the ONEX 4-node pattern by coordinating between different node types:

| Node Type | Role in Registration Workflow | Component |
|-----------|-------------------------------|-----------|
| **ORCHESTRATOR** | Workflow coordination and step sequencing | `NodeRegistrationOrchestrator` |
| **REDUCER** | Compute registration intents from events | `ProtocolReducer` implementation (OMN-889) |
| **EFFECT** | Execute Consul/PostgreSQL registrations | `node_registry_effect` (OMN-890) |
| **COMPUTE** | Aggregate results, evaluate timeouts | Inline workflow steps |

### Reducer-Effect Separation Pattern

The orchestrator implements a functional architecture that cleanly separates:

- **Pure computation** (reducer): Deterministic intent generation from events
- **Side effects** (effect): Infrastructure operations (Consul, PostgreSQL)

```
                     Registration Workflow Internal Architecture

  Introspection      ┌───────────────────────────────────────────────────┐
  Event              │         NodeRegistrationOrchestrator              │
    │                │                                                   │
    │                │  ┌──────────────────────────────────────────┐    │
    ▼                │  │           Reducer (Pure)                  │    │
  ┌─────────┐        │  │                                           │    │
  │ receive │───────►│  │  State + Event ──► (NewState, Intents)   │    │
  │ event   │        │  │                                           │    │
  └─────────┘        │  │  - Deterministic                          │    │
                     │  │  - No I/O                                 │    │
                     │  │  - Idempotent                             │    │
                     │  └──────────────┬───────────────────────────┘    │
                     │                 │                                 │
                     │                 │ intents                         │
                     │                 ▼                                 │
                     │  ┌──────────────────────────────────────────┐    │
                     │  │           Effect (I/O)                    │    │
                     │  │                                           │    │
                     │  │  Intent ──► Side Effect ──► Result       │    │
                     │  │                                           │    │
                     │  │  - Consul registration                    │    │
                     │  │  - PostgreSQL upsert                      │    │
                     │  │  - May retry internally                   │    │
                     │  └──────────────────────────────────────────┘    │
                     │                                                   │
                     └───────────────────────────────────────────────────┘
                                           │
                                           ▼
                                  NodeRegistrationResultEvent
```

## Design Decisions

### 1. Declarative Workflow Pattern

**Decision**: All workflow logic is declared in `contract.yaml`, not Python code.

**Rationale**:
- **Single source of truth**: Workflow behavior is documented in contract
- **Testability**: Contract can be validated without running code
- **Consistency**: All orchestrators follow the same pattern
- **Tooling**: Standard contract parsers can analyze workflow

**Implementation**:
```python
class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Zero custom methods - base class handles everything from contract."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
        # All workflow logic comes from contract.yaml
```

### 2. Domain-Grouped Protocols

**Decision**: `ProtocolReducer` and `ProtocolEffect` are defined together in `protocols.py`.

**Rationale**:
- Protocols are tightly coupled (reducer produces what effect consumes)
- Together they define the complete workflow interface
- Share common type dependencies (`ModelRegistrationIntent`, `ModelReducerState`)
- Per CLAUDE.md "Protocol File Naming" convention

**Documentation**: See `docs/architecture/NODE_REGISTRATION_ORCHESTRATOR_PROTOCOLS.md` for detailed protocol design.

### 3. Immutable Reducer State

**Decision**: `ModelReducerState` is a frozen Pydantic model with `frozenset` for collections.

**Rationale**:
- Thread safety: Concurrent reductions cannot mutate shared state
- Predictability: State transitions are explicit (return new instance)
- Testability: State is snapshot-able and comparable

**Implementation**:
```python
class ModelReducerState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    processed_node_ids: frozenset[UUID] = Field(default_factory=frozenset)
```

### 4. Discriminated Union for Intents

**Decision**: Intent models use a discriminated union with `kind` field.

**Rationale**:
- Type safety: Python type checker can narrow intent types
- Routing: Effect node can dispatch based on `kind`
- Extensibility: New intent types can be added without breaking existing code

**Implementation**:
```python
ModelRegistrationIntent = Annotated[
    ModelConsulRegistrationIntent | ModelPostgresUpsertIntent,
    Field(discriminator="kind"),
]
```

## Workflow Coordination

### Execution Graph

The workflow is defined as a DAG (Directed Acyclic Graph) in `contract.yaml`:

```yaml
execution_graph:
  nodes:
    - node_id: "receive_introspection"
      node_type: effect
      depends_on: []

    - node_id: "read_projection"
      node_type: effect
      depends_on: ["receive_introspection"]

    - node_id: "evaluate_timeout"
      node_type: compute
      depends_on: ["read_projection"]

    - node_id: "compute_intents"
      node_type: reducer
      depends_on: ["evaluate_timeout"]

    - node_id: "execute_consul_registration"
      node_type: effect
      depends_on: ["compute_intents"]

    - node_id: "execute_postgres_registration"
      node_type: effect
      depends_on: ["compute_intents"]

    - node_id: "aggregate_results"
      node_type: compute
      depends_on: ["execute_consul_registration", "execute_postgres_registration"]

    - node_id: "publish_outcome"
      node_type: effect
      depends_on: ["aggregate_results"]
```

### Parallel Execution

The workflow uses parallel execution mode for registration steps:

```yaml
coordination_rules:
  execution_mode: parallel
  parallel_execution_allowed: true
  max_parallel_branches: 2
```

This means:
- `execute_consul_registration` and `execute_postgres_registration` run concurrently
- Both depend only on `compute_intents`, forming a single parallel "wave"
- Reduces end-to-end latency for successful registrations

### Retry Strategy

Two levels of retry are configured:

**1. Workflow Step Retry** (`coordination_rules.max_retries: 3`):
- Retries an entire workflow step when it fails
- No backoff delay between retries
- Used for transient coordination-level failures

**2. Error-Level Retry** (`error_handling.retry_policy`):
- Retries specific error types with exponential backoff
- Delays: 100ms -> 200ms -> 400ms (capped at 5000ms)
- Used for transient infrastructure errors

### Circuit Breaker

The workflow includes circuit breaker protection:

```yaml
circuit_breaker:
  enabled: true
  failure_threshold: 5
  reset_timeout_ms: 60000
```

- Opens after 5 consecutive failures
- Blocks requests for 60 seconds
- Enters half-open state to test recovery

## Thread Safety

### Orchestrator Instance

The `NodeRegistrationOrchestrator` is **NOT thread-safe** at the instance level:

- Each instance should handle one workflow at a time
- For concurrent workflows, create multiple instances
- The orchestrator base class manages workflow state internally

**Recommendation**: Deploy multiple orchestrator instances behind a load balancer for concurrent processing.

### Protocol Implementations

Protocol implementations **MUST** be thread-safe:

**ProtocolReducer**:
- Same instance may process multiple events concurrently
- Treat `ModelReducerState` as immutable (return new instances)
- Avoid instance-level caches that could cause race conditions

**ProtocolEffect**:
- Multiple async tasks may invoke `execute_intent()` simultaneously
- Use `asyncio.Lock` for any shared mutable state
- Ensure underlying clients (Consul, PostgreSQL) are async-safe

See `protocols.py` docstrings for detailed thread safety guidelines.

## Error Handling

### Error Sanitization

All components follow ONEX error sanitization guidelines:

**NEVER include in error messages**:
- Passwords, API keys, tokens, secrets
- Full connection strings with credentials
- PII (names, emails, SSNs, phone numbers)
- Internal IP addresses (in production)
- Raw event payload content

**SAFE to include**:
- Service names (e.g., "consul", "postgres")
- Operation names (e.g., "register", "upsert")
- Correlation IDs (always include for tracing)
- Error codes (e.g., `EnumCoreErrorCode` values)
- Node IDs (UUID, not PII)

### Error Types

The contract defines error handling for specific error types:

| Error Type | Recoverable | Retry Strategy |
|------------|-------------|----------------|
| `ReducerError` | Yes | None (fail immediately) |
| `EffectExecutionError` | Yes | Exponential backoff |
| `AggregationError` | No | None |

## Event Contracts

### Consumed Events

| Topic Pattern | Event Type | Purpose |
|---------------|------------|---------|
| `{env}.{ns}.onex.evt.node-introspection.v1` | `NodeIntrospectionEvent` | Node metadata and capabilities |
| `{env}.{ns}.onex.evt.registry-request-introspection.v1` | `RegistryRequestIntrospectionEvent` | Explicit registration request |
| `{env}.{ns}.onex.internal.runtime-tick.v1` | `RuntimeTick` | Timeout evaluation trigger |

### Published Events

| Topic Pattern | Event Type | Purpose |
|---------------|------------|---------|
| `{env}.{ns}.onex.evt.node-registration-result.v1` | `NodeRegistrationResultEvent` | Final outcome |
| `{env}.{ns}.onex.evt.node-registration-initiated.v1` | `NodeRegistrationInitiated` | Workflow started |
| `{env}.{ns}.onex.evt.node-registration-accepted.v1` | `NodeRegistrationAccepted` | Registration succeeded |
| `{env}.{ns}.onex.evt.node-registration-rejected.v1` | `NodeRegistrationRejected` | Registration failed |
| `{env}.{ns}.onex.evt.node-registration-ack-timed-out.v1` | `NodeRegistrationAckTimedOut` | ACK timeout |
| `{env}.{ns}.onex.evt.node-registration-ack-received.v1` | `NodeRegistrationAckReceived` | ACK received |
| `{env}.{ns}.onex.evt.node-became-active.v1` | `NodeBecameActive` | Node activated |
| `{env}.{ns}.onex.evt.node-liveness-expired.v1` | `NodeLivenessExpired` | Liveness check failed |

## Integration Points

### Dependencies

| Dependency | Type | Module | Status |
|------------|------|--------|--------|
| `reducer_protocol` | Protocol | `ProtocolReducer` | Defined, awaiting impl (OMN-889) |
| `effect_node` | Node | `node_registry_effect` | Placeholder (OMN-890) |
| `projection_reader` | Protocol | `ProtocolProjectionReader` | Not in SPI (OMN-930) |

### External Services

| Service | Purpose | Integration |
|---------|---------|-------------|
| **Consul** | Service discovery registration | Via `ProtocolEffect` |
| **PostgreSQL** | Persistent node registry | Via `ProtocolEffect` |
| **Kafka** | Event bus for communication | Via `ProtocolEventBus` |

## File Structure

```
nodes/node_registration_orchestrator/
├── contract.yaml           # Workflow definition (332 lines)
├── node.py                 # Orchestrator class (123 lines)
├── protocols.py            # ProtocolReducer, ProtocolEffect (349 lines)
├── README.md               # Usage documentation with diagrams
├── __init__.py             # Package exports
└── models/
    ├── __init__.py
    ├── model_intent_execution_result.py
    ├── model_orchestrator_config.py
    ├── model_orchestrator_input.py
    ├── model_orchestrator_output.py
    ├── model_reducer_state.py
    └── model_registration_intent.py  # Discriminated union pattern
```

## Validation Exemptions

This node has documented exemptions in `validation_exemptions.yaml`:

### 1. Domain-Grouped Protocols

**File**: `protocols.py`
**Violation**: `\d+ protocols in one file`
**Reason**: `ProtocolReducer` and `ProtocolEffect` define the complete reducer-effect interface. They are tightly coupled and share type dependencies.

### 2. Discriminated Union Models

**File**: `models/model_registration_intent.py`
**Violation**: `\d+ models in one file`
**Reason**: The 4 models form a cohesive discriminated union pattern. Splitting would break type narrowing.

Both exemptions are justified by architectural patterns documented in CLAUDE.md.

## Implementation Status

| Component | Status | Ticket | Notes |
|-----------|--------|--------|-------|
| Orchestrator Node | Complete | OMN-888 | Declarative pattern |
| Contract Definition | Complete | OMN-888 | Full workflow graph |
| Protocols | Complete | OMN-888 | Thread-safe interfaces |
| Models | Complete | OMN-888 | Input, output, intent, state |
| Effect Node | Placeholder | OMN-890 | Blocked by reducer |
| Reducer Implementation | Pending | OMN-889 | Core logic needed |
| Projection Reader | Pending | OMN-930 | SPI protocol needed |
| Time Injection | Pending | OMN-973 | Contract-driven config |
| Intent Models (Core) | Pending | OMN-912 | Move to omnibase_core |

## Future Work

### OMN-889: Reducer Implementation

The reducer will implement:
- Event deduplication via `processed_node_ids`
- Intent generation with typed payloads
- Rate limiting and batching (optional)

### OMN-890: Effect Node Implementation

The effect node will implement:
- Consul service registration API calls
- PostgreSQL upsert operations
- Result aggregation and error handling

### OMN-930: Projection Reader Integration

Enables the orchestrator to:
- Read current registration state before processing
- Make idempotent decisions based on existing state
- Support the `read_projection` workflow step

### OMN-973: Time Injection Wiring

Enables the orchestrator to:
- Parse `time_injection` from contract
- Receive `now` from `RuntimeTick` events
- Evaluate timeouts with deterministic time

## Related Documentation

- [Protocol Architecture](NODE_REGISTRATION_ORCHESTRATOR_PROTOCOLS.md) - Detailed protocol design
- [Current Node Architecture](CURRENT_NODE_ARCHITECTURE.md) - General ONEX node patterns
- [Error Handling Patterns](../patterns/error_handling_patterns.md) - Error hierarchy
- [Circuit Breaker Thread Safety](CIRCUIT_BREAKER_THREAD_SAFETY.md) - Thread safety patterns
- [Node README](../../src/omnibase_infra/nodes/node_registration_orchestrator/README.md) - Usage and diagrams
