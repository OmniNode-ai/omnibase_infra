# Node Registration Orchestrator

**Type**: ORCHESTRATOR
**Version**: 1.0.0
**Contract Version**: 1.0.0

The Registration Orchestrator coordinates node lifecycle registration workflows by consuming introspection events, computing registration intents via a reducer, and executing those intents through effect nodes.

## Architecture Overview

This orchestrator uses the **declarative pattern** where workflow behavior is 100% driven by `contract.yaml`, not Python code. All workflow logic, retry policies, and result aggregation are handled by the `NodeOrchestrator` base class.

## Workflow Diagrams

### 1. High-Level Event Flow

```mermaid
flowchart TB
    subgraph Input["Input Events"]
        IE[NodeIntrospectionEvent]
        RT[RuntimeTick]
        RRI[RegistryRequestIntrospectionEvent]
    end

    subgraph Orchestrator["Registration Orchestrator Workflow"]
        direction TB
        RI[receive_introspection]
        RP[read_projection]
        ET[evaluate_timeout]
        CI[compute_intents]

        subgraph Parallel["Registration Execution"]
            ECR[execute_consul_registration]
            EPR[execute_postgres_registration]
        end

        AR[aggregate_results]
        PO[publish_outcome]
    end

    subgraph Output["Output Events"]
        NRR[NodeRegistrationResultEvent]
        NRI[NodeRegistrationInitiated]
        NRA[NodeRegistrationAccepted]
        NRJ[NodeRegistrationRejected]
        NRAT[NodeRegistrationAckTimedOut]
        NRAR[NodeRegistrationAckReceived]
        NBA[NodeBecameActive]
        NLE[NodeLivenessExpired]
    end

    subgraph External["External Services"]
        Consul[(Consul)]
        PG[(PostgreSQL)]
    end

    IE --> RI
    RT --> RI
    RRI --> RI

    RI --> RP
    RP --> ET
    ET --> CI
    CI --> ECR
    CI --> EPR
    ECR --> AR
    EPR --> AR
    AR --> PO

    ECR -.-> Consul
    EPR -.-> PG

    PO --> NRR
    PO --> NRI
    PO --> NRA
    PO --> NRJ
    PO --> NRAT
    PO --> NRAR
    PO --> NBA
    PO --> NLE
```

### 2. Detailed Execution Graph with Dependencies

```mermaid
flowchart TD
    subgraph ExecutionGraph["Execution Graph (8 Nodes)"]
        N1["<b>receive_introspection</b><br/><i>type: effect</i><br/>Receive introspection or tick event"]
        N2["<b>read_projection</b><br/><i>type: effect</i><br/>Read registration state from projection"]
        N3["<b>evaluate_timeout</b><br/><i>type: compute</i><br/>Evaluate timeout with injected time"]
        N4["<b>compute_intents</b><br/><i>type: reducer</i><br/>Compute registration intents"]
        N5["<b>execute_consul_registration</b><br/><i>type: effect</i><br/>Execute Consul registration intent"]
        N6["<b>execute_postgres_registration</b><br/><i>type: effect</i><br/>Execute PostgreSQL registration intent"]
        N7["<b>aggregate_results</b><br/><i>type: compute</i><br/>Aggregate registration results"]
        N8["<b>publish_outcome</b><br/><i>type: effect</i><br/>Publish registration outcome event"]
    end

    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N4 --> N6
    N5 --> N7
    N6 --> N7
    N7 --> N8

    style N1 fill:#e1f5fe
    style N2 fill:#e1f5fe
    style N3 fill:#fff3e0
    style N4 fill:#f3e5f5
    style N5 fill:#e1f5fe
    style N6 fill:#e1f5fe
    style N7 fill:#fff3e0
    style N8 fill:#e1f5fe
```

**Legend**:
- Blue (effect): External I/O operations
- Orange (compute): Pure transformations
- Purple (reducer): State aggregation

### 3. Reducer State Machine

```mermaid
stateDiagram-v2
    [*] --> Initial: ModelReducerState.initial()

    Initial --> Processing: Event received
    Processing --> Updated: Valid event
    Processing --> Initial: Duplicate node_id

    state Processing {
        [*] --> ValidateEvent
        ValidateEvent --> CheckDuplicate: Event valid
        ValidateEvent --> Error: Invalid event
        CheckDuplicate --> GenerateIntents: New node
        CheckDuplicate --> Skip: Already processed
        GenerateIntents --> UpdateState
    }

    Updated --> Processing: Next event

    note right of Initial
        processed_node_ids: frozenset()
        pending_registrations: 0
        last_event_timestamp: None
    end note

    note right of Updated
        processed_node_ids: frozenset({node_id})
        pending_registrations: +2 per event
        last_event_timestamp: event.timestamp
    end note
```

### 4. Intent Generation Flow

```mermaid
flowchart LR
    subgraph Input["Introspection Event"]
        NID[node_id]
        NT[node_type]
        NV[node_version]
        CAP[capabilities]
        EP[endpoints]
        CID[correlation_id]
    end

    subgraph Reducer["ProtocolReducer.reduce()"]
        direction TB
        VAL[Validate Event]
        DUP[Check Duplicate]
        GEN[Generate Intents]
    end

    subgraph Intents["Generated Intents"]
        CI["ModelConsulRegistrationIntent<br/><i>kind: 'consul'</i><br/><i>operation: 'register'</i>"]
        PI["ModelPostgresUpsertIntent<br/><i>kind: 'postgres'</i><br/><i>operation: 'upsert'</i>"]
    end

    NID --> VAL
    NT --> VAL
    NV --> VAL
    CAP --> VAL
    EP --> VAL
    CID --> VAL

    VAL --> DUP
    DUP --> GEN

    GEN --> CI
    GEN --> PI
```

### 5. Effect Execution and Result Aggregation

```mermaid
flowchart TB
    subgraph Intents["Registration Intents"]
        CI[ConsulRegistrationIntent]
        PI[PostgresUpsertIntent]
    end

    subgraph Effect["ProtocolEffect.execute_intent()"]
        CE[Execute Consul]
        PE[Execute Postgres]
    end

    subgraph Results["Intent Execution Results"]
        CR["ModelIntentExecutionResult<br/>intent_kind: 'consul'<br/>success: bool<br/>execution_time_ms: float"]
        PR["ModelIntentExecutionResult<br/>intent_kind: 'postgres'<br/>success: bool<br/>execution_time_ms: float"]
    end

    subgraph Aggregation["Result Aggregation"]
        AGG{All Success?}
        SUCCESS["status: 'success'<br/>consul_applied: true<br/>postgres_applied: true"]
        PARTIAL["status: 'partial'<br/>one applied, one failed"]
        FAILED["status: 'failed'<br/>consul_applied: false<br/>postgres_applied: false"]
    end

    subgraph Output["Orchestrator Output"]
        OUT[ModelOrchestratorOutput]
    end

    CI --> CE
    PI --> PE
    CE --> CR
    PE --> PR
    CR --> AGG
    PR --> AGG

    AGG -->|Both succeed| SUCCESS
    AGG -->|One succeeds| PARTIAL
    AGG -->|Both fail| FAILED

    SUCCESS --> OUT
    PARTIAL --> OUT
    FAILED --> OUT
```

### 6. Error Handling and Retry Flow

```mermaid
flowchart TB
    subgraph Workflow["Workflow Execution"]
        START([Start Step])
        EXEC[Execute Step]
        CHECK{Success?}
        DONE([Step Complete])
    end

    subgraph WorkflowRetry["Workflow Step Retry<br/>(coordination_rules.max_retries: 3)"]
        WR_CHECK{Retries < 3?}
        WR_RETRY[Retry Entire Step]
        WR_FAIL[Mark Workflow Failed]
    end

    subgraph ErrorRetry["Error-Level Retry<br/>(error_handling.retry_policy)"]
        ER_CHECK{Retryable Error?}
        ER_BACKOFF["Exponential Backoff<br/>100ms -> 200ms -> 400ms"]
        ER_CIRCUIT{Circuit Open?}
        ER_FAIL[Return Error Result]
    end

    subgraph CircuitBreaker["Circuit Breaker<br/>(failure_threshold: 5)"]
        CB_COUNT[Increment Failure Count]
        CB_CHECK{Count >= 5?}
        CB_OPEN[Open Circuit]
        CB_TIMEOUT["Wait 60s<br/>reset_timeout_ms"]
        CB_HALF[Half-Open State]
    end

    START --> EXEC
    EXEC --> CHECK
    CHECK -->|Yes| DONE
    CHECK -->|No| ER_CHECK

    ER_CHECK -->|Yes| ER_BACKOFF
    ER_CHECK -->|No| ER_FAIL

    ER_BACKOFF --> ER_CIRCUIT
    ER_CIRCUIT -->|No| EXEC
    ER_CIRCUIT -->|Yes| WR_CHECK

    ER_FAIL --> CB_COUNT
    CB_COUNT --> CB_CHECK
    CB_CHECK -->|Yes| CB_OPEN
    CB_CHECK -->|No| WR_CHECK

    CB_OPEN --> CB_TIMEOUT
    CB_TIMEOUT --> CB_HALF
    CB_HALF --> EXEC

    WR_CHECK -->|Yes| WR_RETRY
    WR_CHECK -->|No| WR_FAIL
    WR_RETRY --> EXEC

    style ER_BACKOFF fill:#fff3e0
    style CB_OPEN fill:#ffebee
    style WR_FAIL fill:#ffebee
```

### 7. Complete Workflow State Transitions

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> ReceivingEvent: Event arrives

    state ReceivingEvent {
        [*] --> WaitingForEvent
        WaitingForEvent --> EventReceived: introspection/tick
        EventReceived --> [*]
    }

    ReceivingEvent --> ReadingProjection: Event valid

    state ReadingProjection {
        [*] --> QueryProjection
        QueryProjection --> ProjectionRead: Success
        QueryProjection --> ProjectionError: Failure
        ProjectionError --> QueryProjection: Retry
        ProjectionRead --> [*]
    }

    ReadingProjection --> EvaluatingTimeout: Projection loaded

    state EvaluatingTimeout {
        [*] --> CheckTimeout
        CheckTimeout --> TimeoutExpired: Timeout detected
        CheckTimeout --> NoTimeout: Within threshold
        TimeoutExpired --> [*]
        NoTimeout --> [*]
    }

    EvaluatingTimeout --> ComputingIntents: Timeout evaluated

    state ComputingIntents {
        [*] --> ReducerProcessing
        ReducerProcessing --> IntentsGenerated: New node
        ReducerProcessing --> NoIntents: Duplicate/filtered
        IntentsGenerated --> [*]
        NoIntents --> [*]
    }

    ComputingIntents --> ExecutingRegistrations: Intents ready

    state ExecutingRegistrations {
        [*] --> ConsulRegistration
        [*] --> PostgresRegistration
        ConsulRegistration --> ConsulDone
        PostgresRegistration --> PostgresDone
        ConsulDone --> [*]
        PostgresDone --> [*]
    }

    ExecutingRegistrations --> AggregatingResults: Both complete

    state AggregatingResults {
        [*] --> CollectResults
        CollectResults --> DetermineStatus
        DetermineStatus --> StatusSuccess: All passed
        DetermineStatus --> StatusPartial: Some passed
        DetermineStatus --> StatusFailed: None passed
        StatusSuccess --> [*]
        StatusPartial --> [*]
        StatusFailed --> [*]
    }

    AggregatingResults --> PublishingOutcome: Results aggregated

    state PublishingOutcome {
        [*] --> EmitEvent
        EmitEvent --> EventPublished
        EventPublished --> [*]
    }

    PublishingOutcome --> Idle: Complete
```

## Data Models

### Input Model: `ModelOrchestratorInput`

| Field | Type | Description |
|-------|------|-------------|
| `introspection_event` | `ModelNodeIntrospectionEvent` | The introspection event to process |
| `correlation_id` | `UUID` | Correlation ID for distributed tracing |

### Output Model: `ModelOrchestratorOutput`

| Field | Type | Description |
|-------|------|-------------|
| `correlation_id` | `UUID` | Correlation ID for tracing |
| `status` | `Literal["success", "partial", "failed"]` | Overall workflow status |
| `consul_applied` | `bool` | Whether Consul registration succeeded |
| `postgres_applied` | `bool` | Whether PostgreSQL registration succeeded |
| `consul_error` | `str \| None` | Consul error message if any |
| `postgres_error` | `str \| None` | PostgreSQL error message if any |
| `intent_results` | `list[ModelIntentExecutionResult]` | Results of each intent execution |
| `total_execution_time_ms` | `float` | Total workflow execution time |

### Intent Models

#### `ModelConsulRegistrationIntent`
```python
kind: Literal["consul"]  # Discriminator
operation: str           # "register", "deregister"
node_id: UUID
correlation_id: UUID
payload: ModelConsulIntentPayload
```

#### `ModelPostgresUpsertIntent`
```python
kind: Literal["postgres"]  # Discriminator
operation: str             # "upsert", "delete"
node_id: UUID
correlation_id: UUID
payload: ModelPostgresIntentPayload
```

## Configuration

### Coordination Rules

| Setting | Value | Description |
|---------|-------|-------------|
| `execution_mode` | `sequential` | Steps execute in order |
| `parallel_execution_allowed` | `false` | No parallel branches |
| `max_retries` | `3` | Workflow step retry count |
| `timeout_ms` | `30000` | Overall workflow timeout |
| `checkpoint_enabled` | `true` | Enables recovery checkpoints |
| `rollback_enabled` | `true` | Enables rollback on failure |

### Error Handling

| Setting | Value | Description |
|---------|-------|-------------|
| `retry_policy.max_retries` | `3` | Error-level retry count |
| `retry_policy.initial_delay_ms` | `100` | Initial backoff delay |
| `retry_policy.max_delay_ms` | `5000` | Maximum backoff delay |
| `retry_policy.exponential_base` | `2` | Backoff multiplier |
| `circuit_breaker.enabled` | `true` | Circuit breaker active |
| `circuit_breaker.failure_threshold` | `5` | Failures before open |
| `circuit_breaker.reset_timeout_ms` | `60000` | Time before half-open |

## Dependencies

| Name | Type | Description |
|------|------|-------------|
| `reducer_protocol` | Protocol | For computing registration intents |
| `effect_node` | Node | For executing registration operations |
| `projection_reader` | Protocol | For reading current state (OMN-930) |

## Events

### Consumed Events

| Topic Pattern | Event Type | Description |
|---------------|------------|-------------|
| `{env}.{namespace}.onex.evt.node-introspection.v1` | `NodeIntrospectionEvent` | Node introspection data |
| `{env}.{namespace}.onex.evt.registry-request-introspection.v1` | `RegistryRequestIntrospectionEvent` | Registry request |
| `{env}.{namespace}.onex.internal.runtime-tick.v1` | `RuntimeTick` | Internal tick for timeout evaluation |

### Published Events

| Topic Pattern | Event Type | Description |
|---------------|------------|-------------|
| `{env}.{namespace}.onex.evt.node-registration-result.v1` | `NodeRegistrationResultEvent` | Final registration result |
| `{env}.{namespace}.onex.evt.node-registration-initiated.v1` | `NodeRegistrationInitiated` | Registration started |
| `{env}.{namespace}.onex.evt.node-registration-accepted.v1` | `NodeRegistrationAccepted` | Registration accepted |
| `{env}.{namespace}.onex.evt.node-registration-rejected.v1` | `NodeRegistrationRejected` | Registration rejected |
| `{env}.{namespace}.onex.evt.node-registration-ack-timed-out.v1` | `NodeRegistrationAckTimedOut` | ACK timeout |
| `{env}.{namespace}.onex.evt.node-registration-ack-received.v1` | `NodeRegistrationAckReceived` | ACK received |
| `{env}.{namespace}.onex.evt.node-became-active.v1` | `NodeBecameActive` | Node activated |
| `{env}.{namespace}.onex.evt.node-liveness-expired.v1` | `NodeLivenessExpired` | Liveness expired |

## Coroutine Safety

This orchestrator is **NOT coroutine-safe** for concurrent workflow invocations. Each instance should handle one workflow at a time. For concurrent workflows, create multiple instances.

## Limitations & Implementation Status

This node is part of the MVP implementation for OMN-888. The following limitations apply:

### Current Limitations

| Limitation | Ticket | Description |
|------------|--------|-------------|
| Effect Node Integration | OMN-890 | `NodeRegistryEffect` is implemented in `nodes/effects/registry_effect.py`. The `node_registry_effect` module re-exports from this location. Integration with orchestrator workflow pending reducer implementation. |
| Reducer Not Implemented | OMN-889 | `ProtocolReducer` is defined but no concrete implementation exists. Intent computation is pending. |
| Projection Reader Not Wired | OMN-930 | `ProtocolProjectionReader` protocol does not exist in `omnibase_spi.protocols`. The `read_projection` workflow step cannot execute. |
| Time Injection Not Wired | OMN-973 | Contract declares time injection but orchestrator does not parse or use it. Timeout evaluation uses implicit dispatch context. |
| Intent Models in Infra | OMN-912 | Intent models are currently in `omnibase_infra`. Should be moved to `omnibase_core` for broader reuse. |

### Implementation Status

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Orchestrator Node | **Complete** | `node.py` | Declarative pattern, extends `NodeOrchestrator` |
| Contract | **Complete** | `contract.yaml` | Full workflow definition with coordination rules |
| Protocols | **Complete** | `protocols.py` | `ProtocolReducer`, `ProtocolEffect` defined |
| Models | **Complete** | `models/` | Input, output, intent, state models |
| README | **Complete** | `README.md` | This file |
| Effect Node | **Complete** | `nodes/effects/registry_effect.py` | Alias at `nodes/node_registry_effect/` |
| Reducer Impl | **Pending** | N/A | No implementation yet (OMN-889) |
| Projection Reader | **Pending** | N/A | SPI protocol needed (OMN-930) |

### What Works Today

1. **Contract Parsing**: The contract.yaml is valid and fully defines the workflow
2. **Model Validation**: All input/output models work with Pydantic validation
3. **Protocol Definitions**: Type contracts for reducer and effect are complete
4. **Workflow Structure**: Execution graph with dependencies is defined

### What Does NOT Work Today

1. **End-to-End Registration**: Cannot register nodes (reducer → effect integration pending)
2. **Intent Computation**: Cannot generate intents (reducer not implemented)
3. **Projection Reading**: Cannot read current state (protocol not in SPI)
4. **Timeout Evaluation**: Uses implicit time, not contract-driven injection

Note: The effect node (`NodeRegistryEffect`) is fully implemented and tested. The
blocker for end-to-end registration is the reducer implementation (OMN-889) which
must generate intents that the effect node will execute.

## Related Tickets

- **OMN-888**: Infrastructure MVP Node Registration Orchestrator Workflow
- **OMN-889**: Reducer Implementation (pending)
- **OMN-890**: Effect Node Implementation (pending)
- **OMN-912**: Intent Models in omnibase_core (pending)
- **OMN-930**: Projection Reader Integration
- **OMN-973**: Time Injection Context for Timeout Evaluation

## Related Documentation

- [Node Architecture](../../../../docs/architecture/REGISTRATION_ORCHESTRATOR_ARCHITECTURE.md) - Complete architectural design
- [Protocol Architecture](../../../../docs/architecture/NODE_REGISTRATION_ORCHESTRATOR_PROTOCOLS.md) - Detailed protocol design
- [Validation Exemptions](../../validation/validation_exemptions.yaml) - Exemption for domain-grouped protocols
