> **Navigation**: [Home](../index.md) > Architecture > Overview

# ONEX Architecture Overview

This document provides a high-level overview of the ONEX (OmniNode Execution) architecture used in `omnibase_infra`.

## Design Philosophy

ONEX is built on three core principles:

1. **Contract-Driven**: All behavior is declared in YAML contracts, not hardcoded in Python
2. **Declarative Nodes**: Node classes contain zero custom logic - they extend base classes
3. **Separation of Concerns**: Each node archetype has a single, well-defined responsibility

## System Architecture

### ASCII Diagram

The following diagram shows the complete ONEX runtime architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ONEX RUNTIME                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         EVENT BUS (Kafka)                            │    │
│  │    Topics: introspection, registration, heartbeat, commands          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│            │                    │                    │                       │
│            ▼                    ▼                    ▼                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      MESSAGE DISPATCH ENGINE                          │   │
│  │    Routes events to orchestrators based on topic patterns             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│            │                                                                 │
│            ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         ORCHESTRATOR LAYER                            │   │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────┐  │   │
│  │  │  Registration      │  │   Workflow         │  │   Custom       │  │   │
│  │  │  Orchestrator      │  │   Orchestrator     │  │   Orchestrator │  │   │
│  │  └────────────────────┘  └────────────────────┘  └────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│            │                         │                                       │
│            │ handler_routing         │ emit intents                          │
│            ▼                         ▼                                       │
│  ┌─────────────────────┐   ┌─────────────────────┐                          │
│  │    HANDLER LAYER    │   │    REDUCER LAYER    │                          │
│  │  ┌───────────────┐  │   │  ┌───────────────┐  │                          │
│  │  │ HandlerNode-  │  │   │  │ Registration  │  │                          │
│  │  │ Introspected  │  │   │  │ Reducer       │  │                          │
│  │  └───────────────┘  │   │  │ (FSM-driven)  │  │                          │
│  │  ┌───────────────┐  │   │  └───────────────┘  │                          │
│  │  │ HandlerNode-  │  │   └─────────────────────┘                          │
│  │  │ Acked         │  │             │                                       │
│  │  └───────────────┘  │             │ ModelIntent                           │
│  └─────────────────────┘             ▼                                       │
│                            ┌─────────────────────┐                          │
│                            │    EFFECT LAYER     │                          │
│                            │  ┌───────────────┐  │                          │
│                            │  │ Consul Effect │  │                          │
│                            │  └───────────────┘  │                          │
│                            │  ┌───────────────┐  │                          │
│                            │  │ Postgres      │  │                          │
│                            │  │ Effect        │  │                          │
│                            │  └───────────────┘  │                          │
│                            └─────────────────────┘                          │
│                                      │                                       │
└──────────────────────────────────────│───────────────────────────────────────┘
                                       ▼
                      ┌─────────────────────────────────┐
                      │      EXTERNAL SERVICES          │
                      │  ┌─────────┐  ┌─────────────┐   │
                      │  │ Consul  │  │ PostgreSQL  │   │
                      │  └─────────┘  └─────────────┘   │
                      │  ┌─────────┐  ┌─────────────┐   │
                      │  │  Vault  │  │    Kafka    │   │
                      │  └─────────┘  └─────────────┘   │
                      └─────────────────────────────────┘
```

### Mermaid Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e3f2fd'}}}%%
flowchart TB
    accTitle: ONEX Runtime System Architecture
    accDescr: Complete system architecture diagram showing the ONEX runtime layers. Events flow from Kafka through the Message Dispatch Engine to Orchestrators, which route to Handlers and Reducers. Reducers emit Intents that are executed by Effect nodes, which interact with external services like Consul, PostgreSQL, Vault, and Kafka.

    subgraph ONEX["ONEX Runtime"]
        subgraph EventBus["Event Bus (Kafka)"]
            K[Topics: introspection, registration,<br/>heartbeat, commands]
        end

        subgraph Dispatch["Message Dispatch Engine"]
            MDE[Routes events to orchestrators<br/>based on topic patterns]
        end

        subgraph OrchestratorLayer["Orchestrator Layer"]
            REG_ORCH[Registration<br/>Orchestrator]
            WF_ORCH[Workflow<br/>Orchestrator]
            CUSTOM_ORCH[Custom<br/>Orchestrator]
        end

        subgraph HandlerLayer["Handler Layer"]
            H_INTRO[HandlerNode-<br/>Introspected]
            H_ACKED[HandlerNode-<br/>Acked]
        end

        subgraph ReducerLayer["Reducer Layer"]
            RED[Registration Reducer<br/>FSM-driven]
        end

        subgraph EffectLayer["Effect Layer"]
            CONSUL_EFF[Consul Effect]
            PG_EFF[Postgres Effect]
        end
    end

    subgraph External["External Services"]
        CONSUL[(Consul)]
        PG[(PostgreSQL)]
        VAULT[(Vault)]
        KAFKA[(Kafka)]
    end

    K --> MDE
    MDE --> REG_ORCH
    MDE --> WF_ORCH
    MDE --> CUSTOM_ORCH

    REG_ORCH -->|handler_routing| H_INTRO
    REG_ORCH -->|handler_routing| H_ACKED
    REG_ORCH -->|emit intents| RED

    RED -->|ModelIntent| CONSUL_EFF
    RED -->|ModelIntent| PG_EFF

    CONSUL_EFF --> CONSUL
    PG_EFF --> PG
    EffectLayer --> VAULT
    EffectLayer --> KAFKA
```

## The Four Node Archetypes

ONEX organizes all processing into four node types, each with a specific role:

### 1. ORCHESTRATOR (Workflow Coordination)

**Purpose**: Coordinates workflows by routing events to handlers and managing execution flow.

**Key Characteristics**:
- ONLY node type that can **publish events**
- Routes events to handlers via `handler_routing` in contract
- Coordinates multi-step workflows via `execution_graph`
- Owns no FSM logic (delegates to reducers)

**Example Use Cases**:
- Node registration workflow
- Multi-service transactions
- Saga pattern implementations

```yaml
# contract.yaml for orchestrator
node_type: "ORCHESTRATOR_GENERIC"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model: { name: "ModelNodeIntrospectionEvent" }
      handler: { name: "HandlerNodeIntrospected" }
```

### 2. REDUCER (State + FSM)

**Purpose**: Manages state transitions via FSM and emits intents for effects.

**Key Characteristics**:
- Pure function: `reduce(state, event) → (new_state, intents)`
- No side effects
- FSM transitions defined in contract
- Emits intents for Effect layer to execute

**Example Use Cases**:
- Registration state machine
- Order fulfillment state
- Approval workflows

```yaml
# contract.yaml for reducer
node_type: "REDUCER_GENERIC"
state_machine:
  initial_state: "idle"
  transitions:
    - from_state: "idle"
      to_state: "pending"
      trigger: "request_received"
```

### 3. COMPUTE (Pure Transformations)

**Purpose**: Performs deterministic computations with no side effects.

**Key Characteristics**:
- Pure functions (referentially transparent)
- Same input always produces same output
- No I/O operations

**Example Use Cases**:
- Data validation
- Format conversion
- Business rule evaluation

```yaml
# contract.yaml for compute
node_type: "COMPUTE_GENERIC"
validation_rules:
  - rule_id: "ARCH-001"
    detection_strategy: { type: "ast_pattern" }
```

### 4. EFFECT (External I/O)

**Purpose**: Handles all interactions with external systems.

**Key Characteristics**:
- Named by **capability**, not by technology
- Executes intents emitted by reducers
- May have side effects (database writes, API calls)
- Pluggable handler implementations

**Example Use Cases**:
- Database operations
- Service discovery (Consul)
- Secret management (Vault)
- HTTP API calls

```yaml
# contract.yaml for effect
node_type: "EFFECT_GENERIC"
capabilities:
  - name: "registration.storage"
io_operations:
  - operation: "upsert_record"
```

## Data Flow Patterns

### Event-Driven Flow

#### ASCII Version

```
Event arrives on Kafka topic
         │
         ▼
┌─────────────────────┐
│   Orchestrator      │  1. Receives event via message dispatch
│   (coordinator)     │  2. Routes to handler based on payload type
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Handler           │  3. Processes event
│   (business logic)  │  4. Returns ModelHandlerOutput with events/intents
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Orchestrator      │  5. Publishes returned events
│   (publisher)       │  6. Routes intents to effects
└─────────────────────┘
```

#### Mermaid Version

```mermaid
flowchart TB
    accTitle: Event-Driven Flow
    accDescr: Flow diagram showing how events are processed in ONEX. Events arrive on Kafka, are received by the Orchestrator which routes them to appropriate Handlers based on payload type. Handlers process the event and return a ModelHandlerOutput with events and intents. The Orchestrator then publishes events and routes intents to Effect nodes.

    KAFKA[Event arrives<br/>on Kafka topic] --> ORCH1[Orchestrator<br/>coordinator]

    ORCH1 -->|1. Receives event<br/>2. Routes by payload type| HANDLER[Handler<br/>business logic]

    HANDLER -->|3. Processes event<br/>4. Returns ModelHandlerOutput| ORCH2[Orchestrator<br/>publisher]

    ORCH2 -->|5. Publishes events| KAFKA_OUT[Kafka Topics]
    ORCH2 -->|6. Routes intents| EFFECTS[Effect Nodes]
```

### Intent Flow (Reducer -> Effect)

#### ASCII Version

```
┌─────────────────────┐
│   Reducer           │  1. Receives event
│   (FSM-driven)      │  2. Transitions state
└─────────────────────┘  3. Emits intent(s)
         │
         │ ModelIntent(intent_type="extension",
         │             payload.intent_type="consul.register")
         ▼
┌─────────────────────┐
│   Effect Node       │  4. Receives intent
│   (external I/O)    │  5. Executes via handler
└─────────────────────┘  6. Returns result
         │
         ▼
┌─────────────────────┐
│   External Service  │  7. Actual I/O operation
│   (Consul/Postgres) │
└─────────────────────┘
```

#### Mermaid Version

```mermaid
flowchart TB
    accTitle: Intent Flow from Reducer to Effect
    accDescr: Flow diagram showing how intents flow from Reducers to Effect nodes. The Reducer receives an event, transitions its FSM state, and emits intents wrapped in ModelIntent with an extension intent_type and a payload containing the specific routing key. Effect nodes receive the intent, execute via their handlers, and interact with external services like Consul or PostgreSQL.

    REDUCER[Reducer<br/>FSM-driven] -->|1. Receives event<br/>2. Transitions state<br/>3. Emits intents| INTENT["ModelIntent<br/>intent_type='extension'<br/>payload.intent_type='consul.register'"]

    INTENT --> EFFECT[Effect Node<br/>external I/O]

    EFFECT -->|4. Receives intent<br/>5. Executes via handler<br/>6. Returns result| EXTERNAL[(External Service<br/>Consul/PostgreSQL)]

    EXTERNAL -->|7. Actual I/O operation| RESULT[Operation Complete]
```

## Contract-Driven Architecture

### What Goes in `contract.yaml`

| Section | Purpose | Used By |
|---------|---------|---------|
| `node_type` | Declares archetype | All nodes |
| `input_model` / `output_model` | Typed I/O | All nodes |
| `handler_routing` | Event → Handler mapping | Orchestrators, Effects |
| `state_machine` | FSM definition | Reducers |
| `execution_graph` | Workflow DAG | Orchestrators |
| `io_operations` | External operations | Effects |
| `capabilities` | What the node provides | Effects, Computes |

### What Goes in `node.py`

**Almost nothing!** Node classes are declarative:

```python
class MyNode(NodeArchetype):
    """Docstring describing the capability."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

__all__ = ["MyNode"]
```

All logic is driven by the contract. The runtime reads the contract and wires everything.

## Handler Architecture

Handlers implement business logic and are wired via contracts.

### ASCII Version

```
┌───────────────────────────────────────────────────────────────┐
│                    HANDLER PLUGIN SYSTEM                       │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  contract.yaml                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ handler_routing:                                         │  │
│  │   handlers:                                              │  │
│  │     - event_model: "ModelIntrospectionEvent"            │  │
│  │       handler: "HandlerNodeIntrospected"                │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │            HandlerPluginLoader                           │  │
│  │   - Reads contract.yaml                                  │  │
│  │   - Validates handler protocol (5 methods)               │  │
│  │   - Registers handlers in registry                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                 Handler Instance                         │  │
│  │   Required methods:                                      │  │
│  │   - handler_type (property)                              │  │
│  │   - initialize(config) (async)                           │  │
│  │   - shutdown() (async)                                   │  │
│  │   - handle(envelope) (async)                             │  │
│  │   - describe() (sync)                                    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Mermaid Version

```mermaid
flowchart TB
    accTitle: Handler Plugin System Architecture
    accDescr: Architecture diagram showing the Handler Plugin System. Contract.yaml files declare handler routing rules mapping event models to handler classes. The HandlerPluginLoader reads contracts, validates that handlers implement the required 5-method protocol, and registers them in a registry. Handler instances implement handler_type property, initialize, shutdown, handle, and describe methods.

    subgraph Contract["contract.yaml"]
        ROUTING["handler_routing:<br/>  handlers:<br/>    - event_model: ModelIntrospectionEvent<br/>      handler: HandlerNodeIntrospected"]
    end

    subgraph Loader["HandlerPluginLoader"]
        L1[Reads contract.yaml]
        L2[Validates handler protocol<br/>5 required methods]
        L3[Registers handlers<br/>in registry]
    end

    subgraph Handler["Handler Instance"]
        M1["handler_type (property)"]
        M2["initialize(config) (async)"]
        M3["shutdown() (async)"]
        M4["handle(envelope) (async)"]
        M5["describe() (sync)"]
    end

    Contract --> L1
    L1 --> L2
    L2 --> L3
    L3 --> Handler
```

## Key Design Constraints

| Constraint | Reason |
|------------|--------|
| **Handlers cannot publish events** | Only orchestrators have bus access |
| **Orchestrators have no FSM logic** | Reducers own state transitions |
| **No `Any` types** | Type safety, enforced by CI |
| **Effects named by capability** | "registration.storage" not "postgres" ([naming guide](../reference/contracts.md#capability-naming-convention)) |
| **Immutable state models** | Use `with_*` methods for transitions |

## Package Layering

### ASCII Version

```
┌─────────────────────────────────────────────────┐
│              omnibase_infra                      │
│   Infrastructure implementations                 │
│   - Handlers (Consul, DB, Vault, HTTP)          │
│   - Nodes (Registration, Effects)                │
│   - Runtime (Dispatchers, Loaders)              │
└─────────────────────────────────────────────────┘
                      │
                      │ depends on
                      ▼
┌─────────────────────────────────────────────────┐
│               omnibase_spi                       │
│   Service Provider Interface (protocols)         │
│   - ProtocolHandler                              │
│   - ProtocolDispatcher                           │
│   - ProtocolProjectionReader                     │
└─────────────────────────────────────────────────┘
                      │
                      │ depends on
                      ▼
┌─────────────────────────────────────────────────┐
│              omnibase_core                       │
│   Core models and base classes                   │
│   - NodeEffect, NodeCompute, NodeReducer        │
│   - NodeOrchestrator                             │
│   - ModelONEXContainer                           │
│   - Core enums and types                         │
└─────────────────────────────────────────────────┘
```

### Mermaid Version

```mermaid
flowchart TB
    accTitle: ONEX Package Layering
    accDescr: Dependency diagram showing the three-layer package architecture. omnibase_infra at the top contains infrastructure implementations including handlers for Consul, DB, Vault, and HTTP, plus nodes and runtime components. It depends on omnibase_spi which provides Service Provider Interface protocols like ProtocolHandler, ProtocolDispatcher, and ProtocolProjectionReader. Both depend on omnibase_core at the bottom which provides core models and base classes including NodeEffect, NodeCompute, NodeReducer, NodeOrchestrator, ModelONEXContainer, and core enums.

    subgraph INFRA["omnibase_infra"]
        I1[Infrastructure implementations]
        I2["Handlers (Consul, DB, Vault, HTTP)"]
        I3["Nodes (Registration, Effects)"]
        I4["Runtime (Dispatchers, Loaders)"]
    end

    subgraph SPI["omnibase_spi"]
        S1[Service Provider Interface<br/>protocols]
        S2[ProtocolHandler]
        S3[ProtocolDispatcher]
        S4[ProtocolProjectionReader]
    end

    subgraph CORE["omnibase_core"]
        C1[Core models and base classes]
        C2["NodeEffect, NodeCompute"]
        C3["NodeReducer, NodeOrchestrator"]
        C4[ModelONEXContainer]
        C5[Core enums and types]
    end

    INFRA -->|depends on| SPI
    SPI -->|depends on| CORE
```

## Related Documentation

| Topic | Document |
|-------|----------|
| **Coding standards** | [CLAUDE.md](../../CLAUDE.md) - **authoritative source** for all rules |
| Quick start | [Getting Started](../getting-started/quickstart.md) |
| Node archetypes | [Node Archetypes Reference](../reference/node-archetypes.md) |
| Contract format | [Contract.yaml Reference](../reference/contracts.md) |
| Registration example | [2-Way Registration Walkthrough](../guides/registration-example.md) |
| Implementation patterns | [Pattern Documentation](../patterns/README.md) |
| Handler protocol | [Handler Plugin Loader](../patterns/handler_plugin_loader.md) |

> **Note**: Documentation in `docs/` provides explanations, examples, and tutorials. For authoritative coding rules and standards, always refer to [CLAUDE.md](../../CLAUDE.md).
