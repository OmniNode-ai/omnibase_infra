> **Navigation**: [Home](../index.md) > [As-Is](INDEX.md) > Layering and Terminology

## Layering and Terminology (As-Is)

This document pins down **what each repo “is responsible for” today**, and how key terms
are used across the stack. The goal is to prevent architecture discussions from mixing
concepts that *share names* but *mean different things*.

### Repo roles (current reality)

- **`omnibase_core`**
  - Defines **ONEX node architecture** (kinds/types), base node classes, contracts/models,
    and core primitives (intents, envelopes, runtime router, etc.).
  - “Active but stable”: only bugfix/tech-debt changes per current project policy.

- **`omnibase_spi`**
  - Defines **SPI-pure protocols** (interfaces) for tooling, containers, event buses,
    and workflow orchestration extensions.
  - “Stable until we decide otherwise.”

- **`omnibase_infra3`**
  - Implements **infrastructure adapters** and “real-world” dependencies:
    event bus implementations (Kafka/in-memory), protocol handlers (db/http/consul/vault),
    and specific infra nodes (e.g., registry effect, dual-registration reducer).
  - “Stable implementation reference.”

### The ONEX 4-node vocabulary (Core)

Core explicitly defines the architectural roles:

- **Effect**: performs external I/O (side effects)
- **Compute**: transforms/validates data
- **Reducer**: state aggregation / FSM style processing
- **Orchestrator**: coordinates multi-step workflows
- **Runtime host**: infrastructure that hosts/coordinates execution

Primary reference: `omnibase_core/src/omnibase_core/enums/enum_node_kind.py`.

### “Same word, different thing” pitfalls

#### 1) “Handler”

There are at least two “handler” concepts in play:

- **Core runtime handler**: Core `EnvelopeRouter` routes `ModelOnexEnvelope` to a handler
  keyed by `EnumHandlerType` (HTTP/DATABASE/KAFKA/etc.).
  - Reference: `omnibase_core/src/omnibase_core/runtime/envelope_router.py`
  - Reference: `omnibase_core/src/omnibase_core/enums/enum_handler_type.py`

- **Infra protocol handler**: Infra3 `RuntimeHostProcess` routes an *operation string*
  like `"db.query"` to a handler registered under a prefix like `"db"`.
  - Reference: `omnibase_infra3/src/omnibase_infra/runtime/runtime_host_process.py`
  - Reference: `omnibase_infra3/src/omnibase_infra/runtime/handler_registry.py`

These are conceptually similar (both route to I/O executors), but they are **not the same
message shape** and not the same routing key.

#### 2) “Envelope”

Core has multiple “envelope-ish” models that serve different purposes:

- **`ModelOnexEnvelope`**: canonical inter-service envelope with routing + request/response semantics.
  - Reference: `omnibase_core/src/omnibase_core/models/core/model_onex_envelope.py`

- **`ModelEventEnvelope[T]`**: generic “event wrapper” type with QoS/tracing features.
  - Reference: `omnibase_core/src/omnibase_core/models/events/model_event_envelope.py`

Infra3 runtime host also uses an “envelope” concept, but it’s a **dict-based operation envelope**
validated by `validate_envelope()` (operation presence, prefix registered, correlation_id normalization).

- Reference: `omnibase_infra3/src/omnibase_infra/runtime/envelope_validator.py`

#### 3) “Event bus”

There are three distinct layers:

- **Core event bus protocol** (topic/key/value/headers + subscribe).
  - Reference: `omnibase_core/src/omnibase_core/protocols/event_bus/protocol_event_bus.py`

- **Infra3 event bus implementations** (Kafka + in-memory) that follow the Core protocol shape.
  - Reference: `omnibase_infra3/src/omnibase_infra/event_bus/kafka_event_bus.py`
  - Reference: `omnibase_infra3/src/omnibase_infra/event_bus/inmemory_event_bus.py`

- **SPI event bus protocols** include a broader surface:
  - A base bus that can publish basic events AND publish/consume envelopes.
    - Reference: `omnibase_spi/src/omnibase_spi/protocols/event_bus/protocol_event_bus_mixin.py`
  - Workflow-oriented event sourcing bus contracts (sequence numbers, idempotency keys, projections).
    - Reference: `omnibase_spi/src/omnibase_spi/protocols/workflow_orchestration/protocol_workflow_event_bus.py`

### What this means for architecture review

When a design doc says “handler”, “envelope”, “event bus”, or “runtime”, we must pin it to
one of these **existing, concrete shapes** (Core runtime router vs Infra runtime host vs SPI workflow bus),
or explicitly introduce a new concept with a new name to avoid ambiguity.
