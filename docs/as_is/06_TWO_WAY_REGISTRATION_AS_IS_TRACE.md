## Two-way registration workflow (As-Is Trace)

This document traces the **current** 2-way registration flow as it exists in code today.
It is meant as a concrete grounding example for the “shape of everything” discussion.

### Primary components in Infra3

#### Reducer: builds typed intents (no I/O)

- `NodeDualRegistrationReducer`
  - Reference: `omnibase_infra3/src/omnibase_infra/nodes/reducers/node_dual_registration_reducer.py`

**Emits Core intents**:
- `ModelConsulRegisterIntent` (`kind="consul.register"`)
- `ModelPostgresUpsertRegistrationIntent` (`kind="postgres.upsert_registration"`)

Core intent definitions:
- `omnibase_core/src/omnibase_core/models/intents/__init__.py`
- `omnibase_core/src/omnibase_core/models/intents/model_consul_register_intent.py`
- `omnibase_core/src/omnibase_core/models/intents/model_postgres_upsert_registration_intent.py`

#### Effect: executes external registration (I/O)

- `node_registry_effect` (effect node)
  - Contract: `omnibase_infra3/src/omnibase_infra/nodes/node_registry_effect/v1_0_0/contract.yaml`
  - Implementation: `omnibase_infra3/src/omnibase_infra/nodes/node_registry_effect/v1_0_0/node.py`

**Dependencies resolved by contract/container**:
- Consul handler (`HandlerConsul`)
- DB handler (`HandlerDb`)
- Event bus (`ProtocolEventBus` in the contract, implemented by KafkaEventBus/InMemoryEventBus)

### External systems (executed by the Effect node)

- Consul (service discovery registration)
- PostgreSQL (persistent registration record)

### Messaging surfaces used (as-is)

There are at least two messaging planes involved in the ecosystem:

- Event bus (topic/key/value/headers) used by infra event bus implementations
  - `omnibase_infra3/src/omnibase_infra/event_bus/kafka_event_bus.py`
  - `omnibase_infra3/src/omnibase_infra/event_bus/inmemory_event_bus.py`

- Infra runtime host “operation envelopes” for low-level I/O handlers (db/http/consul/vault)
  - `omnibase_infra3/src/omnibase_infra/runtime/runtime_host_process.py`

The registry effect node can directly call handler dependencies (per its implementation),
and can also publish bus messages.

### What’s important about this trace (as-is facts)

- The reducer/effect split already exists and is already aligned with Core’s typed intent system.
- Infra3’s runtime host is a separate mechanism for routing low-level I/O operations by string prefix;
  it is not the same as Core’s `ModelOnexEnvelope` + `EnvelopeRouter` runtime.
- Any future “two-way registration architecture” needs to state explicitly:
  - where the reducer runs
  - where the effect runs
  - which runtime plane dispatches which messages
  - which envelope/message wrapper is used at each boundary
