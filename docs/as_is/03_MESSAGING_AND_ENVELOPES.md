## Messaging and envelopes (As-Is)

This document inventories the **message wrapper shapes** currently present, how they are
used, and how they relate across Core/SPI/Infra.

### Core: `ModelOnexEnvelope` (inter-service envelope)

- Model: `ModelOnexEnvelope`
- Reference: `omnibase_core/src/omnibase_core/models/core/model_onex_envelope.py`

**Key fields (as-is)**:
- Identity: `envelope_id`, `envelope_version`
- Tracing: `correlation_id`, optional `causation_id`
- Routing: `source_node`, optional `target_node`, optional `handler_type`
- Operation: `operation`
- Data: `payload` (serialized dict)
- Response semantics: `is_response`, `success`, `error`

This model is also used as the unit routed by Core’s in-memory runtime router (see `EnvelopeRouter`).

### Core: `ModelEventEnvelope[T]` (generic event wrapper)

- Model: `ModelEventEnvelope[T]`
- Reference: `omnibase_core/src/omnibase_core/models/events/model_event_envelope.py`

This wrapper is a different “envelope family” than `ModelOnexEnvelope` and carries
QoS/tracing metadata in a different shape.

### Core: other typed message models

Core includes other “message-y” models (not exhaustive):

- `ModelMessagePayload` (typed discriminated union message content)
  - Reference: `omnibase_core/src/omnibase_core/models/operations/model_message_payload.py`

### Core protocols: event bus message + headers

Core’s event bus protocol uses a message + headers pair:

- `ProtocolEventBus`:
  - Reference: `omnibase_core/src/omnibase_core/protocols/event_bus/protocol_event_bus.py`
- `ProtocolEventBusHeaders`:
  - Reference: `omnibase_core/src/omnibase_core/protocols/event_bus/protocol_event_bus_headers.py`
- `ProtocolEventMessage`:
  - Reference: `omnibase_core/src/omnibase_core/protocols/event_bus/protocol_event_message.py`

The event bus protocol surface is **topic/key/value + headers** (not `ModelOnexEnvelope` directly).

### Infra3: concrete event bus message models

Infra3 provides concrete message/header models for its event buses:

- `ModelEventMessage`:
  - Reference: `omnibase_infra3/src/omnibase_infra/event_bus/models/model_event_message.py`
- `ModelEventHeaders`:
  - Reference: `omnibase_infra3/src/omnibase_infra/event_bus/models/model_event_headers.py`

These match the Core protocol shape closely (topic/key/value/headers/offset/partition),
but note there are some **type-level differences** (e.g., `schema_version` is a `str` in infra headers).

### Infra3 runtime host “envelopes” (operation envelopes)

Infra3 also uses a dict-based “envelope” for routing I/O operations:

- Validation and correlation-id normalization:
  - Reference: `omnibase_infra3/src/omnibase_infra/runtime/envelope_validator.py`

**As-is fields** (not formalized as a Pydantic model here):
- `operation`: string like `"db.query"`, `"http.post"`, `"consul.register"`
- `payload`: optional dict
- `correlation_id`: UUID or string (normalized to UUID by validator)

This is conceptually different from `ModelOnexEnvelope`.

### SPI: envelope + workflow messaging protocols

SPI defines protocols that explicitly include envelope publishing and workflow event sourcing:

- `ProtocolEventBusBase` includes:
  - basic publish of `ProtocolEventMessage`
  - envelope publish/subscribe/consume of `ModelOnexEnvelope`
  - Reference: `omnibase_spi/src/omnibase_spi/protocols/event_bus/protocol_event_bus_mixin.py`

- Workflow orchestration message shape includes sequence numbers and idempotency keys:
  - Reference: `omnibase_spi/src/omnibase_spi/protocols/workflow_orchestration/protocol_workflow_event_bus.py`

### Summary: “envelope” is plural today

As-is, the system contains multiple message wrapper shapes:

- `ModelOnexEnvelope` (Core runtime router + inter-service envelope semantics)
- `ModelEventEnvelope[T]` (generic event wrapping)
- Event bus messages (`ModelEventMessage` + headers; topic/key/value)
- Infra runtime host dict-envelopes (operation/payload/correlation_id)

Any proposed architecture needs to be explicit about which of these is “the” canonical wrapper
for each plane (runtime routing vs event bus transport vs workflow event sourcing).
