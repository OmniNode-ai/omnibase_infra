> **Navigation**: [Home](../index.md) > [As-Is](INDEX.md) > Event Bus Shapes

## Event bus shapes (As-Is)

This document captures the “as-is” event bus **interfaces** and **implementations**.

### Core: `ProtocolEventBus` (primary pub/sub abstraction)

Core defines the primary event bus protocol:

- Reference: `omnibase_core/src/omnibase_core/protocols/event_bus/protocol_event_bus.py`

**Core protocol shape (as-is)**:
- `publish(topic, key, value, headers)`
- `subscribe(topic, group_id, on_message)`
- plus environment/group addressing helpers

### Infra3: concrete event bus implementations

Infra3 provides two primary implementations:

- `InMemoryEventBus`:
  - Reference: `omnibase_infra3/src/omnibase_infra/event_bus/inmemory_event_bus.py`
- `KafkaEventBus`:
  - Reference: `omnibase_infra3/src/omnibase_infra/event_bus/kafka_event_bus.py`

Both implement the Core protocol surface via duck typing.

### Infra3: message/header model shapes

Infra3 uses typed models for messages and headers:

- `ModelEventMessage`: `omnibase_infra3/src/omnibase_infra/event_bus/models/model_event_message.py`
- `ModelEventHeaders`: `omnibase_infra3/src/omnibase_infra/event_bus/models/model_event_headers.py`

### SPI: event bus “base” protocol (broader surface)

SPI defines a broader base interface that supports:

- Publishing basic `ProtocolEventMessage` events
- Publishing and consuming **envelopes** (explicitly references `ModelOnexEnvelope`)
- Subscription lifecycle: `start_consuming()` / `stop_consuming()`

Reference: `omnibase_spi/src/omnibase_spi/protocols/event_bus/protocol_event_bus_mixin.py`

### SPI: workflow orchestration event bus (event sourcing shape)

SPI also defines a workflow-oriented event bus abstraction that includes:

- workflow instance identity (`instance_id`)
- ordering (`sequence_number`)
- idempotency (`idempotency_key`)
- projection support hooks

Reference: `omnibase_spi/src/omnibase_spi/protocols/workflow_orchestration/protocol_workflow_event_bus.py`

### As-is “shape facts” to carry forward

- Core and Infra3 are already aligned on a topic/key/value/headers event bus abstraction.
- SPI’s event bus protocols include a second plane: envelope-based and workflow-event-sourcing-based messaging.
- There is no single “one event bus interface” across all repos; there is a base core bus plus SPI extensions.
