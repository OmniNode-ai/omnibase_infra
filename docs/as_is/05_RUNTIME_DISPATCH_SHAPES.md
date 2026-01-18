> **Navigation**: [Home](../index.md) > [As-Is](INDEX.md) > Runtime Dispatch Shapes

## Runtime dispatch shapes (As-Is)

This document captures the two parallel “runtime dispatch” patterns that exist today:

1) Core’s transport-agnostic `EnvelopeRouter` (routes `ModelOnexEnvelope`)
2) Infra3’s `RuntimeHostProcess` (routes dict “operation envelopes” to protocol handlers)

The purpose is not to prefer one; it’s to make the distinction explicit so that workflow
design discussions don’t mix them accidentally.

### Core runtime: `EnvelopeRouter`

- Reference: `omnibase_core/src/omnibase_core/runtime/envelope_router.py`

**Routes**
- Input: `ModelOnexEnvelope` (Core model)
  - Reference: `omnibase_core/src/omnibase_core/models/core/model_onex_envelope.py`

**Routing key**
- `envelope.handler_type` (`EnumHandlerType`)
  - Reference: `omnibase_core/src/omnibase_core/enums/enum_handler_type.py`

**Registry**
- Handlers are registered by `EnumHandlerType`
- Nodes are registered by slug (unique)
- Router has a “register then freeze then execute” thread-safety model

**Execution surface**
- `route_envelope(envelope) -> {handler, handler_type}`
- `execute_with_handler(envelope, instance) -> ModelOnexEnvelope` (response envelope)

Core’s runtime model is “in-memory and transport-agnostic”: it does not require Kafka to function.

### Infra3 runtime: `RuntimeHostProcess`

- Reference: `omnibase_infra3/src/omnibase_infra/runtime/runtime_host_process.py`

**Routes**
- Input: dict envelope with `operation` string (e.g., `"db.query"`, `"http.post"`)
- Validated by:
  - `omnibase_infra3/src/omnibase_infra/runtime/envelope_validator.py`

**Routing key**
- `operation` prefix (substring before first `.`), e.g. `db`, `http`, `consul`, `vault`
- Validated against:
  - `omnibase_infra3/src/omnibase_infra/runtime/handler_registry.py`

**Registry**
- Protocol binding registry maps prefix string → handler class
  - Wired by: `omnibase_infra3/src/omnibase_infra/runtime/wiring.py`

**Execution surface**
- Runtime host subscribes to an event bus topic (default `requests`) and publishes to `responses`
- It instantiates registered protocol handlers and invokes `execute(envelope)` on them

### Relationship between the two runtimes (as-is)

They overlap in *purpose* (route “work” to “handlers”) but differ in:

- Message shape (`ModelOnexEnvelope` vs dict operation envelope)
- Handler identity (`EnumHandlerType` vs string prefix)
- Response shape (response `ModelOnexEnvelope` vs response envelope dict/messages)
- Intended scope (Core routing nodes/handlers vs Infra routing low-level protocol operations)

This is the key “shape” fact to keep in mind when evaluating architecture proposals:
design docs must specify which runtime plane they are targeting, or introduce a bridging layer.
