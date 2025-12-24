## Interface crosswalk (As-Is)

This document is a **lookup table**: given a concept name used in architecture discussions,
it maps to the **concrete types/files** in `omnibase_core`, `omnibase_spi`, and `omnibase_infra3`
that implement or model that concept today.

### Crosswalk table

| Concept | omnibase_core (canonical) | omnibase_spi (stable protocols) | omnibase_infra3 (implementations) |
|---|---|---|---|
| **Node kind (effect/compute/reducer/orchestrator/runtime_host)** | `omnibase_core/src/omnibase_core/enums/enum_node_kind.py` (`EnumNodeKind`) | N/A | N/A |
| **Node type (specific classification)** | `omnibase_core/src/omnibase_core/enums/enum_node_type.py` (`EnumNodeType`) | N/A | N/A |
| **Node lifecycle base** | `omnibase_core/src/omnibase_core/infrastructure/node_core_base.py` (`NodeCoreBase`) | N/A | N/A |
| **Contract-driven effect node** | `omnibase_core/src/omnibase_core/nodes/node_effect.py` (`NodeEffect`) | N/A | Infra effect nodes exist but are not uniformly this base |
| **Contract-driven reducer node** | `omnibase_core/src/omnibase_core/nodes/node_reducer.py` (`NodeReducer`) | N/A | `omnibase_infra3/src/omnibase_infra/nodes/reducers/node_dual_registration_reducer.py` (custom reducer implementation) |
| **Contract-driven orchestrator node** | `omnibase_core/src/omnibase_core/nodes/node_orchestrator.py` (`NodeOrchestrator`) | N/A | N/A |
| **Orchestrator execution engine** | `omnibase_core/src/omnibase_core/mixins/mixin_workflow_execution.py` (`MixinWorkflowExecution`) | Workflow orchestration protocols (see below) | N/A |
| **Core DI container** | `omnibase_core/src/omnibase_core/models/container/model_onex_container.py` (`ModelONEXContainer`) | Container protocols live under `omnibase_spi/src/omnibase_spi/protocols/container/` | Infra nodes often resolve deps via Core container |
| **Core intents (closed set)** | `omnibase_core/src/omnibase_core/models/intents/__init__.py` (`ModelCoreRegistrationIntent`, etc.) | N/A | Used by infra reducers (e.g., dual registration reducer emits Core intents) |
| **Extension intents (open set)** | `omnibase_core/src/omnibase_core/models/reducer/model_intent.py` (`ModelIntent`) | N/A | N/A |
| **Inter-service envelope (request/response + routing)** | `omnibase_core/src/omnibase_core/models/core/model_onex_envelope.py` (`ModelOnexEnvelope`) | Referenced by SPI event bus base protocols | N/A (infra runtime host uses a different envelope shape) |
| **Generic event envelope wrapper** | `omnibase_core/src/omnibase_core/models/events/model_event_envelope.py` (`ModelEventEnvelope[T]`) | `omnibase_core/src/omnibase_core/protocols/event_bus/protocol_event_envelope.py` (`ProtocolEventEnvelope[T]`) | N/A |
| **Core runtime router (in-memory, transport-agnostic)** | `omnibase_core/src/omnibase_core/runtime/envelope_router.py` (`EnvelopeRouter`) | N/A | N/A |
| **Core handler type enum (runtime routing classification)** | `omnibase_core/src/omnibase_core/enums/enum_handler_type.py` (`EnumHandlerType`) | N/A | Infra runtime host uses string prefixes instead |
| **Core event bus protocol (topic/key/value/headers)** | `omnibase_core/src/omnibase_core/protocols/event_bus/protocol_event_bus.py` (`ProtocolEventBus`) | SPI has richer base protocols (below) | Implemented by `omnibase_infra3/src/omnibase_infra/event_bus/*_event_bus.py` |
| **Core event bus headers protocol** | `omnibase_core/src/omnibase_core/protocols/event_bus/protocol_event_bus_headers.py` | N/A | `omnibase_infra3/src/omnibase_infra/event_bus/models/model_event_headers.py` (concrete header model) |
| **Core event bus message protocol** | `omnibase_core/src/omnibase_core/protocols/event_bus/protocol_event_message.py` | N/A | `omnibase_infra3/src/omnibase_infra/event_bus/models/model_event_message.py` (concrete message model) |
| **SPI event bus base protocol (basic + envelope publish/consume)** | N/A | `omnibase_spi/src/omnibase_spi/protocols/event_bus/protocol_event_bus_mixin.py` (`ProtocolEventBusBase`) | Infra buses implement the Core bus protocol; SPI base is broader than infra’s current surface |
| **SPI workflow event bus (event-sourcing shape)** | N/A | `omnibase_spi/src/omnibase_spi/protocols/workflow_orchestration/protocol_workflow_event_bus.py` | N/A |
| **SPI workflow coordinator** | N/A | `omnibase_spi/src/omnibase_spi/protocols/workflow_orchestration/protocol_workflow_event_coordinator.py` | N/A |
| **Infra runtime host (operation routing to protocol handlers)** | N/A | N/A | `omnibase_infra3/src/omnibase_infra/runtime/runtime_host_process.py` (`RuntimeHostProcess`) |
| **Infra operation envelope validation** | N/A | N/A | `omnibase_infra3/src/omnibase_infra/runtime/envelope_validator.py` (`validate_envelope`, `normalize_correlation_id`) |
| **Infra protocol handler registry / wiring** | N/A | N/A | `omnibase_infra3/src/omnibase_infra/runtime/handler_registry.py`, `omnibase_infra3/src/omnibase_infra/runtime/wiring.py` |
| **Infra 2-way registration effect node** | N/A | N/A | `omnibase_infra3/src/omnibase_infra/nodes/node_registry_effect/v1_0_0/node.py` + `contract.yaml` |

### Notes

- This table is **not** asserting which component “should” be canonical; it’s recording what exists.
- The biggest recurring ambiguity in discussions tends to be “envelope/runtime/handler” because
  Core and Infra3 both have routing runtimes with different message shapes.
