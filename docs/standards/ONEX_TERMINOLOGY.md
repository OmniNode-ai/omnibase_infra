# ONEX Infrastructure Terminology

> **Status**: Current | **Last Updated**: 2026-02-19

Canonical term list for `omnibase_infra`-specific vocabulary. This document covers infrastructure-layer terminology only. Core architectural terms (Effect, Compute, Reducer, Orchestrator, Intent, Event, etc.) are defined in `docs/conventions/TERMINOLOGY_GUIDE.md`.

---

## Table of Contents

1. [Overview](#overview)
2. [Canonical vs. Deprecated Terms](#canonical-vs-deprecated-terms)
3. [Transport and Handler Terms](#transport-and-handler-terms)
4. [Model and Type Terms](#model-and-type-terms)
5. [Runtime and Container Terms](#runtime-and-container-terms)
6. [Topic and Event Bus Terms](#topic-and-event-bus-terms)
7. [Configuration Terms](#configuration-terms)
8. [See Also](#see-also)

---

## Overview

This document establishes the authoritative term list for the infrastructure layer. Consistency in naming prevents confusion across code, docs, PR reviews, and Linear tickets. When writing code or documentation, always use the canonical term.

---

## Canonical vs. Deprecated Terms

### Quick Reference Table

| Canonical | Deprecated Synonyms | Notes |
|-----------|---------------------|-------|
| `EnumInfraTransportType` | `TransportType`, `EnumTransportType` | Full enum name required in code |
| `ModelIntent` | `IntentModel`, `IntentPayload`, `ModelSideEffect` | The ONEX side-effect declaration model |
| `ModelPayload*` | `IntentPayload*`, `*PayloadModel` | Prefix for typed intent payload models |
| `effect handler` | `side effect handler`, `side-effect handler` | Lowercase in prose |
| `EnumHandlerType` | `HandlerTypeEnum`, `HandlerKind` | Enum for INFRA_HANDLER, NODE_HANDLER, etc. |
| `EnumHandlerTypeCategory` | `HandlerCategoryEnum`, `HandlerBehavior` | EFFECT, COMPUTE, NONDETERMINISTIC_COMPUTE |
| `ModelInfraErrorContext` | `ErrorContext`, `InfraContext` | Error context factory for all infra errors |
| `RuntimeHostError` | `InfraError`, `BaseInfraError` | Base infrastructure exception class |
| `InfraConnectionError` | `ConnectionError`, `TransportError` | Transport-aware connection failure |
| `ModelONEXContainer` | `container`, `DI container` alone, `ServiceContainer` | Always use full class name in code |
| `ProtocolHandler` | `Handler`, `HandlerInterface` | Envelope-based handler protocol |
| `ProtocolMessageHandler` | `MessageHandler`, `CategoryHandler` | Category-based dispatch handler protocol |
| `wire_infrastructure_services()` | `setup_services()`, `configure_container()` | Container wiring utility function |
| `wire_registration_handlers()` | `setup_handlers()`, `register_handlers()` | Handler wiring utility function |
| `TopicProvisioner` | `TopicManager`, `KafkaSetup` | Startup topic creation component |
| `EventBusKafka` | `KafkaEventBus`, `KafkaBus` | Kafka-backed event bus implementation |
| `EventBusInmemory` | `InMemoryBus`, `InmemoryEventBus` | In-memory event bus for tests |
| `ProjectorShell` | `Projector`, `ProjectionShell` | Contract-driven projector wrapper |
| `HandlerPluginLoader` | `PluginLoader`, `HandlerLoader` | YAML-contract-driven handler loader |
| `MixinAsyncCircuitBreaker` | `CircuitBreaker`, `BreakerMixin` | Circuit breaker mixin for adapters |
| `RuntimeHostProcess` | `Runtime`, `HostRuntime`, `InfraRuntime` | Production event-loop-based runtime |

---

## Transport and Handler Terms

### EnumInfraTransportType

The canonical enum for identifying transport types across the infra layer.

**Canonical values** (use these exact strings):

| Value | String | Used By |
|-------|--------|---------|
| `HTTP` | `"http"` | `HandlerHTTP`, `ServiceHealth` |
| `DATABASE` | `"db"` | `HandlerDb`, `PostgresRepositoryRuntime` |
| `KAFKA` | `"kafka"` | `EventBusKafka`, Kafka adapters |
| `CONSUL` | `"consul"` | `HandlerConsul` |
| `VAULT` | `"vault"` | `HandlerVault` |
| `VALKEY` | `"valkey"` | Planned |
| `GRPC` | `"grpc"` | Planned |
| `RUNTIME` | `"runtime"` | `RuntimeHostProcess` |
| `MCP` | `"mcp"` | `HandlerMCP` |
| `FILESYSTEM` | `"filesystem"` | `HandlerFileSystem` |
| `INMEMORY` | `"inmemory"` | `EventBusInmemory` |
| `QDRANT` | `"qdrant"` | `HandlerQdrant` |
| `GRAPH` | `"graph"` | Planned (Memgraph/Neo4j) |

### Handler Routing Strategies

| Canonical Term | YAML Value | Used When |
|----------------|------------|-----------|
| payload type match | `"payload_type_match"` | Orchestrator handlers routing by event model |
| operation match | `"operation_match"` | Infrastructure handlers routing by operation name |

---

## Model and Type Terms

### ModelIntent (infra usage)

In `omnibase_infra`, `ModelIntent` is always constructed with `intent_type="extension"`. The routing key lives in the typed payload model (`ModelPayload*`).

**Correct construction**:
```python
# Typed payload carries the routing key
payload = ModelPayloadConsulRegister(intent_type="consul.register", ...)

# Outer container always uses "extension"
return ModelIntent(intent_type="extension", target="consul://service/...", payload=payload)
```

**Deprecated**: Do not call these patterns:
- `intent_type="consul.register"` directly on `ModelIntent` (wrong — that belongs on the payload)
- `ModelIntentPayloadBase` (removed in omnibase_core 0.6.2 — extend `BaseModel` directly)

### Typed Payload Models

All typed intent payloads extend `pydantic.BaseModel` directly (not `ModelIntentPayloadBase`).

| Canonical Pattern | Deprecated Pattern |
|-------------------|--------------------|
| `class ModelPayloadConsulRegister(BaseModel)` | `class ModelPayloadConsulRegister(ModelIntentPayloadBase)` |
| `intent_type: Literal["consul.register"]` field on payload | `intent_type` on outer `ModelIntent` |

---

## Runtime and Container Terms

### ModelONEXContainer

The dependency injection container. Always use the full class name in code.

| Canonical | Not This |
|-----------|---------|
| `ModelONEXContainer` | `container` (alone, ambiguous) |
| `container: ModelONEXContainer` (type annotation) | `container: Any` |
| `container.get_service("ProtocolEventBus")` | Direct attribute access |

### Handler Classification Properties

Every handler must expose exactly these two properties:

| Property | Type | Values |
|----------|------|--------|
| `handler_type` | `EnumHandlerType` | `INFRA_HANDLER`, `NODE_HANDLER`, `PROJECTION_HANDLER` |
| `handler_category` | `EnumHandlerTypeCategory` | `EFFECT`, `COMPUTE`, `NONDETERMINISTIC_COMPUTE` |

### Handler No-Publish Constraint

Handlers must not hold or call any event bus reference. The canonical term for this architectural rule is the **handler no-publish constraint**.

| Forbidden in handlers | Canonical alternative |
|-----------------------|----------------------|
| `self._bus.publish(...)` | Orchestrator emits events |
| `self._event_bus` attribute | Not allowed |
| `self.emit(event)` method | Not allowed |

---

## Topic and Event Bus Terms

### Topic Suffix vs. Full Topic

| Term | Meaning | Example |
|------|---------|---------|
| **topic suffix** | The realm-agnostic suffix stored in `platform_topic_suffixes.py` | `onex.evt.platform.node-registration.v1` |
| **full topic** | A suffix composed with a tenant/namespace prefix at runtime | `dev.myservice.onex.evt.platform.node-registration.v1` |
| **provisioned topic** | A topic created by `TopicProvisioner` at startup | Any entry in `ALL_PROVISIONED_TOPIC_SPECS` |

Use "topic suffix" when referring to the constants in `platform_topic_suffixes.py`. Use "full topic" only when referring to the runtime-composed name with tenant prefix.

### EventBusKafka vs. EventBusInmemory

| Term | When to Use |
|------|-------------|
| `EventBusKafka` | Production and E2E integration tests |
| `EventBusInmemory` | Unit tests that need event bus behavior without Kafka |

---

## Configuration Terms

### Infisical-Backed Configuration

| Term | Meaning |
|------|---------|
| **transport config spec** | A `ModelTransportConfigSpec` defining Infisical path and key for a transport setting |
| **config requirements** | `ModelConfigRequirements` — aggregated config needs extracted from a contract |
| **config prefetch** | Runtime pre-fetching of all config values from Infisical before node startup |
| **bootstrap-only `.env`** | The project `.env` in its reduced form — contains only credentials needed to start Infisical, not full service config |

### Correlation ID Rules

Always use these exact phrases:

| Correct | Incorrect |
|---------|-----------|
| "propagate existing correlation ID" | "pass the correlation ID" |
| "auto-generate correlation ID" | "create a new ID" |
| `ModelInfraErrorContext.with_correlation(correlation_id=...)` | Direct error instantiation without context |

---

## See Also

| Topic | Document |
|-------|----------|
| Core ONEX terminology (Effect, Reducer, etc.) | [../conventions/TERMINOLOGY_GUIDE.md](../conventions/TERMINOLOGY_GUIDE.md) |
| Handler system patterns | [../patterns/handler_plugin_loader.md](../patterns/handler_plugin_loader.md) |
| Error handling | [../patterns/error_handling_patterns.md](../patterns/error_handling_patterns.md) |
| Topic taxonomy | [TOPIC_TAXONOMY.md](./TOPIC_TAXONOMY.md) |
| Naming conventions | [../conventions/NAMING_CONVENTIONS.md](../conventions/NAMING_CONVENTIONS.md) |
