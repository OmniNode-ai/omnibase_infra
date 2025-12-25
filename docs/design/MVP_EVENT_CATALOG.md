# MVP Event Catalog

> **Status**: Living Document
> **Document Version**: 1.0.0
> **Phase**: 9 (Event Bus Integration)
> **Ticket**: OMN-57
> **Last Updated**: 2025-12-24
> **Author**: ONEX Infrastructure Team

## Overview

This catalog documents all event types, their schemas, and usage patterns in the ONEX infrastructure event-driven architecture. The system uses Kafka for message transport and Pydantic models for schema validation.

## Table of Contents

1. [Message Categories](#message-categories)
2. [Topic Naming Conventions](#topic-naming-conventions)
3. [Header Schema](#header-schema)
4. [Event Message Structure](#event-message-structure)
5. [Registration Domain Events](#registration-domain-events)
6. [Registration Domain Commands](#registration-domain-commands)
7. [Discovery Domain Events](#discovery-domain-events)
8. [Dispatch Models](#dispatch-models)
9. [Schema Evolution Guidelines](#schema-evolution-guidelines)
10. [Best Practices](#best-practices)

---

## Message Categories

ONEX uses three fundamental message categories defined in `EnumMessageCategory`:

| Category | Description | Topic Suffix | Use Case |
|----------|-------------|--------------|----------|
| **EVENT** | Domain events representing facts that have occurred (past tense, immutable) | `.events` | State changes, audit logs, notifications |
| **COMMAND** | Instructions to perform an action (imperative) | `.commands` | Request-response patterns, action triggers |
| **INTENT** | User intents requiring interpretation and routing | `.intents` | User interactions, workflow triggers |

### Category vs Node Output Type

**Important Distinction**: `EnumMessageCategory` is for **message routing** (Kafka topics), while `EnumNodeOutputType` is for **execution shape validation** (node outputs).

```
EnumMessageCategory (routing):    EVENT, COMMAND, INTENT
EnumNodeOutputType (validation):  EVENT, COMMAND, INTENT, PROJECTION
```

`PROJECTION` is NOT a message category - it's a node output type used only by REDUCER nodes for state consolidation. Projections are internal outputs, not routable messages.

**Source Files**:
- `src/omnibase_infra/enums/enum_message_category.py`
- `src/omnibase_infra/enums/enum_node_output_type.py`

---

## Topic Naming Conventions

ONEX supports two topic naming standards:

### 1. ONEX Kafka Format (Canonical)

```
onex.<domain>.<type>
```

| Component | Description | Examples |
|-----------|-------------|----------|
| `onex` | Fixed namespace prefix | Always "onex" |
| `domain` | Bounded context name | `registration`, `discovery`, `order` |
| `type` | Message category suffix | `events`, `commands`, `intents`, `snapshots` |

**Examples**:
- `onex.registration.events` - Registration domain events
- `onex.discovery.commands` - Discovery service commands
- `onex.checkout.intents` - Checkout user intents

### 2. Environment-Aware Format (Deployment-Specific)

```
<env>.<domain>.<category>.<version>
```

| Component | Description | Valid Values |
|-----------|-------------|--------------|
| `env` | Deployment environment | `dev`, `staging`, `prod`, `test`, `local` |
| `domain` | Bounded context name | Lowercase alphanumeric with hyphens |
| `category` | Message category suffix | `events`, `commands`, `intents` |
| `version` | API version | `v1`, `v2`, etc. |

**Examples**:
- `dev.user.events.v1` - Development user events, version 1
- `prod.order.commands.v2` - Production order commands, version 2
- `staging.payment.intents.v1` - Staging payment intents

### Domain Naming Rules

- Lowercase alphanumeric characters with hyphens
- Must start with a letter
- Single letter domains are valid (e.g., `onex.a.events`)
- Multi-part domains use hyphens (e.g., `order-fulfillment`)

### Topic Type Mapping

| Topic Suffix | EnumMessageCategory | Processing Pattern |
|--------------|--------------------|--------------------|
| `events` | `EVENT` | Reducers, projections |
| `commands` | `COMMAND` | Command handlers |
| `intents` | `INTENT` | Orchestrators |
| `snapshots` | N/A | Materialized views (no category) |

**Source Files**:
- `src/omnibase_infra/enums/enum_topic_standard.py`
- `src/omnibase_infra/models/dispatch/model_topic_parser.py`
- `src/omnibase_infra/models/dispatch/model_parsed_topic.py`

---

## Header Schema

All ONEX messages include standardized headers for tracing, routing, and retry configuration.

### ModelEventHeaders

```python
class ModelEventHeaders(BaseModel):
    # Content
    content_type: str = "application/json"

    # Correlation and Identity
    correlation_id: UUID = Field(default_factory=uuid4)
    message_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Routing
    source: str                              # Required: producing service
    event_type: str                          # Required: type identifier
    schema_version: str = "1.0.0"
    destination: str | None = None
    routing_key: str | None = None
    partition_key: str | None = None

    # Distributed Tracing
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    operation_name: str | None = None

    # Priority and Retry
    priority: Literal["low", "normal", "high", "critical"] = "normal"
    retry_count: int = 0
    max_retries: int = 3
    ttl_seconds: int | None = None
```

### Required vs Optional Headers

| Header | Required | Description |
|--------|----------|-------------|
| `source` | Yes | Service that produced the message |
| `event_type` | Yes | Type identifier for the event |
| `correlation_id` | Auto-generated | UUID for correlating related messages |
| `message_id` | Auto-generated | Unique identifier for this message |
| `timestamp` | Auto-generated | Message creation timestamp |
| `schema_version` | Default "1.0.0" | Version of the message schema |

**Source File**: `src/omnibase_infra/event_bus/models/model_event_headers.py`

---

## Event Message Structure

### ModelEventMessage

The Kafka message envelope:

```python
class ModelEventMessage(BaseModel):
    topic: str                               # Kafka topic
    key: bytes | None = None                 # Partition key
    value: bytes                             # Serialized payload
    headers: ModelEventHeaders               # Structured headers
    offset: str | None = None                # Kafka offset (consumed)
    partition: int | None = None             # Kafka partition
```

### Message Flow

```
Producer                          Kafka                        Consumer
   |                               |                              |
   |  ModelEventMessage            |                              |
   |  - topic: "onex.reg.events"   |                              |
   |  - key: node_id (bytes)       |                              |
   |  - value: JSON payload        |                              |
   |  - headers: correlation_id,   |                              |
   |             source, event_type|                              |
   | ----------------------------->|                              |
   |                               | ----------------------------->|
   |                               |    Deserialize + dispatch     |
   |                               |    based on topic category    |
```

**Source File**: `src/omnibase_infra/event_bus/models/model_event_message.py`

---

## Registration Domain Events

The registration domain implements the ONEX 2-way registration pattern for node lifecycle management.

### Event Flow Diagram

```
Node                    Orchestrator              Reducer              Projection
 |                           |                       |                     |
 |--NodeIntrospected-------->|                       |                     |
 |                           |--RegistrationInitiated-->|------------------>|
 |                           |                       |                     |
 |                           |--RegistrationAccepted--->|------------------>|
 |<------(ack deadline)------|                       |                     |
 |                           |                       |                     |
 |--RegistrationAcked------->|                       |                     |
 |                           |--AckReceived------------>|------------------>|
 |                           |--NodeBecameActive------->|------------------>|
 |                           |                       |                     |
 |--Heartbeat--------------->|                       |                     |
 |--Heartbeat--------------->|  (liveness monitoring)|                     |
```

### ModelNodeIntrospectionEvent

**Purpose**: Node announces its presence and capabilities to the cluster.

**Topic**: `onex.registration.events` or `dev.node.events.v1`

**Category**: EVENT

```python
class ModelNodeIntrospectionEvent(BaseModel):
    # Identity
    node_id: UUID                            # Unique node identifier
    node_type: Literal["effect", "compute", "reducer", "orchestrator"]
    node_version: str = "1.0.0"              # Semantic version

    # Capabilities
    capabilities: ModelNodeCapabilities      # Node capabilities dict
    endpoints: dict[str, str]                # Exposed endpoints (name -> URL)

    # Metadata
    node_role: str | None = None             # Optional role (registry, adapter)
    metadata: ModelNodeMetadata              # Additional node metadata
    correlation_id: UUID                     # Required for idempotency

    # Deployment
    network_id: str | None = None            # Network/cluster identifier
    deployment_id: str | None = None         # Deployment/release identifier
    epoch: int | None = None                 # Registration epoch for ordering

    # Timing
    timestamp: datetime                      # Event timestamp (injected)
```

**Example**:
```json
{
  "node_id": "550e8400-e29b-41d4-a716-446655440000",
  "node_type": "effect",
  "node_version": "1.2.3",
  "capabilities": {"postgres": true, "read": true, "write": true},
  "endpoints": {"health": "http://localhost:8080/health"},
  "correlation_id": "660e8400-e29b-41d4-a716-446655440001",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Source File**: `src/omnibase_infra/models/registration/model_node_introspection_event.py`

---

### ModelNodeHeartbeatEvent

**Purpose**: Periodic liveness signal with health metrics.

**Topic**: `onex.registration.events` or `onex.heartbeat.events`

**Category**: EVENT

```python
class ModelNodeHeartbeatEvent(BaseModel):
    # Identity
    node_id: UUID                            # Node identifier
    node_type: str                           # ONEX node type (relaxed validation)
    node_version: str = "1.0.0"

    # Health Metrics
    uptime_seconds: float                    # Node uptime (>= 0)
    active_operations_count: int = 0         # Active operations (>= 0)
    memory_usage_mb: float | None = None     # Optional memory usage
    cpu_usage_percent: float | None = None   # Optional CPU usage (0-100)

    # Metadata
    correlation_id: UUID | None = None
    timestamp: datetime                      # Event timestamp (injected)
```

**Example**:
```json
{
  "node_id": "550e8400-e29b-41d4-a716-446655440000",
  "node_type": "effect",
  "node_version": "1.2.3",
  "uptime_seconds": 3600.5,
  "active_operations_count": 5,
  "memory_usage_mb": 256.0,
  "cpu_usage_percent": 15.5,
  "timestamp": "2025-01-15T11:30:00Z"
}
```

**Source File**: `src/omnibase_infra/models/registration/model_node_heartbeat_event.py`

---

### ModelNodeRegistrationInitiated

**Purpose**: Orchestrator signals start of registration attempt.

**Topic**: `onex.registration.events`

**Category**: EVENT

```python
class ModelNodeRegistrationInitiated(BaseModel):
    entity_id: UUID          # Entity identifier (= node_id)
    node_id: UUID            # Node being registered
    correlation_id: UUID     # Distributed tracing
    causation_id: UUID       # Triggering NodeIntrospected message_id
    emitted_at: datetime     # Orchestrator emission time (injected)
    registration_attempt_id: UUID  # Unique attempt identifier
```

**Triggering Event**: `NodeIntrospectionEvent`
**FSM Transition**: N/A -> INITIATED

**Source File**: `src/omnibase_infra/models/registration/events/model_node_registration_initiated.py`

---

### ModelNodeRegistrationAccepted

**Purpose**: Orchestrator accepts node registration.

**Topic**: `onex.registration.events`

**Category**: EVENT

```python
class ModelNodeRegistrationAccepted(BaseModel):
    entity_id: UUID          # Entity identifier (= node_id)
    node_id: UUID            # Node being registered
    correlation_id: UUID     # Distributed tracing
    causation_id: UUID       # Triggering event message_id
    emitted_at: datetime     # Orchestrator emission time (injected)
    ack_deadline: datetime   # Deadline for node acknowledgment
```

**FSM Transition**: INITIATED -> AWAITING_ACK

**Source File**: `src/omnibase_infra/models/registration/events/model_node_registration_accepted.py`

---

### ModelNodeRegistrationRejected

**Purpose**: Orchestrator rejects node registration.

**Topic**: `onex.registration.events`

**Category**: EVENT

```python
class ModelNodeRegistrationRejected(BaseModel):
    entity_id: UUID          # Entity identifier (= node_id)
    node_id: UUID            # Node being rejected
    correlation_id: UUID     # Distributed tracing
    causation_id: UUID       # Triggering event message_id
    emitted_at: datetime     # Orchestrator emission time (injected)
    rejection_reason: str    # Human-readable explanation (1-1024 chars)
```

**FSM Transition**: INITIATED -> REJECTED (terminal)

**Common Rejection Reasons**:
- Node version incompatibility
- Capability requirements not met
- Rate limiting exceeded
- Duplicate registration attempt
- Policy violation

**Source File**: `src/omnibase_infra/models/registration/events/model_node_registration_rejected.py`

---

### ModelNodeRegistrationAckReceived

**Purpose**: Orchestrator confirms receipt of node acknowledgment.

**Topic**: `onex.registration.events`

**Category**: EVENT

```python
class ModelNodeRegistrationAckReceived(BaseModel):
    entity_id: UUID          # Entity identifier (= node_id)
    node_id: UUID            # Node that acknowledged
    correlation_id: UUID     # Distributed tracing
    causation_id: UUID       # NodeRegistrationAcked command message_id
    emitted_at: datetime     # Orchestrator emission time (injected)
    liveness_deadline: datetime  # Deadline for next heartbeat
```

**Triggering Command**: `NodeRegistrationAcked`
**FSM Transition**: AWAITING_ACK -> ACTIVE

**Source File**: `src/omnibase_infra/models/registration/events/model_node_registration_ack_received.py`

---

### ModelNodeBecameActive

**Purpose**: Node transitions to active state.

**Topic**: `onex.registration.events`

**Category**: EVENT

```python
class ModelNodeBecameActive(BaseModel):
    entity_id: UUID                  # Entity identifier (= node_id)
    node_id: UUID                    # Activated node
    correlation_id: UUID             # Distributed tracing
    causation_id: UUID               # Triggering event message_id
    emitted_at: datetime             # Orchestrator emission time (injected)
    capabilities: ModelNodeCapabilities  # Node capabilities at activation
```

**FSM Transition**: AWAITING_ACK -> ACTIVE

**Source File**: `src/omnibase_infra/models/registration/events/model_node_became_active.py`

---

### ModelNodeRegistrationAckTimedOut

**Purpose**: Node failed to acknowledge within deadline.

**Topic**: `onex.registration.events`

**Category**: EVENT

```python
class ModelNodeRegistrationAckTimedOut(BaseModel):
    entity_id: UUID          # Entity identifier (= node_id)
    node_id: UUID            # Node that failed to acknowledge
    correlation_id: UUID     # Distributed tracing
    causation_id: UUID       # RuntimeTick that triggered this
    emitted_at: datetime     # Detection time (from RuntimeTick.now)
    deadline_at: datetime    # Original ack deadline that was exceeded
```

**Trigger**: RuntimeTick processing detects expired ack_deadline
**FSM Transition**: AWAITING_ACK -> ACK_TIMED_OUT (terminal)
**Deduplication**: Uses `ack_timeout_emitted_at` marker in projection

**Source File**: `src/omnibase_infra/models/registration/events/model_node_registration_ack_timed_out.py`

---

### ModelNodeLivenessExpired

**Purpose**: Active node failed heartbeat check.

**Topic**: `onex.registration.events`

**Category**: EVENT

```python
class ModelNodeLivenessExpired(BaseModel):
    entity_id: UUID              # Entity identifier (= node_id)
    node_id: UUID                # Node that failed liveness
    correlation_id: UUID         # Distributed tracing
    causation_id: UUID           # RuntimeTick that triggered this
    emitted_at: datetime         # Detection time (from RuntimeTick.now)
    last_heartbeat_at: datetime | None  # Last received heartbeat (or None)
```

**Trigger**: RuntimeTick processing detects expired liveness_deadline
**FSM Transition**: ACTIVE -> LIVENESS_EXPIRED (terminal)
**Deduplication**: Uses `liveness_timeout_emitted_at` marker in projection

**Source File**: `src/omnibase_infra/models/registration/events/model_node_liveness_expired.py`

---

## Registration Domain Commands

Commands are imperative requests from external sources.

### ModelNodeRegistrationAcked

**Purpose**: Node acknowledges its registration acceptance.

**Topic**: `onex.registration.commands`

**Category**: COMMAND

```python
class ModelNodeRegistrationAcked(BaseModel):
    command_id: UUID = Field(default_factory=uuid4)  # Unique command instance
    node_id: UUID                    # Node sending acknowledgment
    correlation_id: UUID             # Links to original registration flow
    timestamp: datetime              # When node sent acknowledgment (injected)
```

**Source**: The registered node itself
**Processing**: Orchestrator validates state, emits `NodeRegistrationAckReceived`

**Validity Conditions**:
- Node must be in AWAITING_ACK state
- If ACTIVE, this is a duplicate ack (no-op)
- If terminal state, ack is too late (rejected)

**Source File**: `src/omnibase_infra/models/registration/commands/model_node_registration_acked.py`

---

## Discovery Domain Events

Node introspection events for service discovery (separate from registration lifecycle).

### ModelNodeIntrospectionEvent (Discovery)

**Purpose**: Node capability discovery and catalog maintenance.

**Topic**: `onex.discovery.events` or `node.introspection`

**Category**: EVENT

This uses the same model as registration but serves a different purpose - populating the service catalog for routing and capability-based discovery.

**Source File**: `src/omnibase_infra/models/discovery/model_node_introspection_event.py`

---

## Dispatch Models

Models used by the message dispatch engine for routing and result tracking.

### ModelDispatchResult

**Purpose**: Captures dispatch operation outcome.

```python
class ModelDispatchResult(BaseModel):
    # Identity
    dispatch_id: UUID = Field(default_factory=uuid4)

    # Status
    status: EnumDispatchStatus          # SUCCESS, HANDLER_ERROR, TIMEOUT, etc.

    # Route Info
    route_id: str | None = None
    dispatcher_id: str | None = None
    topic: str                          # Dispatched topic
    message_category: EnumMessageCategory | None = None
    message_type: str | None = None

    # Timing
    duration_ms: float | None = None
    started_at: datetime
    completed_at: datetime | None = None

    # Outputs
    outputs: list[str] | None = None    # Output topics
    output_count: int = 0

    # Errors
    error_message: str | None = None
    error_code: EnumCoreErrorCode | None = None
    error_details: dict[str, JsonValue] | None = None

    # Retry
    retry_count: int = 0

    # Tracing
    correlation_id: UUID | None = None
    trace_id: UUID | None = None
    span_id: UUID | None = None
```

**Source File**: `src/omnibase_infra/models/dispatch/model_dispatch_result.py`

### ModelDispatchContext

**Purpose**: Carries dispatch metadata with time injection control.

```python
class ModelDispatchContext(BaseModel):
    correlation_id: UUID                # Required tracing ID
    trace_id: UUID | None = None
    now: datetime | None = None         # Time injection (None for deterministic nodes)
    node_kind: EnumNodeKind             # REDUCER, ORCHESTRATOR, EFFECT, COMPUTE
    metadata: dict[str, str] | None = None
```

**Time Injection Rules**:
- REDUCER, COMPUTE: `now` MUST be None (deterministic execution)
- ORCHESTRATOR, EFFECT, RUNTIME_HOST: `now` CAN be provided

**Factory Methods**:
- `ModelDispatchContext.for_reducer(correlation_id=...)` - No time injection
- `ModelDispatchContext.for_orchestrator(correlation_id=..., now=...)` - With time
- `ModelDispatchContext.for_effect(correlation_id=..., now=...)` - With time

**Source File**: `src/omnibase_infra/models/dispatch/model_dispatch_context.py`

### ModelParsedTopic

**Purpose**: Structured topic parsing result for routing.

```python
class ModelParsedTopic(BaseModel):
    raw_topic: str                      # Original topic string
    standard: EnumTopicStandard         # ONEX_KAFKA, ENVIRONMENT_AWARE, UNKNOWN
    domain: str | None = None           # Extracted domain
    category: EnumMessageCategory | None = None  # EVENT, COMMAND, INTENT
    topic_type: EnumTopicType | None = None      # events, commands, intents, snapshots
    environment: str | None = None      # Environment (env-aware format)
    version: str | None = None          # Version (env-aware format)
    is_valid: bool = False              # Parse success
    validation_error: str | None = None
```

**Source File**: `src/omnibase_infra/models/dispatch/model_parsed_topic.py`

---

## Schema Evolution Guidelines

### Versioning Strategy

1. **Schema Version in Headers**: All messages include `schema_version` field
2. **Topic Versioning**: Environment-aware topics include version suffix (`v1`, `v2`)
3. **Backward Compatibility**: New fields should have defaults

### Adding Fields

```python
# Safe: New field with default
new_field: str | None = None
new_field: str = "default_value"
new_field: int = 0

# Unsafe: New required field (breaking change)
new_field: str  # Breaks existing consumers
```

### Removing Fields

1. Deprecate by making optional with default
2. Document deprecation in schema version notes
3. Remove in next major version

### Field Type Changes

| Original | New | Safe? |
|----------|-----|-------|
| `str` | `str | None` | Yes (widen) |
| `str | None` | `str` | No (narrow) |
| `int` | `float` | Depends (test) |
| `list[str]` | `list[str | None]` | Yes (widen) |

### Version Bump Guidelines

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| New optional field | Patch (1.0.0 -> 1.0.1) | Adding `metadata` field |
| New required field | Major (1.0.0 -> 2.0.0) | Adding `tenant_id` required |
| Field deprecation | Minor (1.0.0 -> 1.1.0) | Deprecating `old_field` |
| Field removal | Major (1.0.0 -> 2.0.0) | Removing deprecated field |
| Field type change | Major (usually) | `str` to `int` |

---

## Best Practices

### Event Design

1. **Immutability**: Events are facts - never modify published events
2. **Past Tense Naming**: `UserCreated`, not `CreateUser`
3. **Self-Contained**: Include all data needed by consumers
4. **Causation Chain**: Always include `causation_id` linking to trigger

### Command Design

1. **Imperative Naming**: `CreateUser`, `ProcessPayment`
2. **Idempotency**: Use `command_id` for deduplication
3. **Validation First**: Validate before processing

### Header Usage

1. **Always set `source`**: Identifies producing service
2. **Always set `event_type`**: Enables routing and filtering
3. **Propagate `correlation_id`**: Enables distributed tracing
4. **Use `priority` appropriately**: Default is "normal"

### Topic Selection

1. **Use ONEX Kafka format** for internal services
2. **Use Environment-Aware format** for multi-environment deployments
3. **One domain per topic** - avoid mixing bounded contexts
4. **Version topics** when schema changes are breaking

### Timestamp Injection

1. **Never use `datetime.now()` in models** - timestamps must be injected
2. **Use `emitted_at` for orchestrator events** - represents decision time
3. **Use `timestamp` for node events** - represents observation time
4. **Reducers receive no time** - ensures deterministic replay

### Error Handling

1. **Use `ModelDispatchResult`** for dispatch outcomes
2. **Include `error_code`** from `EnumCoreErrorCode`
3. **Sanitize error messages** - no secrets or PII
4. **Track `retry_count`** for observability

---

## Quick Reference

### Event Models by Domain

| Domain | Model | Category | Topic |
|--------|-------|----------|-------|
| Registration | `ModelNodeIntrospectionEvent` | EVENT | `onex.registration.events` |
| Registration | `ModelNodeHeartbeatEvent` | EVENT | `onex.registration.events` |
| Registration | `ModelNodeRegistrationInitiated` | EVENT | `onex.registration.events` |
| Registration | `ModelNodeRegistrationAccepted` | EVENT | `onex.registration.events` |
| Registration | `ModelNodeRegistrationRejected` | EVENT | `onex.registration.events` |
| Registration | `ModelNodeRegistrationAckReceived` | EVENT | `onex.registration.events` |
| Registration | `ModelNodeBecameActive` | EVENT | `onex.registration.events` |
| Registration | `ModelNodeRegistrationAckTimedOut` | EVENT | `onex.registration.events` |
| Registration | `ModelNodeLivenessExpired` | EVENT | `onex.registration.events` |
| Registration | `ModelNodeRegistrationAcked` | COMMAND | `onex.registration.commands` |
| Discovery | `ModelNodeIntrospectionEvent` | EVENT | `onex.discovery.events` |

### Import Paths

```python
# Registration Events
from omnibase_infra.models.registration import (
    ModelNodeIntrospectionEvent,
    ModelNodeHeartbeatEvent,
    ModelNodeRegistrationInitiated,
    ModelNodeRegistrationAccepted,
    ModelNodeRegistrationRejected,
    ModelNodeRegistrationAckReceived,
    ModelNodeBecameActive,
    ModelNodeRegistrationAckTimedOut,
    ModelNodeLivenessExpired,
)

# Registration Commands
from omnibase_infra.models.registration.commands import (
    ModelNodeRegistrationAcked,
)

# Dispatch Models
from omnibase_infra.models.dispatch import (
    ModelDispatchResult,
    ModelDispatchContext,
    ModelParsedTopic,
    ModelTopicParser,
)

# Event Bus Models
from omnibase_infra.event_bus.models import (
    ModelEventMessage,
    ModelEventHeaders,
)

# Enums
from omnibase_infra.enums import (
    EnumMessageCategory,
    EnumNodeOutputType,
    EnumTopicStandard,
    EnumDispatchStatus,
)
```

---

## Related Documentation

- **CLAUDE.md**: Enum usage guidelines for message routing vs node validation
- **ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md**: C1 Orchestrator design
- **correlation_id_tracking.md**: Distributed tracing patterns
- **error_handling_patterns.md**: Error context and sanitization

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-24 | Initial MVP catalog |
