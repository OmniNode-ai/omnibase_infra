# ONEX Event Streaming Topics — Specification (v1)

**Status**: LOCKED (MVP-SAFE)
**Scope**: Kafka-based event streaming for ONEX node lifecycle, registry workflows, and infrastructure observability
**Non-Goal**: Kafka as a general RPC or query transport

---

## 1. Purpose and Scope

This document defines **Kafka topic contracts** used by ONEX for **event-driven coordination**.

Kafka is used for:
- Lifecycle events
- Registry state changes
- Workflow observability
- Infrastructure signaling

Kafka is **not** used for:
- Strong request/response semantics
- Guaranteed per-node acknowledgements
- Interactive discovery queries

Postgres remains the **source of truth** for registry state.

---

## 2. Event Model Clarification (IMPORTANT)

### Kafka Events Are NOT RPC

Topics such as:
- `onex.registry.node.registered.v1`
- `onex.registry.node.registration_failed.v1`

are **state transition events**, not delivery acknowledgements.

**Guarantees:**
- Events are durable (per retention policy)
- Events are observable
- Events may be missed by offline consumers

**Non-guarantees:**
- A node receiving its own registration event
- Exactly-once delivery to a specific consumer
- Request/response coupling

Nodes must rely on **registry state queries** for authoritative answers.

---

## 3. Topic Naming Convention

Topic naming follows an ONEX-specific convention:

```
onex.<domain>.<entity>.<event>.v<version>
```

| Component | Description | Example |
|-----------|-------------|---------|
| `onex` | Platform namespace | `onex` |
| `<domain>` | Logical domain | `node`, `registry`, `infra` |
| `<entity>` | Entity type | `introspection`, `heartbeat`, `workflow` |
| `<event>` | Event meaning | `published`, `requested`, `completed` |
| `v<version>` | Schema version | `v1` |

---

## 4. Canonical Event Envelope (RECOMMENDED)

All events SHOULD conform to a common envelope:

```json
{
  "event_id": "uuid",
  "event_type": "onex.node.introspection.published",
  "source": "node/postgres-adapter",
  "correlation_id": "uuid",
  "timestamp": "RFC3339",
  "payload": { }
}
```

Benefits:
- Uniform observability
- Versioning clarity
- Easier replay and audit

Payload schemas are versioned independently.

---

## 5. Keying Rules (LOCKED)

Kafka keys MUST follow these rules:

| Event Class | Key |
|-------------|-----|
| Node lifecycle | `node_id` |
| Registry workflow | `workflow_id` |
| Infrastructure signals | `service_name` |
| Trace/error correlation | `correlation_id` |

Arbitrary or per-topic keying is forbidden.

---

## 6. Topic Classes and Semantics

### 6.1 Node Lifecycle Events (Node → Registry)

#### onex.node.introspection.published.v1

Node announces current capabilities.

| Property | Value |
|----------|-------|
| **Meaning** | Node announces current capabilities |
| **Semantic** | Last-write-wins snapshot |
| **Key** | `node_id` |
| **Retention** | delete (1 day) |
| **Compaction** | OPTIONAL (post-MVP) |
| **Partitions** | 3 |

**Payload:**
```json
{
  "node_id": "postgres-adapter-001",
  "node_type": "EFFECT",
  "node_version": "1.0.0",
  "capabilities": {
    "postgres": true,
    "read": true,
    "write": true
  },
  "endpoints": {
    "health": "http://postgres-adapter:8080/health"
  },
  "metadata": {
    "environment": "production",
    "region": "us-east-1"
  },
  "network_id": "onex-prod",
  "deployment_id": "deploy-abc123",
  "epoch": 1
}
```

---

#### onex.node.heartbeat.published.v1

Liveness signal from node.

| Property | Value |
|----------|-------|
| **Meaning** | Liveness signal |
| **Semantic** | Ephemeral |
| **Key** | `node_id` |
| **Retention** | delete (1 hour) |
| **Compaction** | NO |
| **Partitions** | 3 |
| **Frequency** | Every 30 seconds |

**Payload:**
```json
{
  "node_id": "postgres-adapter-001",
  "node_type": "EFFECT",
  "node_version": "1.0.0",
  "uptime_seconds": 3600,
  "active_operations_count": 5,
  "memory_usage_mb": 256.5,
  "cpu_usage_percent": 15.2
}
```

---

#### onex.node.shutdown.announced.v1

Graceful shutdown intent.

| Property | Value |
|----------|-------|
| **Meaning** | Graceful intent to leave |
| **Semantic** | Informational |
| **Key** | `node_id` |
| **Retention** | delete (1 day) |
| **Partitions** | 3 |

**Payload:**
```json
{
  "node_id": "postgres-adapter-001",
  "node_type": "EFFECT",
  "reason": "graceful_shutdown"
}
```

---

### 6.2 Registry Control Events (Registry → System)

#### onex.registry.introspection.requested.v1

Registry requests nodes to re-emit introspection.

| Property | Value |
|----------|-------|
| **Meaning** | Registry requests re-broadcast |
| **Semantic** | Broadcast command (best-effort) |
| **Key** | `request_id` or `"broadcast"` |
| **Retention** | delete (1 hour) |
| **Partitions** | 3 |

**Guarantee**: Best-effort. Nodes may ignore or miss.

**Payload:**
```json
{
  "request_id": "req-introspect-001",
  "target_node_ids": null,
  "target_node_types": null,
  "reason": "registry_startup"
}
```

---

### 6.3 Registry State Change Events (NOT ACKS)

These are **state transition events**, not delivery acknowledgements.

#### onex.registry.node.registered.v1

Registry state transitioned to REGISTERED.

| Property | Value |
|----------|-------|
| **Meaning** | Registry state transitioned to REGISTERED |
| **Semantic** | State change event |
| **Key** | `node_id` |
| **Retention** | delete (7 days) |
| **Partitions** | 3 |
| **Consumers** | Observability, audit, caches |

**Payload:**
```json
{
  "node_id": "postgres-adapter-001",
  "registration_id": "reg-abc123",
  "consul_service_id": "postgres-adapter-001-prod",
  "postgres_row_id": "uuid-xyz789",
  "registered_at": "2025-01-15T10:30:05Z",
  "ttl_seconds": 300
}
```

---

#### onex.registry.node.registration_failed.v1

Registration workflow failed or partially failed.

| Property | Value |
|----------|-------|
| **Meaning** | Registration workflow failed |
| **Semantic** | Outcome event |
| **Key** | `node_id` |
| **Retention** | delete (7 days) |
| **Partitions** | 3 |

**Payload:**
```json
{
  "node_id": "postgres-adapter-001",
  "error_code": "CONSUL_REGISTRATION_FAILED",
  "error_message": "Consul service registration timed out",
  "consul_success": false,
  "postgres_success": true,
  "retry_eligible": true,
  "suggested_retry_delay_seconds": 30
}
```

---

#### onex.registry.node.deregistered.v1

Node removed from registry.

| Property | Value |
|----------|-------|
| **Meaning** | Node removed from registry |
| **Semantic** | State change |
| **Key** | `node_id` |
| **Retention** | delete (7 days) |
| **Partitions** | 3 |

**Payload:**
```json
{
  "node_id": "postgres-adapter-001",
  "reason": "graceful_shutdown",
  "consul_deregistered": true,
  "postgres_deregistered": true,
  "deregistered_at": "2025-01-15T11:00:05Z"
}
```

**Important**: Nodes MUST NOT rely on these events as confirmation of success.

---

### 6.4 Discovery Events (DEPRECATED FOR MVP)

Kafka-based discovery introduces RPC semantics and reply routing complexity.

**MVP Decision:**
- Discovery is performed via registry query APIs/tools
- Kafka discovery topics are OPTIONAL and EXPERIMENTAL

If used post-MVP:
- Responses keyed by `correlation_id`
- Requester filters responses
- Timeouts and retries must be explicit

---

### 6.5 Workflow Observability Events (Internal)

#### onex.registry.workflow.started.v1

| Property | Value |
|----------|-------|
| **Meaning** | Workflow lifecycle started |
| **Semantic** | Audit trail |
| **Key** | `workflow_id` |
| **Retention** | delete (7 days) |
| **Partitions** | 3 |

---

#### onex.registry.workflow.completed.v1

| Property | Value |
|----------|-------|
| **Meaning** | Workflow completed successfully |
| **Semantic** | Audit trail |
| **Key** | `workflow_id` |
| **Retention** | delete (30 days) |
| **Partitions** | 3 |

---

#### onex.registry.workflow.failed.v1

| Property | Value |
|----------|-------|
| **Meaning** | Workflow failed after retries |
| **Semantic** | Audit trail |
| **Key** | `workflow_id` |
| **Retention** | delete (30 days) |
| **Partitions** | 3 |

---

### 6.6 Infrastructure Signals

#### onex.infra.circuit_breaker.state_changed.v1

| Property | Value |
|----------|-------|
| **Meaning** | Circuit breaker transition |
| **Key** | `service_name` |
| **Retention** | delete (7 days) |
| **Partitions** | 3 |

**Payload:**
```json
{
  "service_name": "consul-adapter.production",
  "previous_state": "CLOSED",
  "new_state": "OPEN",
  "failure_count": 5,
  "reset_timeout_seconds": 60
}
```

---

#### onex.infra.error.detected.v1

| Property | Value |
|----------|-------|
| **Meaning** | Infrastructure-level error |
| **Key** | `correlation_id` |
| **Retention** | delete (7 days) |
| **Partitions** | 3 |

**Security Rule:**
- Payloads MUST redact secrets
- Hostnames/IPs allowed only in internal environments

**Payload:**
```json
{
  "error_code": "DATABASE_CONNECTION_ERROR",
  "severity": "error",
  "message": "Connection to PostgreSQL failed",
  "node_type": "EFFECT",
  "node_name": "postgres-adapter",
  "transport_type": "DATABASE",
  "operation": "connect",
  "context": {
    "host": "db.example.com",
    "port": 5432,
    "retry_count": 3
  }
}
```

---

## 7. Retention and Compaction Policy

| Event Type | Retention | Compaction |
|------------|-----------|------------|
| Heartbeats | Short (1h) | NO |
| Introspection | Medium (1d) | Optional |
| Registry state | Medium (7d) | Optional |
| Workflow telemetry | Long (30d) | NO |
| Infra errors | Medium (7d) | NO |

Compaction is deferred until post-MVP unless explicitly required.

---

## 8. Partitioning Guidance (MVP)

**Default:** 3 partitions per topic

Increase only if:
- Measured consumer lag
- Documented throughput requirement

Arbitrary partition counts are forbidden.

---

## 9. Migration from Legacy Topics

| Legacy | New |
|--------|-----|
| `node.introspection` | `onex.node.introspection.published.v1` |
| `node.heartbeat` | `onex.node.heartbeat.published.v1` |
| `node.request_introspection` | `onex.registry.introspection.requested.v1` |

---

## 10. Topic Summary

| Topic | Direction | Key | Retention |
|-------|-----------|-----|-----------|
| `onex.node.introspection.published.v1` | Node → Registry | `node_id` | 1 day |
| `onex.node.heartbeat.published.v1` | Node → Registry | `node_id` | 1 hour |
| `onex.node.shutdown.announced.v1` | Node → Registry | `node_id` | 1 day |
| `onex.registry.introspection.requested.v1` | Registry → Nodes | `request_id` | 1 hour |
| `onex.registry.node.registered.v1` | Internal | `node_id` | 7 days |
| `onex.registry.node.registration_failed.v1` | Internal | `node_id` | 7 days |
| `onex.registry.node.deregistered.v1` | Internal | `node_id` | 7 days |
| `onex.registry.workflow.started.v1` | Internal | `workflow_id` | 7 days |
| `onex.registry.workflow.completed.v1` | Internal | `workflow_id` | 30 days |
| `onex.registry.workflow.failed.v1` | Internal | `workflow_id` | 30 days |
| `onex.infra.circuit_breaker.state_changed.v1` | Internal | `service_name` | 7 days |
| `onex.infra.error.detected.v1` | Internal | `correlation_id` | 7 days |

---

## 11. Final Invariants

- Kafka carries **events**, not truth
- Postgres is the **source of record**
- Reducers are pure
- Effects perform I/O
- Orchestrators coordinate workflows
- No component assumes delivery of a specific event

**If an event is required for correctness, it belongs in the database, not Kafka.**

---

## Related Documentation

- [Correlation ID Tracking](../patterns/correlation_id_tracking.md)
- [Circuit Breaker Implementation](../patterns/circuit_breaker_implementation.md)
- [Error Handling Patterns](../patterns/error_handling_patterns.md)

## Related Tickets

- **OMN-888**: Node Registration Orchestrator (FSM)
- **OMN-889**: Dual Registration Reducer
- **OMN-890**: Registry Effect Node (Done)
- **OMN-891**: Event Models (Done)
- **OMN-893**: IntrospectionMixin (Done)
