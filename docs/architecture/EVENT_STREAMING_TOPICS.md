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

This section provides comprehensive guidance for migrating from legacy topic naming (`node.*`) to ONEX-compliant topic naming (`onex.*`). The migration is designed to be non-disruptive with full backward compatibility during the transition period.

### Legacy to ONEX Canonical Topic Mapping

| Legacy Topic | ONEX Canonical Topic | Purpose |
|--------------|---------------------|---------|
| `node.introspection` | `onex.node.introspection.published.v1` | Node capability announcements |
| `node.heartbeat` | `onex.node.heartbeat.published.v1` | Node liveness signals |
| `node.request_introspection` | `onex.registry.introspection.requested.v1` | Registry re-broadcast requests |

### Naming Convention Differences

| Aspect | Legacy Format | ONEX Format |
|--------|--------------|-------------|
| **Prefix** | `node.` | `onex.` (platform namespace) |
| **Structure** | `<entity>.<name>` | `<domain>.<entity>.<event>.v<version>` |
| **Versioning** | None | Required (`.v1`, `.v2`, etc.) |
| **Example** | `node.introspection` | `onex.node.introspection.published.v1` |

**Why ONEX Format?**
- Explicit versioning enables schema evolution without breaking consumers
- Domain namespace prevents collisions in multi-service environments
- Event suffix (`published`, `requested`) clarifies message semantics
- Consistent with CloudEvents and industry best practices

### Backward Compatibility

**Legacy topics are fully supported** during migration:
- `ModelIntrospectionConfig` defaults to legacy topic names for backward compatibility
- Legacy topic names generate a warning (not an error) during validation
- No code changes required to continue using legacy topics
- Migration can be performed incrementally, node by node

The validation warning looks like:
```
UserWarning: Topic 'node.introspection' does not have version suffix.
Consider using ONEX format: onex.<domain>.<name>.v1
```

### Migration Strategy

**Recommended Approach**: Parallel production with gradual consumer migration.

#### Phase 1: Assessment and Preparation (Day 0)

**Duration**: 1-2 days

1. **Inventory existing topics and consumers**:
   ```bash
   # List all topics
   rpk topic list | grep -E "^node\."

   # List consumer groups using legacy topics
   rpk group list
   rpk group describe <group-name>
   ```

2. **Create ONEX topics** (if not auto-created):
   ```bash
   rpk topic create onex.node.introspection.published.v1 \
       --partitions 3 \
       --config retention.ms=86400000

   rpk topic create onex.node.heartbeat.published.v1 \
       --partitions 3 \
       --config retention.ms=3600000

   rpk topic create onex.registry.introspection.requested.v1 \
       --partitions 3 \
       --config retention.ms=3600000
   ```

3. **Document current consumer groups** for rollback reference

#### Phase 2: Dual Publishing (Day 1-7)

**Duration**: 1 week minimum (allows observation)

Configure producers to publish to both legacy and ONEX topics simultaneously.

**Using ModelIntrospectionConfig for Dual Publishing**:

```python
from uuid import uuid4
from omnibase_infra.models.discovery import ModelIntrospectionConfig
from omnibase_infra.mixins import MixinNodeIntrospection

# Legacy configuration (current default)
legacy_config = ModelIntrospectionConfig(
    node_id=uuid4(),
    node_type="EFFECT",
    event_bus=event_bus,
    # These are the defaults - legacy topic names
    introspection_topic="node.introspection",
    heartbeat_topic="node.heartbeat",
    request_introspection_topic="node.request_introspection",
)

# ONEX configuration (migration target)
onex_config = ModelIntrospectionConfig(
    node_id=uuid4(),
    node_type="EFFECT",
    event_bus=event_bus,
    # ONEX-compliant topic names
    introspection_topic="onex.node.introspection.published.v1",
    heartbeat_topic="onex.node.heartbeat.published.v1",
    request_introspection_topic="onex.registry.introspection.requested.v1",
)
```

**Dual Publishing Implementation**:

```python
from omnibase_infra.models.events import ModelNodeIntrospectionEvent

# Topic constants for migration period
LEGACY_INTROSPECTION_TOPIC = "node.introspection"
ONEX_INTROSPECTION_TOPIC = "onex.node.introspection.published.v1"

LEGACY_HEARTBEAT_TOPIC = "node.heartbeat"
ONEX_HEARTBEAT_TOPIC = "onex.node.heartbeat.published.v1"

async def publish_introspection_dual(
    event_bus: ProtocolEventBus,
    event: ModelNodeIntrospectionEvent,
) -> None:
    """Publish to both legacy and ONEX topics during migration.

    This ensures all consumers (legacy and migrated) receive events.
    """
    # Publish to legacy topic for existing consumers
    await event_bus.publish_envelope(LEGACY_INTROSPECTION_TOPIC, event)

    # Publish to ONEX topic for migrated consumers
    await event_bus.publish_envelope(ONEX_INTROSPECTION_TOPIC, event)
```

**Verification Steps**:
```bash
# Verify messages appear on both topics
rpk topic consume node.introspection --num 1
rpk topic consume onex.node.introspection.published.v1 --num 1

# Compare message rates (should be equal)
rpk topic describe node.introspection
rpk topic describe onex.node.introspection.published.v1
```

#### Phase 3: Consumer Migration (Day 8-14)

**Duration**: 1 week (staggered rollout)

Migrate consumers to ONEX topics one service at a time.

**Consumer Group Strategy**:
- Create **new consumer groups** for ONEX topics
- Do NOT reuse legacy consumer group names (offset tracking would be lost)
- Naming convention: `<service>-onex-v1` (e.g., `registry-introspection-onex-v1`)

**Before (Legacy Consumer)**:
```python
# Legacy consumer configuration
consumer_config = {
    "group.id": "registry-introspection",
    "auto.offset.reset": "earliest",
}
consumer.subscribe(["node.introspection", "node.heartbeat"])
```

**After (ONEX Consumer)**:
```python
# ONEX consumer configuration with new group
consumer_config = {
    "group.id": "registry-introspection-onex-v1",  # New group name
    "auto.offset.reset": "earliest",
}
consumer.subscribe([
    "onex.node.introspection.published.v1",
    "onex.node.heartbeat.published.v1",
])
```

**Node Configuration Migration**:
```python
from uuid import uuid4
from omnibase_infra.models.discovery import ModelIntrospectionConfig
from omnibase_infra.mixins import MixinNodeIntrospection


class MyNode(MixinNodeIntrospection):
    """Node with ONEX-compliant topic configuration."""

    def __init__(self, node_id: uuid4, event_bus: ProtocolEventBus) -> None:
        # Migrate to ONEX topics via configuration
        config = ModelIntrospectionConfig(
            node_id=node_id,
            node_type="EFFECT",
            event_bus=event_bus,
            version="1.0.0",
            # ONEX-compliant topic names
            introspection_topic="onex.node.introspection.published.v1",
            heartbeat_topic="onex.node.heartbeat.published.v1",
            request_introspection_topic="onex.registry.introspection.requested.v1",
        )
        self.initialize_introspection_from_config(config)
```

**Verification Steps**:
```bash
# Monitor consumer lag on new groups
rpk group describe registry-introspection-onex-v1

# Verify no lag accumulation on legacy topics (consumers migrated away)
rpk group describe registry-introspection

# Check consumer offsets are advancing
watch -n 5 'rpk group describe registry-introspection-onex-v1'
```

#### Phase 4: Legacy Deprecation (Day 15-21)

**Duration**: 1 week observation, then cleanup

1. **Stop dual publishing** (publish only to ONEX topics):
   ```python
   # Final configuration - ONEX only
   config = ModelIntrospectionConfig(
       node_id=uuid4(),
       node_type="EFFECT",
       event_bus=event_bus,
       introspection_topic="onex.node.introspection.published.v1",
       heartbeat_topic="onex.node.heartbeat.published.v1",
       request_introspection_topic="onex.registry.introspection.requested.v1",
   )
   ```

2. **Monitor for stragglers**:
   ```bash
   # Check if any consumers still reading legacy topics
   rpk group list | xargs -I {} rpk group describe {}

   # Check for any lag on legacy topics (indicates unmigrated consumers)
   rpk topic describe node.introspection
   ```

3. **Delete legacy topics** after retention period:
   ```bash
   # Wait for retention period (typically 1-7 days)
   # Then delete legacy topics
   rpk topic delete node.introspection
   rpk topic delete node.heartbeat
   rpk topic delete node.request_introspection
   ```

4. **Update documentation** and remove legacy topic references

### Environment-Based Configuration

For staged rollouts, use environment variables to control topic names:

```python
import os
from uuid import uuid4
from omnibase_infra.models.discovery import ModelIntrospectionConfig

# Environment-driven topic selection
USE_ONEX_TOPICS = os.getenv("USE_ONEX_TOPICS", "false").lower() == "true"

if USE_ONEX_TOPICS:
    introspection_topic = "onex.node.introspection.published.v1"
    heartbeat_topic = "onex.node.heartbeat.published.v1"
    request_topic = "onex.registry.introspection.requested.v1"
else:
    introspection_topic = "node.introspection"
    heartbeat_topic = "node.heartbeat"
    request_topic = "node.request_introspection"

config = ModelIntrospectionConfig(
    node_id=uuid4(),
    node_type="EFFECT",
    event_bus=event_bus,
    introspection_topic=introspection_topic,
    heartbeat_topic=heartbeat_topic,
    request_introspection_topic=request_topic,
)
```

**Deployment Strategy**:
```bash
# Canary deployment (10% of nodes)
kubectl set env deployment/node-service USE_ONEX_TOPICS=true --containers=canary

# Full rollout after validation
kubectl set env deployment/node-service USE_ONEX_TOPICS=true
```

### Rollback Procedures

If issues are detected during migration:

#### Rollback During Phase 2 (Dual Publishing)

No rollback needed - legacy consumers continue working.

#### Rollback During Phase 3 (Consumer Migration)

```python
# Revert consumer subscription to legacy topics
consumer.subscribe(["node.introspection", "node.heartbeat"])

# Revert node configuration to legacy topics
config = ModelIntrospectionConfig(
    node_id=node_id,
    node_type="EFFECT",
    event_bus=event_bus,
    # Revert to legacy defaults
    introspection_topic="node.introspection",
    heartbeat_topic="node.heartbeat",
    request_introspection_topic="node.request_introspection",
)
```

#### Rollback After Phase 4 (Legacy Deleted)

If legacy topics were deleted but rollback is needed:

1. **Recreate legacy topics**:
   ```bash
   rpk topic create node.introspection --partitions 3
   rpk topic create node.heartbeat --partitions 3
   rpk topic create node.request_introspection --partitions 3
   ```

2. **Re-enable dual publishing** in producers

3. **Revert consumer configurations** to legacy topics

### Key Migration Considerations

- **Message Format**: Payload schemas remain unchanged; only topic names change
- **Consumer Groups**: Create new consumer groups for ONEX topics to track offset independently
- **Monitoring**: Compare message rates between legacy and ONEX topics during migration
- **Rollback**: Keep legacy topic publishing capability until migration is verified
- **Validation Warnings**: Legacy topics generate warnings but are allowed for backward compatibility
- **Topic ACLs**: Update ACL rules for new ONEX topic names (see Section 12)

### Migration Timeline Summary

| Phase | Duration | Actions | Verification |
|-------|----------|---------|--------------|
| Assessment | 1-2 days | Inventory topics, create ONEX topics | Topics created |
| Dual Publishing | 1 week | Enable dual publishing | Messages on both topics |
| Consumer Migration | 1 week | Migrate consumers one by one | Consumer lag monitored |
| Legacy Deprecation | 1 week | Stop legacy publishing, delete topics | No legacy consumers |

**Total Migration Time**: 3-4 weeks (recommended minimum)

---

## 10. Topic Summary

**Total Topics: 12** (organized by domain: node lifecycle, registry control, registry state, workflow observability, infrastructure signals)

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

## 12. Security and Access Control

### Topic ACL Configuration

In production and multi-tenant environments, Kafka topic access must be controlled using ACLs (Access Control Lists).

#### Recommended ACL Policies

| Topic Pattern | Principal | Operation | Permission |
|--------------|-----------|-----------|------------|
| `onex.node.*` | Node services | WRITE | ALLOW |
| `onex.node.*` | Registry services | READ | ALLOW |
| `onex.registry.*` | Registry services | WRITE | ALLOW |
| `onex.registry.*` | Node services | READ | ALLOW |
| `onex.infra.*` | Infrastructure services | WRITE | ALLOW |
| `onex.infra.*` | Observability services | READ | ALLOW |

#### ACL Configuration Example (Redpanda/Kafka)

```bash
# Allow node services to produce to node lifecycle topics
rpk acl create --allow-principal 'User:node-service' \
    --operation write \
    --topic 'onex.node.*'

# Allow registry to consume node lifecycle events
rpk acl create --allow-principal 'User:registry-service' \
    --operation read \
    --topic 'onex.node.*'

# Allow registry to produce state change events
rpk acl create --allow-principal 'User:registry-service' \
    --operation write \
    --topic 'onex.registry.*'

# Allow observability systems to consume all topics
rpk acl create --allow-principal 'User:observability' \
    --operation read \
    --topic 'onex.*'
```

### Introspection Topic Security

Introspection topics (`onex.node.introspection.published.v1`, `onex.registry.introspection.requested.v1`) require special consideration:

**Security Concerns**:
- Introspection data exposes node capabilities and method signatures
- Attackers could use introspection data for reconnaissance
- Request topic allows triggering re-broadcasts from all nodes

**Recommended Controls**:
1. **Network Segmentation**: Isolate introspection topics to internal network only
2. **Consumer Whitelisting**: Only allow authorized consumers (registry, monitoring)
3. **Rate Limiting**: Limit introspection request frequency to prevent abuse
4. **Audit Logging**: Log all consumers of introspection topics

### Multi-Tenant Considerations

For multi-tenant deployments:

1. **Topic Namespacing**: Consider prefixing topics with tenant ID
   - Example: `tenant-123.onex.node.introspection.published.v1`

2. **Consumer Group Isolation**: Each tenant should use isolated consumer groups

3. **Cross-Tenant Access**: Disable by default; require explicit ACL grants

4. **Topic Quotas**: Set per-tenant quotas to prevent resource exhaustion

---

## 13. Contract Integration

ONEX node contracts declare their event channel dependencies in `contract.yaml`. This enables:
- Static analysis of event topology
- Automatic topic validation during deployment
- Clear documentation of node communication patterns

### Example: Node Contract Event Channels

```yaml
# contract.yaml - Event channel configuration for a registry effect node
meta:
  name: registry-effect
  version: 1.0.0
  node_type: EFFECT

# event_channels is at top level, NOT under meta
event_channels:
  # Topics this node publishes to
  publishes:
    - topic: onex.node.introspection.published.v1
      key: node_id
      event_type: introspection
      description: Announces node capabilities on startup and refresh

    - topic: onex.node.heartbeat.published.v1
      key: node_id
      event_type: heartbeat
      description: Periodic liveness signal

    - topic: onex.node.shutdown.announced.v1
      key: node_id
      event_type: shutdown
      description: Graceful shutdown notification

  # Topics this node subscribes to
  subscribes:
    - topic: onex.registry.introspection.requested.v1
      key: request_id
      event_type: request
      description: Registry broadcast requesting introspection refresh
      handler: handle_introspection_request
```

### Example: Orchestrator Contract with Workflow Events

```yaml
# contract.yaml - Event channel configuration for a registration orchestrator
meta:
  name: registration-orchestrator
  version: 1.0.0
  node_type: ORCHESTRATOR

# event_channels is at top level, NOT under meta
event_channels:
  publishes:
    - topic: onex.registry.node.registered.v1
      key: node_id
      event_type: state_change
      description: Emitted when node registration completes successfully

    - topic: onex.registry.node.registration_failed.v1
      key: node_id
      event_type: state_change
      description: Emitted when node registration fails

    - topic: onex.registry.workflow.started.v1
      key: workflow_id
      event_type: workflow
      description: Workflow audit trail - started

    - topic: onex.registry.workflow.completed.v1
      key: workflow_id
      event_type: workflow
      description: Workflow audit trail - completed

    - topic: onex.registry.workflow.failed.v1
      key: workflow_id
      event_type: workflow
      description: Workflow audit trail - failed

  subscribes:
    - topic: onex.node.introspection.published.v1
      key: node_id
      event_type: introspection
      description: Triggers registration workflow for new/updated nodes
      handler: handle_introspection_event
```

### Event Channel Schema Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `topic` | string | Yes | Full topic name (must match Section 10 topics) |
| `key` | string | Yes | Kafka message key field (see Section 5 keying rules) |
| `event_type` | string | Yes | Logical event category |
| `description` | string | No | Human-readable purpose |
| `handler` | string | No | Method name for subscription handlers |

### Validation Rules

Contract event channels are validated against this specification:

1. **Topic names** must exist in the canonical topic list (Section 10)
2. **Keys** must match the keying rules for that topic class (Section 5)
3. **Publish/subscribe direction** must be consistent with topic semantics
4. **Node types** must match expected publishers (e.g., EFFECT nodes publish lifecycle events)

### Validation Tooling

The `ModelIntrospectionConfig` class provides typed configuration validation using Pydantic:

```python
from uuid import uuid4

from omnibase_infra.mixins import ModelIntrospectionConfig
from pydantic import ValidationError

# Valid configuration
config = ModelIntrospectionConfig(
    node_id=uuid4(),
    node_type="EFFECT",
)

# Invalid - empty node_type raises ValidationError
try:
    config = ModelIntrospectionConfig(
        node_id=uuid4(),
        node_type="",  # Invalid: must have min_length=1
    )
except ValidationError as e:
    print(e)  # String should have at least 1 character
```

**Validation Rules Enforced by ModelIntrospectionConfig**:

| Field | Rule | Description |
|-------|------|-------------|
| `node_id` | UUID format | Must be valid UUID (strings auto-converted) |
| `node_type` | Non-empty | Must have at least 1 character |
| `cache_ttl` | Non-negative | Must be >= 0.0 |
| `extra` | Forbidden | No extra fields allowed (frozen model) |

**Note:** Topic naming validation is available via contract-driven topic configuration (implemented in OMN-881). Nodes can define their event channels in `contract.yaml` and the runtime validates topic names against the canonical list.

See `src/omnibase_infra/models/discovery/model_introspection_config.py` for the configuration model implementation.

### Code Integration Example

The following shows how a node's `contract.yaml` event_channels integrate with `MixinNodeIntrospection` at runtime.

#### Contract Definition

First, define the event channels in `contract.yaml`. The `event_type` field serves as the lookup key that the Python code uses to extract topic names:

```yaml
# contract.yaml - RegistryEffectNode event channel configuration
#
# This contract defines topics for the RegistryEffectNode Python class below.
# The event_type field is the lookup key used by the Python code.
meta:
  name: registry-effect-node
  version: 1.0.0
  node_type: EFFECT

# event_channels is at top level, NOT under meta
event_channels:
  # Topics this node publishes to
  # Python: publishes = {ch.event_type: ch.topic for ch in contract.event_channels.publishes}
  publishes:
    - event_type: introspection                      # Key: "introspection"
      topic: onex.node.introspection.published.v1   # Value: used for introspection_topic
      key: node_id
      description: Node capability announcements on startup and refresh

    - event_type: heartbeat                          # Key: "heartbeat"
      topic: onex.node.heartbeat.published.v1       # Value: used for heartbeat_topic
      key: node_id
      description: Periodic liveness signals (every 30 seconds)

    - event_type: shutdown                           # Key: "shutdown"
      topic: onex.node.shutdown.announced.v1        # Value: NOT used by MixinNodeIntrospection
      key: node_id                                   # (handled separately by node shutdown logic)
      description: Graceful shutdown notification

  # Topics this node subscribes to
  # Python: subscribes = {ch.event_type: ch.topic for ch in contract.event_channels.subscribes}
  subscribes:
    - event_type: request                            # Key: "request"
      topic: onex.registry.introspection.requested.v1  # Value: used for request_introspection_topic
      key: request_id
      description: Registry broadcast requesting introspection refresh
      handler: handle_introspection_request
```

**Contract-to-Code Mapping:**

| Contract `event_type` | Contract `topic` | Module Constant |
|-----------------------|------------------|-----------------|
| `introspection` | `onex.node.introspection.published.v1` | `INTROSPECTION_TOPIC` |
| `heartbeat` | `onex.node.heartbeat.published.v1` | `HEARTBEAT_TOPIC` |
| `shutdown` | `onex.node.shutdown.announced.v1` | *(not in MixinNodeIntrospection)* |
| `request` | `onex.registry.introspection.requested.v1` | `REQUEST_INTROSPECTION_TOPIC` |

**Note:** The contract topics listed above are the canonical ONEX topics. The current implementation uses legacy topic names (`node.introspection`, `node.heartbeat`, `node.request_introspection`). Migration to canonical topic names is tracked in Section 9.

#### Python Integration

The node code reads the contract and initializes `MixinNodeIntrospection` using `ModelIntrospectionConfig`:

```python
# node.py - Initializing MixinNodeIntrospection with configuration
from uuid import UUID

from omnibase_infra.mixins import MixinNodeIntrospection, ModelIntrospectionConfig
from omnibase_infra.event_bus import KafkaEventBus

class RegistryEffectNode(MixinNodeIntrospection):
    """Effect node that uses MixinNodeIntrospection for capability discovery."""

    def __init__(
        self,
        contract: NodeContract,
        event_bus: KafkaEventBus,
    ) -> None:
        # Configure introspection with typed configuration model
        # Topics can be configured via contract.yaml event_channels (implemented in OMN-881).
        # Default topic names in mixin_node_introspection.py are used if not overridden:
        # - INTROSPECTION_TOPIC = "node.introspection"
        # - HEARTBEAT_TOPIC = "node.heartbeat"
        # - REQUEST_INTROSPECTION_TOPIC = "node.request_introspection"
        config = ModelIntrospectionConfig(
            node_id=UUID(contract.metadata.name) if isinstance(contract.metadata.name, str) else contract.metadata.name,
            node_type=contract.metadata.node_type,
            version=contract.metadata.version,
            event_bus=event_bus,
        )
        self.initialize_introspection_from_config(config)
```

**Key Points:**
- Topics are declared in `contract.yaml` for documentation and topology analysis
- `MixinNodeIntrospection` uses module-level topic constants as defaults (`INTROSPECTION_TOPIC`, `HEARTBEAT_TOPIC`, `REQUEST_INTROSPECTION_TOPIC`)
- Contract-driven topic configuration via `ModelIntrospectionConfig` is implemented (OMN-881) and can override defaults
- **Shutdown events are a separate concern**: The `onex.node.shutdown.announced.v1` topic is not managed by `MixinNodeIntrospection`. Nodes should publish shutdown events directly via their shutdown/cleanup logic, not through the introspection mixin.
- Contract validation (see Validation Rules above) ensures topics exist in the canonical list
- This pattern enables static analysis of event topology across all nodes

**Current Behavior:**
Topic names are defined as module-level constants in `mixin_node_introspection.py` as defaults. Contract-driven topic configuration is available via `ModelIntrospectionConfig` (implemented in OMN-881), enabling topology analysis and multi-tenant deployments.

---

## 14. Performance Metrics and Thresholds

### Event Publishing Metrics

The following metrics should be monitored for event streaming health:

| Metric | Expected Threshold | Alert Threshold | Description |
|--------|-------------------|-----------------|-------------|
| `publish_latency_ms` | < 50ms (p50) | > 200ms (p99) | Time to publish event to Kafka |
| `publish_success_rate` | > 99.9% | < 99% | Percentage of successful publishes |
| `event_queue_depth` | < 100 | > 1000 | Pending events in local buffer |
| `circuit_breaker_open_count` | 0 | > 0 | Number of open circuit breakers |

### Heartbeat Monitoring

For node heartbeat events (`onex.node.heartbeat.published.v1`):

| Metric | Expected Value | Alert Condition | Action |
|--------|---------------|-----------------|--------|
| `heartbeat_interval_seconds` | 30 | Missing > 90s | Node may be unhealthy |
| `heartbeat_jitter_seconds` | < 5 | > 15 | Network or scheduling issues |
| `heartbeat_payload_size_bytes` | < 1KB | > 10KB | Payload bloat |

### Consumer Lag Thresholds

| Topic Category | Acceptable Lag | Warning Lag | Critical Lag |
|---------------|----------------|-------------|--------------|
| Heartbeats | < 30s | > 60s | > 120s |
| Introspection | < 60s | > 120s | > 300s |
| Registry state | < 30s | > 60s | > 120s |
| Workflow events | < 120s | > 300s | > 600s |
| Infrastructure | < 60s | > 120s | > 300s |

### Performance Event Fields

Introspection events include performance metrics in the payload:

```json
{
  "node_id": "postgres-adapter-001",
  "node_type": "EFFECT",
  "performance_metrics": {
    "cpu_usage_percent": 15.2,
    "memory_usage_mb": 256.5,
    "uptime_seconds": 3600,
    "active_operations_count": 5
  }
}
```

**Interpretation Guidelines**:

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| `cpu_usage_percent` | < 70% | 70-85% | > 85% |
| `memory_usage_mb` | < 80% of limit | 80-90% of limit | > 90% of limit |
| `active_operations_count` | < pool_size | = pool_size | > pool_size (queuing) |

### Alerting Best Practices

1. **Rate-based alerts**: Alert on rate of change, not absolute values
   - Example: "Heartbeat rate dropped 50% in 5 minutes"

2. **Composite alerts**: Combine metrics for meaningful alerts
   - Example: "High latency AND high CPU" vs just "high latency"

3. **Silence during deployment**: Suppress alerts during planned deployments

4. **Escalation tiers**:
   - Warning: Slack notification
   - Critical: PagerDuty alert
   - Emergency: Phone call

---

## Related Documentation

- [Correlation ID Tracking](../patterns/correlation_id_tracking.md)
- [Circuit Breaker Implementation](../patterns/circuit_breaker_implementation.md)
- [Error Handling Patterns](../patterns/error_handling_patterns.md)

## Future Enhancements

### EnumONEXTopic for Type Safety (Post-MVP)

Consider introducing an `EnumONEXTopic` enum for topic name type safety. Benefits include IDE autocomplete, compile-time typo prevention, and centralized topic management. Trade-offs: reduced flexibility for dynamic topic generation and additional maintenance when adding new topics. Recommended for post-MVP evaluation when topic inventory stabilizes.

---

## Related Tickets

- **OMN-881**: Contract-driven topic configuration (Done)
- **OMN-888**: Node Registration Orchestrator (FSM)
- **OMN-889**: Dual Registration Reducer
- **OMN-890**: Registry Effect Node (Done)
- **OMN-891**: Event Models (Done)
- **OMN-893**: IntrospectionMixin (Done)
