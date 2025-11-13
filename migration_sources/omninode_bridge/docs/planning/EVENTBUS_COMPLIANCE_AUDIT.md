# ONEX EventBus Compliance Audit Report

**Audit Date:** 2025-10-25
**Auditor:** Claude Code (code-review role)
**Correlation ID:** defbd17e-dbd7-418f-985b-38f9869d6f19

## Executive Summary

Comprehensive audit of all 9 nodes in the omninode_bridge repository identified significant EventBus compliance gaps:

- **Total Nodes:** 9
- **✅ Compliant:** 2 (22.2%) - orchestrator, reducer
- **❌ Non-Compliant:** 7 (77.8%)
- **⚠️ Has Event TODOs:** 2 (deployment_receiver_effect, deployment_sender_effect)
- **❌ Direct Kafka Access:** 2 (deployment_receiver_effect, deployment_sender_effect)
- **Estimated Refactoring Effort:** 6.0 hours (0.7 days)

## Compliance Matrix

| Node Name | EventBus | Mixins | Direct Kafka | TODOs | Status |
|-----------|----------|--------|--------------|-------|--------|
| orchestrator | ✅ | ✅ | ✅ NO | ✅ NO | ✅ PASS |
| reducer | ✅ | ✅ | ✅ NO | ✅ NO | ✅ PASS |
| deployment_receiver_effect | ❌ | ✅ | ❌ YES | ⚠️ YES | ❌ FAIL |
| deployment_sender_effect | ❌ | ⚠️ | ❌ YES | ⚠️ YES | ❌ FAIL |
| database_adapter_effect | ❌ | ✅ | ✅ NO | ✅ NO | ❌ FAIL |
| store_effect | ❌ | ⚠️ | ✅ NO | ✅ NO | ❌ FAIL |
| registry | ❌ | ✅ | ✅ NO | ✅ NO | ❌ FAIL |
| codegen_orchestrator | ❌ | ⚠️ | ✅ NO | ✅ NO | ❌ FAIL |
| codegen_metrics_reducer | ❌ | ✅ | ✅ NO | ✅ NO | ❌ FAIL |

## Critical Findings

### High Severity Issues

1. **deployment_receiver_effect** (CRITICAL)
   - Direct Kafka producer usage (line 115)
   - 2 unimplemented event TODOs (lines 400, 638)
   - Missing EventBus integration
   - Estimated effort: 2.2 hours

2. **deployment_sender_effect** (CRITICAL)
   - Direct Kafka producer field (line 78)
   - 1 unimplemented event TODO (line 554)
   - Missing EventBus integration
   - Estimated effort: 1.3 hours

### Medium Severity Issues

3. **database_adapter_effect** - Missing EventBus (0.5h)
4. **store_effect** - Missing EventBus (0.5h)
5. **registry** - Missing EventBus + architectural issues (0.5h)

### Low Severity Issues

6. **codegen_orchestrator** - Missing EventBus (0.5h)
7. **codegen_metrics_reducer** - Partially compliant via MixinIntentPublisher (0.5h)

## Two EventBus Integration Patterns

### Pattern 1: Direct EventBus (Orchestrator Nodes)

**Used by:** orchestrator node
**Best for:** Orchestrator nodes, top-level coordination

```python
# In __init__
from ....services.event_bus import EventBusService

self.event_bus = container.get_service("event_bus")
if self.event_bus is None:
    self.event_bus = EventBusService(
        kafka_client=self.kafka_client,
        node_id=self.node_id,
        namespace=self.default_namespace,
    )
    container.register_service("event_bus", self.event_bus)

# Usage
await self.event_bus.publish_action_event(
    correlation_id=correlation_id,
    action_type="ACTION_NAME",
    payload={"key": "value"},
)
```

**Advantages:**
- Simple and direct
- Immediate event publishing
- Error handling at call site

**Disadvantages:**
- Tight coupling to EventBus service

### Pattern 2: Intent-Based (Reducer/Effect Nodes)

**Used by:** reducer node
**Best for:** Reducer nodes, Effect nodes, pure functions

```python
# Generate intent for orchestrator to handle
intent = ModelIntent(
    intent_type=EnumIntentType.PUBLISH_EVENT.value,
    target="event_bus",
    payload={
        "event_type": event_type.value,
        "topic": topic_name,
        "data": data,
        "correlation_id": correlation_id,
        "timestamp": datetime.now(UTC).isoformat(),
    },
    priority=0,
)
self._pending_event_intents.append(intent)
```

**Advantages:**
- Pure function approach (ONEX v2.0 compliant)
- Decoupled from infrastructure
- Testable without Kafka
- Orchestrator handles execution

**Disadvantages:**
- Requires orchestrator support for intent execution

## Prioritized Refactoring Plan

### Phase 1: Critical Deployment Nodes (Week 1)

**deployment_receiver_effect (2.2h)**
- Remove direct Kafka producer (line 115)
- Add EventBus initialization (Pattern 1: Direct)
- Implement TODO at line 400: DEPLOYMENT_RECEIVED event
- Implement TODO at line 638: DEPLOYMENT_COMPLETED event
- Add lifecycle events: DEPLOYMENT_STARTED, DEPLOYMENT_FAILED

**deployment_sender_effect (1.3h)**
- Remove direct Kafka producer field (line 78)
- Add EventBus initialization (Pattern 1: Direct)
- Implement TODO at line 554: DEPLOYMENT_SENT event
- Add lifecycle events: DEPLOYMENT_INITIATED, DEPLOYMENT_SEND_FAILED

### Phase 2: Infrastructure Nodes (Week 2)

**database_adapter_effect (0.5h)**
- Add EventBus integration (Pattern 2: Intent)
- Add events: DATABASE_WRITE, DATABASE_READ, DATABASE_ERROR

**store_effect (0.5h)**
- Add EventBus integration (Pattern 2: Intent)
- Add events: STORE_WRITE, STORE_READ, STORE_DELETE

### Phase 3: Registry Node (Week 3)

**registry (0.5h)**
- Fix base class inheritance (architectural review needed)
- Add EventBus integration after base class fixed
- Add events: NODE_REGISTERED, NODE_UNREGISTERED, NODE_HEARTBEAT

### Phase 4: Codegen Nodes (Week 4)

**codegen_orchestrator (0.5h)**
- Add EventBus initialization (Pattern 1: Direct)
- Add events: CODEGEN_STARTED, CODEGEN_COMPLETED, CODEGEN_FAILED

**codegen_metrics_reducer (0.5h)**
- Verify MixinIntentPublisher configuration
- Ensure all metric events are published

## Implementation Checklist

### Phase 1: Critical (Week 1)
- [ ] deployment_receiver_effect refactoring
  - [ ] Remove direct Kafka producer
  - [ ] Add EventBus initialization
  - [ ] Implement TODO at line 400
  - [ ] Implement TODO at line 638
  - [ ] Add lifecycle event publishing
  - [ ] Update tests

- [ ] deployment_sender_effect refactoring
  - [ ] Remove direct Kafka producer field
  - [ ] Add EventBus initialization
  - [ ] Implement TODO at line 554
  - [ ] Add lifecycle event publishing
  - [ ] Update tests

### Phase 2: Infrastructure (Week 2)
- [ ] database_adapter_effect refactoring
  - [ ] Add intent-based event publishing
  - [ ] Add database operation events
  - [ ] Update tests

- [ ] store_effect refactoring
  - [ ] Add intent-based event publishing
  - [ ] Add store operation events
  - [ ] Update tests

### Phase 3: Registry Fix (Week 3)
- [ ] registry architectural review
  - [ ] Fix base class inheritance
  - [ ] Add EventBus integration
  - [ ] Add registration events
  - [ ] Update tests

### Phase 4: Codegen (Week 4)
- [ ] codegen_orchestrator refactoring
  - [ ] Add EventBus initialization
  - [ ] Add codegen lifecycle events
  - [ ] Update tests

- [ ] codegen_metrics_reducer verification
  - [ ] Verify intent publishing works
  - [ ] Add any missing metric events
  - [ ] Update tests

## Testing Strategy

### Unit Tests
- [ ] Test EventBus initialization
- [ ] Test event publishing success
- [ ] Test event publishing failure handling
- [ ] Test intent generation (for intent-based nodes)

### Integration Tests
- [ ] Test end-to-end event flow
- [ ] Test event correlation across nodes
- [ ] Test Kafka topic routing
- [ ] Test event schema validation

### Performance Tests
- [ ] Measure event publishing latency
- [ ] Test high-volume event throughput
- [ ] Verify no memory leaks

## Success Criteria

- [ ] All 7 non-compliant nodes refactored
- [ ] No direct Kafka producer usage anywhere
- [ ] All event TODOs implemented
- [ ] EventBus compliance: 100%
- [ ] All tests passing (maintain 92.8%+ coverage)
- [ ] No performance degradation
- [ ] Documentation updated

## Risk Assessment

| Risk | Severity | Impact | Mitigation |
|------|----------|--------|------------|
| Breaking existing functionality | HIGH | CRITICAL | Comprehensive test coverage before refactoring |
| Event schema inconsistencies | MEDIUM | MEDIUM | Use EventBus typed methods (publish_action_event, etc.) |
| Performance degradation | MEDIUM | MEDIUM | Benchmark before/after, use async publishing |
| Registry node architecture | HIGH | MEDIUM | Separate architectural review before EventBus work |
| Timeline slippage | MEDIUM | LOW | Start with critical nodes first (deployment nodes) |

## Detailed Node Analysis

### 1. deployment_receiver_effect (CRITICAL - 2.2h)

**Current State:**
- Base class: `NodeEffect`
- Mixins: HealthCheckMixin
- EventBus: ❌ Missing
- Direct Kafka: ❌ Yes (line 115)
- Event TODOs: ⚠️ 2 (lines 400, 638)

**Issues Found:**
```python
# Line 115 - Direct Kafka producer (REMOVE)
self.kafka_producer = getattr(container, "kafka_producer", None)

# Line 400 - TODO
# TODO: Implement Kafka event publishing

# Line 638 - TODO
# TODO: Actual Kafka publishing implementation
```

**Refactoring Template:**
```python
# In __init__ - Add EventBus initialization
from ....services.event_bus import EventBusService

self.event_bus = container.get_service("event_bus")
if self.event_bus is None:
    self.event_bus = EventBusService(
        kafka_client=self.kafka_client,
        node_id=self.node_id,
        namespace=self.default_namespace,
    )
    container.register_service("event_bus", self.event_bus)

# Replace line 400 TODO
await self.event_bus.publish_action_event(
    correlation_id=correlation_id,
    action_type="DEPLOYMENT_RECEIVED",
    payload={
        "deployment_id": deployment_id,
        "service_name": service_name,
        "status": "received",
        "timestamp": datetime.now(UTC).isoformat(),
    },
)

# Replace line 638 TODO
await self.event_bus.publish_action_event(
    correlation_id=correlation_id,
    action_type="DEPLOYMENT_COMPLETED",
    payload={
        "deployment_id": deployment_id,
        "service_name": service_name,
        "status": "completed",
        "duration_ms": duration_ms,
        "timestamp": datetime.now(UTC).isoformat(),
    },
)
```

**Recommended Events:**
- `DEPLOYMENT_RECEIVED` - When deployment package received
- `DEPLOYMENT_STARTED` - When deployment begins
- `DEPLOYMENT_COMPLETED` - When deployment succeeds
- `DEPLOYMENT_FAILED` - When deployment fails

### 2. deployment_sender_effect (CRITICAL - 1.3h)

**Current State:**
- Base class: `NodeEffect`
- Mixins: None
- EventBus: ❌ Missing
- Direct Kafka: ❌ Yes (line 78)
- Event TODOs: ⚠️ 1 (line 554)

**Issues Found:**
```python
# Line 78 - Direct Kafka producer field (REMOVE)
self._kafka_producer: Optional[Any] = None

# Line 554 - TODO
# TODO: Implement actual Kafka producer integration
```

**Refactoring Template:**
```python
# Remove line 78
# self._kafka_producer: Optional[Any] = None  # DELETE THIS

# In __init__ - Add EventBus initialization
from ....services.event_bus import EventBusService

self.event_bus = container.get_service("event_bus")
if self.event_bus is None:
    self.event_bus = EventBusService(
        kafka_client=self.kafka_client,
        node_id=self.node_id,
        namespace=self.default_namespace,
    )
    container.register_service("event_bus", self.event_bus)

# Replace line 554 TODO
await self.event_bus.publish_action_event(
    correlation_id=correlation_id,
    action_type="DEPLOYMENT_SENT",
    payload={
        "deployment_id": deployment_id,
        "target_host": target_host,
        "service_name": service_name,
        "status": "sent",
        "timestamp": datetime.now(UTC).isoformat(),
    },
)
```

**Recommended Events:**
- `DEPLOYMENT_INITIATED` - When deployment send starts
- `DEPLOYMENT_SENT` - When deployment package sent successfully
- `DEPLOYMENT_SEND_FAILED` - When send operation fails

### 3. database_adapter_effect (HIGH - 0.5h)

**Current State:**
- Base class: `NodeEffect, HealthCheckMixin`
- EventBus: ❌ Missing
- Pattern: Should use Intent-based (Effect node)

**Refactoring Template:**
```python
# Use intent-based pattern for effect node
from ....models.intent import ModelIntent, EnumIntentType

def _generate_database_event_intent(
    self,
    event_type: str,
    data: dict,
    correlation_id: str,
) -> ModelIntent:
    """Generate intent for database operation event."""
    return ModelIntent(
        intent_type=EnumIntentType.PUBLISH_EVENT.value,
        target="event_bus",
        payload={
            "event_type": event_type,
            "topic": "database_operations",
            "data": data,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        priority=0,
    )

# In execute_effect method
intent = self._generate_database_event_intent(
    event_type="DATABASE_WRITE",
    data={
        "table": table_name,
        "operation": "insert",
        "rows_affected": rows_affected,
    },
    correlation_id=correlation_id,
)
# Append to result intents for orchestrator to execute
```

**Recommended Events:**
- `DATABASE_WRITE` - When writing to database
- `DATABASE_READ` - When reading from database
- `DATABASE_ERROR` - When database operation fails

### 4. store_effect (HIGH - 0.5h)

**Current State:**
- Base class: `NodeEffect`
- EventBus: ❌ Missing
- Pattern: Should use Intent-based (Effect node)

**Refactoring Template:**
```python
# Use intent-based pattern (same as database_adapter_effect)
from ....models.intent import ModelIntent, EnumIntentType

def _generate_store_event_intent(
    self,
    event_type: str,
    data: dict,
    correlation_id: str,
) -> ModelIntent:
    """Generate intent for store operation event."""
    return ModelIntent(
        intent_type=EnumIntentType.PUBLISH_EVENT.value,
        target="event_bus",
        payload={
            "event_type": event_type,
            "topic": "store_operations",
            "data": data,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        priority=0,
    )

# In execute_effect method
intent = self._generate_store_event_intent(
    event_type="STORE_WRITE",
    data={
        "key": key,
        "operation": "put",
        "size_bytes": len(value),
    },
    correlation_id=correlation_id,
)
```

**Recommended Events:**
- `STORE_WRITE` - When writing to store
- `STORE_READ` - When reading from store
- `STORE_DELETE` - When deleting from store

### 5. registry (MEDIUM - 0.5h)

**Current State:**
- Base class: `TypedDict` (INCORRECT - architectural issue)
- EventBus: ❌ Missing
- Issue: Not a proper ONEX node class

**Issues Found:**
```python
# Current incorrect implementation
class RegistrationMetrics(TypedDict):
    # ...

class MemoryGrowthMetrics(TypedDict):
    # ...

class KafkaMessageOffsets(TypedDict):
    # ...
```

**Recommendation:**
This node requires **architectural review** before EventBus integration:
1. Determine proper base class (NodeEffect, NodeReducer, or NodeOrchestrator)
2. Refactor from TypedDict to proper ONEX node class
3. Add EventBus integration after base class is fixed

**Recommended Events (after architectural fix):**
- `NODE_REGISTERED` - When node registers with registry
- `NODE_UNREGISTERED` - When node unregisters
- `NODE_HEARTBEAT` - Periodic health signal from nodes

### 6. codegen_orchestrator (MEDIUM - 0.5h)

**Current State:**
- Base class: `NodeOrchestrator`
- EventBus: ❌ Missing
- Pattern: Should use Direct EventBus (Orchestrator node)

**Refactoring Template:**
```python
# In __init__
from ....services.event_bus import EventBusService

self.event_bus = container.get_service("event_bus")
if self.event_bus is None:
    self.event_bus = EventBusService(
        kafka_client=self.kafka_client,
        node_id=self.node_id,
        namespace=self.default_namespace,
    )
    container.register_service("event_bus", self.event_bus)

# Usage in orchestration methods
await self.event_bus.publish_action_event(
    correlation_id=correlation_id,
    action_type="CODEGEN_STARTED",
    payload={
        "template": template_name,
        "target": target_path,
        "generator": generator_type,
    },
)
```

**Recommended Events:**
- `CODEGEN_STARTED` - When code generation starts
- `CODEGEN_COMPLETED` - When code generation completes successfully
- `CODEGEN_FAILED` - When code generation fails

### 7. codegen_metrics_reducer (LOW - 0.5h)

**Current State:**
- Base class: `NodeReducer, MixinIntentPublisher`
- EventBus: ❌ Missing (but uses intent pattern via mixin)
- Pattern: Already uses intent-based approach via MixinIntentPublisher

**Assessment:**
This node is **already mostly compliant** through the use of `MixinIntentPublisher`. The mixin provides intent-based event publishing capabilities.

**Verification Steps:**
1. Verify MixinIntentPublisher is properly configured
2. Ensure all metric events are published via intents
3. No major refactoring needed

**Note:** This is the **lowest priority** refactoring as it already follows ONEX v2.0 patterns.

## Reference Implementations

### Compliant Node: orchestrator

**File:** `src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py`

**Key Implementation Details:**
```python
# Import
from ....services.event_bus import EventBusService

# Initialization
self.event_bus = container.get_service("event_bus")
if self.event_bus is None and not health_check_mode:
    if self.kafka_client:
        self.event_bus = EventBusService(
            kafka_client=self.kafka_client,
            node_id=self.node_id,
            namespace=self.default_namespace,
        )
        container.register_service("event_bus", self.event_bus)

# Usage
success = await self.event_bus.publish_action_event(
    correlation_id=workflow_id,
    action_type="ORCHESTRATE_WORKFLOW",
    payload={
        "operation": "metadata_stamping",
        "input_data": contract.input_data,
        "namespace": self.default_namespace,
        "workflow_type": "metadata_stamping",
    },
)
```

### Compliant Node: reducer

**File:** `src/omninode_bridge/nodes/reducer/v1_0_0/node.py`

**Key Implementation Details:**
```python
# Intent generation
intent = ModelIntent(
    intent_type=EnumIntentType.PUBLISH_EVENT.value,
    target="event_bus",
    payload={
        "event_type": event_type.value,
        "topic": topic_name,
        "data": data,
        "correlation_id": correlation_id,
        "timestamp": datetime.now(UTC).isoformat(),
    },
    priority=0,
)
self._pending_event_intents.append(intent)
```

## Effort Breakdown

| Node | Category | Pattern | Hours | Issues |
|------|----------|---------|-------|--------|
| deployment_receiver_effect | CRITICAL | Direct | 2.2 | Direct Kafka + 2 TODOs |
| deployment_sender_effect | CRITICAL | Direct | 1.3 | Direct Kafka + 1 TODO |
| database_adapter_effect | HIGH | Intent | 0.5 | Missing EventBus |
| store_effect | HIGH | Intent | 0.5 | Missing EventBus |
| registry | MEDIUM | TBD | 0.5 | Missing EventBus + Architecture |
| codegen_orchestrator | MEDIUM | Direct | 0.5 | Missing EventBus |
| codegen_metrics_reducer | LOW | Intent | 0.5 | Minor verification |
| **TOTAL** | | | **6.0** | **7 nodes** |

## Next Steps

1. **Immediate (This Week)**
   - Review this audit report with team
   - Prioritize Phase 1 (deployment nodes) for immediate refactoring
   - Set up development branch for EventBus compliance work

2. **Week 1: Critical Nodes**
   - Refactor deployment_receiver_effect
   - Refactor deployment_sender_effect
   - Update tests and verify functionality

3. **Week 2: Infrastructure Nodes**
   - Refactor database_adapter_effect
   - Refactor store_effect
   - Update tests and verify functionality

4. **Week 3: Registry Review**
   - Conduct architectural review of registry node
   - Refactor registry base class
   - Add EventBus integration

5. **Week 4: Codegen Nodes**
   - Refactor codegen_orchestrator
   - Verify codegen_metrics_reducer
   - Final testing and documentation

## References

- **EventBus Service:** `src/omninode_bridge/services/event_bus.py`
- **Intent Models:** `src/omninode_bridge/models/intent/model_intent.py`
- **Compliant Orchestrator:** `src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py`
- **Compliant Reducer:** `src/omninode_bridge/nodes/reducer/v1_0_0/node.py`
- **ONEX v2.0 Spec:** `/Users/jonah/Code/Archon/docs/ONEX_ARCHITECTURE_PATTERNS_COMPLETE.md`

## Appendix: Audit Commands

For reproducibility, the following commands were used to generate this audit:

```bash
# Discover all nodes
find src/omninode_bridge/nodes -type d -name "v*_*_*" | sort

# Count nodes
find src/omninode_bridge/nodes -type d -name "v*_*_*" | wc -l

# Audit script (see /tmp/audit_nodes.sh)
# Compliance matrix (see /tmp/generate_compliance_report.py)
# Effort estimation (see /tmp/estimate_effort.py)
```

---

**End of Audit Report**
