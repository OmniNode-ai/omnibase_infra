> **ARCHIVED**: This document was consolidated into [KAFKA_SCHEMA_COMPLIANCE.md](../KAFKA_SCHEMA_COMPLIANCE.md) on October 29, 2025.
> See: [docs/meta/EVENT_DOCS_CONSOLIDATION_ANALYSIS_2025_10.md](../../meta/EVENT_DOCS_CONSOLIDATION_ANALYSIS_2025_10.md) for details.

---

# Kafka Event Schema Standardization Guide

**Date**: October 2025
**Status**: Phase 2 Complete - Standardization Framework
**Purpose**: Ensure all Kafka events use OnexEnvelopeV1 format with consistent schemas and versioning

---

## Executive Summary

This guide provides standardization patterns, migration utilities, and best practices for achieving 100% OnexEnvelopeV1 compliance across all 37+ Kafka topics in the omninode_bridge repository.

**Target State**:
- ✅ All events wrapped in ModelOnexEnvelopeV1
- ✅ Consistent schema versioning across all topics
- ✅ Standardized producer/consumer patterns
- ✅ Schema validation at all boundaries

---

## OnexEnvelopeV1 Standard Format

### Envelope Structure

All Kafka events MUST use the following standardized envelope format:

```python
from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import ModelOnexEnvelopeV1

envelope = ModelOnexEnvelopeV1(
    # Envelope metadata (required)
    envelope_version="1.0",
    event_id=UUID,                    # Auto-generated
    event_type="EVENT_TYPE_NAME",     # e.g., "WORKFLOW_STARTED"
    event_version="1.0",
    event_timestamp=datetime.now(UTC),

    # Source information (required)
    source_node_id="service-instance-id",
    source_service="omninode-bridge",
    source_version="1.0.0",
    source_instance="container-id",

    # Correlation and tracing (optional but recommended)
    correlation_id=UUID,              # For request/response correlation
    causation_id="parent-event-id",   # For event causality chains

    # Routing information (optional)
    environment="development",
    region="us-west-2",
    partition_key="correlation-id",   # For ordered delivery

    # Event payload (required)
    payload={
        # Your actual event data here
        "workflow_id": "...",
        "status": "...",
        # ... etc
    },

    # Additional metadata (optional)
    metadata={
        "event_category": "workflow_orchestration",
        "node_type": "orchestrator",
        "namespace": "omninode.bridge"
    }
)
```

### Required Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `envelope_version` | str | ✅ | Envelope format version (always "1.0") |
| `event_id` | UUID | ✅ | Unique event identifier (auto-generated) |
| `event_type` | str | ✅ | Event type name (e.g., "WORKFLOW_STARTED") |
| `event_version` | str | ✅ | Event schema version (e.g., "1.0") |
| `event_timestamp` | datetime | ✅ | Event creation timestamp (UTC) |
| `source_node_id` | str | ✅ | Node/service instance that generated event |
| `payload` | dict | ✅ | Actual event data (deserialized event model) |

### Optional but Recommended Fields

| Field | Type | Description | Use Case |
|-------|------|-------------|----------|
| `correlation_id` | UUID | Request/response correlation ID | Request/response patterns, workflow tracking |
| `causation_id` | str | Parent event ID (event causality) | Event chains, workflow dependencies |
| `partition_key` | str | Kafka partition key | Ordered event delivery within partition |
| `metadata` | dict | Additional envelope metadata | Event categorization, filtering |

---

## Migration Patterns

### Pattern 1: Codegen Topics (Direct Payload Migration)

**Current Format** (❌ Non-compliant):
```python
# Current codegen event publishing
event = CodegenAnalysisRequest(
    correlation_id=uuid4(),
    session_id=uuid4(),
    prd_content="...",
    analysis_type="full",
    timestamp=datetime.now(UTC),
    schema_version="1.0"
)

# Publish raw event
producer.produce(topic, value=event.model_dump_json().encode())
```

**Target Format** (✅ Envelope-wrapped):
```python
from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import ModelOnexEnvelopeV1

# Create event payload
event = CodegenAnalysisRequest(
    correlation_id=uuid4(),
    session_id=uuid4(),
    prd_content="...",
    analysis_type="full",
    workspace_context={},
    timestamp=datetime.now(UTC),
    schema_version="1.0"
)

# Wrap in envelope
envelope = ModelOnexEnvelopeV1(
    event_type="CODEGEN_ANALYSIS_REQUEST",
    source_node_id="omniclaude-instance-1",
    payload=event.model_dump(),  # Serialize event to dict
    correlation_id=event.correlation_id,
    metadata={
        "event_category": "codegen_request",
        "analysis_type": event.analysis_type
    }
)

# Publish envelope
producer.produce(topic, value=envelope.to_bytes())
```

**Migration Steps**:
1. ✅ Keep existing event schemas (CodegenAnalysisRequest, etc.)
2. ✅ Wrap payloads in ModelOnexEnvelopeV1
3. ✅ Update producers to use `envelope.to_bytes()`
4. ✅ Update consumers to use `ModelOnexEnvelopeV1.from_bytes()` then `payload`

### Pattern 2: Service Lifecycle Topics (BaseEvent Refactoring)

**Current Format** (❌ Non-compliant):
```python
# Current service lifecycle event
event = ServiceLifecycleEvent(
    type=EventType.SERVICE_LIFECYCLE,
    event=ServiceEventType.STARTUP,
    service="metadata-stamping",
    timestamp=datetime.now(UTC),
    payload={
        "service_version": "1.0.0",
        "environment": "development"
    }
)

# Publish raw event
producer.produce(topic, value=event.model_dump_json().encode())
```

**Target Format** (✅ Envelope-wrapped):
```python
# Create payload from existing event fields
payload = {
    "service": "metadata-stamping",
    "service_version": "1.0.0",
    "environment": "development",
    "health_status": "healthy"
}

# Wrap in envelope
envelope = ModelOnexEnvelopeV1(
    event_type="SERVICE_STARTUP",
    source_node_id="metadata-stamping-instance-1",
    source_service="metadata-stamping",
    source_version="1.0.0",
    payload=payload,
    metadata={
        "event_category": "service_lifecycle",
        "event_subtype": "startup"
    }
)

# Publish envelope
producer.produce(topic, value=envelope.to_bytes())
```

**Migration Steps**:
1. ✅ Deprecate BaseEvent direct usage
2. ✅ Extract payload fields from BaseEvent
3. ✅ Wrap in ModelOnexEnvelopeV1
4. ✅ Update all service lifecycle publishers
5. ✅ Maintain backward compatibility during migration

---

## Producer Patterns

### Pattern 1: Using KafkaClient.publish_with_envelope()

**Best Practice** (✅ Recommended):
```python
from omninode_bridge.services.kafka_client import KafkaClient

kafka_client = KafkaClient()
await kafka_client.connect()

# Publish with automatic envelope wrapping
success = await kafka_client.publish_with_envelope(
    event_type="WORKFLOW_STARTED",
    source_node_id="orchestrator-instance-1",
    payload={
        "workflow_id": str(workflow_id),
        "status": "processing",
        "timestamp": datetime.now(UTC).isoformat()
    },
    topic="dev.omninode_bridge.onex.evt.stamp-workflow-started.v1",
    correlation_id=workflow_id,
    metadata={
        "event_category": "workflow_orchestration",
        "node_type": "orchestrator"
    }
)

if not success:
    logger.warning("Failed to publish event")
```

**Advantages**:
- ✅ Automatic envelope wrapping
- ✅ Built-in retry logic and DLQ routing
- ✅ Performance metrics tracking
- ✅ Correlation ID propagation
- ✅ Partitioning strategy handled automatically

### Pattern 2: Manual Envelope Creation

**When to Use**: Custom envelope configuration, advanced use cases

```python
from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import ModelOnexEnvelopeV1

# Create custom envelope
envelope = ModelOnexEnvelopeV1.create(
    event_type="CUSTOM_EVENT",
    source_node_id="custom-service",
    payload={"custom": "data"},
    correlation_id=uuid4(),
    metadata={"priority": "high"}
)

# Publish manually
topic = envelope.to_kafka_topic(service_prefix="omninode-bridge")
partition_key = envelope.get_kafka_key()

await kafka_client.publish_raw_event(
    topic=topic,
    event_data=envelope.to_dict(),
    partition_key=partition_key
)
```

---

## Consumer Patterns

### Pattern 1: Envelope Deserialization

**Best Practice** (✅ Recommended):
```python
from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import ModelOnexEnvelopeV1
from aiokafka import AIOKafkaConsumer

consumer = AIOKafkaConsumer(
    "dev.omninode_bridge.onex.evt.*",
    bootstrap_servers="localhost:29092"
)

async for message in consumer:
    try:
        # Deserialize envelope
        envelope = ModelOnexEnvelopeV1.from_bytes(message.value)

        # Extract payload
        payload = envelope.payload

        # Process event based on type
        if envelope.event_type == "WORKFLOW_STARTED":
            await process_workflow_started(payload, envelope.correlation_id)
        elif envelope.event_type == "WORKFLOW_COMPLETED":
            await process_workflow_completed(payload, envelope.correlation_id)

        # Track correlation for debugging
        logger.info(
            f"Processed {envelope.event_type}",
            extra={
                "event_id": str(envelope.event_id),
                "correlation_id": str(envelope.correlation_id),
                "source_node": envelope.source_node_id
            }
        )

    except Exception as e:
        logger.error(f"Failed to process message: {e}")
```

### Pattern 2: Dual-Format Support (Migration Period)

**Use During Migration**:
```python
async for message in consumer:
    try:
        data = json.loads(message.value.decode())

        # Check if message uses envelope format
        if "envelope_version" in data:
            # New envelope format
            envelope = ModelOnexEnvelopeV1.from_dict(data)
            payload = envelope.payload
            correlation_id = envelope.correlation_id
        else:
            # Legacy format (backward compatibility)
            payload = data
            correlation_id = data.get("correlation_id")

        # Process payload (same logic for both formats)
        await process_event(payload, correlation_id)

    except Exception as e:
        logger.error(f"Failed to process message: {e}")
```

---

## Schema Validation

### Producer-Side Validation

**Validate Before Publishing**:
```python
from pydantic import ValidationError

def validate_and_publish(event_schema, event_data, kafka_client):
    """Validate event against schema before publishing."""
    try:
        # Validate event data against Pydantic schema
        validated_event = event_schema(**event_data)

        # Wrap in envelope
        envelope = ModelOnexEnvelopeV1(
            event_type=validated_event.__class__.__name__.upper(),
            source_node_id="service-id",
            payload=validated_event.model_dump()
        )

        # Publish
        await kafka_client.publish_with_envelope(
            event_type=envelope.event_type,
            source_node_id=envelope.source_node_id,
            payload=envelope.payload,
            correlation_id=validated_event.correlation_id
        )

        return True

    except ValidationError as e:
        logger.error(f"Event validation failed: {e}")
        # Send to DLQ or handle error
        return False
```

### Consumer-Side Validation

**Validate After Deserialization**:
```python
from pydantic import ValidationError

async def consume_with_validation(message, expected_schema):
    """Consume and validate event against expected schema."""
    try:
        # Deserialize envelope
        envelope = ModelOnexEnvelopeV1.from_bytes(message.value)

        # Validate payload against expected schema
        validated_payload = expected_schema(**envelope.payload)

        # Process validated event
        return await process_event(validated_payload, envelope.correlation_id)

    except ValidationError as e:
        logger.error(f"Payload validation failed: {e}")
        # Send to DLQ
        await send_to_dlq(message, error=str(e))
    except Exception as e:
        logger.error(f"Envelope deserialization failed: {e}")
        # Send to DLQ
        await send_to_dlq(message, error=str(e))
```

---

## Schema Evolution Strategy

### Backward-Compatible Changes

**✅ Allowed**:
- Adding new optional fields to payload
- Adding new optional fields to metadata
- Adding new event types
- Deprecating fields (keep for 2 major versions)

**Example**:
```python
# v1.0 schema
class CodegenAnalysisRequest(BaseModel):
    correlation_id: UUID
    session_id: UUID
    prd_content: str
    schema_version: str = "1.0"

# v1.1 schema (backward compatible)
class CodegenAnalysisRequest(BaseModel):
    correlation_id: UUID
    session_id: UUID
    prd_content: str
    analysis_type: str = "full"  # New optional field with default
    workspace_context: dict = Field(default_factory=dict)  # New optional field
    schema_version: str = "1.1"  # Updated version
```

### Breaking Changes (Require New Version)

**❌ Requires v2 Topic**:
- Removing required fields
- Changing field types
- Renaming fields
- Changing enum values

**Migration Process**:
1. Create new v2 topic: `omninode_codegen_request_analyze_v2`
2. Publish to both v1 and v2 during migration period
3. Migrate consumers to v2 topic
4. Deprecate v1 topic after 30-day migration window
5. Remove v1 topic after deprecation period

---

## Testing Patterns

### Unit Tests: Envelope Serialization

```python
import pytest
from uuid import uuid4
from datetime import datetime, UTC
from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import ModelOnexEnvelopeV1

def test_envelope_serialization():
    """Test envelope serialization/deserialization."""
    # Create envelope
    envelope = ModelOnexEnvelopeV1(
        event_type="TEST_EVENT",
        source_node_id="test-node",
        payload={"data": "test"},
        correlation_id=uuid4()
    )

    # Serialize
    envelope_bytes = envelope.to_bytes()
    envelope_dict = envelope.to_dict()

    # Deserialize
    deserialized_from_bytes = ModelOnexEnvelopeV1.from_bytes(envelope_bytes)
    deserialized_from_dict = ModelOnexEnvelopeV1.from_dict(envelope_dict)

    # Assertions
    assert deserialized_from_bytes.event_type == "TEST_EVENT"
    assert deserialized_from_dict.payload == {"data": "test"}
    assert deserialized_from_bytes.correlation_id == envelope.correlation_id
```

### Integration Tests: Producer/Consumer

```python
@pytest.mark.asyncio
async def test_event_publishing_with_envelope():
    """Test end-to-end event publishing with envelope."""
    # Setup
    kafka_client = KafkaClient()
    await kafka_client.connect()

    correlation_id = uuid4()

    # Publish event with envelope
    success = await kafka_client.publish_with_envelope(
        event_type="WORKFLOW_STARTED",
        source_node_id="test-orchestrator",
        payload={"workflow_id": str(uuid4()), "status": "processing"},
        correlation_id=correlation_id
    )

    assert success

    # Consume event
    consumer = AIOKafkaConsumer(
        "dev.omninode_bridge.onex.evt.workflow-started.v1",
        bootstrap_servers="localhost:29092",
        auto_offset_reset="earliest"
    )
    await consumer.start()

    message = await consumer.getone()
    envelope = ModelOnexEnvelopeV1.from_bytes(message.value)

    # Verify envelope
    assert envelope.event_type == "WORKFLOW_STARTED"
    assert envelope.correlation_id == correlation_id
    assert envelope.source_node_id == "test-orchestrator"
    assert "workflow_id" in envelope.payload
```

### Schema Evolution Tests

```python
def test_backward_compatible_schema_evolution():
    """Test that v1.1 schema is compatible with v1.0 data."""
    # v1.0 data (missing new optional fields)
    v1_data = {
        "correlation_id": str(uuid4()),
        "session_id": str(uuid4()),
        "prd_content": "test content",
        "timestamp": datetime.now(UTC).isoformat(),
        "schema_version": "1.0"
    }

    # Should successfully parse with v1.1 schema
    v1_1_event = CodegenAnalysisRequest(**v1_data)

    # Verify defaults applied
    assert v1_1_event.analysis_type == "full"
    assert v1_1_event.workspace_context == {}
    assert v1_1_event.schema_version == "1.0"  # Preserved from v1
```

---

## Migration Utilities

### Utility 1: Envelope Wrapper Decorator

```python
from functools import wraps
from typing import Callable
from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import ModelOnexEnvelopeV1

def with_envelope(event_type: str, source_node_id: str):
    """Decorator to automatically wrap event payload in envelope."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute function to get payload
            payload = await func(*args, **kwargs)

            # Wrap in envelope
            envelope = ModelOnexEnvelopeV1(
                event_type=event_type,
                source_node_id=source_node_id,
                payload=payload,
                correlation_id=kwargs.get("correlation_id")
            )

            return envelope
        return wrapper
    return decorator

# Usage
@with_envelope(event_type="ANALYSIS_RESULT", source_node_id="analyzer-1")
async def generate_analysis_result(prd_content: str, correlation_id: UUID):
    """Generate analysis result (automatically wrapped in envelope)."""
    result = await analyze_prd(prd_content)
    return {
        "correlation_id": str(correlation_id),
        "analysis_result": result,
        "confidence": 0.92
    }
```

### Utility 2: Bulk Migration Script

```python
import asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

async def migrate_topic_to_envelope_format(
    source_topic: str,
    target_topic: str,
    event_type: str,
    source_node_id: str
):
    """
    Migrate events from legacy format to envelope format.

    Reads from source_topic, wraps in envelope, publishes to target_topic.
    """
    consumer = AIOKafkaConsumer(
        source_topic,
        bootstrap_servers="localhost:29092",
        auto_offset_reset="earliest"
    )

    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:29092"
    )

    await consumer.start()
    await producer.start()

    migrated_count = 0

    try:
        async for message in consumer:
            # Deserialize legacy event
            legacy_data = json.loads(message.value.decode())

            # Wrap in envelope
            envelope = ModelOnexEnvelopeV1(
                event_type=event_type,
                source_node_id=source_node_id,
                payload=legacy_data,
                correlation_id=UUID(legacy_data.get("correlation_id")) if legacy_data.get("correlation_id") else None
            )

            # Publish to new topic
            await producer.send(
                target_topic,
                value=envelope.to_bytes(),
                key=envelope.get_kafka_key().encode()
            )

            migrated_count += 1

            if migrated_count % 1000 == 0:
                print(f"Migrated {migrated_count} events")

    finally:
        await consumer.stop()
        await producer.stop()

    print(f"Migration complete: {migrated_count} events migrated")

# Usage
asyncio.run(migrate_topic_to_envelope_format(
    source_topic="omninode_codegen_request_analyze_v1",
    target_topic="omninode_codegen_request_analyze_v1_envelope",
    event_type="CODEGEN_ANALYSIS_REQUEST",
    source_node_id="migration-utility"
))
```

---

## Best Practices

### 1. Always Use Correlation IDs

**Why**: Enables request/response correlation and distributed tracing

```python
# Generate correlation ID at workflow start
correlation_id = uuid4()

# Use same correlation ID across all related events
await publish_workflow_started(workflow_id, correlation_id=correlation_id)
await publish_intelligence_requested(workflow_id, correlation_id=correlation_id)
await publish_intelligence_received(workflow_id, correlation_id=correlation_id)
await publish_workflow_completed(workflow_id, correlation_id=correlation_id)
```

### 2. Include Meaningful Metadata

**Why**: Enables event filtering, routing, and analytics

```python
envelope = ModelOnexEnvelopeV1(
    event_type="WORKFLOW_COMPLETED",
    source_node_id="orchestrator-1",
    payload=workflow_result,
    metadata={
        "event_category": "workflow_orchestration",
        "workflow_type": "metadata_stamping",
        "duration_ms": 1250,
        "success": True,
        "environment": "production",
        "version": "1.0.0"
    }
)
```

### 3. Use Partitioning for Ordering

**Why**: Guarantees event ordering within same partition

```python
# Partition by workflow_id for ordered delivery
envelope = ModelOnexEnvelopeV1(
    event_type="WORKFLOW_STATE_TRANSITION",
    source_node_id="orchestrator-1",
    payload=state_data,
    partition_key=str(workflow_id)  # Same workflow always same partition
)
```

### 4. Validate at Boundaries

**Why**: Catch schema errors early, prevent bad data propagation

```python
# Producer: Validate before publishing
try:
    validated_event = EventSchema(**event_data)
    await publish_with_envelope(validated_event)
except ValidationError as e:
    logger.error(f"Invalid event: {e}")
    # Handle error

# Consumer: Validate after deserialization
try:
    envelope = ModelOnexEnvelopeV1.from_bytes(message.value)
    validated_payload = PayloadSchema(**envelope.payload)
    await process_event(validated_payload)
except ValidationError as e:
    logger.error(f"Invalid payload: {e}")
    await send_to_dlq(message)
```

---

## Monitoring and Observability

### Metrics to Track

**Envelope Publishing Metrics**:
```python
# Track from KafkaClient.get_envelope_metrics()
{
  "envelope_publishing": {
    "total_events_published": 1523,
    "total_events_failed": 7,
    "success_rate": 0.9954
  },
  "latency_metrics_ms": {
    "average": 23.45,
    "p95": 45.67,
    "p99": 78.90
  }
}
```

**Schema Validation Metrics**:
```python
schema_validation_failures_total.labels(
    event_type="WORKFLOW_STARTED",
    error_type="missing_required_field"
).inc()

schema_validation_duration_seconds.labels(
    event_type="WORKFLOW_STARTED"
).observe(0.002)  # 2ms
```

### Alerts

```yaml
# Prometheus alert rules
groups:
  - name: schema_compliance_alerts
    interval: 30s
    rules:
      - alert: HighSchemaValidationFailureRate
        expr: |
          rate(schema_validation_failures_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High schema validation failure rate"

      - alert: EnvelopePublishingFailures
        expr: |
          kafka_envelope_success_rate < 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Envelope publishing success rate below 95%"
```

---

## Checklist for OnexEnvelopeV1 Compliance

### For Each Topic

- [ ] Schema defined with Pydantic v2 BaseModel
- [ ] Schema includes `schema_version` field
- [ ] Events wrapped in ModelOnexEnvelopeV1
- [ ] Producers use `publish_with_envelope()` or manual envelope creation
- [ ] Consumers deserialize with `ModelOnexEnvelopeV1.from_bytes()`
- [ ] Correlation IDs propagated across related events
- [ ] Partition keys set for ordered delivery (if needed)
- [ ] Schema validation at producer boundary
- [ ] Schema validation at consumer boundary
- [ ] Integration tests for envelope serialization/deserialization
- [ ] Documentation updated with envelope format
- [ ] Producer/consumer mapping documented

### For Schema Evolution

- [ ] Backward-compatible changes only (or create v2 topic)
- [ ] New fields added as optional with defaults
- [ ] Schema version incremented
- [ ] Migration plan documented
- [ ] Dual-format support during migration (if needed)
- [ ] Schema evolution tests added

---

## Next Steps

1. **Complete Codegen Topics Migration** (Priority 1)
   - Migrate 13 codegen topics to OnexEnvelopeV1
   - Update omniclaude and omniarchon producers/consumers
   - Add schema validation at boundaries

2. **Complete Service Lifecycle Migration** (Priority 1)
   - Migrate 5 service lifecycle topics to OnexEnvelopeV1
   - Refactor BaseEvent to use envelope wrapper
   - Update all service lifecycle publishers

3. **Comprehensive Documentation** (Priority 2)
   - Create SERVICE_LIFECYCLE_EVENTS.md
   - Update all topic documentation with envelope format
   - Document all producer/consumer mappings

4. **Schema Validation Implementation** (Priority 2)
   - Add validation decorators for producers
   - Add validation at consumer boundaries
   - Create schema evolution tests

---

**Last Updated**: October 2025
**Maintained By**: OmniNode Bridge Team
**Status**: Phase 2 Complete - Standardization Framework
