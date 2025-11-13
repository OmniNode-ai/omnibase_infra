# Kafka Event Schema Compliance Guide

**Version**: 1.0.0
**Status**: Active
**Last Updated**: October 29, 2025
**Purpose**: Unified Kafka schema compliance guide covering standardization, validation, and best practices

---

## Executive Summary

This guide provides a comprehensive framework for achieving 100% OnexEnvelopeV1 compliance across all Kafka topics in the omninode_bridge repository. It combines:

- **Standardization Framework**: OnexEnvelopeV1 format, migration patterns, and producer/consumer patterns
- **Validation Framework**: Schema validation at boundaries with comprehensive testing
- **Compliance Workflow**: Best practices, monitoring, and operational procedures

**Target State**:
- ✅ All events wrapped in ModelOnexEnvelopeV1
- ✅ Consistent schema versioning across all topics
- ✅ Standardized producer/consumer patterns
- ✅ Schema validation at all boundaries
- ✅ Comprehensive testing (unit, integration, performance)
- ✅ Observable validation metrics and DLQ routing

---

## Table of Contents

### Part 1: Standardization Framework
1. [OnexEnvelopeV1 Standard Format](#onexenvelopev1-standard-format)
2. [Migration Patterns](#migration-patterns)
3. [Producer Patterns](#producer-patterns)
4. [Consumer Patterns](#consumer-patterns)
5. [Schema Evolution Strategy](#schema-evolution-strategy)

### Part 2: Validation Framework
6. [Producer Validation](#producer-validation)
7. [Consumer Validation](#consumer-validation)
8. [Schema Evolution Testing](#schema-evolution-testing)
9. [Integration Testing](#integration-testing)
10. [Performance Testing](#performance-testing)

### Part 3: Compliance Workflow
11. [Validation Utilities](#validation-utilities)
12. [Best Practices](#best-practices)
13. [Monitoring and Observability](#monitoring-and-observability)
14. [Compliance Checklist](#compliance-checklist)

---

# Part 1: Standardization Framework

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

# Part 2: Validation Framework

## Producer Validation

### Pattern 1: Validation Decorator

**Purpose**: Validate event data before publishing

```python
# src/omninode_bridge/validation/kafka_validators.py

from functools import wraps
from typing import Type, Callable
from pydantic import BaseModel, ValidationError
from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import ModelOnexEnvelopeV1
import logging

logger = logging.getLogger(__name__)

def validate_event_schema(schema: Type[BaseModel], event_type: str):
    """
    Decorator to validate event payload against Pydantic schema before publishing.

    Args:
        schema: Pydantic model class for validation
        event_type: Event type name for envelope

    Returns:
        Decorated async function that validates before publishing

    Example:
        @validate_event_schema(CodegenAnalysisRequest, "CODEGEN_ANALYSIS_REQUEST")
        async def publish_analysis_request(kafka_client, data):
            return await kafka_client.publish_with_envelope(...)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract event data from kwargs or args
            event_data = kwargs.get("event_data") or kwargs.get("payload")

            if not event_data:
                raise ValueError("No event_data or payload provided for validation")

            try:
                # Validate against schema
                validated_event = schema(**event_data)

                # Replace payload with validated data
                kwargs["payload"] = validated_event.model_dump()

                # Execute original function with validated data
                result = await func(*args, **kwargs)

                logger.info(
                    f"Event validated and published successfully",
                    extra={
                        "event_type": event_type,
                        "schema": schema.__name__,
                        "schema_version": validated_event.schema_version
                        if hasattr(validated_event, "schema_version") else "unknown"
                    }
                )

                return result

            except ValidationError as e:
                logger.error(
                    f"Event validation failed",
                    extra={
                        "event_type": event_type,
                        "schema": schema.__name__,
                        "errors": e.errors()
                    }
                )

                # Send to DLQ
                await send_to_dlq(
                    event_data=event_data,
                    error=str(e),
                    event_type=event_type
                )

                raise ValueError(f"Event validation failed: {e}")

        return wrapper
    return decorator


# Usage Example
from omninode_bridge.events.codegen_schemas import CodegenAnalysisRequest

@validate_event_schema(CodegenAnalysisRequest, "CODEGEN_ANALYSIS_REQUEST")
async def publish_analysis_request(kafka_client, event_data, correlation_id):
    """Publish analysis request with automatic validation."""
    return await kafka_client.publish_with_envelope(
        event_type="CODEGEN_ANALYSIS_REQUEST",
        source_node_id="omniclaude-instance-1",
        payload=event_data,  # Already validated by decorator
        correlation_id=correlation_id
    )
```

### Pattern 2: Schema Registry Validator

**Purpose**: Centralized schema validation with registry lookup

```python
# src/omninode_bridge/validation/schema_registry.py

from typing import Type, Dict
from pydantic import BaseModel
from omninode_bridge.events.codegen_schemas import (
    CodegenAnalysisRequest,
    CodegenAnalysisResponse,
    CodegenValidationRequest,
    CodegenValidationResponse,
)


class SchemaRegistry:
    """
    Centralized schema registry for event validation.

    Provides schema lookup by event type and validation utilities.
    """

    _schemas: Dict[str, Type[BaseModel]] = {
        # Codegen Request Schemas
        "CODEGEN_ANALYSIS_REQUEST": CodegenAnalysisRequest,
        "CODEGEN_VALIDATION_REQUEST": CodegenValidationRequest,
        "CODEGEN_PATTERN_REQUEST": CodegenPatternRequest,
        "CODEGEN_MIXIN_REQUEST": CodegenMixinRequest,

        # Codegen Response Schemas
        "CODEGEN_ANALYSIS_RESPONSE": CodegenAnalysisResponse,
        "CODEGEN_VALIDATION_RESPONSE": CodegenValidationResponse,
        "CODEGEN_PATTERN_RESPONSE": CodegenPatternResponse,
        "CODEGEN_MIXIN_RESPONSE": CodegenMixinResponse,

        # Status Schemas
        "CODEGEN_STATUS_EVENT": CodegenStatusEvent,

        # Add bridge event schemas as needed
    }

    @classmethod
    def get_schema(cls, event_type: str) -> Type[BaseModel]:
        """
        Get schema for event type.

        Args:
            event_type: Event type name

        Returns:
            Pydantic model class for validation

        Raises:
            KeyError: If event type not registered
        """
        if event_type not in cls._schemas:
            raise KeyError(f"No schema registered for event type: {event_type}")

        return cls._schemas[event_type]

    @classmethod
    def validate(cls, event_type: str, event_data: dict) -> BaseModel:
        """
        Validate event data against registered schema.

        Args:
            event_type: Event type name
            event_data: Event data to validate

        Returns:
            Validated Pydantic model instance

        Raises:
            ValidationError: If validation fails
        """
        schema = cls.get_schema(event_type)
        return schema(**event_data)

    @classmethod
    def register_schema(cls, event_type: str, schema: Type[BaseModel]):
        """
        Register new schema for event type.

        Args:
            event_type: Event type name
            schema: Pydantic model class
        """
        cls._schemas[event_type] = schema

    @classmethod
    def is_compatible(cls, event_type: str, event_data: dict) -> bool:
        """
        Check if event data is compatible with schema (non-raising).

        Args:
            event_type: Event type name
            event_data: Event data to check

        Returns:
            True if compatible, False otherwise
        """
        try:
            cls.validate(event_type, event_data)
            return True
        except Exception:
            return False


# Usage Example
from omninode_bridge.validation.schema_registry import SchemaRegistry

# Validate before publishing
try:
    validated_event = SchemaRegistry.validate(
        event_type="CODEGEN_ANALYSIS_REQUEST",
        event_data={
            "correlation_id": str(uuid4()),
            "session_id": str(uuid4()),
            "prd_content": "...",
            "analysis_type": "full"
        }
    )

    await kafka_client.publish_with_envelope(
        event_type="CODEGEN_ANALYSIS_REQUEST",
        source_node_id="omniclaude",
        payload=validated_event.model_dump()
    )

except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Handle error
```

---

## Consumer Validation

### Pattern 1: Envelope + Payload Validation

**Purpose**: Validate both envelope and payload at consumer

```python
# src/omninode_bridge/validation/consumer_validators.py

from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import ModelOnexEnvelopeV1
from omninode_bridge.validation.schema_registry import SchemaRegistry
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)


async def consume_with_validation(
    message: bytes,
    expected_event_type: str | None = None
) -> tuple[ModelOnexEnvelopeV1, BaseModel]:
    """
    Consume and validate Kafka message with envelope + payload validation.

    Args:
        message: Raw Kafka message bytes
        expected_event_type: Optional expected event type for validation

    Returns:
        Tuple of (validated_envelope, validated_payload)

    Raises:
        ValidationError: If envelope or payload validation fails
    """
    try:
        # 1. Deserialize envelope
        envelope = ModelOnexEnvelopeV1.from_bytes(message)

        # 2. Verify expected event type (if provided)
        if expected_event_type and envelope.event_type != expected_event_type:
            raise ValueError(
                f"Event type mismatch: expected {expected_event_type}, "
                f"got {envelope.event_type}"
            )

        # 3. Validate payload against schema
        validated_payload = SchemaRegistry.validate(
            event_type=envelope.event_type,
            event_data=envelope.payload
        )

        logger.info(
            f"Message validated successfully",
            extra={
                "event_id": str(envelope.event_id),
                "event_type": envelope.event_type,
                "correlation_id": str(envelope.correlation_id),
                "source_node": envelope.source_node_id
            }
        )

        return envelope, validated_payload

    except ValidationError as e:
        logger.error(
            f"Message validation failed",
            extra={
                "event_type": envelope.event_type if 'envelope' in locals() else "unknown",
                "errors": e.errors()
            }
        )

        # Send to DLQ
        await send_to_dlq(message, error=str(e))

        raise


# Usage Example
from aiokafka import AIOKafkaConsumer

consumer = AIOKafkaConsumer(
    "omninode_codegen_response_analyze_v1",
    bootstrap_servers="localhost:29092"
)

async for message in consumer:
    try:
        # Validate envelope + payload
        envelope, validated_payload = await consume_with_validation(
            message.value,
            expected_event_type="CODEGEN_ANALYSIS_RESPONSE"
        )

        # Process validated payload
        await process_analysis_response(
            payload=validated_payload,
            correlation_id=envelope.correlation_id
        )

    except ValidationError as e:
        # Already sent to DLQ in consume_with_validation
        logger.warning(f"Skipping invalid message: {e}")
```

### Pattern 2: Dual-Format Consumer (Migration Support)

**Purpose**: Support both legacy and envelope formats during migration

```python
async def consume_with_dual_format_support(
    message: bytes,
    legacy_schema: Type[BaseModel],
    envelope_event_type: str
) -> BaseModel:
    """
    Consume message with support for both legacy and envelope formats.

    Args:
        message: Raw Kafka message bytes
        legacy_schema: Pydantic schema for legacy format
        envelope_event_type: Event type for envelope format

    Returns:
        Validated payload (from either format)
    """
    data = json.loads(message.decode())

    # Check format
    if "envelope_version" in data:
        # New envelope format
        envelope = ModelOnexEnvelopeV1.from_dict(data)
        validated_payload = SchemaRegistry.validate(
            event_type=envelope_event_type,
            event_data=envelope.payload
        )
        correlation_id = envelope.correlation_id
    else:
        # Legacy format
        validated_payload = legacy_schema(**data)
        correlation_id = data.get("correlation_id")

    return validated_payload, correlation_id
```

---

## Schema Evolution Testing

### Test 1: Backward Compatibility

**Purpose**: Ensure new schema versions can read old data

```python
# tests/unit/validation/test_schema_evolution.py

import pytest
from uuid import uuid4
from datetime import datetime, UTC
from omninode_bridge.events.codegen_schemas import CodegenAnalysisRequest


def test_backward_compatible_schema_evolution():
    """Test that v1.1 schema can parse v1.0 data."""

    # v1.0 data (missing new optional fields)
    v1_0_data = {
        "correlation_id": str(uuid4()),
        "session_id": str(uuid4()),
        "prd_content": "test content",
        "timestamp": datetime.now(UTC).isoformat(),
        "schema_version": "1.0"
    }

    # Should successfully parse with v1.1 schema
    # (assuming v1.1 added optional fields: analysis_type, workspace_context)
    v1_1_event = CodegenAnalysisRequest(**v1_0_data)

    # Verify defaults applied for new fields
    assert v1_1_event.analysis_type == "full"  # Default value
    assert v1_1_event.workspace_context == {}   # Default value
    assert v1_1_event.schema_version == "1.0"   # Preserved from v1.0


def test_forward_compatible_schema_evolution():
    """Test that v1.0 schema can ignore v1.1 extra fields."""

    # v1.1 data (includes new fields)
    v1_1_data = {
        "correlation_id": str(uuid4()),
        "session_id": str(uuid4()),
        "prd_content": "test content",
        "analysis_type": "quick",      # New in v1.1
        "workspace_context": {},       # New in v1.1
        "timestamp": datetime.now(UTC).isoformat(),
        "schema_version": "1.1"
    }

    # v1.0 schema should ignore extra fields (Pydantic default behavior)
    # Note: This requires Pydantic Config(extra="ignore")
    v1_0_event = CodegenAnalysisRequest(**v1_1_data)

    # Verify core fields parsed
    assert v1_0_event.prd_content == "test content"
    assert v1_0_event.schema_version == "1.1"


def test_breaking_schema_change_detection():
    """Test that breaking changes are detected."""

    # Simulate v2.0 with breaking change (renamed field)
    v2_0_data = {
        "correlation_id": str(uuid4()),
        "session_id": str(uuid4()),
        "prd_markdown": "test content",  # RENAMED from prd_content
        "analysis_type": "full",
        "timestamp": datetime.now(UTC).isoformat(),
        "schema_version": "2.0"
    }

    # Should fail validation with v1.x schema
    with pytest.raises(ValidationError) as exc_info:
        CodegenAnalysisRequest(**v2_0_data)

    # Verify error mentions missing required field
    errors = exc_info.value.errors()
    assert any("prd_content" in str(error) for error in errors)
```

### Test 2: Envelope Versioning

**Purpose**: Ensure envelope version compatibility

```python
def test_envelope_version_compatibility():
    """Test that envelope v1.0 is stable and versioned."""

    from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import ModelOnexEnvelopeV1

    envelope = ModelOnexEnvelopeV1(
        event_type="TEST_EVENT",
        source_node_id="test-node",
        payload={"data": "test"}
    )

    # Verify envelope version
    assert envelope.envelope_version == "1.0"

    # Serialize and deserialize
    envelope_bytes = envelope.to_bytes()
    deserialized = ModelOnexEnvelopeV1.from_bytes(envelope_bytes)

    # Verify version preserved
    assert deserialized.envelope_version == "1.0"
    assert deserialized.event_type == "TEST_EVENT"
```

---

## Integration Testing

### Test 1: End-to-End Event Publishing with Validation

```python
# tests/integration/validation/test_event_publishing_validation.py

import pytest
from uuid import uuid4
from omninode_bridge.services.kafka_client import KafkaClient
from omninode_bridge.validation.schema_registry import SchemaRegistry
from omninode_bridge.events.codegen_schemas import CodegenAnalysisRequest


@pytest.mark.asyncio
async def test_validated_event_publishing():
    """Test end-to-end event publishing with schema validation."""

    # Setup
    kafka_client = KafkaClient()
    await kafka_client.connect()

    correlation_id = uuid4()
    session_id = uuid4()

    # 1. Create event data
    event_data = {
        "correlation_id": str(correlation_id),
        "session_id": str(session_id),
        "prd_content": "# Test PRD\n\nRequirements here...",
        "analysis_type": "full",
        "workspace_context": {},
        "schema_version": "1.0"
    }

    # 2. Validate against schema
    validated_event = SchemaRegistry.validate(
        event_type="CODEGEN_ANALYSIS_REQUEST",
        event_data=event_data
    )

    assert isinstance(validated_event, CodegenAnalysisRequest)

    # 3. Publish with envelope
    success = await kafka_client.publish_with_envelope(
        event_type="CODEGEN_ANALYSIS_REQUEST",
        source_node_id="test-publisher",
        payload=validated_event.model_dump(),
        correlation_id=correlation_id
    )

    assert success

    # 4. Consume and validate
    from aiokafka import AIOKafkaConsumer
    from omninode_bridge.nodes.registry.v1_0_0.models.model_onex_envelope_v1 import ModelOnexEnvelopeV1

    consumer = AIOKafkaConsumer(
        "omninode_codegen_request_analyze_v1",
        bootstrap_servers="localhost:29092",
        auto_offset_reset="latest"
    )
    await consumer.start()

    message = await consumer.getone(timeout=5.0)

    # 5. Deserialize envelope
    envelope = ModelOnexEnvelopeV1.from_bytes(message.value)

    assert envelope.event_type == "CODEGEN_ANALYSIS_REQUEST"
    assert envelope.correlation_id == correlation_id

    # 6. Validate payload
    validated_payload = SchemaRegistry.validate(
        event_type=envelope.event_type,
        event_data=envelope.payload
    )

    assert validated_payload.prd_content == event_data["prd_content"]
    assert validated_payload.correlation_id == correlation_id


@pytest.mark.asyncio
async def test_invalid_event_rejected():
    """Test that invalid events are rejected at producer."""

    kafka_client = KafkaClient()
    await kafka_client.connect()

    # Invalid event data (missing required field)
    invalid_event_data = {
        "correlation_id": str(uuid4()),
        # Missing session_id (required)
        "prd_content": "test",
        "schema_version": "1.0"
    }

    # Should raise ValidationError
    with pytest.raises(ValidationError):
        SchemaRegistry.validate(
            event_type="CODEGEN_ANALYSIS_REQUEST",
            event_data=invalid_event_data
        )
```

### Test 2: Consumer Validation Integration

```python
@pytest.mark.asyncio
async def test_consumer_validation_integration():
    """Test consumer validates messages correctly."""

    from omninode_bridge.validation.consumer_validators import consume_with_validation

    # Setup publisher
    kafka_client = KafkaClient()
    await kafka_client.connect()

    correlation_id = uuid4()

    # Publish valid event
    await kafka_client.publish_with_envelope(
        event_type="CODEGEN_ANALYSIS_REQUEST",
        source_node_id="test-publisher",
        payload={
            "correlation_id": str(correlation_id),
            "session_id": str(uuid4()),
            "prd_content": "test content",
            "schema_version": "1.0"
        }
    )

    # Setup consumer
    consumer = AIOKafkaConsumer(
        "omninode_codegen_request_analyze_v1",
        bootstrap_servers="localhost:29092",
        auto_offset_reset="latest"
    )
    await consumer.start()

    # Consume and validate
    message = await consumer.getone(timeout=5.0)

    envelope, validated_payload = await consume_with_validation(
        message.value,
        expected_event_type="CODEGEN_ANALYSIS_REQUEST"
    )

    # Verify validated
    assert envelope.correlation_id == correlation_id
    assert isinstance(validated_payload, CodegenAnalysisRequest)
```

---

## Performance Testing

### Test 1: Validation Overhead Benchmark

```python
# tests/performance/test_validation_overhead.py

import pytest
import time
from uuid import uuid4
from omninode_bridge.validation.schema_registry import SchemaRegistry


@pytest.mark.performance
def test_schema_validation_performance():
    """Benchmark schema validation overhead."""

    event_data = {
        "correlation_id": str(uuid4()),
        "session_id": str(uuid4()),
        "prd_content": "test content",
        "analysis_type": "full",
        "workspace_context": {},
        "schema_version": "1.0"
    }

    iterations = 10000

    # Benchmark validation
    start = time.perf_counter()
    for _ in range(iterations):
        SchemaRegistry.validate(
            event_type="CODEGEN_ANALYSIS_REQUEST",
            event_data=event_data
        )
    duration = time.perf_counter() - start

    avg_time_ms = (duration / iterations) * 1000

    # Assert validation overhead < 1ms per event
    assert avg_time_ms < 1.0, f"Validation too slow: {avg_time_ms:.2f}ms"

    print(f"Validation performance: {avg_time_ms:.4f}ms per event")


@pytest.mark.performance
@pytest.mark.asyncio
async def test_end_to_end_validation_performance():
    """Benchmark end-to-end publishing with validation."""

    kafka_client = KafkaClient()
    await kafka_client.connect()

    event_data = {
        "correlation_id": str(uuid4()),
        "session_id": str(uuid4()),
        "prd_content": "test content",
        "schema_version": "1.0"
    }

    iterations = 100

    # Benchmark with validation
    start = time.perf_counter()
    for _ in range(iterations):
        validated = SchemaRegistry.validate(
            "CODEGEN_ANALYSIS_REQUEST",
            event_data
        )
        await kafka_client.publish_with_envelope(
            event_type="CODEGEN_ANALYSIS_REQUEST",
            source_node_id="test",
            payload=validated.model_dump()
        )
    duration = time.perf_counter() - start

    avg_time_ms = (duration / iterations) * 1000

    # Assert total overhead < 50ms per event
    assert avg_time_ms < 50.0, f"E2E validation too slow: {avg_time_ms:.2f}ms"

    print(f"E2E validation performance: {avg_time_ms:.2f}ms per event")
```

---

# Part 3: Compliance Workflow

## Validation Utilities

### Utility 1: DLQ Helper

```python
# src/omninode_bridge/validation/dlq_helper.py

from datetime import datetime, UTC
from typing import Any
import logging

logger = logging.getLogger(__name__)


async def send_to_dlq(
    message: bytes | None = None,
    event_data: dict | None = None,
    error: str = "",
    event_type: str = ""
):
    """
    Send failed message to DLQ for investigation.

    Args:
        message: Raw Kafka message bytes (if available)
        event_data: Event data dict (if deserialized)
        error: Error message
        event_type: Event type
    """
    from omninode_bridge.services.kafka_client import KafkaClient

    kafka_client = KafkaClient()

    dlq_payload = {
        "original_message": message.decode() if message else None,
        "event_data": event_data,
        "error": error,
        "event_type": event_type,
        "failed_at": datetime.now(UTC).isoformat()
    }

    dlq_topic = f"omninode_codegen_dlq_{event_type.lower()}_v1"

    try:
        await kafka_client.publish_raw_event(
            topic=dlq_topic,
            event_data=dlq_payload
        )

        logger.info(f"Message sent to DLQ: {dlq_topic}")

    except Exception as e:
        logger.error(f"Failed to send to DLQ: {e}")
```

### Utility 2: Validation Metrics

```python
# src/omninode_bridge/validation/metrics.py

from prometheus_client import Counter, Histogram

# Validation metrics
validation_success_total = Counter(
    "event_validation_success_total",
    "Total successful event validations",
    ["event_type", "validation_stage"]
)

validation_failure_total = Counter(
    "event_validation_failure_total",
    "Total failed event validations",
    ["event_type", "error_type", "validation_stage"]
)

validation_duration_seconds = Histogram(
    "event_validation_duration_seconds",
    "Event validation duration in seconds",
    ["event_type", "validation_stage"]
)


def track_validation_metrics(
    event_type: str,
    validation_stage: str,
    success: bool,
    duration: float,
    error_type: str = ""
):
    """Track validation metrics for monitoring."""

    if success:
        validation_success_total.labels(
            event_type=event_type,
            validation_stage=validation_stage
        ).inc()
    else:
        validation_failure_total.labels(
            event_type=event_type,
            error_type=error_type,
            validation_stage=validation_stage
        ).inc()

    validation_duration_seconds.labels(
        event_type=event_type,
        validation_stage=validation_stage
    ).observe(duration)
```

### Utility 3: Envelope Wrapper Decorator

```python
# src/omninode_bridge/validation/envelope_decorators.py

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

### Utility 4: Bulk Migration Script

```python
# scripts/migrate_topic_to_envelope_format.py

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

### 5. Design for Schema Evolution

**Why**: Enable backward/forward compatibility without service disruption

```python
# Use optional fields with defaults
class EventPayload(BaseModel):
    # Required fields (never remove)
    correlation_id: UUID
    timestamp: datetime

    # Optional fields with defaults (safe to add)
    analysis_type: str = "full"
    workspace_context: dict = Field(default_factory=dict)

    # Deprecated fields (mark but keep for 2 major versions)
    legacy_field: str | None = Field(None, deprecated=True)

    # Version tracking
    schema_version: str = "1.1"
```

### 6. Implement DLQ Routing

**Why**: Preserve failed messages for investigation and replay

```python
async def process_message_with_dlq(message):
    """Process message with automatic DLQ routing on failure."""
    try:
        envelope, payload = await consume_with_validation(message.value)
        await process_event(payload)
    except ValidationError as e:
        # Send to DLQ for investigation
        await send_to_dlq(
            message=message.value,
            error=str(e),
            event_type="VALIDATION_FAILURE"
        )
    except Exception as e:
        # Send processing errors to DLQ
        await send_to_dlq(
            message=message.value,
            error=str(e),
            event_type="PROCESSING_FAILURE"
        )
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
# Track validation success/failure
validation_success_total.labels(
    event_type="WORKFLOW_STARTED",
    validation_stage="producer"
).inc()

validation_failure_total.labels(
    event_type="WORKFLOW_STARTED",
    error_type="missing_required_field",
    validation_stage="consumer"
).inc()

validation_duration_seconds.labels(
    event_type="WORKFLOW_STARTED",
    validation_stage="producer"
).observe(0.002)  # 2ms
```

### Prometheus Alert Rules

```yaml
# Prometheus alert rules for schema compliance
groups:
  - name: schema_compliance_alerts
    interval: 30s
    rules:
      # High validation failure rate
      - alert: HighSchemaValidationFailureRate
        expr: |
          rate(validation_failure_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High schema validation failure rate (>5%)"
          description: "Event type {{ $labels.event_type }} has {{ $value }} validation failures/sec"

      # Envelope publishing failures
      - alert: EnvelopePublishingFailures
        expr: |
          kafka_envelope_success_rate < 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Envelope publishing success rate below 95%"
          description: "Success rate: {{ $value | humanizePercentage }}"

      # DLQ message accumulation
      - alert: DLQMessageAccumulation
        expr: |
          kafka_consumer_lag{topic=~".*_dlq_.*"} > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "DLQ accumulating messages"
          description: "DLQ topic {{ $labels.topic }} has {{ $value }} unprocessed messages"

      # Schema evolution breaking changes
      - alert: SchemaEvolutionBreakingChange
        expr: |
          increase(validation_failure_total{error_type="schema_version_mismatch"}[1h]) > 100
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Potential schema breaking change detected"
          description: "Event type {{ $labels.event_type }} has {{ $value }} version mismatch errors"
```

### Grafana Dashboard Metrics

**Key Metrics to Visualize**:
1. **Validation Success Rate** - By event type, validation stage
2. **Validation Duration** - P50, P95, P99 latencies
3. **DLQ Message Rate** - Messages per second to DLQ
4. **Schema Version Distribution** - Which versions are in use
5. **Envelope Publishing Success Rate** - Overall envelope health
6. **Event Type Distribution** - Which events are most common

---

## Compliance Checklist

### For Each Topic

**Schema Definition**:
- [ ] Schema defined with Pydantic v2 BaseModel
- [ ] Schema includes `schema_version` field
- [ ] Schema registered in SchemaRegistry
- [ ] Schema supports backward compatibility (optional fields with defaults)

**Producer Implementation**:
- [ ] Events wrapped in ModelOnexEnvelopeV1
- [ ] Producers use `publish_with_envelope()` or manual envelope creation
- [ ] Schema validation at producer boundary
- [ ] Validation decorator or SchemaRegistry.validate() implemented
- [ ] DLQ routing configured for validation failures
- [ ] Correlation IDs propagated across related events
- [ ] Partition keys set for ordered delivery (if needed)

**Consumer Implementation**:
- [ ] Consumers deserialize with `ModelOnexEnvelopeV1.from_bytes()`
- [ ] Schema validation at consumer boundary
- [ ] Envelope + payload validation implemented
- [ ] DLQ routing configured for invalid messages
- [ ] Dual-format support (if during migration)

**Testing**:
- [ ] Unit tests for schema validation (producer/consumer)
- [ ] Unit tests for envelope serialization/deserialization
- [ ] Integration tests for end-to-end event flow
- [ ] Schema evolution tests (backward/forward compatibility)
- [ ] Performance tests for validation overhead (<1ms target)
- [ ] Breaking change detection tests

**Documentation**:
- [ ] Event schema documented with examples
- [ ] Producer/consumer patterns documented
- [ ] Schema evolution strategy documented
- [ ] Migration plan documented (if applicable)

**Monitoring**:
- [ ] Validation metrics tracked (success/failure/duration)
- [ ] Envelope publishing metrics tracked
- [ ] DLQ monitoring configured
- [ ] Prometheus alerts configured
- [ ] Grafana dashboard created

### For Schema Evolution

**Backward Compatibility**:
- [ ] New fields added as optional with defaults
- [ ] No required fields removed
- [ ] No field types changed
- [ ] No fields renamed
- [ ] Schema version incremented
- [ ] Backward compatibility tests passing

**Forward Compatibility**:
- [ ] Old schemas ignore new fields (Config(extra="ignore"))
- [ ] Forward compatibility tests passing

**Breaking Changes**:
- [ ] New v2 topic created (if breaking change required)
- [ ] Dual-publishing to v1 and v2 during migration
- [ ] Consumer migration plan documented
- [ ] Deprecation timeline defined (30-day minimum)
- [ ] Breaking change tests passing

---

## Summary

This unified compliance guide provides:

✅ **Standardization Framework**: OnexEnvelopeV1 format, migration patterns, producer/consumer patterns
✅ **Validation Framework**: Schema validation at boundaries with decorators, schema registry, and comprehensive testing
✅ **Schema Evolution**: Backward/forward compatibility testing and migration strategies
✅ **Best Practices**: Correlation IDs, metadata, partitioning, boundary validation, DLQ routing
✅ **Monitoring**: Prometheus metrics, alerts, Grafana dashboards
✅ **Compliance Checklist**: Topic-by-topic and schema evolution validation

**Performance Targets**:
- Schema validation overhead: <1ms per event
- End-to-end validation: <50ms per event
- Envelope publishing success rate: >95%
- Schema validation success rate: >95%

**Next Steps**:
1. Implement validation for all 37+ Kafka topics
2. Run comprehensive testing (unit, integration, performance)
3. Deploy validation framework to production
4. Monitor metrics and tune thresholds
5. Document lessons learned and optimize patterns

---

**Maintained By**: OmniNode Bridge Team
**Last Updated**: October 29, 2025
**Version**: 1.0.0
**Status**: Active
