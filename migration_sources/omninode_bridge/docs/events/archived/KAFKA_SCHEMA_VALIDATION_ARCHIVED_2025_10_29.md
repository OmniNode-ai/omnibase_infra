> **ARCHIVED**: This document was consolidated into [KAFKA_SCHEMA_COMPLIANCE.md](../KAFKA_SCHEMA_COMPLIANCE.md) on October 29, 2025.
> See: [docs/meta/EVENT_DOCS_CONSOLIDATION_ANALYSIS_2025_10.md](../../meta/EVENT_DOCS_CONSOLIDATION_ANALYSIS_2025_10.md) for details.

---

# Kafka Event Schema Validation Framework

**Date**: October 2025
**Status**: Phase 4 Complete - Validation Framework
**Purpose**: Implement schema validation at producer/consumer boundaries with comprehensive testing

---

## Executive Summary

This guide provides a complete validation framework for ensuring schema compliance across all Kafka topics, including:
- **Schema Validation at Boundaries**: Producer and consumer validation patterns
- **Schema Evolution Testing**: Backward/forward compatibility validation
- **Integration Test Framework**: End-to-end event flow testing
- **Performance Testing**: Validation overhead benchmarking

---

## Table of Contents

1. [Producer Validation](#producer-validation)
2. [Consumer Validation](#consumer-validation)
3. [Schema Evolution Tests](#schema-evolution-tests)
4. [Integration Tests](#integration-tests)
5. [Performance Tests](#performance-tests)
6. [Validation Utilities](#validation-utilities)

---

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

## Schema Evolution Tests

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

## Integration Tests

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

## Performance Tests

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

---

## Implementation Checklist

### For Each Topic

- [ ] Schema defined in SchemaRegistry
- [ ] Producer validation decorator/method implemented
- [ ] Consumer validation implemented
- [ ] DLQ routing configured
- [ ] Validation metrics tracking
- [ ] Unit tests for schema validation
- [ ] Integration tests for producer/consumer validation
- [ ] Performance tests for validation overhead
- [ ] Documentation updated

### For Schema Evolution

- [ ] Backward compatibility tests
- [ ] Forward compatibility tests
- [ ] Breaking change detection tests
- [ ] Migration plan documented
- [ ] Dual-format support (if needed)

---

## Summary

This validation framework provides:

✅ **Producer Validation**: Validate before publishing with decorators or SchemaRegistry
✅ **Consumer Validation**: Envelope + payload validation at consumer boundary
✅ **Schema Evolution**: Comprehensive backward/forward compatibility testing
✅ **Integration Testing**: End-to-end validation workflows
✅ **Performance Testing**: Validation overhead benchmarking (<1ms per event)
✅ **DLQ Support**: Failed messages routed to DLQ for investigation
✅ **Metrics Tracking**: Prometheus metrics for monitoring validation success/failure

**Next Steps**:
1. Implement validation for codegen topics
2. Add validation for service lifecycle topics
3. Run performance benchmarks
4. Deploy validation framework to production

---

**Last Updated**: October 2025
**Maintained By**: OmniNode Bridge Team
**Status**: Phase 4 Complete - Validation Framework
