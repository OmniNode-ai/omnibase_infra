# Test Generation Event Schemas Implementation

**Status**: ✅ Complete
**Date**: October 30, 2025
**Correlation ID**: 16c2acae-d574-4530-ae26-ecc50e69a589

## Overview

This document describes the implementation of Kafka event schemas for test generation observability in NodeTestGeneratorEffect.

## Implementation Summary

### Files Modified

1. **`src/omninode_bridge/events/codegen_schemas.py`**
   - Added 3 new event models
   - Added 3 new topic constants
   - Updated module docstring

### Event Schemas Defined

#### 1. ModelEventTestGenerationStarted

**Purpose**: Published when test generation begins

**Fields**:
- `correlation_id: UUID` - Correlation ID for tracing
- `workflow_id: UUID | None` - Parent workflow ID (optional)
- `timestamp: datetime` - Event timestamp (auto-generated)
- `test_contract_name: str` - Name of test contract
- `test_types: list[str]` - Types of tests to generate
- `node_name: str` - Name of node being tested
- `output_directory: str` - Where tests will be written
- `schema_version: str` - Event schema version (default: "1.0")

**Kafka Topic**: `dev.omninode-bridge.test-generation.started.v1`

#### 2. ModelEventTestGenerationCompleted

**Purpose**: Published when test generation succeeds

**Fields**:
- `correlation_id: UUID` - Correlation ID for tracing
- `workflow_id: UUID | None` - Parent workflow ID (optional)
- `timestamp: datetime` - Event timestamp (auto-generated)
- `generated_files: list[str]` - List of generated test file paths
- `file_count: int` - Total number of test files generated
- `duration_seconds: float` - Total generation duration
- `quality_score: float` - Quality score (0.0-1.0)
- `test_coverage_estimate: float | None` - Estimated coverage % (optional)
- `metadata: dict[str, Any]` - Additional metadata
- `schema_version: str` - Event schema version (default: "1.0")

**Kafka Topic**: `dev.omninode-bridge.test-generation.completed.v1`

#### 3. ModelEventTestGenerationFailed

**Purpose**: Published when test generation fails

**Fields**:
- `correlation_id: UUID` - Correlation ID for tracing
- `workflow_id: UUID | None` - Parent workflow ID (optional)
- `timestamp: datetime` - Event timestamp (auto-generated)
- `error_code: str` - Error code for classification
- `error_message: str` - Human-readable error message
- `stack_trace: str | None` - Full stack trace (optional)
- `failed_test_type: str | None` - Specific test type that failed (optional)
- `partial_files_generated: list[str]` - Files generated before failure
- `metadata: dict[str, Any]` - Additional context
- `schema_version: str` - Event schema version (default: "1.0")

**Kafka Topic**: `dev.omninode-bridge.test-generation.failed.v1`

### Topic Constants Defined

```python
TOPIC_TEST_GENERATION_STARTED = "dev.omninode-bridge.test-generation.started.v1"
TOPIC_TEST_GENERATION_COMPLETED = "dev.omninode-bridge.test-generation.completed.v1"
TOPIC_TEST_GENERATION_FAILED = "dev.omninode-bridge.test-generation.failed.v1"
```

## Usage Example

### Import the Schemas

```python
from uuid import uuid4
from src.omninode_bridge.events.codegen_schemas import (
    ModelEventTestGenerationStarted,
    ModelEventTestGenerationCompleted,
    ModelEventTestGenerationFailed,
    TOPIC_TEST_GENERATION_STARTED,
    TOPIC_TEST_GENERATION_COMPLETED,
    TOPIC_TEST_GENERATION_FAILED,
)
```

### Publish Started Event

```python
started_event = ModelEventTestGenerationStarted(
    correlation_id=uuid4(),
    workflow_id=uuid4(),
    test_contract_name="postgres_crud_effect_tests",
    test_types=["unit", "integration", "contract"],
    node_name="NodePostgresCrudEffect",
    output_directory="./generated_nodes/postgres_crud_effect/tests",
)

await kafka_publisher.publish(
    topic=TOPIC_TEST_GENERATION_STARTED,
    event=started_event
)
```

### Publish Completed Event

```python
completed_event = ModelEventTestGenerationCompleted(
    correlation_id=correlation_id,
    workflow_id=workflow_id,
    generated_files=[
        "./generated_nodes/postgres_crud_effect/tests/test_unit.py",
        "./generated_nodes/postgres_crud_effect/tests/test_integration.py",
    ],
    file_count=2,
    duration_seconds=45.2,
    quality_score=0.92,
    test_coverage_estimate=85.5,
    metadata={
        "unit_tests": 12,
        "integration_tests": 8,
        "total_assertions": 47,
    },
)

await kafka_publisher.publish(
    topic=TOPIC_TEST_GENERATION_COMPLETED,
    event=completed_event
)
```

### Publish Failed Event

```python
import traceback

try:
    # Test generation logic
    pass
except Exception as e:
    failed_event = ModelEventTestGenerationFailed(
        correlation_id=correlation_id,
        workflow_id=workflow_id,
        error_code=type(e).__name__,
        error_message=str(e),
        stack_trace=traceback.format_exc(),
        failed_test_type="unit",
        metadata={
            "node_name": "NodePostgresCrudEffect",
            "test_contract_name": "postgres_crud_effect_tests",
        },
    )

    await kafka_publisher.publish(
        topic=TOPIC_TEST_GENERATION_FAILED,
        event=failed_event
    )
```

## Integration with NodeTestGeneratorEffect

The test generator node should publish events at these lifecycle stages:

```python
async def execute_effect(
    self, contract: ModelContractTestGeneration
) -> ModelResult:
    correlation_id = contract.correlation_id
    start_time = time.time()

    # 1. Publish started event
    await self._publish_started_event(contract, correlation_id)

    try:
        # 2. Generate tests
        result = await self._generate_tests(contract)

        # 3. Publish completed event
        duration = time.time() - start_time
        await self._publish_completed_event(
            correlation_id=correlation_id,
            result=result,
            duration_seconds=duration,
        )

        return result

    except Exception as e:
        # 4. Publish failed event
        await self._publish_failed_event(
            correlation_id=correlation_id,
            error=e,
        )
        raise
```

## Validation Tests

All event schemas have been validated:

```bash
poetry run python -c "
from uuid import UUID
from src.omninode_bridge.events.codegen_schemas import (
    ModelEventTestGenerationStarted,
    ModelEventTestGenerationCompleted,
    ModelEventTestGenerationFailed
)

# Test instantiation with valid data
started = ModelEventTestGenerationStarted(...)
completed = ModelEventTestGenerationCompleted(...)
failed = ModelEventTestGenerationFailed(...)
"
```

**Result**: ✅ All schemas validated successfully

## Next Steps

1. **Create Kafka Topics**
   - See [Test Generation Kafka Topics](../test_generation_kafka_topics.md)
   - Run topic creation script

2. **Implement Event Publishing in NodeTestGeneratorEffect**
   - Add Kafka publisher dependency injection
   - Implement event publishing at lifecycle stages
   - Add error handling for publish failures

3. **Add Monitoring**
   - Set up Kafka consumer for events
   - Create dashboards for test generation metrics
   - Configure alerting for failed events

4. **Testing**
   - Write unit tests for event schema validation
   - Integration tests for event publishing
   - End-to-end tests for complete workflow

## Design Decisions

### Why Three Separate Events?

Instead of a single "TestGenerationEvent" with status field, we use three separate event types:

1. **Clear Intent**: Each event type has a specific purpose
2. **Type Safety**: Different fields for different states (completed has files, failed has errors)
3. **Kafka Best Practice**: Separate topics enable different retention policies
4. **Consumer Optimization**: Consumers can subscribe only to events they need

### Why OnexEnvelopeV1 Format?

Following the existing codegen event patterns ensures:

1. **Consistency**: All events in the system use the same format
2. **Tooling Reuse**: Existing Kafka consumers and monitoring work unchanged
3. **Evolution Support**: Envelope format supports schema versioning
4. **Interoperability**: Events can be consumed by any OnexEnvelopeV1-aware system

### Why Optional Fields?

Fields like `workflow_id`, `test_coverage_estimate`, and `stack_trace` are optional to support:

1. **Flexibility**: Not all test generation is part of a workflow
2. **Graceful Degradation**: Missing optional data doesn't prevent event publication
3. **Evolution**: Easy to add fields without breaking existing consumers

## Compliance Checklist

- ✅ All event models extend EventBase
- ✅ Follows OnexEnvelopeV1 format conventions
- ✅ Includes correlation_id for tracing
- ✅ Timestamps auto-generated with UTC timezone
- ✅ Schema version field included (default: "1.0")
- ✅ Comprehensive docstrings with flow documentation
- ✅ Type hints on all fields
- ✅ Validation constraints (ge, le) where appropriate
- ✅ Example payloads in model_config
- ✅ Topic constants follow naming convention
- ✅ Compatible with existing codegen event patterns

## References

- **Event Schemas**: [src/omninode_bridge/events/codegen_schemas.py](/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/events/codegen_schemas.py)
- **Kafka Topics**: [docs/test_generation_kafka_topics.md](../test_generation_kafka_topics.md)
- **Existing Codegen Events**: [src/omninode_bridge/events/models/codegen_events.py](/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/events/models/codegen_events.py)
- **EventBase**: [src/omninode_bridge/events/models/base.py](/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/events/models/base.py)
