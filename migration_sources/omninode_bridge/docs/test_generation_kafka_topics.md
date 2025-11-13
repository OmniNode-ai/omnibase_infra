# Test Generation Kafka Topic Configuration

## Overview

This document defines the Kafka topic configuration for test generation observability events published by `NodeTestGeneratorEffect`.

## Topics

### 1. omninode_test_generation_started_v1

**Purpose**: Track when test generation workflows begin

**Configuration**:
- **Partitions**: 3 (allows parallel processing of test generation requests)
- **Replication Factor**: 1 (single replica for development, increase for production)
- **Retention**: 7 days (604800000 ms)
- **Cleanup Policy**: delete

**Event Schema**: `ModelEventTestGenerationStarted`

**Key Fields**:
- `correlation_id`: UUID for tracing
- `test_contract_name`: Name of test contract
- `test_types`: Types of tests being generated
- `node_name`: Name of node being tested
- `output_directory`: Where tests will be written

### 2. omninode_test_generation_completed_v1

**Purpose**: Track successful test generation completions with metrics

**Configuration**:
- **Partitions**: 3
- **Replication Factor**: 1
- **Retention**: 7 days (604800000 ms)
- **Cleanup Policy**: delete

**Event Schema**: `ModelEventTestGenerationCompleted`

**Key Fields**:
- `correlation_id`: UUID for tracing
- `generated_files`: List of generated test files
- `file_count`: Total number of files generated
- `duration_seconds`: How long generation took
- `quality_score`: Quality score (0.0-1.0)
- `test_coverage_estimate`: Estimated coverage percentage

### 3. omninode_test_generation_failed_v1

**Purpose**: Track test generation failures for debugging and alerting

**Configuration**:
- **Partitions**: 3
- **Replication Factor**: 1
- **Retention**: 30 days (2592000000 ms) - longer retention for debugging
- **Cleanup Policy**: delete

**Event Schema**: `ModelEventTestGenerationFailed`

**Key Fields**:
- `correlation_id`: UUID for tracing
- `error_code`: Classification code (e.g., TEMPLATE_NOT_FOUND)
- `error_message`: Human-readable error
- `stack_trace`: Full stack trace for debugging
- `failed_test_type`: Specific test type that failed
- `partial_files_generated`: Files generated before failure

## Topic Creation Script

Add the following to your Kafka topic creation script (e.g., `scripts/create_test_generation_kafka_topics.py`):

```python
from confluent_kafka.admin import AdminClient, NewTopic

KAFKA_TOPICS_TEST_GENERATION = [
    {
        "name": "dev.omninode-bridge.test-generation.started.v1",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": "604800000",  # 7 days
            "cleanup.policy": "delete",
            "compression.type": "snappy",
        },
    },
    {
        "name": "dev.omninode-bridge.test-generation.completed.v1",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": "604800000",  # 7 days
            "cleanup.policy": "delete",
            "compression.type": "snappy",
        },
    },
    {
        "name": "dev.omninode-bridge.test-generation.failed.v1",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": "2592000000",  # 30 days (longer for debugging)
            "cleanup.policy": "delete",
            "compression.type": "snappy",
        },
    },
]

def create_test_generation_topics(bootstrap_servers: str = "192.168.86.200:9092"):
    """Create test generation Kafka topics."""
    admin_client = AdminClient({"bootstrap.servers": bootstrap_servers})

    new_topics = [
        NewTopic(
            topic["name"],
            num_partitions=topic["partitions"],
            replication_factor=topic["replication_factor"],
            config=topic["config"],
        )
        for topic in KAFKA_TOPICS_TEST_GENERATION
    ]

    fs = admin_client.create_topics(new_topics)

    for topic, f in fs.items():
        try:
            f.result()
            print(f"✅ Topic {topic} created successfully")
        except Exception as e:
            print(f"❌ Failed to create topic {topic}: {e}")

if __name__ == "__main__":
    create_test_generation_topics()
```

## Production Recommendations

For production environments, consider these adjustments:

1. **Replication Factor**: Increase to 3 for high availability
2. **Partitions**: Scale based on throughput requirements
3. **Retention**: Adjust based on compliance and debugging needs
4. **Compression**: Use `snappy` or `lz4` for better performance
5. **Min In-Sync Replicas**: Set to 2 for durability

## Monitoring

Monitor these metrics for each topic:

- **Message Rate**: Messages/second being published
- **Consumer Lag**: Delay in processing events
- **Error Rate**: Failed events per topic
- **Partition Distribution**: Balanced message distribution

## Integration with NodeTestGeneratorEffect

The test generator node will publish events at these lifecycle stages:

1. **Started**: Immediately when `execute_effect()` begins
2. **Completed**: After all test files successfully generated
3. **Failed**: On any error during generation

Example integration:

```python
async def execute_effect(self, contract: ModelContractTestGeneration) -> ModelResult:
    # Publish started event
    await self._publish_event(
        TOPIC_TEST_GENERATION_STARTED,
        ModelEventTestGenerationStarted(
            correlation_id=contract.correlation_id,
            test_contract_name=contract.contract_name,
            test_types=contract.test_types,
            node_name=contract.node_name,
            output_directory=contract.output_directory,
        )
    )

    try:
        # Generate tests...
        result = await self._generate_tests(contract)

        # Publish completed event
        await self._publish_event(
            TOPIC_TEST_GENERATION_COMPLETED,
            ModelEventTestGenerationCompleted(
                correlation_id=contract.correlation_id,
                generated_files=result.files,
                file_count=len(result.files),
                duration_seconds=result.duration,
                quality_score=result.quality_score,
            )
        )

        return result

    except Exception as e:
        # Publish failed event
        await self._publish_event(
            TOPIC_TEST_GENERATION_FAILED,
            ModelEventTestGenerationFailed(
                correlation_id=contract.correlation_id,
                error_code=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
            )
        )
        raise
```

## Related Documentation

- [Code Generation Event Schemas](../src/omninode_bridge/events/codegen_schemas.py)
- [OnexEnvelopeV1 Format](../src/omninode_bridge/events/models/base.py)
- [Kafka Infrastructure Guide](./KAFKA_INFRASTRUCTURE.md) *(planned)*
