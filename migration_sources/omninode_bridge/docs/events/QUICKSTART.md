# Event Infrastructure Quickstart Guide

**Get up and running with the code generation event infrastructure in 15 minutes.**

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup (5 minutes)](#quick-setup-5-minutes)
3. [Publishing Events](#publishing-events)
4. [Consuming Events](#consuming-events)
5. [Monitoring Events](#monitoring-events)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

---

## Prerequisites

Before starting, ensure you have:

### Required Software

- **Docker Desktop** (or Docker Engine + Docker Compose)
  - Version: 20.10+ recommended
  - Download: https://www.docker.com/products/docker-desktop

- **Python 3.11+**
  - Check version: `python --version`
  - Download: https://www.python.org/downloads/

- **Poetry** (Python package manager)
  - Install: `curl -sSL https://install.python-poetry.org | python3 -`
  - Verify: `poetry --version`

### System Requirements

- **RAM**: 2GB minimum (4GB recommended)
- **Disk**: 5GB free space
- **Network**: Internet access for pulling Docker images

### Network Configuration

Redpanda requires hostname resolution for broker discovery. Add this entry to `/etc/hosts` **(one-time setup)**:

```bash
# Add Redpanda hostname to /etc/hosts
echo "127.0.0.1 omninode-bridge-redpanda" | sudo tee -a /etc/hosts
```

This is Kafka/Redpanda-specific due to its two-step broker discovery protocol.

---

## Quick Setup (5 minutes)

### Step 1: Clone Repository

```bash
# Clone omninode_bridge repository
git clone https://github.com/OmniNode-ai/omninode_bridge.git
cd omninode_bridge
```

### Step 2: Install Dependencies

```bash
# Install Python dependencies with Poetry
poetry install

# Verify installation
poetry run python --version
poetry run python -c "from omninode_bridge.events.codegen_schemas import CodegenAnalysisRequest; print('Schemas loaded successfully!')"
```

### Step 3: Start Redpanda Cluster

```bash
# Create Docker network (if not exists)
docker network create omninode-bridge-network 2>/dev/null || true

# Start Redpanda and create topics
docker-compose -f docker-compose.codegen.yml up -d

# Wait for Redpanda to be ready (30 seconds)
echo "Waiting for Redpanda to start..."
sleep 30

# Verify topics were created
docker exec omninode-bridge-redpanda rpk topic list | grep omninode_codegen
```

**Expected Output**:
```
omninode_codegen_dlq_analyze_v1
omninode_codegen_dlq_mixin_v1
omninode_codegen_dlq_pattern_v1
omninode_codegen_dlq_validate_v1
omninode_codegen_request_analyze_v1
omninode_codegen_request_mixin_v1
omninode_codegen_request_pattern_v1
omninode_codegen_request_validate_v1
omninode_codegen_response_analyze_v1
omninode_codegen_response_mixin_v1
omninode_codegen_response_pattern_v1
omninode_codegen_response_validate_v1
omninode_codegen_status_session_v1
```

### Step 4: Verify Infrastructure

```bash
# Check Redpanda health
docker exec omninode-bridge-redpanda rpk cluster health

# Check topic details
docker exec omninode-bridge-redpanda rpk topic describe omninode_codegen_request_analyze_v1
```

**Expected Output** (cluster health):
```
CLUSTER HEALTH OVERVIEW
=======================
Healthy:                     true
Controller ID:               0
All replicas:                13
Leader replicas:             13
Out of sync replicas:        0
Under-replicated replicas:   0
```

**Setup Complete!** You now have a working event infrastructure.

---

## Publishing Events

### Example 1: Publish Analysis Request

Create a file `examples/publish_analysis_request.py`:

```python
"""Example: Publishing a code generation analysis request."""

import asyncio
import json
from datetime import datetime
from uuid import uuid4

from aiokafka import AIOKafkaProducer

from omninode_bridge.events.codegen_schemas import CodegenAnalysisRequest


async def publish_analysis_request():
    """Publish PRD analysis request to Kafka."""

    # Create Kafka producer
    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:19092",
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        compression_type="gzip",
    )

    await producer.start()

    try:
        # Create analysis request
        request = CodegenAnalysisRequest(
            correlation_id=uuid4(),
            session_id=uuid4(),
            prd_content="""
# User Authentication Service

## Requirements
- User registration with email/password
- JWT token generation
- Password hashing with bcrypt
- Token refresh mechanism

## Architecture
- Node type: Effect
- Database: PostgreSQL
- Cache: Redis (optional)
            """,
            analysis_type="full",
            workspace_context={
                "project_path": "/Users/dev/myproject",
                "language": "python",
                "framework": "onex",
            },
        )

        # Publish to topic
        topic = "omninode_codegen_request_analyze_v1"
        await producer.send_and_wait(
            topic, value=request.model_dump(mode="json")
        )

        print(f"✅ Published analysis request to {topic}")
        print(f"   Correlation ID: {request.correlation_id}")
        print(f"   Session ID: {request.session_id}")
        print(f"   Analysis type: {request.analysis_type}")

    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(publish_analysis_request())
```

**Run the example**:

```bash
# Run the publisher
poetry run python examples/publish_analysis_request.py

# Verify message was published
docker exec omninode-bridge-redpanda rpk topic consume omninode_codegen_request_analyze_v1 --num 1
```

### Example 2: Publish Validation Request

Create a file `examples/publish_validation_request.py`:

```python
"""Example: Publishing a code validation request."""

import asyncio
import json
from uuid import uuid4

from aiokafka import AIOKafkaProducer

from omninode_bridge.events.codegen_schemas import CodegenValidationRequest


async def publish_validation_request():
    """Publish code validation request to Kafka."""

    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:19092",
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        compression_type="gzip",
    )

    await producer.start()

    try:
        # Create validation request
        request = CodegenValidationRequest(
            correlation_id=uuid4(),
            session_id=uuid4(),
            code_content='''
class NodeUserAuthEffect(NodeEffect):
    """Handle user authentication operations."""

    async def execute_effect(self, contract: ModelContractEffect) -> ModelResult:
        """Execute authentication effect."""
        async with self.transaction_manager.begin():
            # Validate credentials
            user = await self._validate_user(contract.input_data)

            # Generate JWT token
            token = self._generate_jwt(user)

            return ModelResult(success=True, data={"token": token})
            ''',
            node_type="effect",
            contracts=[
                {
                    "contract_type": "effect",
                    "name": "UserAuthContract",
                    "version": "1.0",
                }
            ],
            validation_type="full",
        )

        # Publish to topic
        topic = "omninode_codegen_request_validate_v1"
        await producer.send_and_wait(topic, value=request.model_dump(mode="json"))

        print(f"✅ Published validation request to {topic}")
        print(f"   Correlation ID: {request.correlation_id}")
        print(f"   Node type: {request.node_type}")
        print(f"   Validation type: {request.validation_type}")

    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(publish_validation_request())
```

**Run the example**:

```bash
poetry run python examples/publish_validation_request.py
```

### Example 3: Publish Status Update

Create a file `examples/publish_status_update.py`:

```python
"""Example: Publishing session status updates."""

import asyncio
import json
from uuid import UUID

from aiokafka import AIOKafkaProducer

from omninode_bridge.events.codegen_schemas import CodegenStatusEvent


async def publish_status_updates(session_id: UUID):
    """Publish real-time status updates for a session."""

    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:19092",
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        compression_type="gzip",
    )

    await producer.start()

    try:
        # Status 1: Analyzing
        status1 = CodegenStatusEvent(
            session_id=session_id,
            status="analyzing",
            progress_percentage=25.0,
            message="Analyzing PRD for node requirements",
            metadata={
                "current_step": "prd_analysis",
                "steps_completed": ["initialization"],
                "steps_remaining": ["code_generation", "validation", "finalization"],
            },
        )

        # Status 2: Generating
        status2 = CodegenStatusEvent(
            session_id=session_id,
            status="generating",
            progress_percentage=50.0,
            message="Generating node implementation",
            metadata={
                "current_step": "code_generation",
                "steps_completed": ["initialization", "prd_analysis"],
                "steps_remaining": ["validation", "finalization"],
            },
        )

        # Status 3: Validating
        status3 = CodegenStatusEvent(
            session_id=session_id,
            status="validating",
            progress_percentage=75.0,
            message="Validating generated code",
            metadata={
                "current_step": "validation",
                "steps_completed": ["initialization", "prd_analysis", "code_generation"],
                "steps_remaining": ["finalization"],
            },
        )

        # Status 4: Completed
        status4 = CodegenStatusEvent(
            session_id=session_id,
            status="completed",
            progress_percentage=100.0,
            message="Code generation completed successfully",
            metadata={
                "current_step": "finalization",
                "steps_completed": [
                    "initialization",
                    "prd_analysis",
                    "code_generation",
                    "validation",
                    "finalization",
                ],
                "steps_remaining": [],
            },
        )

        topic = "omninode_codegen_status_session_v1"

        # Publish status updates with delays
        for i, status in enumerate([status1, status2, status3, status4], 1):
            await producer.send_and_wait(topic, value=status.model_dump(mode="json"))
            print(
                f"✅ Published status {i}/4: {status.status} ({status.progress_percentage}%)"
            )
            await asyncio.sleep(2)  # Simulate processing time

    finally:
        await producer.stop()


if __name__ == "__main__":
    from uuid import uuid4

    session_id = uuid4()
    print(f"Session ID: {session_id}")
    asyncio.run(publish_status_updates(session_id))
```

**Run the example**:

```bash
poetry run python examples/publish_status_update.py
```

---

## Consuming Events

### Example 4: Consume Analysis Requests

Create a file `examples/consume_analysis_requests.py`:

```python
"""Example: Consuming analysis requests (omniarchon consumer)."""

import asyncio
import json

from aiokafka import AIOKafkaConsumer

from omninode_bridge.events.codegen_schemas import CodegenAnalysisRequest


async def consume_analysis_requests():
    """Consume and process analysis requests."""

    consumer = AIOKafkaConsumer(
        "omninode_codegen_request_analyze_v1",
        bootstrap_servers="localhost:19092",
        group_id="example_analysis_consumer",
        auto_offset_reset="earliest",  # Start from beginning for demo
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    await consumer.start()

    print("🔄 Consuming analysis requests... (Press Ctrl+C to stop)")
    print()

    try:
        async for msg in consumer:
            # Deserialize to Pydantic model
            request = CodegenAnalysisRequest(**msg.value)

            # Process request
            print(f"📥 Received analysis request:")
            print(f"   Correlation ID: {request.correlation_id}")
            print(f"   Session ID: {request.session_id}")
            print(f"   Analysis type: {request.analysis_type}")
            print(f"   PRD length: {len(request.prd_content)} characters")
            print(f"   Workspace: {request.workspace_context.get('project_path', 'N/A')}")
            print()

            # Here you would:
            # 1. Perform semantic analysis on PRD
            # 2. Extract requirements and architecture
            # 3. Publish CodegenAnalysisResponse to response topic

    except KeyboardInterrupt:
        print("\n⏹️  Stopped consuming")
    finally:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(consume_analysis_requests())
```

**Run the consumer** (in a separate terminal):

```bash
# Terminal 1: Start consumer
poetry run python examples/consume_analysis_requests.py

# Terminal 2: Publish a request
poetry run python examples/publish_analysis_request.py
```

### Example 5: Consume Response Events

Create a file `examples/consume_responses.py`:

```python
"""Example: Consuming all response events (omniclaude consumer)."""

import asyncio
import json

from aiokafka import AIOKafkaConsumer

from omninode_bridge.events.codegen_schemas import (
    CodegenAnalysisResponse,
    CodegenValidationResponse,
    CodegenPatternResponse,
    CodegenMixinResponse,
)


async def consume_responses():
    """Consume all response types."""

    consumer = AIOKafkaConsumer(
        "omninode_codegen_response_analyze_v1",
        "omninode_codegen_response_validate_v1",
        "omninode_codegen_response_pattern_v1",
        "omninode_codegen_response_mixin_v1",
        bootstrap_servers="localhost:19092",
        group_id="example_response_consumer",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    await consumer.start()

    print("🔄 Consuming response events... (Press Ctrl+C to stop)")
    print()

    try:
        async for msg in consumer:
            # Route based on topic
            if "analyze" in msg.topic:
                response = CodegenAnalysisResponse(**msg.value)
                print(f"📥 Analysis Response:")
                print(f"   Correlation ID: {response.correlation_id}")
                print(f"   Confidence: {response.confidence:.2%}")
                print(f"   Processing time: {response.processing_time_ms}ms")

            elif "validate" in msg.topic:
                response = CodegenValidationResponse(**msg.value)
                print(f"📥 Validation Response:")
                print(f"   Correlation ID: {response.correlation_id}")
                print(f"   Is valid: {response.is_valid}")
                print(f"   Quality score: {response.quality_score:.2%}")
                print(f"   ONEX compliance: {response.onex_compliance_score:.2%}")

            elif "pattern" in msg.topic:
                response = CodegenPatternResponse(**msg.value)
                print(f"📥 Pattern Response:")
                print(f"   Correlation ID: {response.correlation_id}")
                print(f"   Total matches: {response.total_matches}")
                print(f"   Results: {len(response.pattern_result)} nodes")

            elif "mixin" in msg.topic:
                response = CodegenMixinResponse(**msg.value)
                print(f"📥 Mixin Response:")
                print(f"   Correlation ID: {response.correlation_id}")
                print(f"   Recommendations: {response.total_recommendations} mixins")

            print()

    except KeyboardInterrupt:
        print("\n⏹️  Stopped consuming")
    finally:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(consume_responses())
```

**Run the consumer**:

```bash
poetry run python examples/consume_responses.py
```

---

## Monitoring Events

### Example 6: Monitor DLQ Topics

Create a file `examples/monitor_dlq.py`:

```python
"""Example: Monitoring dead letter queue topics."""

import asyncio

from omninode_bridge.monitoring.codegen_dlq_monitor import CodegenDLQMonitor


async def monitor_dlq():
    """Monitor DLQ topics for failed events."""

    # Initialize DLQ monitor
    monitor = CodegenDLQMonitor(
        kafka_config={"bootstrap_servers": "localhost:19092"},
        alert_threshold=5,  # Alert after 5 DLQ messages
    )

    print("🔍 Starting DLQ monitoring...")
    print("   Alert threshold: 5 messages per topic")
    print("   Monitoring topics:")
    for topic in monitor.DLQ_TOPICS:
        print(f"     - {topic}")
    print()

    try:
        # Start monitoring (runs continuously)
        await monitor.start_monitoring()

    except KeyboardInterrupt:
        print("\n⏹️  Stopping DLQ monitoring...")
        await monitor.stop_monitoring()

        # Print final statistics
        stats = await monitor.get_dlq_stats()
        print("\n📊 Final DLQ Statistics:")
        print(f"   Total DLQ messages: {stats['total_dlq_messages']}")
        print("   Per-topic counts:")
        for topic, count in stats["dlq_counts"].items():
            print(f"     - {topic}: {count}")


if __name__ == "__main__":
    asyncio.run(monitor_dlq())
```

**Run the monitor**:

```bash
poetry run python examples/monitor_dlq.py
```

### Example 7: Trace Session Events

Create a file `examples/trace_session.py`:

```python
"""Example: Tracing session events for debugging."""

import asyncio
from uuid import UUID

from omninode_bridge.dashboard.codegen_event_tracer import CodegenEventTracer
from omninode_bridge.infrastructure.postgres_connection_manager import (
    PostgresConnectionManager,
    ModelPostgresConfig,
)


async def trace_session(session_id: UUID):
    """Trace all events for a code generation session."""

    # Initialize database connection
    config = ModelPostgresConfig.from_environment()
    db_manager = PostgresConnectionManager(config)
    await db_manager.initialize()

    # Create event tracer
    tracer = CodegenEventTracer(db_manager)

    print(f"🔍 Tracing session: {session_id}")
    print()

    # Trace session events
    trace = await tracer.trace_session_events(session_id, time_range_hours=24)

    print(f"📊 Session Trace Results:")
    print(f"   Total events: {trace['total_events']}")
    print(f"   Session duration: {trace['session_duration_ms']}ms")
    print(f"   Status: {trace['status']}")
    print()

    if trace["events"]:
        print("📋 Event Timeline:")
        for event in trace["events"]:
            print(f"   [{event['timestamp']}] {event['event_type']}")
            print(f"      Topic: {event['topic']}")
            print(f"      Status: {event['status']}")
            if "processing_time_ms" in event:
                print(f"      Processing time: {event['processing_time_ms']}ms")
            print()

    # Get performance metrics
    metrics = await tracer.get_session_metrics(session_id)

    print(f"⚡ Performance Metrics:")
    print(f"   Success rate: {metrics['success_rate'] * 100:.1f}%")
    print(f"   Average response time: {metrics['avg_response_time_ms']}ms")
    print(f"   P95 response time: {metrics['p95_response_time_ms']}ms")

    # Cleanup
    await db_manager.close()


if __name__ == "__main__":
    # Example session ID - replace with actual session ID
    session_id = UUID("123e4567-e89b-12d3-a456-426614174000")
    asyncio.run(trace_session(session_id))
```

**Run the tracer**:

```bash
# Set database environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5436
export POSTGRES_DATABASE=omninode_bridge
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your-password

# Run tracer
poetry run python examples/trace_session.py
```

---

## Troubleshooting

### Common Issues

#### 1. Topics Not Created

**Symptom**: `rpk topic list` shows no codegen topics

**Solution**:
```bash
# Manually create topics
docker-compose -f docker-compose.codegen.yml up topic-creator

# Verify topics
docker exec omninode-bridge-redpanda rpk topic list | grep omninode_codegen
```

#### 2. Connection Refused

**Symptom**: `KafkaConnectionError: Connection refused`

**Solution**:
```bash
# Check Redpanda is running
docker ps | grep redpanda

# Check Redpanda logs
docker logs omninode-bridge-redpanda

# Verify port mapping
docker port omninode-bridge-redpanda 19092
# Should show: 19092/tcp -> 0.0.0.0:19092

# Test connection
telnet localhost 19092
```

#### 3. Hostname Resolution Failure

**Symptom**: `KafkaConnectionError: Node not ready`

**Solution**:
```bash
# Verify /etc/hosts entry
grep omninode-bridge-redpanda /etc/hosts
# Should show: 127.0.0.1 omninode-bridge-redpanda

# If missing, add it:
echo "127.0.0.1 omninode-bridge-redpanda" | sudo tee -a /etc/hosts
```

#### 4. Consumer Lag

**Symptom**: Consumer not receiving messages

**Solution**:
```bash
# Check consumer group lag
docker exec omninode-bridge-redpanda rpk group describe example_analysis_consumer

# Reset consumer offsets to earliest
docker exec omninode-bridge-redpanda rpk group seek example_analysis_consumer --to start

# Check topic for messages
docker exec omninode-bridge-redpanda rpk topic consume omninode_codegen_request_analyze_v1 --num 5
```

#### 5. Schema Validation Errors

**Symptom**: `ValidationError: 1 validation error for CodegenAnalysisRequest`

**Solution**:
```python
# Validate your data before publishing
from omninode_bridge.events.codegen_schemas import CodegenAnalysisRequest

try:
    request = CodegenAnalysisRequest(
        correlation_id="not-a-uuid",  # This will fail
        session_id=uuid4(),
        prd_content="Test"
    )
except ValueError as e:
    print(f"Validation error: {e}")
    # Fix the data and retry
```

#### 6. DLQ Messages Accumulating

**Symptom**: High DLQ message count

**Solution**:
```bash
# Check DLQ topics
docker exec omninode-bridge-redpanda rpk topic consume omninode_codegen_dlq_analyze_v1 --num 10

# Investigate error patterns in DLQ messages
# Fix root cause in consumer code
# Optionally reset DLQ topic
docker exec omninode-bridge-redpanda rpk topic delete omninode_codegen_dlq_analyze_v1
docker exec omninode-bridge-redpanda rpk topic create omninode_codegen_dlq_analyze_v1 --partitions 1 --replicas 1
```

### Health Checks

```bash
# 1. Check Redpanda cluster health
docker exec omninode-bridge-redpanda rpk cluster health

# 2. Check topic status
docker exec omninode-bridge-redpanda rpk topic list

# 3. Check consumer groups
docker exec omninode-bridge-redpanda rpk group list

# 4. Check partition distribution
docker exec omninode-bridge-redpanda rpk topic describe omninode_codegen_request_analyze_v1

# 5. Monitor message throughput
docker exec omninode-bridge-redpanda rpk topic produce omninode_codegen_request_analyze_v1 --num 100
docker exec omninode-bridge-redpanda rpk topic consume omninode_codegen_request_analyze_v1 --num 100
```

### Useful Commands

```bash
# List all topics
docker exec omninode-bridge-redpanda rpk topic list

# Describe a topic
docker exec omninode-bridge-redpanda rpk topic describe omninode_codegen_request_analyze_v1

# Produce test message
echo '{"test": "message"}' | docker exec -i omninode-bridge-redpanda rpk topic produce omninode_codegen_request_analyze_v1

# Consume messages
docker exec omninode-bridge-redpanda rpk topic consume omninode_codegen_request_analyze_v1 --num 10

# List consumer groups
docker exec omninode-bridge-redpanda rpk group list

# Describe consumer group
docker exec omninode-bridge-redpanda rpk group describe example_analysis_consumer

# Delete consumer group
docker exec omninode-bridge-redpanda rpk group delete example_analysis_consumer

# Delete topic
docker exec omninode-bridge-redpanda rpk topic delete omninode_codegen_request_analyze_v1

# View Redpanda logs
docker logs omninode-bridge-redpanda --tail 100 -f
```

---

## Next Steps

### 1. Explore Advanced Features

- **Schema Evolution**: Learn about backward-compatible schema changes
- **DLQ Handling**: Implement retry logic for failed events
- **Event Tracing**: Use correlation IDs to trace request/response chains
- **Performance Tuning**: Optimize producer/consumer configurations

### 2. Production Readiness

- **Enable TLS/SSL**: Secure Kafka connections
- **Add Authentication**: Implement SASL/SCRAM authentication
- **Configure ACLs**: Restrict topic access by service
- **Monitoring**: Set up Prometheus metrics and alerting
- **Multi-Broker Cluster**: Deploy Redpanda with replication

### 3. Integration Examples

- **omniclaude Integration**: Connect AI code generator
- **omniarchon Integration**: Implement intelligence processing
- **Event Store**: Persist events for audit trail
- **Stream Processing**: Use Kafka Streams for real-time analytics

### 4. Documentation

- **[EVENT_SYSTEM_GUIDE.md](./EVENT_SYSTEM_GUIDE.md)**: Comprehensive event system guide with infrastructure and patterns
- **[codegen-topics-config.yaml](./codegen-topics-config.yaml)**: Topic configuration reference
- **[codegen_schemas.py](../../src/omninode_bridge/events/codegen_schemas.py)**: Event schema definitions
- **[MVP Plan](../../!!!MVP_PLAN_EVENT_INFRASTRUCTURE!!!.md)**: Original implementation plan

---

**Questions or Issues?**
- Check the [Troubleshooting](#troubleshooting) section
- Review the [EVENT_SYSTEM_GUIDE.md](./EVENT_SYSTEM_GUIDE.md)
- Open an issue on GitHub

**Happy Eventing!** 🎉
