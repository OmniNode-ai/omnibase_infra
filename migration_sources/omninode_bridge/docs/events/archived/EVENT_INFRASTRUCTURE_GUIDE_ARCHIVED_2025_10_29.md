> **ARCHIVED**: This document was consolidated into [EVENT_SYSTEM_GUIDE.md](../EVENT_SYSTEM_GUIDE.md) (v2.0.0) on October 29, 2025.
> See: [docs/meta/EVENT_DOCS_CONSOLIDATION_ANALYSIS_2025_10.md](../../meta/EVENT_DOCS_CONSOLIDATION_ANALYSIS_2025_10.md) for details.

---

# Event Infrastructure Guide - Autonomous Code Generation

**Version**: 1.0
**Last Updated**: October 2025
**Status**: Production-ready Event Infrastructure

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Topic Organization](#topic-organization)
3. [Event Schemas](#event-schemas)
4. [Consumer Group Patterns](#consumer-group-patterns)
5. [Error Handling & DLQ Strategy](#error-handling--dlq-strategy)
6. [Monitoring & Alerting](#monitoring--alerting)
7. [Performance Characteristics](#performance-characteristics)
8. [Best Practices](#best-practices)

---

## Architecture Overview

The event infrastructure enables autonomous code generation workflows between **omniclaude** (AI code generator) and **omniarchon** (intelligence processing service) using Kafka/Redpanda as the event backbone.

### High-Level Flow

```
┌─────────────┐                                    ┌─────────────┐
│ omniclaude  │                                    │ omniarchon  │
│  (Client)   │                                    │ (Intelligence)│
└──────┬──────┘                                    └──────┬──────┘
       │                                                  │
       │ 1. Publish Request                              │
       │───────────────────────────────────────────────► │
       │   Topic: omninode_codegen_request_*_v1          │
       │                                                  │
       │                                    2. Process   │
       │                                    Intelligence │
       │                                                  │
       │ 3. Consume Response                             │
       │ ◄─────────────────────────────────────────────  │
       │   Topic: omninode_codegen_response_*_v1         │
       │                                                  │
       │ 4. Both publish status updates                  │
       │───────────────────────────────────────────────► │
       │   Topic: omninode_codegen_status_session_v1     │
       │                                                  │
       │ 5. Failed events → DLQ                          │
       │───────────────────────────────────────────────► │
       │   Topic: omninode_codegen_dlq_*_v1              │
       │                                                  │
```

### Event Flow Stages

1. **Request Phase**: omniclaude publishes analysis/validation/pattern/mixin requests
2. **Processing Phase**: omniarchon consumes requests, performs intelligence operations
3. **Response Phase**: omniarchon publishes results, omniclaude consumes responses
4. **Status Updates**: Both services publish real-time status updates
5. **Error Handling**: Failed events routed to DLQ topics for investigation

### Key Components

- **Redpanda Cluster**: Kafka-compatible event streaming platform (1GB RAM, 1 CPU)
- **13 Kafka Topics**: 4 request + 4 response + 1 status + 4 DLQ topics
- **2 Consumer Groups**: `omniclaude_codegen_consumer` and `omniarchon_codegen_intelligence`
- **Event Schemas**: 9 Pydantic v2 schemas with versioning support
- **DLQ Monitor**: Threshold-based alerting for failed events
- **Event Tracer**: Correlation-based event tracing for debugging

---

## Topic Organization

### Topic Naming Convention

All topics follow the pattern: `omninode_codegen_{category}_{operation}_v{version}`

- **Prefix**: `omninode_codegen_` (namespace for code generation events)
- **Category**: `request`, `response`, `status`, `dlq` (event category)
- **Operation**: `analyze`, `validate`, `pattern`, `mixin`, `session` (event type)
- **Version**: `v1`, `v2`, etc. (schema versioning)

### Topic Catalog (13 Topics)

#### Request Topics (4 topics)
| Topic Name | Partitions | Retention | Purpose |
|------------|-----------|-----------|---------|
| `omninode_codegen_request_analyze_v1` | 3 | 7 days | PRD analysis requests |
| `omninode_codegen_request_validate_v1` | 3 | 7 days | Code validation requests |
| `omninode_codegen_request_pattern_v1` | 3 | 7 days | Pattern matching requests |
| `omninode_codegen_request_mixin_v1` | 3 | 7 days | Mixin recommendation requests |

**Flow**: omniclaude (producer) → omniarchon (consumer)

#### Response Topics (4 topics)
| Topic Name | Partitions | Retention | Purpose |
|------------|-----------|-----------|---------|
| `omninode_codegen_response_analyze_v1` | 3 | 7 days | PRD analysis results |
| `omninode_codegen_response_validate_v1` | 3 | 7 days | Code validation results |
| `omninode_codegen_response_pattern_v1` | 3 | 7 days | Similar node patterns |
| `omninode_codegen_response_mixin_v1` | 3 | 7 days | Mixin recommendations |

**Flow**: omniarchon (producer) → omniclaude (consumer)

#### Status Topics (1 topic)
| Topic Name | Partitions | Retention | Purpose |
|------------|-----------|-----------|---------|
| `omninode_codegen_status_session_v1` | 6 | 3 days | Real-time session status |

**Flow**: Both services (producers) → Monitoring systems (consumers)

**Partitioning Strategy**: Partition by `session_id` for event ordering within sessions

#### Dead Letter Queue Topics (4 topics)
| Topic Name | Partitions | Retention | Purpose |
|------------|-----------|-----------|---------|
| `omninode_codegen_dlq_analyze_v1` | 1 | 30 days | Failed analysis events |
| `omninode_codegen_dlq_validate_v1` | 1 | 30 days | Failed validation events |
| `omninode_codegen_dlq_pattern_v1` | 1 | 30 days | Failed pattern events |
| `omninode_codegen_dlq_mixin_v1` | 1 | 30 days | Failed mixin events |

**Flow**: Both services (producers) → DLQ Monitor (consumer)

**Retention Policy**: 30 days for forensic analysis and debugging

### Topic Configuration Details

**Common Settings**:
- **Compression**: gzip (reduces network and storage overhead)
- **Cleanup Policy**: delete (time-based retention)
- **Replication Factor**: 1 (development/single-broker deployment)

**Performance Tuning**:
- **Max Message Size**: 1MB per message (configurable via `max_message_bytes`)
- **Producer Batch Size**: 16KB (balances latency and throughput)
- **Producer Linger**: 10ms (allows batching before send)
- **Consumer Fetch Min**: 1KB minimum fetch size
- **Consumer Fetch Max Wait**: 500ms (reduces consumer lag)

---

## Event Schemas

All event schemas are Pydantic v2 models with strict typing and validation. They support schema versioning for backward compatibility.

### Schema Versioning Strategy

- **Current Version**: 1.0
- **Evolution Strategy**: Backward-compatible
- **Version Field**: All schemas include `schema_version` field
- **New Fields**: Add as optional fields to maintain compatibility

### Schema Catalog (9 Schemas)

#### 1. CodegenAnalysisRequest

**Purpose**: Request PRD analysis from omniarchon intelligence
**Topic**: `omninode_codegen_request_analyze_v1`
**Flow**: omniclaude → omniarchon

```python
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import Dict, Any
from pydantic import BaseModel, Field

class CodegenAnalysisRequest(BaseModel):
    correlation_id: UUID = Field(..., description="Request correlation ID for tracing")
    session_id: UUID = Field(..., description="Code generation session ID")
    prd_content: str = Field(..., description="Raw PRD markdown content")
    analysis_type: str = Field(
        default="full",
        description="Type of analysis: 'full', 'requirements', 'architecture'"
    )
    workspace_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workspace context including file paths and metadata"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp (UTC)"
    )
    schema_version: str = Field(default="1.0", description="Event schema version")

# Example Usage
request = CodegenAnalysisRequest(
    correlation_id=uuid4(),
    session_id=uuid4(),
    prd_content="# User Authentication Service\n\n## Requirements\n...",
    analysis_type="full",
    workspace_context={
        "project_path": "/path/to/project",
        "language": "python",
        "framework": "onex"
    }
)
```

#### 2. CodegenAnalysisResponse

**Purpose**: Return PRD analysis results to omniclaude
**Topic**: `omninode_codegen_response_analyze_v1`
**Flow**: omniarchon → omniclaude

```python
class CodegenAnalysisResponse(BaseModel):
    correlation_id: UUID = Field(..., description="Request correlation ID for matching")
    session_id: UUID = Field(..., description="Code generation session ID")
    analysis_result: Dict[str, Any] = Field(
        ...,
        description="Semantic analysis results including requirements, architecture, dependencies"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Analysis confidence score (0.0-1.0)"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp (UTC)"
    )
    schema_version: str = Field(default="1.0", description="Event schema version")

# Example Usage
response = CodegenAnalysisResponse(
    correlation_id=request.correlation_id,  # Match request
    session_id=request.session_id,
    analysis_result={
        "requirements": ["User authentication", "JWT tokens", "Password hashing"],
        "architecture": {
            "node_type": "effect",
            "mixins": ["ValidationMixin", "AuthenticationMixin"],
            "dependencies": ["bcrypt", "pyjwt"]
        },
        "complexity": "medium"
    },
    confidence=0.92,
    processing_time_ms=1250
)
```

#### 3. CodegenValidationRequest

**Purpose**: Request code validation for ONEX compliance
**Topic**: `omninode_codegen_request_validate_v1`
**Flow**: omniclaude → omniarchon

```python
from typing import List

class CodegenValidationRequest(BaseModel):
    correlation_id: UUID = Field(..., description="Request correlation ID for tracing")
    session_id: UUID = Field(..., description="Code generation session ID")
    code_content: str = Field(..., description="Generated code to validate")
    node_type: str = Field(
        ...,
        description="Type of node: 'effect', 'compute', 'reducer', 'orchestrator'"
    )
    contracts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Associated contracts for compliance checking"
    )
    validation_type: str = Field(
        default="full",
        description="Type of validation: 'full', 'syntax', 'compliance', 'performance'"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp (UTC)"
    )
    schema_version: str = Field(default="1.0", description="Event schema version")

# Example Usage
validation_request = CodegenValidationRequest(
    correlation_id=uuid4(),
    session_id=session_id,
    code_content='''
class NodeUserAuthEffect(NodeEffect):
    async def execute_effect(self, contract: ModelContractEffect) -> ModelResult:
        # Implementation...
        pass
    ''',
    node_type="effect",
    contracts=[
        {"contract_type": "effect", "name": "UserAuthContract", "version": "1.0"}
    ],
    validation_type="full"
)
```

#### 4. CodegenValidationResponse

**Purpose**: Return code validation results with quality scores
**Topic**: `omninode_codegen_response_validate_v1`
**Flow**: omniarchon → omniclaude

```python
class CodegenValidationResponse(BaseModel):
    correlation_id: UUID = Field(..., description="Request correlation ID for matching")
    session_id: UUID = Field(..., description="Code generation session ID")
    validation_result: Dict[str, Any] = Field(
        ...,
        description="Validation results including errors, warnings, and suggestions"
    )
    quality_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall code quality score (0.0-1.0)"
    )
    onex_compliance_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="ONEX compliance score (0.0-1.0)"
    )
    is_valid: bool = Field(..., description="Whether code passes all validation checks")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp (UTC)"
    )
    schema_version: str = Field(default="1.0", description="Event schema version")

# Example Usage
validation_response = CodegenValidationResponse(
    correlation_id=validation_request.correlation_id,
    session_id=validation_request.session_id,
    validation_result={
        "errors": [],
        "warnings": ["Consider adding docstring to execute_effect method"],
        "suggestions": ["Use more descriptive variable names"],
        "compliance_checks": {
            "naming_convention": "pass",
            "method_signature": "pass",
            "contract_adherence": "pass",
            "error_handling": "warning"
        }
    },
    quality_score=0.87,
    onex_compliance_score=0.95,
    is_valid=True,
    processing_time_ms=450
)
```

#### 5. CodegenPatternRequest

**Purpose**: Find similar node implementations for reference
**Topic**: `omninode_codegen_request_pattern_v1`
**Flow**: omniclaude → omniarchon

```python
class CodegenPatternRequest(BaseModel):
    correlation_id: UUID = Field(..., description="Request correlation ID for tracing")
    session_id: UUID = Field(..., description="Code generation session ID")
    node_description: str = Field(
        ...,
        description="Description of desired node functionality"
    )
    node_type: str = Field(
        ...,
        description="Type of node to find: 'effect', 'compute', 'reducer', etc."
    )
    limit: int = Field(
        default=5, ge=1, le=20,
        description="Maximum number of similar nodes to return"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp (UTC)"
    )
    schema_version: str = Field(default="1.0", description="Event schema version")

# Example Usage
pattern_request = CodegenPatternRequest(
    correlation_id=uuid4(),
    session_id=session_id,
    node_description="Database write operation with transaction support",
    node_type="effect",
    limit=5
)
```

#### 6. CodegenPatternResponse

**Purpose**: Return similar node patterns with implementation details
**Topic**: `omninode_codegen_response_pattern_v1`
**Flow**: omniarchon → omniclaude

```python
class CodegenPatternResponse(BaseModel):
    correlation_id: UUID = Field(..., description="Request correlation ID for matching")
    session_id: UUID = Field(..., description="Code generation session ID")
    pattern_result: List[Dict[str, Any]] = Field(
        ...,
        description="List of similar nodes with similarity scores and implementation details"
    )
    total_matches: int = Field(
        ...,
        description="Total number of matches found (before limit)"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp (UTC)"
    )
    schema_version: str = Field(default="1.0", description="Event schema version")

# Example Usage
pattern_response = CodegenPatternResponse(
    correlation_id=pattern_request.correlation_id,
    session_id=pattern_request.session_id,
    pattern_result=[
        {
            "node_name": "NodeDatabaseWriterEffect",
            "similarity_score": 0.94,
            "file_path": "src/nodes/database/node_database_writer_effect.py",
            "implementation_summary": "Database write with asyncpg transaction support",
            "key_patterns": ["async context manager", "transaction rollback", "error handling"]
        },
        {
            "node_name": "NodeCacheWriterEffect",
            "similarity_score": 0.87,
            "file_path": "src/nodes/cache/node_cache_writer_effect.py",
            "implementation_summary": "Redis cache write with connection pooling",
            "key_patterns": ["connection pool", "retry logic", "TTL management"]
        }
    ],
    total_matches=12,
    processing_time_ms=850
)
```

#### 7. CodegenMixinRequest

**Purpose**: Get mixin recommendations based on requirements
**Topic**: `omninode_codegen_request_mixin_v1`
**Flow**: omniclaude → omniarchon

```python
class CodegenMixinRequest(BaseModel):
    correlation_id: UUID = Field(..., description="Request correlation ID for tracing")
    session_id: UUID = Field(..., description="Code generation session ID")
    requirements: List[str] = Field(
        ...,
        description="List of functional requirements for the node"
    )
    node_type: str = Field(..., description="Type of node being generated")
    existing_mixins: List[str] = Field(
        default_factory=list,
        description="Mixins already selected (to avoid duplicates)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp (UTC)"
    )
    schema_version: str = Field(default="1.0", description="Event schema version")

# Example Usage
mixin_request = CodegenMixinRequest(
    correlation_id=uuid4(),
    session_id=session_id,
    requirements=[
        "Input validation",
        "Caching support",
        "Retry logic",
        "Metrics tracking"
    ],
    node_type="effect",
    existing_mixins=[]
)
```

#### 8. CodegenMixinResponse

**Purpose**: Return recommended mixins with implementation guidance
**Topic**: `omninode_codegen_response_mixin_v1`
**Flow**: omniarchon → omniclaude

```python
class CodegenMixinResponse(BaseModel):
    correlation_id: UUID = Field(..., description="Request correlation ID for matching")
    session_id: UUID = Field(..., description="Code generation session ID")
    mixin_recommendations: List[Dict[str, Any]] = Field(
        ...,
        description="List of recommended mixins with rationale and implementation guidance"
    )
    total_recommendations: int = Field(
        ...,
        description="Total number of recommendations made"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp (UTC)"
    )
    schema_version: str = Field(default="1.0", description="Event schema version")

# Example Usage
mixin_response = CodegenMixinResponse(
    correlation_id=mixin_request.correlation_id,
    session_id=mixin_request.session_id,
    mixin_recommendations=[
        {
            "mixin_name": "ValidationMixin",
            "confidence": 0.98,
            "rationale": "Provides input validation for 'Input validation' requirement",
            "implementation_notes": "Add validate() method, use pydantic validators",
            "dependencies": ["pydantic"]
        },
        {
            "mixin_name": "CachingMixin",
            "confidence": 0.92,
            "rationale": "Implements caching for 'Caching support' requirement",
            "implementation_notes": "Configure cache TTL, use Redis backend",
            "dependencies": ["redis"]
        },
        {
            "mixin_name": "RetryMixin",
            "confidence": 0.89,
            "rationale": "Adds retry logic for 'Retry logic' requirement",
            "implementation_notes": "Configure max retries, backoff strategy",
            "dependencies": ["tenacity"]
        }
    ],
    total_recommendations=3,
    processing_time_ms=320
)
```

#### 9. CodegenStatusEvent

**Purpose**: Real-time status updates for session monitoring
**Topic**: `omninode_codegen_status_session_v1`
**Flow**: Both services → Monitoring systems

```python
class CodegenStatusEvent(BaseModel):
    session_id: UUID = Field(..., description="Code generation session ID")
    status: str = Field(
        ...,
        description="Current status: 'pending', 'analyzing', 'generating', 'validating', 'completed', 'failed'"
    )
    progress_percentage: float = Field(
        ..., ge=0.0, le=100.0,
        description="Progress percentage (0.0-100.0)"
    )
    message: str = Field(..., description="Human-readable status message")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata including current step, errors, etc."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp (UTC)"
    )
    schema_version: str = Field(default="1.0", description="Event schema version")

# Example Usage
status_event = CodegenStatusEvent(
    session_id=session_id,
    status="analyzing",
    progress_percentage=25.0,
    message="Analyzing PRD for node requirements",
    metadata={
        "current_step": "prd_analysis",
        "steps_completed": ["initialization"],
        "steps_remaining": ["code_generation", "validation", "finalization"]
    }
)
```

---

## Consumer Group Patterns

### Consumer Group Configuration

Consumer groups enable parallel processing, load balancing, and fault tolerance across multiple consumer instances.

#### 1. omniclaude_codegen_consumer

**Purpose**: Consume code generation responses from omniarchon
**Topics**:
- `omninode_codegen_response_analyze_v1`
- `omninode_codegen_response_validate_v1`
- `omninode_codegen_response_pattern_v1`
- `omninode_codegen_response_mixin_v1`

**Configuration**:
```python
consumer_config = {
    "bootstrap_servers": "localhost:19092",
    "group_id": "omniclaude_codegen_consumer",
    "auto_offset_reset": "latest",  # Only consume new messages
    "enable_auto_commit": True,  # Auto-commit offsets
    "session_timeout_ms": 30000,  # 30s session timeout
    "heartbeat_interval_ms": 10000,  # 10s heartbeat
    "max_poll_interval_ms": 300000,  # 5m max poll interval
}
```

**Consumer Pattern**:
```python
import asyncio
from aiokafka import AIOKafkaConsumer
from omninode_bridge.events.codegen_schemas import CodegenAnalysisResponse
import json
from uuid import UUID

async def consume_responses():
    consumer = AIOKafkaConsumer(
        "omninode_codegen_response_analyze_v1",
        "omninode_codegen_response_validate_v1",
        "omninode_codegen_response_pattern_v1",
        "omninode_codegen_response_mixin_v1",
        bootstrap_servers="localhost:19092",
        group_id="omniclaude_codegen_consumer",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )

    await consumer.start()
    try:
        async for msg in consumer:
            # Route based on topic
            if "analyze" in msg.topic:
                response = CodegenAnalysisResponse(**msg.value)
                await handle_analysis_response(response)
            elif "validate" in msg.topic:
                response = CodegenValidationResponse(**msg.value)
                await handle_validation_response(response)
            elif "pattern" in msg.topic:
                response = CodegenPatternResponse(**msg.value)
                await handle_pattern_response(response)
            elif "mixin" in msg.topic:
                response = CodegenMixinResponse(**msg.value)
                await handle_mixin_response(response)
    finally:
        await consumer.stop()

async def handle_analysis_response(response: CodegenAnalysisResponse):
    print(f"Analysis response received for session {response.session_id}")
    print(f"Confidence: {response.confidence:.2%}")
    print(f"Processing time: {response.processing_time_ms}ms")
    # Process analysis results...
```

#### 2. omniarchon_codegen_intelligence

**Purpose**: Consume code generation requests for intelligence processing
**Topics**:
- `omninode_codegen_request_analyze_v1`
- `omninode_codegen_request_validate_v1`
- `omninode_codegen_request_pattern_v1`
- `omninode_codegen_request_mixin_v1`

**Configuration**:
```python
consumer_config = {
    "bootstrap_servers": "localhost:19092",
    "group_id": "omniarchon_codegen_intelligence",
    "auto_offset_reset": "latest",
    "enable_auto_commit": True,
    "session_timeout_ms": 30000,
    "heartbeat_interval_ms": 10000,
    "max_poll_interval_ms": 600000,  # 10m for long intelligence operations
}
```

**Consumer Pattern**:
```python
from omninode_bridge.events.codegen_schemas import (
    CodegenAnalysisRequest,
    CodegenValidationRequest,
    CodegenPatternRequest,
    CodegenMixinRequest
)

async def consume_requests():
    consumer = AIOKafkaConsumer(
        "omninode_codegen_request_analyze_v1",
        "omninode_codegen_request_validate_v1",
        "omninode_codegen_request_pattern_v1",
        "omninode_codegen_request_mixin_v1",
        bootstrap_servers="localhost:19092",
        group_id="omniarchon_codegen_intelligence",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )

    await consumer.start()
    try:
        async for msg in consumer:
            # Route based on topic
            if "analyze" in msg.topic:
                request = CodegenAnalysisRequest(**msg.value)
                await process_analysis_request(request)
            elif "validate" in msg.topic:
                request = CodegenValidationRequest(**msg.value)
                await process_validation_request(request)
            elif "pattern" in msg.topic:
                request = CodegenPatternRequest(**msg.value)
                await process_pattern_request(request)
            elif "mixin" in msg.topic:
                request = CodegenMixinRequest(**msg.value)
                await process_mixin_request(request)
    finally:
        await consumer.stop()

async def process_analysis_request(request: CodegenAnalysisRequest):
    print(f"Processing analysis request {request.correlation_id}")
    # Perform intelligence analysis...
    # Publish response to omninode_codegen_response_analyze_v1
```

### Consumer Group Best Practices

1. **Partition Assignment**: Let Kafka auto-assign partitions for load balancing
2. **Offset Management**: Use auto-commit for simple cases, manual commit for at-least-once semantics
3. **Session Timeout**: Set to 30s for responsive failure detection
4. **Heartbeat Interval**: Set to 10s (1/3 of session timeout)
5. **Max Poll Interval**: Set based on maximum processing time (5-10 minutes)
6. **Graceful Shutdown**: Always call `consumer.stop()` to commit offsets
7. **Error Handling**: Catch exceptions, publish to DLQ, continue processing

---

## Error Handling & DLQ Strategy

### Dead Letter Queue (DLQ) Architecture

When event processing fails, events are routed to Dead Letter Queue (DLQ) topics for investigation and retry.

### DLQ Workflow

```
┌───────────────┐
│ Request Topic │
└───────┬───────┘
        │
        ▼
   ┌─────────┐         Success
   │Consumer │────────────────────►  Process normally
   └────┬────┘
        │
        │ Failure
        ▼
   ┌─────────────┐
   │ Error Check │
   └──────┬──────┘
          │
          │ Retry exhausted
          ▼
   ┌────────────┐
   │  DLQ Topic │  ◄─── Failed event + error context
   └────────────┘
          │
          ▼
   ┌─────────────┐
   │ DLQ Monitor │  ◄─── Alert if threshold exceeded
   └─────────────┘
```

### DLQ Event Format

```python
dlq_event = {
    "original_event": {
        # Original event payload
        "correlation_id": "uuid",
        "session_id": "uuid",
        # ... original fields
    },
    "error": {
        "error_type": "ValidationError",
        "error_message": "Invalid node_type: 'invalid'",
        "stack_trace": "Traceback (most recent call last):\n  ...",
        "timestamp": "2025-10-14T10:30:00Z"
    },
    "retry_count": 3,
    "max_retries": 3,
    "dlq_timestamp": "2025-10-14T10:30:05Z",
    "source_topic": "omninode_codegen_request_analyze_v1",
    "consumer_group": "omniarchon_codegen_intelligence"
}
```

### Error Handling Pattern

```python
import asyncio
from aiokafka import AIOKafkaProducer
from omnibase_core import ModelOnexError
import traceback
import json
from datetime import datetime

async def process_with_dlq(
    request_event: dict,
    source_topic: str,
    dlq_topic: str,
    max_retries: int = 3
):
    """Process event with DLQ fallback on failure."""
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # Attempt to process event
            result = await process_event(request_event)
            return result

        except Exception as e:
            retry_count += 1

            if retry_count > max_retries:
                # Max retries exhausted - send to DLQ
                await send_to_dlq(
                    original_event=request_event,
                    error=e,
                    source_topic=source_topic,
                    dlq_topic=dlq_topic,
                    retry_count=retry_count
                )
                raise  # Re-raise for caller to handle

            else:
                # Retry with exponential backoff
                backoff_seconds = 2 ** retry_count  # 2s, 4s, 8s
                print(f"Retry {retry_count}/{max_retries} after {backoff_seconds}s")
                await asyncio.sleep(backoff_seconds)

async def send_to_dlq(
    original_event: dict,
    error: Exception,
    source_topic: str,
    dlq_topic: str,
    retry_count: int
):
    """Send failed event to DLQ topic."""
    dlq_event = {
        "original_event": original_event,
        "error": {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "retry_count": retry_count,
        "max_retries": 3,
        "dlq_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_topic": source_topic,
        "consumer_group": "omniarchon_codegen_intelligence"
    }

    # Publish to DLQ topic
    producer = AIOKafkaProducer(bootstrap_servers="localhost:19092")
    await producer.start()
    try:
        await producer.send_and_wait(
            dlq_topic,
            value=json.dumps(dlq_event).encode('utf-8')
        )
        print(f"Event sent to DLQ: {dlq_topic}")
    finally:
        await producer.stop()
```

### DLQ Monitoring

The `CodegenDLQMonitor` provides threshold-based alerting for DLQ topics.

```python
from omninode_bridge.monitoring.codegen_dlq_monitor import CodegenDLQMonitor

# Initialize DLQ monitor
monitor = CodegenDLQMonitor(
    kafka_config={"bootstrap_servers": "localhost:19092"},
    alert_threshold=10,  # Alert after 10 DLQ messages
    alert_webhook_url="https://alerts.example.com/webhook"
)

# Start monitoring (runs continuously)
await monitor.start_monitoring()

# Get DLQ statistics
stats = await monitor.get_dlq_stats()
print(f"Total DLQ messages: {stats['total_dlq_messages']}")
print(f"DLQ counts: {stats['dlq_counts']}")

# Stop monitoring
await monitor.stop_monitoring()
```

**DLQ Monitor Features**:
- Threshold-based alerting (configurable per topic)
- Webhook notifications for alerts
- 15-minute alert cooldown to prevent spam
- Per-topic message counting
- Structured logging with correlation IDs

---

## Monitoring & Alerting

### Event Tracing

The `CodegenEventTracer` provides correlation-based event tracing for debugging.

```python
from omninode_bridge.dashboard.codegen_event_tracer import CodegenEventTracer
from omninode_bridge.infrastructure.postgres_connection_manager import (
    PostgresConnectionManager,
    ModelPostgresConfig
)
from uuid import UUID

# Initialize database connection
config = ModelPostgresConfig.from_environment()
db_manager = PostgresConnectionManager(config)
await db_manager.initialize()

# Create event tracer
tracer = CodegenEventTracer(db_manager)

# Trace session events
session_id = UUID("123e4567-e89b-12d3-a456-426614174000")
trace = await tracer.trace_session_events(session_id, time_range_hours=24)

print(f"Session: {trace['session_id']}")
print(f"Total events: {trace['total_events']}")
print(f"Duration: {trace['session_duration_ms']}ms")
print(f"Status: {trace['status']}")

for event in trace['events']:
    print(f"  [{event['timestamp']}] {event['event_type']}: {event['topic']}")

# Get performance metrics
metrics = await tracer.get_session_metrics(session_id)
print(f"Success rate: {metrics['success_rate'] * 100:.1f}%")
print(f"Average response time: {metrics['avg_response_time_ms']}ms")
print(f"P95 response time: {metrics['p95_response_time_ms']}ms")

# Find correlated events
correlation_id = UUID("456e7890-e89b-12d3-a456-426614174000")
events = await tracer.find_correlated_events(correlation_id)
print(f"Found {len(events)} correlated events")
```

### Key Metrics to Monitor

1. **Topic Health**:
   - Consumer lag (should be <100 messages)
   - Partition distribution (even distribution across partitions)
   - Replication status (all replicas in-sync)

2. **Consumer Performance**:
   - Processing time per message (<5s average)
   - Error rate (<1% of messages)
   - DLQ count (should be minimal)

3. **Producer Performance**:
   - Publish latency (<100ms p95)
   - Batch size efficiency (>50% batch utilization)
   - Compression ratio (>2x with gzip)

4. **Session Metrics**:
   - Session duration (track p50, p95, p99)
   - Success rate (>95% target)
   - Bottleneck identification (topics with >5s avg response time)

### Monitoring Tools

1. **Redpanda Console**: Web UI at http://localhost:8080 (if deployed)
2. **Prometheus Metrics**: Expose Kafka metrics for Prometheus scraping
3. **DLQ Monitor**: Custom monitoring for dead letter queues
4. **Event Tracer**: Database-backed event tracing and correlation

---

## Performance Characteristics

### Throughput & Latency Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Producer latency (p95) | <100ms | End-to-end publish time |
| Consumer latency (p95) | <50ms | Message consumption time |
| Processing time (avg) | <2s | Request → Response time |
| Processing time (p95) | <5s | 95th percentile processing |
| Throughput | 1000+ msg/s | Per topic throughput |
| DLQ rate | <1% | Percentage of failed events |

### Topic Performance

**Request/Response Topics** (3 partitions each):
- Throughput: ~3000 messages/sec (1000 per partition)
- Latency: <100ms p95 (producer) + <50ms p95 (consumer)
- Storage: ~7 days retention × average message rate

**Status Topic** (6 partitions):
- Throughput: ~6000 messages/sec (1000 per partition)
- Latency: <50ms p95 (status updates are non-critical)
- Storage: 3 days retention (shorter for transient status)

**DLQ Topics** (1 partition each):
- Throughput: <10 messages/sec (minimal failures expected)
- Latency: Not critical (failure handling is async)
- Storage: 30 days retention (forensic analysis)

### Resource Requirements

**Redpanda Cluster**:
- Memory: 1GB minimum (configured in docker-compose)
- CPU: 1 core minimum (single-broker deployment)
- Disk: 10GB minimum (for 7-day retention)
- Network: 100Mbps minimum (for throughput targets)

**Consumer Instances**:
- Memory: 256MB per consumer instance
- CPU: 0.5 cores per consumer instance
- Network: 50Mbps per consumer instance

### Scaling Considerations

**Horizontal Scaling**:
- Add consumer instances to consumer groups (auto-rebalancing)
- Increase partition count for higher throughput (requires topic recreation)
- Add broker nodes for higher availability (production deployment)

**Vertical Scaling**:
- Increase Redpanda memory allocation for larger message buffers
- Increase CPU cores for better compression performance
- Increase disk IOPS for faster log writes

---

## Best Practices

### Event Publishing

1. **Always set correlation_id**: Enable request/response matching
2. **Include session_id**: Track events across entire code generation session
3. **Use schema_version**: Support backward-compatible schema evolution
4. **Validate before publishing**: Catch errors early with Pydantic validation
5. **Handle publish failures**: Implement retry logic with exponential backoff
6. **Monitor publish latency**: Alert if p95 latency exceeds 100ms

### Event Consumption

1. **Idempotent processing**: Design consumers to handle duplicate messages
2. **Graceful error handling**: Use DLQ pattern for failed events
3. **Commit offsets carefully**: Use auto-commit for simple cases, manual for critical
4. **Monitor consumer lag**: Alert if lag exceeds 100 messages
5. **Scale horizontally**: Add consumer instances for increased throughput
6. **Track processing time**: Identify bottlenecks with metrics

### Schema Evolution

1. **Add new fields as optional**: Maintain backward compatibility
2. **Never remove existing fields**: Deprecate instead
3. **Version schema_version field**: Enable version-specific handling
4. **Test compatibility**: Verify old consumers can read new schemas
5. **Document breaking changes**: Communicate schema migrations clearly

### Monitoring & Operations

1. **Monitor DLQ topics**: Alert on threshold exceeded (>10 messages)
2. **Track correlation chains**: Use event tracer for debugging
3. **Log structured data**: Include correlation_id, session_id in all logs
4. **Alert on high latency**: >5s processing time indicates bottleneck
5. **Review DLQ messages**: Investigate failures weekly
6. **Optimize topic configuration**: Adjust retention, partitions based on usage

### Datetime and Timezone Handling

**Critical Requirement**: All timestamps in event schemas MUST be timezone-aware (UTC).

#### Why Timezone-Aware Datetimes?

1. **PostgreSQL Compliance**: Database TIMESTAMPTZ columns expect timezone-aware datetimes
2. **Cross-Service Consistency**: Services in different timezones must share consistent timestamps
3. **No Ambiguity**: Timezone-aware datetimes eliminate DST and timezone conversion issues
4. **ISO 8601 Compatibility**: Enables proper serialization for Kafka event payloads

#### Correct Datetime Usage

```python
from datetime import datetime, timezone

# ✅ CORRECT: Timezone-aware datetime (UTC)
timestamp = datetime.now(timezone.utc)

# ❌ INCORRECT: Naive datetime (no timezone)
timestamp = datetime.utcnow()  # Deprecated in Python 3.12+
```

#### Schema Default Factories

All event schemas use timezone-aware datetime factories:

```python
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class EventSchema(BaseModel):
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp (UTC)"
    )
```

#### Common Pitfalls to Avoid

1. **DO NOT use `datetime.utcnow()`**: Returns naive datetime (no timezone info)
2. **DO NOT use `datetime.now()`**: Returns local time without timezone
3. **DO use `datetime.now(timezone.utc)`**: Returns UTC with timezone info
4. **Always import `timezone`**: `from datetime import datetime, timezone`

#### Database Considerations

PostgreSQL `TIMESTAMPTZ` columns:
- **Expect**: Timezone-aware datetime objects
- **Reject**: Naive datetime objects (causes runtime errors)
- **Store**: All timestamps in UTC internally
- **Return**: Timezone-aware datetime objects with UTC timezone

#### Testing Datetime Fields

```python
from datetime import datetime, timezone
import pytest

def test_timezone_aware_timestamp():
    """Verify all timestamps are timezone-aware."""
    event = CodegenAnalysisRequest(
        correlation_id=uuid4(),
        session_id=uuid4(),
        prd_content="Test content"
    )

    # Verify timestamp is timezone-aware
    assert event.timestamp.tzinfo is not None
    assert event.timestamp.tzinfo == timezone.utc
```

#### Migration from Naive Datetimes

If migrating from naive datetimes:

```python
# Old (naive datetime)
timestamp = datetime.utcnow()

# New (timezone-aware)
timestamp = datetime.now(timezone.utc)

# Converting existing naive UTC datetime to timezone-aware
naive_datetime = datetime.utcnow()
aware_datetime = naive_datetime.replace(tzinfo=timezone.utc)
```

### Security Considerations

1. **Use TLS for production**: Enable encryption in transit
2. **Authenticate consumers**: Use SASL/SCRAM for authentication
3. **Authorize topic access**: Implement ACLs for topic permissions
4. **Encrypt sensitive data**: Use application-level encryption for PII
5. **Rotate credentials**: Regularly rotate Kafka credentials
6. **Audit access logs**: Track who accessed which topics

---

## Appendix: Topic Creation Script

```bash
#!/bin/bash
# Create all 13 codegen topics with proper configuration

BOOTSTRAP_SERVERS="localhost:19092"

# Request topics
kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_request_analyze_v1 \
  --partitions 3 --replication-factor 1 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_request_validate_v1 \
  --partitions 3 --replication-factor 1 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_request_pattern_v1 \
  --partitions 3 --replication-factor 1 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_request_mixin_v1 \
  --partitions 3 --replication-factor 1 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

# Response topics
kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_response_analyze_v1 \
  --partitions 3 --replication-factor 1 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_response_validate_v1 \
  --partitions 3 --replication-factor 1 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_response_pattern_v1 \
  --partitions 3 --replication-factor 1 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_response_mixin_v1 \
  --partitions 3 --replication-factor 1 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

# Status topic
kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_status_session_v1 \
  --partitions 6 --replication-factor 1 \
  --config retention.ms=259200000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

# DLQ topics
kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_dlq_analyze_v1 \
  --partitions 1 --replication-factor 1 \
  --config retention.ms=2592000000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_dlq_validate_v1 \
  --partitions 1 --replication-factor 1 \
  --config retention.ms=2592000000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_dlq_pattern_v1 \
  --partitions 1 --replication-factor 1 \
  --config retention.ms=2592000000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

kafka-topics --bootstrap-server $BOOTSTRAP_SERVERS --create --if-not-exists \
  --topic omninode_codegen_dlq_mixin_v1 \
  --partitions 1 --replication-factor 1 \
  --config retention.ms=2592000000 \
  --config cleanup.policy=delete \
  --config compression.type=gzip

echo "All 13 codegen topics created successfully!"
```

---

**Document Version**: 1.0
**Last Updated**: October 2025
**Maintainer**: OmniNode Bridge Team

---

## Related Documentation

### Event Infrastructure
- [Event System Guide](./EVENT_SYSTEM_GUIDE.md) - Event publishing patterns and best practices
- [Quickstart Guide](./QUICKSTART.md) - Quick setup guide
- [Kafka Schema Registry](./KAFKA_SCHEMA_REGISTRY.md) - OnexEnvelopeV1 format specification
- [Schema Validation](./KAFKA_SCHEMA_VALIDATION.md) - Schema validation patterns

### API & Integration
- [API Event Schemas](../api/event-schemas.md) - API event definitions
- [Hook Events](../api/hook-events.md) - Hook event types and payloads
- [API Reference](../api/API_REFERENCE.md) - Complete API documentation

### Architecture
- [Kafka Topic Strategy](../planning/KAFKA_TOPIC_STRATEGY.md) - Topic design and strategy
- [ADR-014: Event-Driven Architecture](../architecture/adrs/adr-014-event-driven-architecture-kafka.md) - Architectural decision
- [Service Architecture](../architecture/service-architecture.md) - System architecture overview

### Research & Implementation
- [Event-Driven Database Adapter Research](../research/EVENT_DRIVEN_DATABASE_ADAPTER_RESEARCH.md) - Database adapter patterns
- [Completion Status & Roadmap](../research/COMPLETION_STATUS_AND_INTEGRATION_ROADMAP.md) - Integration roadmap

### Code References
- [codegen-topics-config.yaml](./codegen-topics-config.yaml) - Topic configuration reference
- [codegen_schemas.py](../../src/omninode_bridge/events/codegen_schemas.py) - Event schema definitions
