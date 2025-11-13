# Event System Guide - OmniNode Bridge

**Complete Event Infrastructure Documentation**
**Version**: 2.1.0
**Last Updated**: October 29, 2025
**Status**: Complete
**Note**: Consolidated with EVENT_INFRASTRUCTURE_GUIDE + EVENT_SCHEMAS (Oct 2025)

---

## Table of Contents

1. [Overview](#overview)
2. [Kafka/Redpanda Infrastructure](#kafkaredpanda-infrastructure)
3. [Event Schemas](#event-schemas)
4. [Hook Event Schemas](#hook-event-schemas)
5. [Producer Implementation](#producer-implementation)
6. [Consumer Patterns](#consumer-patterns)
7. [Event Tracing](#event-tracing)
8. [DLQ Monitoring](#dlq-monitoring)
9. [Operations Guide](#operations-guide)
10. [Performance Tuning](#performance-tuning)
11. [Performance Characteristics](#performance-characteristics)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

---

## Overview

### Event-Driven Architecture

OmniNode Bridge implements an event-driven architecture using Kafka/Redpanda for asynchronous communication between services. This architecture provides:

- **Loose Coupling**: Services communicate via events, not direct dependencies
- **Scalability**: Multiple consumers can process events in parallel
- **Reliability**: Events are persisted and can be replayed
- **Observability**: Complete audit trail of all system events

### Event Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Event-Driven Architecture                        │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  omniclaude (Client Service)                         │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 1. User requests code generation/validation                    │  │
│  │ 2. Publishes request events to Kafka                           │  │
│  │ 3. Subscribes to response topics for results                   │  │
│  └────────────────────┬───────────────────────────────────────────┘  │
└────────────────────────┼───────────────────────────────────────────┘
                         │ Publishes events (OnexEnvelopeV1)
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Kafka/Redpanda (Event Infrastructure)                   │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 13 Topics:                                                     │  │
│  │  • Request Topics (4): analyze, validate, pattern, mixin       │  │
│  │  • Response Topics (4): analysis results, validation, etc.     │  │
│  │  • Status Topics (1): Real-time status updates (6 partitions)  │  │
│  │  • DLQ Topics (4): Failed events for investigation             │  │
│  └────────────────────┬───────────────────────────────────────────┘  │
└────────────────────────┼───────────────────────────────────────────┘
                         │ Consumes events (OnexEnvelopeV1)
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│               omniarchon (Intelligence Service)                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 1. Subscribes to request topics                                │  │
│  │ 2. Processes events with AI intelligence                       │  │
│  │ 3. Publishes response events with results                      │  │
│  │ 4. Publishes status updates during processing                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Code Generation Workflow Context

The event infrastructure specifically supports **autonomous code generation workflows** between:
- **omniclaude**: AI code generator (client service)
- **omniarchon**: Intelligence processing service

**Event Flow Stages**:
1. **Request Phase**: omniclaude publishes analysis/validation/pattern/mixin requests
2. **Processing Phase**: omniarchon consumes requests, performs intelligence operations
3. **Response Phase**: omniarchon publishes results, omniclaude consumes responses
4. **Status Updates**: Both services publish real-time status updates
5. **Error Handling**: Failed events routed to DLQ topics for investigation

### 13 Kafka Topics

```
Request Topics (omniclaude → omniarchon):
├── omninode_codegen_request_analyze_v1     # PRD analysis requests
├── omninode_codegen_request_validate_v1    # Code validation requests
├── omninode_codegen_request_pattern_v1     # Pattern matching requests
└── omninode_codegen_request_mixin_v1       # Mixin recommendation requests

Response Topics (omniarchon → omniclaude):
├── omninode_codegen_response_analyze_v1    # Analysis results
├── omninode_codegen_response_validate_v1   # Validation results
├── omninode_codegen_response_pattern_v1    # Pattern matches
└── omninode_codegen_response_mixin_v1      # Mixin recommendations

Status Topics (bidirectional):
└── omninode_codegen_status_session_v1      # Real-time session status (6 partitions)

DLQ Topics (failed events):
├── omninode_codegen_dlq_analyze_v1         # Failed analysis events
├── omninode_codegen_dlq_validate_v1        # Failed validation events
├── omninode_codegen_dlq_pattern_v1         # Failed pattern events
└── omninode_codegen_dlq_mixin_v1           # Failed mixin events
```

### Topic Naming Convention

All topics follow the pattern: `omninode_codegen_{category}_{operation}_v{version}`

- **Prefix**: `omninode_codegen_` (namespace for code generation events)
- **Category**: `request`, `response`, `status`, `dlq` (event category)
- **Operation**: `analyze`, `validate`, `pattern`, `mixin`, `session` (event type)
- **Version**: `v1`, `v2`, etc. (schema versioning)

### Key Components

| Component | Purpose | File Path |
|-----------|---------|-----------|
| **Event Schemas** | Pydantic v2 models for all events | `src/omninode_bridge/events/codegen_schemas.py` |
| **Kafka Producer** | Event publishing with OnexEnvelopeV1 | `src/omninode_bridge/services/kafka_client.py` |
| **Event Tracer** | Database-backed event tracing | `src/omninode_bridge/dashboard/codegen_event_tracer.py` |
| **DLQ Monitor** | Dead Letter Queue monitoring | `src/omninode_bridge/events/dlq_monitor.py` |
| **Topics Config** | Topic configuration and partitions | `docs/events/codegen-topics-config.yaml` |

---

## Kafka/Redpanda Infrastructure

### Redpanda Setup

**Why Redpanda?**
- Drop-in replacement for Apache Kafka
- Simpler configuration (no Zookeeper)
- Better performance (C++ implementation)
- Smaller resource footprint

**Docker Compose Configuration**:
```yaml
# docker-compose.yml

redpanda:
  image: docker.redpanda.com/vectorized/redpanda:latest
  container_name: omninode-bridge-redpanda
  command:
    - redpanda
    - start
    - --smp=1
    - --memory=512M
    - --reserve-memory=0M
    - --overprovisioned
    - --set redpanda.auto_create_topics_enabled=true
    - --kafka-addr=PLAINTEXT://0.0.0.0:29092,EXTERNAL://0.0.0.0:9092
    - --advertise-kafka-addr=PLAINTEXT://omninode-bridge-redpanda:29092,EXTERNAL://localhost:9092
  ports:
    - 9092:9092   # External Kafka API
    - 29092:29092 # Internal Kafka API
    - 9644:9644   # Redpanda Admin API
  environment:
    REDPANDA_ENVIRONMENT: development
  healthcheck:
    test: ["CMD-SHELL", "rpk cluster info || exit 1"]
    interval: 30s
    timeout: 10s
    retries: 5
```

### Hostname Configuration (ONE-TIME)

Kafka/Redpanda requires hostname resolution due to its two-step broker discovery protocol.

**Linux/macOS**:
```bash
# Add to /etc/hosts (ONE-TIME)
echo "127.0.0.1 omninode-bridge-redpanda" | sudo tee -a /etc/hosts

# Verify
grep omninode-bridge-redpanda /etc/hosts
```

**Windows (Administrator)**:
```powershell
# Add to hosts file
echo 127.0.0.1 omninode-bridge-redpanda >> C:\Windows\System32\drivers\etc\hosts

# Verify
type C:\Windows\System32\drivers\etc\hosts | findstr omninode-bridge-redpanda
```

### Topic Configuration

**File**: `docs/events/codegen-topics-config.yaml`

```yaml
# Request topic configuration example
omninode_codegen_request_analyze_v1:
  partitions: 3  # Parallel processing
  replication_factor: 1  # Single-node development
  retention_ms: 604800000  # 7 days
  cleanup_policy: "delete"
  compression_type: "gzip"
  description: "PRD analysis requests from omniclaude to omniarchon"
```

**Topic Configuration Reference**:
```yaml
performance:
  # Message size limits
  max_message_bytes: 1048576  # 1MB per message

  # Producer configuration
  producer_acks: 1  # Wait for leader acknowledgment
  producer_compression: "gzip"
  producer_batch_size: 16384  # 16KB
  producer_linger_ms: 10  # Wait up to 10ms for batching

  # Consumer configuration
  consumer_fetch_min_bytes: 1024  # 1KB minimum fetch
  consumer_fetch_max_wait_ms: 500  # Wait up to 500ms
  consumer_max_partition_fetch_bytes: 1048576  # 1MB per partition
```

### Creating Topics

```bash
# Connect to Redpanda container
docker exec -it omninode-bridge-redpanda bash

# Create topic with specific configuration
rpk topic create omninode_codegen_request_analyze_v1 \
  --partitions 3 \
  --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config compression.type=gzip

# List all topics
rpk topic list

# Describe topic details
rpk topic describe omninode_codegen_request_analyze_v1

# View topic messages (consume last 10)
rpk topic consume omninode_codegen_request_analyze_v1 --num 10 --format json
```

### Consumer Groups

```yaml
# Consumer group configuration (docs/events/codegen-topics-config.yaml)

consumer_groups:
  omniclaude_codegen_consumer:
    topics:
      - omninode_codegen_response_analyze_v1
      - omninode_codegen_response_validate_v1
      - omninode_codegen_response_pattern_v1
      - omninode_codegen_response_mixin_v1
    auto_offset_reset: "latest"  # Start from latest on first run
    enable_auto_commit: true  # Automatically commit offsets
    session_timeout_ms: 30000  # 30 seconds
    heartbeat_interval_ms: 10000  # 10 seconds

  omniarchon_codegen_intelligence:
    topics:
      - omninode_codegen_request_analyze_v1
      - omninode_codegen_request_validate_v1
      - omninode_codegen_request_pattern_v1
      - omninode_codegen_request_mixin_v1
    auto_offset_reset: "latest"
    enable_auto_commit: true
    session_timeout_ms: 30000
    heartbeat_interval_ms: 10000
```

**Managing Consumer Groups**:
```bash
# List consumer groups
rpk group list

# Describe consumer group offsets
rpk group describe omniclaude_codegen_consumer

# Reset consumer group offsets (BE CAREFUL)
rpk group seek omniclaude_codegen_consumer --to start

# Delete consumer group (BE CAREFUL - only when no active consumers)
rpk group delete omniclaude_codegen_consumer
```

---

## Event Schemas

### OnexEnvelopeV1 Format

All Kafka events use the standardized **OnexEnvelopeV1** envelope format for consistency:

```python
# src/omninode_bridge/events/codegen_schemas.py

class OnexEnvelopeV1(BaseModel):
    """
    Standardized event envelope for all Kafka messages.

    Ensures:
    - Version compatibility
    - Correlation tracking
    - Event type identification
    - Temporal ordering
    """
    envelope_version: str = "1.0"
    correlation_id: UUID
    session_id: UUID | None = None
    event_type: str
    timestamp: datetime
    payload: dict[str, Any]
```

**Example Event** (JSON format):
```json
{
  "envelope_version": "1.0",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "event_type": "CodegenAnalysisRequest",
  "timestamp": "2025-10-16T12:34:56.789Z",
  "payload": {
    "prd_content": "As a user, I want...",
    "analysis_type": "full",
    "workspace_context": {
      "project_root": "/path/to/project",
      "files": ["file1.py", "file2.py"]
    }
  }
}
```

### Schema Versioning Strategy

All event schemas follow a versioning strategy for backward compatibility:

- **Current Version**: 1.0
- **Evolution Strategy**: Backward-compatible changes only
- **Version Field**: All schemas include `schema_version` field
- **New Fields**: Add as optional fields to maintain compatibility
- **Deprecated Fields**: Mark as deprecated but never remove
- **Breaking Changes**: Require new version (e.g., v2) and separate topic

### 9 Event Schemas

**File**: `src/omninode_bridge/events/codegen_schemas.py`

#### 1. CodegenAnalysisRequest

```python
class CodegenAnalysisRequest(BaseModel):
    """
    Schema for PRD analysis requests.

    Published to: omninode_codegen_request_analyze_v1
    Flow: omniclaude → omniarchon
    """
    correlation_id: UUID
    session_id: UUID
    prd_content: str
    analysis_type: EnumAnalysisType = EnumAnalysisType.FULL
    workspace_context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = "1.0"
```

**Example Usage**:
```python
from uuid import uuid4
from omninode_bridge.events.codegen_schemas import CodegenAnalysisRequest
from omninode_bridge.events.enums import EnumAnalysisType

request = CodegenAnalysisRequest(
    correlation_id=uuid4(),
    session_id=uuid4(),
    prd_content="# PRD: New Feature\n\nImplement feature X",
    analysis_type=EnumAnalysisType.FULL,
    workspace_context={
        "project_root": "/path/to/project",
        "files": ["file1.py", "file2.py"]
    }
)
```

#### 2. CodegenAnalysisResponse

```python
class CodegenAnalysisResponse(BaseModel):
    """
    Schema for PRD analysis responses.

    Published to: omninode_codegen_response_analyze_v1
    Flow: omniarchon → omniclaude
    """
    correlation_id: UUID
    session_id: UUID
    analysis_result: dict[str, Any]
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = "1.0"
```

#### 3. CodegenValidationRequest

```python
class CodegenValidationRequest(BaseModel):
    """
    Schema for code validation requests.

    Published to: omninode_codegen_request_validate_v1
    Flow: omniclaude → omniarchon
    """
    correlation_id: UUID
    session_id: UUID
    code_content: str
    node_type: EnumNodeType
    contracts: list[dict[str, Any]] = Field(default_factory=list)
    validation_type: EnumValidationType = EnumValidationType.FULL
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = "1.0"
```

#### 4. CodegenValidationResponse

```python
class CodegenValidationResponse(BaseModel):
    """
    Schema for code validation responses.

    Published to: omninode_codegen_response_validate_v1
    Flow: omniarchon → omniclaude
    """
    correlation_id: UUID
    session_id: UUID
    validation_result: dict[str, Any]
    quality_score: float = Field(..., ge=0.0, le=1.0)
    onex_compliance_score: float = Field(..., ge=0.0, le=1.0)
    is_valid: bool
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = "1.0"
```

#### 5. CodegenPatternRequest

```python
class CodegenPatternRequest(BaseModel):
    """
    Schema for pattern matching requests.

    Published to: omninode_codegen_request_pattern_v1
    Flow: omniclaude → omniarchon
    """
    correlation_id: UUID
    session_id: UUID
    node_description: str
    node_type: EnumNodeType
    limit: int = Field(default=5, ge=1, le=20)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = "1.0"
```

#### 6. CodegenPatternResponse

```python
class CodegenPatternResponse(BaseModel):
    """
    Schema for pattern matching responses.

    Published to: omninode_codegen_response_pattern_v1
    Flow: omniarchon → omniclaude
    """
    correlation_id: UUID
    session_id: UUID
    pattern_result: list[dict[str, Any]]
    total_matches: int
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = "1.0"
```

#### 7. CodegenMixinRequest

```python
class CodegenMixinRequest(BaseModel):
    """
    Schema for mixin recommendation requests.

    Published to: omninode_codegen_request_mixin_v1
    Flow: omniclaude → omniarchon
    """
    correlation_id: UUID
    session_id: UUID
    requirements: list[str]
    node_type: EnumNodeType
    existing_mixins: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = "1.0"
```

#### 8. CodegenMixinResponse

```python
class CodegenMixinResponse(BaseModel):
    """
    Schema for mixin recommendation responses.

    Published to: omninode_codegen_response_mixin_v1
    Flow: omniarchon → omniclaude
    """
    correlation_id: UUID
    session_id: UUID
    mixin_recommendations: list[dict[str, Any]]
    total_recommendations: int
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = "1.0"
```

#### 9. CodegenStatusEvent

```python
class CodegenStatusEvent(BaseModel):
    """
    Schema for code generation status updates.

    Published to: omninode_codegen_status_session_v1
    Flow: Both omniclaude and omniarchon → monitoring/debugging
    """
    session_id: UUID
    status: EnumSessionStatus
    progress_percentage: float = Field(..., ge=0.0, le=100.0)
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = "1.0"
```

### Enums

```python
# src/omninode_bridge/events/enums.py

class EnumAnalysisType(str, Enum):
    """Analysis depth types."""
    QUICK = "quick"  # Fast surface-level analysis
    PARTIAL = "partial"  # Moderate depth analysis
    FULL = "full"  # Comprehensive deep analysis

class EnumValidationType(str, Enum):
    """Code validation types."""
    SYNTAX = "syntax"  # Syntax checking only
    SEMANTIC = "semantic"  # Semantic analysis
    COMPLIANCE = "compliance"  # ONEX compliance checking
    FULL = "full"  # All validation types

class EnumNodeType(str, Enum):
    """ONEX v2.0 node types."""
    EFFECT = "effect"  # I/O and side effects
    COMPUTE = "compute"  # Pure computation
    REDUCER = "reducer"  # Aggregation and state
    ORCHESTRATOR = "orchestrator"  # Workflow coordination

class EnumSessionStatus(str, Enum):
    """Code generation session statuses."""
    PENDING = "pending"  # Not started
    PROCESSING = "processing"  # In progress
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed with errors
    CANCELLED = "cancelled"  # User cancelled
```

---

## Hook Event Schemas

### Overview

**Hook events** use a different base event structure than code generation events. While codegen events use **OnexEnvelopeV1**, hook events use a **BaseEvent** structure with **EventSource** metadata for comprehensive service monitoring and operations.

**Key Differences**:
- **Codegen Events** (OnexEnvelopeV1): Code generation workflows, correlation tracking, session management
- **Hook Events** (BaseEvent): Service lifecycle, tool registration, proxy operations, intelligence gathering

### Base Event Structure

All hook events follow a standard envelope format with source tracking:

```json
{
  "event_id": "uuid4",
  "event_type": "string",
  "timestamp": "ISO8601",
  "version": "semver",
  "source": {
    "service_name": "string",
    "service_version": "string",
    "instance_id": "string"
  },
  "correlation_id": "uuid4",
  "trace_id": "string",
  "metadata": {},
  "payload": {}
}
```

**Schema Definition**:
```python
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from uuid import UUID

class EventSource(BaseModel):
    service_name: str = Field(..., description="Name of the originating service")
    service_version: str = Field(..., description="Version of the originating service")
    instance_id: str = Field(..., description="Unique instance identifier")

class BaseEvent(BaseModel):
    event_id: UUID = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event creation timestamp (UTC)"
    )
    version: str = Field("1.0.0", description="Event schema version")
    source: EventSource = Field(..., description="Event source information")
    correlation_id: Optional[UUID] = Field(None, description="Request correlation ID")
    trace_id: Optional[str] = Field(None, description="Distributed trace ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    payload: Dict[str, Any] = Field(..., description="Event-specific payload")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
```

### Service Lifecycle Events

#### Service Started Event

Published when a service successfully starts and registers with the system.

**Event Type**: `service.lifecycle.started`

**Example**:
```json
{
  "event_type": "service.lifecycle.started",
  "payload": {
    "service_info": {
      "name": "omniagent",
      "version": "1.2.3",
      "type": "agent_service",
      "instance_id": "omniagent-prod-001",
      "host": "10.0.1.15",
      "port": 8000,
      "health_check_url": "http://10.0.1.15:8000/health",
      "capabilities": [
        "code_generation",
        "documentation",
        "testing"
      ],
      "dependencies": [
        {
          "service": "omnimcp",
          "type": "required",
          "version": ">=1.0.0"
        }
      ],
      "resources": {
        "cpu_limit": "2000m",
        "memory_limit": "4Gi",
        "disk_usage": "10Gi"
      }
    },
    "startup_info": {
      "startup_time_ms": 3500,
      "initialization_steps": [
        {
          "step": "load_configuration",
          "duration_ms": 150,
          "status": "success"
        },
        {
          "step": "connect_database",
          "duration_ms": 250,
          "status": "success"
        },
        {
          "step": "register_with_consul",
          "duration_ms": 100,
          "status": "success"
        }
      ]
    }
  }
}
```

**Schema Definition**:
```python
class ServiceInfo(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., regex=r'^\d+\.\d+\.\d+')
    type: str = Field(..., description="Service type classification")
    instance_id: str = Field(..., description="Unique instance identifier")
    host: str = Field(..., description="Service host address")
    port: int = Field(..., ge=1, le=65535)
    health_check_url: str = Field(..., description="Health check endpoint URL")
    capabilities: List[str] = Field(default_factory=list)
    dependencies: List[Dict[str, Any]] = Field(default_factory=list)
    resources: Dict[str, str] = Field(default_factory=dict)

class StartupStep(BaseModel):
    step: str
    duration_ms: int = Field(..., ge=0)
    status: str = Field(..., regex=r'^(success|failed|warning)$')
    details: Optional[Dict[str, Any]] = None

class StartupInfo(BaseModel):
    startup_time_ms: int = Field(..., ge=0)
    initialization_steps: List[StartupStep] = Field(default_factory=list)

class ServiceStartedPayload(BaseModel):
    service_info: ServiceInfo
    startup_info: StartupInfo
```

#### Service Stopped Event

Published when a service gracefully shuts down.

**Event Type**: `service.lifecycle.stopped`

**Example**:
```json
{
  "event_type": "service.lifecycle.stopped",
  "payload": {
    "service_info": {
      "name": "omniagent",
      "instance_id": "omniagent-prod-001"
    },
    "shutdown_info": {
      "shutdown_reason": "graceful_shutdown",
      "uptime_seconds": 86400,
      "final_metrics": {
        "requests_processed": 15420,
        "errors_encountered": 23,
        "avg_response_time_ms": 150
      },
      "cleanup_steps": [
        {
          "step": "drain_connections",
          "duration_ms": 2000,
          "status": "success"
        },
        {
          "step": "save_state",
          "duration_ms": 500,
          "status": "success"
        }
      ]
    }
  }
}
```

#### Service Health Changed Event

Published when a service health status changes (healthy → degraded → unhealthy).

**Event Type**: `service.lifecycle.health_changed`

**Example**:
```json
{
  "event_type": "service.lifecycle.health_changed",
  "payload": {
    "service_info": {
      "name": "omniagent",
      "instance_id": "omniagent-prod-001"
    },
    "health_change": {
      "previous_status": "healthy",
      "current_status": "degraded",
      "change_reason": "high_error_rate",
      "health_details": {
        "database_connection": "healthy",
        "external_apis": "degraded",
        "memory_usage": "warning",
        "cpu_usage": "healthy"
      },
      "recovery_actions": [
        "reduce_traffic",
        "restart_external_connections"
      ]
    }
  }
}
```

### Tool Registration Events

#### Tool Discovered Event

Published when a new tool is discovered and registered in the system.

**Event Type**: `tool.registration.discovered`

**Example**:
```json
{
  "event_type": "tool.registration.discovered",
  "payload": {
    "tool_info": {
      "name": "code_analyzer",
      "version": "2.1.0",
      "service_source": "omniagent",
      "category": "code_analysis",
      "description": "Analyze code quality and patterns",
      "input_schema": {
        "type": "object",
        "properties": {
          "code": {"type": "string"},
          "language": {"type": "string"},
          "analysis_type": {"type": "string", "enum": ["quality", "security", "performance"]}
        },
        "required": ["code", "language"]
      },
      "output_schema": {
        "type": "object",
        "properties": {
          "score": {"type": "number", "minimum": 0, "maximum": 100},
          "issues": {"type": "array"},
          "recommendations": {"type": "array"}
        }
      },
      "execution_requirements": {
        "max_execution_time_ms": 30000,
        "memory_limit_mb": 512,
        "requires_filesystem": true,
        "requires_network": false
      }
    },
    "discovery_context": {
      "discovery_method": "service_registration",
      "discovery_timestamp": "2024-01-15T10:30:00Z",
      "registration_source": "automatic"
    }
  }
}
```

#### Tool Executed Event

Published when a tool execution completes (successfully or with errors).

**Event Type**: `tool.execution.completed`

**Example**:
```json
{
  "event_type": "tool.execution.completed",
  "payload": {
    "execution_info": {
      "tool_name": "code_analyzer",
      "execution_id": "exec_550e8400-e29b-41d4-a716-446655440000",
      "requester_service": "omniplan",
      "target_service": "omniagent"
    },
    "execution_details": {
      "start_time": "2024-01-15T10:35:00Z",
      "end_time": "2024-01-15T10:35:02.5Z",
      "duration_ms": 2500,
      "status": "success",
      "input_size_bytes": 15420,
      "output_size_bytes": 3240
    },
    "performance_metrics": {
      "cpu_usage_percent": 45.2,
      "memory_peak_mb": 256,
      "io_operations": 127,
      "network_calls": 0
    },
    "intelligence_data": {
      "patterns_identified": [
        {
          "pattern_type": "code_quality",
          "confidence": 0.92,
          "description": "High complexity function detected"
        }
      ],
      "recommendations": [
        {
          "type": "refactoring",
          "priority": "medium",
          "description": "Consider breaking down large function"
        }
      ]
    }
  }
}
```

### Proxy Event Schemas

#### Request Routed Event

Published when a request is routed through the proxy to a target service.

**Event Type**: `proxy.request.routed`

**Example**:
```json
{
  "event_type": "proxy.request.routed",
  "payload": {
    "request_info": {
      "request_id": "req_550e8400-e29b-41d4-a716-446655440000",
      "source_service": "omniplan",
      "target_service": "omniagent",
      "tool_name": "generate_code",
      "method": "POST",
      "path": "/tools/generate_code"
    },
    "routing_decision": {
      "routing_strategy": "load_balanced",
      "selected_instance": "omniagent-prod-002",
      "routing_factors": {
        "cpu_utilization": 0.45,
        "current_load": 23,
        "response_time_avg": 120,
        "health_score": 0.95
      },
      "alternative_instances": [
        "omniagent-prod-001",
        "omniagent-prod-003"
      ]
    },
    "caching_info": {
      "cache_key": "hash_of_request_content",
      "cache_hit": false,
      "cache_ttl_seconds": 300,
      "cacheable": true
    }
  }
}
```

#### Cache Event

Published when a cache hit or miss occurs in the proxy layer.

**Event Type**: `proxy.cache.hit` or `proxy.cache.miss`

**Example**:
```json
{
  "event_type": "proxy.cache.hit",
  "payload": {
    "cache_info": {
      "cache_key": "hash_of_request_content",
      "request_id": "req_550e8400-e29b-41d4-a716-446655440000",
      "cached_at": "2024-01-15T10:30:00Z",
      "ttl_remaining_seconds": 180,
      "cache_size_bytes": 5240
    },
    "performance_impact": {
      "response_time_saved_ms": 1800,
      "bandwidth_saved_bytes": 5240,
      "cpu_saved_percent": 25.0
    },
    "adaptive_caching": {
      "hit_rate_trend": 0.78,
      "ttl_adjustment": "extend",
      "confidence_score": 0.85
    }
  }
}
```

### Intelligence Event Schemas

#### Intelligence Pattern Discovered Event

Published when the intelligence system discovers a new pattern in service behavior.

**Event Type**: `intelligence.pattern.discovered`

**Example**:
```json
{
  "event_type": "intelligence.pattern.discovered",
  "payload": {
    "pattern_info": {
      "pattern_id": "pattern_550e8400-e29b-41d4-a716-446655440000",
      "pattern_type": "service_communication",
      "confidence_score": 0.87,
      "discovery_method": "statistical_analysis",
      "pattern_data": {
        "source_services": ["omniplan", "omnimemory"],
        "target_service": "omniagent",
        "frequency": "high",
        "success_rate": 0.94,
        "avg_response_time": 145,
        "peak_usage_hours": ["09:00-11:00", "14:00-16:00"]
      }
    },
    "analysis_context": {
      "analysis_window": {
        "start_time": "2024-01-14T00:00:00Z",
        "end_time": "2024-01-15T00:00:00Z",
        "sample_size": 1547
      },
      "data_sources": [
        "hook_events",
        "proxy_logs",
        "performance_metrics"
      ]
    },
    "actionable_insights": {
      "optimization_opportunities": [
        {
          "type": "caching",
          "potential_improvement": "25% faster response",
          "implementation_effort": "low"
        },
        {
          "type": "load_balancing",
          "potential_improvement": "improved resource utilization",
          "implementation_effort": "medium"
        }
      ],
      "risk_indicators": [
        {
          "type": "single_point_of_failure",
          "severity": "medium",
          "mitigation": "add service redundancy"
        }
      ]
    }
  }
}
```

#### Performance Anomaly Detected Event

Published when the intelligence system detects performance anomalies in services.

**Event Type**: `intelligence.performance.anomaly_detected`

**Example**:
```json
{
  "event_type": "intelligence.performance.anomaly_detected",
  "payload": {
    "anomaly_info": {
      "anomaly_id": "anomaly_550e8400-e29b-41d4-a716-446655440000",
      "detection_time": "2024-01-15T10:45:00Z",
      "anomaly_type": "response_time_spike",
      "severity": "high",
      "affected_services": ["omniagent"],
      "affected_tools": ["code_analyzer", "documentation_generator"]
    },
    "metrics_data": {
      "baseline_metrics": {
        "avg_response_time_ms": 150,
        "p95_response_time_ms": 300,
        "error_rate": 0.02,
        "throughput_rps": 45
      },
      "anomaly_metrics": {
        "avg_response_time_ms": 1200,
        "p95_response_time_ms": 2500,
        "error_rate": 0.15,
        "throughput_rps": 12
      },
      "degradation_factor": 8.0
    },
    "root_cause_analysis": {
      "probable_causes": [
        {
          "cause": "database_connection_pool_exhaustion",
          "confidence": 0.85,
          "evidence": [
            "high_connection_wait_times",
            "database_error_spike"
          ]
        },
        {
          "cause": "memory_pressure",
          "confidence": 0.65,
          "evidence": [
            "gc_pressure_increase",
            "memory_usage_spike"
          ]
        }
      ],
      "recommended_actions": [
        {
          "action": "increase_database_connection_pool",
          "priority": "high",
          "estimated_impact": "70% improvement"
        },
        {
          "action": "restart_service_instances",
          "priority": "medium",
          "estimated_impact": "temporary_relief"
        }
      ]
    }
  }
}
```

### Configuration Event Schemas

#### Configuration Updated Event

Published when system configuration changes occur.

**Event Type**: `configuration.updated`

**Example**:
```json
{
  "event_type": "configuration.updated",
  "payload": {
    "change_info": {
      "configuration_key": "service.omniagent.worker_pool_size",
      "previous_value": "4",
      "new_value": "8",
      "change_source": "consul_kv_update",
      "changed_by": "system_administrator",
      "change_reason": "performance_optimization"
    },
    "change_impact": {
      "affected_services": ["omniagent"],
      "restart_required": true,
      "estimated_downtime_seconds": 30,
      "backward_compatible": true
    },
    "rollback_info": {
      "rollback_available": true,
      "rollback_window_hours": 24,
      "rollback_steps": [
        "revert_consul_key",
        "restart_affected_services"
      ]
    }
  }
}
```

### Error Event Schemas

#### Service Error Event

Published when critical service errors occur.

**Event Type**: `error.service.critical`

**Example**:
```json
{
  "event_type": "error.service.critical",
  "payload": {
    "error_info": {
      "error_id": "error_550e8400-e29b-41d4-a716-446655440000",
      "service_name": "omniagent",
      "instance_id": "omniagent-prod-001",
      "error_type": "database_connection_failure",
      "severity": "critical",
      "message": "Failed to connect to PostgreSQL database after 3 retries",
      "stack_trace": "Traceback...",
      "first_occurrence": "2024-01-15T10:50:00Z",
      "occurrence_count": 5
    },
    "context_info": {
      "request_id": "req_550e8400-e29b-41d4-a716-446655440000",
      "user_id": "user_12345",
      "operation": "tool_execution",
      "tool_name": "code_analyzer",
      "environment": "production"
    },
    "impact_assessment": {
      "affected_users": 25,
      "affected_operations": ["code_analysis", "documentation_generation"],
      "estimated_recovery_time_minutes": 15,
      "business_impact": "medium"
    },
    "recovery_actions": {
      "automatic_recovery_attempted": true,
      "recovery_steps": [
        "reconnect_database",
        "fallback_to_readonly_replica"
      ],
      "manual_intervention_required": false
    }
  }
}
```

### Hook Event Processing

#### Event Processor Implementation

```python
# event_processing/processor.py
from typing import Dict, Any, Type
from pydantic import BaseModel, ValidationError
import logging

class EventProcessor:
    def __init__(self):
        self.schema_registry: Dict[str, Type[BaseModel]] = {}
        self.event_handlers: Dict[str, list] = {}
        self.logger = logging.getLogger(__name__)

    def register_schema(self, event_type: str, schema_class: Type[BaseModel]):
        """Register event schema for validation"""
        self.schema_registry[event_type] = schema_class

    def register_handler(self, event_type: str, handler_func):
        """Register event handler function"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler_func)

    async def process_event(self, raw_event: Dict[str, Any]) -> bool:
        """Process and validate incoming event"""
        try:
            # Validate base event structure
            base_event = BaseEvent(**raw_event)

            # Validate event-specific payload
            event_type = base_event.event_type
            if event_type in self.schema_registry:
                schema_class = self.schema_registry[event_type]
                validated_payload = schema_class(**base_event.payload)
                base_event.payload = validated_payload.dict()

            # Process with registered handlers
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    await handler(base_event)

            self.logger.info(f"Successfully processed event {base_event.event_id}")
            return True

        except ValidationError as e:
            self.logger.error(f"Event validation failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Event processing failed: {e}")
            return False

# Schema registration
processor = EventProcessor()

# Register service lifecycle schemas
processor.register_schema("service.lifecycle.started", ServiceStartedPayload)
processor.register_schema("service.lifecycle.stopped", ServiceStoppedPayload)
processor.register_schema("service.lifecycle.health_changed", ServiceHealthChangedPayload)

# Register tool schemas
processor.register_schema("tool.registration.discovered", ToolDiscoveredPayload)
processor.register_schema("tool.execution.completed", ToolExecutedPayload)

# Register proxy schemas
processor.register_schema("proxy.request.routed", RequestRoutedPayload)
processor.register_schema("proxy.cache.hit", CacheHitPayload)

# Register intelligence schemas
processor.register_schema("intelligence.pattern.discovered", PatternDiscoveredPayload)
processor.register_schema("intelligence.performance.anomaly_detected", PerformanceAnomalyPayload)
```

#### Event Publisher for Hook Events

```python
# event_processing/publisher.py
import json
import asyncio
from aiokafka import AIOKafkaProducer
from typing import Dict, Any
import uuid
from datetime import datetime, timezone

class EventPublisher:
    def __init__(self, bootstrap_servers: str):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

    async def start(self):
        await self.producer.start()

    async def stop(self):
        await self.producer.stop()

    async def publish_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source_service: str,
        source_version: str,
        instance_id: str,
        correlation_id: str = None,
        trace_id: str = None
    ):
        """Publish hook event to Kafka topic"""

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "source": {
                "service_name": source_service,
                "service_version": source_version,
                "instance_id": instance_id
            },
            "correlation_id": correlation_id,
            "trace_id": trace_id,
            "metadata": {},
            "payload": payload
        }

        # Determine topic based on event type
        topic = self._get_topic_for_event(event_type)

        try:
            await self.producer.send_and_wait(topic, event)
            return event["event_id"]
        except Exception as e:
            raise Exception(f"Failed to publish event: {e}")

    def _get_topic_for_event(self, event_type: str) -> str:
        """Map event type to Kafka topic"""
        topic_mapping = {
            "service.lifecycle": "hooks.service_lifecycle",
            "tool.registration": "hooks.tool_registration",
            "tool.execution": "hooks.tool_execution",
            "proxy.request": "proxy.requests",
            "proxy.cache": "proxy.cache",
            "intelligence.pattern": "intelligence.patterns",
            "intelligence.performance": "intelligence.performance",
            "configuration": "configuration.changes",
            "error": "system.errors"
        }

        for prefix, topic in topic_mapping.items():
            if event_type.startswith(prefix):
                return topic

        return "events.general"
```

### Hook Event Topics

**Recommended Kafka Topics for Hook Events**:

| Topic | Event Types | Description |
|-------|-------------|-------------|
| `hooks.service_lifecycle` | service.lifecycle.* | Service start/stop/health changes |
| `hooks.tool_registration` | tool.registration.* | Tool discovery and registration |
| `hooks.tool_execution` | tool.execution.* | Tool execution events |
| `proxy.requests` | proxy.request.* | Request routing through proxy |
| `proxy.cache` | proxy.cache.* | Cache hit/miss events |
| `intelligence.patterns` | intelligence.pattern.* | Pattern discovery events |
| `intelligence.performance` | intelligence.performance.* | Performance anomaly detection |
| `configuration.changes` | configuration.* | Configuration change events |
| `system.errors` | error.* | System error events |

**Note**: These topics are separate from the code generation topics (`omninode_codegen_*`) and serve different operational purposes.

---

## Producer Implementation

### KafkaEventProducer

**File**: `src/omninode_bridge/services/kafka_client.py`

```python
class KafkaEventProducer:
    """
    Production Kafka event producer with OnexEnvelopeV1 format.

    Features:
    - OnexEnvelopeV1 envelope format
    - Correlation ID tracking
    - DLQ for failed events
    - Dual client support (aiokafka, confluent-kafka)
    """

    def __init__(self, bootstrap_servers: str = "localhost:29092"):
        self.bootstrap_servers = bootstrap_servers
        self._producer: AIOKafkaProducer | None = None

    async def initialize(self) -> None:
        """Initialize Kafka producer."""
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            compression_type="gzip",
            acks=1,  # Wait for leader acknowledgment
            max_request_size=1048576,  # 1MB max message size
        )
        await self._producer.start()
        logger.info("Kafka producer initialized",
                   bootstrap_servers=self.bootstrap_servers)

    async def publish_event(
        self,
        topic: str,
        event: BaseModel,
        correlation_id: UUID,
        session_id: UUID | None = None
    ) -> None:
        """
        Publish event to Kafka with OnexEnvelopeV1 envelope.

        Args:
            topic: Kafka topic name
            event: Event payload (Pydantic model)
            correlation_id: UUID for request/response correlation
            session_id: Optional session tracking ID

        Raises:
            OnexError: If publish fails
        """
        # Create OnexEnvelopeV1
        envelope = OnexEnvelopeV1(
            envelope_version="1.0",
            correlation_id=correlation_id,
            session_id=session_id,
            event_type=event.__class__.__name__,
            timestamp=datetime.utcnow(),
            payload=event.model_dump()
        )

        # Serialize to JSON
        message = json.dumps(envelope.model_dump(), default=str)

        try:
            # Publish to Kafka (partition by correlation_id for ordering)
            await self._producer.send(
                topic=topic,
                value=message.encode('utf-8'),
                key=str(correlation_id).encode('utf-8')  # Partition key
            )

            logger.info("Event published",
                       topic=topic,
                       event_type=envelope.event_type,
                       correlation_id=str(correlation_id))

        except Exception as e:
            # Publish to DLQ (derive proper DLQ topic name)
            dlq_topic = self._dlq_topic_for(topic)
            await self._publish_to_dlq(dlq_topic, envelope, error=str(e))

            logger.error("Event publish failed, sent to DLQ",
                        topic=topic,
                        dlq_topic=dlq_topic,
                        error=str(e))

    def _dlq_topic_for(self, topic: str) -> str:
        """
        Derive DLQ topic name from original topic.

        Converts:
        - omninode_codegen_request_analyze_v1 → omninode_codegen_dlq_analyze_v1
        - omninode_codegen_response_validate_v1 → omninode_codegen_dlq_validate_v1

        Args:
            topic: Original topic name

        Returns:
            Corresponding DLQ topic name
        """
        # Extract operation from topic (e.g., "analyze", "validate", "pattern", "mixin")
        # Pattern: omninode_codegen_{request/response}_{operation}_v1
        parts = topic.split('_')
        if len(parts) >= 4:
            operation = parts[3]  # Extract operation (analyze, validate, etc.)
            return f"omninode_codegen_dlq_{operation}_v1"
        # Fallback to simple suffix if pattern doesn't match
        return f"{topic}_dlq"
```

### Publishing Events

**Example**: Publishing Analysis Request

```python
from uuid import uuid4
from omninode_bridge.events.codegen_schemas import CodegenAnalysisRequest
from omninode_bridge.events.enums import EnumAnalysisType
from omninode_bridge.services.kafka_client import KafkaEventProducer

# Initialize producer
producer = KafkaEventProducer(bootstrap_servers="localhost:29092")
await producer.initialize()

# Create event
correlation_id = uuid4()
session_id = uuid4()

request = CodegenAnalysisRequest(
    correlation_id=correlation_id,
    session_id=session_id,
    prd_content="# PRD: Feature X\n\nImplement new feature",
    analysis_type=EnumAnalysisType.FULL
)

# Publish event
await producer.publish_event(
    topic="omninode_codegen_request_analyze_v1",
    event=request,
    correlation_id=correlation_id,
    session_id=session_id
)

# Event is now in Kafka with OnexEnvelopeV1 envelope
```

### Correlation ID Tracking

Correlation IDs enable request/response matching:

```python
# Publisher side (omniclaude)
correlation_id = uuid4()

# Publish request
await producer.publish_event(
    topic="omninode_codegen_request_analyze_v1",
    event=request,
    correlation_id=correlation_id  # Track this
)

# Consumer side (omniarchon)
# Process request, then publish response with SAME correlation_id
response = CodegenAnalysisResponse(
    correlation_id=correlation_id,  # SAME ID
    session_id=session_id,
    analysis_result={"requirements": [...]}
)

await producer.publish_event(
    topic="omninode_codegen_response_analyze_v1",
    event=response,
    correlation_id=correlation_id  # SAME ID for matching
)
```

### Partitioning Strategy

Events are partitioned by `correlation_id` for ordered processing:

```python
# Producer automatically partitions by correlation_id
await self._producer.send(
    topic=topic,
    value=message.encode('utf-8'),
    key=str(correlation_id).encode('utf-8')  # Partition key
)

# Benefits:
# 1. All events for same correlation_id go to same partition
# 2. Events within partition maintain order
# 3. Parallel processing across partitions
```

---

## Consumer Patterns

### Basic Consumer

```python
from aiokafka import AIOKafkaConsumer
import json

class CodegenResponseConsumer:
    """Consumer for code generation response events."""

    def __init__(self, bootstrap_servers: str = "localhost:29092"):
        self.bootstrap_servers = bootstrap_servers
        self._consumer: AIOKafkaConsumer | None = None

    async def initialize(self) -> None:
        """Initialize Kafka consumer."""
        self._consumer = AIOKafkaConsumer(
            "omninode_codegen_response_analyze_v1",
            "omninode_codegen_response_validate_v1",
            "omninode_codegen_response_pattern_v1",
            "omninode_codegen_response_mixin_v1",
            bootstrap_servers=self.bootstrap_servers,
            group_id="omniclaude_codegen_consumer",
            auto_offset_reset="latest",  # Start from latest on first run
            enable_auto_commit=True,  # Auto-commit offsets
            session_timeout_ms=30000,  # 30 seconds
            heartbeat_interval_ms=10000  # 10 seconds
        )
        await self._consumer.start()
        logger.info("Kafka consumer initialized",
                   group_id="omniclaude_codegen_consumer")

    async def consume_events(self) -> None:
        """Consume and process events."""
        try:
            async for message in self._consumer:
                # Deserialize envelope
                envelope_data = json.loads(message.value.decode('utf-8'))
                envelope = OnexEnvelopeV1.model_validate(envelope_data)

                # Route by event type
                await self._handle_event(envelope)

        except Exception as e:
            logger.error("Consumer error", error=str(e))
            raise

    async def _handle_event(self, envelope: OnexEnvelopeV1) -> None:
        """Handle event by type."""
        event_handlers = {
            "CodegenAnalysisResponse": self._handle_analysis_response,
            "CodegenValidationResponse": self._handle_validation_response,
            "CodegenPatternResponse": self._handle_pattern_response,
            "CodegenMixinResponse": self._handle_mixin_response,
        }

        handler = event_handlers.get(envelope.event_type)
        if handler:
            await handler(envelope)
        else:
            logger.warning("Unknown event type", event_type=envelope.event_type)

    async def _handle_analysis_response(self, envelope: OnexEnvelopeV1) -> None:
        """Handle analysis response event."""
        response = CodegenAnalysisResponse.model_validate(envelope.payload)

        logger.info("Analysis response received",
                   correlation_id=str(response.correlation_id),
                   confidence=response.confidence)

        # Process response (e.g., update UI, trigger next step)
        await self._process_analysis_result(response)
```

### Consumer with Manual Offset Management

```python
class ManualOffsetConsumer:
    """Consumer with manual offset management for critical processing."""

    async def consume_with_manual_commits(self) -> None:
        """Consume events with manual offset commits."""
        # Initialize consumer with auto-commit disabled
        consumer = AIOKafkaConsumer(
            "omninode_codegen_request_analyze_v1",
            bootstrap_servers="localhost:29092",
            group_id="omniarchon_codegen_intelligence",
            enable_auto_commit=False,  # Manual commit
            auto_offset_reset="latest"
        )

        await consumer.start()

        try:
            async for message in consumer:
                try:
                    # Deserialize and process event
                    envelope_data = json.loads(message.value.decode('utf-8'))
                    envelope = OnexEnvelopeV1.model_validate(envelope_data)

                    # Process event
                    await self._process_event(envelope)

                    # Manually commit offset AFTER successful processing
                    await consumer.commit()

                    logger.info("Event processed and committed",
                               partition=message.partition,
                               offset=message.offset)

                except Exception as e:
                    # Don't commit offset on error - allows retry
                    logger.error("Event processing failed, will retry",
                                partition=message.partition,
                                offset=message.offset,
                                error=str(e))

                    # Optional: Send to DLQ after N retries
                    await self._send_to_dlq(envelope, error=str(e))

        finally:
            await consumer.stop()
```

### Parallel Consumer

```python
import asyncio

class ParallelConsumer:
    """Consumer with parallel event processing."""

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def consume_with_parallelism(self) -> None:
        """Consume events with parallel processing."""
        consumer = AIOKafkaConsumer(
            "omninode_codegen_request_analyze_v1",
            bootstrap_servers="localhost:29092",
            group_id="omniarchon_parallel_consumer",
            enable_auto_commit=False
        )

        await consumer.start()

        try:
            async for message in consumer:
                # Process events in parallel with semaphore control
                asyncio.create_task(
                    self._process_with_semaphore(message, consumer)
                )

        finally:
            await consumer.stop()

    async def _process_with_semaphore(
        self,
        message,
        consumer: AIOKafkaConsumer
    ) -> None:
        """Process event with semaphore control."""
        async with self._semaphore:  # Limit concurrent processing
            try:
                # Deserialize event
                envelope_data = json.loads(message.value.decode('utf-8'))
                envelope = OnexEnvelopeV1.model_validate(envelope_data)

                # Process event (parallel execution)
                await self._process_event(envelope)

                # Commit offset
                await consumer.commit({
                    TopicPartition(message.topic, message.partition):
                        OffsetAndMetadata(message.offset + 1, "")
                })

            except Exception as e:
                logger.error("Parallel processing error",
                            partition=message.partition,
                            offset=message.offset,
                            error=str(e))
```

### Consumer Group Best Practices

1. **Partition Assignment**: Let Kafka auto-assign partitions for load balancing
   - Kafka rebalances partitions when consumers join/leave the group
   - Each partition is consumed by exactly one consumer in the group

2. **Offset Management**: Use auto-commit for simple cases, manual commit for at-least-once semantics
   - Auto-commit: Convenient but may lose messages on crashes
   - Manual commit: More control but requires careful error handling

3. **Session Timeout**: Set to 30s for responsive failure detection
   - Too short: False positives on slow processing
   - Too long: Slow detection of actual failures

4. **Heartbeat Interval**: Set to 10s (1/3 of session timeout)
   - Ensures timely liveness detection
   - Prevents false positive rebalances

5. **Max Poll Interval**: Set based on maximum processing time (5-10 minutes)
   - Must be longer than longest expected processing time
   - Prevents rebalancing during legitimate long-running operations

6. **Graceful Shutdown**: Always call `consumer.stop()` to commit offsets
   - Ensures no message loss on shutdown
   - Triggers immediate rebalance for remaining consumers

7. **Error Handling**: Catch exceptions, publish to DLQ, continue processing
   - Don't let single message failure stop entire consumer
   - Track failure metrics for monitoring

---

## Event Tracing

### Database-Backed Tracing

**File**: `src/omninode_bridge/dashboard/codegen_event_tracer.py`

```python
class CodegenEventTracer:
    """
    Database-backed event tracing for debugging and monitoring.

    Features:
    - Store all events in PostgreSQL
    - Query by correlation_id, session_id, event_type
    - Performance metrics and analytics
    """

    def __init__(self, db_client: PostgresConnectionManager):
        self.db_client = db_client

    async def trace_event(
        self,
        envelope: OnexEnvelopeV1,
        topic: str,
        partition: int,
        offset: int
    ) -> None:
        """Store event in database for tracing."""
        async with self.db_client.transaction() as tx:
            await tx.execute(
                """
                INSERT INTO event_logs (
                    event_id,
                    correlation_id,
                    session_id,
                    event_type,
                    topic,
                    partition,
                    offset,
                    payload,
                    timestamp,
                    created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                uuid4(),
                envelope.correlation_id,
                envelope.session_id,
                envelope.event_type,
                topic,
                partition,
                offset,
                json.dumps(envelope.payload),
                envelope.timestamp,
                datetime.utcnow()
            )

    async def get_events_by_correlation(
        self,
        correlation_id: UUID
    ) -> list[dict]:
        """Retrieve all events for a correlation ID."""
        async with self.db_client.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    event_id,
                    correlation_id,
                    session_id,
                    event_type,
                    topic,
                    partition,
                    offset,
                    payload,
                    timestamp,
                    created_at
                FROM event_logs
                WHERE correlation_id = $1
                ORDER BY timestamp ASC
                """,
                correlation_id
            )

            return [dict(row) for row in rows]

    async def get_session_timeline(
        self,
        session_id: UUID
    ) -> list[dict]:
        """Get complete timeline of events for a session."""
        async with self.db_client.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    event_type,
                    correlation_id,
                    timestamp,
                    payload->>'message' as message,
                    payload->>'status' as status
                FROM event_logs
                WHERE session_id = $1
                ORDER BY timestamp ASC
                """,
                session_id
            )

            return [dict(row) for row in rows]
```

### Querying Event History

```python
# Get all events for a specific request/response
correlation_id = UUID("550e8400-e29b-41d4-a716-446655440000")
events = await tracer.get_events_by_correlation(correlation_id)

for event in events:
    print(f"{event['timestamp']}: {event['event_type']}")
    # 2025-10-16 12:34:56: CodegenAnalysisRequest
    # 2025-10-16 12:34:57: CodegenStatusEvent
    # 2025-10-16 12:35:02: CodegenAnalysisResponse

# Get session timeline
session_id = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
timeline = await tracer.get_session_timeline(session_id)

for event in timeline:
    print(f"{event['timestamp']}: {event['message']} ({event['status']})")
    # 2025-10-16 12:34:56: Analysis started (processing)
    # 2025-10-16 12:34:58: Extracting requirements (processing)
    # 2025-10-16 12:35:02: Analysis complete (completed)
```

---

## DLQ Monitoring

### Dead Letter Queue (DLQ) Monitoring

**File**: `src/omninode_bridge/events/dlq_monitor.py`

```python
class DLQMonitor:
    """
    Monitor Dead Letter Queue topics for failed events.

    Features:
    - Threshold-based alerting (>10 failures → alert)
    - Real-time monitoring of all DLQ topics
    - Automatic retry logic for transient failures
    - Event analysis and reporting
    """

    DLQ_TOPICS = [
        "omninode_codegen_dlq_analyze_v1",
        "omninode_codegen_dlq_validate_v1",
        "omninode_codegen_dlq_pattern_v1",
        "omninode_codegen_dlq_mixin_v1",
    ]

    ALERT_THRESHOLD = 10  # Alert if >10 failed events

    async def monitor_dlqs(self) -> dict[str, int]:
        """
        Monitor all DLQ topics and trigger alerts if thresholds exceeded.

        Returns:
            Dict mapping DLQ topic to failure count
        """
        dlq_counts = {}

        for dlq_topic in self.DLQ_TOPICS:
            # Get message count from Kafka
            count = await self._get_topic_message_count(dlq_topic)
            dlq_counts[dlq_topic] = count

            # Trigger alert if threshold exceeded
            if count > self.ALERT_THRESHOLD:
                await self._trigger_alert(dlq_topic, count)

        return dlq_counts

    async def _trigger_alert(self, dlq_topic: str, count: int) -> None:
        """Send alert for DLQ threshold breach."""
        logger.error("DLQ threshold exceeded",
                    dlq_topic=dlq_topic,
                    count=count,
                    threshold=self.ALERT_THRESHOLD)

        # Publish alert event
        await self.event_producer.publish_event(
            topic="omninode_system_alerts",
            event=DLQAlertEvent(
                dlq_topic=dlq_topic,
                failure_count=count,
                threshold=self.ALERT_THRESHOLD,
                timestamp=datetime.utcnow()
            ),
            correlation_id=uuid4()
        )
```

### DLQ Event Format

When events fail and are sent to DLQ topics, they include the original event plus error context:

```python
dlq_event = {
    "original_event": {
        # Original event payload
        "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
        "session_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
        # ... original fields
    },
    "error": {
        "error_type": "ValidationError",
        "error_message": "Invalid node_type: 'invalid'",
        "stack_trace": "Traceback (most recent call last):\n  ...",
        "timestamp": "2025-10-29T10:30:00Z"
    },
    "retry_count": 3,
    "max_retries": 3,
    "dlq_timestamp": "2025-10-29T10:30:05Z",
    "source_topic": "omninode_codegen_request_analyze_v1",
    "consumer_group": "omniarchon_codegen_intelligence"
}
```

**Key Fields**:
- `original_event`: Complete original event for replay
- `error`: Full error context including stack trace
- `retry_count`: Number of retry attempts before DLQ
- `dlq_timestamp`: When event was sent to DLQ
- `source_topic`: Original topic for replay routing

### DLQ Analysis

```bash
# List DLQ topics and message counts
for topic in $(rpk topic list | grep dlq); do
  count=$(rpk topic describe $topic | grep "partition 0" | awk '{print $3}')
  echo "$topic: $count messages"
done

# Consume DLQ messages for analysis
rpk topic consume omninode_codegen_dlq_analyze_v1 --num 10 --format json | jq

# Example DLQ message
{
  "envelope_version": "1.0",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "CodegenAnalysisRequest",
  "timestamp": "2025-10-16T12:34:56.789Z",
  "payload": {...},
  "error": "Connection timeout",
  "retry_count": 3,
  "dlq_timestamp": "2025-10-16T12:35:10.123Z"
}
```

### Retry Strategy

```python
class DLQRetryHandler:
    """Retry handler for DLQ messages."""

    async def retry_dlq_messages(self, dlq_topic: str) -> int:
        """
        Retry messages from DLQ topic.

        Args:
            dlq_topic: DLQ topic name

        Returns:
            Number of successfully retried messages
        """
        success_count = 0

        # Consume from DLQ
        consumer = AIOKafkaConsumer(
            dlq_topic,
            bootstrap_servers="localhost:29092",
            group_id="dlq_retry_handler",
            auto_offset_reset="earliest"
        )

        await consumer.start()

        try:
            async for message in consumer:
                # Deserialize DLQ message
                dlq_data = json.loads(message.value.decode('utf-8'))

                # Check retry count
                retry_count = dlq_data.get('retry_count', 0)

                if retry_count >= 5:
                    logger.warning("Max retries exceeded, skipping",
                                  correlation_id=dlq_data['correlation_id'])
                    continue

                # Retry: Publish back to original topic
                original_topic = dlq_topic.replace('_dlq', '')

                try:
                    await self.producer.publish_event(
                        topic=original_topic,
                        event=self._reconstruct_event(dlq_data),
                        correlation_id=UUID(dlq_data['correlation_id'])
                    )

                    success_count += 1
                    logger.info("DLQ message retried successfully",
                               correlation_id=dlq_data['correlation_id'])

                except Exception as e:
                    logger.error("DLQ retry failed",
                                correlation_id=dlq_data['correlation_id'],
                                error=str(e))

        finally:
            await consumer.stop()

        return success_count
```

---

## Operations Guide

### Starting Event Infrastructure

```bash
# 1. Start Redpanda
docker-compose up -d redpanda

# 2. Wait for Redpanda to be healthy
docker-compose ps redpanda  # Check status

# 3. Create topics (if not auto-created)
docker exec omninode-bridge-redpanda rpk topic create \
  omninode_codegen_request_analyze_v1 --partitions 3

# 4. Verify topics exist
docker exec omninode-bridge-redpanda rpk topic list

# 5. Start application services
PYTHONPATH=src poetry run uvicorn metadata_stamping.main:app --reload
```

### Monitoring Topics

```bash
# List all topics
rpk topic list

# Describe topic (partitions, replicas, config)
rpk topic describe omninode_codegen_request_analyze_v1

# Get topic statistics
rpk topic describe-storage omninode_codegen_request_analyze_v1

# Monitor consumer lag
rpk group describe omniclaude_codegen_consumer

# View recent messages
rpk topic consume omninode_codegen_status_session_v1 --num 5 --format json
```

### Managing Consumer Groups

```bash
# List consumer groups
rpk group list

# Describe consumer group offsets
rpk group describe omniclaude_codegen_consumer

# Check consumer lag (read-only)
rpk group describe omniclaude_codegen_consumer

# Reset consumer group (BE CAREFUL)
rpk group seek omniclaude_codegen_consumer --to start  # Replay all messages
```

### Troubleshooting Kafka Issues

```bash
# Check Redpanda logs
docker logs omninode-bridge-redpanda --tail 100 --follow

# Check cluster health
docker exec omninode-bridge-redpanda rpk cluster info

# Test connectivity
docker exec omninode-bridge-redpanda rpk cluster health

# Delete topic (BE CAREFUL - data loss)
rpk topic delete omninode_codegen_request_analyze_v1
```

---

## Performance Tuning

### Producer Tuning

```python
# High-throughput producer configuration
producer = AIOKafkaProducer(
    bootstrap_servers="localhost:29092",
    compression_type="gzip",  # Compress messages (gzip, snappy, lz4)
    acks=1,  # Wait for leader (1), all replicas (all), or none (0)
    max_request_size=1048576,  # 1MB max message size
    batch_size=16384,  # 16KB batch size
    linger_ms=10,  # Wait up to 10ms for batching
    buffer_memory=33554432,  # 32MB buffer
    max_in_flight_requests_per_connection=5  # Pipeline requests
)

# Benefits:
# - Batching: Reduces network round-trips (10-50% improvement)
# - Compression: Reduces network bandwidth (20-80% improvement)
# - Pipelining: Parallelizes requests (30-100% improvement)
```

### Consumer Tuning

```python
# High-throughput consumer configuration
consumer = AIOKafkaConsumer(
    "omninode_codegen_request_analyze_v1",
    bootstrap_servers="localhost:29092",
    group_id="omniarchon_codegen_intelligence",
    fetch_min_bytes=1024,  # 1KB minimum fetch (reduce requests)
    fetch_max_wait_ms=500,  # Wait up to 500ms for fetch_min_bytes
    max_partition_fetch_bytes=1048576,  # 1MB per partition
    max_poll_records=500,  # Process 500 records per poll
    session_timeout_ms=30000,  # 30 seconds session timeout
    heartbeat_interval_ms=10000,  # 10 seconds heartbeat
)

# Benefits:
# - Larger fetch: Reduces polling overhead (20-40% improvement)
# - Batching: Process multiple records together (30-60% improvement)
```

### Performance Metrics

```python
# Monitor producer metrics
producer_metrics = {
    "request_rate": 150,  # requests/second
    "request_latency_avg_ms": 12,  # average latency
    "request_latency_p99_ms": 45,  # p99 latency
    "batch_size_avg": 14500,  # average batch size
    "compression_rate_avg": 0.65,  # compression ratio
    "buffer_available_bytes": 30000000,  # available buffer
}

# Monitor consumer metrics
consumer_metrics = {
    "records_consumed_rate": 1200,  # records/second
    "fetch_rate": 25,  # fetches/second
    "fetch_latency_avg_ms": 20,  # average fetch latency
    "records_lag_max": 150,  # maximum lag
}
```

---

## Performance Characteristics

### Throughput & Latency Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Producer latency (p95)** | <100ms | End-to-end publish time |
| **Consumer latency (p95)** | <50ms | Message consumption time |
| **Processing time (avg)** | <2s | Request → Response time |
| **Processing time (p95)** | <5s | 95th percentile processing |
| **Throughput** | 1000+ msg/s | Per topic throughput |
| **DLQ rate** | <1% | Percentage of failed events |

### Topic Performance Breakdown

**Request/Response Topics** (3 partitions each):
- **Throughput**: ~3000 messages/sec (1000 per partition)
- **Latency**: <100ms p95 (producer) + <50ms p95 (consumer)
- **Storage**: ~7 days retention × average message rate
- **Message Size**: Average 2-5KB per message

**Status Topic** (6 partitions):
- **Throughput**: ~6000 messages/sec (1000 per partition)
- **Latency**: <50ms p95 (status updates are non-critical)
- **Storage**: 3 days retention (shorter for transient status)
- **Partition Strategy**: Partition by `session_id` for event ordering

**DLQ Topics** (1 partition each):
- **Throughput**: <10 messages/sec (minimal failures expected)
- **Latency**: Not critical (failure handling is async)
- **Storage**: 30 days retention (forensic analysis)
- **Alert Threshold**: >10 messages triggers alert

### Resource Requirements

**Redpanda Cluster**:
- **Memory**: 1GB minimum (configured in docker-compose)
- **CPU**: 1 core minimum (single-broker deployment)
- **Disk**: 10GB minimum (for 7-day retention)
- **Network**: 100Mbps minimum (for throughput targets)
- **IOPS**: 1000+ IOPS for disk writes

**Consumer Instances**:
- **Memory**: 256MB per consumer instance
- **CPU**: 0.5 cores per consumer instance
- **Network**: 50Mbps per consumer instance
- **Concurrency**: 10-100 concurrent message processing

### Scaling Considerations

**Horizontal Scaling**:
- **Add consumer instances** to consumer groups (auto-rebalancing)
  - Increases throughput proportionally
  - No downtime required
  - Limited by partition count (max 1 consumer per partition)

- **Increase partition count** for higher throughput
  - Requires topic recreation (with data migration)
  - Enables more parallel consumers
  - Consider initial over-provisioning

- **Add broker nodes** for higher availability (production)
  - Increases replication capacity
  - Improves fault tolerance
  - Requires replication factor adjustment

**Vertical Scaling**:
- **Increase Redpanda memory** allocation for larger message buffers
  - Improves batching efficiency
  - Reduces disk I/O pressure
  - Recommended for high-throughput scenarios

- **Increase CPU cores** for better compression performance
  - gzip compression is CPU-intensive
  - Parallel partition processing benefits

- **Increase disk IOPS** for faster log writes
  - SSD recommended for production
  - Consider RAID configurations
  - Monitor disk utilization metrics

---

## Best Practices

### Event Publishing

1. **Always set correlation_id**: Enable request/response matching and distributed tracing
   - Use same correlation_id for related events (request → response)
   - Generate new UUID for each independent operation
   - Include in all log messages for correlation

2. **Include session_id**: Track events across entire code generation session
   - Enables session-level debugging and analytics
   - Required for session timeline reconstruction
   - Optional but highly recommended for multi-step workflows

3. **Use schema_version**: Support backward-compatible schema evolution
   - Include in all event schemas
   - Increment for breaking changes
   - Document version differences

4. **Validate before publishing**: Catch errors early with Pydantic validation
   - Use `model_validate()` to validate payloads
   - Handle ValidationError gracefully
   - Log validation failures for debugging

5. **Handle publish failures**: Implement retry logic with exponential backoff
   - Retry transient failures (network, broker unavailable)
   - Use DLQ for permanent failures (validation, schema errors)
   - Set max retries limit (3-5 attempts recommended)

6. **Monitor publish latency**: Alert if p95 latency exceeds 100ms
   - Track per-topic latency metrics
   - Investigate spikes in publish time
   - Consider producer configuration tuning

### Event Consumption

1. **Idempotent processing**: Design consumers to handle duplicate messages
   - Kafka guarantees at-least-once delivery
   - Use correlation_id or message_id for deduplication
   - Store processed message IDs in cache or database

2. **Graceful error handling**: Use DLQ pattern for failed events
   - Catch all exceptions during processing
   - Log error context (correlation_id, stack trace)
   - Publish to DLQ after max retries exhausted

3. **Commit offsets carefully**: Use auto-commit for simple cases, manual for critical
   - Auto-commit: Simple but may lose messages on crash
   - Manual commit: More control, commit only after successful processing
   - Never commit before processing completes

4. **Monitor consumer lag**: Alert if lag exceeds 100 messages
   - Consumer lag = latest offset - committed offset
   - High lag indicates slow processing or consumer failures
   - Scale horizontally if lag persists

5. **Scale horizontally**: Add consumer instances for increased throughput
   - Kafka auto-rebalances partitions across consumers
   - Each partition assigned to exactly one consumer
   - Maximum consumers = partition count

6. **Track processing time**: Identify bottlenecks with metrics
   - Measure time from message consumption to completion
   - Alert if p95 processing time exceeds 5s
   - Profile slow operations for optimization

### Schema Evolution

1. **Add new fields as optional**: Maintain backward compatibility
   - Use `Field(default=...)` or `Field(default_factory=...)`
   - Old consumers ignore unknown fields
   - New consumers provide defaults for missing fields

2. **Never remove existing fields**: Deprecate instead
   - Mark fields as deprecated in docstrings
   - Log warnings when deprecated fields are used
   - Remove only in new major version (v2, v3, etc.)

3. **Version schema_version field**: Enable version-specific handling
   - Check `schema_version` in consumers
   - Handle different versions appropriately
   - Document version differences in schemas

4. **Test compatibility**: Verify old consumers can read new schemas
   - Deploy new producers first (forward compatibility)
   - Test old consumers with new events
   - Deploy new consumers after validation

5. **Document breaking changes**: Communicate schema migrations clearly
   - Update schema documentation
   - Notify all service teams
   - Provide migration timeline

### Monitoring & Operations

1. **Monitor DLQ topics**: Alert on threshold exceeded (>10 messages)
   - DLQ indicates production issues
   - Investigate error patterns immediately
   - Implement fixes and replay messages

2. **Track correlation chains**: Use event tracer for debugging
   - Query events by correlation_id
   - Reconstruct request → response flows
   - Identify missing or delayed events

3. **Log structured data**: Include correlation_id, session_id in all logs
   - Enable log correlation with events
   - Simplify distributed debugging
   - Use structured logging (JSON format)

4. **Alert on high latency**: >5s processing time indicates bottleneck
   - Identify slow operations in processing pipeline
   - Optimize database queries, API calls
   - Consider caching or parallel processing

5. **Review DLQ messages**: Investigate failures weekly
   - Analyze error patterns and root causes
   - Implement fixes for common failures
   - Track failure rate over time

6. **Optimize topic configuration**: Adjust retention, partitions based on usage
   - Monitor disk usage and retention periods
   - Increase partitions for higher throughput
   - Balance partition count with consumer count

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
from uuid import uuid4

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
   - Configure Kafka SSL/TLS certificates
   - Encrypt all broker-to-broker communication
   - Encrypt client-to-broker communication

2. **Authenticate consumers**: Use SASL/SCRAM for authentication
   - Require authentication for all clients
   - Use strong passwords or certificates
   - Rotate credentials regularly

3. **Authorize topic access**: Implement ACLs for topic permissions
   - Grant read/write permissions per topic
   - Restrict admin operations
   - Audit access logs regularly

4. **Encrypt sensitive data**: Use application-level encryption for PII
   - Encrypt sensitive fields in event payloads
   - Store encryption keys securely (e.g., Vault)
   - Consider field-level encryption for compliance

5. **Rotate credentials**: Regularly rotate Kafka credentials
   - Rotate every 90 days (or per policy)
   - Automate rotation with secrets management
   - Test rotation procedures regularly

6. **Audit access logs**: Track who accessed which topics
   - Enable Kafka audit logging
   - Monitor for suspicious access patterns
   - Integrate with SIEM for alerting

---

## Troubleshooting

### Common Issues

#### Issue: Kafka Connection Failures

**Symptom**:
```
Failed to resolve 'omninode-bridge-redpanda'
```

**Solution**:
```bash
# 1. Verify hostname configuration
grep omninode-bridge-redpanda /etc/hosts

# 2. If not found, add it
echo "127.0.0.1 omninode-bridge-redpanda" | sudo tee -a /etc/hosts

# 3. Restart Redpanda
docker-compose restart redpanda

# 4. Test connectivity
docker exec omninode-bridge-redpanda rpk cluster info
```

#### Issue: Consumer Lag

**Symptom**:
```
Consumer lag increasing, messages not processing fast enough
```

**Solution**:
```bash
# 1. Check consumer lag
rpk group describe omniclaude_codegen_consumer

# 2. Increase consumer parallelism
# Add more consumers to the group (scale horizontally)

# 3. Increase partition count (preserves data and offsets)
rpk topic add-partitions omninode_codegen_request_analyze_v1 --num 6

# 4. Optimize consumer code (batch processing, async)
```

#### Issue: DLQ Growing

**Symptom**:
```
DLQ topic message count exceeding threshold
```

**Solution**:
```bash
# 1. Analyze DLQ messages
rpk topic consume omninode_codegen_dlq_analyze_v1 --num 10 --format json | jq

# 2. Identify common error patterns
# Look for repeated error messages

# 3. Fix root cause in code

# 4. Retry DLQ messages
python -m omninode_bridge.events.dlq_retry_handler \
  --topic omninode_codegen_dlq_analyze_v1

# 5. Purge DLQ if messages are no longer relevant (BE CAREFUL)
rpk topic delete omninode_codegen_dlq_analyze_v1
rpk topic create omninode_codegen_dlq_analyze_v1 --partitions 1
```

#### Issue: Message Serialization Errors

**Symptom**:
```
ValidationError: invalid UUID format
```

**Solution**:
```python
# Ensure UUIDs are serialized as strings
envelope = OnexEnvelopeV1(
    correlation_id=correlation_id,  # UUID object
    session_id=session_id,  # UUID object
    ...
)

# Pydantic v2 automatically serializes UUIDs as strings in JSON
message = envelope.model_dump_json()  # UUIDs → strings
```

---

## Quick Reference

### Event Topics

| Topic | Type | Direction | Partitions |
|-------|------|-----------|------------|
| `omninode_codegen_request_analyze_v1` | Request | omniclaude → omniarchon | 3 |
| `omninode_codegen_request_validate_v1` | Request | omniclaude → omniarchon | 3 |
| `omninode_codegen_request_pattern_v1` | Request | omniclaude → omniarchon | 3 |
| `omninode_codegen_request_mixin_v1` | Request | omniclaude → omniarchon | 3 |
| `omninode_codegen_response_analyze_v1` | Response | omniarchon → omniclaude | 3 |
| `omninode_codegen_response_validate_v1` | Response | omniarchon → omniclaude | 3 |
| `omninode_codegen_response_pattern_v1` | Response | omniarchon → omniclaude | 3 |
| `omninode_codegen_response_mixin_v1` | Response | omniarchon → omniclaude | 3 |
| `omninode_codegen_status_session_v1` | Status | Bidirectional | 6 |
| `omninode_codegen_dlq_analyze_v1` | DLQ | Failed events | 1 |
| `omninode_codegen_dlq_validate_v1` | DLQ | Failed events | 1 |
| `omninode_codegen_dlq_pattern_v1` | DLQ | Failed events | 1 |
| `omninode_codegen_dlq_mixin_v1` | DLQ | Failed events | 1 |

### Kafka Commands

```bash
# Topic management
rpk topic create <topic> --partitions 3
rpk topic list
rpk topic describe <topic>
rpk topic delete <topic>

# Consumer group management
rpk group list
rpk group describe <group>
rpk group seek <group> --to start|end

# Message inspection
rpk topic consume <topic> --num 10 --format json
rpk topic produce <topic>  # Interactive producer

# Cluster management
rpk cluster info
rpk cluster health
```

### Python Event Publishing

```python
# Quick publish template
from omninode_bridge.services.kafka_client import KafkaEventProducer
from omninode_bridge.events.codegen_schemas import CodegenAnalysisRequest

producer = KafkaEventProducer()
await producer.initialize()

await producer.publish_event(
    topic="omninode_codegen_request_analyze_v1",
    event=request,
    correlation_id=correlation_id
)
```

---

## Appendix: Topic Creation Script

Complete bash script for creating all 13 codegen topics with proper configuration:

```bash
#!/bin/bash
# Create all 13 codegen topics with proper configuration

BOOTSTRAP_SERVERS="localhost:29092"

echo "Creating code generation event topics..."

# Request topics
echo "Creating request topics..."
rpk topic create omninode_codegen_request_analyze_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 3 --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

rpk topic create omninode_codegen_request_validate_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 3 --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

rpk topic create omninode_codegen_request_pattern_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 3 --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

rpk topic create omninode_codegen_request_mixin_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 3 --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

# Response topics
echo "Creating response topics..."
rpk topic create omninode_codegen_response_analyze_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 3 --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

rpk topic create omninode_codegen_response_validate_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 3 --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

rpk topic create omninode_codegen_response_pattern_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 3 --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

rpk topic create omninode_codegen_response_mixin_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 3 --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

# Status topic
echo "Creating status topic..."
rpk topic create omninode_codegen_status_session_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 6 --replicas 1 \
  --topic-config retention.ms=259200000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

# DLQ topics
echo "Creating DLQ topics..."
rpk topic create omninode_codegen_dlq_analyze_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 1 --replicas 1 \
  --topic-config retention.ms=2592000000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

rpk topic create omninode_codegen_dlq_validate_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 1 --replicas 1 \
  --topic-config retention.ms=2592000000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

rpk topic create omninode_codegen_dlq_pattern_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 1 --replicas 1 \
  --topic-config retention.ms=2592000000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

rpk topic create omninode_codegen_dlq_mixin_v1 \
  --brokers $BOOTSTRAP_SERVERS \
  --partitions 1 --replicas 1 \
  --topic-config retention.ms=2592000000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

echo "All 13 codegen topics created successfully!"
echo ""
echo "Verifying topics..."
rpk topic list --brokers $BOOTSTRAP_SERVERS | grep omninode_codegen
```

**Usage**:
```bash
# Make script executable
chmod +x create_codegen_topics.sh

# Run script
./create_codegen_topics.sh
```

**Configuration Details**:
- **Request/Response Topics**: 3 partitions, 7 days retention (604800000ms)
- **Status Topic**: 6 partitions, 3 days retention (259200000ms)
- **DLQ Topics**: 1 partition, 30 days retention (2592000000ms)
- **All Topics**: gzip compression, delete cleanup policy, 1 replica

---

## Related Documentation

- **[Architecture Guide](../architecture/ARCHITECTURE.md)** - Event-driven architecture overview
- **[API Reference](../api/API_REFERENCE.md)** - Complete API documentation
- **[Bridge Nodes Guide](../guides/BRIDGE_NODES_GUIDE.md)** - Bridge node integration
- **[Operations Guide](../operations/OPERATIONS_GUIDE.md)** 🚧 *(Planned)* - Deployment and monitoring
- **[Getting Started](../GETTING_STARTED.md)** - Quick start guide

---

**Maintained By**: omninode_bridge team
**Last Updated**: October 29, 2025
**Document Version**: 2.1.0
**Status**: Consolidated with EVENT_INFRASTRUCTURE_GUIDE + EVENT_SCHEMAS (Oct 2025)
**Next Review**: November 29, 2025
