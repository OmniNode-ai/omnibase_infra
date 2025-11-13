# Kafka Event Schema Registry - Complete Reference

**Date**: October 2025
**Status**: Phase 3 Complete - Comprehensive Documentation
**Purpose**: Complete schema documentation with producer/consumer mapping for all 37+ Kafka topics

---

## Table of Contents

1. [Overview](#overview)
2. [Topic Catalog](#topic-catalog)
3. [Schema Definitions](#schema-definitions)
4. [Producer/Consumer Matrix](#producerconsumer-matrix)
5. [Event Flow Diagrams](#event-flow-diagrams)
6. [Consumer Groups](#consumer-groups)
7. [Quick Reference](#quick-reference)

---

## Overview

This schema registry provides a complete reference for all Kafka topics in the omninode_bridge ecosystem, including:
- **37+ Topics** across 4 categories
- **9 Event Schemas** with full Pydantic v2 definitions
- **Complete Producer/Consumer Mapping** for all topics
- **Consumer Group Configuration** for all services
- **Event Flow Diagrams** for major workflows

---

## Topic Catalog

### Summary by Category

| Category | Topics | OnexEnvelopeV1 | Documentation | Status |
|----------|--------|----------------|---------------|--------|
| Codegen Topics | 13 | ❌ → ✅ | ✅ Complete | Migration Required |
| Bridge Orchestrator | 12 | ✅ | ✅ Complete | Production Ready |
| Bridge Reducer | 7 | ✅ | ✅ Complete | Production Ready |
| Service Lifecycle | 5 | ❌ → ✅ | ⚠️ Partial | Migration Required |
| **Total** | **37** | **51% → 100%** | **86% → 100%** | **In Progress** |

---

## Schema Definitions

### 1. Codegen Event Schemas (9 schemas)

**Location**: `src/omninode_bridge/events/codegen_schemas.py`

#### CodegenAnalysisRequest

**Purpose**: Request PRD analysis from omniarchon intelligence
**Topic**: `omninode_codegen_request_analyze_v1`
**Flow**: omniclaude → omniarchon

```python
class CodegenAnalysisRequest(BaseModel):
    """PRD analysis request schema."""
    correlation_id: UUID = Field(..., description="Request correlation ID")
    session_id: UUID = Field(..., description="Code generation session ID")
    prd_content: str = Field(..., description="Raw PRD markdown content")
    analysis_type: EnumAnalysisType = Field(
        default=EnumAnalysisType.FULL,
        description="full|partial|quick"
    )
    workspace_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Workspace context (file paths, metadata)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Request timestamp (UTC)"
    )
    schema_version: str = Field(default="1.0", description="Schema version")
```

**Kafka Topic Configuration**:
- Partitions: 3
- Replication: 1
- Retention: 7 days
- Compression: gzip

**Producers**: omniclaude
**Consumers**: omniarchon (consumer group: `omniarchon_codegen_intelligence`)

#### CodegenAnalysisResponse

**Purpose**: Return PRD analysis results
**Topic**: `omninode_codegen_response_analyze_v1`
**Flow**: omniarchon → omniclaude

```python
class CodegenAnalysisResponse(BaseModel):
    """PRD analysis response schema."""
    correlation_id: UUID = Field(..., description="Request correlation ID")
    session_id: UUID = Field(..., description="Session ID")
    analysis_result: dict[str, Any] = Field(
        ...,
        description="Requirements, architecture, dependencies"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    processing_time_ms: int = Field(..., description="Processing time")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = Field(default="1.0")
```

**Producers**: omniarchon
**Consumers**: omniclaude (consumer group: `omniclaude_codegen_consumer`)

#### CodegenValidationRequest

**Purpose**: Request code validation for ONEX compliance
**Topic**: `omninode_codegen_request_validate_v1`
**Flow**: omniclaude → omniarchon

```python
class CodegenValidationRequest(BaseModel):
    """Code validation request schema."""
    correlation_id: UUID
    session_id: UUID
    code_content: str = Field(..., description="Generated code to validate")
    node_type: EnumNodeType = Field(
        ...,
        description="effect|compute|reducer|orchestrator"
    )
    contracts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Associated contracts"
    )
    validation_type: EnumValidationType = Field(
        default=EnumValidationType.FULL,
        description="full|syntax|semantic|compliance"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = Field(default="1.0")
```

**Producers**: omniclaude
**Consumers**: omniarchon

#### CodegenValidationResponse

**Purpose**: Return code validation results with quality scores
**Topic**: `omninode_codegen_response_validate_v1`
**Flow**: omniarchon → omniclaude

```python
class CodegenValidationResponse(BaseModel):
    """Code validation response schema."""
    correlation_id: UUID
    session_id: UUID
    validation_result: dict[str, Any] = Field(
        ...,
        description="Errors, warnings, suggestions"
    )
    quality_score: float = Field(..., ge=0.0, le=1.0)
    onex_compliance_score: float = Field(..., ge=0.0, le=1.0)
    is_valid: bool = Field(..., description="Pass all validation checks")
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = Field(default="1.0")
```

**Producers**: omniarchon
**Consumers**: omniclaude

#### CodegenPatternRequest

**Purpose**: Request pattern matching for similar nodes
**Topic**: `omninode_codegen_request_pattern_v1`
**Flow**: omniclaude → omniarchon

```python
class CodegenPatternRequest(BaseModel):
    """Pattern matching request schema."""
    correlation_id: UUID
    session_id: UUID
    node_description: str = Field(..., description="Desired node functionality")
    node_type: EnumNodeType
    limit: int = Field(default=5, ge=1, le=20, description="Max results")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = Field(default="1.0")
```

**Producers**: omniclaude
**Consumers**: omniarchon

#### CodegenPatternResponse

**Purpose**: Return similar node patterns
**Topic**: `omninode_codegen_response_pattern_v1`
**Flow**: omniarchon → omniclaude

```python
class CodegenPatternResponse(BaseModel):
    """Pattern matching response schema."""
    correlation_id: UUID
    session_id: UUID
    pattern_result: list[dict[str, Any]] = Field(
        ...,
        description="Similar nodes with scores and details"
    )
    total_matches: int = Field(..., description="Total matches before limit")
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = Field(default="1.0")
```

**Producers**: omniarchon
**Consumers**: omniclaude

#### CodegenMixinRequest

**Purpose**: Request mixin recommendations
**Topic**: `omninode_codegen_request_mixin_v1`
**Flow**: omniclaude → omniarchon

```python
class CodegenMixinRequest(BaseModel):
    """Mixin recommendation request schema."""
    correlation_id: UUID
    session_id: UUID
    requirements: list[str] = Field(..., description="Functional requirements")
    node_type: EnumNodeType
    existing_mixins: list[str] = Field(
        default_factory=list,
        description="Already selected mixins"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = Field(default="1.0")
```

**Producers**: omniclaude
**Consumers**: omniarchon

#### CodegenMixinResponse

**Purpose**: Return mixin recommendations
**Topic**: `omninode_codegen_response_mixin_v1`
**Flow**: omniarchon → omniclaude

```python
class CodegenMixinResponse(BaseModel):
    """Mixin recommendation response schema."""
    correlation_id: UUID
    session_id: UUID
    mixin_recommendations: list[dict[str, Any]] = Field(
        ...,
        description="Recommended mixins with rationale"
    )
    total_recommendations: int
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = Field(default="1.0")
```

**Producers**: omniarchon
**Consumers**: omniclaude

#### CodegenStatusEvent

**Purpose**: Real-time session status updates
**Topic**: `omninode_codegen_status_session_v1`
**Flow**: Both omniclaude and omniarchon → monitoring

```python
class CodegenStatusEvent(BaseModel):
    """Code generation status update schema."""
    session_id: UUID
    status: EnumSessionStatus = Field(
        ...,
        description="pending|processing|completed|failed|cancelled"
    )
    progress_percentage: float = Field(..., ge=0.0, le=100.0)
    message: str = Field(..., description="Human-readable status")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Current step, errors, etc."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = Field(default="1.0")
```

**Producers**: omniclaude, omniarchon
**Consumers**: Monitoring systems, omniclaude (for remote status), omniarchon (for remote status)

---

### 2. Bridge Orchestrator Event Schemas

**Location**: `src/omninode_bridge/nodes/orchestrator/v1_0_0/models/enum_workflow_event.py`

#### EnumWorkflowEvent

All orchestrator events use OnexEnvelopeV1 wrapper with these event types:

```python
class EnumWorkflowEvent(str, Enum):
    """Workflow orchestration event types."""

    # Workflow Lifecycle (9 events)
    WORKFLOW_STARTED = "stamp_workflow_started"
    WORKFLOW_COMPLETED = "stamp_workflow_completed"
    WORKFLOW_FAILED = "stamp_workflow_failed"
    STEP_COMPLETED = "workflow_step_completed"
    INTELLIGENCE_REQUESTED = "onextree_intelligence_requested"
    INTELLIGENCE_RECEIVED = "onextree_intelligence_received"
    STAMP_CREATED = "metadata_stamp_created"
    HASH_GENERATED = "blake3_hash_generated"
    STATE_TRANSITION = "workflow_state_transition"

    # Node Introspection (3 events)
    NODE_INTROSPECTION = "node_introspection"
    REGISTRY_REQUEST_INTROSPECTION = "registry_request_introspection"
    NODE_HEARTBEAT = "node_heartbeat"
```

**Envelope Payload Examples**:

```python
# WORKFLOW_STARTED payload
{
    "workflow_id": "uuid",
    "status": "processing",
    "node_id": "orchestrator-instance-1",
    "timestamp": "2025-10-15T12:00:00Z"
}

# INTELLIGENCE_REQUESTED payload
{
    "workflow_id": "uuid",
    "content": "file content to analyze",
    "file_path": "/path/to/file.py",
    "analysis_type": "quality_assessment"
}

# NODE_INTROSPECTION payload
{
    "node_id": "orchestrator-001",
    "node_type": "orchestrator",
    "capabilities": {...},
    "endpoints": {...},
    "performance_profile": {...}
}
```

**Producers**: NodeBridgeOrchestrator
**Consumers**:
- Registry (for introspection events)
- Analytics (for workflow events)
- Storage (for stamp/hash events)
- OnexTree (for intelligence requests)

---

### 3. Bridge Reducer Event Schemas

**Location**: `src/omninode_bridge/nodes/reducer/v1_0_0/models/enum_reducer_event.py`

#### EnumReducerEvent

All reducer events use OnexEnvelopeV1 wrapper with these event types:

```python
class EnumReducerEvent(str, Enum):
    """Aggregation and state reduction event types."""

    # Aggregation Lifecycle (5 events)
    AGGREGATION_STARTED = "aggregation_started"
    BATCH_PROCESSED = "batch_processed"
    STATE_PERSISTED = "state_persisted"
    AGGREGATION_COMPLETED = "aggregation_completed"
    AGGREGATION_FAILED = "aggregation_failed"

    # FSM Management (2 events)
    FSM_STATE_INITIALIZED = "fsm_state_initialized"
    FSM_STATE_TRANSITIONED = "fsm_state_transitioned"

    # Node Introspection (3 events - shared with orchestrator)
    NODE_INTROSPECTION = "node_introspection"
    REGISTRY_REQUEST_INTROSPECTION = "registry_request_introspection"
    NODE_HEARTBEAT = "node_heartbeat"
```

**Envelope Payload Examples**:

```python
# AGGREGATION_STARTED payload
{
    "aggregation_id": "uuid",
    "aggregation_type": "NAMESPACE_GROUPING",
    "total_items": 1000
}

# STATE_PERSISTED payload
{
    "aggregation_id": "uuid",
    "namespaces_count": 5,
    "total_items_persisted": 1000
}
```

**Producers**: NodeBridgeReducer
**Consumers**:
- Analytics (for aggregation events)
- Storage (for persistence events)
- Registry (for FSM and introspection events)

---

### 4. Service Lifecycle Event Schemas

**Location**: `src/omninode_bridge/models/events.py`

#### ServiceLifecycleEvent

```python
class ServiceLifecycleEvent(BaseEvent):
    """Service lifecycle event schema."""
    type: Literal[EventType.SERVICE_LIFECYCLE] = EventType.SERVICE_LIFECYCLE
    event: ServiceEventType  # startup|shutdown|health_check|etc.

    service_version: str | None
    environment: str | None
    instance_id: str | None
    health_status: str | None
    dependencies: dict[str, str] | None
```

**Event Types**: STARTUP, SHUTDOWN, HEALTH_CHECK, REGISTRATION, DEREGISTRATION, READY, ERROR

**Producers**: All services (Orchestrator, Reducer, Registry, MetadataStamping)
**Consumers**: Monitoring systems, Health monitors

---

## Producer/Consumer Matrix

### Complete Mapping Table

| Topic | Producer(s) | Consumer(s) | Consumer Group(s) | Partitions | Retention |
|-------|-------------|-------------|-------------------|------------|-----------|
| **Codegen Request Topics** |
| `omninode_codegen_request_analyze_v1` | omniclaude | omniarchon | `omniarchon_codegen_intelligence` | 3 | 7d |
| `omninode_codegen_request_validate_v1` | omniclaude | omniarchon | `omniarchon_codegen_intelligence` | 3 | 7d |
| `omninode_codegen_request_pattern_v1` | omniclaude | omniarchon | `omniarchon_codegen_intelligence` | 3 | 7d |
| `omninode_codegen_request_mixin_v1` | omniclaude | omniarchon | `omniarchon_codegen_intelligence` | 3 | 7d |
| **Codegen Response Topics** |
| `omninode_codegen_response_analyze_v1` | omniarchon | omniclaude | `omniclaude_codegen_consumer` | 3 | 7d |
| `omninode_codegen_response_validate_v1` | omniarchon | omniclaude | `omniclaude_codegen_consumer` | 3 | 7d |
| `omninode_codegen_response_pattern_v1` | omniarchon | omniclaude | `omniclaude_codegen_consumer` | 3 | 7d |
| `omninode_codegen_response_mixin_v1` | omniarchon | omniclaude | `omniclaude_codegen_consumer` | 3 | 7d |
| **Codegen Status Topics** |
| `omninode_codegen_status_session_v1` | omniclaude, omniarchon | Monitoring | `monitoring_codegen_status` | 6 | 3d |
| **Codegen DLQ Topics** |
| `omninode_codegen_dlq_analyze_v1` | KafkaClient | DLQMonitor | `dlq_monitor_codegen` | 1 | 30d |
| `omninode_codegen_dlq_validate_v1` | KafkaClient | DLQMonitor | `dlq_monitor_codegen` | 1 | 30d |
| `omninode_codegen_dlq_pattern_v1` | KafkaClient | DLQMonitor | `dlq_monitor_codegen` | 1 | 30d |
| `omninode_codegen_dlq_mixin_v1` | KafkaClient | DLQMonitor | `dlq_monitor_codegen` | 1 | 30d |
| **Bridge Workflow Topics** |
| `dev.omninode_bridge.onex.evt.stamp-workflow-started.v1` | Orchestrator | Registry, Analytics | `registry_workflow_events`, `analytics_workflow_events` | 3 | 7d |
| `dev.omninode_bridge.onex.evt.stamp-workflow-completed.v1` | Orchestrator | Registry, Analytics | `registry_workflow_events`, `analytics_workflow_events` | 3 | 7d |
| `dev.omninode_bridge.onex.evt.stamp-workflow-failed.v1` | Orchestrator | Registry, Analytics, Alerting | `registry_workflow_events`, `analytics_workflow_events`, `alerting_workflow_failures` | 3 | 7d |
| `dev.omninode_bridge.onex.evt.workflow-step-completed.v1` | Orchestrator | Analytics | `analytics_workflow_events` | 3 | 3d |
| `dev.omninode_bridge.onex.evt.onextree-intelligence-requested.v1` | Orchestrator | OnexTree | `onextree_intelligence_processor` | 3 | 7d |
| `dev.omninode_bridge.onex.evt.onextree-intelligence-received.v1` | Orchestrator | Analytics | `analytics_workflow_events` | 3 | 7d |
| `dev.omninode_bridge.onex.evt.metadata-stamp-created.v1` | Orchestrator | Storage | `storage_stamp_persister` | 3 | 7d |
| `dev.omninode_bridge.onex.evt.blake3-hash-generated.v1` | Orchestrator | Storage | `storage_hash_persister` | 3 | 7d |
| `dev.omninode_bridge.onex.evt.workflow-state-transition.v1` | Orchestrator | Registry | `registry_fsm_tracker` | 3 | 7d |
| **Bridge Introspection Topics** |
| `dev.omninode_bridge.onex.evt.node-introspection.v1` | All Nodes | Registry, Monitoring | `registry_introspection_collectors`, `monitoring_introspection_agents` | 3 | 24h |
| `dev.omninode_bridge.onex.evt.registry-request-introspection.v1` | Registry | All Nodes | `node_introspection_handlers_orchestrator`, `node_introspection_handlers_reducer` | 3 | 1h |
| `dev.omninode_bridge.onex.evt.node-heartbeat.v1` | All Nodes | Registry, Health Monitor | `health_monitors`, `failure_detectors` | 3 | 30m |
| **Bridge Aggregation Topics** |
| `dev.omninode_bridge.onex.evt.aggregation-started.v1` | Reducer | Analytics | `analytics_aggregation_events` | 3 | 3d |
| `dev.omninode_bridge.onex.evt.batch-processed.v1` | Reducer | Analytics | `analytics_aggregation_events` | 3 | 3d |
| `dev.omninode_bridge.onex.evt.state-persisted.v1` | Reducer | Storage | `storage_state_persister` | 3 | 7d |
| `dev.omninode_bridge.onex.evt.aggregation-completed.v1` | Reducer | Analytics | `analytics_aggregation_events` | 3 | 7d |
| `dev.omninode_bridge.onex.evt.aggregation-failed.v1` | Reducer | Analytics, Alerting | `analytics_aggregation_events`, `alerting_aggregation_failures` | 3 | 7d |
| `dev.omninode_bridge.onex.evt.fsm-state-initialized.v1` | Reducer | Registry | `registry_fsm_tracker` | 3 | 7d |
| `dev.omninode_bridge.onex.evt.fsm-state-transitioned.v1` | Reducer | Registry | `registry_fsm_tracker` | 3 | 7d |
| **Service Lifecycle Topics** |
| `dev.omninode_bridge.onex.evt.startup.v1` | All Services | Monitoring | `monitoring_service_lifecycle` | 3 | 24h |
| `dev.omninode_bridge.onex.evt.shutdown.v1` | All Services | Monitoring | `monitoring_service_lifecycle` | 3 | 24h |
| `dev.omninode_bridge.onex.evt.health-check.v1` | All Services | Health Monitor | `health_monitors` | 3 | 1h |
| `dev.omninode_bridge.onex.evt.tool-call.v1` | MCP Services | Analytics | `analytics_tool_execution` | 3 | 3d |
| `dev.omninode_bridge.onex.met.performance.v1` | All Services | Prometheus | `prometheus_metrics_collector` | 3 | 24h |

---

## Event Flow Diagrams

### Codegen Workflow (Request/Response Pattern)

```
┌─────────────┐                                                    ┌─────────────┐
│ omniclaude  │                                                    │ omniarchon  │
└──────┬──────┘                                                    └──────┬──────┘
       │                                                                  │
       │ 1. Publish Analysis Request                                     │
       │───────────────────────────────────────────────────────────────►│
       │   Topic: omninode_codegen_request_analyze_v1                    │
       │   Correlation ID: req-123                                       │
       │                                                                  │
       │                                         2. Process Intelligence │
       │                                            (RAG, Code Analysis) │
       │                                                                  │
       │ 3. Consume Analysis Response                                    │
       │◄───────────────────────────────────────────────────────────────│
       │   Topic: omninode_codegen_response_analyze_v1                   │
       │   Correlation ID: req-123 (matched)                             │
       │                                                                  │
       │ 4. Publish Validation Request                                   │
       │───────────────────────────────────────────────────────────────►│
       │   Topic: omninode_codegen_request_validate_v1                   │
       │   Correlation ID: req-456                                       │
       │                                                                  │
       │ 5. Consume Validation Response                                  │
       │◄───────────────────────────────────────────────────────────────│
       │   Topic: omninode_codegen_response_validate_v1                  │
       │   Correlation ID: req-456 (matched)                             │
       │                                                                  │
       │ Both publish status updates throughout                          │
       │───────────────────────────────────────────────────────────────►│
       │   Topic: omninode_codegen_status_session_v1                     │
       │   Session ID: session-789 (all events share same session ID)    │
```

### Bridge Workflow (Orchestrator → Reducer)

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ Orchestrator │         │   Registry   │         │   Reducer    │
└──────┬───────┘         └──────┬───────┘         └──────┬───────┘
       │                        │                        │
       │ 1. Workflow Started    │                        │
       │──────────────────────►│                        │
       │   correlation_id: wf-1 │                        │
       │                        │                        │
       │ 2. Intelligence Request│                        │
       │──────────────────────►│                        │
       │                        │                        │
       │ 3. Stamp Created       │                        │
       │──────────────────────►│                        │
       │                        │                        │
       │ 4. Workflow Completed  │                        │
       │──────────────────────►│                        │
       │                        │                        │
       │ 5. Pass stamped items  │                        │
       │──────────────────────────────────────────────►│
       │                        │                        │
       │                        │ 6. Aggregation Started │
       │                        │◄───────────────────────│
       │                        │                        │
       │                        │ 7. State Persisted     │
       │                        │◄───────────────────────│
       │                        │                        │
       │                        │ 8. Aggregation Complete│
       │                        │◄───────────────────────│
```

---

## Consumer Groups

### Codegen Consumer Groups

**omniclaude_codegen_consumer**:
- Topics: All codegen response topics (4 topics)
- Consumers: omniclaude instances
- Auto Offset Reset: latest
- Session Timeout: 30s
- Heartbeat Interval: 10s

**omniarchon_codegen_intelligence**:
- Topics: All codegen request topics (4 topics)
- Consumers: omniarchon instances
- Auto Offset Reset: latest
- Session Timeout: 30s
- Heartbeat Interval: 10s

**dlq_monitor_codegen**:
- Topics: All codegen DLQ topics (4 topics)
- Consumers: DLQMonitor instances
- Auto Offset Reset: earliest
- Session Timeout: 30s

### Bridge Consumer Groups

**registry_introspection_collectors**:
- Topics: node-introspection.v1
- Consumers: NodeRegistry instances
- Auto Offset Reset: latest

**registry_workflow_events**:
- Topics: workflow-started.v1, workflow-completed.v1, workflow-failed.v1, fsm-state-*.v1
- Consumers: NodeRegistry instances

**analytics_workflow_events**:
- Topics: All workflow events
- Consumers: Analytics pipeline
- Auto Offset Reset: earliest (process historical data)

**storage_stamp_persister**:
- Topics: metadata-stamp-created.v1, blake3-hash-generated.v1
- Consumers: Storage service

**health_monitors**:
- Topics: node-heartbeat.v1, health-check.v1
- Consumers: Health monitoring systems
- Auto Offset Reset: latest

---

## Quick Reference

### Topic Naming Patterns

**Codegen Topics**:
```
omninode_codegen_{category}_{operation}_v{version}
Examples:
- omninode_codegen_request_analyze_v1
- omninode_codegen_response_validate_v1
- omninode_codegen_dlq_pattern_v1
```

**Bridge Topics**:
```
{env}.omninode_bridge.onex.{type}.{event-slug}.v{version}
Examples:
- dev.omninode_bridge.onex.evt.stamp-workflow-started.v1
- dev.omninode_bridge.onex.evt.aggregation-completed.v1
- dev.omninode_bridge.onex.met.performance.v1
```

### Schema Versions

All schemas currently at version 1.0:
- `schema_version: "1.0"` (codegen schemas)
- `event_version: "1.0"` (bridge events in envelope)
- Topic suffix: `_v1` or `.v1`

### Envelope Compliance Status

✅ **Fully Compliant** (19 topics):
- All bridge orchestrator topics (12)
- All bridge reducer topics (7)

❌ **Requires Migration** (18 topics):
- All codegen topics (13)
- All service lifecycle topics (5)

---

**Last Updated**: October 2025
**Maintained By**: OmniNode Bridge Team
**Status**: Phase 3 Complete - Comprehensive Documentation
**Next Phase**: Schema Validation Implementation
