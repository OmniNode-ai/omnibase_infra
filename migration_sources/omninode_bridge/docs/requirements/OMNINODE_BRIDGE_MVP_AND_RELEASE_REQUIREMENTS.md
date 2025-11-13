# OmniNode Bridge MVP and Release Requirements Analysis

**Date**: October 22, 2025 (Analysis) | **Updated**: October 29, 2025 (Status Review)
**Repository**: `/Volumes/PRO-G40/Code/omninode_bridge`
**Analysis Scope**: MVP requirements for node generation workflow and releaseable product requirements
**Status**: ✅ **MVP COMPLETE** - Phase 1 & 2 delivered October 2025

---

## Executive Summary

The omninode_bridge repository contains a production-ready MVP foundation with **20+ ONEX-compliant nodes**, comprehensive event infrastructure (13 Kafka topics), and battle-tested orchestration patterns. The repository has **92.8% test coverage** (501 tests) and demonstrates mature patterns for workflow coordination, metrics aggregation, and service mesh integration.

**Key Findings**:
- ✅ **Strong Foundation**: Production-grade orchestrators, reducers, and service clients already implemented
- ✅ **Event Infrastructure**: Complete Kafka/Redpanda integration with DLQ, circuit breakers, and resilience patterns
- ✅ **LlamaIndex Integration**: Event-driven workflow orchestration already in use for complex multi-step workflows
- ⚠️ **Gap Analysis**: Node generation workflow requires adapting existing patterns to omniclaude integration
- 🎯 **MVP Focus**: Reuse 90% of existing patterns; build 10% new integration layer

---

## 1. Current Architecture Analysis

### 1.1 Existing Orchestrators (Workflow Coordination)

**NodeBridgeWorkflowOrchestrator** (`src/omninode_bridge/nodes/orchestrator/v1_0_0/workflow_node.py`)

**Architecture**: LlamaIndex Workflows-based event-driven orchestration
**Performance**: <2000ms end-to-end latency target, 100+ workflows/second throughput

**Workflow Pattern** (6-step pipeline):
```
1. validate_input       → ValidationCompletedEvent
2. generate_hash        → HashGeneratedEvent
3. create_stamp         → StampCreatedEvent | IntelligenceRequestedEvent
4. enrich_intelligence  → IntelligenceReceivedEvent (optional)
5. persist_state        → PersistenceCompletedEvent
6. complete_workflow    → StopEvent (final result)
```

**Key Features**:
- ✅ Type-safe workflow steps with `@step` decorator
- ✅ Automatic event routing and validation
- ✅ Context management for shared state across steps
- ✅ Graceful error handling with centralized `handle_workflow_error()`
- ✅ Kafka event publishing at each lifecycle stage
- ✅ Conditional routing (intelligence enrichment optional)
- ✅ Correlation ID propagation for distributed tracing

**Integration Points**:
- MetadataStamping service (HTTP) for BLAKE3 hash generation
- OnexTree service (HTTP) for AI enrichment (optional)
- KafkaClient for event publishing (9 event types)
- PostgreSQL for workflow state persistence

**Reusability for Node Generation**: **HIGH (95%)**
- Replace `validate_input` → `validate_prd` (PRD validation)
- Replace `generate_hash` → `analyze_requirements` (RAG intelligence)
- Replace `create_stamp` → `generate_node_code` (code generation)
- Replace `enrich_intelligence` → `validate_node_quality` (ONEX compliance)
- Keep `persist_state` → `persist_generation_metadata` (unchanged)
- Keep `complete_workflow` → `complete_generation` (minimal changes)

---

### 1.2 Existing Reducers (Metrics Aggregation)

**NodeBridgeReducer** (`src/omninode_bridge/nodes/reducer/v1_0_0/node.py`)

**Architecture**: Pure functional reducer with FSM state management
**Performance**: >1000 items/second aggregation throughput, <100ms latency for 1000 items

**Reduction Strategy**:
```python
# Streaming aggregation with windowing
async def execute_reduction(contract: ModelContractReducer) -> ModelReducerOutputState:
    1. Stream metadata from input (async iterator with batching)
    2. Group by aggregation strategy (namespace/time_window/file_type)
    3. Compute aggregations (count, sum, avg, distinct)
    4. Update FSM state for each workflow (in-memory)
    5. Return aggregation results with intents for side effects
```

**Aggregation Types** (6 strategies):
- `NAMESPACE_GROUPING` - Multi-tenant isolation and grouping (primary)
- `TIME_WINDOW` - Windowed aggregation (default: 5000ms windows)
- `FILE_TYPE_GROUPING` - Group by content type
- `SIZE_BUCKETS` - Group by file size ranges
- `WORKFLOW_GROUPING` - Group by workflow ID
- `CUSTOM` - User-defined aggregation logic

**FSM State Management**:
```python
class FSMStateManager:
    """
    Contract-driven FSM state tracker
    - Loads states/transitions from FSM subcontract YAML
    - Validates state transitions dynamically
    - PostgreSQL persistence for recovery
    - Transition history tracking
    """
```

**Reusability for Node Generation**: **MODERATE (70%)**
- Metrics for generation tracking (nodes generated per hour, success rate, quality scores)
- FSM state tracking for generation workflows (pending → analyzing → generating → validating → completed/failed)
- Aggregation by node type, domain, complexity
- Requires new aggregation types: `NODE_TYPE_GROUPING`, `DOMAIN_GROUPING`, `QUALITY_BUCKETS`

---

### 1.3 Service Mesh Integration

#### Kafka Event Bus

**KafkaClient** (`src/omninode_bridge/services/kafka_client.py`)

**Features**:
- ✅ Async Kafka producer with aiokafka (0.11.0)
- ✅ Circuit breaker protection (`@KAFKA_CIRCUIT_BREAKER()`)
- ✅ Dead Letter Queue (DLQ) support with configurable suffix
- ✅ Exponential backoff retry (max 3 attempts, 1.0s base)
- ✅ Performance optimizations:
  - Environment-aware batching (dev: 0ms linger, prod: 10ms linger)
  - Compression (dev: none, staging: gzip, prod: lz4)
  - Configurable batch sizes (via `BatchSizeManager`)
- ✅ Intelligent partitioning (balanced/round-robin/key-based)
- ✅ Metrics tracking (message count, bytes sent, batch count, latencies)

**Topics Configuration** (13 existing topics):
```yaml
Request Topics (4):
  - omninode_codegen_request_analyze_v1
  - omninode_codegen_request_validate_v1
  - omninode_codegen_request_pattern_v1
  - omninode_codegen_request_mixin_v1

Response Topics (4):
  - omninode_codegen_response_analyze_v1
  - omninode_codegen_response_validate_v1
  - omninode_codegen_response_pattern_v1
  - omninode_codegen_response_mixin_v1

Status Topics (1):
  - omninode_codegen_status_session_v1 (6 partitions)

Dead Letter Queue (4):
  - omninode_codegen_dlq_analyze_v1
  - omninode_codegen_dlq_validate_v1
  - omninode_codegen_dlq_pattern_v1
  - omninode_codegen_dlq_mixin_v1
```

**Event Schemas** (`src/omninode_bridge/events/codegen_schemas.py`):
- 9 Pydantic v2 models with strict typing and versioning
- OnexEnvelopeV1 format for all events
- Correlation ID tracking across all events

**Reusability for Node Generation**: **VERY HIGH (100%)**
- Existing topics already support node generation workflows
- Event schemas ready for omniclaude integration
- No changes required to Kafka infrastructure

---

#### Consul Service Discovery

**ConsulClient** (`src/omninode_bridge/services/metadata_stamping/registry/consul_client.py`)

**Features**:
- Service registration with health checks
- Service discovery with DNS/HTTP
- Key-value configuration management
- Health monitoring integration

**Status**: Implemented but not actively used in current MVP

---

#### OnexTree Intelligence Service

**OnexTreeClient** (`src/omninode_bridge/clients/onextree_client.py`)

**Features**:
- HTTP client for AI-powered content analysis
- Circuit breaker protection for resilience
- Optional intelligence enrichment in workflows
- Graceful degradation on failure

**Integration Pattern**:
```python
# Optional intelligence enrichment step in workflow
if ctx.data.get("enable_intelligence", False):
    intelligence_data = await onextree_client.analyze_content(content)
else:
    intelligence_data = None  # Skip intelligence
```

**Reusability for Node Generation**: **HIGH (85%)**
- Replace content analysis → node quality analysis
- Add ONEX compliance validation
- Add pattern matching for similar nodes

---

### 1.4 Infrastructure Patterns

#### Circuit Breakers

**Implementation**: Custom circuit breaker pattern with state machine
**Location**:
- `src/omninode_bridge/clients/circuit_breaker.py`
- `src/metadata_stamping/distributed/circuit_breaker.py`

**Configuration**:
```python
KAFKA_CIRCUIT_BREAKER = CircuitBreaker(
    failure_threshold=5,      # Open after 5 consecutive failures
    recovery_timeout=60,      # Close after 60s
    half_open_max_calls=3     # Test with 3 calls before full close
)
```

**Applied To**:
- Kafka producer connections
- PostgreSQL connections
- HTTP service clients (OnexTree, MetadataStamping)

---

#### Health Checks

**HealthCheckMixin** (`src/omninode_bridge/nodes/mixins/health_mixin.py`)

**Features**:
- Component-level health tracking
- Critical vs non-critical component classification
- Timeout enforcement per component
- Overall health aggregation

**Health States**:
- `HEALTHY` - All critical components passing
- `DEGRADED` - Non-critical components failing
- `UNHEALTHY` - Critical components failing

**Usage Pattern**:
```python
class NodeBridgeReducer(NodeReducer, HealthCheckMixin):
    def initialize_health_checks(self):
        self.register_component_check(
            "aggregation_buffer",
            self._check_aggregation_buffer_health,
            critical=False,
            timeout_seconds=1.0
        )
```

---

#### Monitoring & Observability

**Components**:
- OpenTelemetry integration (API 1.27.0, SDK 1.27.0)
- Structured logging with audit trail
- Performance metrics tracking (latencies, throughput, error rates)
- Event tracing via correlation IDs

**Metrics Tracked**:
- Workflow execution times (p50, p95, p99)
- Aggregation throughput (items/second)
- Kafka publish latencies
- Database query times
- Circuit breaker state transitions

---

## 2. MVP Requirements for Node Generation Workflow

### 2.1 MVP Scope Definition

**Goal**: Enable omniclaude to generate ONEX-compliant nodes using omninode_bridge orchestration patterns

**MVP Features** (✅ All Delivered October 2025):
1. ✅ PRD analysis via Archon MCP intelligence - **COMPLETE**
2. ✅ Node type detection (Effect/Compute/Reducer/Orchestrator) - **COMPLETE**
3. ✅ Code generation using LlamaIndex workflow orchestration - **COMPLETE**
4. ✅ ONEX compliance validation - **COMPLETE**
5. ✅ Event-driven communication between omniclaude and omniarchon - **COMPLETE**
6. ✅ Metrics tracking for generation workflows - **COMPLETE**

**Out of Scope for MVP** (Deferred to Post-MVP):
- ⏳ Advanced pattern matching across projects
- ⏳ Multi-node coordination for complex systems
- ⏳ Automated testing generation
- ⏳ Performance optimization beyond basic quality checks

---

### 2.2 Required Components for MVP

#### 2.2.1 Node Generation Orchestrator

**New Node**: `NodeCodegenOrchestrator` (extends `NodeBridgeWorkflowOrchestrator` pattern)

**Workflow Steps**:
```python
class NodeCodegenOrchestrator(Workflow):
    @step
    async def validate_prd(self, ctx, ev: StartEvent) -> PRDValidatedEvent:
        """Validate PRD structure and requirements"""

    @step
    async def gather_intelligence(self, ctx, ev: PRDValidatedEvent) -> IntelligenceGatheredEvent:
        """Query Archon MCP for similar nodes, patterns, mixins"""

    @step
    async def detect_node_type(self, ctx, ev: IntelligenceGatheredEvent) -> NodeTypeDetectedEvent:
        """Classify as Effect/Compute/Reducer/Orchestrator"""

    @step
    async def generate_code(self, ctx, ev: NodeTypeDetectedEvent) -> CodeGeneratedEvent:
        """Generate node implementation using templates + intelligence"""

    @step
    async def validate_quality(self, ctx, ev: CodeGeneratedEvent) -> QualityValidatedEvent:
        """Run ONEX compliance checks, linting, type checking"""

    @step
    async def persist_generation(self, ctx, ev: QualityValidatedEvent) -> PersistenceCompletedEvent:
        """Save generated code, metadata, and metrics"""

    @step
    async def complete_generation(self, ctx, ev: PersistenceCompletedEvent) -> StopEvent:
        """Return final generation result with file paths"""
```

**Implementation Estimate**: 3-4 days
- Day 1: Adapt workflow skeleton from `NodeBridgeWorkflowOrchestrator`
- Day 2: Implement PRD validation and intelligence gathering steps
- Day 3: Implement code generation and quality validation steps
- Day 4: Testing and integration with existing Kafka topics

**Reuse Percentage**: 95% from existing orchestrator pattern

---

#### 2.2.2 Generation Metrics Reducer

**New Node**: `NodeCodegenMetricsReducer` (extends `NodeBridgeReducer` pattern)

**Aggregation Strategy**:
```python
aggregation_types = [
    EnumAggregationType.NODE_TYPE_GROUPING,      # Effect/Compute/Reducer/Orchestrator
    EnumAggregationType.DOMAIN_GROUPING,         # api_development/ml_training/etc
    EnumAggregationType.QUALITY_BUCKETS,         # Low/Medium/High quality scores
    EnumAggregationType.TIME_WINDOW,             # Hourly/daily generation rates
]

metrics_to_track = {
    "total_nodes_generated": int,
    "success_rate": float,
    "average_quality_score": float,
    "average_generation_time_ms": float,
    "nodes_by_type": dict[str, int],
    "intelligence_usage_rate": float,
}
```

**FSM States**:
```python
generation_workflow_states = {
    "PENDING": "Waiting for PRD analysis",
    "ANALYZING": "Gathering intelligence from Archon MCP",
    "GENERATING": "Generating node code",
    "VALIDATING": "Running quality checks",
    "COMPLETED": "Generation successful",
    "FAILED": "Generation failed with errors",
}
```

**Implementation Estimate**: 2-3 days
- Day 1: Add new aggregation types to `EnumAggregationType`
- Day 2: Extend aggregation logic for node generation metrics
- Day 3: Testing and integration with orchestrator events

**Reuse Percentage**: 70% from existing reducer pattern

---

#### 2.2.3 Event Bus Integration

**Status**: ✅ **Already Complete**

**Existing Topics** (reuse as-is):
- `omninode_codegen_request_analyze_v1` - PRD analysis requests from omniclaude
- `omninode_codegen_response_analyze_v1` - Intelligence results from omniarchon
- `omninode_codegen_request_validate_v1` - Quality validation requests
- `omninode_codegen_response_validate_v1` - Validation results
- `omninode_codegen_status_session_v1` - Real-time session status updates
- DLQ topics for failed events

**Event Flow**:
```
omniclaude (CLI)
    → publish(omninode_codegen_request_analyze_v1, prd_content)

omniarchon (Intelligence Service)
    → consume(omninode_codegen_request_analyze_v1)
    → process with RAG/Qdrant/Memgraph
    → publish(omninode_codegen_response_analyze_v1, intelligence_data)

omniclaude (CLI)
    → consume(omninode_codegen_response_analyze_v1)
    → trigger NodeCodegenOrchestrator workflow
    → publish(omninode_codegen_status_session_v1, workflow_progress)
```

**Implementation Estimate**: 0 days (already complete)

**Reuse Percentage**: 100% from existing event infrastructure

---

#### 2.2.4 Registry Service for Generated Nodes

**Purpose**: Track generated nodes for discovery, introspection, and versioning

**Functionality**:
- Node metadata storage (name, type, version, domain, quality score)
- Node search and discovery
- Version tracking and history
- Quality metrics aggregation

**Implementation Options**:

**Option A**: Extend existing `NodeBridgeRegistry` pattern
- Registry node already exists at `src/omninode_bridge/nodes/registry/`
- Add node generation metadata to registry schema
- Reuse introspection mixin for node discovery

**Option B**: Use PostgreSQL directly via DatabaseAdapter
- Leverage existing `NodeDatabaseAdapterEffect` for CRUD operations
- Add `generated_nodes` table to schema
- Use existing connection pooling and circuit breakers

**Recommendation**: **Option A** (extend NodeBridgeRegistry)
- More ONEX-compliant (uses registry node pattern)
- Supports introspection and service discovery out of the box
- Better separation of concerns

**Implementation Estimate**: 3-4 days
- Day 1: Analyze existing registry node implementation
- Day 2: Extend registry schema for node generation metadata
- Day 3: Implement search and discovery endpoints
- Day 4: Testing and integration

**Reuse Percentage**: 80% from existing registry node pattern

---

### 2.3 MVP Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      omniclaude CLI                              │
│  (User invokes: omniclaude generate node --prd requirements.md) │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ 1. Publish PRD analysis request
                 ▼
         ┌──────────────────┐
         │   Kafka Topic    │
         │ request_analyze  │
         └──────────────────┘
                 │
                 │ 2. Consume request
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              omniarchon Intelligence Service                     │
│  - RAG query for similar nodes                                   │
│  - Pattern matching across projects                              │
│  - Mixin recommendations                                         │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ 3. Publish intelligence results
                 ▼
         ┌──────────────────┐
         │   Kafka Topic    │
         │ response_analyze │
         └──────────────────┘
                 │
                 │ 4. Consume results
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│         NodeCodegenOrchestrator (omninode_bridge)                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LlamaIndex Workflow (6 steps)                            │   │
│  │ 1. validate_prd        → PRDValidatedEvent               │   │
│  │ 2. gather_intelligence → IntelligenceGatheredEvent       │   │
│  │ 3. detect_node_type    → NodeTypeDetectedEvent           │   │
│  │ 4. generate_code       → CodeGeneratedEvent              │   │
│  │ 5. validate_quality    → QualityValidatedEvent           │   │
│  │ 6. persist_generation  → GenerationCompletedEvent        │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────┬───────────────────┬────────────────────────────┘
                 │                   │
                 │ Parallel          │ Parallel
                 ▼                   ▼
    ┌───────────────────┐   ┌──────────────────────┐
    │ NodeCodegen       │   │ NodeCodegenRegistry  │
    │ MetricsReducer    │   │ (Node Discovery)     │
    │ (Aggregation)     │   │                      │
    └───────────────────┘   └──────────────────────┘
                 │                   │
                 │ Store             │ Store
                 ▼                   ▼
         ┌──────────────────────────────┐
         │       PostgreSQL             │
         │ - generation_metrics         │
         │ - generated_nodes_registry   │
         │ - workflow_executions        │
         └──────────────────────────────┘
```

---

## 3. Releaseable Product Requirements

### 3.1 Production Readiness Criteria

#### 3.1.1 Advanced Workflow Orchestration

**Current State**: LlamaIndex Workflows implementation supports basic orchestration
**Gap**: Multi-step dependency management, conditional branching, parallel execution

**Requirements**:
- ✅ Parallel intelligence gathering (RAG + Qdrant + Memgraph simultaneously)
- ✅ Conditional quality gates (retry generation if quality < threshold)
- ✅ Multi-node coordination (generate multiple dependent nodes in correct order)
- ✅ Workflow state recovery (resume from last successful step on failure)
- ✅ Workflow templates (reusable workflow definitions for common patterns)

**Implementation**:
```python
class AdvancedCodegenOrchestrator(Workflow):
    @step
    async def parallel_intelligence_gathering(
        self, ctx, ev: PRDValidatedEvent
    ) -> list[IntelligenceSourceEvent]:
        """
        Fan-out to multiple intelligence sources in parallel
        Returns: [RAGEvent, QdrantEvent, MemgraphEvent]
        """

    @step
    async def aggregate_intelligence(
        self, ctx, ev: list[IntelligenceSourceEvent]
    ) -> IntelligenceAggregatedEvent:
        """
        Fan-in: Combine intelligence from multiple sources
        Apply confidence scoring and conflict resolution
        """

    @step
    async def quality_gate_checkpoint(
        self, ctx, ev: CodeGeneratedEvent
    ) -> Union[QualityPassedEvent, RetryGenerationEvent]:
        """
        Conditional routing based on quality score
        If quality < 0.7: Retry with different strategy
        If quality >= 0.7: Continue to completion
        """
```

**Implementation Estimate**: 5-7 days
**Priority**: HIGH (required for production)

---

#### 3.1.2 Multi-Node Coordination

**Purpose**: Generate multiple interdependent nodes in correct order (e.g., Effect → Compute → Reducer → Orchestrator)

**Requirements**:
- Dependency graph construction from PRD analysis
- Topological sort for correct generation order
- Shared context across node generations (types, models, contracts)
- Rollback on partial failure (all-or-nothing generation)

**Implementation**:
```python
class MultiNodeCoordinator:
    async def generate_node_system(
        self,
        prd_content: str,
        system_architecture: dict[str, Any]
    ) -> list[GeneratedNode]:
        """
        Generate multiple interdependent nodes

        1. Parse PRD and extract node dependencies
        2. Build dependency graph (DAG)
        3. Topologically sort nodes
        4. Generate nodes in correct order
        5. Validate cross-node references
        6. Commit all or rollback on failure
        """
```

**Implementation Estimate**: 7-10 days
**Priority**: MEDIUM (post-MVP enhancement)

---

#### 3.1.3 Performance Optimization

**Targets** (based on existing benchmarks):
- Node generation workflow: <5000ms end-to-end (p95)
- Intelligence gathering: <2000ms (p95, parallel execution)
- Code generation: <1000ms per node (p95)
- Quality validation: <500ms (p95)
- Throughput: 50+ nodes/minute per instance

**Optimizations**:
- ✅ Parallel intelligence gathering (implemented in Archon MCP)
- ⚠️ Template caching for common node patterns
- ⚠️ Incremental code generation (generate diffs, not full files)
- ⚠️ Async I/O for file operations
- ⚠️ Connection pooling optimization (PostgreSQL, Kafka, HTTP)

**Implementation Estimate**: 3-5 days
**Priority**: MEDIUM (post-MVP tuning)

---

#### 3.1.4 Monitoring and Alerting

**Requirements**:
- Real-time generation metrics dashboard
- Alerting on quality degradation (quality score drop >20%)
- Alerting on failure rate spikes (>10% failures in 5min window)
- Alerting on latency spikes (p95 latency >2x baseline)
- Event tracing visualization (correlation ID tracking)

**Implementation**:
```python
# Prometheus metrics
generation_total = Counter('node_generation_total', ['node_type', 'domain'])
generation_duration = Histogram('node_generation_duration_seconds', ['node_type'])
generation_quality = Gauge('node_generation_quality_score', ['node_type'])
generation_failures = Counter('node_generation_failures_total', ['error_type'])

# Alert rules (Prometheus AlertManager)
- alert: HighGenerationFailureRate
  expr: rate(node_generation_failures_total[5m]) > 0.1

- alert: QualityDegradation
  expr: avg_over_time(node_generation_quality_score[1h]) < 0.7

- alert: LatencySpike
  expr: histogram_quantile(0.95, node_generation_duration_seconds) > 10
```

**Implementation Estimate**: 5-7 days
**Priority**: HIGH (required for production)

---

### 3.2 Missing Infrastructure Components

#### 3.2.1 Health Checks Status

**Current State**: ✅ **Comprehensive health check system already implemented**

**Existing Components**:
- `HealthCheckMixin` for node-level health tracking
- Component-level health checks with timeout enforcement
- Health states: HEALTHY, DEGRADED, UNHEALTHY
- Health check CLI tool: `src/omninode_bridge/nodes/health_check_cli.py`

**Usage Pattern**:
```python
# Health check registration
self.register_component_check(
    "kafka_producer",
    self._check_kafka_health,
    critical=True,      # Critical for overall health
    timeout_seconds=2.0
)

# Health check execution
health_status = await self.get_overall_health()
# Returns: {"status": "HEALTHY", "components": {...}, "timestamp": ...}
```

**Missing for Production**:
- ⚠️ Health check HTTP endpoint for Kubernetes liveness/readiness probes
- ⚠️ Health check aggregation across multiple instances
- ⚠️ Health trend analysis (degradation detection over time)

**Implementation Estimate**: 2-3 days (HTTP endpoint + aggregation)

---

#### 3.2.2 Circuit Breakers Status

**Current State**: ✅ **Production-grade circuit breakers already implemented**

**Existing Components**:
- `CircuitBreaker` class with state machine (CLOSED → OPEN → HALF_OPEN)
- Applied to: Kafka producer, PostgreSQL client, HTTP service clients
- Configuration: Failure threshold, recovery timeout, half-open max calls
- Decorator pattern for easy application: `@KAFKA_CIRCUIT_BREAKER()`

**Configuration**:
```python
KAFKA_CIRCUIT_BREAKER = CircuitBreaker(
    failure_threshold=5,      # Open after 5 consecutive failures
    recovery_timeout=60,      # Attempt recovery after 60s
    half_open_max_calls=3     # Test with 3 calls before full recovery
)
```

**Missing for Production**:
- ⚠️ Circuit breaker metrics export (state transitions, failure counts)
- ⚠️ Circuit breaker dashboard visualization
- ⚠️ Manual circuit breaker control (admin API to force open/close)

**Implementation Estimate**: 2-3 days (metrics + dashboard + API)

---

#### 3.2.3 Distributed Tracing

**Current State**: ⚠️ **Partial implementation via correlation IDs**

**Existing**:
- Correlation ID propagation through workflow context
- Correlation ID tracking in Kafka events
- Correlation ID logging in structured logs

**Missing**:
- OpenTelemetry distributed tracing (full span propagation)
- Jaeger/Zipkin integration for trace visualization
- Cross-service trace correlation (omniclaude → omniarchon → omninode_bridge)

**Implementation**:
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider

# Initialize tracer
tracer_provider = TracerProvider()
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
tracer_provider.add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)
trace.set_tracer_provider(tracer_provider)

# Usage in workflow
with tracer.start_as_current_span("node_generation"):
    with tracer.start_as_current_span("prd_validation"):
        validate_prd(prd_content)
    with tracer.start_as_current_span("code_generation"):
        generate_code(intelligence_data)
```

**Implementation Estimate**: 5-7 days
**Priority**: MEDIUM (valuable for debugging, not critical for MVP)

---

#### 3.2.4 Rate Limiting

**Current State**: ✅ **Rate limiting service already implemented**

**Existing Components**:
- `RateLimitingService` at `src/omninode_bridge/services/rate_limiting_service.py`
- Token bucket and leaky bucket algorithms
- Per-user and per-endpoint rate limiting
- Redis-backed distributed rate limiting

**Missing for Production**:
- ⚠️ Integration with node generation workflow (apply rate limits to generation requests)
- ⚠️ Dynamic rate limit adjustment based on system load
- ⚠️ Rate limit metrics and monitoring

**Implementation Estimate**: 2-3 days (integration + monitoring)

---

## 4. Integration Points

### 4.1 omniclaude Integration

**Interface**: CLI command for node generation

**Command Structure**:
```bash
omniclaude generate node \
  --prd requirements.md \
  --node-type effect \
  --domain api_development \
  --enable-intelligence \
  --output-dir ./generated/
```

**Integration Flow**:
```python
# omniclaude CLI → KafkaClient
kafka_client = KafkaClient(bootstrap_servers="localhost:29092")

# Publish PRD analysis request
request = CodegenAnalysisRequest(
    correlation_id=uuid4(),
    session_id=uuid4(),
    prd_content=prd_file.read_text(),
    analysis_type="full",
    workspace_context={"project_path": "/path/to/project"}
)

await kafka_client.publish_event(
    "omninode_codegen_request_analyze_v1",
    request.model_dump()
)

# Consume intelligence response
response = await kafka_client.consume_event(
    "omninode_codegen_response_analyze_v1",
    correlation_id=request.correlation_id,
    timeout=30.0
)

# Trigger node generation workflow
orchestrator = NodeCodegenOrchestrator(container)
result = await orchestrator.run(
    correlation_id=request.correlation_id,
    prd_content=prd_content,
    intelligence_data=response.intelligence_data,
    enable_quality_validation=True
)

# Return generated files
print(f"Generated node: {result.file_path}")
print(f"Quality score: {result.quality_score}")
```

**Implementation Estimate**: 3-4 days
**Dependencies**: NodeCodegenOrchestrator, KafkaClient (both exist)

---

### 4.2 omniarchon Integration

**Interface**: Kafka event bus (already implemented)

**Event Flow**:
```
omniclaude → [omninode_codegen_request_analyze_v1] → omniarchon
omniarchon → [omninode_codegen_response_analyze_v1] → omniclaude
```

**Intelligence Services** (existing in Archon MCP):
- `perform_rag_query` - RAG intelligence for similar nodes
- `search_code_examples` - Code example search
- `assess_code_quality` - ONEX compliance scoring
- `search_similar_entities` - Vector similarity search
- `quality_weighted_search` - Quality-weighted vector search

**Integration Status**: ✅ **Already complete** (Kafka topics and schemas ready)

---

### 4.3 OnexTree Integration

**Purpose**: Optional AI enrichment for node generation quality

**Integration Point**: Optional step in `NodeCodegenOrchestrator` workflow

**Usage**:
```python
@step
async def enrich_with_onextree(
    self, ctx, ev: CodeGeneratedEvent
) -> IntelligenceReceivedEvent:
    """
    Optional OnexTree AI enrichment
    - Code quality analysis
    - Security vulnerability detection
    - Performance optimization suggestions
    """
    if ctx.data.get("enable_onextree", False):
        onextree_client = self.container.get_service("onextree_client")
        intelligence = await onextree_client.analyze_code(
            code=ev.generated_code,
            language="python",
            analysis_type="full"
        )
        return IntelligenceReceivedEvent(intelligence_data=intelligence)
    else:
        return IntelligenceReceivedEvent(intelligence_data=None)
```

**Integration Status**: ✅ **Client already implemented** (just needs workflow integration)

---

## 5. Estimated Effort

### 5.1 MVP Implementation (✅ Completed October 2025)

| Component | Effort | Priority | Status | Completion Date |
|-----------|--------|----------|--------|----------------|
| NodeCodegenOrchestrator | 3-4 days | P0 | ✅ COMPLETE | October 2025 |
| NodeCodegenMetricsReducer | 2-3 days | P0 | ✅ COMPLETE | October 2025 |
| NodeCodegenRegistry | 3-4 days | P1 | ✅ COMPLETE | October 2025 |
| omniclaude CLI Integration | 3-4 days | P0 | ✅ COMPLETE | October 2025 |
| Event Bus Integration | 0 days | ✅ COMPLETE | October 2025 |
| Testing & Documentation | 3-4 days | P0 | ✅ COMPLETE | October 2025 |

**Total MVP Effort**: ✅ **Completed in October 2025** (originally estimated 14-19 days)

**Critical Path**: ✅ **All critical path items delivered**
1. ✅ NodeCodegenOrchestrator (3-4 days) - **COMPLETE**
2. ✅ omniclaude CLI Integration (3-4 days) - **COMPLETE**
3. ✅ Testing & Documentation (3-4 days) - **COMPLETE**

---

### 5.2 Release Implementation (3-4 weeks)

| Component | Effort | Priority | Dependencies |
|-----------|--------|----------|--------------|
| Advanced Workflow Orchestration | 5-7 days | P0 | MVP complete |
| Multi-Node Coordination | 7-10 days | P1 | Advanced orchestration |
| Performance Optimization | 3-5 days | P1 | MVP benchmarks |
| Monitoring & Alerting | 5-7 days | P0 | Prometheus, Grafana |
| Health Check HTTP Endpoints | 2-3 days | P0 | HealthCheckMixin |
| Circuit Breaker Metrics | 2-3 days | P1 | CircuitBreaker |
| Distributed Tracing | 5-7 days | P2 | OpenTelemetry |
| Rate Limiting Integration | 2-3 days | P1 | RateLimitingService |

**Total Release Effort**: 31-45 days (4-6 weeks with parallelization)

**Critical Path** (P0 only):
1. Advanced Workflow Orchestration (5-7 days)
2. Monitoring & Alerting (5-7 days)
3. Health Check HTTP Endpoints (2-3 days)
**Total**: 12-17 days (2-3 weeks)

---

### 5.3 Team Composition Recommendations

**MVP Team** (2-3 weeks):
- 1 Senior Engineer (Workflow orchestration, architecture)
- 1 Mid-level Engineer (Integration, testing)
- 1 DevOps Engineer (Infrastructure, Kafka, PostgreSQL)

**Release Team** (3-4 weeks):
- 2 Senior Engineers (Advanced orchestration, multi-node coordination)
- 2 Mid-level Engineers (Performance optimization, monitoring)
- 1 DevOps Engineer (Observability, alerting, deployment)
- 1 QA Engineer (Load testing, integration testing)

---

## 6. Risk Assessment

### 6.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| LlamaIndex workflow complexity | HIGH | MEDIUM | Leverage existing `NodeBridgeWorkflowOrchestrator` pattern (95% reuse) |
| Kafka topic schema evolution | MEDIUM | LOW | Schemas already versioned (v1), use backward-compatible changes |
| Intelligence latency spikes | MEDIUM | MEDIUM | Implement timeout enforcement (30s) and graceful degradation |
| Multi-node coordination deadlocks | HIGH | LOW | Use DAG topological sort, implement timeout with rollback |
| PostgreSQL connection exhaustion | MEDIUM | LOW | Existing connection pooling (10-50 connections), circuit breakers |

---

### 6.2 Integration Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| omniclaude CLI changes breaking event contracts | HIGH | LOW | Event schemas are Pydantic v2 with strict validation |
| omniarchon service unavailability | MEDIUM | MEDIUM | Circuit breaker + graceful degradation (skip intelligence) |
| OnexTree service latency | LOW | MEDIUM | Optional step, timeout enforcement (5s) |
| Kafka broker downtime | HIGH | LOW | DLQ for failed events, producer retry with backoff |

---

### 6.3 Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| High memory usage from workflow context | MEDIUM | MEDIUM | Context cleanup after workflow completion |
| Kafka topic partition hotspots | MEDIUM | LOW | Intelligent partitioning (balanced strategy) |
| PostgreSQL query performance degradation | MEDIUM | MEDIUM | Existing indexes on JSONB fields, partitioned metrics table |
| FSM state cache growth | LOW | MEDIUM | Periodic cleanup of terminal states (>24h old) |

---

## 7. Recommendations

### 7.1 MVP Priorities (✅ All P0 and P1 Items Completed)

**CRITICAL (P0)** - ✅ **MVP COMPLETE (October 2025)**:
1. ✅ NodeCodegenOrchestrator with 6-step workflow - **DELIVERED**
2. ✅ omniclaude CLI integration with Kafka event bus - **DELIVERED**
3. ✅ Basic metrics tracking via NodeCodegenMetricsReducer - **DELIVERED**
4. ✅ Testing and documentation - **DELIVERED**

**IMPORTANT (P1)** - ✅ **ALL DELIVERED (October 2025)**:
1. ✅ NodeCodegenRegistry for node discovery - **DELIVERED**
2. ✅ Quality validation with ONEX compliance checks - **DELIVERED**
3. ✅ Health check HTTP endpoints for Kubernetes - **DELIVERED**

**NICE TO HAVE (P2)** - ⏳ **Deferred to Post-MVP**:
1. ⏳ Advanced workflow orchestration (parallel intelligence gathering)
2. ⏳ Multi-node coordination
3. ⏳ Distributed tracing with Jaeger
4. ⏳ Performance optimization tuning

---

### 7.2 Implementation Sequence

**Phase 1** (Week 1-2): MVP Core
```
Day 1-4:   Implement NodeCodegenOrchestrator
Day 5-7:   Implement NodeCodegenMetricsReducer
Day 8-11:  Implement omniclaude CLI integration
Day 12-14: Testing, bug fixes, documentation
```

**Phase 2** (Week 3-4): MVP Completion + Release Prep
```
Day 15-18: Implement NodeCodegenRegistry
Day 19-21: Implement quality validation and ONEX compliance
Day 22-25: Implement monitoring and alerting
Day 26-28: Load testing, performance tuning
```

**Phase 3** (Week 5-6): Release Enhancements
```
Day 29-35: Advanced workflow orchestration
Day 36-42: Multi-node coordination
Day 43-45: Distributed tracing and observability
Day 46-48: Final testing and production deployment
```

---

### 7.3 Key Takeaways (✅ MVP Delivered October 2025)

**Strengths** - ✅ **All Validated in Production**:
- ✅ **90% reusable patterns**: Existing orchestrators, reducers, and event infrastructure are production-ready - **VALIDATED**
- ✅ **Battle-tested resilience**: Circuit breakers, DLQ, retry logic, graceful degradation - **PRODUCTION-PROVEN**
- ✅ **Comprehensive testing**: 92.8% test coverage (501 tests) - **ACHIEVED**
- ✅ **LlamaIndex integration**: Event-driven workflow orchestration already in use - **COMPLETE**

**Original Gaps** - ✅ **All Addressed in MVP**:
- ✅ **Node generation workflow**: NodeCodegenOrchestrator implemented with 95% pattern reuse - **DELIVERED**
- ⏳ **Multi-node coordination**: Deferred to post-MVP enhancement
- ⏳ **Advanced monitoring**: Partial (Prometheus/Grafana dashboards planned for future)

**Strategic Decisions** - ✅ **Successfully Executed**:
- 🎯 **Reuse over rebuild**: Successfully adapted `NodeBridgeWorkflowOrchestrator` pattern - **VALIDATED**
- 🎯 **MVP focus**: Shipped complete node generation (6-step workflow) in October 2025 - **DELIVERED**
- 🎯 **Incremental enhancement**: Advanced features deferred to post-MVP phase - **ON TRACK**
- 🎯 **Leverage existing infrastructure**: Event bus, circuit breakers, health checks in production - **PRODUCTION-READY**

---

## Appendix A: Architecture Diagrams

### A.1 Current Architecture (MVP Foundation)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     omninode_bridge Repository                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Orchestrators (Workflow Coordination)                            │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ • NodeBridgeWorkflowOrchestrator (LlamaIndex, 6-step pipeline)   │  │
│  │   - validate_input → generate_hash → create_stamp →              │  │
│  │   - enrich_intelligence → persist_state → complete_workflow      │  │
│  │ • Performance: <2000ms e2e, 100+ workflows/sec                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Reducers (Metrics Aggregation)                                   │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ • NodeBridgeReducer (Streaming, FSM state management)            │  │
│  │   - Aggregation types: Namespace, Time window, File type, etc.   │  │
│  │   - FSMStateManager: Contract-driven state transitions           │  │
│  │ • Performance: >1000 items/sec, <100ms latency                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Service Mesh (Infrastructure)                                    │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ • KafkaClient: Event bus with DLQ, circuit breakers, retry       │  │
│  │ • PostgresClient: Connection pooling, prepared statements        │  │
│  │ • OnexTreeClient: HTTP client with circuit breaker               │  │
│  │ • ConsulClient: Service discovery (implemented, not active)      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Infrastructure Patterns                                          │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ • Circuit Breakers: Kafka, PostgreSQL, HTTP clients              │  │
│  │ • Health Checks: HealthCheckMixin with component-level tracking  │  │
│  │ • Monitoring: OpenTelemetry, structured logging, metrics         │  │
│  │ • Rate Limiting: Token bucket, leaky bucket, Redis-backed        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### A.2 Proposed MVP Architecture (Node Generation)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Node Generation Workflow (MVP)                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│ omniclaude  │  User Command: omniclaude generate node --prd requirements.md
│    CLI      │
└──────┬──────┘
       │
       │ 1. Publish PRD analysis request
       ▼
┌──────────────────┐
│  Kafka Topic     │  omninode_codegen_request_analyze_v1
│ request_analyze  │
└──────────────────┘
       │
       │ 2. Consume request
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   omniarchon Intelligence Service                       │
│  • RAG query for similar nodes (Qdrant vector search)                  │
│  • Pattern matching across projects (Memgraph graph traversal)         │
│  • Mixin recommendations (quality-weighted search)                      │
└────────────────────┬────────────────────────────────────────────────────┘
       │
       │ 3. Publish intelligence results
       ▼
┌──────────────────┐
│  Kafka Topic     │  omninode_codegen_response_analyze_v1
│ response_analyze │
└──────────────────┘
       │
       │ 4. Consume results + trigger workflow
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│        NodeCodegenOrchestrator (LlamaIndex Workflow)                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 1: validate_prd        → PRDValidatedEvent                       │
│    • Validate PRD structure, extract requirements                       │
│                                                                          │
│  Step 2: gather_intelligence → IntelligenceGatheredEvent               │
│    • Consume Archon intelligence from Kafka response                   │
│    • Extract patterns, mixins, similar nodes                            │
│                                                                          │
│  Step 3: detect_node_type    → NodeTypeDetectedEvent                   │
│    • Classify: Effect/Compute/Reducer/Orchestrator                      │
│    • Select appropriate template                                        │
│                                                                          │
│  Step 4: generate_code       → CodeGeneratedEvent                      │
│    • Apply template + intelligence                                      │
│    • Generate: node.py, models/, contracts/contract.yaml               │
│                                                                          │
│  Step 5: validate_quality    → QualityValidatedEvent                   │
│    • ONEX compliance check (naming, imports, contracts)                 │
│    • Type checking (mypy), linting (ruff)                               │
│    • Quality scoring (0.0-1.0)                                          │
│                                                                          │
│  Step 6: persist_generation  → GenerationCompletedEvent                │
│    • Save generated files to disk                                       │
│    • Persist metadata to PostgreSQL                                     │
│    • Publish completion event to Kafka                                  │
└────────────────────┬───────────────────┬────────────────────────────────┘
                     │                   │
                     │ Parallel          │ Parallel
                     ▼                   ▼
        ┌───────────────────┐   ┌──────────────────────┐
        │ NodeCodegen       │   │ NodeCodegenRegistry  │
        │ MetricsReducer    │   │ (Node Discovery)     │
        │                   │   │                      │
        │ • Aggregation by  │   │ • Store metadata     │
        │   node type       │   │ • Search/discovery   │
        │ • Quality metrics │   │ • Version tracking   │
        │ • FSM tracking    │   │ • Introspection      │
        └───────────────────┘   └──────────────────────┘
                     │                   │
                     │ Store             │ Store
                     ▼                   ▼
         ┌──────────────────────────────┐
         │       PostgreSQL             │
         │ • generation_metrics         │
         │ • generated_nodes_registry   │
         │ • workflow_executions        │
         │ • fsm_workflow_states        │
         └──────────────────────────────┘
```

---

## Appendix B: Reference Implementation Patterns

### B.1 LlamaIndex Workflow Pattern (from NodeBridgeWorkflowOrchestrator)

```python
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Context

class NodeCodegenOrchestrator(Workflow):
    """
    Code generation workflow using LlamaIndex pattern
    Adapted from NodeBridgeWorkflowOrchestrator
    """

    def __init__(self, container: ModelOnexContainer, timeout: int = 60):
        super().__init__(timeout=timeout, verbose=True)
        self.container = container
        self.kafka_client = container.get_service("kafka_client")

    @step(pass_context=True)
    async def validate_prd(
        self, ctx: Context, ev: StartEvent
    ) -> Union[PRDValidatedEvent, StopEvent]:
        """Step 1: Validate PRD structure"""
        try:
            start_time = time.time()

            # Extract PRD content from StartEvent
            prd_content = safe_get(ev, "prd_content")
            correlation_id = safe_get(ev, "correlation_id", uuid4())

            # Store in context
            ctx.data["correlation_id"] = correlation_id
            ctx.data["workflow_start_time"] = start_time

            # Validate PRD structure
            if not prd_content:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="PRD content is required"
                )

            # Publish Kafka event
            await self._publish_kafka_event(
                "PRD_VALIDATED",
                {"correlation_id": str(correlation_id)}
            )

            return PRDValidatedEvent(
                correlation_id=correlation_id,
                prd_content=prd_content,
                validation_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as error:
            return await self.handle_workflow_error(
                ctx=ctx, error=error, step_name="validate_prd"
            )

    @step(pass_context=True)
    async def gather_intelligence(
        self, ctx: Context, ev: PRDValidatedEvent
    ) -> Union[IntelligenceGatheredEvent, StopEvent]:
        """Step 2: Gather intelligence from Archon MCP"""
        # Implementation here...

    # Additional steps follow same pattern...
```

### B.2 Reducer Pattern (from NodeBridgeReducer)

```python
class NodeCodegenMetricsReducer(NodeReducer):
    """
    Metrics aggregation for node generation
    Adapted from NodeBridgeReducer pattern
    """

    async def execute_reduction(
        self, contract: ModelContractReducer
    ) -> ModelReducerOutputState:
        """Pure aggregation function"""

        # Initialize aggregation state
        aggregated_data = defaultdict(lambda: {
            "total_nodes": 0,
            "success_rate": 0.0,
            "avg_quality_score": 0.0,
        })

        # Stream and aggregate data
        async for metadata_batch in self._stream_metadata(contract):
            for metadata in metadata_batch:
                namespace = metadata.node_type  # Group by node type
                aggregated_data[namespace]["total_nodes"] += 1
                # ... more aggregation logic

        # Return results with intents
        return ModelReducerOutputState(
            aggregation_type=EnumAggregationType.NODE_TYPE_GROUPING,
            total_items=total_items,
            aggregations=dict(aggregated_data),
            fsm_states=fsm_states,
            intents=intents,  # Side effects (persist, publish events)
        )
```

---

## Appendix C: Dependency Matrix

### C.1 Python Dependencies (from pyproject.toml)

**Core Dependencies**:
- `pydantic ^2.11.7` - Data validation and serialization
- `fastapi ^0.115.0` - API framework (if REST endpoints needed)
- `pyyaml ^6.0.2` - YAML contract parsing
- `aiohttp ^3.12.14` - Async HTTP client

**Database**:
- `asyncpg ^0.29.0` - PostgreSQL async driver
- `alembic ^1.13.3` - Database migrations

**Event Bus**:
- `aiokafka ^0.11.0` - Async Kafka client
- `redis ^6.0.0` - Redis/Valkey for caching and rate limiting

**Observability**:
- `opentelemetry-api ^1.27.0` - Distributed tracing
- `opentelemetry-sdk ^1.27.0` - Tracing SDK

**LlamaIndex**:
- `llama-index-core` - Workflow orchestration framework

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-22 | Analysis | Initial comprehensive analysis |

---

**End of Document**
