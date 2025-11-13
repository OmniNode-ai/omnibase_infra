# Metrics and Registry System Guide

**Version:** 1.0.0
**Last Updated:** 2025-10-24
**ONEX v2.0 Compliance:** Full

## Overview

This guide covers the **Metrics & Registry System** for omninode_bridge, which provides:

1. **Code Generation Metrics Aggregation** - Real-time metrics collection and aggregation for code generation workflows
2. **FSM State Tracking** - Workflow state management with FSM pattern
3. **Node Registry** - Service discovery and dual registration system (Consul + PostgreSQL)
4. **Search & Discovery API** - Comprehensive node search and capability-based discovery

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kafka Event Bus                          │
│  (codegen.generation-started, stage-completed, etc.)        │
└───────────────┬─────────────────────────────────────────────┘
                │
                │ Events
                ↓
┌───────────────────────────────────────────────────────────────┐
│          CodegenMetricsAggregator (Reducer)                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Aggregation Strategies:                                 │  │
│  │  • NODE_TYPE_GROUPING (Effect/Compute/Reducer/Orch)     │  │
│  │  • QUALITY_BUCKETS (Low/Medium/High)                    │  │
│  │  • TIME_WINDOW (Hourly/Daily/Weekly/Monthly)            │  │
│  │  • DOMAIN_GROUPING (API/ML/Data)                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  Performance: >1000 events/sec, <100ms aggregation            │
└──────────────────────┬─────────────────────────────────────────┘
                       │
                       │ Aggregated Metrics
                       ↓
┌───────────────────────────────────────────────────────────────┐
│                 PostgreSQL Database                           │
│  • codegen_metrics_aggregated                                 │
│  • workflow_state_history                                     │
│  • workflow_state_current                                     │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│                FSMStateTracker                                │
│  States: PENDING → ANALYZING → GENERATING →                  │
│          VALIDATING → COMPLETED/FAILED                        │
│  Features: Guard conditions, transition validation            │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│              NodeRegistryService (Effect)                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Dual Registration:                                      │  │
│  │  • Consul (service discovery)                            │  │
│  │  • PostgreSQL (tool orchestration)                       │  │
│  │  Fallback: Graceful degradation on backend failures     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  Performance: <100ms registration, 50+ ops/sec                │
└──────────────────────┬─────────────────────────────────────────┘
                       │
                       │ Query/Search
                       ↓
┌───────────────────────────────────────────────────────────────┐
│                    SearchAPI                                  │
│  • Full-text search                                           │
│  • Capability-based discovery                                 │
│  • Version filtering                                          │
│  • Pagination & sorting                                       │
│  Performance: <50ms queries                                   │
└───────────────────────────────────────────────────────────────┘
```

## Component 1: Code Generation Metrics Aggregator

### Purpose

Aggregates code generation metrics from Kafka events with real-time streaming processing and multi-dimensional analysis.

### Features

- **4 Aggregation Strategies**:
  - NODE_TYPE_GROUPING: By ONEX node type
  - QUALITY_BUCKETS: By quality score ranges
  - TIME_WINDOW: By time windows
  - DOMAIN_GROUPING: By application domain

- **Statistical Computations**:
  - Duration: avg, p50, p95, p99, min, max
  - Quality: avg, min, max
  - Cost: total tokens, total cost USD
  - Stage: per-stage performance

- **Performance**:
  - >1000 events/second throughput
  - <100ms aggregation latency for 1000 items
  - <50ms per operation target

### Usage

```python
from src.omninode_bridge.reducers.codegen_metrics_aggregator import (
    CodegenMetricsAggregator,
    ModelNodeGenerationCompleted,
    EnumNodeType,
    EnumDomain,
)

# Initialize aggregator
aggregator = CodegenMetricsAggregator(
    max_buffer_size=10000,
    flush_interval_seconds=60,
    enable_postgres_persistence=True,
)

# Process generation event
completion_event = ModelNodeGenerationCompleted(
    workflow_id=workflow_id,
    node_type=EnumNodeType.EFFECT,
    domain=EnumDomain.API,
    total_duration_seconds=2.5,
    quality_score=0.88,
    total_tokens=1200,
    total_cost_usd=0.12,
)

await aggregator.process_event(completion_event)

# Get metrics
metrics = await aggregator.get_metrics()
print(f"Events processed: {metrics['events_processed_total']}")
print(f"Active workflows: {metrics['active_workflows']}")
```

### Event Models

**NODE_GENERATION_STARTED**:
```python
{
  "workflow_id": "uuid",
  "node_type": "effect|compute|reducer|orchestrator",
  "domain": "api|ml|data|...",
  "timestamp": "ISO-8601"
}
```

**NODE_GENERATION_STAGE_COMPLETED**:
```python
{
  "workflow_id": "uuid",
  "stage_name": "contract_generation",
  "stage_number": 1,
  "duration_ms": 150,
  "success": true,
  "tokens_consumed": 100,
  "cost_usd": 0.01
}
```

**NODE_GENERATION_COMPLETED**:
```python
{
  "workflow_id": "uuid",
  "node_type": "effect",
  "domain": "api",
  "total_duration_seconds": 2.5,
  "quality_score": 0.88,
  "total_tokens": 1200,
  "total_cost_usd": 0.12
}
```

**NODE_GENERATION_FAILED**:
```python
{
  "workflow_id": "uuid",
  "node_type": "effect",
  "failed_stage": "validation",
  "error_message": "Quality score below threshold"
}
```

### Aggregation Output

```python
{
  "aggregation_id": "uuid",
  "window_type": "hourly",
  "window_start": "2025-10-24T10:00:00Z",
  "window_end": "2025-10-24T11:00:00Z",
  "aggregation_type": "node_type_grouping",
  "group_key": "effect",
  "total_generations": 150,
  "successful_generations": 142,
  "failed_generations": 8,
  "success_rate": 0.9467,
  "duration_statistics": {
    "avg_seconds": 2.3,
    "p50_seconds": 2.1,
    "p95_seconds": 3.5,
    "p99_seconds": 4.2
  },
  "quality_statistics": {
    "avg_score": 0.87,
    "min_score": 0.62,
    "max_score": 0.98
  },
  "cost_statistics": {
    "total_tokens": 180000,
    "total_cost_usd": 18.00,
    "avg_tokens_per_generation": 1200,
    "avg_cost_per_generation": 0.12
  }
}
```

## Component 2: FSM State Tracker

### Purpose

Tracks workflow state transitions through the 6-phase code generation process with guard conditions and history tracking.

### FSM States

```
PENDING → ANALYZING → GENERATING → VALIDATING → COMPLETED
                                                ↓
                                              FAILED
```

**State Descriptions**:
- **PENDING**: Initial state, awaiting requirements
- **ANALYZING**: Intelligence gathering phase
- **GENERATING**: Code generation phase
- **VALIDATING**: Quality validation phase
- **COMPLETED**: Successful completion (terminal)
- **FAILED**: Failure at any stage (terminal, can retry)

### State Transitions

| From State | To State | Event | Guard Condition |
|-----------|----------|-------|-----------------|
| PENDING | ANALYZING | START_ANALYSIS | has_requirements |
| ANALYZING | GENERATING | START_GENERATION | has_analysis_results |
| GENERATING | VALIDATING | START_VALIDATION | has_generated_code |
| VALIDATING | COMPLETED | COMPLETE_WORKFLOW | validation_passed |
| Any | FAILED | FAIL_WORKFLOW | none |
| FAILED | PENDING | RETRY_WORKFLOW | none |

### Usage

```python
from src.omninode_bridge.reducers.fsm_state_tracker import (
    FSMStateTracker,
    EnumWorkflowState,
    EnumTransitionEvent,
)

# Initialize tracker
tracker = FSMStateTracker(enable_postgres_persistence=True)

# Initialize workflow
workflow_id = uuid4()
await tracker.initialize_workflow(
    workflow_id=workflow_id,
    initial_state=EnumWorkflowState.PENDING,
    metadata={"node_type": "effect", "domain": "api"},
)

# Transition to analyzing
success, error = await tracker.transition_state(
    workflow_id=workflow_id,
    to_state=EnumWorkflowState.ANALYZING,
    event=EnumTransitionEvent.START_ANALYSIS,
    metadata={"has_requirements": True},
)

if not success:
    print(f"Transition failed: {error}")

# Get current state
workflow = await tracker.get_workflow_state(workflow_id)
print(f"Current state: {workflow.current_state}")
print(f"Transition count: {workflow.transition_count}")

# Get transition history
history = await tracker.get_transition_history(workflow_id)
for transition in history:
    print(f"{transition.from_state} → {transition.to_state} ({transition.event})")
```

### Performance

- <10ms per state transition
- <5ms for state lookups
- Support for 10,000+ concurrent workflows

## Component 3: Node Registry Service

### Purpose

Manages node registration with dual-backend strategy for service discovery (Consul) and tool orchestration (PostgreSQL).

### Features

- **Dual Registration**:
  - Consul: Service discovery and health monitoring
  - PostgreSQL: Tool orchestration and queryable registry
  - Graceful degradation on backend failures

- **Health Monitoring**:
  - Heartbeat tracking
  - Health status: healthy, degraded, unhealthy, unknown
  - Automatic health updates

- **Capabilities Tracking**:
  - Node capabilities registry
  - Capability-based discovery
  - Version tracking

### Usage

```python
from src.omninode_bridge.registry.node_registry_service import (
    NodeRegistryService,
    ModelNodeIntrospection,
    ModelNodeRegistrationInput,
    ModelCapability,
    EnumNodeType,
)

# Initialize registry
registry = NodeRegistryService(
    enable_consul=True,
    enable_postgres=True,
    enable_kafka=True,
)

# Register node
node_introspection = ModelNodeIntrospection(
    node_id="my-effect-node-001",
    node_name="MyEffectNode",
    node_type=EnumNodeType.EFFECT,
    version="1.0.0",
    capabilities=[
        ModelCapability(
            name="http_api",
            description="HTTP API endpoint for external requests"
        ),
        ModelCapability(
            name="data_validation",
            description="Input data validation"
        ),
    ],
    metadata={
        "author": "my-team",
        "domain": "api",
        "deployment": "production",
    },
)

result = await registry.register_node(
    ModelNodeRegistrationInput(node_introspection=node_introspection)
)

if result.success:
    print(f"Node registered: {result.registration_result.node_id}")
    print(f"Consul: {result.registration_result.consul_registered}")
    print(f"PostgreSQL: {result.registration_result.postgres_registered}")
else:
    print(f"Registration failed: {result.error_message}")

# Update health
await registry.update_node_health("my-effect-node-001", EnumHealthStatus.HEALTHY)

# Query nodes
query_result = await registry.query_nodes(
    ModelNodeQueryInput(
        query_filters={
            "node_type": "effect",
            "capability": "http_api",
            "health_status": "healthy",
        }
    )
)

print(f"Found {query_result.total_count} nodes")
for node in query_result.nodes:
    print(f"  - {node['node_name']} ({node['version']})")

# Deregister node
await registry.deregister_node("my-effect-node-001")
```

### Performance

- <100ms registration operations
- 50+ registrations per second
- 99% dual-registration consistency target

## Component 4: Search & Discovery API

### Purpose

Provides comprehensive search and discovery capabilities for registered nodes with full-text search, filtering, and pagination.

### Features

- **Full-Text Search**: Search across node names, capabilities, and metadata
- **Multi-Facet Filtering**: Node type, health status, capabilities, version ranges
- **Pagination & Sorting**: Efficient result navigation
- **Relevance Scoring**: Ranked search results
- **Performance**: <50ms queries for typical searches

### Usage

```python
from src.omninode_bridge.registry.search_api import (
    SearchAPI,
    ModelSearchQuery,
    EnumSortBy,
    EnumSortOrder,
)

# Initialize search API
search_api = SearchAPI(registry_service=registry)

# Full-text search
query = ModelSearchQuery(
    query="API endpoint",
    node_type=EnumNodeType.EFFECT,
    health_status=EnumHealthStatus.HEALTHY,
    page=1,
    page_size=10,
    sort_by=EnumSortBy.RELEVANCE,
    sort_order=EnumSortOrder.DESC,
)

response = await search_api.search(query)

print(f"Found {response.total_count} nodes ({response.total_pages} pages)")
for result in response.results:
    print(f"  {result.node_name} (relevance: {result.relevance_score:.2f})")
    print(f"    Type: {result.node_type}, Health: {result.health_status}")
    print(f"    Capabilities: {[c['name'] for c in result.capabilities]}")

# Discover by capability
nodes_with_kafka = await search_api.discover_by_capability("kafka_consumer")

# Discover by type
orchestrators = await search_api.discover_by_type(EnumNodeType.ORCHESTRATOR)

# Get healthy nodes by type
healthy_effects = await search_api.get_healthy_nodes_by_type(EnumNodeType.EFFECT)
```

### Search Filters

- `query`: Text search across names and capabilities
- `node_type`: Filter by ONEX node type
- `health_status`: Filter by health status
- `capability`: Filter by capability name
- `version`: Filter by exact version
- `version_min`/`version_max`: Version range filtering
- `quality_score_min`: Minimum quality score (future)

### Sorting Options

- `RELEVANCE`: Sort by search relevance score
- `NAME`: Alphabetical by node name
- `VERSION`: Semantic version ordering
- `REGISTRATION_TIME`: By registration timestamp
- `LAST_HEARTBEAT`: By most recent heartbeat

## Database Schema

### Tables

**registered_nodes**:
```sql
CREATE TABLE registered_nodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_name VARCHAR(255) NOT NULL,
    node_type VARCHAR(50) NOT NULL,
    version VARCHAR(50) NOT NULL,
    capabilities JSONB DEFAULT '[]',
    endpoints JSONB DEFAULT '{}',
    health_status VARCHAR(50) DEFAULT 'unknown',
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    registration_timestamp TIMESTAMP WITH TIME ZONE,
    consul_registered BOOLEAN DEFAULT FALSE,
    postgres_registered BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'
);
```

**codegen_metrics_aggregated**:
```sql
CREATE TABLE codegen_metrics_aggregated (
    aggregation_id UUID PRIMARY KEY,
    window_type VARCHAR(50) NOT NULL,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,
    aggregation_type VARCHAR(50) NOT NULL,
    group_key VARCHAR(255) NOT NULL,
    total_generations INTEGER DEFAULT 0,
    successful_generations INTEGER DEFAULT 0,
    failed_generations INTEGER DEFAULT 0,
    success_rate NUMERIC(5, 4) DEFAULT 0.0,
    duration_statistics JSONB,
    quality_statistics JSONB,
    cost_statistics JSONB,
    UNIQUE (window_type, window_start, aggregation_type, group_key)
);
```

**workflow_state_history**:
```sql
CREATE TABLE workflow_state_history (
    transition_id UUID PRIMARY KEY,
    workflow_id UUID NOT NULL,
    from_state VARCHAR(50) NOT NULL,
    to_state VARCHAR(50) NOT NULL,
    event VARCHAR(50) NOT NULL,
    transition_timestamp TIMESTAMP WITH TIME ZONE,
    guard_conditions_met BOOLEAN DEFAULT TRUE,
    reason TEXT,
    metadata JSONB DEFAULT '{}'
);
```

**workflow_state_current**:
```sql
CREATE TABLE workflow_state_current (
    workflow_id UUID PRIMARY KEY,
    current_state VARCHAR(50) NOT NULL,
    previous_state VARCHAR(50),
    state_entry_time TIMESTAMP WITH TIME ZONE,
    transition_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);
```

### Migration

Run the migration to create all tables:

```bash
psql -U postgres -d omninode_bridge -f migrations/add_node_registry_tables.sql
```

## Monitoring and Alerting

### Grafana Dashboard

Import the dashboard from `monitoring/grafana_dashboard_codegen.json`.

**Key Metrics**:

1. **Generation Success Rate**: Percentage of successful generations
2. **Generations Per Hour**: Throughput over time
3. **Average Generation Duration**: p50/p95/p99 latencies
4. **Average Quality Score**: Overall quality metric
5. **Generations by Node Type**: Distribution pie chart
6. **Quality Bucket Distribution**: Low/Medium/High classification
7. **Domain Distribution**: Generations by domain
8. **Cost Tracking**: Total cost and token usage
9. **Workflow State Distribution**: FSM state distribution
10. **Registered Nodes by Type**: Registry node counts
11. **Registry Health Status**: Node health distribution
12. **Dual Registration Consistency**: Consul/PostgreSQL sync metric

### Prometheus Metrics

**Metrics Aggregator**:
- `codegen_generations_total` - Total generations counter
- `codegen_generations_successful_total` - Successful generations
- `codegen_generations_failed_total` - Failed generations
- `codegen_generation_duration_seconds` - Duration histogram
- `codegen_quality_score` - Quality score gauge
- `aggregation_duration_seconds` - Aggregation latency histogram
- `aggregation_buffer_size` - Buffer utilization gauge
- `aggregation_items_per_second` - Throughput gauge

**FSM State Tracker**:
- `workflow_state_transitions_total` - State transition counter
- `workflow_state_count` - Current state distribution gauge
- `active_workflows_count` - Active workflows gauge

**Node Registry**:
- `registrations_total` - Total registrations counter
- `registrations_successful_total` - Successful registrations
- `registration_duration_seconds` - Registration latency histogram
- `registered_nodes_total` - Total registered nodes gauge
- `consul_postgres_consistency` - Dual registration consistency gauge

## Testing

### Run Integration Tests

```bash
# Test metrics aggregator
pytest tests/integration/test_metrics_reducer.py -v

# Test node registry
pytest tests/integration/test_node_registry.py -v

# Run all integration tests
pytest tests/integration/ -v --asyncio-mode=auto
```

### Test Coverage

- Unit tests: >90% coverage target
- Integration tests: End-to-end workflows
- Performance tests: Validate latency and throughput targets

## Performance Requirements

| Component | Metric | Target | Max |
|-----------|--------|--------|-----|
| Metrics Aggregator | Throughput | >1000 events/sec | - |
| Metrics Aggregator | Aggregation Latency | <100ms for 1000 items | <500ms |
| Metrics Aggregator | Operation Time | <50ms per operation | <100ms |
| FSM State Tracker | Transition Time | <10ms | <50ms |
| FSM State Tracker | Lookup Time | <5ms | <10ms |
| Node Registry | Registration Time | <100ms | <500ms |
| Node Registry | Throughput | >50 ops/sec | - |
| Search API | Query Time | <50ms | <100ms |

## Troubleshooting

### High Aggregation Latency

**Symptoms**: Aggregation taking >100ms

**Solutions**:
1. Reduce buffer size
2. Increase flush interval
3. Check buffer usage metrics
4. Verify PostgreSQL performance

### Registry Inconsistency

**Symptoms**: Consul and PostgreSQL out of sync

**Solutions**:
1. Check `consul_postgres_consistency` metric
2. Run reconciliation (scheduled every 5 minutes)
3. Verify both backends healthy
4. Check network connectivity

### Search Performance Degradation

**Symptoms**: Queries taking >50ms

**Solutions**:
1. Check node count (optimize for <1000 nodes)
2. Add database indexes on query fields
3. Reduce page size
4. Use more specific filters

## Best Practices

1. **Metrics Aggregation**:
   - Use appropriate buffer sizes (1000-10000)
   - Set flush intervals based on latency requirements
   - Monitor buffer usage to prevent overflow
   - Enable PostgreSQL persistence for durability

2. **FSM State Tracking**:
   - Always check guard conditions before transitions
   - Log transition failures for debugging
   - Cleanup terminal workflows periodically
   - Use correlation IDs for tracing

3. **Node Registry**:
   - Update health status regularly (every 30-60s)
   - Include detailed capabilities for discovery
   - Use meaningful metadata for search
   - Monitor dual registration consistency

4. **Search & Discovery**:
   - Use pagination for large result sets
   - Combine filters for precise queries
   - Cache frequently-used searches
   - Monitor query performance

## Future Enhancements

- [ ] Kafka integration for real-time event streaming
- [ ] Consul client implementation for service discovery
- [ ] Quality score integration in search
- [ ] Advanced analytics and trend analysis
- [ ] Alert rules and thresholds
- [ ] Horizontal scaling for high throughput
- [ ] Machine learning for anomaly detection

## References

- [Bridge Nodes Guide](./BRIDGE_NODES_GUIDE.md)
- [Architecture Guide](../architecture/ARCHITECTURE.md)
- [Database Guide](../database/DATABASE_GUIDE.md)
- ONEX v2.0 Specification (see Archon repository: `docs/ONEX_ARCHITECTURE_PATTERNS_COMPLETE.md`)
- Contract: `contracts/reducers/codegen_metrics.yaml`
- Contract: `contracts/registry/node_registry.yaml`
