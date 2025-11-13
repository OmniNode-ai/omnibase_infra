# Bridge Nodes Implementation Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-02
**Status**: Phase 1 Partially Complete

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Status](#implementation-status)
4. [NodeBridgeReducer](#nodebridgereducer)
5. [NodeBridgeOrchestrator](#nodebridgeorchestrator-pending)
6. [Contract System](#contract-system)
7. [FSM State Machine](#fsm-state-machine)
8. [Workflow Patterns](#workflow-patterns)
9. [Integration Guide](#integration-guide)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The **omninode_bridge** project implements ONEX v2.0 compliant bridge nodes that coordinate between MetadataStampingService and OnexTree intelligence services. Bridge nodes follow the **contract-driven architecture** pattern where functionality is composed through **subcontracts** rather than inheritance.

### Core Principles

- **Contract-Driven**: Nodes are configured via YAML contracts with subcontract composition
- **Suffix-Based Naming**: All ONEX entities use suffix-based naming (e.g., `NodeBridgeReducer`, `ModelBridgeState`)
- **Dependency Injection**: ModelONEXContainer provides all service dependencies
- **Streaming Architecture**: Async iterators for efficient data processing
- **FSM-Driven Workflows**: Finite state machines manage workflow states

### Bridge System Purpose

```
MetadataStampingService
         ↓ (stamp requests)
NodeBridgeOrchestrator ←→ OnexTree Intelligence
         ↓ (stamped content + metadata)
NodeBridgeReducer
         ↓ (aggregated state)
PostgreSQL Bridge State Store
```

**Data Flow**:
1. **Input**: Stamp requests from MetadataStampingService
2. **Orchestration**: NodeBridgeOrchestrator coordinates stamping workflow
3. **Intelligence**: OnexTree provides AI-enhanced validation/routing (optional)
4. **Reduction**: NodeBridgeReducer aggregates metadata and state
5. **Storage**: Bridge state persisted with FSM tracking

---

## Architecture

### ONEX v2.0 Node Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ModelONEXContainer                       │
│  (Dependency Injection: PostgreSQL, Kafka, OnexTree, etc.)  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   NodeBridgeOrchestrator                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Contract: ModelContractOrchestrator                   │  │
│  │  - workflow_coordination (subcontract)                │  │
│  │  - routing (subcontract)                              │  │
│  │  - fsm (subcontract)                                  │  │
│  │  - event_type (subcontract)                           │  │
│  └───────────────────────────────────────────────────────┘  │
│  │                                                            │
│  ├─► execute_orchestration(contract) → StampResponse        │
│  ├─► Route to MetadataStampingService                       │
│  ├─► Route to OnexTree (optional intelligence)              │
│  ├─► Manage FSM state transitions                           │
│  └─► Publish Kafka events                                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    NodeBridgeReducer                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Contract: ModelContractReducer                        │  │
│  │  - aggregation (subcontract)                          │  │
│  │  - state_management (subcontract)                     │  │
│  │  - fsm (subcontract)                                  │  │
│  └───────────────────────────────────────────────────────┘  │
│  │                                                            │
│  ├─► execute_reduction(contract) → AggregatedState          │
│  ├─► Stream metadata from orchestrator                      │
│  ├─► Aggregate by namespace                                 │
│  ├─► Track FSM states                                       │
│  └─► Persist to PostgreSQL                                  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
           ┌──────────────┐
           │  PostgreSQL  │
           │ Bridge State │
           └──────────────┘
```

### Directory Structure

```
src/omninode_bridge/nodes/
│
├── orchestrator/
│   └── v1_0_0/
│       ├── __init__.py
│       ├── node.py                          # NodeBridgeOrchestrator (PENDING)
│       │
│       ├── contracts/
│       │   └── contract.yaml                # Main contract (PENDING)
│       │
│       └── models/                          # All models PENDING
│           ├── __init__.py
│           ├── model_input_state.py
│           ├── model_output_state.py
│           ├── enum_workflow_state.py
│           └── enum_workflow_event.py
│
└── reducer/
    └── v1_0_0/
        ├── __init__.py
        ├── node.py                          # NodeBridgeReducer (COMPLETE ✅)
        │
        ├── contracts/
        │   └── contract.yaml                # Main contract (PENDING)
        │
        └── models/
            ├── __init__.py
            ├── enum_aggregation_type.py     # COMPLETE ✅
            ├── model_bridge_state.py        # COMPLETE ✅
            ├── model_stamp_metadata_input.py # COMPLETE ✅
            ├── model_input_state.py         # PENDING (referenced by node)
            └── model_output_state.py        # PENDING (referenced by node)
```

---

## Implementation Status

### ✅ Completed

1. **NodeBridgeReducer (Core Implementation)**
   - Full `execute_reduction()` implementation with:
     - Streaming data aggregation via async iterators
     - Namespace-based grouping
     - FSM state tracking across workflows
     - Performance metrics (duration, items/sec)
     - State persistence hooks (pending actual DB integration)

2. **Reducer Models**
   - ✅ `EnumAggregationType` - Aggregation strategy types
   - ✅ `ModelBridgeState` - PostgreSQL-persisted state model
   - ✅ `ModelStampMetadataInput` - Stamp metadata input

### ⏳ Pending

1. **NodeBridgeOrchestrator**
   - ❌ Node implementation (placeholder only)
   - ❌ All models (placeholders)
   - ❌ Contract YAML

2. **Reducer Completion**
   - ❌ `ModelReducerInputState` - Input state model
   - ❌ `ModelReducerOutputState` - Output state model
   - ❌ Contract YAML with subcontracts
   - ❌ Actual PostgreSQL persistence implementation
   - ❌ Unit tests
   - ❌ Integration tests

3. **System Integration**
   - ❌ Orchestrator → Reducer workflow
   - ❌ MetadataStampingService integration
   - ❌ OnexTree intelligence integration
   - ❌ Kafka event publishing
   - ❌ End-to-end tests

---

## NodeBridgeReducer

**Status**: ✅ Core Implementation Complete

### Purpose

Aggregates stamping metadata across workflows and manages bridge state persistence.

### Key Features

- **Streaming Aggregation**: Async iterator-based data streaming
- **Namespace Grouping**: Multi-tenant aggregation with namespace isolation
- **FSM State Tracking**: Track workflow states across reduction windows
- **Windowed Processing**: Configurable time windows (default: 5000ms)
- **Batch Processing**: Configurable batch size (default: 100 items)
- **Performance Monitoring**: Track aggregation duration and throughput

### Implementation

```python
class NodeBridgeReducer(NodeReducer):
    """
    Bridge Reducer for metadata aggregation and state management.

    Aggregates stamping metadata:
    1. Receive stamp metadata stream from orchestrator
    2. Group by aggregation strategy (namespace, time window, etc.)
    3. Compute statistics (total stamps, namespaces, file types)
    4. Track FSM states for workflows
    5. Persist aggregated state to PostgreSQL
    6. Return aggregation results
    """

    async def execute_reduction(
        self,
        contract: ModelContractReducer,
    ) -> ModelReducerOutputState:
        """
        Execute metadata aggregation and state reduction.

        Returns:
            ModelReducerOutputState with:
            - aggregation_type: Strategy used
            - total_items: Number of items aggregated
            - total_size_bytes: Sum of all file sizes
            - namespaces: List of unique namespaces
            - aggregations: Detailed aggregation data per namespace
            - fsm_states: Workflow FSM states
            - aggregation_duration_ms: Performance metric
            - items_per_second: Throughput metric
        """
```

### Aggregation Strategies

Defined in `EnumAggregationType`:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `NAMESPACE_GROUPING` | Group by namespace (primary) | Multi-tenant aggregation |
| `TIME_WINDOW` | Group by time windows | Time-series analysis |
| `FILE_TYPE_GROUPING` | Group by content_type | File type statistics |
| `SIZE_BUCKETS` | Group by file size ranges | Size distribution analysis |
| `WORKFLOW_GROUPING` | Group by workflow_id | Workflow-level metrics |
| `CUSTOM` | Custom strategy via config | Advanced use cases |

### Streaming Configuration

```python
# Contract configuration (pending YAML implementation)
streaming:
  mode: "windowed"        # Time-based windowing
  window_size_ms: 5000    # 5 second windows
  batch_size: 100         # Process 100 items per batch
```

### State Persistence

**Model**: `ModelBridgeState`

```python
class ModelBridgeState(BaseModel):
    """
    Persistent bridge state for cumulative aggregation tracking.

    Database Mapping:
    - Table: bridge_state
    - Primary Key: state_id
    - Indexes: namespace, last_updated, current_fsm_state
    """

    # State identity
    state_id: UUID
    version: int                    # Optimistic locking

    # O.N.E. v0.1 compliance
    namespace: str
    metadata_version: str = "0.1"

    # Cumulative statistics
    total_stamps: int = 0
    total_size_bytes: int = 0
    unique_file_types: set[str]
    unique_workflows: set[str]

    # FSM tracking
    current_fsm_state: str
    fsm_state_history: list[dict[str, Any]]

    # Temporal tracking
    created_at: datetime
    last_updated: datetime
    last_aggregation_at: datetime | None

    # Performance metrics
    total_aggregations: int = 0
    avg_aggregation_duration_ms: float = 0.0
```

### Example Usage

```python
from omnibase_core.models.container import ModelONEXContainer
from omnibase_core.models.contracts import ModelContractReducer
from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

# Initialize with DI container
container = ModelONEXContainer(
    postgresql_client=postgresql_service,
    kafka_producer=kafka_service,
    # ... other services
)

# Create reducer instance
reducer = NodeBridgeReducer(container)

# Create contract with streaming input
contract = ModelContractReducer(
    correlation_id=uuid4(),
    input_stream=metadata_stream,  # Async iterator
)

# Execute reduction
result = await reducer.execute_reduction(contract)

print(f"Aggregated {result.total_items} items across {len(result.namespaces)} namespaces")
print(f"Duration: {result.aggregation_duration_ms:.2f}ms")
print(f"Throughput: {result.items_per_second:.0f} items/sec")
```

---

## NodeBridgeOrchestrator (PENDING)

**Status**: ⏳ Pending Implementation

### Purpose

Coordinates metadata stamping workflows by routing requests between MetadataStampingService and OnexTree intelligence.

### Planned Features

- **Workflow Coordination**: Multi-step stamping workflows
- **Service Routing**: Route to MetadataStamping and OnexTree services
- **FSM State Management**: Track workflow state transitions
- **Event Publishing**: Publish workflow events to Kafka
- **Error Handling**: Graceful degradation and retry logic

### Planned Contract Composition

```yaml
# orchestrator/v1_0_0/contracts/contract.yaml

contract_version: {major: 1, minor: 0, patch: 0}
node_type: ORCHESTRATOR

# Subcontracts
workflow_coordination:
  $ref: "./subcontracts/workflow_steps.yaml"

routing:
  $ref: "./subcontracts/routing_rules.yaml"

fsm:
  $ref: "./subcontracts/fsm_states.yaml"

event_type:
  $ref: "./subcontracts/events.yaml"
```

### Planned Workflow Steps

1. **Validate Input** - Validate stamp request data
2. **OnexTree Intelligence** (optional) - AI-enhanced validation/routing
3. **Hash Generation** - BLAKE3 hash via MetadataStampingService
4. **Stamp Creation** - Create stamp with O.N.E. v0.1 compliance
5. **Event Publishing** - Publish `stamp_created` event
6. **FSM Transition** - Update workflow state

---

## Contract System

### Contract Architecture

ONEX v2.0 uses **subcontract composition** to add capabilities to nodes:

```python
# Base contract
class ModelContractReducer(ModelContractBase):
    node_type: EnumNodeType = EnumNodeType.REDUCER
    correlation_id: UUID

    # Optional subcontracts
    aggregation: ModelAggregationSubcontract | None = None
    state_management: ModelStateManagementSubcontract | None = None
    fsm: ModelFSMSubcontract | None = None
    caching: ModelCachingSubcontract | None = None
```

### Available Subcontracts

From `omnibase_core.models.contracts.subcontracts`:

| Subcontract | Purpose | Nodes |
|-------------|---------|-------|
| `ModelAggregationSubcontract` | Data aggregation strategies | Reducer |
| `ModelStateManagementSubcontract` | State persistence (PostgreSQL) | Reducer, Orchestrator |
| `ModelFSMSubcontract` | Finite state machine patterns | Orchestrator, Reducer |
| `ModelWorkflowCoordinationSubcontract` | Workflow orchestration | Orchestrator |
| `ModelRoutingSubcontract` | Service routing & load balancing | Orchestrator |
| `ModelEventTypeSubcontract` | Event definitions & publishing | Orchestrator, Effect |
| `ModelCachingSubcontract` | Cache strategies | All |
| `ModelConfigurationSubcontract` | Configuration management | All |

### Import Paths

```python
# Infrastructure (node base classes)
from omnibase_core.infrastructure.node_orchestrator import NodeOrchestrator
from omnibase_core.infrastructure.node_reducer import NodeReducer

# Contracts
from omnibase_core.models.contracts.model_contract_orchestrator import (
    ModelContractOrchestrator,
)
from omnibase_core.models.contracts.model_contract_reducer import (
    ModelContractReducer,
)

# Subcontracts
from omnibase_core.models.contracts.subcontracts import (
    ModelAggregationSubcontract,
    ModelStateManagementSubcontract,
    ModelFSMSubcontract,
    ModelWorkflowCoordinationSubcontract,
    ModelRoutingSubcontract,
    ModelEventTypeSubcontract,
)

# Container
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
```

---

## FSM State Machine

### Orchestrator FSM States

```
┌─────────┐
│ pending │ (Initial state - workflow created)
└────┬────┘
     │ start_workflow
     ▼
┌────────────┐
│ processing │ (Workflow executing)
└─────┬──────┘
      │
      ├─► ┌───────────┐
      │   │ completed │ (Success - terminal state)
      │   └───────────┘
      │
      └─► ┌────────┐
          │ failed │ (Error - terminal state)
          └────────┘
```

**Transitions**:
- `pending → processing`: Workflow started
- `processing → completed`: All steps succeeded
- `processing → failed`: Step failed or timeout

### Reducer FSM States

```
┌──────┐
│ idle │ (No active aggregation)
└──┬───┘
   │ start_aggregation
   ▼
┌────────────┐
│ aggregating│ (Active aggregation window)
└─────┬──────┘
      │ window_complete
      ▼
┌────────────┐
│ persisting │ (Writing to PostgreSQL)
└─────┬──────┘
      │ persistence_complete
      ▼
┌──────┐
│ idle │ (Ready for next window)
└──────┘
```

**Transitions**:
- `idle → aggregating`: New aggregation window started
- `aggregating → persisting`: Window complete, saving state
- `persisting → idle`: State saved, ready for next window

---

## Workflow Patterns

### Pattern 1: Simple Stamping Workflow

```python
# 1. User requests stamp via MetadataStampingService API
POST /api/v1/metadata-stamping/stamp
{
    "content": "Hello World",
    "namespace": "my_app"
}

# 2. MetadataStampingService invokes NodeBridgeOrchestrator
orchestrator_input = {
    "content": "Hello World",
    "namespace": "my_app",
    "workflow_id": uuid4()
}

# 3. NodeBridgeOrchestrator coordinates:
#    - FSM: pending → processing
#    - Generate BLAKE3 hash
#    - Create stamp
#    - Publish 'stamp_created' event
#    - FSM: processing → completed

# 4. Stamp metadata flows to NodeBridgeReducer
reducer_input = {
    "stamp_id": "...",
    "file_hash": "...",
    "namespace": "my_app",
    "workflow_id": uuid4(),
    "workflow_state": "completed",
    ...
}

# 5. NodeBridgeReducer aggregates:
#    - Group by namespace
#    - Update statistics
#    - Track FSM state
#    - Persist to PostgreSQL

# 6. Return stamped content to user
```

### Pattern 2: Intelligence-Enhanced Workflow

```python
# 1. Same as Pattern 1, but with OnexTree routing

# 2. NodeBridgeOrchestrator adds intelligence step:
#    - Route to OnexTree for validation
#    - OnexTree analyzes content
#    - Returns intelligence data
#    - Orchestrator enriches stamp metadata

# 3. Enhanced metadata flows to reducer with intelligence_data:
reducer_input = {
    "stamp_id": "...",
    "intelligence_data": {
        "content_analysis": "...",
        "security_scan": "...",
        "ai_classification": "..."
    },
    ...
}

# 4. Reducer aggregates with intelligence insights
# 5. Persisted state includes intelligence metrics
```

### Pattern 3: Batch Stamping Workflow

```python
# 1. Batch request with multiple items
POST /api/v1/metadata-stamping/batch
{
    "items": [
        {"content": "File 1", "namespace": "ns1"},
        {"content": "File 2", "namespace": "ns2"},
        ...
    ]
}

# 2. NodeBridgeOrchestrator processes each item:
#    - Parallel workflow execution
#    - Shared correlation_id for batch tracking
#    - FSM state per item

# 3. NodeBridgeReducer streams batch results:
#    - Windowed aggregation (5s windows)
#    - Batch processing (100 items/batch)
#    - Namespace-based grouping
#    - Incremental state updates
```

---

## Integration Guide

### MetadataStampingService Integration

**Pending**: NodeBridgeOrchestrator implementation

```python
# Expected integration pattern:

# 1. MetadataStampingService creates stamp request
from omninode_bridge.nodes.orchestrator.v1_0_0 import NodeBridgeOrchestrator

async def create_stamp(content: str, namespace: str) -> dict:
    # Invoke orchestrator
    contract = ModelContractOrchestrator(
        correlation_id=uuid4(),
        input_state={
            "content": content,
            "namespace": namespace
        }
    )

    result = await orchestrator.execute_orchestration(contract)
    return result.to_dict()
```

### OnexTree Intelligence Integration

**Pending**: NodeBridgeOrchestrator routing implementation

```python
# Expected OnexTree routing:

async def _route_to_onextree(
    self,
    step: dict,
    contract: ModelContractOrchestrator
) -> dict:
    """Route to OnexTree for AI intelligence analysis."""

    onextree_client = self.container.get_service('onextree_client')

    intelligence_request = {
        "content": contract.input_state.content,
        "analysis_type": "security_and_classification"
    }

    result = await onextree_client.analyze(intelligence_request)
    return result
```

### PostgreSQL Integration

**Pending**: Actual database persistence implementation

```python
# Expected PostgreSQL persistence:

async def _persist_state(
    self,
    aggregated_data: dict[str, dict],
    fsm_states: dict[str, str],
    contract: ModelContractReducer,
) -> None:
    """Persist aggregated state to PostgreSQL."""

    postgresql_client = self.container.get_service('postgresql_client')

    async with postgresql_client.transaction() as tx:
        for namespace, data in aggregated_data.items():
            bridge_state = ModelBridgeState(
                namespace=namespace,
                total_stamps=data["total_stamps"],
                total_size_bytes=data["total_size_bytes"],
                # ...
            )

            await tx.execute(
                """
                INSERT INTO bridge_state (...)
                VALUES (...)
                ON CONFLICT (namespace)
                DO UPDATE SET ...
                """
            )
```

---

## Troubleshooting

### Common Issues

#### Issue: Import errors for subcontracts

```python
ImportError: cannot import name 'ModelAggregationSubcontract' from 'omnibase_core.models.contracts.subcontracts'
```

**Solution**: Check `omnibase_core` version. Subcontracts may not be available in all versions. The current implementation includes fallback handling:

```python
try:
    from omnibase_core.models.contracts.subcontracts import (
        ModelAggregationSubcontract,
    )
except ImportError:
    ModelAggregationSubcontract = None  # type: ignore
```

#### Issue: Missing model definitions

```python
ImportError: cannot import name 'ModelReducerInputState'
```

**Solution**: Model placeholders need to be implemented. See [Implementation Status](#implementation-status) for pending models.

#### Issue: Contract YAML not loaded

```python
FileNotFoundError: contract.yaml not found
```

**Solution**: Contract YAML files are currently placeholders. The node implementation is designed to work without them by using contract object defaults.

### Development Workflow

1. **Local Development**
   ```bash
   # Run in standalone mode (without omnibase runtime)
   cd src/omninode_bridge/nodes/reducer/v1_0_0
   python node.py  # Fallback mode
   ```

2. **With omnibase Runtime**
   ```bash
   # Run via omnibase node runtime
   omnibase run --node reducer --version 1.0.0 --contract contract.yaml
   ```

3. **Testing**
   ```bash
   # Unit tests
   pytest tests/unit/nodes/test_reducer.py

   # Integration tests
   pytest tests/integration/test_orchestrator_reducer_flow.py
   ```

### Migration Notes

**When omnibase_core is fully available:**

1. **Complete Model Implementations**
   - Implement `ModelReducerInputState`
   - Implement `ModelReducerOutputState`
   - Create Orchestrator models

2. **Complete Contract YAMLs**
   - Define subcontract references
   - Configure aggregation strategies
   - Define FSM states and transitions

3. **Implement PostgreSQL Persistence**
   - Replace `_log_state_persistence()` with actual DB writes
   - Add transaction management
   - Add connection pooling

4. **Add Kafka Integration**
   - Implement event publishing
   - Add event consumers for testing
   - Configure OnexEnvelopeV1 format

5. **Complete Orchestrator**
   - Implement `NodeBridgeOrchestrator`
   - Add service routing logic
   - Add FSM state management
   - Add workflow coordination

---

## Next Steps

### Immediate Priorities

1. **Complete Reducer Models**
   - [ ] Implement `ModelReducerInputState`
   - [ ] Implement `ModelReducerOutputState`
   - [ ] Add comprehensive field validation

2. **Complete Reducer Contract**
   - [ ] Create contract.yaml with subcontracts
   - [ ] Define aggregation strategies
   - [ ] Define FSM states
   - [ ] Add streaming configuration

3. **Implement Orchestrator**
   - [ ] Complete `NodeBridgeOrchestrator` implementation
   - [ ] Create orchestrator models
   - [ ] Create orchestrator contract
   - [ ] Add workflow coordination logic

4. **Add Tests**
   - [ ] Unit tests for NodeBridgeReducer
   - [ ] Unit tests for NodeBridgeOrchestrator
   - [ ] Integration tests for workflow
   - [ ] Performance tests

5. **System Integration**
   - [ ] PostgreSQL persistence
   - [ ] Kafka event publishing
   - [ ] MetadataStampingService integration
   - [ ] OnexTree intelligence integration

### Long-Term Roadmap

**Phase 2**: Advanced Features (Weeks 3-4)
- Enhanced metadata extraction with multi-modal capabilities
- Advanced caching with Redis
- Performance tuning and horizontal scaling
- Distributed tracing and monitoring

**Phase 3**: Production Readiness (Weeks 5-6)
- Comprehensive error handling and retry logic
- Circuit breakers for service resilience
- Advanced monitoring and alerting
- Load testing and performance optimization
- Security hardening and audit logging

---

## References

### Local Documentation
- [API Reference](./API_REFERENCE.md) - Detailed API documentation
- [BRIDGE_NODE_IMPLEMENTATION_PLAN.md](../BRIDGE_NODE_IMPLEMENTATION_PLAN.md) - Implementation plan
- [CLAUDE.md](../CLAUDE.md) - Project-specific development guide

### External References
- [ONEX Architecture Patterns](https://github.com/OmniNode-ai/Archon/blob/main/docs/ONEX_ARCHITECTURE_PATTERNS_COMPLETE.md)
- [omnibase_core Infrastructure](https://github.com/omnibase/omnibase_core)
- [O.N.E. v0.1 Protocol Specification](../docs/protocol/)

### Code Examples
- [Reducer Implementation](src/omninode_bridge/nodes/reducer/v1_0_0/node.py)
- [Model Examples](src/omninode_bridge/nodes/reducer/v1_0_0/models/)

---

**Document Version**: 1.0.0
**Maintained By**: omninode_bridge team
**Last Review**: 2025-10-02
