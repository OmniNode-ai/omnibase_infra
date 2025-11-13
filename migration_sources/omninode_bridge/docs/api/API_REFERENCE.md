# Bridge Nodes API Reference

**Version**: 1.0.0
**Last Updated**: 2025-10-02
**Status**: Complete

---

## Table of Contents

1. [Overview](#overview)
2. [NodeBridgeOrchestrator API](#nodebridgeorchestrator-api)
3. [NodeBridgeReducer API](#nodebridgereducer-api)
4. [Contract Models](#contract-models)
5. [Data Models](#data-models)
6. [Enumerations](#enumerations)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

---

## Overview

This document provides comprehensive API reference for the bridge nodes in the omninode_bridge project. Both nodes are ONEX v2.0 compliant and use contract-driven architecture with dependency injection.

### Import Paths

```python
# Orchestrator
from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator
from omninode_bridge.nodes.orchestrator.v1_0_0.models import (
    EnumWorkflowState,
    EnumWorkflowEvent,
    ModelStampRequestInput,
    ModelStampResponseOutput,
)

# Reducer
from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer
from omninode_bridge.nodes.reducer.v1_0_0.models import (
    EnumAggregationType,
    ModelBridgeState,
    ModelStampMetadataInput,
    ModelReducerInputState,
    ModelReducerOutputState,
)

# Contracts (from omnibase_core)
from omnibase_core.models.contracts import (
    ModelContractOrchestrator,
    ModelContractReducer,
)
from omnibase_core.models.container import ModelONEXContainer
```

---

## NodeBridgeOrchestrator API

### Class Definition

```python
class NodeBridgeOrchestrator(NodeOrchestrator):
    """
    Bridge Orchestrator for stamping workflow coordination.

    Coordinates multi-step stamping workflows with FSM state management,
    service routing, and event publishing.
    """
```

### Constructor

```python
def __init__(self, container: ModelONEXContainer) -> None:
    """
    Initialize Bridge Orchestrator with dependency injection container.

    Args:
        container: ONEX container with service dependencies

    Raises:
        OnexError: If container is invalid or initialization fails

    Configuration Keys (from container.config):
        - metadata_stamping_service_url: MetadataStamping service endpoint
        - onextree_service_url: OnexTree intelligence service endpoint
        - kafka_broker_url: Kafka broker connection string
        - default_namespace: Default namespace for operations
    """
```

**Example:**
```python
from omnibase_core.models.container import ModelONEXContainer

# Create container with configuration
container = ModelONEXContainer(
    config={
        "metadata_stamping_service_url": "http://metadata-stamping:8053",
        "onextree_service_url": "http://onextree:8080",
        "kafka_broker_url": "localhost:9092",
        "default_namespace": "omninode.bridge"
    }
)

# Initialize orchestrator
orchestrator = NodeBridgeOrchestrator(container)
```

### Core Methods

#### execute_orchestration

```python
async def execute_orchestration(
    self,
    contract: ModelContractOrchestrator
) -> ModelStampResponseOutput:
    """
    Execute stamping workflow orchestration.

    Workflow Steps:
    1. Validate input and transition FSM to 'processing'
    2. Route to OnexTree for intelligence (if enabled)
    3. Route to MetadataStampingService for hash generation
    4. Create metadata stamp with O.N.E. v0.1 compliance
    5. Publish events to Kafka
    6. Transition FSM to 'completed'
    7. Return stamped content with metadata

    Args:
        contract: Orchestrator contract with workflow configuration.
                 Must include:
                 - correlation_id: UUID for workflow tracking
                 - input_data: Input data for stamping

    Returns:
        ModelStampResponseOutput with:
        - stamp_id: Unique stamp identifier
        - file_hash: BLAKE3 hash of content
        - stamped_content: Content with embedded stamp
        - stamp_metadata: Complete stamp metadata
        - namespace: Namespace for the stamp
        - workflow_state: Final FSM state
        - processing_time_ms: Total processing time
        - hash_generation_time_ms: Hash generation time
        - workflow_steps_executed: Number of steps completed
        - intelligence_data: OnexTree analysis (if enabled)

    Raises:
        OnexError: If workflow execution fails
                  Error codes:
                  - VALIDATION_ERROR: Invalid input data
                  - OPERATION_FAILED: Workflow step failed

    Performance:
        - Target: < 50ms for standard workflows
        - With OnexTree: < 150ms
    """
```

**Example:**
```python
from uuid import uuid4
from omnibase_core.models.contracts import ModelContractOrchestrator

# Create contract
contract = ModelContractOrchestrator(
    correlation_id=uuid4(),
    input_data={
        "content": "Hello World",
        "namespace": "my_app",
        "file_path": "/data/hello.txt"
    }
)

# Execute orchestration
result = await orchestrator.execute_orchestration(contract)

print(f"Stamp ID: {result.stamp_id}")
print(f"File Hash: {result.file_hash}")
print(f"Processing Time: {result.processing_time_ms:.2f}ms")
print(f"Workflow State: {result.workflow_state.value}")
```

#### get_workflow_state

```python
def get_workflow_state(
    self,
    workflow_id: UUID
) -> Optional[EnumWorkflowState]:
    """
    Get current FSM state for a workflow.

    Args:
        workflow_id: Workflow correlation ID

    Returns:
        Current workflow state, or None if workflow not found

    Example:
        >>> state = orchestrator.get_workflow_state(workflow_id)
        >>> if state == EnumWorkflowState.COMPLETED:
        ...     print("Workflow completed successfully")
    """
```

#### get_stamping_metrics

```python
def get_stamping_metrics(self) -> dict[str, dict[str, float]]:
    """
    Get current stamping performance metrics.

    Returns:
        Dictionary of metrics by operation type:
        {
            "workflow_orchestration": {
                "total_operations": 100,
                "successful_operations": 95,
                "failed_operations": 5,
                "total_time_ms": 5000.0,
                "avg_time_ms": 50.0,
                "min_time_ms": 30.0,
                "max_time_ms": 150.0
            }
        }

    Example:
        >>> metrics = orchestrator.get_stamping_metrics()
        >>> workflow_metrics = metrics["workflow_orchestration"]
        >>> print(f"Success rate: {workflow_metrics['successful_operations'] / workflow_metrics['total_operations']:.1%}")
    """
```

### Internal Methods

#### _execute_workflow_steps

```python
async def _execute_workflow_steps(
    self,
    contract: ModelContractOrchestrator,
    workflow_id: UUID
) -> list[dict[str, Any]]:
    """
    Execute workflow steps defined in contract.

    Routes to appropriate services based on step configuration:
    - 'validation': Validate input data
    - 'onextree_intelligence': Route to OnexTree for AI analysis
    - 'hash_generation': Route to MetadataStampingService
    - 'stamp_creation': Create stamp with O.N.E. v0.1 compliance

    Returns:
        List of step execution results
    """
```

#### _transition_state

```python
async def _transition_state(
    self,
    workflow_id: UUID,
    current: EnumWorkflowState,
    target: EnumWorkflowState
) -> EnumWorkflowState:
    """
    Transition FSM state with validation.

    Validates transition is allowed and publishes state change event.

    Raises:
        OnexError: If transition is invalid (e.g., completed → processing)
    """
```

#### _publish_event

```python
async def _publish_event(
    self,
    event_type: EnumWorkflowEvent,
    data: dict[str, Any]
) -> None:
    """
    Publish event to Kafka using EventType subcontract.

    Event types:
    - WORKFLOW_STARTED
    - WORKFLOW_COMPLETED
    - WORKFLOW_FAILED
    - STEP_COMPLETED
    - STATE_TRANSITION
    - HASH_GENERATED
    - STAMP_CREATED
    - INTELLIGENCE_REQUESTED
    - INTELLIGENCE_RECEIVED
    """
```

---

## NodeBridgeReducer API

### Class Definition

```python
class NodeBridgeReducer(NodeReducer):
    """
    Bridge Reducer for metadata aggregation and state management.

    Aggregates stamping metadata with streaming architecture and
    FSM state tracking.
    """
```

### Constructor

```python
def __init__(self, container: ModelONEXContainer) -> None:
    """
    Initialize NodeBridgeReducer with dependency injection container.

    Args:
        container: ONEX DI container with service dependencies
    """
```

**Example:**
```python
from omnibase_core.models.container import ModelONEXContainer

# Create container
container = ModelONEXContainer(
    postgresql_client=postgresql_service,
    kafka_producer=kafka_service,
)

# Initialize reducer
reducer = NodeBridgeReducer(container)
```

### Core Methods

#### execute_reduction

```python
async def execute_reduction(
    self,
    contract: ModelContractReducer
) -> ModelReducerOutputState:
    """
    Execute metadata aggregation and state reduction.

    Aggregation Strategy:
    1. Stream stamp metadata from input (async iterator)
    2. Group by namespace using windowing
    3. Compute aggregations (count, sum, avg, distinct)
    4. Update FSM state for each workflow
    5. Persist aggregated state to PostgreSQL
    6. Return aggregation results

    Args:
        contract: Reducer contract with aggregation configuration.
                 Must include one of:
                 - input_stream: Async iterator of metadata items
                 - input_state: Dict or list of metadata items

    Returns:
        ModelReducerOutputState with:
        - aggregation_type: Strategy used (e.g., NAMESPACE_GROUPING)
        - total_items: Number of items aggregated
        - total_size_bytes: Sum of all file sizes
        - namespaces: List of unique namespaces processed
        - aggregations: Detailed aggregation data per namespace
        - fsm_states: Workflow FSM states by workflow_id
        - aggregation_duration_ms: Time taken for aggregation
        - items_per_second: Throughput metric

    Raises:
        OnexError: If reduction fails or validation errors occur

    Performance:
        - Target: > 1000 items/second
        - Latency: < 100ms for 1000 items
    """
```

**Example:**
```python
from omnibase_core.models.contracts import ModelContractReducer

# Create contract with streaming input
async def metadata_stream():
    for item in metadata_items:
        yield item

contract = ModelContractReducer(
    correlation_id=uuid4(),
    input_stream=metadata_stream()
)

# Execute reduction
result = await reducer.execute_reduction(contract)

print(f"Aggregated {result.total_items} items")
print(f"Namespaces: {', '.join(result.namespaces)}")
print(f"Throughput: {result.items_per_second:.0f} items/sec")
print(f"Duration: {result.aggregation_duration_ms:.2f}ms")

# Access per-namespace aggregations
for namespace, data in result.aggregations.items():
    print(f"  {namespace}:")
    print(f"    Total stamps: {data['total_stamps']}")
    print(f"    Total size: {data['total_size_bytes']:,} bytes")
    print(f"    File types: {', '.join(data['file_types'])}")
```

### Internal Methods

#### _stream_metadata

```python
async def _stream_metadata(
    self,
    contract: ModelContractReducer,
    batch_size: int = 100
) -> AsyncIterator[list[ModelReducerInputState]]:
    """
    Stream metadata from input using streaming configuration.

    Implements windowed streaming with batching for efficient processing.

    Yields:
        Batches of stamp metadata items
    """
```

#### _persist_state

```python
async def _persist_state(
    self,
    aggregated_data: dict[str, dict[str, Any]],
    fsm_states: dict[str, str],
    contract: ModelContractReducer
) -> None:
    """
    Persist aggregated state to PostgreSQL.

    Uses StateManagement subcontract for transaction management.

    Creates or updates ModelBridgeState records for each namespace.
    """
```

---

## Contract Models

### ModelContractOrchestrator

```python
class ModelContractOrchestrator(ModelContractBase):
    """
    Contract for orchestrator node configuration.

    Base Fields:
        node_type: EnumNodeType.ORCHESTRATOR
        correlation_id: UUID for workflow tracking
        input_data: Input data for processing

    Optional Subcontracts:
        workflow_coordination: Workflow step configuration
        routing: Service routing rules
        fsm: FSM state machine configuration
        event_type: Event definitions for publishing
    """
```

**Example:**
```python
from omnibase_core.models.contracts import ModelContractOrchestrator
from uuid import uuid4

contract = ModelContractOrchestrator(
    correlation_id=uuid4(),
    input_data={
        "content": "File content",
        "namespace": "app.domain",
        "file_path": "/data/file.txt"
    }
)
```

### ModelContractReducer

```python
class ModelContractReducer(ModelContractBase):
    """
    Contract for reducer node configuration.

    Base Fields:
        node_type: EnumNodeType.REDUCER
        correlation_id: UUID for tracking
        input_state: Input data for reduction
        input_stream: Async iterator of input items (optional)

    Optional Subcontracts:
        aggregation: Aggregation strategies
        state_management: PostgreSQL persistence config
        fsm: FSM state tracking config
        caching: Result caching config
    """
```

**Example:**
```python
from omnibase_core.models.contracts import ModelContractReducer

# With list input
contract = ModelContractReducer(
    correlation_id=uuid4(),
    input_state={
        "items": [metadata1, metadata2, ...]
    }
)

# With streaming input
async def stream_data():
    for item in large_dataset:
        yield item

contract = ModelContractReducer(
    correlation_id=uuid4(),
    input_stream=stream_data()
)
```

---

## Data Models

### ModelStampResponseOutput

```python
class ModelStampResponseOutput(BaseModel):
    """
    Output from orchestrator stamping workflow.

    Fields:
        stamp_id: str - Unique stamp identifier
        file_hash: str - BLAKE3 hash of content
        stamped_content: str - Content with embedded stamp
        stamp_metadata: dict[str, Any] - Complete stamp metadata
        namespace: str - Namespace for multi-tenant isolation
        op_id: UUID - Operation correlation ID
        version: int - Stamp version (default: 1)
        metadata_version: str - O.N.E. protocol version (default: "0.1")
        workflow_state: EnumWorkflowState - Final workflow FSM state
        workflow_id: UUID - Workflow correlation ID
        intelligence_data: dict[str, Any] | None - OnexTree analysis data
        processing_time_ms: float - Total processing time
        hash_generation_time_ms: float - Hash generation time
        workflow_steps_executed: int - Number of workflow steps completed
        created_at: datetime - Stamp creation timestamp
        completed_at: datetime - Workflow completion timestamp
    """
```

### ModelReducerOutputState

```python
class ModelReducerOutputState(BaseModel):
    """
    Output from reducer aggregation.

    Fields:
        aggregation_type: EnumAggregationType - Strategy used
        total_items: int - Number of items aggregated
        total_size_bytes: int - Sum of all file sizes
        namespaces: list[str] - Unique namespaces processed
        aggregations: dict[str, dict[str, Any]] - Per-namespace data
        fsm_states: dict[str, str] - Workflow states by workflow_id
        aggregation_duration_ms: float - Processing time
        items_per_second: float - Throughput metric
    """
```

**Aggregations Structure:**
```python
{
    "namespace1": {
        "total_stamps": 100,
        "total_size_bytes": 10485760,
        "file_types": ["application/pdf", "image/jpeg"],
        "workflow_ids": ["uuid1", "uuid2", ...]
    },
    "namespace2": {
        ...
    }
}
```

### ModelBridgeState

```python
class ModelBridgeState(BaseModel):
    """
    PostgreSQL-persisted bridge state for cumulative tracking.

    Fields:
        state_id: UUID - Unique state record identifier
        version: int - Version for optimistic locking
        namespace: str - Namespace for isolation
        metadata_version: str - O.N.E. protocol version
        total_stamps: int - Cumulative stamp count
        total_size_bytes: int - Cumulative file size
        unique_file_types: set[str] - All content types encountered
        unique_workflows: set[str] - All workflow IDs processed
        current_fsm_state: str - Current bridge FSM state
        fsm_state_history: list[dict] - State transition history
        created_at: datetime - State record creation
        last_updated: datetime - Last state update
        last_aggregation_at: datetime | None - Most recent aggregation
        total_aggregations: int - Count of aggregation operations
        avg_aggregation_duration_ms: float - Average aggregation time
        metadata: dict[str, Any] - Extended custom metadata
        configuration: dict[str, Any] - Bridge-specific config
    """
```

### ModelStampMetadataInput

```python
class ModelStampMetadataInput(BaseModel):
    """
    Input metadata for stamp aggregation.

    Fields:
        stamp_id: str - Unique stamp identifier
        file_hash: str - BLAKE3 hash
        file_path: str - Path to stamped file
        file_size: int - File size in bytes
        namespace: str - Multi-tenant namespace
        content_type: str - MIME type
        workflow_id: UUID - Associated workflow
        workflow_state: str - Current FSM state
        created_at: datetime - Creation timestamp
        processing_time_ms: float - Processing duration
    """
```

---

## Enumerations

### EnumWorkflowState

```python
class EnumWorkflowState(str, Enum):
    """
    FSM states for workflow orchestration.

    Values:
        PENDING: Workflow created, not yet started
        PROCESSING: Workflow actively executing
        COMPLETED: Workflow finished successfully (terminal)
        FAILED: Workflow failed with error (terminal)

    Methods:
        can_transition_to(target: EnumWorkflowState) -> bool:
            Check if transition to target state is valid

        is_terminal() -> bool:
            Check if this is a terminal state (no further transitions)
    """
```

**Valid Transitions:**
```python
# Allowed transitions
PENDING → PROCESSING
PROCESSING → COMPLETED
PROCESSING → FAILED

# Not allowed
COMPLETED → PROCESSING  # Terminal state
FAILED → PROCESSING     # Terminal state
PENDING → COMPLETED     # Must go through PROCESSING
```

**Example:**
```python
from omninode_bridge.nodes.orchestrator.v1_0_0.models import EnumWorkflowState

current_state = EnumWorkflowState.PENDING
target_state = EnumWorkflowState.PROCESSING

if current_state.can_transition_to(target_state):
    print(f"Transition {current_state.value} → {target_state.value} is valid")

if current_state.is_terminal():
    print("Cannot transition from terminal state")
```

### EnumWorkflowEvent

```python
class EnumWorkflowEvent(str, Enum):
    """
    Event types published during workflow execution.

    Values:
        WORKFLOW_STARTED: Workflow execution began
        WORKFLOW_COMPLETED: Workflow finished successfully
        WORKFLOW_FAILED: Workflow failed with error
        STEP_COMPLETED: Individual step completed
        STATE_TRANSITION: FSM state changed
        HASH_GENERATED: BLAKE3 hash created
        STAMP_CREATED: Metadata stamp created
        INTELLIGENCE_REQUESTED: OnexTree analysis requested
        INTELLIGENCE_RECEIVED: OnexTree analysis completed

    Methods:
        get_topic_name(namespace: str) -> str:
            Get Kafka topic name for this event type
    """
```

**Topic Naming:**
```python
event = EnumWorkflowEvent.STAMP_CREATED
topic = event.get_topic_name("omninode.bridge")
# Returns: "omninode.bridge.stamp_created"
```

### EnumAggregationType

```python
class EnumAggregationType(str, Enum):
    """
    Aggregation strategies for metadata reduction.

    Values:
        NAMESPACE_GROUPING: Group by namespace (primary strategy)
        TIME_WINDOW: Group by time windows (configurable duration)
        FILE_TYPE_GROUPING: Group by content_type/file_type
        SIZE_BUCKETS: Group by file size ranges
        WORKFLOW_GROUPING: Group by workflow_id
        CUSTOM: Custom aggregation via configuration
    """
```

---

## Error Handling

### OnexError

All nodes raise `OnexError` from `omnibase_core.errors` for failures:

```python
from omnibase_core.errors import OnexError, CoreErrorCode

try:
    result = await orchestrator.execute_orchestration(contract)
except OnexError as e:
    print(f"Error code: {e.code}")
    print(f"Message: {e.message}")
    print(f"Details: {e.details}")
    print(f"Cause: {e.cause}")
```

### Common Error Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| `VALIDATION_ERROR` | Input validation failed | Missing required fields, invalid data types |
| `OPERATION_FAILED` | Operation execution failed | Service unavailable, timeout, processing error |
| `STATE_ERROR` | Invalid FSM state transition | Attempting invalid transition (e.g., COMPLETED → PROCESSING) |
| `RESOURCE_ERROR` | Resource unavailable | Database connection failed, service unreachable |

### Error Handling Best Practices

```python
from omnibase_core.errors import OnexError, CoreErrorCode
from omnibase_core.enums import EnumLogLevel
from omnibase_core.logging import emit_log_event

async def safe_orchestration(contract):
    """Example of comprehensive error handling."""
    try:
        result = await orchestrator.execute_orchestration(contract)
        return result

    except OnexError as e:
        if e.code == CoreErrorCode.VALIDATION_ERROR:
            # Handle validation errors
            emit_log_event(
                EnumLogLevel.WARNING,
                "Invalid input data for workflow",
                {"error": str(e), "workflow_id": str(contract.correlation_id)}
            )
            # Return None or raise
            return None

        elif e.code == CoreErrorCode.OPERATION_FAILED:
            # Handle operational failures
            emit_log_event(
                EnumLogLevel.ERROR,
                "Workflow execution failed",
                {"error": str(e), "workflow_id": str(contract.correlation_id)}
            )
            # Retry logic here
            raise

        else:
            # Unknown error, re-raise
            raise

    except Exception as e:
        # Unexpected error
        emit_log_event(
            EnumLogLevel.CRITICAL,
            "Unexpected error in workflow orchestration",
            {
                "error": str(e),
                "error_type": type(e).__name__,
                "workflow_id": str(contract.correlation_id)
            }
        )
        raise
```

---

## Examples

### Example 1: Simple Stamping Workflow

```python
from uuid import uuid4
from omnibase_core.models.contracts import ModelContractOrchestrator
from omnibase_core.models.container import ModelONEXContainer
from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator

# Initialize orchestrator
container = ModelONEXContainer(config={
    "metadata_stamping_service_url": "http://metadata-stamping:8053",
    "kafka_broker_url": "localhost:9092",
    "default_namespace": "my_app"
})
orchestrator = NodeBridgeOrchestrator(container)

# Create workflow contract
contract = ModelContractOrchestrator(
    correlation_id=uuid4(),
    input_data={
        "content": "Important document content",
        "namespace": "my_app.documents",
        "file_path": "/data/important.pdf"
    }
)

# Execute orchestration
result = await orchestrator.execute_orchestration(contract)

# Use the result
print(f"✅ Stamp created: {result.stamp_id}")
print(f"📄 File hash: {result.file_hash}")
print(f"⏱️ Processing time: {result.processing_time_ms:.2f}ms")
print(f"📊 Workflow state: {result.workflow_state.value}")
```

### Example 2: Batch Aggregation

```python
from omnibase_core.models.contracts import ModelContractReducer
from omnibase_core.models.container import ModelONEXContainer
from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

# Initialize reducer
container = ModelONEXContainer(
    postgresql_client=postgresql_service,
)
reducer = NodeBridgeReducer(container)

# Create batch of metadata items
metadata_items = [
    {
        "stamp_id": "stamp-1",
        "file_hash": "hash1",
        "file_path": "/data/file1.pdf",
        "file_size": 1024000,
        "namespace": "app1",
        "content_type": "application/pdf",
        "workflow_id": uuid4(),
        "workflow_state": "completed"
    },
    # ... more items
]

# Create contract with batch
contract = ModelContractReducer(
    correlation_id=uuid4(),
    input_state={"items": metadata_items}
)

# Execute reduction
result = await reducer.execute_reduction(contract)

# Analyze results
print(f"📊 Aggregation Results:")
print(f"  Total items: {result.total_items}")
print(f"  Total size: {result.total_size_bytes:,} bytes")
print(f"  Namespaces: {len(result.namespaces)}")
print(f"  Throughput: {result.items_per_second:.0f} items/sec")

# Per-namespace breakdown
for namespace, data in result.aggregations.items():
    print(f"\n  {namespace}:")
    print(f"    Stamps: {data['total_stamps']}")
    print(f"    Size: {data['total_size_bytes']:,} bytes")
    print(f"    Types: {', '.join(data['file_types'])}")
```

### Example 3: Streaming Aggregation

```python
from omnibase_core.models.contracts import ModelContractReducer
from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

# Create async metadata stream
async def metadata_stream():
    """Stream metadata from database or queue."""
    async for batch in database.stream_metadata(batch_size=100):
        for item in batch:
            yield item

# Create streaming contract
contract = ModelContractReducer(
    correlation_id=uuid4(),
    input_stream=metadata_stream()
)

# Execute reduction with streaming
result = await reducer.execute_reduction(contract)

print(f"Streamed and aggregated {result.total_items} items in {result.aggregation_duration_ms:.2f}ms")
```

### Example 4: Intelligence-Enhanced Workflow

```python
from omnibase_core.models.contracts import ModelContractOrchestrator
from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator

# Initialize with OnexTree enabled
container = ModelONEXContainer(config={
    "metadata_stamping_service_url": "http://metadata-stamping:8053",
    "onextree_service_url": "http://onextree:8080",
    "kafka_broker_url": "localhost:9092"
})
orchestrator = NodeBridgeOrchestrator(container)

# Create workflow with intelligence request
contract = ModelContractOrchestrator(
    correlation_id=uuid4(),
    input_data={
        "content": "Sensitive document",
        "namespace": "secure.documents",
        "enable_intelligence": True  # Request OnexTree analysis
    }
)

# Execute with intelligence
result = await orchestrator.execute_orchestration(contract)

# Check intelligence results
if result.intelligence_data:
    print("🧠 OnexTree Intelligence:")
    print(f"  Analysis type: {result.intelligence_data['analysis_type']}")
    print(f"  Confidence: {result.intelligence_data['confidence_score']}")
    print(f"  Recommendations: {', '.join(result.intelligence_data['recommendations'])}")
```

### Example 5: Workflow State Tracking

```python
from omninode_bridge.nodes.orchestrator.v1_0_0.models import EnumWorkflowState

# Start workflow
workflow_id = uuid4()
contract = ModelContractOrchestrator(
    correlation_id=workflow_id,
    input_data={"content": "Test"}
)

# Execute asynchronously (don't wait for completion)
asyncio.create_task(orchestrator.execute_orchestration(contract))

# Poll for workflow state
while True:
    state = orchestrator.get_workflow_state(workflow_id)

    if state == EnumWorkflowState.PENDING:
        print("⏳ Workflow pending...")
    elif state == EnumWorkflowState.PROCESSING:
        print("🔄 Workflow processing...")
    elif state == EnumWorkflowState.COMPLETED:
        print("✅ Workflow completed!")
        break
    elif state == EnumWorkflowState.FAILED:
        print("❌ Workflow failed!")
        break

    await asyncio.sleep(0.1)
```

### Example 6: Performance Monitoring

```python
# Get orchestrator metrics
metrics = orchestrator.get_stamping_metrics()

for operation, data in metrics.items():
    print(f"\n📊 {operation}:")
    print(f"  Total operations: {data['total_operations']}")
    print(f"  Success rate: {data['successful_operations'] / data['total_operations']:.1%}")
    print(f"  Average time: {data['avg_time_ms']:.2f}ms")
    print(f"  Min time: {data['min_time_ms']:.2f}ms")
    print(f"  Max time: {data['max_time_ms']:.2f}ms")
```

---

## Performance Characteristics

### NodeBridgeOrchestrator

| Operation | Target | Typical |
|-----------|--------|---------|
| Standard workflow | < 50ms | ~30-40ms |
| With OnexTree intelligence | < 150ms | ~100-120ms |
| Concurrent workflows | 100+/sec | 150/sec |
| FSM state transition | < 1ms | ~0.5ms |
| Event publishing | < 5ms | ~2-3ms |

### NodeBridgeReducer

| Operation | Target | Typical |
|-----------|--------|---------|
| Throughput | > 1000 items/sec | ~1500 items/sec |
| Aggregation latency (1000 items) | < 100ms | ~60-80ms |
| Streaming window (5000ms) | < 10ms overhead | ~5ms |
| PostgreSQL persistence | < 50ms | ~20-30ms |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-10-02 | Initial API reference with complete implementations |

---

## Phase 3: Intelligent Code Generation API

### Overview

Phase 3 adds intelligent, pattern-driven code generation with LLM enhancement to the omninode_bridge system. These APIs enable template variant selection, mixin recommendation, pattern matching, and enhanced context building for production-quality code generation.

**Key Features**:
- Template variant selection based on requirements analysis
- Intelligent mixin recommendation with conflict resolution
- Production pattern library with 5+ pattern generators
- Enhanced LLM context building (target: <8K tokens)
- Subcontract processing for ONEX v2.0 compliance

### Import Paths

```python
# Template Selection
from omninode_bridge.codegen.template_selector import TemplateSelector, ModelTemplateSelection

# Pattern Library
from omninode_bridge.codegen.pattern_library import ProductionPatternLibrary, ModelPatternMatch

# Mixin Intelligence
from omninode_bridge.codegen.mixins.mixin_recommender import MixinRecommender
from omninode_bridge.codegen.mixins.models import ModelMixinRecommendation

# Context Building
from omninode_bridge.codegen.context_builder import EnhancedContextBuilder, ModelLLMContext

# Subcontract Processing (in template_engine)
from omninode_bridge.codegen.template_engine import TemplateEngine
```

---

## TemplateSelector API

### Class Definition

```python
class TemplateSelector:
    """
    Intelligent template variant selector.

    Analyzes requirements and selects optimal template variant with confidence
    scoring and pattern recommendations.
    """
```

### Constructor

```python
def __init__(self, template_root: Optional[Path] = None) -> None:
    """
    Initialize template selector.

    Args:
        template_root: Root directory for templates (uses default if None)
    """
```

### Core Method: select_template

```python
def select_template(
    self,
    requirements: Any,
    node_type: str,
    target_environment: Optional[str] = None,
) -> ModelTemplateSelection:
    """
    Select optimal template variant based on requirements.

    Args:
        requirements: Contract or requirements object
        node_type: Node type (effect/compute/reducer/orchestrator)
        target_environment: Target environment (development/staging/production)

    Returns:
        ModelTemplateSelection with:
        - variant: Selected template variant enum
        - confidence: Confidence score (0.0-1.0)
        - template_path: Path to selected template
        - patterns: Recommended patterns for this variant
        - rationale: Human-readable explanation
        - selection_time_ms: Time taken for selection

    Performance:
        - Target: <5ms per selection
        - Accuracy: >95% correct template selection
    """
```

**Example:**
```python
from omninode_bridge.codegen.template_selector import TemplateSelector

selector = TemplateSelector()

# Select template variant
result = selector.select_template(
    requirements=contract,
    node_type="effect",
    target_environment="production"
)

print(f"Selected: {result.variant.value}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Rationale: {result.rationale}")
print(f"Patterns: {', '.join(result.patterns)}")
```

---

## ProductionPatternLibrary API

### Class Definition

```python
class ProductionPatternLibrary:
    """
    Unified interface to discover, match, and apply production patterns.

    Provides pattern discovery, similarity-based matching, and usage tracking.
    """
```

### Constructor

```python
def __init__(self) -> None:
    """Initialize pattern library with all available pattern generators."""
```

### Core Methods

#### find_patterns

```python
def find_patterns(
    self,
    requirements: Any,
    node_type: str,
    max_results: int = 10
) -> list[ModelPatternMatch]:
    """
    Find matching patterns based on requirements.

    Args:
        requirements: Contract or requirements object
        node_type: Node type filter
        max_results: Maximum number of patterns to return

    Returns:
        List of ModelPatternMatch with:
        - pattern_info: Pattern metadata
        - relevance_score: Match relevance (0.0-1.0)
        - match_reason: Explanation of why pattern matched
        - confidence: Confidence in match (0.0-1.0)

    Performance:
        - Target: <10ms per search
        - Accuracy: >90% pattern match relevance
    """
```

#### get_pattern_code

```python
def get_pattern_code(self, pattern_name: str, **kwargs) -> str:
    """
    Generate code for a specific pattern.

    Args:
        pattern_name: Name of pattern to generate
        **kwargs: Pattern-specific configuration

    Returns:
        Generated code string
    """
```

**Example:**
```python
from omninode_bridge.codegen.pattern_library import ProductionPatternLibrary

library = ProductionPatternLibrary()

# Find matching patterns
matches = library.find_patterns(
    requirements=contract,
    node_type="effect",
    max_results=5
)

for match in matches:
    print(f"Pattern: {match.pattern_info.name}")
    print(f"  Category: {match.pattern_info.category.value}")
    print(f"  Relevance: {match.relevance_score:.2f}")
    print(f"  Reason: {match.match_reason}")

    # Generate code for pattern
    code = library.get_pattern_code(
        match.pattern_info.name,
        service_name="my_service"
    )
```

---

## MixinRecommender API

### Class Definition

```python
class MixinRecommender:
    """
    Intelligent mixin recommendation engine.

    Analyzes requirements and recommends appropriate mixins with conflict
    detection and resolution.
    """
```

### Constructor

```python
def __init__(
    self,
    scorer: Optional[MixinScorer] = None,
    conflict_resolver: Optional[ConflictResolver] = None
) -> None:
    """
    Initialize mixin recommender.

    Args:
        scorer: Mixin scoring engine (uses default if None)
        conflict_resolver: Conflict resolution engine (uses default if None)
    """
```

### Core Method: recommend_mixins

```python
def recommend_mixins(
    self,
    requirements: Any,
    node_type: str,
    max_recommendations: int = 10,
    min_score_threshold: float = 0.5
) -> list[ModelMixinRecommendation]:
    """
    Recommend mixins based on requirement analysis.

    Args:
        requirements: Contract or requirements object
        node_type: Node type for context
        max_recommendations: Maximum mixins to recommend
        min_score_threshold: Minimum relevance score (0.0-1.0)

    Returns:
        List of ModelMixinRecommendation with:
        - mixin_name: Name of recommended mixin
        - relevance_score: How relevant to requirements (0.0-1.0)
        - rationale: Why this mixin was recommended
        - code_snippet: Code to inject
        - conflicts: List of conflicting mixins
        - dependencies: Required dependencies

    Performance:
        - Target: <20ms per recommendation set
        - Accuracy: >90% useful recommendations
    """
```

**Example:**
```python
from omninode_bridge.codegen.mixins.mixin_recommender import MixinRecommender

recommender = MixinRecommender()

# Get mixin recommendations
recommendations = recommender.recommend_mixins(
    requirements=contract,
    node_type="effect",
    max_recommendations=5,
    min_score_threshold=0.6
)

for rec in recommendations:
    print(f"Mixin: {rec.mixin_name}")
    print(f"  Score: {rec.relevance_score:.2f}")
    print(f"  Rationale: {rec.rationale}")
    print(f"  Dependencies: {', '.join(rec.dependencies)}")

    if rec.conflicts:
        print(f"  Conflicts with: {', '.join(rec.conflicts)}")
```

---

## EnhancedContextBuilder API

### Class Definition

```python
class EnhancedContextBuilder:
    """
    Builds comprehensive LLM context from Phase 3 components.

    Aggregates data from contract requirements, template selection,
    pattern matching, and mixin recommendations.
    """
```

### Constructor

```python
def __init__(self, max_tokens: int = 8000) -> None:
    """
    Initialize context builder.

    Args:
        max_tokens: Maximum token count for generated context
    """
```

### Core Method: build_context

```python
def build_context(
    self,
    operation_name: str,
    operation_description: str,
    node_type: str,
    template_selection: ModelTemplateSelection,
    pattern_matches: list[ModelPatternMatch],
    mixin_recommendations: list[ModelMixinRecommendation],
    requirements: dict[str, Any]
) -> ModelLLMContext:
    """
    Build comprehensive LLM context from all Phase 3 inputs.

    Args:
        operation_name: Name of operation to implement
        operation_description: Detailed operation description
        node_type: Node type (effect/compute/reducer/orchestrator)
        template_selection: Selected template variant
        pattern_matches: Matched production patterns
        mixin_recommendations: Recommended mixins
        requirements: Additional requirements and constraints

    Returns:
        ModelLLMContext with:
        - operation_name, operation_description, node_type
        - template_variant: Selected variant name
        - patterns: Pattern data with code examples
        - mixins: Mixin data with usage instructions
        - requirements: Structured requirements
        - examples: Similar code examples
        - best_practices: ONEX best practices
        - estimated_tokens: Token count estimate

    Performance:
        - Target: <50ms per context build
        - Size Target: <8K tokens per context
    """
```

**Example:**
```python
from omninode_bridge.codegen.context_builder import EnhancedContextBuilder

builder = EnhancedContextBuilder(max_tokens=8000)

# Build comprehensive context
context = builder.build_context(
    operation_name="fetch_user_data",
    operation_description="Fetch user data from PostgreSQL with caching",
    node_type="effect",
    template_selection=template_result,
    pattern_matches=pattern_results,
    mixin_recommendations=mixin_results,
    requirements={"database": "postgresql", "cache": True}
)

print(f"Context built: {context.estimated_tokens} tokens")
print(f"Template: {context.template_variant}")
print(f"Patterns: {len(context.patterns)}")
print(f"Mixins: {len(context.mixins)}")

# Convert to dict for LLM API
context_dict = context.to_dict()
```

---

## TemplateEngine Subcontract Processing

The `TemplateEngine` class includes subcontract processing capabilities for Phase 3:

### Method: process_subcontracts

```python
def process_subcontracts(
    self,
    contract: dict[str, Any],
    node_type: str
) -> dict[str, str]:
    """
    Generate subcontract YAML files based on contract configuration.

    Supports 6 subcontract types:
    - API: External API integration subcontracts
    - Compute: Computational operation subcontracts
    - Database: Database operation subcontracts
    - Event: Event publishing/consuming subcontracts
    - State: State management subcontracts
    - Workflow: Workflow coordination subcontracts

    Args:
        contract: Contract dictionary with subcontract specifications
        node_type: Node type for context

    Returns:
        Dictionary mapping subcontract filenames to YAML content

    Example:
        {
            "api_subcontract.yaml": "...",
            "database_subcontract.yaml": "...",
            "event_subcontract.yaml": "..."
        }
    """
```

**Example:**
```python
from omninode_bridge.codegen.template_engine import TemplateEngine

engine = TemplateEngine()

# Generate subcontracts
subcontracts = engine.process_subcontracts(
    contract={
        "subcontracts": {
            "database": {
                "type": "postgresql",
                "operations": ["read", "write"]
            },
            "events": {
                "topics": ["user.created", "user.updated"]
            }
        }
    },
    node_type="effect"
)

for filename, content in subcontracts.items():
    print(f"Generated: {filename}")
```

---

## Phase 3 Data Models

### ModelTemplateSelection

```python
@dataclass
class ModelTemplateSelection:
    """Results of template variant selection."""

    variant: EnumTemplateVariant
    confidence: float  # 0.0-1.0
    template_path: Optional[Path] = None
    patterns: list[str] = field(default_factory=list)
    rationale: str = ""
    selection_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

### ModelPatternMatch

```python
@dataclass
class ModelPatternMatch:
    """Represents a matched pattern with scoring."""

    pattern_info: ModelPatternInfo
    relevance_score: float  # 0.0-1.0
    match_reason: str = ""
    confidence: float = 0.9
```

### ModelMixinRecommendation

```python
class ModelMixinRecommendation(BaseModel):
    """Recommendation for a specific mixin with explanation."""

    mixin_name: str
    relevance_score: float  # 0.0-1.0
    rationale: str
    code_snippet: str
    conflicts: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    usage_instructions: str = ""
    priority: int = 0
```

### ModelLLMContext

```python
@dataclass
class ModelLLMContext:
    """Comprehensive LLM generation context."""

    operation_name: str
    operation_description: str
    node_type: str
    template_variant: str
    patterns: list[dict[str, Any]] = field(default_factory=list)
    mixins: list[dict[str, Any]] = field(default_factory=list)
    requirements: dict[str, Any] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    best_practices: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 0
```

---

## Phase 3 Performance Targets

| Component | Target | Metric |
|-----------|--------|--------|
| TemplateSelector | <5ms | Selection time |
| TemplateSelector | >95% | Selection accuracy |
| PatternLibrary | <10ms | Pattern search time |
| PatternLibrary | >90% | Match relevance |
| MixinRecommender | <20ms | Recommendation time |
| MixinRecommender | >90% | Useful recommendations |
| EnhancedContextBuilder | <50ms | Context build time |
| EnhancedContextBuilder | <8K tokens | Context size |

---

## Phase 3 End-to-End Example

```python
from omninode_bridge.codegen.template_selector import TemplateSelector
from omninode_bridge.codegen.pattern_library import ProductionPatternLibrary
from omninode_bridge.codegen.mixins.mixin_recommender import MixinRecommender
from omninode_bridge.codegen.context_builder import EnhancedContextBuilder

# Initialize Phase 3 components
template_selector = TemplateSelector()
pattern_library = ProductionPatternLibrary()
mixin_recommender = MixinRecommender()
context_builder = EnhancedContextBuilder(max_tokens=8000)

# Step 1: Select template variant
template_result = template_selector.select_template(
    requirements=contract,
    node_type="effect",
    target_environment="production"
)
print(f"Template: {template_result.variant.value} (confidence: {template_result.confidence:.2%})")

# Step 2: Find matching patterns
pattern_results = pattern_library.find_patterns(
    requirements=contract,
    node_type="effect",
    max_results=5
)
print(f"Found {len(pattern_results)} matching patterns")

# Step 3: Get mixin recommendations
mixin_results = mixin_recommender.recommend_mixins(
    requirements=contract,
    node_type="effect",
    max_recommendations=5,
    min_score_threshold=0.6
)
print(f"Recommended {len(mixin_results)} mixins")

# Step 4: Build comprehensive LLM context
llm_context = context_builder.build_context(
    operation_name="fetch_user_data",
    operation_description="Fetch user data from PostgreSQL with caching",
    node_type="effect",
    template_selection=template_result,
    pattern_matches=pattern_results,
    mixin_recommendations=mixin_results,
    requirements={"database": "postgresql", "cache": True}
)
print(f"Context: {llm_context.estimated_tokens} tokens")

# Use context for code generation
# (Pass to LLM API or template engine)
```

---

## Related Documentation

- [Code Generation Guide](../guides/CODE_GENERATION_GUIDE.md) - Complete code generation workflow
- [Bridge Nodes Guide](../guides/BRIDGE_NODES_GUIDE.md) - Comprehensive implementation guide
- [BRIDGE_NODE_IMPLEMENTATION_PLAN.md](../planning/BRIDGE_NODE_IMPLEMENTATION_PLAN.md) - Implementation plan
- [Phase 3 Task Breakdown](../planning/PHASE_3_TASK_BREAKDOWN.md) - Phase 3 planning and tasks
- [CLAUDE.md](../../CLAUDE.md) - Project-specific development guide

---

**Document Version**: 2.0.0 (Phase 3)
**Maintained By**: omninode_bridge team
**Last Review**: 2025-11-06
