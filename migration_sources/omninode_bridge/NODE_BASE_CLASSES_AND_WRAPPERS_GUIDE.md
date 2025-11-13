# Node Base Classes and Convenience Wrappers - Comprehensive Guide

**Date**: 2025-11-05
**Purpose**: Complete understanding of omnibase_core node architecture for code generation
**Scope**: Base classes, convenience wrappers, MRO, and usage patterns

---

## Executive Summary

omnibase_core provides two ways to create ONEX v2.0 nodes:

1. **Convenience Wrappers** (Recommended): Pre-composed classes with standard mixins → `ModelServiceEffect`, `ModelServiceCompute`, etc.
2. **Custom Composition**: Manual mixin assembly for specialized requirements → Inherit from base classes directly

**Key Insight**: Convenience wrappers include `MixinNodeService` as the **first** mixin, enabling persistent service mode (long-lived MCP servers, tool invocation handling). This is critical for service-oriented architectures.

---

## Table of Contents

1. [Base Node Classes](#base-node-classes)
2. [Convenience Wrappers](#convenience-wrappers)
3. [MixinNodeService](#mixinnodeservice)
4. [Method Resolution Order (MRO)](#method-resolution-order-mro)
5. [Decision Matrix](#decision-matrix)
6. [Import Paths](#import-paths)
7. [Code Examples](#code-examples)

---

## Base Node Classes

### Overview

All ONEX nodes inherit from one of four base classes. These provide the **fundamental execution semantics** for each node type.

| Base Class | Purpose | Key Method | Stability | Version |
|------------|---------|------------|-----------|---------|
| `NodeEffect` | Side effects, I/O operations | `execute_effect()` | Stable | v1.0.0 |
| `NodeCompute` | Pure computations, transformations | `process()` | Stable | v1.0.0 |
| `NodeReducer` | Aggregation, state reduction | `process()` | Stable | v1.0.0 |
| `NodeOrchestrator` | Workflow coordination | `process()` | Stable | v1.0.0 |

**Stability Guarantee**: Abstract method signatures are **frozen**. Breaking changes require major version bump (v2.0.0).

---

### 1. NodeEffect

**File**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/nodes/node_effect.py`

#### Purpose
Manages side effects and external interactions with transaction support, retry policies, and circuit breaker patterns.

#### Key Capabilities
- **Transaction Management**: `ModelEffectTransaction` with automatic rollback
- **Retry Logic**: Exponential backoff with configurable max attempts
- **Circuit Breaker**: Failure threshold detection for fault tolerance
- **Atomic File Operations**: Data integrity guarantees for file I/O
- **Event Bus Integration**: State change notifications
- **Performance Monitoring**: Effect-specific metrics tracking

#### Core Attributes
```python
default_timeout_ms: int = 30000
default_retry_delay_ms: int = 1000
max_concurrent_effects: int = 10
active_transactions: dict[UUID, ModelEffectTransaction]
circuit_breakers: dict[str, ModelCircuitBreaker]
effect_handlers: dict[EnumEffectType, Callable]
effect_semaphore: asyncio.Semaphore
effect_metrics: dict[str, dict[str, float]]
```

#### Required Method
```python
async def execute_effect(
    self,
    contract: ModelContractEffect
) -> ModelEffectOutput:
    """
    REQUIRED INTERFACE: Execute side effect based on contract.

    Args:
        contract: Effect contract specifying operation configuration

    Returns:
        ModelEffectOutput with transaction status and metadata

    Raises:
        ModelOnexError: If effect execution fails or rollback fails
    """
```

#### Built-in Effect Handlers
- **FILE_OPERATION**: Read, write, move, delete with atomic guarantees
- **EVENT_EMISSION**: Emit events to event bus

#### Thread Safety Warning
⚠️ Circuit breaker state NOT thread-safe. Transactions NOT shareable across threads. Create separate instances per thread for concurrent effects.

---

### 2. NodeCompute

**File**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/nodes/node_compute.py`

#### Purpose
Pure computational operations with deterministic guarantees, caching, and parallel processing support.

#### Key Capabilities
- **Pure Function Patterns**: No side effects, deterministic outputs
- **Computational Pipeline**: Input → Transform → Output
- **Caching Layer**: `ModelComputeCache` with TTL and eviction policies
- **Parallel Processing**: ThreadPoolExecutor for batch operations
- **Algorithm Registry**: Register custom computation functions
- **Performance Optimization**: Cache hit tracking, latency monitoring

#### Core Attributes
```python
max_parallel_workers: int = 4
cache_ttl_minutes: int = 30
performance_threshold_ms: float = 100.0
computation_cache: ModelComputeCache
thread_pool: ThreadPoolExecutor | None
computation_registry: dict[str, Callable]
computation_metrics: dict[str, dict[str, float]]
```

#### Required Method
```python
async def process(
    self,
    input_data: ModelComputeInput[T_Input]
) -> ModelComputeOutput[T_Output]:
    """
    REQUIRED: Execute pure computation.

    STABLE INTERFACE: Frozen for code generation.

    Args:
        input_data: Strongly typed computation input

    Returns:
        ModelComputeOutput with performance metrics

    Raises:
        ModelOnexError: If computation fails or threshold exceeded
    """
```

#### Built-in Computations
- **default**: Identity transformation
- **string_uppercase**: Convert string to uppercase
- **sum_numbers**: Sum list of numbers

#### Thread Safety Warning
⚠️ Instance NOT thread-safe due to mutable cache state. Use separate instances per thread OR implement cache locking.

---

### 3. NodeReducer

**File**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/nodes/node_reducer.py`

#### Purpose
Data aggregation and state reduction operations with streaming support and conflict resolution.

#### Key Capabilities
- **Multiple Reduction Types**: FOLD, ACCUMULATE, MERGE, AGGREGATE, NORMALIZE
- **Streaming Support**: BATCH, INCREMENTAL, WINDOWED modes
- **Conflict Resolution**: Configurable strategies (LATEST_WINS, MERGE_DEEP, CUSTOM)
- **Performance Optimization**: Batch processing for large datasets
- **RSD Data Processing**: Ticket metadata aggregation, priority normalization, dependency cycle detection
- **Memory-Efficient**: Streaming windows for large datasets

#### Core Attributes
```python
default_batch_size: int = 1000
max_memory_usage_mb: int = 512
streaming_buffer_size: int = 10000
reduction_functions: dict[EnumReductionType, Callable]
reduction_metrics: dict[str, dict[str, float]]
active_windows: dict[str, ModelStreamingWindow]
```

#### Required Method
```python
async def process(
    self,
    input_data: ModelReducerInput[T_Input]
) -> ModelReducerOutput[T_Output]:
    """
    REQUIRED: Stream-based reduction with conflict resolution.

    STABLE INTERFACE: Frozen for code generation.

    Args:
        input_data: Strongly typed reduction input with configuration

    Returns:
        ModelReducerOutput with processing statistics

    Raises:
        ModelOnexError: If reduction fails or memory limits exceeded
    """
```

#### Pure FSM Pattern
⚠️ **IMPORTANT**: NodeReducer uses **Pure FSM Pattern**:
- No mutable instance state for business logic
- All state passed through input/output
- Side effects emitted as `ModelIntent` objects
- Intents collected in `ModelReducerOutput.intents` for Effect node execution

**Example Intent Emission**:
```python
intents: list[ModelIntent] = []

# Intent to log metrics
intents.append(
    ModelIntent(
        intent_type="log_metric",
        target="metrics_service",
        payload={
            "metric_type": "reduction_metrics",
            "processing_time_ms": processing_time,
            "success": True,
        },
        priority=3,
    )
)

return ModelReducerOutput(
    result=result,
    intents=intents,  # Effect node processes these
    ...
)
```

---

### 4. NodeOrchestrator

**File**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/nodes/node_orchestrator.py`

#### Purpose
Workflow coordination and control flow management with action emission patterns and parallel execution coordination.

#### Key Capabilities
- **Workflow Coordination**: Control flow with dependency tracking
- **Action Emission**: Thunk patterns for deferred execution
- **Conditional Branching**: Runtime state-based decision making
- **Parallel Execution**: Coordinated parallel step execution with load balancing
- **Dependency-Aware Ordering**: Topological sort for dependency graphs
- **Error Recovery**: Partial failure handling with configurable strategies
- **RSD Workflow Management**: Ticket lifecycle state transitions

#### Core Attributes
```python
max_concurrent_workflows: int = 5
default_step_timeout_ms: int = 30000
action_emission_enabled: bool = True
active_workflows: dict[UUID, ModelOrchestratorInput]
workflow_states: dict[UUID, EnumWorkflowState]
load_balancer: ModelLoadBalancer
emitted_actions: dict[UUID, list[ModelAction]]
workflow_semaphore: asyncio.Semaphore
orchestration_metrics: dict[str, dict[str, float]]
condition_functions: dict[str, Callable]
```

#### Required Method
```python
async def process(
    self,
    input_data: ModelOrchestratorInput
) -> ModelOrchestratorOutput:
    """
    REQUIRED: Execute workflow coordination with action emission.

    STABLE INTERFACE: Frozen for code generation.

    Args:
        input_data: Strongly typed orchestration input with workflow config

    Returns:
        ModelOrchestratorOutput with execution results

    Raises:
        ModelOnexError: If workflow coordination fails
    """
```

#### Execution Modes
- **SEQUENTIAL**: Execute steps one after another
- **PARALLEL**: Execute independent steps concurrently
- **BATCH**: Load-balanced batch processing

#### Built-in Conditions
- **always_true**: Always execute step
- **always_false**: Never execute step
- **has_previous_results**: Execute if previous results exist
- **previous_step_success**: Execute if previous step succeeded

---

## Convenience Wrappers

### Overview

Convenience wrappers are **pre-composed node classes** that eliminate boilerplate by combining base classes with commonly used mixins. They provide production-ready capabilities out of the box.

**Key Benefit**: One import, one inheritance, zero configuration.

---

### 1. ModelServiceEffect

**File**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/models/nodes/node_services/model_service_effect.py`

#### Composition
```python
class ModelServiceEffect(
    MixinNodeService,        # Persistent service mode (FIRST!)
    NodeEffect,              # Effect semantics
    MixinHealthCheck,        # Health monitoring
    MixinEventBus,           # Event publishing
    MixinMetrics,            # Performance metrics
):
    ...
```

#### Included Capabilities
✅ **Persistent service mode** - Long-lived MCP servers, tool invocation handling
✅ **Transaction management** - Automatic rollback on failures
✅ **Retry logic** - Configurable retry with exponential backoff
✅ **Circuit breaker** - Fault tolerance for cascading failures
✅ **Health check endpoint** - `GET /health` → `{"status": "healthy", ...}`
✅ **Event emission** - `await self.publish_event(...)` → publishes to event bus
✅ **Performance metrics** - Request latency, throughput, error rates

#### When to Use
- Database operations requiring transactions
- External API calls needing circuit breaker protection
- File I/O operations with health monitoring
- Message queue producers with event coordination

#### Usage Example
```python
from omnibase_core.models.nodes.node_services import ModelServiceEffect
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect

class NodeDatabaseWriterEffect(ModelServiceEffect):
    """Database writer with automatic health checks, events, and metrics."""

    async def execute_effect(self, contract: ModelContractEffect) -> dict:
        # Just write your business logic!
        result = await self.database.write(contract.input_data)

        # Emit event automatically tracked with metrics
        await self.publish_event(
            event_type="write_completed",
            payload={"records_written": result["count"]},
            correlation_id=contract.correlation_id
        )

        return {"status": "success", "data": result}
```

---

### 2. ModelServiceCompute

**File**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/models/nodes/node_services/model_service_compute.py`

#### Composition
```python
class ModelServiceCompute(
    MixinNodeService,        # Persistent service mode (FIRST!)
    NodeCompute,             # Pure computation semantics
    MixinHealthCheck,        # Health monitoring
    MixinCaching,            # Result caching
    MixinMetrics,            # Performance metrics
):
    ...
```

#### Included Capabilities
✅ **Persistent service mode** - Long-lived tool services
✅ **Result caching** - LRU cache with configurable TTL
✅ **Cache key generation** - Deterministic key generation from inputs
✅ **Cache hit/miss tracking** - Metrics for cache performance
✅ **Health check** - Cache service health included in node health
✅ **Performance metrics** - Computation latency, cache hit ratio

#### When to Use
- Data transformation pipelines benefiting from caching
- Expensive calculations with repeatable inputs (ML inference)
- Pure functions requiring performance monitoring
- Stateless processors with deterministic outputs

#### Usage Example
```python
from omnibase_core.models.nodes.node_services import ModelServiceCompute
from omnibase_core.models.contracts.model_contract_compute import ModelContractCompute

class NodeDataTransformerCompute(ModelServiceCompute):
    """Data transformer with automatic caching and metrics."""

    async def execute_compute(self, contract: ModelContractCompute) -> dict:
        # Check cache first (automatic via MixinCaching)
        cache_key = self.generate_cache_key(contract.input_data)
        cached_result = await self.get_cached(cache_key)

        if cached_result:
            return cached_result  # Cache hit!

        # Perform expensive computation
        result = await self._transform_data(contract.input_data)

        # Cache result for 10 minutes
        await self.set_cached(cache_key, result, ttl_seconds=600)

        return result
```

#### Why Caching Matters
Compute nodes often perform expensive operations (ML inference, complex transformations, aggregations). Caching eliminates redundant computation for identical inputs, reducing latency from seconds to milliseconds.

---

### 3. ModelServiceReducer

**File**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/models/nodes/node_services/model_service_reducer.py`

#### Composition
```python
class ModelServiceReducer(
    MixinNodeService,        # Persistent service mode (FIRST!)
    NodeReducer,             # Aggregation semantics
    MixinHealthCheck,        # Health monitoring
    MixinCaching,            # Result caching
    MixinMetrics,            # Performance metrics
):
    ...
```

#### Included Capabilities
✅ **Persistent service mode** - Long-lived tool services
✅ **Aggregation result caching** - Avoids re-computing expensive aggregations
✅ **State persistence health** - Monitors state storage availability
✅ **Performance metrics** - Aggregation latency, data volume processed
✅ **Cache invalidation** - Automatic cache clearing on state changes

#### When to Use
- Metrics aggregators caching 5-minute rollups
- Log analyzers caching daily summaries
- Analytics reducers caching computed KPIs
- Stream processors reducing event streams to summaries

#### Usage Example
```python
from omnibase_core.models.nodes.node_services import ModelServiceReducer
from omnibase_core.models.contracts.model_contract_reducer import ModelContractReducer

class NodeMetricsAggregatorReducer(ModelServiceReducer):
    """Metrics aggregator with automatic caching and health checks."""

    async def execute_reduction(self, contract: ModelContractReducer) -> dict:
        # Check cache for recent aggregation
        cache_key = self.generate_cache_key(contract.aggregation_window)
        cached_result = await self.get_cached(cache_key)

        if cached_result:
            return cached_result  # Avoid re-aggregating

        # Perform expensive aggregation over large dataset
        aggregated_data = await self._aggregate_metrics(contract.input_data)

        # Cache aggregated result for 5 minutes
        await self.set_cached(cache_key, aggregated_data, ttl_seconds=300)

        return aggregated_data
```

#### Why Caching Matters
Reducers often aggregate large datasets (sum, average, group-by operations). Caching aggregated results eliminates redundant computation for repeated queries over the same time window or dataset.

---

### 4. ModelServiceOrchestrator

**File**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/models/nodes/node_services/model_service_orchestrator.py`

#### Composition
```python
class ModelServiceOrchestrator(
    MixinNodeService,        # Persistent service mode (FIRST!)
    NodeOrchestrator,        # Workflow coordination
    MixinHealthCheck,        # Health monitoring
    MixinEventBus,           # Event publishing
    MixinMetrics,            # Performance metrics
):
    ...
```

#### Included Capabilities
✅ **Persistent service mode** - Long-lived tool services
✅ **Event-driven coordination** - Workflow lifecycle events (started, completed, failed)
✅ **Subnode health aggregation** - Overall workflow health based on subnode health
✅ **Correlation tracking** - All workflow events share correlation ID
✅ **Performance metrics** - Workflow duration, step counts, success rates
✅ **Error propagation** - Failed subnodes trigger workflow failure events

#### When to Use
- Multi-step workflow coordination requiring event-driven communication
- Dependency management across multiple subnodes
- Workflow lifecycle tracking (started, in-progress, completed, failed)
- Parallel execution coordination with result aggregation

#### Usage Example
```python
from omnibase_core.models.nodes.node_services import ModelServiceOrchestrator
from omnibase_core.models.contracts.model_contract_orchestrator import ModelContractOrchestrator

class NodeWorkflowOrchestrator(ModelServiceOrchestrator):
    """Workflow orchestrator with automatic event coordination and metrics."""

    async def execute_orchestration(self, contract: ModelContractOrchestrator) -> dict:
        # Emit workflow started event
        await self.publish_event(
            event_type="workflow_started",
            payload={"workflow_id": str(contract.workflow_id)},
            correlation_id=contract.correlation_id
        )

        # Coordinate subnode execution
        results = await self._execute_workflow(contract)

        # Emit workflow completed event
        await self.publish_event(
            event_type="workflow_completed",
            payload={
                "workflow_id": str(contract.workflow_id),
                "steps_completed": len(results)
            },
            correlation_id=contract.correlation_id
        )

        return results
```

#### Why MixinEventBus is Critical
Orchestrators emit many events during workflow execution:
- Workflow lifecycle events (started, completed, failed)
- Subnode coordination events
- Progress updates
- Error notifications

---

## MixinNodeService

**File**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/mixins/mixin_node_service.py`

### Purpose
Enables nodes to run as **persistent services** that respond to TOOL_INVOCATION events, providing tool-as-a-service functionality for MCP, GraphQL, and other integrations.

### Why It's First in MRO
Service mode must be initialized **before** other mixins to properly establish the persistent service lifecycle. This enables:
- TOOL_INVOCATION handling
- Long-lived MCP server patterns
- Proper service shutdown

### Key Capabilities

#### 1. Service Lifecycle Management
```python
async def start_service_mode(self) -> None:
    """
    Start the node in persistent service mode.

    1. Publishes introspection on startup
    2. Subscribes to TOOL_INVOCATION events
    3. Starts health monitoring
    4. Enters async event loop
    """

async def stop_service_mode(self) -> None:
    """
    Stop the service mode gracefully.

    1. Emits NODE_SHUTDOWN event
    2. Cancels health monitoring
    3. Waits for active invocations to complete
    4. Cleanup resources
    """
```

#### 2. Tool Invocation Handling
```python
async def handle_tool_invocation(self, event: ModelToolInvocationEvent) -> None:
    """
    Handle a TOOL_INVOCATION event.

    1. Validates the target is this node
    2. Converts event to input state
    3. Calls node.run() with proper context
    4. Emits TOOL_RESPONSE event with results
    """
```

#### 3. Health Monitoring
```python
def get_service_health(self) -> dict[str, Any]:
    """
    Get current service health status.

    Returns:
        {
            "status": "healthy" | "unhealthy",
            "uptime_seconds": int,
            "active_invocations": int,
            "total_invocations": int,
            "successful_invocations": int,
            "failed_invocations": int,
            "success_rate": float,
            "node_id": str,
            "node_name": str,
            "shutdown_requested": bool,
        }
    """
```

#### 4. Graceful Shutdown
```python
def add_shutdown_callback(self, callback: Callable[[], None]) -> None:
    """Add a callback to be called during shutdown."""
```

### Core Attributes
```python
_service_running: bool
_service_task: asyncio.Task[None] | None
_health_task: asyncio.Task[None] | None
_active_invocations: set[UUID]
_total_invocations: int
_successful_invocations: int
_failed_invocations: int
_start_time: float | None
_shutdown_requested: bool
_shutdown_callbacks: list[Callable[[], None]]
```

### Event Bus Integration
MixinNodeService requires an event bus for TOOL_INVOCATION subscription. It tries multiple strategies to find the event bus:
1. `_get_event_bus()` method (from MixinEventBus)
2. Direct `event_bus` attribute
3. `container.get_service("event_bus")`

### Signal Handling
Registers signal handlers for graceful shutdown:
- SIGTERM → Graceful shutdown
- SIGINT (Ctrl+C) → Graceful shutdown

---

## Method Resolution Order (MRO)

### Overview
Python's MRO (Method Resolution Order) determines which method gets called when multiple classes define the same method. Understanding MRO is **critical** for mixin composition.

### MRO Principles
1. **Child classes take precedence** over parent classes
2. **Left-to-right order** in inheritance list matters
3. **All `__init__` methods are called** via `super().__init__()`

### Standard Service MRO

#### ModelServiceEffect
```
ModelServiceEffect
→ MixinNodeService
→ NodeEffect
→ MixinHealthCheck
→ MixinEventBus
→ MixinMetrics
→ NodeCoreBase
→ ABC
```

#### ModelServiceCompute
```
ModelServiceCompute
→ MixinNodeService
→ NodeCompute
→ MixinHealthCheck
→ MixinCaching
→ MixinMetrics
→ NodeCoreBase
→ ABC
```

#### ModelServiceReducer
```
ModelServiceReducer
→ MixinNodeService
→ NodeReducer
→ MixinHealthCheck
→ MixinCaching
→ MixinMetrics
→ NodeCoreBase
→ ABC
```

#### ModelServiceOrchestrator
```
ModelServiceOrchestrator
→ MixinNodeService
→ NodeOrchestrator
→ MixinHealthCheck
→ MixinEventBus
→ MixinMetrics
→ NodeCoreBase
→ ABC
```

### Why MixinNodeService is First

**Critical Reason**: Service mode initialization must happen **before** other mixin initialization to:
1. Establish persistent service lifecycle
2. Enable TOOL_INVOCATION handling
3. Set up event loop for async operations
4. Provide proper shutdown coordination

**Example Problem Without MixinNodeService First**:
```python
# ❌ WRONG - MixinHealthCheck initializes before service mode
class BadServiceEffect(
    NodeEffect,
    MixinNodeService,
    MixinHealthCheck,
):
    pass

# ✅ CORRECT - MixinNodeService initializes first
class GoodServiceEffect(
    MixinNodeService,
    NodeEffect,
    MixinHealthCheck,
):
    pass
```

### Custom Composition MRO Best Practices

Always put the **most specific** class first (node type), followed by mixins in **order of dependency**:

#### Example 1: Validation before Security
```python
class SecureProcessor(
    NodeCompute,
    MixinValidation,      # Validate FIRST
    MixinSecurity,        # Secure AFTER validation
    MixinMetrics,
):
    pass
```

#### Example 2: Retry before Circuit Breaker
```python
class ResilientApiClient(
    NodeEffect,
    MixinRetry,           # Retry FIRST
    MixinCircuitBreaker,  # Circuit break AFTER retries exhausted
    MixinMetrics,
):
    pass
```

---

## Decision Matrix

### Use Convenience Wrappers When:

✅ You need the standard set of capabilities (health, metrics, events/caching)
✅ You're building a typical ONEX node (database adapters, API clients, etc.)
✅ You want minimal boilerplate and fast development
✅ You trust the ONEX team's mixin selection for your node type
✅ **You need persistent service mode** (MCP servers, tool providers)

### Use Custom Composition When:

✅ You need specialized mixin combinations (e.g., Retry + CircuitBreaker + Timeout)
✅ You're building a unique node with specific requirements
✅ You need fine-grained control over mixin initialization order
✅ You want to exclude certain mixins (e.g., no caching for lightweight nodes)
✅ You don't need persistent service mode (one-shot execution)

### Decision Tree

```
Start: Do I need persistent service mode (long-lived MCP server)?
  │
  ├─ YES → Use Convenience Wrapper (ModelService*)
  │   │
  │   └─ Which node type?
  │       ├─ Side effects, I/O → ModelServiceEffect
  │       ├─ Pure computation → ModelServiceCompute
  │       ├─ Aggregation → ModelServiceReducer
  │       └─ Workflow coordination → ModelServiceOrchestrator
  │
  └─ NO → Do I need standard capabilities (health, metrics, events/caching)?
      │
      ├─ YES → Still use Convenience Wrapper (easier)
      │
      └─ NO → Custom Composition
          │
          └─ Which mixins do I need?
              ├─ Flow Control: MixinRetry, MixinCircuitBreaker
              ├─ Monitoring: MixinHealthCheck, MixinMetrics, MixinLogging
              ├─ Data: MixinCaching, MixinSerialization
              ├─ Communication: MixinEventBus
              └─ Security: MixinSecurity, MixinValidation
```

---

## Import Paths

### Base Node Classes
```python
from omnibase_core.nodes.node_effect import NodeEffect
from omnibase_core.nodes.node_compute import NodeCompute
from omnibase_core.nodes.node_reducer import NodeReducer
from omnibase_core.nodes.node_orchestrator import NodeOrchestrator
```

### Convenience Wrappers
```python
from omnibase_core.models.nodes.node_services import ModelServiceEffect
from omnibase_core.models.nodes.node_services import ModelServiceCompute
from omnibase_core.models.nodes.node_services import ModelServiceReducer
from omnibase_core.models.nodes.node_services import ModelServiceOrchestrator

# OR individual imports
from omnibase_core.models.nodes.node_services.model_service_effect import ModelServiceEffect
from omnibase_core.models.nodes.node_services.model_service_compute import ModelServiceCompute
from omnibase_core.models.nodes.node_services.model_service_reducer import ModelServiceReducer
from omnibase_core.models.nodes.node_services.model_service_orchestrator import ModelServiceOrchestrator
```

### MixinNodeService
```python
from omnibase_core.mixins.mixin_node_service import MixinNodeService
```

### Common Mixins
```python
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck
from omnibase_core.mixins.mixin_event_bus import MixinEventBus
from omnibase_core.mixins.mixin_metrics import MixinMetrics
from omnibase_core.mixins.mixin_caching import MixinCaching
from omnibase_core.mixins.mixin_retry import MixinRetry
from omnibase_core.mixins.mixin_circuit_breaker import MixinCircuitBreaker
from omnibase_core.mixins.mixin_logging import MixinLogging
from omnibase_core.mixins.mixin_security import MixinSecurity
from omnibase_core.mixins.mixin_validation import MixinValidation
from omnibase_core.mixins.mixin_serialization import MixinSerialization
```

### Container
```python
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
```

### Contracts
```python
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.contracts.model_contract_compute import ModelContractCompute
from omnibase_core.models.contracts.model_contract_reducer import ModelContractReducer
from omnibase_core.models.contracts.model_contract_orchestrator import ModelContractOrchestrator
```

---

## Code Examples

### Example 1: Using Convenience Wrapper (Recommended)

```python
"""
Database writer using ModelServiceEffect convenience wrapper.
Includes automatic health checks, events, and metrics.
"""
from omnibase_core.models.nodes.node_services import ModelServiceEffect
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect

class NodeDatabaseWriterEffect(ModelServiceEffect):
    """Database writer with automatic health checks, events, and metrics."""

    def __init__(self, container: ModelONEXContainer):
        super().__init__(container)
        # No need to initialize mixins - done automatically!

    async def execute_effect(self, contract: ModelContractEffect) -> dict:
        """Execute database write operation."""
        # Just write your business logic!
        result = await self.database.write(contract.input_data)

        # Emit event automatically tracked with metrics
        await self.publish_event(
            event_type="write_completed",
            payload={"records_written": result["count"]},
            correlation_id=contract.correlation_id
        )

        return {"status": "success", "data": result}

# Usage in persistent service mode
if __name__ == "__main__":
    container = ModelONEXContainer.create_default()
    node = NodeDatabaseWriterEffect(container)

    # Start as persistent service (MCP server)
    await node.start_service_mode()
```

### Example 2: Custom Composition for Specialized Requirements

```python
"""
Fault-tolerant API client with custom mixin composition.
Adds retry and circuit breaker for transient failure handling.
Omits event bus (not needed for API client).
"""
from omnibase_core.nodes.node_effect import NodeEffect
from omnibase_core.mixins.mixin_retry import MixinRetry
from omnibase_core.mixins.mixin_circuit_breaker import MixinCircuitBreaker
from omnibase_core.mixins.mixin_metrics import MixinMetrics
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect

class ResilientApiClient(
    NodeEffect,
    MixinRetry,
    MixinCircuitBreaker,
    MixinMetrics,
):
    """
    Custom composition for fault-tolerant API client.

    MRO: ResilientApiClient → NodeEffect → MixinRetry
         → MixinCircuitBreaker → MixinMetrics → NodeCoreBase
    """

    def __init__(self, container: ModelONEXContainer):
        super().__init__(container)
        # Configure retry policy
        self.retry_max_attempts = 3
        self.retry_backoff_factor = 2.0

        # Configure circuit breaker
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout_ms = 60000

    async def execute_effect(self, contract: ModelContractEffect) -> dict:
        """Execute API call with automatic retry and circuit breaker."""
        # Retry and circuit breaker handled automatically by mixins!
        result = await self._call_external_api(contract.input_data)
        return {"status": "success", "data": result}

# Usage (one-shot execution)
if __name__ == "__main__":
    container = ModelONEXContainer.create_default()
    node = ResilientApiClient(container)

    # Direct execution (not service mode)
    contract = ModelContractEffect(...)
    result = await node.execute_effect(contract)
```

### Example 3: High-Performance Compute Node (No Caching)

```python
"""
High-throughput stream processing compute node.
Omits caching (data is never repeated).
Includes only health checks and metrics for monitoring.
"""
from omnibase_core.nodes.node_compute import NodeCompute
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck
from omnibase_core.mixins.mixin_metrics import MixinMetrics
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.model_compute_input import ModelComputeInput
from omnibase_core.models.model_compute_output import ModelComputeOutput

class StreamProcessorCompute(
    NodeCompute,
    MixinHealthCheck,
    MixinMetrics,
):
    """
    Custom composition for high-throughput stream processing.

    Omits caching (data never repeated).
    Includes only health checks and metrics for monitoring.
    """

    async def process(self, input_data: ModelComputeInput) -> ModelComputeOutput:
        """Process streaming data without caching."""
        # Process data in real-time
        result = await self._process_stream(input_data.data)

        return ModelComputeOutput(
            result=result,
            operation_id=input_data.operation_id,
            computation_type=input_data.computation_type,
            processing_time_ms=0.0,  # Calculated by base class
            cache_hit=False,
            parallel_execution_used=False,
        )
```

### Example 4: Secure Data Processor

```python
"""
Secure data processor with validation and redaction.
Custom composition for processing sensitive data.
"""
from omnibase_core.nodes.node_compute import NodeCompute
from omnibase_core.mixins.mixin_security import MixinSecurity
from omnibase_core.mixins.mixin_validation import MixinValidation
from omnibase_core.mixins.mixin_logging import MixinLogging
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.model_compute_input import ModelComputeInput
from omnibase_core.models.model_compute_output import ModelComputeOutput

class SecureDataProcessor(
    NodeCompute,
    MixinValidation,     # Validate FIRST
    MixinSecurity,       # Secure AFTER validation
    MixinLogging,
):
    """
    Custom composition for processing sensitive data.

    Adds security (redaction) and validation (fail-fast).
    Omits caching (never cache sensitive data).

    MRO: SecureDataProcessor → NodeCompute → MixinValidation
         → MixinSecurity → MixinLogging → NodeCoreBase
    """

    async def process(self, input_data: ModelComputeInput) -> ModelComputeOutput:
        """Process sensitive data with validation and redaction."""
        # Validation happens automatically via MixinValidation
        # Security redaction happens automatically via MixinSecurity

        result = await self._process_sensitive_data(input_data.data)

        # Log without sensitive fields (redacted by MixinSecurity)
        self.log_info(f"Processed secure data: {result}")

        return ModelComputeOutput(
            result=result,
            operation_id=input_data.operation_id,
            computation_type=input_data.computation_type,
            processing_time_ms=0.0,
            cache_hit=False,
            parallel_execution_used=False,
        )
```

### Example 5: Using ModelServiceCompute with Caching

```python
"""
ML inference compute node using ModelServiceCompute with caching.
Demonstrates cache-aware implementation for expensive operations.
"""
from omnibase_core.models.nodes.node_services import ModelServiceCompute
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.contracts.model_contract_compute import ModelContractCompute

class NodeMLInferenceCompute(ModelServiceCompute):
    """ML inference with automatic result caching."""

    async def execute_compute(self, contract: ModelContractCompute) -> dict:
        """Execute ML inference with caching."""
        # Generate cache key from input features
        cache_key = self.generate_cache_key(contract.input_data)

        # Check cache first (automatic via MixinCaching)
        cached_result = await self.get_cached(cache_key)
        if cached_result:
            return cached_result  # Cache hit - instant response!

        # Cache miss - perform expensive ML inference
        result = await self._run_ml_model(contract.input_data)

        # Cache result for 1 hour (models don't change often)
        await self.set_cached(cache_key, result, ttl_seconds=3600)

        return result

# Usage in persistent service mode
if __name__ == "__main__":
    container = ModelONEXContainer.create_default()
    node = NodeMLInferenceCompute(container)

    # Start as persistent service (MCP server)
    await node.start_service_mode()

    # Service now responds to TOOL_INVOCATION events
    # Inference results are cached automatically
```

---

## Summary

### Key Takeaways

1. **Four Base Classes**: NodeEffect, NodeCompute, NodeReducer, NodeOrchestrator provide fundamental node semantics
2. **Four Convenience Wrappers**: ModelServiceEffect, ModelServiceCompute, ModelServiceReducer, ModelServiceOrchestrator pre-compose standard mixins
3. **MixinNodeService is First**: Enables persistent service mode for long-lived MCP servers and tool invocation
4. **MRO Matters**: Mixin order determines initialization sequence and method resolution
5. **Use Wrappers by Default**: 80% of nodes should use convenience wrappers
6. **Custom Composition for Edge Cases**: 20% of nodes need specialized mixin combinations

### When to Use Each Approach

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Standard database adapter | `ModelServiceEffect` | Needs transactions, events, health checks |
| Standard data transformer | `ModelServiceCompute` | Needs caching, health checks, metrics |
| Standard metrics aggregator | `ModelServiceReducer` | Needs caching, health checks, metrics |
| Standard workflow orchestrator | `ModelServiceOrchestrator` | Needs events, health checks, metrics |
| Fault-tolerant API client | Custom composition | Needs retry + circuit breaker, no events |
| High-throughput stream processor | Custom composition | No caching needed, minimize overhead |
| Secure data processor | Custom composition | Needs validation + security, no caching |

### Performance Characteristics

| Service Wrapper | Overhead per Call | Memory per Instance | Recommended Use Cases |
|----------------|-------------------|---------------------|----------------------|
| **ModelServiceEffect** | ~10-20ms | ~10-20KB | Database adapters, API clients, file I/O |
| **ModelServiceCompute** | ~5-15ms | ~15-30KB (with cache) | Data transformers, ML inference, calculations |
| **ModelServiceOrchestrator** | ~15-30ms | ~20-40KB | Workflow coordinators, multi-step processes |
| **ModelServiceReducer** | ~10-25ms | ~15-35KB (with cache) | Metrics aggregators, log analyzers, analytics |

### Next Steps

1. Choose the appropriate convenience wrapper OR decide on custom composition
2. Implement your business logic in `execute_*` or `process()` method
3. Add custom health checks if needed (override `get_health_checks()`)
4. Test with mocked and real mixins
5. Monitor metrics and health in production

---

## References

- **Base Classes**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/nodes/`
- **Convenience Wrappers**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/models/nodes/node_services/`
- **MixinNodeService**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/mixins/mixin_node_service.py`
- **Mixin Metadata**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/mixins/mixin_metadata.yaml`
- **Container Documentation**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/models/container/model_onex_container.py`

**Last Updated**: 2025-11-05
**Version**: 1.0.0
**Status**: Complete
