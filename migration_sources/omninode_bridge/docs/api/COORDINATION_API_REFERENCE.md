# Phase 4 Coordination API Reference

**Version**: 1.0
**Status**: ✅ Production-Ready
**Last Updated**: 2025-11-06

---

## Table of Contents

1. [Signal Coordination API](#signal-coordination-api)
2. [Routing Orchestration API](#routing-orchestration-api)
3. [Context Distribution API](#context-distribution-api)
4. [Dependency Resolution API](#dependency-resolution-api)
5. [Data Models](#data-models)
6. [Exceptions](#exceptions)

---

## Signal Coordination API

### SignalCoordinator

High-performance signal coordinator for agent-to-agent communication.

#### Constructor

```python
SignalCoordinator(
    state: ThreadSafeState,
    metrics_collector: Optional[MetricsCollector] = None,
    max_history_size: int = 10000
)
```

**Parameters**:
- `state`: ThreadSafeState instance for centralized storage
- `metrics_collector`: Optional metrics collector for signal tracking
- `max_history_size`: Maximum signal history per coordination session (default: 10,000)

**Example**:
```python
from omninode_bridge.agents.coordination import ThreadSafeState, SignalCoordinator
from omninode_bridge.agents.metrics import MetricsCollector

state = ThreadSafeState()
metrics = MetricsCollector()
await metrics.start()

coordinator = SignalCoordinator(
    state=state,
    metrics_collector=metrics,
    max_history_size=5000
)
```

---

#### signal_coordination_event()

Send coordination signal to participating agents.

```python
async def signal_coordination_event(
    coordination_id: str,
    event_type: str,
    event_data: dict,
    sender_agent_id: Optional[str] = None,
    recipient_agents: Optional[list[str]] = None,
    metadata: Optional[dict] = None
) -> bool
```

**Parameters**:
- `coordination_id`: Coordination session identifier
- `event_type`: Signal type (`agent_initialized`, `agent_completed`, `dependency_resolved`, `inter_agent_message`)
- `event_data`: Signal-specific event data
- `sender_agent_id`: Agent that sent signal (default: `"system"`)
- `recipient_agents`: List of recipient agent IDs (empty = broadcast)
- `metadata`: Additional metadata (correlation_id, priority, etc.)

**Returns**: `True` if signal sent successfully, `False` otherwise

**Performance**: <100ms target (actual: ~3ms)

**Example**:
```python
# Send agent completed signal
success = await coordinator.signal_coordination_event(
    coordination_id="codegen-session-1",
    event_type="agent_completed",
    event_data={
        "agent_id": "model-gen",
        "result_summary": "Generated 5 models",
        "quality_score": 0.95,
        "execution_time_ms": 1234.5
    },
    sender_agent_id="model-gen"
)

if success:
    print("Signal sent successfully")
```

---

#### subscribe_to_signals()

Subscribe to coordination signals with filtering.

```python
async def subscribe_to_signals(
    coordination_id: str,
    agent_id: str,
    signal_types: Optional[list[str]] = None
) -> AsyncIterator[CoordinationSignal]
```

**Parameters**:
- `coordination_id`: Coordination session to subscribe to
- `agent_id`: Subscribing agent ID
- `signal_types`: List of signal types to receive (None = all types)

**Yields**: `CoordinationSignal` objects matching subscription

**Example**:
```python
# Subscribe to completion signals
async for signal in coordinator.subscribe_to_signals(
    coordination_id="codegen-session-1",
    agent_id="validator-gen",
    signal_types=["agent_completed", "dependency_resolved"]
):
    if signal.signal_type == "agent_completed":
        print(f"Agent {signal.event_data['agent_id']} completed")
        # Process completion
        break
```

---

#### get_signal_history()

Retrieve signal history with optional filtering.

```python
def get_signal_history(
    coordination_id: str,
    filters: Optional[dict] = None,
    limit: Optional[int] = None
) -> list[CoordinationSignal]
```

**Parameters**:
- `coordination_id`: Coordination session identifier
- `filters`: Optional filters (`signal_type`, `sender_agent_id`, etc.)
- `limit`: Maximum number of signals to return (None = all)

**Returns**: List of `CoordinationSignal` (most recent first)

**Example**:
```python
# Get last 10 completion signals
history = coordinator.get_signal_history(
    coordination_id="codegen-session-1",
    filters={"signal_type": "agent_completed"},
    limit=10
)

for signal in history:
    print(f"Agent: {signal.sender_agent_id}, Time: {signal.timestamp}")
```

---

#### get_signal_metrics()

Get signal metrics for coordination session.

```python
def get_signal_metrics(
    coordination_id: str
) -> SignalMetrics
```

**Parameters**:
- `coordination_id`: Coordination session identifier

**Returns**: `SignalMetrics` with aggregated statistics

**Example**:
```python
metrics = coordinator.get_signal_metrics("codegen-session-1")
print(f"Total signals: {metrics.total_signals_sent}")
print(f"Avg propagation: {metrics.average_propagation_ms:.2f}ms")
print(f"Max propagation: {metrics.max_propagation_ms:.2f}ms")
```

---

## Routing Orchestration API

### SmartRoutingOrchestrator

Orchestrates multiple routing strategies with priority-based consolidation.

#### Constructor

```python
SmartRoutingOrchestrator(
    metrics_collector: Optional[MetricsCollector] = None,
    state: Optional[ThreadSafeState] = None,
    max_history_size: int = 1000
)
```

**Parameters**:
- `metrics_collector`: Optional metrics collector for performance tracking
- `state`: Optional ThreadSafeState for state access
- `max_history_size`: Maximum routing history records (default: 1,000)

**Example**:
```python
from omninode_bridge.agents.coordination import SmartRoutingOrchestrator
from omninode_bridge.agents.metrics import MetricsCollector

metrics = MetricsCollector()
await metrics.start()

orchestrator = SmartRoutingOrchestrator(
    metrics_collector=metrics,
    max_history_size=500
)
```

---

#### add_router()

Add a routing strategy.

```python
def add_router(
    router: BaseRouter
) -> None
```

**Parameters**:
- `router`: Router to add (ConditionalRouter, ParallelRouter, StateAnalysisRouter, PriorityRouter)

**Example**:
```python
from omninode_bridge.agents.coordination import (
    ConditionalRouter,
    ConditionalRule,
    RoutingDecision
)

# Create conditional router
rules = [
    ConditionalRule(
        rule_id="error_handling",
        name="Retry on Error",
        condition_key="error_count",
        condition_operator=">",
        condition_value=0,
        decision=RoutingDecision.RETRY,
        priority=90
    )
]
router = ConditionalRouter(rules=rules)

# Add to orchestrator
orchestrator.add_router(router)
```

---

#### remove_router()

Remove a routing strategy.

```python
def remove_router(
    strategy: RoutingStrategy
) -> None
```

**Parameters**:
- `strategy`: Strategy to remove (CONDITIONAL, PARALLEL, STATE_ANALYSIS, PRIORITY)

**Example**:
```python
from omninode_bridge.agents.coordination import RoutingStrategy

orchestrator.remove_router(RoutingStrategy.PARALLEL)
```

---

#### route()

Make routing decision using all registered routers.

```python
def route(
    state: dict[str, Any],
    current_task: str,
    execution_time: float = 0.0,
    retry_count: int = 0,
    custom_data: Optional[dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> dict[str, Any]
```

**Parameters**:
- `state`: Current state snapshot
- `current_task`: Task being evaluated
- `execution_time`: Time spent executing current task (ms)
- `retry_count`: Number of retries for current task
- `custom_data`: Additional context data
- `correlation_id`: Optional correlation ID for tracing

**Returns**: Dictionary with routing decision and metadata

**Performance**: <5ms target (actual: ~0.018ms)

**Example**:
```python
state_snapshot = {
    "error_count": 0,
    "completed_tasks": ["parse_contract"],
    "task_priority": 80
}

result = orchestrator.route(
    state=state_snapshot,
    current_task="generate_model",
    execution_time=45.2,
    retry_count=0
)

print(f"Decision: {result['decision']}")  # e.g., "continue"
print(f"Confidence: {result['confidence']}")  # e.g., 0.9
print(f"Reasoning: {result['reasoning']}")
print(f"Routing time: {result['routing_time_ms']:.2f}ms")
```

---

#### get_history()

Get routing history.

```python
def get_history(
    task: Optional[str] = None,
    limit: Optional[int] = None
) -> list[RoutingHistoryRecord]
```

**Parameters**:
- `task`: Filter by task name (None = all tasks)
- `limit`: Maximum number of records (None = all)

**Returns**: List of `RoutingHistoryRecord` (most recent first)

**Example**:
```python
history = orchestrator.get_history(task="generate_model", limit=10)

for record in history:
    print(f"Task: {record.context.current_task}")
    print(f"Decision: {record.result.decision}")
    print(f"Routing time: {record.routing_time_ms:.2f}ms")
```

---

#### get_stats()

Get routing statistics.

```python
def get_stats() -> dict[str, Any]
```

**Returns**: Dictionary with routing statistics

**Example**:
```python
stats = orchestrator.get_stats()
print(f"Total routings: {stats['total_routings']}")
print(f"Avg routing time: {stats['avg_routing_time_ms']:.2f}ms")
print(f"Decisions: {stats['decisions']}")
print(f"Strategies: {stats['strategies']}")
```

---

#### clear_history()

Clear routing history.

```python
def clear_history() -> None
```

**Example**:
```python
orchestrator.clear_history()
```

---

### BaseRouter

Abstract base class for routing strategies.

#### evaluate()

Evaluate routing decision based on state and context.

```python
@abstractmethod
def evaluate(
    state: dict[str, Any],
    context: RoutingContext
) -> RoutingResult
```

**Parameters**:
- `state`: Current state snapshot
- `context`: Routing context

**Returns**: `RoutingResult` with decision and reasoning

---

### ConditionalRouter

Routes based on state conditions.

#### Constructor

```python
ConditionalRouter(
    rules: list[ConditionalRule],
    metrics_collector: Optional[MetricsCollector] = None
)
```

**Parameters**:
- `rules`: List of conditional rules (evaluated by priority)
- `metrics_collector`: Optional metrics collector

**Example**:
```python
rules = [
    ConditionalRule(
        rule_id="simple_model",
        name="Simple Model Detection",
        condition_key="field_count",
        condition_operator="<=",
        condition_value=5,
        decision=RoutingDecision.CONTINUE,
        next_task="basic_generator",
        priority=90
    )
]

router = ConditionalRouter(rules=rules)
result = router.evaluate(state={"field_count": 3}, context=context)
```

---

### ParallelRouter

Identifies tasks for parallel execution.

#### Constructor

```python
ParallelRouter(
    parallelization_hints: Optional[list[ParallelizationHint]] = None,
    metrics_collector: Optional[MetricsCollector] = None
)
```

**Parameters**:
- `parallelization_hints`: Optional hints for parallel execution
- `metrics_collector`: Optional metrics collector

**Example**:
```python
from omninode_bridge.agents.coordination import ParallelizationHint

hints = [
    ParallelizationHint(
        task_group=["generate_model", "generate_validator", "generate_test"],
        dependencies=["parse_contract"]
    )
]

router = ParallelRouter(parallelization_hints=hints)
result = router.evaluate(state=state, context=context)

if result.decision == RoutingDecision.PARALLEL:
    parallel_tasks = result.metadata["parallel_tasks"]
    print(f"Can parallelize with: {parallel_tasks}")
```

---

### StateAnalysisRouter

Analyzes state complexity for routing decisions.

#### Constructor

```python
StateAnalysisRouter(
    max_complexity_score: float = 0.8,
    error_handling_decision: RoutingDecision = RoutingDecision.RETRY,
    metrics_collector: Optional[MetricsCollector] = None
)
```

**Parameters**:
- `max_complexity_score`: Maximum complexity before branching (0.0-1.0)
- `error_handling_decision`: Decision to make when errors detected
- `metrics_collector`: Optional metrics collector

**Example**:
```python
router = StateAnalysisRouter(
    max_complexity_score=0.7,
    error_handling_decision=RoutingDecision.RETRY
)

result = router.evaluate(state=state, context=context)

metrics = result.metadata["complexity_metrics"]
print(f"Complexity score: {metrics['complexity_score']:.2f}")
print(f"Has errors: {metrics['has_errors']}")
```

---

### PriorityRouter

Routes based on task priority levels.

#### Constructor

```python
PriorityRouter(
    config: Optional[PriorityRoutingConfig] = None,
    metrics_collector: Optional[MetricsCollector] = None
)
```

**Parameters**:
- `config`: Optional priority routing configuration
- `metrics_collector`: Optional metrics collector

**Example**:
```python
from omninode_bridge.agents.coordination import PriorityRoutingConfig

config = PriorityRoutingConfig(
    high_priority_threshold=80,
    low_priority_threshold=20,
    high_priority_decision=RoutingDecision.CONTINUE,
    low_priority_decision=RoutingDecision.SKIP
)

router = PriorityRouter(config=config)
result = router.evaluate(state={"task_priority": 90}, context=context)
```

---

## Context Distribution API

### ContextDistributor

Agent context distribution system for parallel coordination.

#### Constructor

```python
ContextDistributor(
    state: ThreadSafeState,
    metrics_collector: Optional[MetricsCollector] = None,
    default_resource_allocation: Optional[ResourceAllocation] = None,
    default_coordination_protocols: Optional[CoordinationProtocols] = None
)
```

**Parameters**:
- `state`: ThreadSafeState instance for context storage
- `metrics_collector`: Optional MetricsCollector for distribution tracking
- `default_resource_allocation`: Default resource limits for agents
- `default_coordination_protocols`: Default coordination protocols

**Example**:
```python
from omninode_bridge.agents.coordination import (
    ContextDistributor,
    ResourceAllocation,
    CoordinationProtocols
)

distributor = ContextDistributor(
    state=state,
    metrics_collector=metrics,
    default_resource_allocation=ResourceAllocation(
        max_execution_time_ms=300000,
        max_retry_attempts=3,
        quality_threshold=0.8
    )
)
```

---

#### distribute_agent_context()

Distribute specialized context to each parallel agent.

```python
async def distribute_agent_context(
    coordination_state: dict[str, Any],
    agent_assignments: dict[str, dict[str, Any]],
    shared_intelligence: Optional[SharedIntelligence] = None,
    resource_allocations: Optional[dict[str, ResourceAllocation]] = None,
    coordination_protocols: Optional[dict[str, CoordinationProtocols]] = None
) -> dict[str, AgentContext]
```

**Parameters**:
- `coordination_state`: Coordination workflow state (must contain `coordination_id` and `session_id`)
- `agent_assignments`: Agent ID to assignment mapping
- `shared_intelligence`: Optional shared intelligence (type registry, patterns, etc.)
- `resource_allocations`: Optional per-agent resource allocations
- `coordination_protocols`: Optional per-agent coordination protocols

**Returns**: Dictionary mapping agent_id to `AgentContext`

**Performance**: <200ms per agent target (actual: ~15ms)

**Raises**:
- `ValueError`: If coordination_state is missing required fields
- `RuntimeError`: If distribution fails

**Example**:
```python
contexts = await distributor.distribute_agent_context(
    coordination_state={
        "coordination_id": "coord-123",
        "session_id": "session-456"
    },
    agent_assignments={
        "model_gen": {
            "agent_role": "model_generator",
            "objective": "Generate Pydantic models",
            "tasks": ["parse_contract", "generate_models"],
            "input_data": {"contract_path": "./contract.yaml"},
            "dependencies": []
        },
        "validator_gen": {
            "agent_role": "validator_generator",
            "objective": "Generate validators",
            "tasks": ["generate_validators"],
            "dependencies": ["model_gen"]
        }
    }
)

# Access agent context
model_gen_context = contexts["model_gen"]
print(f"Tasks: {model_gen_context.agent_assignment.tasks}")
```

---

#### get_agent_context()

Retrieve context for specific agent.

```python
def get_agent_context(
    coordination_id: str,
    agent_id: str
) -> Optional[AgentContext]
```

**Parameters**:
- `coordination_id`: Coordination workflow ID
- `agent_id`: Agent identifier

**Returns**: `AgentContext` if found, `None` otherwise

**Performance**: <5ms target (actual: ~0.5ms)

**Example**:
```python
context = distributor.get_agent_context("coord-123", "model_gen")
if context:
    print(f"Agent role: {context.coordination_metadata.agent_role}")
    print(f"Quality threshold: {context.resource_allocation.quality_threshold}")
```

---

#### update_shared_intelligence()

Update shared intelligence across agents.

```python
def update_shared_intelligence(
    update_request: ContextUpdateRequest
) -> dict[str, bool]
```

**Parameters**:
- `update_request`: Context update request

**Returns**: Dictionary mapping agent_id to success status

**Example**:
```python
from omninode_bridge.agents.coordination import ContextUpdateRequest

results = distributor.update_shared_intelligence(
    ContextUpdateRequest(
        coordination_id="coord-123",
        update_type="type_registry",
        update_data={"CustomType": "class CustomType(BaseModel): ..."},
        target_agents=None,  # Update all agents
        increment_version=True
    )
)

for agent_id, success in results.items():
    print(f"Agent {agent_id}: {'✅' if success else '❌'}")
```

---

#### list_coordination_contexts()

List all agent IDs with contexts for a coordination workflow.

```python
def list_coordination_contexts(
    coordination_id: str
) -> list[str]
```

**Parameters**:
- `coordination_id`: Coordination workflow ID

**Returns**: List of agent IDs

**Example**:
```python
agent_ids = distributor.list_coordination_contexts("coord-123")
print(f"Agents: {agent_ids}")
```

---

#### clear_coordination_contexts()

Clear all contexts for a coordination workflow.

```python
def clear_coordination_contexts(
    coordination_id: str
) -> bool
```

**Parameters**:
- `coordination_id`: Coordination workflow ID

**Returns**: `True` if contexts were cleared, `False` if no contexts found

**Example**:
```python
# After workflow completion
cleared = distributor.clear_coordination_contexts("coord-123")
if cleared:
    print("Contexts cleared successfully")
```

---

## Dependency Resolution API

### DependencyResolver

Dependency resolver for multi-agent coordination.

#### Constructor

```python
DependencyResolver(
    signal_coordinator: ISignalCoordinator,
    metrics_collector: MetricsCollector,
    state: Optional[ThreadSafeState] = None,
    max_concurrent_resolutions: int = 10
)
```

**Parameters**:
- `signal_coordinator`: Signal coordinator for event signaling
- `metrics_collector`: Metrics collector for performance tracking
- `state`: Optional shared state for resource availability checks
- `max_concurrent_resolutions`: Maximum concurrent dependency resolutions (default: 10)

**Example**:
```python
from omninode_bridge.agents.coordination import DependencyResolver

resolver = DependencyResolver(
    signal_coordinator=signal_coordinator,
    metrics_collector=metrics,
    state=state,
    max_concurrent_resolutions=20
)
```

---

#### resolve_agent_dependencies()

Resolve all dependencies for an agent.

```python
async def resolve_agent_dependencies(
    coordination_id: str,
    agent_context: dict[str, Any]
) -> bool
```

**Parameters**:
- `coordination_id`: Coordination session ID
- `agent_context`: Agent context containing dependency specifications

**Returns**: `True` if all dependencies resolved successfully, `False` otherwise

**Performance**: <2s total target (actual: <500ms)

**Raises**:
- `DependencyResolutionError`: If dependency resolution fails
- `DependencyTimeoutError`: If dependency resolution times out

**Example**:
```python
agent_context = {
    "agent_id": "validator-gen",
    "dependencies": [
        {
            "dependency_id": "model_gen_complete",
            "type": "agent_completion",
            "target": "model-gen",
            "timeout": 120,
            "metadata": {"agent_id": "model-gen"}
        }
    ]
}

try:
    success = await resolver.resolve_agent_dependencies(
        coordination_id="coord-123",
        agent_context=agent_context
    )

    if success:
        print("All dependencies resolved")
    else:
        print("Dependency resolution failed")

except DependencyTimeoutError as e:
    print(f"Timeout: {e}")
```

---

#### resolve_dependency()

Resolve a single dependency.

```python
async def resolve_dependency(
    coordination_id: str,
    dependency: Dependency
) -> DependencyResolutionResult
```

**Parameters**:
- `coordination_id`: Coordination session ID
- `dependency`: Dependency to resolve

**Returns**: `DependencyResolutionResult` with resolution status and metrics

**Raises**:
- `DependencyTimeoutError`: If dependency resolution times out

**Example**:
```python
from omninode_bridge.agents.coordination import Dependency, DependencyType

dependency = Dependency(
    dependency_id="resource_ready",
    dependency_type=DependencyType.RESOURCE_AVAILABILITY,
    target="resource-1",
    timeout=60
)

result = await resolver.resolve_dependency("coord-123", dependency)

print(f"Success: {result.success}")
print(f"Duration: {result.duration_ms:.2f}ms")
print(f"Attempts: {result.attempts}")
```

---

#### get_dependency_status()

Get status of specific dependency.

```python
def get_dependency_status(
    coordination_id: str,
    dependency_id: str
) -> dict[str, Any]
```

**Parameters**:
- `coordination_id`: Coordination session ID
- `dependency_id`: Dependency identifier

**Returns**: Dictionary with dependency status information

**Example**:
```python
status = resolver.get_dependency_status("coord-123", "model_gen_complete")
print(f"Resolved: {status['resolved']}")
print(f"Status: {status['status']}")
print(f"Resolved at: {status['resolved_at']}")
```

---

#### mark_resource_available()

Mark a resource as available/unavailable.

```python
async def mark_resource_available(
    resource_id: str,
    available: bool = True
) -> None
```

**Parameters**:
- `resource_id`: Resource identifier
- `available`: Availability status

**Example**:
```python
# Mark resource as available
await resolver.mark_resource_available("resource-1", available=True)

# Mark resource as unavailable
await resolver.mark_resource_available("resource-1", available=False)
```

---

#### update_quality_gate_score()

Update quality gate score.

```python
async def update_quality_gate_score(
    gate_id: str,
    score: float
) -> None
```

**Parameters**:
- `gate_id`: Quality gate identifier
- `score`: Quality gate score (0.0-1.0)

**Example**:
```python
# Update quality gate score
await resolver.update_quality_gate_score("coverage_gate", 0.85)
```

---

#### clear_coordination_dependencies()

Clear dependencies for coordination session.

```python
def clear_coordination_dependencies(
    coordination_id: str
) -> None
```

**Parameters**:
- `coordination_id`: Coordination session ID

**Example**:
```python
# After workflow completion
resolver.clear_coordination_dependencies("coord-123")
```

---

#### get_pending_dependencies_count()

Get count of pending dependencies.

```python
def get_pending_dependencies_count(
    coordination_id: str
) -> int
```

**Parameters**:
- `coordination_id`: Coordination session ID

**Returns**: Number of pending dependencies

**Example**:
```python
pending_count = resolver.get_pending_dependencies_count("coord-123")
print(f"Pending dependencies: {pending_count}")
```

---

## Data Models

### Signal Models

#### CoordinationSignal

```python
@dataclass
class CoordinationSignal:
    signal_type: SignalType
    sender_agent_id: str
    recipient_agents: list[str]
    event_data: dict[str, Any]
    metadata: dict[str, Any]
    coordination_id: str
    timestamp: datetime
```

#### SignalType (Enum)

```python
class SignalType(str, Enum):
    AGENT_INITIALIZED = "agent_initialized"
    AGENT_COMPLETED = "agent_completed"
    DEPENDENCY_RESOLVED = "dependency_resolved"
    INTER_AGENT_MESSAGE = "inter_agent_message"
```

#### SignalMetrics

```python
@dataclass
class SignalMetrics:
    total_signals_sent: int = 0
    average_propagation_ms: float = 0.0
    max_propagation_ms: float = 0.0
    signals_by_type: dict[str, int] = field(default_factory=dict)
```

---

### Routing Models

#### RoutingDecision (Enum)

```python
class RoutingDecision(str, Enum):
    ERROR = "error"
    END = "end"
    RETRY = "retry"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    BRANCH = "branch"
    SKIP = "skip"
    CONTINUE = "continue"
```

#### RoutingResult

```python
@dataclass
class RoutingResult:
    decision: RoutingDecision
    confidence: float  # 0.0-1.0
    reasoning: str
    strategy: RoutingStrategy
    next_task: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### RoutingStrategy (Enum)

```python
class RoutingStrategy(str, Enum):
    CONDITIONAL = "conditional"
    STATE_ANALYSIS = "state_analysis"
    PARALLEL = "parallel"
    PRIORITY = "priority"
```

#### ConditionalRule

```python
@dataclass
class ConditionalRule:
    rule_id: str
    name: str
    condition_key: str
    condition_operator: str  # "==", "!=", ">", "<", ">=", "<=", "in", "not_in", "contains", "not_contains"
    condition_value: Any
    decision: RoutingDecision
    priority: int
    next_task: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### ParallelizationHint

```python
@dataclass
class ParallelizationHint:
    task_group: list[str]
    dependencies: list[str]
    estimated_duration_ms: float = 0.0
```

---

### Context Models

#### AgentContext

```python
@dataclass
class AgentContext:
    coordination_metadata: CoordinationMetadata
    shared_intelligence: SharedIntelligence
    agent_assignment: AgentAssignment
    coordination_protocols: CoordinationProtocols
    resource_allocation: ResourceAllocation
    context_version: int
```

#### CoordinationMetadata

```python
@dataclass
class CoordinationMetadata:
    session_id: str
    coordination_id: str
    agent_id: str
    agent_role: str
    parent_workflow_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
```

#### SharedIntelligence

```python
@dataclass
class SharedIntelligence:
    type_registry: dict[str, Any] = field(default_factory=dict)
    pattern_library: dict[str, list] = field(default_factory=dict)
    validation_rules: dict[str, Any] = field(default_factory=dict)
    naming_conventions: dict[str, str] = field(default_factory=dict)
    dependency_graph: dict[str, list] = field(default_factory=dict)
```

#### AgentAssignment

```python
@dataclass
class AgentAssignment:
    objective: str
    tasks: list[str]
    input_data: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    output_requirements: dict[str, Any] = field(default_factory=dict)
    success_criteria: dict[str, Any] = field(default_factory=dict)
```

#### ResourceAllocation

```python
@dataclass
class ResourceAllocation:
    max_execution_time_ms: int = 300000  # 5 minutes
    max_retry_attempts: int = 3
    max_memory_mb: int = 512
    quality_threshold: float = 0.8
    timeout_ms: int = 30000
    concurrency_limit: int = 10
```

#### CoordinationProtocols

```python
@dataclass
class CoordinationProtocols:
    update_interval_ms: int = 5000
    heartbeat_interval_ms: int = 10000
    status_update_channel: str = "state"
    result_delivery_channel: str = "state"
    error_reporting_channel: str = "state"
    coordination_endpoint: Optional[str] = None
```

---

### Dependency Models

#### Dependency

```python
@dataclass
class Dependency:
    dependency_id: str
    dependency_type: DependencyType
    target: str
    timeout: int = 120
    max_retries: int = 3
    status: DependencyStatus = DependencyStatus.PENDING
    retry_count: int = 0
    resolved_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### DependencyType (Enum)

```python
class DependencyType(str, Enum):
    AGENT_COMPLETION = "agent_completion"
    RESOURCE_AVAILABILITY = "resource_availability"
    QUALITY_GATE = "quality_gate"
```

#### DependencyStatus (Enum)

```python
class DependencyStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    FAILED = "failed"
    TIMEOUT = "timeout"
```

#### DependencyResolutionResult

```python
@dataclass
class DependencyResolutionResult:
    coordination_id: str
    dependency_id: str
    success: bool
    duration_ms: float
    attempts: int
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

---

## Exceptions

### DependencyResolutionError

Raised when dependency resolution fails.

```python
class DependencyResolutionError(Exception):
    def __init__(
        self,
        coordination_id: str,
        dependency_id: str,
        error_message: str
    ):
        self.coordination_id = coordination_id
        self.dependency_id = dependency_id
        self.error_message = error_message
        super().__init__(f"Dependency '{dependency_id}' failed: {error_message}")
```

**Example**:
```python
try:
    await resolver.resolve_agent_dependencies(coordination_id, agent_context)
except DependencyResolutionError as e:
    print(f"Resolution failed for {e.dependency_id}: {e.error_message}")
```

---

### DependencyTimeoutError

Raised when dependency resolution times out.

```python
class DependencyTimeoutError(Exception):
    def __init__(
        self,
        coordination_id: str,
        dependency_id: str,
        timeout_seconds: int
    ):
        self.coordination_id = coordination_id
        self.dependency_id = dependency_id
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Dependency '{dependency_id}' timed out after {timeout_seconds}s")
```

**Example**:
```python
try:
    await resolver.resolve_dependency(coordination_id, dependency)
except DependencyTimeoutError as e:
    print(f"Timeout after {e.timeout_seconds}s for {e.dependency_id}")
```

---

## Usage Examples

### Example 1: Complete Coordination Workflow

```python
import asyncio
from omninode_bridge.agents.coordination import (
    ThreadSafeState,
    SignalCoordinator,
    ContextDistributor,
    DependencyResolver,
)
from omninode_bridge.agents.metrics import MetricsCollector

async def complete_workflow():
    # Setup
    state = ThreadSafeState()
    metrics = MetricsCollector()
    await metrics.start()

    signal_coordinator = SignalCoordinator(state=state, metrics_collector=metrics)
    context_distributor = ContextDistributor(state=state, metrics_collector=metrics)
    dependency_resolver = DependencyResolver(
        signal_coordinator=signal_coordinator,
        metrics_collector=metrics,
        state=state
    )

    # Define agents
    agent_assignments = {
        "model_gen": {
            "objective": "Generate models",
            "tasks": ["parse_contract", "generate_models"],
            "dependencies": []
        },
        "validator_gen": {
            "objective": "Generate validators",
            "tasks": ["generate_validators"],
            "dependencies": ["model_gen"]
        }
    }

    # Distribute context
    contexts = await context_distributor.distribute_agent_context(
        coordination_state={"coordination_id": "coord-123", "session_id": "session-456"},
        agent_assignments=agent_assignments
    )

    # Execute model_gen
    await signal_coordinator.signal_coordination_event(
        coordination_id="coord-123",
        event_type="agent_initialized",
        event_data={"agent_id": "model_gen", "ready": True}
    )

    await asyncio.sleep(0.5)  # Simulate work

    await signal_coordinator.signal_coordination_event(
        coordination_id="coord-123",
        event_type="agent_completed",
        event_data={"agent_id": "model_gen", "quality_score": 0.95}
    )

    # Execute validator_gen (after resolving dependency)
    validator_context = {
        "agent_id": "validator_gen",
        "dependencies": [
            {
                "dependency_id": "model_gen_complete",
                "type": "agent_completion",
                "target": "model_gen",
                "timeout": 120,
                "metadata": {"agent_id": "model_gen"}
            }
        ]
    }

    success = await dependency_resolver.resolve_agent_dependencies(
        coordination_id="coord-123",
        agent_context=validator_context
    )

    if success:
        await signal_coordinator.signal_coordination_event(
            coordination_id="coord-123",
            event_type="agent_initialized",
            event_data={"agent_id": "validator_gen", "ready": True}
        )

        await asyncio.sleep(0.5)  # Simulate work

        await signal_coordinator.signal_coordination_event(
            coordination_id="coord-123",
            event_type="agent_completed",
            event_data={"agent_id": "validator_gen", "quality_score": 0.92}
        )

    # Cleanup
    context_distributor.clear_coordination_contexts("coord-123")
    dependency_resolver.clear_coordination_dependencies("coord-123")

    await metrics.stop()

asyncio.run(complete_workflow())
```

---

## Performance Characteristics

| Component | Operation | Target | Actual | Status |
|-----------|-----------|--------|--------|--------|
| **SignalCoordinator** | Signal propagation | <100ms | 3ms | ✅ 97% faster |
| | Bulk operations (100 signals) | <1s | 310ms | ✅ 3x faster |
| **RoutingOrchestrator** | Routing decision | <5ms | 0.018ms | ✅ 424x faster |
| | Throughput | 100+ ops/sec | 56K ops/sec | ✅ 560x faster |
| **ContextDistributor** | Context distribution | <200ms/agent | 15ms/agent | ✅ 13x faster |
| | Context retrieval | <5ms | 0.5ms | ✅ 10x faster |
| **DependencyResolver** | Total resolution | <2s | <500ms | ✅ 4x faster |
| | Single dependency | <100ms | <50ms | ✅ 2x faster |

---

**Version**: 1.0
**Status**: ✅ Production-Ready
**Last Updated**: 2025-11-06
**Test Coverage**: 95%+
