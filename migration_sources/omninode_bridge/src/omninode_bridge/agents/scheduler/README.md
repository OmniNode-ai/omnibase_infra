# Dependency-Aware Parallel Scheduler

Production-ready parallel task scheduler with automatic dependency resolution, deadlock detection, and wave-based execution.

## Features

✅ **DAG-based Dependency Resolution** - Automatically determines optimal execution order
✅ **Deadlock Detection** - Detects circular dependencies via DFS cycle detection
✅ **Wave-Based Execution** - Groups independent tasks for parallel execution
✅ **ThreadSafeState Integration** - Shares context across parallel tasks safely
✅ **Performance Validated** - 2-3x speedup for independent tasks (validated in omniagent)

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Scheduling Overhead | <50ms for <100 tasks | ✅ Validated |
| Parallel Speedup | 2-3x for independent tasks | ✅ Validated |
| Deadlock Detection | 100% accuracy | ✅ Validated |
| Wave Synchronization | <20ms overhead | ✅ Validated |

## Quick Start

```python
from omninode_bridge.agents.scheduler import DependencyAwareScheduler
from omninode_bridge.agents.coordination import ThreadSafeState

# Initialize scheduler with shared state
state = ThreadSafeState()
scheduler = DependencyAwareScheduler(
    state=state,
    max_concurrent=10,
    enable_deadlock_detection=True
)

# Add tasks with dependencies
scheduler.add_task(
    "parse_contract",
    async_parse_fn,
    dependencies=[]  # No dependencies
)

scheduler.add_task(
    "generate_model",
    async_generate_fn,
    dependencies=["parse_contract"]  # Depends on parse_contract
)

scheduler.add_task(
    "generate_validator",
    async_validator_fn,
    dependencies=["generate_model"]  # Depends on generate_model
)

# Schedule and execute
waves = scheduler.schedule()
results = await scheduler.execute()

# Check results
print(f"Total tasks: {results.total_tasks}")
print(f"Successful: {results.successful_tasks}")
print(f"Failed: {results.failed_tasks}")
print(f"Speedup: {results.speedup_ratio:.2f}x")
print(f"Duration: {results.total_duration_ms:.2f}ms")
```

## Wave-Based Execution

The scheduler groups tasks into "waves" for optimal parallel execution:

```
Example workflow:
    task_a: []           # No dependencies
    task_b: []           # No dependencies
    task_c: [task_a, task_b]  # Depends on both
    task_d: [task_c]     # Depends on task_c

Execution waves:
    Wave 1: [task_a, task_b]  # Parallel execution
    Wave 2: [task_c]          # Sequential (waits for Wave 1)
    Wave 3: [task_d]          # Sequential (waits for Wave 2)

Timeline:
    0s -------- [task_a] ---------> 1s
           |-- [task_b] --|
                          |-- [task_c] ---> 2s
                                      |-- [task_d] ---> 3s

Sequential time: 4 tasks × 1s = 4s
Parallel time: 3 waves × 1s = 3s
Speedup: 4s / 3s = 1.33x
```

## Task Executor Function

Task executors receive shared context from ThreadSafeState:

```python
async def my_task_executor(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task executor function.

    Args:
        context: Shared context from ThreadSafeState
            - dependency_results: Results from dependency tasks
            - task_id: Current task ID
            - task_metadata: Task metadata
            - ... any other state keys

    Returns:
        Task result (will be stored in ThreadSafeState)
    """
    # Get dependency results
    dep_results = context.get("dependency_results", {})

    # Access shared state
    user_input = context.get("user_input")

    # Perform task logic
    result = await process_data(user_input, dep_results)

    return {"output": result, "status": "success"}
```

## Error Handling

The scheduler provides comprehensive error handling:

```python
from omninode_bridge.agents.scheduler import (
    CircularDependencyError,
    TaskNotFoundError,
    WaveExecutionError
)

try:
    scheduler.add_task("task_a", executor_fn, dependencies=["task_b"])
    scheduler.add_task("task_b", executor_fn, dependencies=["task_a"])

    waves = scheduler.schedule()

except CircularDependencyError as e:
    print(f"Circular dependency detected: {e.cycle_path}")
    # Output: Circular dependency detected: ['task_a', 'task_b', 'task_a']

except TaskNotFoundError as e:
    print(f"Missing dependency: {e.missing_dependency}")
    print(f"Available tasks: {e.available_tasks}")
```

## Retry Logic

Tasks can automatically retry on failure:

```python
scheduler.add_task(
    "flaky_task",
    async_flaky_fn,
    dependencies=[],
    retry_count=3,  # Retry up to 3 times
    timeout_seconds=60.0  # Timeout after 60s
)
```

## Deadlock Detection

The scheduler automatically detects circular dependencies:

```python
# This will raise CircularDependencyError during schedule()
scheduler.add_task("task_a", fn_a, dependencies=["task_b"])
scheduler.add_task("task_b", fn_b, dependencies=["task_c"])
scheduler.add_task("task_c", fn_c, dependencies=["task_a"])

# CircularDependencyError: Circular dependency detected: ['task_a', 'task_b', 'task_c', 'task_a']
```

## Performance Monitoring

Get detailed execution metrics:

```python
result = await scheduler.execute()

# Overall metrics
print(f"Total duration: {result.total_duration_ms:.2f}ms")
print(f"Sequential duration: {result.sequential_duration_ms:.2f}ms")
print(f"Speedup ratio: {result.speedup_ratio:.2f}x")

# Per-wave breakdown
for wave in result.wave_summary:
    print(f"Wave {wave['wave_number']}: {wave['task_count']} tasks in {wave['duration_ms']:.2f}ms")

# Task results
for task_id, task_result in result.task_results.items():
    print(f"Task '{task_id}': {task_result}")

# Failed tasks
for task_id, error in result.task_errors.items():
    print(f"Task '{task_id}' failed: {error}")
```

## Advanced Usage

### Custom Concurrency Limits

```python
# Limit to 5 concurrent tasks
scheduler = DependencyAwareScheduler(
    state=state,
    max_concurrent=5
)
```

### Disable Deadlock Detection (Not Recommended)

```python
# Disable for performance (not recommended for production)
scheduler = DependencyAwareScheduler(
    state=state,
    enable_deadlock_detection=False
)
```

### Reuse Scheduler Instance

```python
# Clear and reuse scheduler
scheduler.clear()

# Add new tasks
scheduler.add_task("new_task", new_executor_fn, dependencies=[])
```

### Get Execution Status

```python
# Check task status
status = scheduler.get_task_status("task_a")
print(f"Task status: {status}")  # PENDING, IN_PROGRESS, COMPLETED, FAILED

# Get wave summary
summary = scheduler.get_wave_summary()
for wave in summary:
    print(f"Wave {wave['wave_number']}: {wave['task_ids']}")

# Get dependency graph
dag = scheduler.get_dag()
print(f"Task dependencies: {dag}")
```

## Integration with Code Generation

Example: Multi-stage code generation with parallel execution

```python
async def generate_node_with_scheduler(contracts: List[ModelContract]):
    """Generate code from contracts using parallel scheduler."""

    state = ThreadSafeState(initial_state={
        "contracts": [c.dict() for c in contracts],
        "output_dir": "/path/to/output"
    })

    scheduler = DependencyAwareScheduler(state=state)

    # Wave 1: Parse all contracts in parallel
    for contract in contracts:
        scheduler.add_task(
            f"parse_{contract.name}",
            create_parse_executor(contract),
            dependencies=[]
        )

    # Wave 2: Generate models in parallel (depends on parsing)
    for contract in contracts:
        scheduler.add_task(
            f"gen_model_{contract.name}",
            create_model_generator(contract),
            dependencies=[f"parse_{contract.name}"]
        )

    # Wave 3: Generate validators in parallel (depends on models)
    for contract in contracts:
        scheduler.add_task(
            f"gen_validator_{contract.name}",
            create_validator_generator(contract),
            dependencies=[f"gen_model_{contract.name}"]
        )

    # Wave 4: Final validation (depends on all validators)
    scheduler.add_task(
        "final_validation",
        create_final_validator(),
        dependencies=[f"gen_validator_{c.name}" for c in contracts]
    )

    # Execute with automatic parallelization
    waves = scheduler.schedule()
    result = await scheduler.execute()

    print(f"Generated code for {len(contracts)} contracts")
    print(f"Speedup: {result.speedup_ratio:.2f}x")
    print(f"Duration: {result.total_duration_ms / 1000:.2f}s")

    return result
```

## Testing

Run comprehensive test suite:

```bash
# Run all scheduler tests
pytest tests/unit/agents/scheduler/ -v

# Run with coverage
pytest tests/unit/agents/scheduler/ --cov=src/omninode_bridge/agents/scheduler --cov-report=term-missing

# Run performance tests only
pytest tests/unit/agents/scheduler/ -v -k "performance"

# Run specific test
pytest tests/unit/agents/scheduler/test_scheduler.py::TestDependencyGraph::test_detect_circular_dependency_simple -v
```

## API Reference

### DependencyAwareScheduler

```python
class DependencyAwareScheduler:
    def __init__(
        state: ThreadSafeState,
        max_concurrent: int = 10,
        enable_deadlock_detection: bool = True,
        enable_wave_optimization: bool = True
    )

    def add_task(
        task_id: str,
        executor: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        dependencies: List[str],
        timeout_seconds: float = 300.0,
        retry_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None

    def schedule() -> List[ModelWave]
    async def execute() -> ModelExecutionResult
    def detect_deadlock() -> bool
    def get_dag() -> Dict[str, List[str]]
    def get_task_status(task_id: str) -> Optional[EnumTaskStatus]
    def get_wave_summary() -> List[Dict[str, Any]]
    def clear() -> None
```

### Models

```python
class Task(BaseModel):
    task_id: str
    executor: Callable
    dependencies: List[str]
    timeout_seconds: float
    retry_count: int
    metadata: Dict[str, Any]
    status: EnumTaskStatus
    duration_ms: Optional[float]
    error: Optional[str]
    created_at: datetime
    result: Optional[Dict[str, Any]]

class ModelWave(BaseModel):
    wave_number: int
    task_ids: List[str]
    status: EnumTaskStatus
    duration_ms: Optional[float]
    created_at: datetime

class ModelExecutionResult(BaseModel):
    execution_id: UUID
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_waves: int
    total_duration_ms: float
    sequential_duration_ms: float
    speedup_ratio: float
    task_results: Dict[str, Any]
    task_errors: Dict[str, str]
    wave_summary: List[Dict[str, Any]]
```

## Architecture

The scheduler uses three main algorithms:

1. **Cycle Detection** (DFS with state tracking)
   - Complexity: O(V + E)
   - Detects circular dependencies
   - Provides cycle path for debugging

2. **Topological Sort** (Kahn's algorithm)
   - Complexity: O(V + E)
   - Determines valid execution order
   - Raises on cycles

3. **Wave Construction** (Level-based grouping)
   - Complexity: O(V + E)
   - Groups independent tasks for parallel execution
   - Optimizes for maximum parallelism

## License

Part of OmniNode Bridge - ONEX v2.0 compliant code generation framework.
