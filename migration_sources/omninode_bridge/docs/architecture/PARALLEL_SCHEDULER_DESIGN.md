# Dependency-Aware Parallel Scheduler Architecture Design

**Version**: 1.0
**Status**: Design Phase
**Target Implementation**: Wave 3 (depends on ThreadSafeState from Wave 2)
**Expected Performance**: 2-3x speedup for multi-agent workflows
**Last Updated**: 2025-11-06

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Class Design](#class-design)
4. [Dependency Graph](#dependency-graph)
5. [Wave-Based Execution](#wave-based-execution)
6. [Deadlock Detection](#deadlock-detection)
7. [Error Handling](#error-handling)
8. [Performance Optimization](#performance-optimization)
9. [Integration Design](#integration-design)
10. [Testing Strategy](#testing-strategy)
11. [Implementation Plan](#implementation-plan)

---

## Executive Summary

### Purpose

Design a production-ready **dependency-aware parallel scheduler** that enables multi-agent code generation workflows to execute with 2-3x speedup while maintaining correctness through dependency resolution and deadlock detection.

### Key Features

✅ **Dependency Resolution** - DAG-based dependency graph with topological sort
✅ **Parallel Execution** - Wave-based scheduling executes independent tasks concurrently
✅ **Deadlock Detection** - Circular dependency prevention via cycle detection
✅ **Thread Safety** - Integration with ThreadSafeState for shared context
✅ **Error Recovery** - Graceful failure handling with retry strategies
✅ **Performance** - 2-3x speedup validated from omniagent benchmarks

### Design Principles

1. **Correctness First** - Dependencies always respected, no race conditions
2. **ONEX v2.0 Compliance** - Suffix-based naming, contract-driven, Pydantic models
3. **Observable** - Comprehensive metrics and logging at each stage
4. **Resilient** - Graceful degradation on partial failures
5. **Type Safe** - Strong typing with Pydantic v2 throughout

### Performance Targets

| Metric | Target | Validation Source |
|--------|--------|-------------------|
| **Parallel Speedup** | 2.25x-4.17x | omniagent benchmarks |
| **Scheduling Overhead** | <50ms | Research findings |
| **Dependency Resolution** | <10ms per wave | Research findings |
| **Independent Tasks** | 100% parallel | By design |
| **Wave Synchronization** | <20ms | asyncio.Barrier overhead |

### Research Validation

**Source**: `docs/research/OMNIAGENT_AGENT_FUNCTIONALITY_RESEARCH.md`

- **Pattern 1**: Dependency-Aware Parallel Scheduling (Lines 89-157)
- **Pattern 2**: Thread-Safe State Management (Lines 160-213)
- **Pattern 8**: Staged Parallel Execution (Lines 722-801)
- **Workflow 1**: Full Parallel Multi-Domain Workflow (Lines 1199-1295)

**Proven Results**:
- Independent parallel execution: **>2x speedup**
- Mixed parallel/dependency: **>1.5x speedup**
- Coordination overhead: **50-100ms per signal**

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DependencyAwareScheduler                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Public API                                                    │  │
│  │  • add_task(task, dependencies)                              │  │
│  │  • schedule() → List[List[Task]]  # Wave construction        │  │
│  │  • execute() → Dict[str, Any]     # Execute all waves        │  │
│  │  • detect_deadlock() → bool       # Circular dep check       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Core Components                                               │  │
│  │                                                               │  │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐│  │
│  │  │ DependencyGraph │  │ WaveConstructor  │  │ WaveExecutor ││  │
│  │  │                 │  │                  │  │              ││  │
│  │  │ • DAG storage   │  │ • Topological    │  │ • Parallel   ││  │
│  │  │ • Cycle detect  │  │   sort           │  │   execution  ││  │
│  │  │ • Validation    │  │ • Wave grouping  │  │ • Barrier    ││  │
│  │  │                 │  │ • Optimization   │  │   sync       ││  │
│  │  └────────┬────────┘  └────────┬─────────┘  └──────┬───────┘│  │
│  │           │                    │                    │        │  │
│  └───────────┼────────────────────┼────────────────────┼────────┘  │
│              │                    │                    │            │
│              └────────────────────┼────────────────────┘            │
│                                   │                                 │
│  ┌───────────────────────────────▼──────────────────────────────┐  │
│  │ Integration Layer                                            │  │
│  │                                                              │  │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │  │
│  │  │ ThreadSafeState  │  │ PerformanceMetrics│ │ AgentReg  │ │  │
│  │  │ (Shared Context) │  │ (Monitoring)      │ │ (Discovery)│ │  │
│  │  └──────────────────┘  └──────────────────┘  └───────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Execution Flow

```
1. TASK REGISTRATION
   │
   ├─► scheduler.add_task(task_a, dependencies=[])
   ├─► scheduler.add_task(task_b, dependencies=[])
   ├─► scheduler.add_task(task_c, dependencies=['task_a', 'task_b'])
   │
2. DEPENDENCY GRAPH CONSTRUCTION
   │
   ├─► Build DAG from task dependencies
   ├─► Validate all dependencies exist
   ├─► Detect circular dependencies (fail fast)
   │
3. WAVE CONSTRUCTION (Topological Sort)
   │
   ├─► Wave 1: [task_a, task_b]  # No dependencies, run in parallel
   ├─► Wave 2: [task_c]           # Depends on Wave 1, run after
   │
4. WAVE EXECUTION (Parallel within, Sequential between)
   │
   ├─► Wave 1 Execution:
   │   ├─► Launch task_a (asyncio.create_task)
   │   ├─► Launch task_b (asyncio.create_task)
   │   ├─► Barrier sync: await all tasks complete
   │   └─► Update ThreadSafeState with results
   │
   ├─► Wave 2 Execution:
   │   ├─► Launch task_c (can now access task_a, task_b results)
   │   ├─► Barrier sync: await completion
   │   └─► Update ThreadSafeState
   │
5. RESULT AGGREGATION
   │
   └─► Collect all task results
       ├─► Performance metrics (execution time, speedup ratio)
       ├─► Error summary (if any failures)
       └─► Return consolidated results
```

### Key Algorithms

**1. Topological Sort (Wave Construction)**
```python
def topological_sort(graph: Dict[str, List[str]]) -> List[List[str]]:
    """
    Group tasks into waves via topological sort.

    Algorithm: Kahn's algorithm with wave grouping
    Complexity: O(V + E) where V=tasks, E=dependencies
    """
    # Wave construction via level-based traversal
    # Tasks with same max_depth_from_root go in same wave
```

**2. Cycle Detection (Deadlock Prevention)**
```python
def detect_cycle(graph: Dict[str, List[str]]) -> bool:
    """
    Detect circular dependencies via DFS.

    Algorithm: DFS with state tracking (WHITE/GRAY/BLACK)
    Complexity: O(V + E)
    """
    # If GRAY node encountered during DFS → cycle exists
```

**3. Wave Optimization**
```python
def optimize_waves(waves: List[List[str]]) -> List[List[str]]:
    """
    Optimize wave construction for maximum parallelism.

    Strategies:
    - Independent task promotion (move tasks earlier if safe)
    - Wave merging (combine waves with no inter-dependencies)
    - Resource-aware scheduling (balance load across waves)
    """
```

---

## Class Design

### Core Classes

#### 1. DependencyAwareScheduler

**Primary scheduler class coordinating all scheduling operations.**

```python
from typing import Dict, List, Any, Callable, Awaitable, Optional
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio

from omnibase_core.models.enums.enum_task_status import EnumTaskStatus
from omninode_bridge.coordination.thread_safe_state import ThreadSafeState
from omninode_bridge.coordination.models.model_task import ModelTask
from omninode_bridge.coordination.models.model_wave import ModelWave
from omninode_bridge.coordination.models.model_execution_result import ModelExecutionResult


class DependencyAwareScheduler:
    """
    Dependency-aware parallel task scheduler with wave-based execution.

    Features:
    - DAG-based dependency resolution
    - Wave construction via topological sort
    - Parallel execution within waves
    - Deadlock detection via cycle detection
    - ThreadSafeState integration for shared context

    Performance:
    - Scheduling overhead: <50ms
    - Parallel speedup: 2-3x for independent tasks
    - Wave synchronization: <20ms per wave

    Example:
        >>> scheduler = DependencyAwareScheduler(state=shared_state)
        >>> scheduler.add_task("parse_contract", async_parse_fn, dependencies=[])
        >>> scheduler.add_task("generate_model", async_generate_fn, dependencies=["parse_contract"])
        >>> results = await scheduler.execute()
        >>> print(f"Speedup: {results['performance']['speedup_ratio']}x")
    """

    def __init__(
        self,
        state: ThreadSafeState,
        max_concurrent: int = 10,
        enable_deadlock_detection: bool = True,
        enable_wave_optimization: bool = True
    ):
        """
        Initialize scheduler with shared state and configuration.

        Args:
            state: ThreadSafeState for shared context across tasks
            max_concurrent: Maximum concurrent tasks (semaphore limit)
            enable_deadlock_detection: Enable circular dependency detection
            enable_wave_optimization: Enable wave construction optimization
        """
        self.state = state
        self.max_concurrent = max_concurrent
        self.enable_deadlock_detection = enable_deadlock_detection
        self.enable_wave_optimization = enable_wave_optimization

        # Task registry
        self._tasks: Dict[str, ModelTask] = {}

        # Dependency graph (adjacency list)
        self._dependency_graph: Dict[str, List[str]] = {}

        # Execution tracking
        self._execution_id: UUID | None = None
        self._waves: List[ModelWave] = []
        self._start_time: float | None = None

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def add_task(
        self,
        task_id: str,
        executor: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        dependencies: List[str],
        timeout_seconds: float = 300.0,
        retry_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add task to scheduler with dependencies.

        Args:
            task_id: Unique task identifier
            executor: Async function to execute (receives shared state)
            dependencies: List of task_ids that must complete first
            timeout_seconds: Task execution timeout
            retry_count: Number of retries on failure
            metadata: Optional task metadata

        Raises:
            ValueError: If task_id already exists
            ValueError: If dependency references non-existent task
        """
        if task_id in self._tasks:
            raise ValueError(f"Task '{task_id}' already exists")

        # Validate dependencies (deferred validation if task hasn't been added yet)
        # Will be fully validated in schedule()

        task = ModelTask(
            task_id=task_id,
            executor=executor,
            dependencies=dependencies,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            metadata=metadata or {},
            status=EnumTaskStatus.PENDING,
            created_at=datetime.utcnow()
        )

        self._tasks[task_id] = task
        self._dependency_graph[task_id] = dependencies.copy()

    def schedule(self) -> List[ModelWave]:
        """
        Construct execution waves via topological sort.

        Returns:
            List of waves (each wave is list of task_ids that can run in parallel)

        Raises:
            ValueError: If task references non-existent dependency
            ValueError: If circular dependency detected (deadlock)

        Algorithm:
            1. Validate dependency graph
            2. Detect circular dependencies (if enabled)
            3. Perform topological sort with wave grouping
            4. Optimize waves (if enabled)
            5. Return wave schedule

        Performance:
            - Complexity: O(V + E) where V=tasks, E=dependencies
            - Typical: <10ms for <100 tasks
        """
        # Step 1: Validate dependency graph
        self._validate_dependency_graph()

        # Step 2: Detect circular dependencies
        if self.enable_deadlock_detection:
            if self.detect_deadlock():
                raise ValueError("Circular dependency detected (deadlock)")

        # Step 3: Topological sort with wave grouping
        waves = self._topological_sort_with_waves()

        # Step 4: Optimize waves
        if self.enable_wave_optimization:
            waves = self._optimize_waves(waves)

        # Step 5: Create Wave models
        self._waves = [
            ModelWave(
                wave_number=i + 1,
                task_ids=wave_task_ids,
                status=EnumTaskStatus.PENDING,
                created_at=datetime.utcnow()
            )
            for i, wave_task_ids in enumerate(waves)
        ]

        return self._waves

    async def execute(self) -> ModelExecutionResult:
        """
        Execute all waves with parallel execution within each wave.

        Returns:
            ModelExecutionResult with task results and performance metrics

        Raises:
            ValueError: If schedule() not called first

        Execution Flow:
            For each wave:
                1. Launch all tasks in wave concurrently (asyncio.create_task)
                2. Wait for wave completion (asyncio.gather or barrier)
                3. Update ThreadSafeState with results
                4. Track errors and retries

        Performance:
            - Independent tasks: 100% parallel (within wave)
            - Wave synchronization: <20ms overhead
            - Target speedup: 2-3x over sequential
        """
        if not self._waves:
            raise ValueError("Must call schedule() before execute()")

        self._execution_id = uuid4()
        self._start_time = asyncio.get_event_loop().time()

        # Track results
        task_results: Dict[str, Any] = {}
        task_errors: Dict[str, Exception] = {}

        # Execute waves sequentially (parallel within each wave)
        for wave in self._waves:
            wave_start = asyncio.get_event_loop().time()
            wave.status = EnumTaskStatus.IN_PROGRESS

            # Launch all tasks in wave concurrently
            wave_tasks = {}
            for task_id in wave.task_ids:
                task = self._tasks[task_id]
                task.status = EnumTaskStatus.IN_PROGRESS

                # Create task with semaphore for concurrency control
                wave_tasks[task_id] = asyncio.create_task(
                    self._execute_task_with_semaphore(task, task_results, task_errors)
                )

            # Wait for wave completion
            wave_results = await asyncio.gather(*wave_tasks.values(), return_exceptions=True)

            # Process wave results
            for task_id, result in zip(wave_tasks.keys(), wave_results):
                if isinstance(result, Exception):
                    task_errors[task_id] = result
                    self._tasks[task_id].status = EnumTaskStatus.FAILED
                    self._tasks[task_id].error = str(result)
                else:
                    task_results[task_id] = result
                    self._tasks[task_id].status = EnumTaskStatus.COMPLETED

            wave.status = EnumTaskStatus.COMPLETED
            wave.duration_ms = (asyncio.get_event_loop().time() - wave_start) * 1000

        # Calculate performance metrics
        total_duration = asyncio.get_event_loop().time() - self._start_time
        sequential_time = sum(
            self._tasks[task_id].duration_ms / 1000
            for task_id in task_results.keys()
        )
        speedup_ratio = sequential_time / total_duration if total_duration > 0 else 1.0

        return ModelExecutionResult(
            execution_id=self._execution_id,
            total_tasks=len(self._tasks),
            successful_tasks=len(task_results),
            failed_tasks=len(task_errors),
            total_waves=len(self._waves),
            total_duration_ms=total_duration * 1000,
            sequential_duration_ms=sequential_time * 1000,
            speedup_ratio=speedup_ratio,
            task_results=task_results,
            task_errors={k: str(v) for k, v in task_errors.items()},
            wave_summary=[
                {
                    "wave_number": wave.wave_number,
                    "task_count": len(wave.task_ids),
                    "duration_ms": wave.duration_ms,
                    "status": wave.status.value
                }
                for wave in self._waves
            ]
        )

    def detect_deadlock(self) -> bool:
        """
        Detect circular dependencies via DFS cycle detection.

        Returns:
            True if circular dependency exists, False otherwise

        Algorithm:
            DFS with state tracking (WHITE/GRAY/BLACK)
            - WHITE: Unvisited
            - GRAY: Currently in DFS stack (visiting)
            - BLACK: Fully processed

            If GRAY node encountered → cycle exists

        Complexity: O(V + E)
        """
        # State tracking for DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        node_state = {task_id: WHITE for task_id in self._tasks.keys()}

        def dfs_visit(node: str) -> bool:
            """Visit node and detect cycles."""
            if node_state[node] == GRAY:
                # Currently visiting → cycle detected
                return True

            if node_state[node] == BLACK:
                # Already processed → no cycle from this node
                return False

            # Mark as currently visiting
            node_state[node] = GRAY

            # Visit dependencies
            for dep in self._dependency_graph.get(node, []):
                if dfs_visit(dep):
                    return True

            # Mark as fully processed
            node_state[node] = BLACK
            return False

        # Check all nodes
        for task_id in self._tasks.keys():
            if node_state[task_id] == WHITE:
                if dfs_visit(task_id):
                    return True

        return False

    # Private helper methods

    def _validate_dependency_graph(self) -> None:
        """Validate all dependencies reference existing tasks."""
        all_task_ids = set(self._tasks.keys())

        for task_id, dependencies in self._dependency_graph.items():
            for dep in dependencies:
                if dep not in all_task_ids:
                    raise ValueError(
                        f"Task '{task_id}' depends on non-existent task '{dep}'"
                    )

    def _topological_sort_with_waves(self) -> List[List[str]]:
        """
        Perform topological sort with wave grouping via Kahn's algorithm.

        Returns:
            List of waves (each wave is list of task_ids)
        """
        # Calculate in-degree for each node (number of dependencies each task has)
        in_degree = {task_id: 0 for task_id in self._tasks.keys()}
        for task_id, dependencies in self._dependency_graph.items():
            in_degree[task_id] = len(dependencies)

        # Initialize queue with nodes having in-degree 0
        current_wave = [
            task_id for task_id, degree in in_degree.items() if degree == 0
        ]

        waves = []

        while current_wave:
            waves.append(current_wave)
            next_wave = []

            # Process current wave
            for task_id in current_wave:
                # Reduce in-degree of dependents
                for dependent_id, dependencies in self._dependency_graph.items():
                    if task_id in dependencies:
                        in_degree[dependent_id] -= 1

                        # If all dependencies satisfied, add to next wave
                        if in_degree[dependent_id] == 0:
                            next_wave.append(dependent_id)

            current_wave = next_wave

        return waves

    def _optimize_waves(self, waves: List[List[str]]) -> List[List[str]]:
        """
        Optimize wave construction for maximum parallelism.

        Strategies:
        - Independent task promotion (move tasks earlier if safe)
        - Wave merging (combine waves with no inter-dependencies)
        """
        # Future optimization: Implement advanced wave optimization
        # For MVP: Return waves as-is
        return waves

    async def _execute_task_with_semaphore(
        self,
        task: ModelTask,
        results: Dict[str, Any],
        errors: Dict[str, Exception]
    ) -> Dict[str, Any]:
        """Execute task with semaphore for concurrency control."""
        async with self._semaphore:
            task_start = asyncio.get_event_loop().time()

            try:
                # Get shared state snapshot for task
                shared_context = self.state.get_snapshot()

                # Add dependency results to context
                dep_results = {
                    dep_id: results[dep_id]
                    for dep_id in task.dependencies
                    if dep_id in results
                }
                shared_context['dependency_results'] = dep_results

                # Execute task with timeout
                result = await asyncio.wait_for(
                    task.executor(shared_context),
                    timeout=task.timeout_seconds
                )

                # Update ThreadSafeState with result
                self.state.set(
                    f"task_result_{task.task_id}",
                    result,
                    step_id=task.task_id
                )

                task.duration_ms = (asyncio.get_event_loop().time() - task_start) * 1000
                return result

            except Exception as e:
                task.duration_ms = (asyncio.get_event_loop().time() - task_start) * 1000

                # Retry logic
                if task.retry_count > 0:
                    task.retry_count -= 1
                    return await self._execute_task_with_semaphore(task, results, errors)

                raise e
```

#### 2. ModelTask

**Task representation with metadata and execution details.**

```python
from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Dict, Any, List, Optional
from datetime import datetime
from omnibase_core.models.enums.enum_task_status import EnumTaskStatus


class ModelTask(BaseModel):
    """
    Task representation for scheduler.

    Attributes:
        task_id: Unique task identifier
        executor: Async function to execute
        dependencies: List of task_ids that must complete first
        timeout_seconds: Task execution timeout
        retry_count: Number of retries remaining
        metadata: Optional task metadata
        status: Current task status
        duration_ms: Execution duration (populated after execution)
        error: Error message if failed
        created_at: Task creation timestamp
    """
    task_id: str = Field(..., description="Unique task identifier")
    executor: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]] = Field(
        ...,
        description="Async function to execute",
        exclude=True  # Don't serialize executor function
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Task IDs that must complete first"
    )
    timeout_seconds: float = Field(
        default=300.0,
        description="Task execution timeout"
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries remaining"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional task metadata"
    )
    status: EnumTaskStatus = Field(
        default=EnumTaskStatus.PENDING,
        description="Current task status"
    )
    duration_ms: Optional[float] = Field(
        default=None,
        description="Execution duration in milliseconds"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Task creation timestamp"
    )

    class Config:
        arbitrary_types_allowed = True  # Allow Callable types
```

#### 3. ModelWave

**Wave representation for grouped parallel tasks.**

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from omnibase_core.models.enums.enum_task_status import EnumTaskStatus


class ModelWave(BaseModel):
    """
    Wave representation for parallel task execution.

    A wave is a group of tasks that can execute in parallel
    because they have no inter-dependencies.

    Attributes:
        wave_number: Sequential wave number (1, 2, 3...)
        task_ids: List of task IDs in this wave
        status: Current wave status
        duration_ms: Wave execution duration
        created_at: Wave creation timestamp
    """
    wave_number: int = Field(..., description="Sequential wave number")
    task_ids: List[str] = Field(..., description="Task IDs in this wave")
    status: EnumTaskStatus = Field(
        default=EnumTaskStatus.PENDING,
        description="Current wave status"
    )
    duration_ms: Optional[float] = Field(
        default=None,
        description="Wave execution duration in milliseconds"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Wave creation timestamp"
    )
```

#### 4. ModelExecutionResult

**Execution result with performance metrics.**

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from uuid import UUID


class ModelExecutionResult(BaseModel):
    """
    Execution result with performance metrics.

    Attributes:
        execution_id: Unique execution identifier
        total_tasks: Total number of tasks executed
        successful_tasks: Number of tasks that completed successfully
        failed_tasks: Number of tasks that failed
        total_waves: Total number of waves executed
        total_duration_ms: Total execution time (wall clock)
        sequential_duration_ms: Estimated sequential execution time
        speedup_ratio: Parallel speedup (sequential_time / total_time)
        task_results: Dictionary of task_id → result
        task_errors: Dictionary of task_id → error message
        wave_summary: Summary of each wave execution
    """
    execution_id: UUID = Field(..., description="Unique execution identifier")
    total_tasks: int = Field(..., description="Total number of tasks")
    successful_tasks: int = Field(..., description="Successful task count")
    failed_tasks: int = Field(..., description="Failed task count")
    total_waves: int = Field(..., description="Total number of waves")
    total_duration_ms: float = Field(..., description="Total execution time (ms)")
    sequential_duration_ms: float = Field(
        ...,
        description="Estimated sequential execution time (ms)"
    )
    speedup_ratio: float = Field(
        ...,
        description="Parallel speedup ratio (sequential / parallel)"
    )
    task_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task ID to result mapping"
    )
    task_errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Task ID to error message mapping"
    )
    wave_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summary of each wave execution"
    )
```

#### 5. EnumTaskStatus

**Task/Wave status enumeration.**

```python
from enum import Enum


class EnumTaskStatus(str, Enum):
    """
    Task and wave execution status.

    States:
    - PENDING: Not yet started
    - IN_PROGRESS: Currently executing
    - COMPLETED: Successfully completed
    - FAILED: Failed with error
    - CANCELLED: Cancelled before completion
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

---

## Dependency Graph

### Graph Representation

**Data Structure**: Adjacency list (Dict[str, List[str]])

```python
# Example dependency graph
dependency_graph = {
    "parse_contract": [],                      # No dependencies
    "generate_model": ["parse_contract"],      # Depends on parse_contract
    "generate_validator": ["generate_model"],  # Depends on generate_model
    "generate_tests": ["generate_validator"],  # Depends on generate_validator
    "final_validation": ["generate_model", "generate_validator", "generate_tests"]
}
```

### Validation

**Validation Rules**:

1. **Existence Check**: All referenced dependencies must exist as tasks
2. **Acyclicity**: No circular dependencies (enforced by cycle detection)
3. **Reachability**: All tasks must be reachable from root nodes

```python
def _validate_dependency_graph(self) -> None:
    """
    Validate dependency graph for correctness.

    Checks:
    1. All dependencies reference existing tasks
    2. No self-dependencies
    3. No cycles (if deadlock detection enabled)

    Raises:
        ValueError: If validation fails
    """
    all_task_ids = set(self._tasks.keys())

    for task_id, dependencies in self._dependency_graph.items():
        # Check 1: All dependencies exist
        for dep in dependencies:
            if dep not in all_task_ids:
                raise ValueError(
                    f"Task '{task_id}' depends on non-existent task '{dep}'"
                )

        # Check 2: No self-dependencies
        if task_id in dependencies:
            raise ValueError(f"Task '{task_id}' cannot depend on itself")
```

### Topological Sort Algorithm

**Algorithm**: Modified Kahn's algorithm with wave grouping

**Complexity**: O(V + E) where V = number of tasks, E = number of dependencies

```python
def _topological_sort_with_waves(self) -> List[List[str]]:
    """
    Perform topological sort with wave grouping.

    Algorithm (Kahn's algorithm):
    1. Calculate in-degree for each node (number of incoming edges)
    2. Initialize queue with nodes having in-degree 0 (no dependencies)
    3. While queue not empty:
       a. Process all nodes in queue as current wave
       b. For each node, reduce in-degree of dependents
       c. Add dependents with in-degree 0 to next wave
    4. Return list of waves

    Returns:
        List of waves where each wave is a list of task_ids

    Example:
        Input:
            task_a: []
            task_b: []
            task_c: [task_a, task_b]
            task_d: [task_c]

        Output:
            [
                ["task_a", "task_b"],  # Wave 1: Independent
                ["task_c"],            # Wave 2: Depends on Wave 1
                ["task_d"]             # Wave 3: Depends on Wave 2
            ]
    """
    # Step 1: Calculate in-degree
    in_degree = {task_id: 0 for task_id in self._tasks.keys()}

    for task_id, dependencies in self._dependency_graph.items():
        for dep in dependencies:
            in_degree[task_id] += 1

    # Step 2: Initialize first wave (tasks with no dependencies)
    current_wave = [
        task_id for task_id, degree in in_degree.items() if degree == 0
    ]

    waves = []

    # Step 3: Process waves
    while current_wave:
        waves.append(current_wave.copy())
        next_wave = []

        # Process each task in current wave
        for task_id in current_wave:
            # Reduce in-degree of dependents
            for dependent_id, dependencies in self._dependency_graph.items():
                if task_id in dependencies:
                    in_degree[dependent_id] -= 1

                    # If all dependencies satisfied, add to next wave
                    if in_degree[dependent_id] == 0:
                        next_wave.append(dependent_id)

        current_wave = next_wave

    return waves
```

### Graph Visualization

**ASCII Representation**:

```
Example workflow:

    ┌──────────────┐  ┌──────────────┐
    │ parse_contract│  │load_templates│
    └──────┬───────┘  └──────┬───────┘
           │                 │
           └────────┬────────┘
                    ▼
           ┌────────────────┐
           │ generate_model │
           └────────┬───────┘
                    │
           ┌────────┴────────┐
           ▼                 ▼
    ┌──────────────┐  ┌──────────────┐
    │gen_validator │  │ gen_tests    │
    └──────┬───────┘  └──────┬───────┘
           │                 │
           └────────┬────────┘
                    ▼
           ┌────────────────┐
           │final_validation│
           └────────────────┘

Wave 1: [parse_contract, load_templates]  # Parallel (2 tasks)
Wave 2: [generate_model]                  # Sequential (depends on Wave 1)
Wave 3: [gen_validator, gen_tests]        # Parallel (both depend on Wave 2)
Wave 4: [final_validation]                # Sequential (depends on Wave 3)

Speedup calculation:
- Sequential time: 5 tasks × 1s = 5s
- Parallel time: 4 waves × 1s = 4s
- Speedup: 5s / 4s = 1.25x
```

---

## Wave-Based Execution

### Wave Construction

**Goal**: Group tasks into waves such that:
1. All tasks in a wave can execute in parallel (no inter-dependencies)
2. Waves execute sequentially (dependencies between waves)
3. Maximum parallelism achieved within each wave

**Construction Algorithm**: Topological sort with level-based grouping

```python
# Pseudo-code for wave construction
waves = []
remaining_tasks = set(all_task_ids)

while remaining_tasks:
    # Find tasks with all dependencies satisfied
    ready_tasks = []
    for task_id in remaining_tasks:
        if all(dep in completed_tasks for dep in dependencies[task_id]):
            ready_tasks.append(task_id)

    # Create wave from ready tasks
    waves.append(ready_tasks)

    # Mark as completed and remove from remaining
    completed_tasks.update(ready_tasks)
    remaining_tasks.difference_update(ready_tasks)
```

### Parallel Execution within Wave

**Execution Pattern**: Launch all tasks concurrently, wait for wave completion

```python
async def _execute_wave(self, wave: ModelWave) -> Dict[str, Any]:
    """
    Execute all tasks in wave in parallel.

    Pattern:
    1. Launch all tasks concurrently (asyncio.create_task)
    2. Control concurrency via semaphore
    3. Wait for all tasks to complete (asyncio.gather)
    4. Update shared state with results
    5. Track errors and partial failures

    Returns:
        Dictionary of task_id → result for wave
    """
    wave_tasks = {}

    # Launch all tasks in wave
    for task_id in wave.task_ids:
        task = self._tasks[task_id]
        wave_tasks[task_id] = asyncio.create_task(
            self._execute_task_with_semaphore(task)
        )

    # Wait for all tasks to complete
    results = await asyncio.gather(
        *wave_tasks.values(),
        return_exceptions=True  # Capture exceptions without failing
    )

    # Process results
    wave_results = {}
    wave_errors = {}

    for task_id, result in zip(wave_tasks.keys(), results):
        if isinstance(result, Exception):
            wave_errors[task_id] = result
        else:
            wave_results[task_id] = result

    return wave_results, wave_errors
```

### Barrier Synchronization

**Purpose**: Ensure all tasks in wave complete before starting next wave

**Implementation**: asyncio.gather with return_exceptions=True

```python
# Barrier synchronization via asyncio.gather
results = await asyncio.gather(
    *[task_coroutine for task_coroutine in wave_tasks],
    return_exceptions=True  # Don't fail on single task error
)

# All tasks in wave are now complete
# Safe to proceed to next wave
```

### Wave Optimization Strategies

**1. Independent Task Promotion**

Move tasks to earlier waves if dependencies allow:

```python
# Before optimization:
# Wave 1: [task_a]
# Wave 2: [task_b]  # Actually independent of task_a
# Wave 3: [task_c]  # Depends on task_b

# After optimization:
# Wave 1: [task_a, task_b]  # Parallel execution
# Wave 2: [task_c]
```

**2. Wave Merging**

Merge consecutive waves with no inter-dependencies:

```python
# Before optimization:
# Wave 1: [task_a]
# Wave 2: [task_b]  # Independent of Wave 1
# Wave 3: [task_c]  # Depends on Wave 2

# After optimization (if safe):
# Wave 1: [task_a, task_b]  # Merged if truly independent
# Wave 2: [task_c]
```

**3. Resource-Aware Scheduling**

Balance load across waves based on estimated execution time:

```python
# If task_a takes 10s and task_b takes 1s
# Might be better to schedule task_b in next wave
# to avoid blocking other tasks
```

---

## Deadlock Detection

### Circular Dependency Detection

**Problem**: Circular dependencies create deadlock (tasks wait forever)

**Example**:
```python
task_a depends on task_b
task_b depends on task_c
task_c depends on task_a  # Cycle! Deadlock!
```

**Solution**: DFS-based cycle detection before execution

### Detection Algorithm

**Algorithm**: Depth-First Search with state tracking

**States**:
- **WHITE (0)**: Unvisited node
- **GRAY (1)**: Currently in DFS stack (being visited)
- **BLACK (2)**: Fully processed (visit complete)

**Detection Rule**: If we encounter a GRAY node during DFS → cycle exists

```python
def detect_deadlock(self) -> bool:
    """
    Detect circular dependencies via DFS cycle detection.

    Algorithm:
    1. Initialize all nodes as WHITE (unvisited)
    2. For each unvisited node:
       a. Perform DFS from that node
       b. Mark node as GRAY when visiting
       c. If GRAY node encountered → cycle detected
       d. Mark node as BLACK when done visiting
    3. Return True if cycle found, False otherwise

    Complexity: O(V + E)

    Example:
        task_a → task_b → task_c → task_a  # Cycle!

        DFS trace:
        1. Visit task_a (mark GRAY)
        2. Visit task_b (mark GRAY)
        3. Visit task_c (mark GRAY)
        4. Visit task_a again → Already GRAY! → CYCLE DETECTED
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    node_state = {task_id: WHITE for task_id in self._tasks.keys()}

    def dfs_visit(node: str) -> bool:
        """Visit node and detect cycles."""
        if node_state[node] == GRAY:
            # Currently visiting → cycle detected
            return True

        if node_state[node] == BLACK:
            # Already processed → no cycle from this node
            return False

        # Mark as currently visiting
        node_state[node] = GRAY

        # Visit all dependencies
        for dep in self._dependency_graph.get(node, []):
            if dfs_visit(dep):
                return True

        # Mark as fully processed
        node_state[node] = BLACK
        return False

    # Check all nodes (handles disconnected components)
    for task_id in self._tasks.keys():
        if node_state[task_id] == WHITE:
            if dfs_visit(task_id):
                return True

    return False
```

### Deadlock Prevention Strategies

**1. Fail Fast (MVP)**

Detect cycles during `schedule()` and raise error:

```python
if self.enable_deadlock_detection:
    if self.detect_deadlock():
        raise ValueError("Circular dependency detected (deadlock)")
```

**2. Cycle Breaking (Future)**

Automatically break cycles by removing minimum edges:

```python
def break_cycle(self, cycle: List[str]) -> None:
    """
    Break cycle by removing edge with lowest priority.

    Strategy:
    - Find edge in cycle with lowest priority weight
    - Remove that edge from dependency graph
    - Retry topological sort
    """
    # Future implementation
```

**3. Timeout Detection (Future)**

Detect deadlock via timeout during execution:

```python
# If wave takes >timeout → possible deadlock
# Investigate which tasks are blocked
```

### Reporting Circular Dependencies

**Enhanced Error Messages**:

```python
def _find_cycle_path(self) -> List[str]:
    """
    Find and return the circular dependency path.

    Returns:
        List of task_ids forming the cycle

    Example:
        ["task_a", "task_b", "task_c", "task_a"]
    """
    # Implementation using DFS to track path
    # Future enhancement for better error messages
```

**Error Message**:
```
ValueError: Circular dependency detected (deadlock)
Cycle path: task_a → task_b → task_c → task_a

Please review task dependencies and break the cycle.
```

---

## Error Handling

### Failure Modes

**1. Task Execution Failure**

Individual task fails during execution:

```python
# Failure modes:
# - Exception raised in executor
# - Timeout exceeded
# - Resource unavailable
```

**Handling**:
```python
try:
    result = await task.executor(shared_context)
except asyncio.TimeoutError:
    # Timeout exceeded
    if task.retry_count > 0:
        # Retry with exponential backoff
        await asyncio.sleep(2 ** (initial_retries - task.retry_count))
        task.retry_count -= 1
        return await self._execute_task_with_semaphore(task)
    else:
        # Max retries exceeded
        raise
except Exception as e:
    # Other exception
    if task.retry_count > 0:
        task.retry_count -= 1
        return await self._execute_task_with_semaphore(task)
    else:
        raise
```

**2. Partial Wave Failure**

Some tasks in wave succeed, others fail:

```python
# Wave execution with return_exceptions=True
results = await asyncio.gather(
    *wave_tasks,
    return_exceptions=True  # Capture exceptions without propagating
)

# Separate successes from failures
successes = {k: v for k, v in zip(task_ids, results) if not isinstance(v, Exception)}
failures = {k: v for k, v in zip(task_ids, results) if isinstance(v, Exception)}
```

**Handling**:
```python
# Strategy: Continue with successful tasks, track failures
# - Store successful results in ThreadSafeState
# - Log failures for investigation
# - Dependent tasks may skip or use partial results
```

**3. Complete Wave Failure**

All tasks in wave fail:

```python
if len(wave_failures) == len(wave.task_ids):
    # All tasks failed
    # Option 1: Abort remaining waves
    # Option 2: Continue with degraded results
    # Option 3: Retry entire wave
```

### Recovery Strategies

**1. Retry with Exponential Backoff**

```python
async def _execute_task_with_retry(
    self,
    task: ModelTask,
    initial_retries: int
) -> Dict[str, Any]:
    """
    Execute task with retry and exponential backoff.

    Backoff strategy:
    - Retry 1: Wait 2^0 = 1 second
    - Retry 2: Wait 2^1 = 2 seconds
    - Retry 3: Wait 2^2 = 4 seconds
    """
    retries_remaining = initial_retries

    while retries_remaining >= 0:
        try:
            return await task.executor(shared_context)
        except Exception as e:
            if retries_remaining == 0:
                raise

            # Exponential backoff
            wait_seconds = 2 ** (initial_retries - retries_remaining)
            await asyncio.sleep(wait_seconds)
            retries_remaining -= 1
```

**2. Graceful Degradation**

```python
# Continue execution with partial results
if wave_failures:
    logger.warning(
        f"Wave {wave.wave_number} had {len(wave_failures)} failures, "
        f"continuing with {len(wave_successes)} successful tasks"
    )

    # Mark failed tasks in ThreadSafeState
    for task_id in wave_failures:
        self.state.set(f"task_failed_{task_id}", True)

    # Dependent tasks can check for failures
    # and skip or use alternative logic
```

**3. Compensating Actions (Saga Pattern)**

```python
# Future: Implement saga pattern for transactional workflows
class SagaScheduler(DependencyAwareScheduler):
    """
    Scheduler with saga pattern for compensating actions.

    If task fails:
    1. Execute compensating action for that task
    2. Execute compensating actions for all completed tasks in reverse
    3. Restore system to consistent state
    """
    # Future implementation
```

### Error Propagation

**Dependency Error Propagation**:

```python
# If task_a fails and task_b depends on task_a:
# Option 1: Skip task_b (no input available)
# Option 2: Execute task_b with partial/empty input
# Option 3: Fail task_b immediately

if any(dep_id in task_errors for dep_id in task.dependencies):
    # Dependency failed
    if task.metadata.get('allow_partial_dependencies', False):
        # Execute with available results
        pass
    else:
        # Skip this task
        task.status = EnumTaskStatus.CANCELLED
        task.error = "Dependency failure"
        continue
```

### Error Reporting

**Comprehensive Error Summary**:

```python
class ModelExecutionResult:
    """Execution result with detailed error information."""

    task_errors: Dict[str, str]  # task_id → error message
    wave_failures: List[Dict[str, Any]]  # Per-wave failure summary

    def get_error_summary(self) -> str:
        """
        Generate human-readable error summary.

        Returns:
            Formatted error report
        """
        if not self.task_errors:
            return "No errors"

        summary = f"Execution failed with {len(self.task_errors)} task errors:\n\n"

        for task_id, error in self.task_errors.items():
            summary += f"❌ {task_id}: {error}\n"

        return summary
```

---

## Performance Optimization

### Target Performance Metrics

| Metric | Target | Validation |
|--------|--------|------------|
| **Scheduling Overhead** | <50ms | Topological sort + validation |
| **Dependency Resolution** | <10ms per wave | O(V + E) complexity |
| **Wave Synchronization** | <20ms | asyncio.gather overhead |
| **Parallel Speedup** | 2.25x-4.17x | Independent tasks |
| **Memory Overhead** | <100MB | Task metadata storage |

### Optimization Techniques

**1. Minimize Scheduling Overhead**

```python
# Strategy: Pre-compute dependency graph, reuse for multiple executions
class DependencyAwareScheduler:
    def __init__(self):
        self._dependency_graph_cached = False
        self._waves_cached: List[ModelWave] = []

    def schedule(self) -> List[ModelWave]:
        """Compute waves once, reuse for multiple executions."""
        if self._dependency_graph_cached and self._waves_cached:
            return self._waves_cached

        # Compute waves
        waves = self._topological_sort_with_waves()

        # Cache for reuse
        self._dependency_graph_cached = True
        self._waves_cached = waves

        return waves
```

**2. Efficient Task Queuing**

```python
# Strategy: Use asyncio.Queue with priority for task scheduling
import asyncio
from queue import PriorityQueue

class PriorityScheduler(DependencyAwareScheduler):
    """Scheduler with priority-based task queuing."""

    def __init__(self):
        super().__init__()
        self._task_queue = asyncio.PriorityQueue()

    async def _queue_wave_tasks(self, wave: ModelWave):
        """Queue wave tasks with priority."""
        for task_id in wave.task_ids:
            task = self._tasks[task_id]
            priority = task.metadata.get('priority', 10)
            await self._task_queue.put((priority, task_id, task))
```

**3. Thread Pool for CPU-Intensive Tasks**

```python
# Strategy: Offload CPU-intensive tasks to thread pool
from concurrent.futures import ThreadPoolExecutor
import asyncio

class HybridScheduler(DependencyAwareScheduler):
    """Scheduler with thread pool for CPU-intensive tasks."""

    def __init__(self, max_cpu_workers: int = 4):
        super().__init__()
        self._cpu_executor = ThreadPoolExecutor(max_workers=max_cpu_workers)

    async def _execute_task_hybrid(self, task: ModelTask):
        """Execute task with hybrid async/thread pool approach."""
        if task.metadata.get('cpu_intensive', False):
            # Run in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._cpu_executor,
                self._sync_execute_task,
                task
            )
        else:
            # Run async
            result = await task.executor(shared_context)

        return result
```

**4. Resource Allocation**

```python
# Strategy: Allocate resources based on task requirements
class ResourceAwareScheduler(DependencyAwareScheduler):
    """Scheduler with resource-aware allocation."""

    def __init__(self, max_memory_mb: int = 4096, max_cpu_cores: int = 8):
        super().__init__()
        self.max_memory_mb = max_memory_mb
        self.max_cpu_cores = max_cpu_cores
        self._current_memory_usage = 0
        self._current_cpu_usage = 0

    async def _execute_wave_with_resources(self, wave: ModelWave):
        """Execute wave with resource allocation."""
        # Sort tasks by resource requirements
        tasks_sorted = sorted(
            wave.task_ids,
            key=lambda tid: self._tasks[tid].metadata.get('memory_mb', 0),
            reverse=True
        )

        # Schedule tasks respecting resource limits
        # ...
```

### Benchmarking Strategy

**Performance Benchmarks**:

```python
# tests/performance/test_scheduler_performance.py

import pytest
import asyncio
from omninode_bridge.coordination.scheduler import DependencyAwareScheduler

@pytest.mark.performance
async def test_scheduling_overhead():
    """
    Test scheduling overhead for various graph sizes.

    Target: <50ms for <100 tasks
    """
    scheduler = DependencyAwareScheduler()

    # Add 100 tasks with various dependencies
    for i in range(100):
        deps = [f"task_{j}" for j in range(i) if j % 10 == 0]
        scheduler.add_task(
            f"task_{i}",
            async_dummy_executor,
            dependencies=deps
        )

    # Measure scheduling time
    start = asyncio.get_event_loop().time()
    waves = scheduler.schedule()
    duration_ms = (asyncio.get_event_loop().time() - start) * 1000

    # Assert performance target
    assert duration_ms < 50, f"Scheduling overhead {duration_ms}ms exceeds target 50ms"

@pytest.mark.performance
async def test_parallel_speedup():
    """
    Test parallel speedup for independent tasks.

    Target: 2-3x speedup over sequential
    """
    scheduler = DependencyAwareScheduler()

    # Add 10 independent tasks (each takes ~1s)
    for i in range(10):
        scheduler.add_task(
            f"task_{i}",
            async_sleep_executor,  # Sleeps for 1 second
            dependencies=[]
        )

    scheduler.schedule()

    # Execute in parallel
    start = asyncio.get_event_loop().time()
    result = await scheduler.execute()
    parallel_time = asyncio.get_event_loop().time() - start

    # Calculate speedup
    sequential_time = result.sequential_duration_ms / 1000
    speedup = sequential_time / parallel_time

    # Assert speedup target
    assert speedup >= 2.0, f"Speedup {speedup}x below target 2.0x"
```

---

## Integration Design

### ThreadSafeState Integration

**Purpose**: Share context and results across parallel tasks safely

**Integration Points**:

1. **Task Execution Context**
   ```python
   # Before executing task, get state snapshot
   shared_context = self.state.get_snapshot()

   # Add dependency results
   shared_context['dependency_results'] = {
       dep_id: results[dep_id] for dep_id in task.dependencies
   }

   # Execute task with context
   result = await task.executor(shared_context)
   ```

2. **Result Storage**
   ```python
   # After task completion, update state
   self.state.set(
       f"task_result_{task.task_id}",
       result,
       step_id=task.task_id
   )
   ```

3. **State Version Tracking**
   ```python
   # ThreadSafeState tracks versions for audit trail
   current_version = self.state.get_version()

   # After wave completion, checkpoint version
   self.state.set(
       f"wave_{wave.wave_number}_checkpoint",
       current_version
   )
   ```

**ThreadSafeState API** (from omniagent research):

```python
class ThreadSafeState:
    """
    Thread-safe state container from omniagent research.

    Features:
    - RLock for thread safety
    - Deep copy for data isolation
    - Version tracking
    - Change history auditing
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self._state = initial_state or {}
        self._lock = threading.RLock()
        self._version = 0
        self._history: List[Dict[str, Any]] = []

    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe get with deep copy."""
        with self._lock:
            return deepcopy(self._state.get(key, default))

    def set(self, key: str, value: Any, step_id: Optional[str] = None) -> None:
        """Thread-safe set with audit trail."""
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = deepcopy(value)
            self._version += 1

            self._history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'step_id': step_id,
                'key': key,
                'old_value': old_value,
                'new_value': deepcopy(value),
                'version': self._version
            })

    def get_snapshot(self) -> Dict[str, Any]:
        """Get complete state snapshot."""
        with self._lock:
            return deepcopy(self._state)
```

### Performance Metrics Integration

**Prometheus Metrics**:

```python
from prometheus_client import Histogram, Counter, Gauge

# Scheduling metrics
scheduling_duration_seconds = Histogram(
    'scheduler_scheduling_duration_seconds',
    'Scheduling duration (topological sort + validation)',
    buckets=[0.001, 0.005, 0.010, 0.025, 0.050, 0.100]
)

# Execution metrics
execution_duration_seconds = Histogram(
    'scheduler_execution_duration_seconds',
    'Total execution duration (all waves)',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# Wave metrics
wave_count = Gauge(
    'scheduler_wave_count',
    'Number of waves in current execution'
)

wave_duration_seconds = Histogram(
    'scheduler_wave_duration_seconds',
    'Duration of wave execution',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)

# Speedup metrics
speedup_ratio = Gauge(
    'scheduler_speedup_ratio',
    'Parallel speedup ratio (sequential_time / parallel_time)'
)

# Task metrics
task_total = Counter(
    'scheduler_task_total',
    'Total tasks executed',
    ['status']  # Labels: completed, failed, cancelled
)

task_duration_seconds = Histogram(
    'scheduler_task_duration_seconds',
    'Task execution duration',
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 60.0]
)
```

### Agent Registry Integration

**Purpose**: Discover available agents for task execution

**Integration**:

```python
# Scheduler queries agent registry for available agents
from omninode_bridge.nodes.registry.v1_0_0.node import NodeBridgeRegistry

class AgentAwareScheduler(DependencyAwareScheduler):
    """Scheduler with agent registry integration."""

    def __init__(self, agent_registry: NodeBridgeRegistry):
        super().__init__()
        self.agent_registry = agent_registry

    async def _assign_task_to_agent(self, task: ModelTask) -> str:
        """
        Assign task to available agent based on capabilities.

        Returns:
            Agent ID that will execute task
        """
        # Query registry for agents with required capabilities
        required_capabilities = task.metadata.get('required_capabilities', [])

        available_agents = await self.agent_registry.find_agents(
            capabilities=required_capabilities,
            status='healthy'
        )

        if not available_agents:
            raise ValueError(f"No available agents for task {task.task_id}")

        # Select agent (round-robin, load balancing, etc.)
        selected_agent = available_agents[0]

        return selected_agent.agent_id
```

---

## Testing Strategy

### Unit Tests

**Test Coverage**: Core scheduler logic in isolation

```python
# tests/unit/coordination/test_scheduler.py

import pytest
from omninode_bridge.coordination.scheduler import DependencyAwareScheduler
from omninode_bridge.coordination.thread_safe_state import ThreadSafeState

class TestDependencyGraph:
    """Test dependency graph construction and validation."""

    def test_add_task_with_dependencies(self):
        """Test adding tasks with valid dependencies."""
        scheduler = DependencyAwareScheduler(state=ThreadSafeState())

        scheduler.add_task("task_a", async_dummy, dependencies=[])
        scheduler.add_task("task_b", async_dummy, dependencies=["task_a"])

        assert "task_a" in scheduler._tasks
        assert "task_b" in scheduler._tasks
        assert scheduler._dependency_graph["task_b"] == ["task_a"]

    def test_add_task_invalid_dependency(self):
        """Test adding task with non-existent dependency."""
        scheduler = DependencyAwareScheduler(state=ThreadSafeState())

        scheduler.add_task("task_a", async_dummy, dependencies=["non_existent"])

        with pytest.raises(ValueError, match="non-existent"):
            scheduler.schedule()

    def test_detect_circular_dependency(self):
        """Test circular dependency detection."""
        scheduler = DependencyAwareScheduler(state=ThreadSafeState())

        scheduler.add_task("task_a", async_dummy, dependencies=["task_c"])
        scheduler.add_task("task_b", async_dummy, dependencies=["task_a"])
        scheduler.add_task("task_c", async_dummy, dependencies=["task_b"])

        assert scheduler.detect_deadlock() == True

class TestWaveConstruction:
    """Test wave construction via topological sort."""

    def test_independent_tasks_single_wave(self):
        """Test independent tasks grouped into single wave."""
        scheduler = DependencyAwareScheduler(state=ThreadSafeState())

        scheduler.add_task("task_a", async_dummy, dependencies=[])
        scheduler.add_task("task_b", async_dummy, dependencies=[])
        scheduler.add_task("task_c", async_dummy, dependencies=[])

        waves = scheduler.schedule()

        assert len(waves) == 1
        assert set(waves[0].task_ids) == {"task_a", "task_b", "task_c"}

    def test_sequential_dependencies_multiple_waves(self):
        """Test sequential dependencies create multiple waves."""
        scheduler = DependencyAwareScheduler(state=ThreadSafeState())

        scheduler.add_task("task_a", async_dummy, dependencies=[])
        scheduler.add_task("task_b", async_dummy, dependencies=["task_a"])
        scheduler.add_task("task_c", async_dummy, dependencies=["task_b"])

        waves = scheduler.schedule()

        assert len(waves) == 3
        assert waves[0].task_ids == ["task_a"]
        assert waves[1].task_ids == ["task_b"]
        assert waves[2].task_ids == ["task_c"]

    def test_mixed_dependencies_optimal_waves(self):
        """Test mixed dependencies create optimal wave grouping."""
        scheduler = DependencyAwareScheduler(state=ThreadSafeState())

        # Wave 1: task_a, task_b (independent)
        # Wave 2: task_c (depends on both)
        scheduler.add_task("task_a", async_dummy, dependencies=[])
        scheduler.add_task("task_b", async_dummy, dependencies=[])
        scheduler.add_task("task_c", async_dummy, dependencies=["task_a", "task_b"])

        waves = scheduler.schedule()

        assert len(waves) == 2
        assert set(waves[0].task_ids) == {"task_a", "task_b"}
        assert waves[1].task_ids == ["task_c"]

class TestDeadlockDetection:
    """Test deadlock detection algorithms."""

    def test_no_cycle_returns_false(self):
        """Test no cycle detection returns False."""
        scheduler = DependencyAwareScheduler(state=ThreadSafeState())

        scheduler.add_task("task_a", async_dummy, dependencies=[])
        scheduler.add_task("task_b", async_dummy, dependencies=["task_a"])

        assert scheduler.detect_deadlock() == False

    def test_simple_cycle_returns_true(self):
        """Test simple 2-node cycle detection."""
        scheduler = DependencyAwareScheduler(state=ThreadSafeState())

        scheduler.add_task("task_a", async_dummy, dependencies=["task_b"])
        scheduler.add_task("task_b", async_dummy, dependencies=["task_a"])

        assert scheduler.detect_deadlock() == True

    def test_complex_cycle_returns_true(self):
        """Test complex multi-node cycle detection."""
        scheduler = DependencyAwareScheduler(state=ThreadSafeState())

        scheduler.add_task("task_a", async_dummy, dependencies=["task_b"])
        scheduler.add_task("task_b", async_dummy, dependencies=["task_c"])
        scheduler.add_task("task_c", async_dummy, dependencies=["task_d"])
        scheduler.add_task("task_d", async_dummy, dependencies=["task_a"])

        assert scheduler.detect_deadlock() == True
```

### Integration Tests

**Test Coverage**: End-to-end scheduler execution with real tasks

```python
# tests/integration/coordination/test_scheduler_integration.py

import pytest
import asyncio
from omninode_bridge.coordination.scheduler import DependencyAwareScheduler
from omninode_bridge.coordination.thread_safe_state import ThreadSafeState

@pytest.mark.integration
async def test_parallel_execution_independent_tasks():
    """
    Test parallel execution of independent tasks.

    Validates:
    - All tasks execute in parallel (single wave)
    - Results stored in ThreadSafeState
    - Performance meets speedup target
    """
    state = ThreadSafeState()
    scheduler = DependencyAwareScheduler(state=state)

    # Add 10 independent tasks (each sleeps 1s)
    for i in range(10):
        scheduler.add_task(
            f"task_{i}",
            create_sleep_executor(1.0),
            dependencies=[]
        )

    waves = scheduler.schedule()
    assert len(waves) == 1  # All in single wave

    result = await scheduler.execute()

    # Validate parallel execution
    assert result.total_duration_ms < 1500  # Should complete in ~1s, not 10s
    assert result.speedup_ratio > 5.0  # Should be ~10x speedup
    assert result.successful_tasks == 10

    # Validate results in ThreadSafeState
    for i in range(10):
        assert state.get(f"task_result_task_{i}") is not None

@pytest.mark.integration
async def test_sequential_execution_dependency_chain():
    """
    Test sequential execution of dependency chain.

    Validates:
    - Tasks execute in correct order
    - Dependencies satisfied before execution
    - Results propagate through chain
    """
    state = ThreadSafeState()
    scheduler = DependencyAwareScheduler(state=state)

    # Create chain: task_a → task_b → task_c
    scheduler.add_task("task_a", create_increment_executor(0), dependencies=[])
    scheduler.add_task("task_b", create_increment_executor(1), dependencies=["task_a"])
    scheduler.add_task("task_c", create_increment_executor(2), dependencies=["task_b"])

    waves = scheduler.schedule()
    assert len(waves) == 3  # One wave per task

    result = await scheduler.execute()

    # Validate sequential execution
    assert result.successful_tasks == 3

    # Validate results
    assert state.get("task_result_task_a")['value'] == 1  # 0 + 1
    assert state.get("task_result_task_b")['value'] == 2  # 1 + 1
    assert state.get("task_result_task_c")['value'] == 3  # 2 + 1

@pytest.mark.integration
async def test_mixed_parallel_sequential_execution():
    """
    Test mixed parallel and sequential execution.

    Validates:
    - Correct wave construction for mixed dependencies
    - Parallel execution within waves
    - Sequential execution between waves
    - Speedup vs fully sequential
    """
    state = ThreadSafeState()
    scheduler = DependencyAwareScheduler(state=state)

    # Wave 1: task_a, task_b (parallel)
    # Wave 2: task_c (depends on both)
    # Wave 3: task_d (depends on task_c)
    scheduler.add_task("task_a", create_sleep_executor(1.0), dependencies=[])
    scheduler.add_task("task_b", create_sleep_executor(1.0), dependencies=[])
    scheduler.add_task("task_c", create_sleep_executor(1.0), dependencies=["task_a", "task_b"])
    scheduler.add_task("task_d", create_sleep_executor(1.0), dependencies=["task_c"])

    waves = scheduler.schedule()
    assert len(waves) == 3
    assert set(waves[0].task_ids) == {"task_a", "task_b"}
    assert waves[1].task_ids == ["task_c"]
    assert waves[2].task_ids == ["task_d"]

    result = await scheduler.execute()

    # Validate execution time (~3s for parallel vs ~4s for sequential)
    assert result.total_duration_ms < 3500
    assert result.speedup_ratio > 1.2  # Should be ~1.33x speedup

@pytest.mark.integration
async def test_error_handling_partial_failure():
    """
    Test error handling with partial wave failure.

    Validates:
    - Failed tasks recorded in result
    - Successful tasks complete normally
    - Dependent tasks handle missing inputs
    """
    state = ThreadSafeState()
    scheduler = DependencyAwareScheduler(state=state)

    # Wave 1: task_a (success), task_b (fails)
    # Wave 2: task_c (depends on both, should handle failure)
    scheduler.add_task("task_a", create_success_executor(), dependencies=[])
    scheduler.add_task("task_b", create_failure_executor(), dependencies=[])
    scheduler.add_task(
        "task_c",
        create_partial_dependency_executor(),
        dependencies=["task_a", "task_b"]
    )

    result = await scheduler.execute()

    # Validate partial failure
    assert result.successful_tasks == 2  # task_a and task_c
    assert result.failed_tasks == 1      # task_b
    assert "task_b" in result.task_errors

    # Validate task_c handled partial dependencies
    assert state.get("task_result_task_c")['handled_partial'] == True
```

### Performance Tests

**Test Coverage**: Validate performance targets

```python
# tests/performance/test_scheduler_performance.py

import pytest
import asyncio
import time
from omninode_bridge.coordination.scheduler import DependencyAwareScheduler
from omninode_bridge.coordination.thread_safe_state import ThreadSafeState

@pytest.mark.performance
async def test_scheduling_overhead_small_graph():
    """
    Test scheduling overhead for small graph (<10 tasks).

    Target: <10ms
    """
    scheduler = DependencyAwareScheduler(state=ThreadSafeState())

    for i in range(10):
        deps = [f"task_{i-1}"] if i > 0 else []
        scheduler.add_task(f"task_{i}", async_dummy, dependencies=deps)

    start = time.perf_counter()
    scheduler.schedule()
    duration_ms = (time.perf_counter() - start) * 1000

    assert duration_ms < 10, f"Scheduling overhead {duration_ms}ms exceeds 10ms target"

@pytest.mark.performance
async def test_scheduling_overhead_large_graph():
    """
    Test scheduling overhead for large graph (100 tasks).

    Target: <50ms
    """
    scheduler = DependencyAwareScheduler(state=ThreadSafeState())

    for i in range(100):
        # Create complex dependency structure
        deps = [f"task_{j}" for j in range(i) if i % 5 == 0 and j < i]
        scheduler.add_task(f"task_{i}", async_dummy, dependencies=deps)

    start = time.perf_counter()
    scheduler.schedule()
    duration_ms = (time.perf_counter() - start) * 1000

    assert duration_ms < 50, f"Scheduling overhead {duration_ms}ms exceeds 50ms target"

@pytest.mark.performance
async def test_parallel_speedup_independent_tasks():
    """
    Test parallel speedup for fully independent tasks.

    Target: >2x speedup
    """
    state = ThreadSafeState()
    scheduler = DependencyAwareScheduler(state=state)

    # Add 20 independent CPU-bound tasks
    for i in range(20):
        scheduler.add_task(
            f"task_{i}",
            create_cpu_bound_executor(),  # CPU-intensive work
            dependencies=[]
        )

    scheduler.schedule()
    result = await scheduler.execute()

    # Validate speedup
    assert result.speedup_ratio >= 2.0, \
        f"Speedup {result.speedup_ratio}x below 2.0x target"

@pytest.mark.performance
async def test_parallel_speedup_mixed_dependencies():
    """
    Test parallel speedup for mixed parallel/sequential.

    Target: >1.5x speedup
    """
    state = ThreadSafeState()
    scheduler = DependencyAwareScheduler(state=state)

    # Create mixed dependency structure
    # Wave 1: 10 parallel tasks
    # Wave 2: 5 parallel tasks (depend on Wave 1)
    # Wave 3: 1 task (depends on Wave 2)

    for i in range(10):
        scheduler.add_task(f"wave1_task_{i}", create_sleep_executor(1.0), dependencies=[])

    for i in range(5):
        scheduler.add_task(
            f"wave2_task_{i}",
            create_sleep_executor(1.0),
            dependencies=[f"wave1_task_{j}" for j in range(10)]
        )

    scheduler.add_task(
        "wave3_task",
        create_sleep_executor(1.0),
        dependencies=[f"wave2_task_{i}" for i in range(5)]
    )

    scheduler.schedule()
    result = await scheduler.execute()

    # Sequential time: 16 tasks × 1s = 16s
    # Parallel time: 3 waves × 1s = 3s
    # Expected speedup: ~5.3x
    assert result.speedup_ratio >= 1.5, \
        f"Speedup {result.speedup_ratio}x below 1.5x target"

@pytest.mark.performance
async def test_wave_synchronization_overhead():
    """
    Test wave synchronization overhead.

    Target: <20ms per wave
    """
    state = ThreadSafeState()
    scheduler = DependencyAwareScheduler(state=state)

    # Create 10 waves with 1 task each (worst case for overhead)
    for i in range(10):
        deps = [f"task_{i-1}"] if i > 0 else []
        scheduler.add_task(
            f"task_{i}",
            create_instant_executor(),  # Instant execution
            dependencies=deps
        )

    scheduler.schedule()
    result = await scheduler.execute()

    # Calculate average wave overhead
    # Total time - task time = overhead
    task_time_ms = sum(result.wave_summary[i]['duration_ms'] for i in range(10))
    overhead_per_wave_ms = task_time_ms / 10

    assert overhead_per_wave_ms < 20, \
        f"Wave overhead {overhead_per_wave_ms}ms exceeds 20ms target"
```

---

## Implementation Plan

### File Structure

```
src/omninode_bridge/coordination/
├── __init__.py
├── scheduler.py                           # DependencyAwareScheduler
├── thread_safe_state.py                   # ThreadSafeState (from Wave 2)
├── models/
│   ├── __init__.py
│   ├── model_task.py                      # ModelTask
│   ├── model_wave.py                      # ModelWave
│   ├── model_execution_result.py          # ModelExecutionResult
│   └── enum_task_status.py                # EnumTaskStatus

tests/unit/coordination/
├── __init__.py
├── test_scheduler.py                      # Unit tests
├── test_dependency_graph.py
├── test_wave_construction.py
├── test_deadlock_detection.py

tests/integration/coordination/
├── __init__.py
├── test_scheduler_integration.py          # Integration tests
├── test_parallel_execution.py
├── test_error_handling.py

tests/performance/
├── __init__.py
├── test_scheduler_performance.py          # Performance benchmarks

docs/architecture/
└── PARALLEL_SCHEDULER_DESIGN.md           # This document
```

### Implementation Order

**Wave 2**: ThreadSafeState (Prerequisite)
```
1. Implement ThreadSafeState (from omniagent research)
   - RLock for thread safety
   - Deep copy for isolation
   - Version tracking
   - Change history

2. Unit tests for ThreadSafeState
   - Thread safety validation
   - Concurrent access tests
   - Performance benchmarks

Duration: 2-3 days
```

**Wave 3**: Core Scheduler (This Design)
```
Phase 1: Data Models (Day 1)
├─ ModelTask
├─ ModelWave
├─ ModelExecutionResult
└─ EnumTaskStatus

Phase 2: Dependency Graph (Days 2-3)
├─ DependencyAwareScheduler.__init__()
├─ add_task()
├─ _validate_dependency_graph()
└─ Unit tests

Phase 3: Topological Sort (Days 4-5)
├─ _topological_sort_with_waves()
├─ schedule()
└─ Unit tests for wave construction

Phase 4: Deadlock Detection (Day 6)
├─ detect_deadlock()
├─ DFS cycle detection
└─ Unit tests

Phase 5: Execution Engine (Days 7-9)
├─ execute()
├─ _execute_wave()
├─ _execute_task_with_semaphore()
├─ ThreadSafeState integration
└─ Integration tests

Phase 6: Error Handling (Days 10-11)
├─ Retry logic
├─ Partial failure handling
├─ Error propagation
└─ Integration tests

Phase 7: Performance Optimization (Days 12-13)
├─ Wave optimization
├─ Resource allocation
└─ Performance benchmarks

Phase 8: Documentation & Polish (Day 14)
├─ API documentation
├─ Usage examples
└─ Final integration tests

Duration: 14 days (2 weeks)
```

### Testing Milestones

**Milestone 1**: Dependency Graph Validation
```
✅ Add tasks with dependencies
✅ Validate dependency existence
✅ Detect circular dependencies
✅ Handle invalid inputs
```

**Milestone 2**: Wave Construction
```
✅ Independent tasks → single wave
✅ Sequential dependencies → multiple waves
✅ Mixed dependencies → optimal waves
✅ Complex graphs → correct ordering
```

**Milestone 3**: Parallel Execution
```
✅ Execute independent tasks in parallel
✅ Execute waves sequentially
✅ ThreadSafeState integration
✅ Result aggregation
```

**Milestone 4**: Error Handling
```
✅ Handle task failures
✅ Handle partial wave failures
✅ Retry logic with backoff
✅ Graceful degradation
```

**Milestone 5**: Performance Validation
```
✅ Scheduling overhead <50ms
✅ Parallel speedup >2x for independent tasks
✅ Wave synchronization <20ms
✅ Memory overhead <100MB
```

### Success Criteria

**Functional**:
- ✅ All dependency constraints respected
- ✅ No race conditions in parallel execution
- ✅ Circular dependencies detected and rejected
- ✅ Graceful error handling with partial failures

**Performance**:
- ✅ Scheduling overhead <50ms for <100 tasks
- ✅ Parallel speedup 2.25x-4.17x for independent tasks
- ✅ Wave synchronization <20ms overhead
- ✅ Memory overhead <100MB for typical workflows

**Quality**:
- ✅ >90% test coverage (unit + integration)
- ✅ Type safety with Pydantic v2
- ✅ Comprehensive error messages
- ✅ Production-ready logging and metrics

### Integration with Phase 4 Code Generation

**Usage Example**:

```python
# Phase 4 code generation workflow
from omninode_bridge.coordination.scheduler import DependencyAwareScheduler
from omninode_bridge.coordination.thread_safe_state import ThreadSafeState

async def phase4_code_generation(contracts: List[ModelContract]):
    """
    Generate code from contracts using parallel scheduler.

    Workflow:
    - Wave 1: Parse all contracts in parallel
    - Wave 2: Generate models in parallel (depend on parsing)
    - Wave 3: Generate validators in parallel (depend on models)
    - Wave 4: Generate tests in parallel (depend on validators)
    - Wave 5: Final validation (depends on all)
    """
    # Initialize scheduler with shared state
    state = ThreadSafeState()
    scheduler = DependencyAwareScheduler(
        state=state,
        max_concurrent=10,
        enable_deadlock_detection=True
    )

    # Wave 1: Parse contracts
    for contract in contracts:
        scheduler.add_task(
            f"parse_{contract.name}",
            create_parse_executor(contract),
            dependencies=[]
        )

    # Wave 2: Generate models
    for contract in contracts:
        scheduler.add_task(
            f"generate_model_{contract.name}",
            create_model_generator(contract),
            dependencies=[f"parse_{contract.name}"]
        )

    # Wave 3: Generate validators
    for contract in contracts:
        scheduler.add_task(
            f"generate_validator_{contract.name}",
            create_validator_generator(contract),
            dependencies=[f"generate_model_{contract.name}"]
        )

    # Wave 4: Generate tests
    for contract in contracts:
        scheduler.add_task(
            f"generate_tests_{contract.name}",
            create_test_generator(contract),
            dependencies=[f"generate_validator_{contract.name}"]
        )

    # Wave 5: Final validation
    scheduler.add_task(
        "final_validation",
        create_final_validator(),
        dependencies=[
            f"generate_tests_{c.name}" for c in contracts
        ]
    )

    # Schedule and execute
    waves = scheduler.schedule()
    result = await scheduler.execute()

    # Report results
    print(f"Generated code for {len(contracts)} contracts")
    print(f"Total duration: {result.total_duration_ms / 1000:.2f}s")
    print(f"Speedup: {result.speedup_ratio:.2f}x")
    print(f"Success rate: {result.successful_tasks / result.total_tasks * 100:.1f}%")

    return result
```

---

## Conclusion

This design document provides a comprehensive architecture for a **dependency-aware parallel scheduler** that achieves:

✅ **2-3x speedup** for multi-agent code generation workflows
✅ **Correctness** through dependency resolution and deadlock detection
✅ **Type safety** with Pydantic v2 throughout
✅ **Production readiness** with comprehensive error handling
✅ **ONEX v2.0 compliance** with suffix-based naming and contracts

**Next Steps**:

1. **Review & Approval** - Review design with team, gather feedback
2. **Wave 2 Implementation** - Implement ThreadSafeState (prerequisite)
3. **Wave 3 Implementation** - Implement scheduler (this design)
4. **Performance Validation** - Run benchmarks, validate 2-3x speedup
5. **Phase 4 Integration** - Integrate with code generation pipeline

**Timeline**: 2-3 weeks total (Wave 2: 2-3 days, Wave 3: 14 days)

---

**Document Version**: 1.0
**Author**: OmniNode Bridge Architecture Team
**Review Date**: 2025-11-06
**Implementation Target**: Wave 3 (Q4 2025)
