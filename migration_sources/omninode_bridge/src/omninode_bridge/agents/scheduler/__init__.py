"""
Dependency-aware parallel scheduler for agent coordination.

This package provides production-ready parallel task scheduling with automatic
dependency resolution, deadlock detection, and wave-based execution.

Performance targets (validated from omniagent):
- Scheduling overhead: <50ms for <100 tasks
- Parallel speedup: 2-3x for independent tasks
- Deadlock detection: 100% accuracy

Example:
    ```python
    from omninode_bridge.agents.scheduler import DependencyAwareScheduler
    from omninode_bridge.agents.coordination import ThreadSafeState

    state = ThreadSafeState()
    scheduler = DependencyAwareScheduler(state=state)

    # Add tasks
    scheduler.add_task("parse", async_parse_fn, dependencies=[])
    scheduler.add_task("generate", async_generate_fn, dependencies=["parse"])

    # Schedule and execute
    waves = scheduler.schedule()
    results = await scheduler.execute()

    print(f"Speedup: {results.speedup_ratio}x")
    ```
"""

from .dag import DependencyGraph
from .exceptions import (
    CircularDependencyError,
    InvalidTaskError,
    SchedulerError,
    SchedulingTimeoutError,
    TaskNotFoundError,
    WaveExecutionError,
)
from .models import EnumTaskStatus, ModelExecutionResult, ModelWave, Task
from .scheduler import DependencyAwareScheduler

__all__ = [
    # Main scheduler
    "DependencyAwareScheduler",
    # DAG operations
    "DependencyGraph",
    # Models
    "Task",
    "ModelWave",
    "ModelExecutionResult",
    "EnumTaskStatus",
    # Exceptions
    "SchedulerError",
    "CircularDependencyError",
    "TaskNotFoundError",
    "WaveExecutionError",
    "SchedulingTimeoutError",
    "InvalidTaskError",
]
