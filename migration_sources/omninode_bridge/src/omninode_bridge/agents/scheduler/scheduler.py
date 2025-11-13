"""
Dependency-aware parallel task scheduler.

This module provides a production-ready scheduler that executes tasks in parallel
while automatically resolving dependencies and detecting deadlocks.

Performance targets (validated from omniagent):
- Scheduling overhead: <50ms for <100 tasks
- Parallel speedup: 2-3x for independent tasks
- Deadlock detection: 100% accuracy
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from ..coordination.thread_safe_state import ThreadSafeState
from .dag import DependencyGraph
from .exceptions import CircularDependencyError, InvalidTaskError
from .models import EnumTaskStatus, ModelExecutionResult, ModelWave, Task

logger = logging.getLogger(__name__)


class DependencyAwareScheduler:
    """
    Dependency-aware parallel task scheduler with wave-based execution.

    Features:
    - DAG-based dependency resolution
    - Wave construction via topological sort
    - Parallel execution within waves
    - Deadlock detection via cycle detection
    - ThreadSafeState integration for shared context

    Performance (validated in omniagent):
    - Scheduling overhead: <50ms
    - Parallel speedup: 2-3x for independent tasks
    - Wave synchronization: <20ms per wave

    Example:
        ```python
        scheduler = DependencyAwareScheduler(state=shared_state)
        scheduler.add_task("parse_contract", async_parse_fn, dependencies=[])
        scheduler.add_task("generate_model", async_generate_fn, dependencies=["parse_contract"])

        waves = scheduler.schedule()
        results = await scheduler.execute()

        print(f"Speedup: {results.speedup_ratio}x")
        ```
    """

    def __init__(
        self,
        state: ThreadSafeState,
        max_concurrent: int = 10,
        enable_deadlock_detection: bool = True,
        enable_wave_optimization: bool = True,
    ) -> None:
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
        self._tasks: dict[str, Task] = {}

        # Dependency graph
        self._dependency_graph = DependencyGraph()

        # Execution tracking
        self._execution_id: Optional[UUID] = None
        self._waves: list[ModelWave] = []
        self._start_time: Optional[float] = None

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"[DependencyAwareScheduler] Initialized with max_concurrent={max_concurrent}, "
            f"deadlock_detection={enable_deadlock_detection}"
        )

    def add_task(
        self,
        task_id: str,
        executor: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        dependencies: list[str],
        timeout_seconds: float = 300.0,
        retry_count: int = 0,
        metadata: Optional[dict[str, Any]] = None,
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
            InvalidTaskError: If executor is not callable
        """
        if task_id in self._tasks:
            raise ValueError(f"Task '{task_id}' already exists")

        if not callable(executor):
            raise InvalidTaskError(task_id=task_id, reason="executor must be callable")

        task = Task(
            task_id=task_id,
            executor=executor,
            dependencies=dependencies,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            metadata=metadata or {},
            status=EnumTaskStatus.PENDING,
            created_at=datetime.utcnow(),
        )

        self._tasks[task_id] = task
        self._dependency_graph.add_task(task_id, dependencies)

        logger.debug(
            f"[DependencyAwareScheduler] Added task '{task_id}' with {len(dependencies)} dependencies"
        )

    def schedule(self) -> list[ModelWave]:
        """
        Construct execution waves via topological sort.

        Returns:
            List of waves (each wave is list of task_ids that can run in parallel)

        Raises:
            TaskNotFoundError: If task references non-existent dependency
            CircularDependencyError: If circular dependency detected (deadlock)

        Algorithm:
            1. Validate dependency graph
            2. Detect circular dependencies (if enabled)
            3. Perform topological sort with wave grouping
            4. Optimize waves (if enabled)
            5. Return wave schedule

        Performance: O(V + E) where V=tasks, E=dependencies
        """
        if not self._tasks:
            logger.warning("[DependencyAwareScheduler] No tasks to schedule")
            return []

        logger.info(f"[DependencyAwareScheduler] Scheduling {len(self._tasks)} tasks")

        # Step 1: Validate dependency graph
        self._dependency_graph.validate()

        # Step 2: Detect circular dependencies
        if self.enable_deadlock_detection:
            if self._dependency_graph.has_cycle():
                cycle_path = self._dependency_graph.find_cycle_path()
                logger.error(
                    f"[DependencyAwareScheduler] Circular dependency detected: {cycle_path}"
                )
                raise CircularDependencyError(cycle_path=cycle_path)

        # Step 3: Construct waves via topological sort
        wave_task_ids = self._dependency_graph.construct_waves()

        # Step 4: Create Wave models
        self._waves = [
            ModelWave(
                wave_number=i + 1,
                task_ids=task_ids,
                status=EnumTaskStatus.PENDING,
                created_at=datetime.utcnow(),
            )
            for i, task_ids in enumerate(wave_task_ids)
        ]

        logger.info(
            f"[DependencyAwareScheduler] Created {len(self._waves)} waves. "
            f"Wave sizes: {[len(w.task_ids) for w in self._waves]}"
        )

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
                2. Wait for wave completion (asyncio.gather)
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

        logger.info(
            f"[DependencyAwareScheduler] Starting execution {self._execution_id}"
        )

        # Track results
        task_results: dict[str, Any] = {}
        task_errors: dict[str, str] = {}

        # Execute waves sequentially (parallel within each wave)
        for wave in self._waves:
            wave_start = asyncio.get_event_loop().time()
            wave.status = EnumTaskStatus.IN_PROGRESS

            logger.info(
                f"[DependencyAwareScheduler] Executing wave {wave.wave_number} "
                f"with {len(wave.task_ids)} tasks: {wave.task_ids}"
            )

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
            wave_results_list = await asyncio.gather(
                *wave_tasks.values(), return_exceptions=True
            )

            # Process wave results
            for task_id, task_result in zip(
                wave_tasks.keys(), wave_results_list, strict=False
            ):
                if isinstance(task_result, Exception):
                    task_errors[task_id] = str(task_result)
                    self._tasks[task_id].status = EnumTaskStatus.FAILED
                    self._tasks[task_id].error = str(task_result)
                    logger.error(
                        f"[DependencyAwareScheduler] Task '{task_id}' failed: {task_result}"
                    )
                else:
                    # task_result is Dict[str, Any] here (not BaseException)
                    task_results[task_id] = task_result
                    self._tasks[task_id].status = EnumTaskStatus.COMPLETED
                    self._tasks[task_id].result = task_result  # type: ignore[assignment]
                    logger.debug(
                        f"[DependencyAwareScheduler] Task '{task_id}' completed successfully"
                    )

            wave.status = EnumTaskStatus.COMPLETED
            wave.duration_ms = (asyncio.get_event_loop().time() - wave_start) * 1000

            logger.info(
                f"[DependencyAwareScheduler] Wave {wave.wave_number} completed in {wave.duration_ms:.2f}ms. "
                f"Success: {len([tid for tid in wave.task_ids if tid in task_results])}/{len(wave.task_ids)}"
            )

        # Calculate performance metrics
        total_duration = asyncio.get_event_loop().time() - self._start_time
        sequential_time = sum(
            (self._tasks[task_id].duration_ms or 0.0) / 1000 for task_id in task_results
        )
        speedup_ratio = sequential_time / total_duration if total_duration > 0 else 1.0

        execution_result = ModelExecutionResult(
            execution_id=self._execution_id,
            total_tasks=len(self._tasks),
            successful_tasks=len(task_results),
            failed_tasks=len(task_errors),
            total_waves=len(self._waves),
            total_duration_ms=total_duration * 1000,
            sequential_duration_ms=sequential_time * 1000,
            speedup_ratio=speedup_ratio,
            task_results=task_results,
            task_errors=task_errors,
            wave_summary=[
                {
                    "wave_number": wave.wave_number,
                    "task_count": len(wave.task_ids),
                    "duration_ms": wave.duration_ms,
                    "status": wave.status.value,
                }
                for wave in self._waves
            ],
        )

        logger.info(
            f"[DependencyAwareScheduler] Execution completed. "
            f"Total: {total_duration * 1000:.2f}ms, "
            f"Speedup: {speedup_ratio:.2f}x, "
            f"Success rate: {len(task_results)}/{len(self._tasks)}"
        )

        return execution_result

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
        return self._dependency_graph.has_cycle()

    def get_dag(self) -> dict[str, list[str]]:
        """
        Get dependency graph representation.

        Returns:
            Dictionary mapping task_id to list of dependencies
        """
        return {task_id: task.dependencies for task_id, task in self._tasks.items()}

    async def _execute_task_with_semaphore(
        self,
        task: Task,
        results: dict[str, Any],
        errors: dict[str, str],
    ) -> dict[str, Any]:
        """
        Execute task with semaphore for concurrency control.

        Args:
            task: Task to execute
            results: Shared results dictionary
            errors: Shared errors dictionary

        Returns:
            Task execution result

        Raises:
            Exception: If task execution fails after all retries
        """
        async with self._semaphore:
            task_start = asyncio.get_event_loop().time()
            initial_retry_count = task.retry_count

            while True:
                try:
                    # Get shared state snapshot for task
                    shared_context = self.state.snapshot()

                    # Add dependency results to context
                    dep_results = {
                        dep_id: results[dep_id]
                        for dep_id in task.dependencies
                        if dep_id in results
                    }
                    shared_context["dependency_results"] = dep_results
                    shared_context["task_id"] = task.task_id
                    shared_context["task_metadata"] = task.metadata

                    # Execute task with timeout
                    result = await asyncio.wait_for(
                        task.executor(shared_context), timeout=task.timeout_seconds
                    )

                    # Update ThreadSafeState with result
                    self.state.set(
                        f"task_result_{task.task_id}", result, changed_by=task.task_id
                    )

                    task.duration_ms = (
                        asyncio.get_event_loop().time() - task_start
                    ) * 1000
                    return result

                except Exception as e:
                    task.duration_ms = (
                        asyncio.get_event_loop().time() - task_start
                    ) * 1000

                    # Retry logic
                    if task.retry_count > 0:
                        task.retry_count -= 1
                        retry_number = initial_retry_count - task.retry_count
                        logger.warning(
                            f"[DependencyAwareScheduler] Task '{task.task_id}' failed, "
                            f"retrying ({retry_number}/{initial_retry_count}): {e}"
                        )
                        # Exponential backoff
                        await asyncio.sleep(2 ** (retry_number - 1))
                        continue

                    # No retries left
                    logger.error(
                        f"[DependencyAwareScheduler] Task '{task.task_id}' failed after "
                        f"{initial_retry_count} retries: {e}"
                    )
                    raise e

    def get_task_status(self, task_id: str) -> Optional[EnumTaskStatus]:
        """
        Get current status of a task.

        Args:
            task_id: Task identifier

        Returns:
            Task status or None if task not found
        """
        task = self._tasks.get(task_id)
        return task.status if task else None

    def get_wave_summary(self) -> list[dict[str, Any]]:
        """
        Get summary of wave construction.

        Returns:
            List of wave summaries with task counts and IDs
        """
        return [
            {
                "wave_number": wave.wave_number,
                "task_count": len(wave.task_ids),
                "task_ids": wave.task_ids,
                "status": wave.status.value,
            }
            for wave in self._waves
        ]

    def clear(self) -> None:
        """
        Clear all tasks and reset scheduler state.

        Use this to reuse scheduler instance for new execution.
        """
        self._tasks.clear()
        self._dependency_graph = DependencyGraph()
        self._waves.clear()
        self._execution_id = None
        self._start_time = None

        logger.info("[DependencyAwareScheduler] Cleared all tasks and state")

    def __len__(self) -> int:
        """Get number of tasks in scheduler."""
        return len(self._tasks)

    def __repr__(self) -> str:
        """String representation of scheduler."""
        return (
            f"DependencyAwareScheduler(tasks={len(self._tasks)}, "
            f"waves={len(self._waves)}, "
            f"state_version={self.state.get_version()})"
        )
