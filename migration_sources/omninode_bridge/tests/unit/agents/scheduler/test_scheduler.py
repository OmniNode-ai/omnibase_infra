"""
Comprehensive unit tests for dependency-aware parallel scheduler.

Test coverage:
- Dependency graph construction and validation
- Circular dependency detection
- Topological sorting
- Wave construction
- Parallel execution
- Error handling
- Performance benchmarks
"""

import asyncio
import time
from typing import Any

import pytest

from omninode_bridge.agents.coordination.thread_safe_state import ThreadSafeState
from omninode_bridge.agents.scheduler import (
    CircularDependencyError,
    DependencyAwareScheduler,
    DependencyGraph,
    EnumTaskStatus,
    InvalidTaskError,
    TaskNotFoundError,
)

# Test helpers


async def dummy_executor(context: dict[str, Any]) -> dict[str, Any]:
    """Dummy executor that returns immediately."""
    return {"result": "success", "task_id": context.get("task_id")}


async def sleep_executor(duration: float = 0.1):
    """Create executor that sleeps for specified duration."""

    async def executor(context: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(duration)
        return {
            "result": "success",
            "task_id": context.get("task_id"),
            "slept": duration,
        }

    return executor


async def failing_executor(context: dict[str, Any]) -> dict[str, Any]:
    """Executor that always fails."""
    raise ValueError("Task failed intentionally")


async def increment_executor(base_value: int):
    """Create executor that increments a base value."""

    async def executor(context: dict[str, Any]) -> dict[str, Any]:
        dep_results = context.get("dependency_results", {})
        # Get value from first dependency if exists
        prev_value = base_value
        if dep_results:
            first_dep = list(dep_results.values())[0]
            prev_value = first_dep.get("value", base_value)
        return {"value": prev_value + 1, "task_id": context.get("task_id")}

    return executor


# DependencyGraph Tests


class TestDependencyGraph:
    """Test dependency graph operations."""

    def test_add_task_with_dependencies(self):
        """Test adding tasks with valid dependencies."""
        graph = DependencyGraph()

        graph.add_task("task_a", dependencies=[])
        graph.add_task("task_b", dependencies=["task_a"])

        assert len(graph) == 2
        assert "task_a" in graph
        assert "task_b" in graph

    def test_add_task_duplicate_raises(self):
        """Test adding duplicate task raises error."""
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=[])

        with pytest.raises(ValueError, match="already exists"):
            graph.add_task("task_a", dependencies=[])

    def test_validate_invalid_dependency(self):
        """Test validation detects non-existent dependencies."""
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=["non_existent"])

        with pytest.raises(TaskNotFoundError, match="non_existent"):
            graph.validate()

    def test_detect_circular_dependency_simple(self):
        """Test detection of simple 2-node cycle."""
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=["task_b"])
        graph.add_task("task_b", dependencies=["task_a"])

        assert graph.has_cycle() is True

    def test_detect_circular_dependency_complex(self):
        """Test detection of complex multi-node cycle."""
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=["task_b"])
        graph.add_task("task_b", dependencies=["task_c"])
        graph.add_task("task_c", dependencies=["task_d"])
        graph.add_task("task_d", dependencies=["task_a"])

        assert graph.has_cycle() is True

    def test_no_cycle_returns_false(self):
        """Test no cycle detection returns False."""
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=[])
        graph.add_task("task_b", dependencies=["task_a"])
        graph.add_task("task_c", dependencies=["task_b"])

        assert graph.has_cycle() is False

    def test_find_cycle_path(self):
        """Test finding circular dependency path."""
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=["task_b"])
        graph.add_task("task_b", dependencies=["task_c"])
        graph.add_task("task_c", dependencies=["task_a"])

        cycle_path = graph.find_cycle_path()

        assert cycle_path is not None
        assert len(cycle_path) == 4  # [task_x, task_y, task_z, task_x]
        assert cycle_path[0] == cycle_path[-1]  # Cycle closes

    def test_topological_sort(self):
        """Test topological sorting."""
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=[])
        graph.add_task("task_b", dependencies=["task_a"])
        graph.add_task("task_c", dependencies=["task_b"])

        sorted_tasks = graph.topological_sort()

        # task_a must come before task_b, task_b before task_c
        assert sorted_tasks.index("task_a") < sorted_tasks.index("task_b")
        assert sorted_tasks.index("task_b") < sorted_tasks.index("task_c")

    def test_topological_sort_raises_on_cycle(self):
        """Test topological sort raises on circular dependency."""
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=["task_b"])
        graph.add_task("task_b", dependencies=["task_a"])

        with pytest.raises(CircularDependencyError):
            graph.topological_sort()

    def test_construct_waves_independent(self):
        """Test wave construction for independent tasks."""
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=[])
        graph.add_task("task_b", dependencies=[])
        graph.add_task("task_c", dependencies=[])

        waves = graph.construct_waves()

        # All independent tasks should be in wave 1
        assert len(waves) == 1
        assert set(waves[0]) == {"task_a", "task_b", "task_c"}

    def test_construct_waves_sequential(self):
        """Test wave construction for sequential dependencies."""
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=[])
        graph.add_task("task_b", dependencies=["task_a"])
        graph.add_task("task_c", dependencies=["task_b"])

        waves = graph.construct_waves()

        # Sequential dependencies should create 3 waves
        assert len(waves) == 3
        assert waves[0] == ["task_a"]
        assert waves[1] == ["task_b"]
        assert waves[2] == ["task_c"]

    def test_construct_waves_mixed(self):
        """Test wave construction for mixed dependencies."""
        graph = DependencyGraph()
        # Wave 1: task_a, task_b (independent)
        # Wave 2: task_c (depends on both)
        graph.add_task("task_a", dependencies=[])
        graph.add_task("task_b", dependencies=[])
        graph.add_task("task_c", dependencies=["task_a", "task_b"])

        waves = graph.construct_waves()

        assert len(waves) == 2
        assert set(waves[0]) == {"task_a", "task_b"}
        assert waves[1] == ["task_c"]

    def test_get_independent_tasks(self):
        """Test getting tasks with no dependencies."""
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=[])
        graph.add_task("task_b", dependencies=[])
        graph.add_task("task_c", dependencies=["task_a"])

        independent = graph.get_independent_tasks()

        assert set(independent) == {"task_a", "task_b"}


# DependencyAwareScheduler Tests


class TestDependencyAwareScheduler:
    """Test dependency-aware scheduler."""

    def test_add_task(self):
        """Test adding tasks to scheduler."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_a", dummy_executor, dependencies=[])

        assert len(scheduler) == 1
        assert scheduler.get_task_status("task_a") == EnumTaskStatus.PENDING

    def test_add_task_duplicate_raises(self):
        """Test adding duplicate task raises error."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_a", dummy_executor, dependencies=[])

        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_task("task_a", dummy_executor, dependencies=[])

    def test_add_task_invalid_executor(self):
        """Test adding task with non-callable executor raises error."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        with pytest.raises(InvalidTaskError, match="must be callable"):
            scheduler.add_task("task_a", "not_callable", dependencies=[])  # type: ignore

    def test_schedule_creates_waves(self):
        """Test schedule creates execution waves."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_a", dummy_executor, dependencies=[])
        scheduler.add_task("task_b", dummy_executor, dependencies=["task_a"])

        waves = scheduler.schedule()

        assert len(waves) == 2
        assert waves[0].task_ids == ["task_a"]
        assert waves[1].task_ids == ["task_b"]

    def test_schedule_detects_circular_dependency(self):
        """Test schedule detects circular dependencies."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_a", dummy_executor, dependencies=["task_b"])
        scheduler.add_task("task_b", dummy_executor, dependencies=["task_a"])

        with pytest.raises(CircularDependencyError):
            scheduler.schedule()

    def test_schedule_validates_dependencies(self):
        """Test schedule validates all dependencies exist."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_a", dummy_executor, dependencies=["non_existent"])

        with pytest.raises(TaskNotFoundError, match="non_existent"):
            scheduler.schedule()

    @pytest.mark.asyncio
    async def test_execute_requires_schedule(self):
        """Test execute raises if schedule not called."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_a", dummy_executor, dependencies=[])

        with pytest.raises(ValueError, match="Must call schedule"):
            await scheduler.execute()

    @pytest.mark.asyncio
    async def test_execute_independent_tasks_parallel(self):
        """Test independent tasks execute in parallel."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        # Add 3 independent tasks that sleep for 0.1s each
        for i in range(3):
            executor = await sleep_executor(0.1)
            scheduler.add_task(f"task_{i}", executor, dependencies=[])

        scheduler.schedule()

        start = time.time()
        result = await scheduler.execute()
        duration = time.time() - start

        # Should complete in ~0.1s (parallel) not ~0.3s (sequential)
        assert duration < 0.2  # Allow some overhead
        assert result.successful_tasks == 3
        assert result.failed_tasks == 0
        # Speedup should be close to 3x
        assert result.speedup_ratio > 2.0

    @pytest.mark.asyncio
    async def test_execute_sequential_tasks(self):
        """Test sequential dependencies execute in order."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        # Create chain: task_a → task_b → task_c
        scheduler.add_task("task_a", await increment_executor(0), dependencies=[])
        scheduler.add_task(
            "task_b", await increment_executor(0), dependencies=["task_a"]
        )
        scheduler.add_task(
            "task_c", await increment_executor(0), dependencies=["task_b"]
        )

        scheduler.schedule()
        result = await scheduler.execute()

        # Verify results propagate through chain
        assert result.task_results["task_a"]["value"] == 1  # 0 + 1
        assert result.task_results["task_b"]["value"] == 2  # 1 + 1
        assert result.task_results["task_c"]["value"] == 3  # 2 + 1

    @pytest.mark.asyncio
    async def test_execute_updates_thread_safe_state(self):
        """Test execution updates ThreadSafeState with results."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_a", dummy_executor, dependencies=[])
        scheduler.schedule()

        await scheduler.execute()

        # Verify result stored in state
        result = state.get("task_result_task_a")
        assert result is not None
        assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_execute_with_failures(self):
        """Test execution with some task failures."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_success", dummy_executor, dependencies=[])
        scheduler.add_task("task_fail", failing_executor, dependencies=[])

        scheduler.schedule()
        result = await scheduler.execute()

        assert result.successful_tasks == 1
        assert result.failed_tasks == 1
        assert "task_fail" in result.task_errors

    @pytest.mark.asyncio
    async def test_execute_with_retry(self):
        """Test task retry on failure."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        attempt_count = {"count": 0}

        async def retry_executor(context: dict[str, Any]) -> dict[str, Any]:
            attempt_count["count"] += 1
            if attempt_count["count"] < 2:
                raise ValueError("First attempt fails")
            return {"result": "success", "attempts": attempt_count["count"]}

        scheduler.add_task("task_retry", retry_executor, dependencies=[], retry_count=2)
        scheduler.schedule()

        result = await scheduler.execute()

        assert result.successful_tasks == 1
        assert attempt_count["count"] == 2  # Failed once, succeeded on retry

    def test_get_wave_summary(self):
        """Test getting wave summary."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_a", dummy_executor, dependencies=[])
        scheduler.add_task("task_b", dummy_executor, dependencies=[])
        scheduler.schedule()

        summary = scheduler.get_wave_summary()

        assert len(summary) == 1
        assert summary[0]["wave_number"] == 1
        assert summary[0]["task_count"] == 2

    def test_clear_resets_scheduler(self):
        """Test clear resets scheduler state."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_a", dummy_executor, dependencies=[])
        scheduler.schedule()

        scheduler.clear()

        assert len(scheduler) == 0
        assert len(scheduler.get_wave_summary()) == 0

    def test_detect_deadlock(self):
        """Test deadlock detection."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_a", dummy_executor, dependencies=["task_b"])
        scheduler.add_task("task_b", dummy_executor, dependencies=["task_a"])

        assert scheduler.detect_deadlock() is True

    def test_get_dag(self):
        """Test getting dependency graph."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        scheduler.add_task("task_a", dummy_executor, dependencies=[])
        scheduler.add_task("task_b", dummy_executor, dependencies=["task_a"])

        dag = scheduler.get_dag()

        assert dag["task_a"] == []
        assert dag["task_b"] == ["task_a"]


# Performance Tests


class TestSchedulerPerformance:
    """Test scheduler performance targets."""

    def test_scheduling_overhead_small_graph(self):
        """Test scheduling overhead for small graph (<10 tasks)."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        # Add 10 tasks with dependencies
        for i in range(10):
            deps = [f"task_{i-1}"] if i > 0 else []
            scheduler.add_task(f"task_{i}", dummy_executor, dependencies=deps)

        start = time.time()
        scheduler.schedule()
        duration_ms = (time.time() - start) * 1000

        # Target: <10ms for small graphs
        assert duration_ms < 10, f"Scheduling took {duration_ms}ms, exceeds 10ms target"

    def test_scheduling_overhead_large_graph(self):
        """Test scheduling overhead for large graph (100 tasks)."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        # Add 100 tasks with complex dependencies
        for i in range(100):
            deps = [f"task_{j}" for j in range(i) if i % 5 == 0 and j < i]
            scheduler.add_task(f"task_{i}", dummy_executor, dependencies=deps)

        start = time.time()
        scheduler.schedule()
        duration_ms = (time.time() - start) * 1000

        # Target: <50ms for <100 tasks
        assert duration_ms < 50, f"Scheduling took {duration_ms}ms, exceeds 50ms target"

    @pytest.mark.asyncio
    async def test_parallel_speedup_target(self):
        """Test parallel speedup meets 2x target."""
        state = ThreadSafeState()
        scheduler = DependencyAwareScheduler(state=state)

        # Add 10 independent tasks (each takes 0.05s)
        for i in range(10):
            executor = await sleep_executor(0.05)
            scheduler.add_task(f"task_{i}", executor, dependencies=[])

        scheduler.schedule()
        result = await scheduler.execute()

        # Target: 2x speedup
        assert (
            result.speedup_ratio >= 2.0
        ), f"Speedup {result.speedup_ratio}x below 2.0x target"
