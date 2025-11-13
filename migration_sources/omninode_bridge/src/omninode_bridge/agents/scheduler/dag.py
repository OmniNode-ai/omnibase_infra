"""
Dependency graph operations for parallel scheduler.

This module provides directed acyclic graph (DAG) operations including:
- Cycle detection via DFS
- Topological sorting via Kahn's algorithm
- Wave construction for parallel execution
- Graph validation
"""

import logging
from typing import Optional

from .exceptions import CircularDependencyError, TaskNotFoundError

logger = logging.getLogger(__name__)


class DependencyGraph:
    """
    Directed acyclic graph (DAG) for task dependencies.

    Features:
    - Cycle detection via DFS
    - Topological sorting for execution order
    - Wave construction for parallel execution
    - Dependency validation

    Example:
        ```python
        graph = DependencyGraph()
        graph.add_task("task_a", dependencies=[])
        graph.add_task("task_b", dependencies=["task_a"])
        graph.add_task("task_c", dependencies=["task_a"])

        # Detect cycles
        has_cycle = graph.has_cycle()  # Returns False

        # Topological sort
        sorted_tasks = graph.topological_sort()  # ["task_a", "task_b", "task_c"]

        # Wave construction
        waves = graph.construct_waves()  # [["task_a"], ["task_b", "task_c"]]
        ```
    """

    def __init__(self) -> None:
        """Initialize empty dependency graph."""
        # Adjacency list: task_id → list of dependencies
        self._graph: dict[str, list[str]] = {}

        # Reverse adjacency list: task_id → list of dependents
        self._reverse_graph: dict[str, list[str]] = {}

    def add_task(self, task_id: str, dependencies: list[str]) -> None:
        """
        Add task to dependency graph.

        Args:
            task_id: Task identifier
            dependencies: List of task IDs this task depends on

        Raises:
            ValueError: If task_id already exists
        """
        if task_id in self._graph:
            raise ValueError(f"Task '{task_id}' already exists in graph")

        self._graph[task_id] = dependencies.copy()

        # Update reverse graph
        if task_id not in self._reverse_graph:
            self._reverse_graph[task_id] = []

        for dep in dependencies:
            if dep not in self._reverse_graph:
                self._reverse_graph[dep] = []
            self._reverse_graph[dep].append(task_id)

    def validate(self) -> None:
        """
        Validate dependency graph.

        Validates that all dependencies reference existing tasks.

        Raises:
            TaskNotFoundError: If dependency references non-existent task
        """
        all_task_ids = set(self._graph.keys())

        for task_id, dependencies in self._graph.items():
            for dep in dependencies:
                if dep not in all_task_ids:
                    raise TaskNotFoundError(
                        task_id=task_id,
                        missing_dependency=dep,
                        available_tasks=list(all_task_ids),
                    )

    def has_cycle(self) -> bool:
        """
        Detect cycles in dependency graph via DFS.

        Uses DFS with state tracking:
        - WHITE (0): Unvisited
        - GRAY (1): Currently in DFS stack (visiting)
        - BLACK (2): Fully processed

        If GRAY node encountered → cycle exists.

        Returns:
            True if cycle exists, False otherwise

        Complexity: O(V + E) where V=tasks, E=dependencies
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        node_state = {task_id: WHITE for task_id in self._graph}

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
            for dep in self._graph.get(node, []):
                if dfs_visit(dep):
                    return True

            # Mark as fully processed
            node_state[node] = BLACK
            return False

        # Check all nodes (handles disconnected components)
        for task_id in self._graph:
            if node_state[task_id] == WHITE:
                if dfs_visit(task_id):
                    return True

        return False

    def find_cycle_path(self) -> Optional[list[str]]:
        """
        Find and return the circular dependency path.

        Returns:
            List of task_ids forming the cycle, or None if no cycle

        Example:
            ["task_a", "task_b", "task_c", "task_a"]
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        node_state = {task_id: WHITE for task_id in self._graph}
        path: list[str] = []
        cycle: Optional[list[str]] = None

        def dfs_visit(node: str) -> bool:
            """Visit node and track path."""
            nonlocal cycle

            if node_state[node] == GRAY:
                # Cycle detected - construct cycle path
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                return True

            if node_state[node] == BLACK:
                return False

            node_state[node] = GRAY
            path.append(node)

            for dep in self._graph.get(node, []):
                if dfs_visit(dep):
                    return True

            path.pop()
            node_state[node] = BLACK
            return False

        for task_id in self._graph:
            if node_state[task_id] == WHITE:
                if dfs_visit(task_id):
                    return cycle

        return None

    def topological_sort(self) -> list[str]:
        """
        Perform topological sort via Kahn's algorithm.

        Returns:
            List of task IDs in topologically sorted order

        Raises:
            CircularDependencyError: If cycle detected

        Complexity: O(V + E)
        """
        # Calculate in-degree for each node (number of dependencies)
        in_degree = {task_id: len(deps) for task_id, deps in self._graph.items()}

        # Initialize queue with nodes having in-degree 0 (no dependencies)
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        sorted_tasks: list[str] = []

        while queue:
            # Process next task with no dependencies
            current = queue.pop(0)
            sorted_tasks.append(current)

            # Reduce in-degree of dependents (tasks that depend on current)
            for dependent in self._reverse_graph.get(current, []):
                in_degree[dependent] -= 1

                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # If not all tasks processed → cycle exists
        if len(sorted_tasks) != len(self._graph):
            cycle_path = self.find_cycle_path()
            raise CircularDependencyError(cycle_path=cycle_path)

        return sorted_tasks

    def construct_waves(self) -> list[list[str]]:
        """
        Construct execution waves via topological sort with level grouping.

        Groups tasks into waves where all tasks in a wave can execute in parallel
        (no inter-dependencies). Waves execute sequentially.

        Returns:
            List of waves, where each wave is a list of task IDs

        Raises:
            CircularDependencyError: If cycle detected

        Complexity: O(V + E)

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
        # Calculate in-degree for each node
        in_degree = {task_id: 0 for task_id in self._graph}

        for task_id, dependencies in self._graph.items():
            in_degree[task_id] = len(dependencies)

        # Initialize first wave (tasks with no dependencies)
        current_wave = [task_id for task_id, degree in in_degree.items() if degree == 0]

        waves: list[list[str]] = []
        processed_tasks: set[str] = set()

        while current_wave:
            waves.append(current_wave.copy())
            processed_tasks.update(current_wave)
            next_wave: list[str] = []

            # Process each task in current wave
            for task_id in current_wave:
                # Reduce in-degree of dependents
                for dependent in self._reverse_graph.get(task_id, []):
                    in_degree[dependent] -= 1

                    # If all dependencies satisfied, add to next wave
                    if in_degree[dependent] == 0 and dependent not in processed_tasks:
                        next_wave.append(dependent)

            current_wave = next_wave

        # Validate all tasks processed
        if len(processed_tasks) != len(self._graph):
            cycle_path = self.find_cycle_path()
            raise CircularDependencyError(cycle_path=cycle_path)

        return waves

    def get_independent_tasks(self) -> list[str]:
        """
        Get all tasks with no dependencies.

        Returns:
            List of task IDs with no dependencies
        """
        return [task_id for task_id, deps in self._graph.items() if not deps]

    def get_task_level(self, task_id: str) -> int:
        """
        Get execution level (depth) of task in dependency graph.

        Level 0: No dependencies
        Level N: Depends on tasks at level N-1

        Args:
            task_id: Task to get level for

        Returns:
            Task execution level

        Raises:
            ValueError: If task_id not in graph
        """
        if task_id not in self._graph:
            raise ValueError(f"Task '{task_id}' not in graph")

        # Calculate levels via BFS
        levels: dict[str, int] = {}
        queue = [(tid, 0) for tid in self.get_independent_tasks()]

        while queue:
            current, level = queue.pop(0)

            if current in levels:
                levels[current] = max(levels[current], level)
            else:
                levels[current] = level

            for dependent in self._reverse_graph.get(current, []):
                queue.append((dependent, level + 1))

        return levels.get(task_id, 0)

    def __len__(self) -> int:
        """Get number of tasks in graph."""
        return len(self._graph)

    def __contains__(self, task_id: str) -> bool:
        """Check if task is in graph."""
        return task_id in self._graph

    def __repr__(self) -> str:
        """String representation of dependency graph."""
        return f"DependencyGraph(tasks={len(self._graph)})"
