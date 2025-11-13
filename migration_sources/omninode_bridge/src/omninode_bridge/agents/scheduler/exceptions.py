"""
Custom exceptions for dependency-aware parallel scheduler.

This module provides a hierarchy of typed exceptions for scheduler operations
with structured context for debugging and error handling.
"""

from typing import Optional


class SchedulerError(Exception):
    """Base exception for scheduler errors."""

    pass


class CircularDependencyError(SchedulerError):
    """
    Raised when circular dependencies are detected in task graph.

    Circular dependencies create deadlock where tasks wait indefinitely
    for each other to complete.

    Attributes:
        cycle_path: List of task IDs forming the cycle
    """

    def __init__(self, cycle_path: Optional[list[str]] = None):
        """
        Initialize CircularDependencyError.

        Args:
            cycle_path: Optional list of task IDs forming the cycle
        """
        self.cycle_path = cycle_path

        if cycle_path:
            path_str = " → ".join(cycle_path)
            message = f"Circular dependency detected: {path_str}"
        else:
            message = "Circular dependency detected in task graph"

        super().__init__(message)


class TaskNotFoundError(SchedulerError):
    """
    Raised when a task references a non-existent dependency.

    Attributes:
        task_id: Task that has the invalid dependency
        missing_dependency: Dependency that doesn't exist
        available_tasks: List of all available task IDs
    """

    def __init__(
        self,
        task_id: str,
        missing_dependency: str,
        available_tasks: Optional[list[str]] = None,
    ):
        """
        Initialize TaskNotFoundError.

        Args:
            task_id: Task with invalid dependency
            missing_dependency: Missing dependency ID
            available_tasks: List of available task IDs
        """
        self.task_id = task_id
        self.missing_dependency = missing_dependency
        self.available_tasks = available_tasks or []

        message = (
            f"Task '{task_id}' depends on non-existent task '{missing_dependency}'"
        )

        if available_tasks:
            if len(available_tasks) <= 10:
                tasks_display = ", ".join(available_tasks)
            else:
                tasks_display = ", ".join(available_tasks[:10]) + "..."
            message += f". Available tasks: {tasks_display}"

        super().__init__(message)


class WaveExecutionError(SchedulerError):
    """
    Raised when wave execution fails.

    Attributes:
        wave_number: Wave that failed
        failed_tasks: List of task IDs that failed
        error_details: Detailed error information per task
    """

    def __init__(
        self,
        wave_number: int,
        failed_tasks: list[str],
        error_details: Optional[dict[str, str]] = None,
    ):
        """
        Initialize WaveExecutionError.

        Args:
            wave_number: Wave number that failed
            failed_tasks: List of failed task IDs
            error_details: Detailed error messages per task
        """
        self.wave_number = wave_number
        self.failed_tasks = failed_tasks
        self.error_details = error_details or {}

        message = (
            f"Wave {wave_number} execution failed. "
            f"Failed tasks: {', '.join(failed_tasks)}"
        )

        super().__init__(message)


class SchedulingTimeoutError(SchedulerError):
    """
    Raised when scheduling or execution times out.

    Attributes:
        timeout_seconds: Timeout duration
        operation: Operation that timed out
    """

    def __init__(self, timeout_seconds: float, operation: str):
        """
        Initialize SchedulingTimeoutError.

        Args:
            timeout_seconds: Timeout duration
            operation: Operation that timed out
        """
        self.timeout_seconds = timeout_seconds
        self.operation = operation

        super().__init__(f"Timeout after {timeout_seconds}s for operation: {operation}")


class InvalidTaskError(SchedulerError):
    """
    Raised when task configuration is invalid.

    Attributes:
        task_id: Invalid task ID
        reason: Reason for invalidity
    """

    def __init__(self, task_id: str, reason: str):
        """
        Initialize InvalidTaskError.

        Args:
            task_id: Task ID
            reason: Reason for invalidity
        """
        self.task_id = task_id
        self.reason = reason

        super().__init__(f"Task '{task_id}' is invalid: {reason}")
