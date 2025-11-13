"""
Pydantic models for dependency-aware parallel scheduler.

This module provides type-safe data models for task scheduling, wave execution,
and result tracking with ONEX v2.0 compliance.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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


class Task(BaseModel):
    """
    Task specification for scheduler.

    Attributes:
        task_id: Unique task identifier
        executor: Async function to execute (receives shared context)
        dependencies: List of task_ids that must complete first
        timeout_seconds: Task execution timeout
        retry_count: Number of retries remaining
        metadata: Optional task metadata
        status: Current task status
        duration_ms: Execution duration (populated after execution)
        error: Error message if failed
        created_at: Task creation timestamp
        result: Task execution result (populated after execution)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str = Field(..., description="Unique task identifier", min_length=1)
    executor: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] = Field(
        ..., description="Async function to execute"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Task IDs that must complete first"
    )
    timeout_seconds: float = Field(
        default=300.0, description="Task execution timeout in seconds", gt=0
    )
    retry_count: int = Field(default=0, description="Number of retries remaining", ge=0)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional task metadata"
    )
    status: EnumTaskStatus = Field(
        default=EnumTaskStatus.PENDING, description="Current task status"
    )
    duration_ms: Optional[float] = Field(
        default=None, description="Execution duration in milliseconds"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Task creation timestamp"
    )
    result: Optional[dict[str, Any]] = Field(
        default=None, description="Task execution result"
    )


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

    wave_number: int = Field(..., description="Sequential wave number", ge=1)
    task_ids: list[str] = Field(..., description="Task IDs in this wave", min_length=1)
    status: EnumTaskStatus = Field(
        default=EnumTaskStatus.PENDING, description="Current wave status"
    )
    duration_ms: Optional[float] = Field(
        default=None, description="Wave execution duration in milliseconds"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Wave creation timestamp"
    )


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
    total_tasks: int = Field(..., description="Total number of tasks", ge=0)
    successful_tasks: int = Field(..., description="Successful task count", ge=0)
    failed_tasks: int = Field(..., description="Failed task count", ge=0)
    total_waves: int = Field(..., description="Total number of waves", ge=0)
    total_duration_ms: float = Field(..., description="Total execution time (ms)", ge=0)
    sequential_duration_ms: float = Field(
        ..., description="Estimated sequential execution time (ms)", ge=0
    )
    speedup_ratio: float = Field(
        ..., description="Parallel speedup ratio (sequential / parallel)", ge=0
    )
    task_results: dict[str, Any] = Field(
        default_factory=dict, description="Task ID to result mapping"
    )
    task_errors: dict[str, str] = Field(
        default_factory=dict, description="Task ID to error message mapping"
    )
    wave_summary: list[dict[str, Any]] = Field(
        default_factory=list, description="Summary of each wave execution"
    )
