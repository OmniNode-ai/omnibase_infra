"""
Workflow models for staged parallel execution.

This module provides models for 6-phase code generation pipelines with
parallel contract processing.

Performance targets:
- Full pipeline: <5s for typical contract
- Stage transition overhead: <100ms
- Parallelism speedup: 2.25x-4.17x
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID, uuid4


class EnumStageStatus(str, Enum):
    """Stage execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EnumStepType(str, Enum):
    """Step types for code generation workflow."""

    PARSE_CONTRACT = "parse_contract"
    GENERATE_MODEL = "generate_model"
    GENERATE_VALIDATOR = "generate_validator"
    GENERATE_TEST = "generate_test"
    VALIDATE_QUALITY = "validate_quality"
    PACKAGE_NODE = "package_node"


@dataclass
class WorkflowStep:
    """
    Individual workflow step with dependencies.

    A step represents a single unit of work within a stage.
    Steps can depend on other steps, and are executed in dependency order.

    Attributes:
        step_id: Unique step identifier
        step_type: Type of step (parse, generate_model, etc.)
        dependencies: List of step IDs this step depends on
        input_data: Input data for step execution
        executor: Async function that executes the step
        status: Current execution status
        result: Step execution result (after completion)
        error: Error message (if failed)
        duration_ms: Execution duration in milliseconds
        created_at: Step creation timestamp
        started_at: Execution start timestamp
        completed_at: Execution completion timestamp
        retry_count: Number of retries on failure
        metadata: Additional metadata

    Example:
        ```python
        step = WorkflowStep(
            step_id="parse_contract_1",
            step_type=EnumStepType.PARSE_CONTRACT,
            dependencies=[],
            input_data={"contract_path": "path/to/contract.yaml"},
            executor=parse_contract_async
        )
        ```
    """

    step_id: str
    step_type: EnumStepType
    dependencies: list[str]
    input_data: dict[str, Any]
    executor: Callable[[dict[str, Any]], Any]  # Async callable
    status: EnumStageStatus = EnumStageStatus.PENDING
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate step configuration."""
        if not self.step_id:
            raise ValueError("step_id cannot be empty")
        if not callable(self.executor):
            raise ValueError(f"executor must be callable for step '{self.step_id}'")


@dataclass
class WorkflowStage:
    """
    Workflow stage with parallel step execution.

    A stage represents a logical phase in the workflow (e.g., parsing, model generation).
    All steps within a stage can be executed in parallel if stage.parallel=True.

    Attributes:
        stage_id: Unique stage identifier
        stage_number: Stage sequence number (1-based)
        stage_name: Human-readable stage name
        steps: List of steps in this stage
        dependencies: List of stage IDs this stage depends on
        parallel: Execute steps in parallel (default: True)
        status: Current execution status
        duration_ms: Execution duration in milliseconds
        created_at: Stage creation timestamp
        started_at: Execution start timestamp
        completed_at: Execution completion timestamp
        metadata: Additional metadata

    Example:
        ```python
        stage = WorkflowStage(
            stage_id="stage_parse",
            stage_number=1,
            stage_name="Parse Contracts",
            steps=[step1, step2, step3],
            dependencies=[],  # First stage has no dependencies
            parallel=True  # Execute all steps in parallel
        )
        ```
    """

    stage_id: str
    stage_number: int
    stage_name: str
    steps: list[WorkflowStep]
    dependencies: list[str] = field(default_factory=list)
    parallel: bool = True
    status: EnumStageStatus = EnumStageStatus.PENDING
    duration_ms: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate stage configuration."""
        if not self.stage_id:
            raise ValueError("stage_id cannot be empty")
        if self.stage_number < 1:
            raise ValueError(f"stage_number must be >= 1, got {self.stage_number}")
        if not self.steps:
            raise ValueError(f"stage '{self.stage_id}' must have at least one step")

    @property
    def total_steps(self) -> int:
        """Get total number of steps in stage."""
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        """Get number of completed steps."""
        return sum(1 for step in self.steps if step.status == EnumStageStatus.COMPLETED)

    @property
    def failed_steps(self) -> int:
        """Get number of failed steps."""
        return sum(1 for step in self.steps if step.status == EnumStageStatus.FAILED)

    @property
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return all(
            step.status in (EnumStageStatus.COMPLETED, EnumStageStatus.SKIPPED)
            for step in self.steps
        )

    @property
    def has_failures(self) -> bool:
        """Check if any steps failed."""
        return any(step.status == EnumStageStatus.FAILED for step in self.steps)


@dataclass
class StepResult:
    """
    Result of step execution.

    Attributes:
        step_id: Step identifier
        status: Execution status
        result: Step result data
        error: Error message (if failed)
        duration_ms: Execution duration
        retry_attempts: Number of retry attempts made
        metadata: Additional metadata

    Example:
        ```python
        result = StepResult(
            step_id="parse_contract_1",
            status=EnumStageStatus.COMPLETED,
            result={"parsed_data": {...}},
            duration_ms=234.5
        )
        ```
    """

    step_id: str
    status: EnumStageStatus
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retry_attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """
    Result of stage execution.

    Attributes:
        stage_id: Stage identifier
        stage_number: Stage sequence number
        status: Execution status
        step_results: Results for all steps in stage
        duration_ms: Execution duration
        total_steps: Total number of steps
        successful_steps: Number of successful steps
        failed_steps: Number of failed steps
        speedup_ratio: Parallel speedup achieved (sequential / parallel)
        metadata: Additional metadata

    Example:
        ```python
        result = StageResult(
            stage_id="stage_parse",
            stage_number=1,
            status=EnumStageStatus.COMPLETED,
            step_results={...},
            duration_ms=1234.5,
            total_steps=10,
            successful_steps=10,
            failed_steps=0,
            speedup_ratio=2.5
        )
        ```
    """

    stage_id: str
    stage_number: int
    status: EnumStageStatus
    step_results: dict[str, StepResult]
    duration_ms: float
    total_steps: int
    successful_steps: int
    failed_steps: int
    speedup_ratio: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """
    Result of complete workflow execution.

    Attributes:
        workflow_id: Workflow identifier
        status: Overall execution status
        stage_results: Results for all stages
        total_duration_ms: Total execution duration
        total_stages: Total number of stages
        successful_stages: Number of successful stages
        failed_stages: Number of failed stages
        total_steps: Total number of steps across all stages
        successful_steps: Total number of successful steps
        failed_steps: Total number of failed steps
        overall_speedup: Overall parallel speedup achieved
        created_at: Workflow creation timestamp
        started_at: Execution start timestamp
        completed_at: Execution completion timestamp
        metadata: Additional metadata

    Example:
        ```python
        result = WorkflowResult(
            workflow_id="codegen-session-1",
            status=EnumStageStatus.COMPLETED,
            stage_results=[...],
            total_duration_ms=4567.8,
            total_stages=6,
            successful_stages=6,
            failed_stages=0,
            total_steps=50,
            successful_steps=50,
            failed_steps=0,
            overall_speedup=2.3
        )
        ```
    """

    workflow_id: str
    status: EnumStageStatus
    stage_results: list[StageResult]
    total_duration_ms: float
    total_stages: int
    successful_stages: int
    failed_stages: int
    total_steps: int
    successful_steps: int
    failed_steps: int
    overall_speedup: float
    created_at: datetime
    started_at: datetime
    completed_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_steps == 0:
            return 0.0
        return self.successful_steps / self.total_steps

    @property
    def stage_success_rate(self) -> float:
        """Calculate stage-level success rate."""
        if self.total_stages == 0:
            return 0.0
        return self.successful_stages / self.total_stages


@dataclass
class WorkflowConfig:
    """
    Workflow configuration.

    Attributes:
        workflow_id: Unique workflow identifier
        workflow_name: Human-readable workflow name
        max_concurrent_steps: Maximum concurrent steps per stage
        enable_stage_recovery: Enable stage-level recovery on failures
        enable_step_retry: Enable automatic step retries
        step_retry_count: Number of retries per step
        stage_timeout_seconds: Timeout per stage (None = no timeout)
        step_timeout_seconds: Timeout per step (None = no timeout)
        collect_metrics: Enable metrics collection
        signal_on_stage_complete: Send signals on stage completion
        metadata: Additional configuration metadata

    Example:
        ```python
        config = WorkflowConfig(
            workflow_id="codegen-session-1",
            workflow_name="Contract Code Generation",
            max_concurrent_steps=10,
            enable_stage_recovery=True,
            enable_step_retry=True,
            step_retry_count=2
        )
        ```
    """

    workflow_id: str
    workflow_name: str
    max_concurrent_steps: int = 10
    enable_stage_recovery: bool = True
    enable_step_retry: bool = True
    step_retry_count: int = 2
    stage_timeout_seconds: Optional[float] = None
    step_timeout_seconds: Optional[float] = 300.0  # 5 minutes default
    collect_metrics: bool = True
    signal_on_stage_complete: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.workflow_id:
            raise ValueError("workflow_id cannot be empty")
        if self.max_concurrent_steps < 1:
            raise ValueError(
                f"max_concurrent_steps must be >= 1, got {self.max_concurrent_steps}"
            )
        if self.step_retry_count < 0:
            raise ValueError(
                f"step_retry_count must be >= 0, got {self.step_retry_count}"
            )
