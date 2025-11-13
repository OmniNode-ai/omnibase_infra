"""Pydantic models for GitHub Actions workflows."""

from enum import Enum
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(str, Enum):
    """Supported GitHub workflow event types."""

    PUSH = "push"
    PULL_REQUEST = "pull_request"
    ISSUE_COMMENT = "issue_comment"
    PULL_REQUEST_REVIEW = "pull_request_review"
    PULL_REQUEST_REVIEW_COMMENT = "pull_request_review_comment"
    ISSUES = "issues"
    SCHEDULE = "schedule"
    WORKFLOW_DISPATCH = "workflow_dispatch"


class JobStatus(str, Enum):
    """Job execution status options."""

    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class PermissionLevel(str, Enum):
    """GitHub Actions permission levels."""

    READ = "read"
    WRITE = "write"
    NONE = "none"


class PermissionSet(BaseModel):
    """GitHub Actions permissions configuration."""

    contents: PermissionLevel | None = None
    pull_requests: PermissionLevel | None = None
    issues: PermissionLevel | None = None
    actions: PermissionLevel | None = None
    id_token: PermissionLevel | None = None
    checks: PermissionLevel | None = None
    statuses: PermissionLevel | None = None
    deployments: PermissionLevel | None = None
    security_events: PermissionLevel | None = None


class WorkflowStep(BaseModel):
    """Individual step in a workflow job."""

    name: str = Field(..., description="Step name")
    id: str | None = Field(default=None, description="Step identifier")
    uses: str | None = Field(default=None, description="Action to use")
    run: str | None = Field(default=None, description="Shell command to run")
    with_: dict[str, Any] | None = Field(
        default=None, alias="with", description="Action inputs"
    )
    env: dict[str, str] | None = Field(
        default=None, description="Environment variables"
    )
    if_: str | None = Field(
        default=None, alias="if", description="Conditional expression"
    )
    continue_on_error: bool | None = Field(
        default=None, alias="continue-on-error", description="Continue on step failure"
    )
    timeout_minutes: int | None = Field(
        default=None, alias="timeout-minutes", description="Step timeout in minutes"
    )
    shell: str | None = Field(default=None, description="Shell to use for run commands")
    working_directory: str | None = Field(default=None, alias="working-directory")

    @field_validator("uses", "run")
    @classmethod
    def validate_step_action(cls, v, info):
        """Ensure step has either 'uses' or 'run' but not both."""
        if info.field_name == "uses" and v is not None:
            # If 'uses' is provided, 'run' should not be
            return v
        elif info.field_name == "run" and v is not None:
            # If 'run' is provided, 'uses' should not be
            return v
        return v

    model_config = ConfigDict(populate_by_name=True)


class MatrixStrategy(BaseModel):
    """Matrix strategy for parallel job execution."""

    matrix: dict[str, list[Any]] = Field(..., description="Matrix variables")
    include: list[dict[str, Any]] | None = Field(
        default=None, description="Additional matrix combinations"
    )
    exclude: list[dict[str, Any]] | None = Field(
        default=None, description="Excluded matrix combinations"
    )
    fail_fast: bool | None = Field(
        default=None, alias="fail-fast", description="Fail fast on first failure"
    )
    max_parallel: int | None = Field(
        default=None, alias="max-parallel", description="Maximum parallel jobs"
    )

    @field_validator("matrix")
    @classmethod
    def validate_matrix_not_empty(cls, v):
        """Validate that matrix is not empty."""
        if not v:
            raise ValueError("Matrix strategy must have at least one variable")
        return v

    model_config = ConfigDict(populate_by_name=True)


class WorkflowJob(BaseModel):
    """GitHub Actions workflow job definition."""

    name: str | None = Field(default=None, description="Job display name")
    runs_on: Union[str, list[str]] = Field(
        ..., alias="runs-on", description="Runner specification"
    )
    needs: Union[str, list[str]] | None = Field(
        default=None, description="Job dependencies"
    )
    if_: str | None = Field(
        default=None, alias="if", description="Job conditional expression"
    )
    permissions: Union[PermissionSet, dict[str, str]] | None = Field(
        default=None, description="Job permissions"
    )
    environment: Union[str, dict[str, Any]] | None = Field(
        default=None, description="Deployment environment"
    )
    timeout_minutes: int | None = Field(
        default=None, alias="timeout-minutes", description="Job timeout"
    )
    strategy: MatrixStrategy | None = Field(default=None, description="Matrix strategy")
    container: Union[str, dict[str, Any]] | None = Field(
        default=None, description="Container specification"
    )
    services: dict[str, dict[str, Any]] | None = Field(
        default=None, description="Service containers"
    )
    steps: list[WorkflowStep] = Field(..., description="Job steps")
    outputs: dict[str, str] | None = Field(default=None, description="Job outputs")

    @field_validator("runs_on")
    @classmethod
    def validate_runs_on(cls, v):
        """Validate runner specification."""
        if isinstance(v, str):
            return v
        elif isinstance(v, list):
            if not v:  # Empty list is invalid
                raise ValueError("runs_on list cannot be empty")
            if all(isinstance(item, str) for item in v):
                return v
            else:
                raise ValueError("All items in runs_on list must be strings")
        else:
            raise ValueError("runs_on must be a string or list of strings")

    @field_validator("steps")
    @classmethod
    def validate_steps_not_empty(cls, v):
        """Validate that steps list is not empty."""
        if not v:
            raise ValueError("Job must have at least one step")
        return v

    model_config = ConfigDict(populate_by_name=True)


class WorkflowTrigger(BaseModel):
    """Workflow trigger configuration."""

    event_type: EventType
    types: list[str] | None = Field(None, description="Event activity types")
    branches: list[str] | None = Field(None, description="Branch filters")
    paths: list[str] | None = Field(None, description="Path filters")
    tags: list[str] | None = Field(None, description="Tag filters")
    schedule: list[dict[str, str]] | None = Field(None, description="Cron schedule")


class WorkflowConfig(BaseModel):
    """Complete GitHub Actions workflow configuration."""

    name: str = Field(..., description="Workflow name")
    on: Union[EventType, list[EventType], dict[str, Any]] = Field(
        ..., description="Workflow triggers"
    )
    env: dict[str, str] | None = Field(
        default=None, description="Global environment variables"
    )
    defaults: dict[str, Any] | None = Field(
        default=None, description="Default settings"
    )
    concurrency: Union[str, dict[str, Any]] | None = Field(
        default=None, description="Concurrency settings"
    )
    permissions: Union[PermissionSet, dict[str, str]] | None = Field(
        default=None, description="Workflow permissions"
    )
    jobs: dict[str, WorkflowJob] = Field(..., description="Workflow jobs")

    @field_validator("jobs")
    @classmethod
    def validate_jobs_not_empty(cls, v):
        """Ensure at least one job is defined."""
        if not v:
            raise ValueError("At least one job must be defined")
        return v

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to YAML-serializable dictionary."""
        data = self.model_dump(by_alias=True, exclude_none=True, mode="json")

        # Handle special field name conversions
        for job_id, job_data in data.get("jobs", {}).items():
            # Convert step 'with_' back to 'with'
            for step in job_data.get("steps", []):
                if "with_" in step:
                    step["with"] = step.pop("with_")
                if "if_" in step:
                    step["if"] = step.pop("if_")

            # Convert job 'if_' back to 'if'
            if "if_" in job_data:
                job_data["if"] = job_data.pop("if_")

        return data

    @classmethod
    def from_yaml_dict(cls, data: dict[str, Any]) -> "WorkflowConfig":
        """Create WorkflowConfig from YAML-loaded dictionary.

        Args:
            data: Dictionary loaded from YAML

        Returns:
            WorkflowConfig instance

        Raises:
            PydanticValidationError: If data is invalid
        """
        # Handle field name conversions from YAML format
        processed_data = data.copy()

        # Convert jobs data
        if "jobs" in processed_data:
            for job_id, job_data in processed_data["jobs"].items():
                # Convert job 'if' to 'if_'
                if "if" in job_data:
                    job_data["if_"] = job_data.pop("if")

                # Convert steps data
                if "steps" in job_data:
                    for step in job_data["steps"]:
                        # Convert step 'with' to 'with_'
                        if "with" in step:
                            step["with_"] = step.pop("with")
                        # Convert step 'if' to 'if_'
                        if "if" in step:
                            step["if_"] = step.pop("if")

        return cls(**processed_data)

    @classmethod
    def from_yaml_string(cls, yaml_content: str) -> "WorkflowConfig":
        """Create WorkflowConfig from YAML string.

        Args:
            yaml_content: YAML content as string

        Returns:
            WorkflowConfig instance

        Raises:
            yaml.YAMLError: If YAML is invalid
            PydanticValidationError: If data is invalid
        """
        data = yaml.safe_load(yaml_content)
        return cls.from_yaml_dict(data)

    @classmethod
    def from_yaml_file(cls, file_path: Union[str, Path]) -> "WorkflowConfig":
        """Create WorkflowConfig from YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            WorkflowConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
            PydanticValidationError: If data is invalid
        """
        file_path = Path(file_path)

        with open(file_path, encoding="utf-8") as f:
            yaml_content = f.read()

        return cls.from_yaml_string(yaml_content)

    def to_yaml_string(self) -> str:
        """Convert to YAML string.

        Returns:
            YAML string representation
        """
        from ..generators.workflow_generator import YAMLFormatter

        formatter = YAMLFormatter()
        return formatter.format_yaml(self.to_yaml_dict())

    def to_yaml_file(self, file_path: Union[str, Path]) -> None:
        """Save workflow to YAML file.

        Args:
            file_path: Output file path

        Raises:
            IOError: If file cannot be written
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml_string())

    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)
