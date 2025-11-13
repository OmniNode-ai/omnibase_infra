#!/usr/bin/env python3
"""
Pydantic models for GitHub Actions workflows.

This module provides comprehensive Pydantic models for representing and validating
GitHub Actions workflow files, enabling type-safe workflow generation and validation.
"""

from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(str, Enum):
    """GitHub Actions event types."""

    PUSH = "push"
    PULL_REQUEST = "pull_request"
    PULL_REQUEST_REVIEW = "pull_request_review"
    PULL_REQUEST_REVIEW_COMMENT = "pull_request_review_comment"
    ISSUE_COMMENT = "issue_comment"
    ISSUES = "issues"
    SCHEDULE = "schedule"
    WORKFLOW_DISPATCH = "workflow_dispatch"
    WORKFLOW_RUN = "workflow_run"
    WORKFLOW_CALL = "workflow_call"
    REPOSITORY_DISPATCH = "repository_dispatch"
    RELEASE = "release"
    CREATE = "create"
    DELETE = "delete"
    FORK = "fork"
    GOLLUM = "gollum"
    PAGE_BUILD = "page_build"
    PUBLIC = "public"
    WATCH = "watch"
    DISCUSSION = "discussion"
    DISCUSSION_COMMENT = "discussion_comment"


class PermissionLevel(str, Enum):
    """Permission levels for GitHub Actions."""

    READ = "read"
    WRITE = "write"
    NONE = "none"


class PermissionSet(BaseModel):
    """GitHub Actions permissions configuration."""

    actions: Optional[PermissionLevel] = None
    checks: Optional[PermissionLevel] = None
    contents: Optional[PermissionLevel] = None
    deployments: Optional[PermissionLevel] = None
    id_token: Optional[PermissionLevel] = Field(None, alias="id-token")
    issues: Optional[PermissionLevel] = None
    metadata: Optional[PermissionLevel] = None
    packages: Optional[PermissionLevel] = None
    pages: Optional[PermissionLevel] = None
    pull_requests: Optional[PermissionLevel] = Field(None, alias="pull-requests")
    repository_projects: Optional[PermissionLevel] = Field(
        None, alias="repository-projects"
    )
    security_events: Optional[PermissionLevel] = Field(None, alias="security-events")
    statuses: Optional[PermissionLevel] = None

    model_config = ConfigDict(populate_by_name=True)


class WorkflowStep(BaseModel):
    """A single step in a GitHub Actions job."""

    name: str
    uses: Optional[str] = None
    run: Optional[str] = None
    with_: Optional[dict[str, Any]] = Field(None, alias="with")
    env: Optional[dict[str, str]] = None
    if_: Optional[str] = Field(None, alias="if")
    continue_on_error: Optional[bool] = Field(None, alias="continue-on-error")
    timeout_minutes: Optional[int] = Field(None, alias="timeout-minutes")
    working_directory: Optional[str] = Field(None, alias="working-directory")
    shell: Optional[str] = None
    id: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("uses")
    @classmethod
    def validate_uses(cls, v, info):
        """Ensure either 'uses' or 'run' is provided."""
        if not v and not info.data.get("run"):
            raise ValueError("Either 'uses' or 'run' must be provided")
        return v

    @field_validator("run")
    @classmethod
    def validate_run(cls, v, info):
        """Ensure either 'uses' or 'run' is provided."""
        if not v and not info.data.get("uses"):
            raise ValueError("Either 'uses' or 'run' must be provided")
        return v


class MatrixStrategy(BaseModel):
    """Matrix strategy for job execution."""

    matrix: dict[str, list[Any]]
    fail_fast: Optional[bool] = Field(True, alias="fail-fast")
    max_parallel: Optional[int] = Field(None, alias="max-parallel")
    include: Optional[list[dict[str, Any]]] = None
    exclude: Optional[list[dict[str, Any]]] = None

    model_config = ConfigDict(populate_by_name=True)


class JobStrategy(BaseModel):
    """Job execution strategy configuration."""

    matrix: Optional[MatrixStrategy] = None
    fail_fast: Optional[bool] = Field(True, alias="fail-fast")
    max_parallel: Optional[int] = Field(None, alias="max-parallel")

    model_config = ConfigDict(populate_by_name=True)


class WorkflowJob(BaseModel):
    """A job in a GitHub Actions workflow."""

    name: Optional[str] = None
    runs_on: Union[str, list[str]] = Field(alias="runs-on")
    steps: list[WorkflowStep]
    needs: Optional[Union[str, list[str]]] = None
    if_: Optional[str] = Field(None, alias="if")
    permissions: Optional[Union[PermissionSet, str]] = None
    environment: Optional[Union[str, dict[str, Any]]] = None
    concurrency: Optional[Union[str, dict[str, Any]]] = None
    outputs: Optional[dict[str, str]] = None
    env: Optional[dict[str, str]] = None
    defaults: Optional[dict[str, Any]] = None
    strategy: Optional[JobStrategy] = None
    timeout_minutes: Optional[int] = Field(None, alias="timeout-minutes")
    continue_on_error: Optional[bool] = Field(None, alias="continue-on-error")
    container: Optional[Union[str, dict[str, Any]]] = None
    services: Optional[dict[str, dict[str, Any]]] = None

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("steps")
    @classmethod
    def steps_not_empty(cls, v):
        """Ensure at least one step is provided."""
        if not v:
            raise ValueError("At least one step must be provided")
        return v


class EventConfig(BaseModel):
    """Configuration for specific event types."""

    branches: Optional[list[str]] = None
    branches_ignore: Optional[list[str]] = Field(None, alias="branches-ignore")
    tags: Optional[list[str]] = None
    tags_ignore: Optional[list[str]] = Field(None, alias="tags-ignore")
    paths: Optional[list[str]] = None
    paths_ignore: Optional[list[str]] = Field(None, alias="paths-ignore")
    types: Optional[list[str]] = None

    model_config = ConfigDict(populate_by_name=True)


class ScheduleConfig(BaseModel):
    """Configuration for scheduled workflows."""

    cron: str


class WorkflowConfig(BaseModel):
    """Complete GitHub Actions workflow configuration."""

    name: str
    on: Union[
        str,
        list[str],
        dict[str, Union[EventConfig, list[ScheduleConfig], dict[str, Any]]],
    ]
    jobs: dict[str, WorkflowJob]
    permissions: Optional[Union[PermissionSet, str]] = None
    env: Optional[dict[str, str]] = None
    defaults: Optional[dict[str, Any]] = None
    concurrency: Optional[Union[str, dict[str, Any]]] = None
    run_name: Optional[str] = Field(None, alias="run-name")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("jobs")
    @classmethod
    def jobs_not_empty(cls, v):
        """Ensure at least one job is provided."""
        if not v:
            raise ValueError("At least one job must be provided")
        return v

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to YAML-compatible dictionary."""
        data = self.model_dump(by_alias=True, exclude_none=True)

        # Handle special cases for proper YAML output
        if isinstance(data.get("on"), list):
            # Convert list of events to proper format
            events = data["on"]
            if len(events) == 1:
                data["on"] = events[0]

        return data

    model_config = ConfigDict(populate_by_name=True)


class WorkflowTemplate(BaseModel):
    """Template for generating workflows."""

    name: str
    description: str
    template_type: str
    parameters: dict[str, Any] = {}

    def generate_workflow(self, **kwargs) -> WorkflowConfig:
        """Generate a workflow from this template."""
        # This would be implemented by specific template classes
        raise NotImplementedError("Subclasses must implement generate_workflow")


# Additional validation and utility functions
def validate_cron_expression(cron: str) -> bool:
    """Validate a cron expression format."""
    parts = cron.split()
    if len(parts) != 5:
        return False

    # Basic validation - more sophisticated validation could be added
    for part in parts:
        if not (
            part.isdigit() or part == "*" or "/" in part or "-" in part or "," in part
        ):
            return False

    return True


def validate_action_reference(action_ref: str) -> bool:
    """Validate GitHub Actions action reference format."""
    if action_ref.startswith("./"):
        # Local action
        return True

    if "@" not in action_ref:
        return False

    parts = action_ref.split("@")
    if len(parts) != 2:
        return False

    action_name, version = parts
    return "/" in action_name  # Should be owner/repo format


# Predefined runner configurations
GITHUB_HOSTED_RUNNERS = [
    "ubuntu-latest",
    "ubuntu-22.04",
    "ubuntu-20.04",
    "windows-latest",
    "windows-2022",
    "windows-2019",
    "macos-latest",
    "macos-13",
    "macos-12",
    "macos-11",
]


# Common action references
COMMON_ACTIONS = {
    "checkout": "actions/checkout@v4",
    "setup-python": "actions/setup-python@v5",
    "setup-node": "actions/setup-node@v4",
    "cache": "actions/cache@v4",
    "upload-artifact": "actions/upload-artifact@v4",
    "download-artifact": "actions/download-artifact@v4",
    "claude-code": "anthropics/claude-code-action@v1",
}
