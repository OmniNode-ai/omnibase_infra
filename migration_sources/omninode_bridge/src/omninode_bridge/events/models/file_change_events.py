"""
File change event models for git hook automation.

This module defines events published by git hooks when file changes are detected
during git operations (push, commit, etc.). These events trigger automated
metadata stamping workflows in the omninode ecosystem.

Pattern:
    Git Hook → Kafka (file_changes topic) → Stamping Workflow

Example:
    # Git pre-push hook publishes file change event
    event = ModelFileChangeEvent(
        files=["src/module.py", "tests/test_module.py"],
        repo_name="omninode_bridge",
        branch="main",
        operation="pre-push",
    )
    await publish_to_kafka(TOPIC_FILE_CHANGES, event)

ONEX v2.0 Compliance:
- Suffix-based naming: ModelFileChangeEvent
- Strong typing with Pydantic v2
- Strict validation with extra="forbid"
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field

from .base import EventBase

# Kafka topic for file change events
TOPIC_FILE_CHANGES = "dev.omninode_bridge.onex.evt.file-changes.v1"


class EnumGitOperation(str, Enum):
    """Git operation types that trigger file change events."""

    PRE_PUSH = "pre-push"
    PRE_COMMIT = "pre-commit"
    POST_COMMIT = "post-commit"
    POST_MERGE = "post-merge"


class ModelFileChangeEvent(EventBase):
    """
    Event published when file changes are detected by git hooks.

    This event triggers automated metadata stamping workflows for changed files.
    Published by git hooks with minimal overhead (<2s execution time requirement).

    Attributes:
        event_type: Always "FILE_CHANGE_DETECTED" for this event type
        event_id: Unique identifier for this event
        correlation_id: Correlation ID for tracing through workflow
        timestamp: When event was created (UTC)
        files: List of changed file paths (relative to repo root)
        repo_name: Repository name (e.g., "omninode_bridge")
        repo_path: Absolute path to repository root
        branch: Git branch name
        operation: Git operation that triggered this event
        commit_sha: Git commit SHA (if available)
        author_name: Git author name (if available)
        author_email: Git author email (if available)
    """

    # Event metadata
    event_type: Literal["FILE_CHANGE_DETECTED"] = Field(
        default="FILE_CHANGE_DETECTED",
        description="Event type identifier",
    )
    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for tracing through workflow",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When event was created (UTC)",
    )

    # File change details
    files: list[str] = Field(
        ...,
        description="List of changed file paths (relative to repo root)",
        min_length=1,
        examples=[["src/module.py", "tests/test_module.py"]],
    )
    repo_name: str = Field(
        ...,
        description="Repository name (e.g., 'omninode_bridge')",
        examples=["omninode_bridge", "omnibase_core"],
    )
    repo_path: str = Field(
        ...,
        description="Absolute path to repository root",
        examples=["/Users/jonah/Code/omninode_bridge"],
    )
    branch: str = Field(
        ...,
        description="Git branch name",
        examples=["main", "feature/stamping-automation", "mvp_requirement_completion"],
    )
    operation: EnumGitOperation = Field(
        ...,
        description="Git operation that triggered this event",
    )

    # Optional git metadata
    commit_sha: str | None = Field(
        default=None,
        description="Git commit SHA (if available)",
        examples=["b059427abc123def456"],  # pragma: allowlist secret
    )
    author_name: str | None = Field(
        default=None,
        description="Git author name (if available)",
    )
    author_email: str | None = Field(
        default=None,
        description="Git author email (if available)",
    )

    def to_kafka_topic(self) -> str:
        """Generate Kafka topic name for this event."""
        return TOPIC_FILE_CHANGES

    def to_dict(self) -> dict:
        """Convert event to dictionary for Kafka publishing."""
        return self.model_dump(mode="json")


class ModelFileChangeProcessingResult(EventBase):
    """
    Result of processing file change events through stamping workflow.

    Published after metadata stamping workflow completes processing of file changes.

    Attributes:
        event_id: Unique identifier for this result event
        correlation_id: Correlation ID from original file change event
        original_event_id: Reference to original file change event
        processed_at: When processing completed (UTC)
        success: Whether processing succeeded
        files_processed: Number of files successfully processed
        files_failed: Number of files that failed processing
        error_message: Error message if processing failed
        processing_duration_ms: How long processing took
        stamps_created: Number of metadata stamps created
    """

    # Event metadata
    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this result event",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID from original file change event",
    )
    original_event_id: UUID = Field(
        ...,
        description="Reference to original file change event ID",
    )
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When processing completed (UTC)",
    )

    # Processing results
    success: bool = Field(
        ...,
        description="Whether processing succeeded",
    )
    files_processed: int = Field(
        ...,
        ge=0,
        description="Number of files successfully processed",
    )
    files_failed: int = Field(
        default=0,
        ge=0,
        description="Number of files that failed processing",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if processing failed",
    )
    processing_duration_ms: float | None = Field(
        default=None,
        ge=0,
        description="How long processing took in milliseconds",
    )
    stamps_created: int = Field(
        default=0,
        ge=0,
        description="Number of metadata stamps created",
    )

    def to_kafka_topic(self) -> str:
        """Generate Kafka topic name for this event."""
        return "dev.omninode_bridge.onex.evt.file-change-processing-result.v1"

    def to_dict(self) -> dict:
        """Convert event to dictionary for Kafka publishing."""
        return self.model_dump(mode="json")
