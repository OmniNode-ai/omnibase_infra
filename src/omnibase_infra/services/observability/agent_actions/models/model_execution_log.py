# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Execution Log Model.

This module defines the model for agent execution log events consumed
from Kafka. Execution logs track the full lifecycle of an agent execution,
from start to completion (or failure).

Design Decisions:
    - extra="allow": Phase 1 flexibility - required fields typed, extras preserved
    - raw_payload: Optional field to preserve complete payload for schema tightening
    - created_at AND updated_at: Both required for lifecycle tracking and TTL
    - TTL keys off updated_at (not created_at) to avoid deleting in-flight executions

Idempotency:
    Table: agent_execution_logs
    Unique Key: execution_id (UUID)
    Conflict Action: DO UPDATE (lifecycle record - started -> completed)

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> log = ModelExecutionLog(
    ...     execution_id=uuid4(),
    ...     correlation_id=uuid4(),
    ...     agent_name="api-architect",
    ...     status="completed",
    ...     created_at=datetime.now(UTC),
    ...     updated_at=datetime.now(UTC),
    ... )
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelExecutionLog(BaseModel):
    """Agent execution log event model.

    Represents the lifecycle of an agent execution, from start to completion.
    Unlike other observability models, this supports upsert semantics to track
    status transitions (started -> running -> completed/failed).

    Attributes:
        execution_id: Unique identifier for this execution (idempotency key).
        correlation_id: Request correlation ID linking related events.
        agent_name: Name of the agent being executed.
        status: Current status of the execution (started, running, completed, failed).
        created_at: Timestamp when the execution started.
        updated_at: Timestamp of last status update (TTL key for lifecycle records).
        started_at: Optional explicit start timestamp.
        completed_at: Optional completion timestamp.
        duration_ms: Optional total duration in milliseconds.
        exit_code: Optional exit code if execution completed.
        error_message: Optional error message if execution failed.
        input_summary: Optional summary of execution input.
        output_summary: Optional summary of execution output.
        metadata: Optional additional metadata about the execution.
        raw_payload: Optional complete raw payload for Phase 2 schema tightening.

    Example:
        >>> log = ModelExecutionLog(
        ...     execution_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     agent_name="testing",
        ...     status="completed",
        ...     created_at=datetime.now(UTC),
        ...     updated_at=datetime.now(UTC),
        ...     duration_ms=5432,
        ...     exit_code=0,
        ... )
    """

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
    )

    # ---- Required Fields ----
    execution_id: UUID = Field(
        ...,
        description="Unique identifier for this execution (idempotency key).",
    )
    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID linking related events.",
    )
    agent_name: str = Field(  # ONEX_EXCLUDE: entity_reference - external payload
        ..., description="Name of the agent being executed."
    )
    status: str = Field(  # ONEX_EXCLUDE: string_status - external payload
        ..., description="Current status (started, running, completed, failed)."
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the execution started.",
    )
    updated_at: datetime = Field(
        ...,
        description="Timestamp of last status update (TTL key for lifecycle records).",
    )

    # ---- Optional Fields ----
    started_at: datetime | None = Field(
        default=None,
        description="Explicit start timestamp.",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Completion timestamp.",
    )
    duration_ms: int | None = Field(
        default=None,
        description="Total duration in milliseconds.",
    )
    exit_code: int | None = Field(
        default=None,
        description="Exit code if execution completed.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if execution failed.",
    )
    input_summary: str | None = Field(
        default=None,
        description="Summary of execution input.",
    )
    output_summary: str | None = Field(
        default=None,
        description="Summary of execution output.",
    )
    metadata: dict[str, JsonType] | None = Field(
        default=None,
        description="Additional metadata about the execution.",
    )
    raw_payload: dict[str, JsonType] | None = Field(
        default=None,
        description="Complete raw payload for Phase 2 schema tightening.",
    )


__all__ = ["ModelExecutionLog"]
