# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Agent Status Event Model.

This module defines the model for agent status events consumed from Kafka.
Agent status events provide real-time visibility into agent state, progress,
and current phase within a session.

Note:
    This is a local model for Kafka event deserialization. When omnibase_core
    ships EnumAgentState + ModelAgentStatus (OMN-1847), this model should be
    replaced with an import from omnibase_core.

Design Decisions:
    - frozen=True: Immutability for thread safety
    - extra="forbid": Strict validation ensures schema compliance
    - from_attributes=True: ORM/pytest-xdist compatibility
    - created_at: Required for TTL cleanup job (Phase 2)
    - progress: Bounded 0.0-1.0 for percentage representation

Idempotency:
    Table: agent_status_events
    Unique Key: id (UUID)
    Conflict Action: DO NOTHING (append-only audit log)

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> event = ModelAgentStatusEvent(
    ...     id=uuid4(),
    ...     correlation_id=uuid4(),
    ...     agent_name="polymorphic-agent",
    ...     session_id="session-abc123",
    ...     state="working",
    ...     message="Processing code review",
    ...     created_at=datetime.now(UTC),
    ... )
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelAgentStatusEvent(BaseModel):
    """Agent status event model for Kafka deserialization.

    Represents a single status report from an agent, including its current
    state, progress, and phase. Uses frozen=True for thread safety and
    extra="forbid" for strict schema compliance.

    Attributes:
        id: Unique event identifier (idempotency key).
        correlation_id: Request correlation ID for tracing.
        agent_name: Name of the reporting agent.
        session_id: Session identifier for grouping status events.
        state: Current agent state (idle, working, blocked, etc.).
        status_schema_version: Schema version for forward compatibility.
        message: Human-readable status message.
        progress: Optional progress indicator 0.0-1.0.
        current_phase: Optional current workflow phase name.
        current_task: Optional current task description.
        blocking_reason: Optional reason for blocked state.
        created_at: Timestamp when event was created (TTL key).
        metadata: Additional metadata about the status event.

    Example:
        >>> event = ModelAgentStatusEvent(
        ...     id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     agent_name="code-reviewer",
        ...     session_id="session-xyz",
        ...     state="working",
        ...     message="Reviewing PR #42",
        ...     progress=0.5,
        ...     current_phase="analysis",
        ...     created_at=datetime.now(UTC),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # ---- Required Fields ----
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique event identifier (idempotency key).",
    )
    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID for tracing.",
    )
    agent_name: str = Field(  # ONEX_EXCLUDE: entity_reference - external payload
        ...,
        description="Name of the reporting agent.",
    )
    session_id: str = Field(  # ONEX_EXCLUDE: string_id - external session identifier
        ...,
        description="Session identifier for grouping status events.",
    )
    state: str = Field(
        ...,
        description="Current agent state (idle, working, blocked, etc.).",
    )
    message: str = Field(
        ...,
        description="Human-readable status message.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when event was created (TTL key).",
    )

    # ---- Optional Fields ----
    status_schema_version: int = Field(
        default=1,
        description="Schema version for forward compatibility.",
    )
    progress: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Progress indicator 0.0-1.0.",
    )
    current_phase: str | None = Field(
        default=None,
        description="Current workflow phase name.",
    )
    current_task: str | None = Field(
        default=None,
        description="Current task description.",
    )
    blocking_reason: str | None = Field(
        default=None,
        description="Reason for blocked state.",
    )
    metadata: dict[str, JsonType] | None = Field(
        default=None,
        description="Additional metadata about the status event.",
    )

    def __str__(self) -> str:
        """Return concise string representation for logging.

        Includes key identifying fields but excludes metadata.
        """
        id_short = str(self.id)[:8]
        progress_part = (
            f", progress={self.progress}" if self.progress is not None else ""
        )
        phase_part = f", phase={self.current_phase}" if self.current_phase else ""
        return (
            f"AgentStatusEvent(id={id_short}, agent={self.agent_name}, "
            f"state={self.state}{progress_part}{phase_part})"
        )


__all__ = ["ModelAgentStatusEvent"]
