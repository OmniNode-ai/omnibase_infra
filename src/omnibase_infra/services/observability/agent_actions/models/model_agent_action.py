# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Agent Action Model.

This module defines the model for agent action events consumed from Kafka.
Agent actions represent individual tool calls, decisions, errors, and
successes recorded during agent execution.

Design Decisions:
    - extra="allow": Phase 1 flexibility - required fields typed, extras preserved
    - raw_payload: Optional field to preserve complete payload for schema tightening
    - created_at: Required for TTL cleanup job (Phase 2)

Idempotency:
    Table: agent_actions
    Unique Key: id (UUID)
    Conflict Action: DO NOTHING (append-only audit log)

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> action = ModelAgentAction(
    ...     id=uuid4(),
    ...     correlation_id=uuid4(),
    ...     agent_name="polymorphic-agent",
    ...     action_type="tool_call",
    ...     action_name="Bash",
    ...     created_at=datetime.now(UTC),
    ... )
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelAgentAction(BaseModel):
    """Agent action event model.

    Represents a single action performed by an agent, such as a tool call,
    decision, error, or success. Uses extra="allow" for Phase 1 flexibility
    while ensuring required fields are typed.

    Attributes:
        id: Unique identifier for this action (idempotency key).
        correlation_id: Request correlation ID linking related actions.
        agent_name: Name of the agent that performed this action.
        action_type: Type of action (tool_call, decision, error, success).
        action_name: Specific name of the action or tool.
        created_at: Timestamp when the action was recorded (TTL key).
        status: Optional status of the action (started, completed, failed).
        duration_ms: Optional duration of the action in milliseconds.
        result: Optional result summary or outcome.
        error_message: Optional error message if action failed.
        metadata: Optional additional metadata about the action.
        raw_payload: Optional complete raw payload for Phase 2 schema tightening.

    Example:
        >>> action = ModelAgentAction(
        ...     id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     agent_name="code-reviewer",
        ...     action_type="decision",
        ...     action_name="approve_pr",
        ...     created_at=datetime.now(UTC),
        ...     status="completed",
        ...     duration_ms=1234,
        ... )
    """

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
    )

    # ---- Required Fields ----
    id: UUID = Field(
        ...,
        description="Unique identifier for this action (idempotency key).",
    )
    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID linking related actions.",
    )
    agent_name: str = Field(  # ONEX_EXCLUDE: entity_reference - external payload
        ..., description="Name of the agent that performed this action."
    )
    action_type: str = Field(
        ...,
        description="Type of action (tool_call, decision, error, success).",
    )
    action_name: str = Field(  # ONEX_EXCLUDE: entity_reference - external payload
        ..., description="Specific name of the action or tool."
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the action was recorded (TTL key).",
    )

    # ---- Optional Fields ----
    status: str | None = Field(
        default=None,
        description="Status of the action (started, completed, failed).",
    )
    duration_ms: int | None = Field(
        default=None,
        description="Duration of the action in milliseconds.",
    )
    result: str | None = Field(
        default=None,
        description="Result summary or outcome of the action.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if the action failed.",
    )
    metadata: dict[str, JsonType] | None = Field(
        default=None,
        description="Additional metadata about the action.",
    )
    raw_payload: dict[str, JsonType] | None = Field(
        default=None,
        description="Complete raw payload for Phase 2 schema tightening.",
    )

    def __str__(self) -> str:
        """Return concise string representation for logging.

        Includes key identifying fields but excludes metadata and raw_payload.
        """
        id_short = str(self.id)[:8]
        status_part = f", status={self.status}" if self.status else ""
        return (
            f"AgentAction(id={id_short}, agent={self.agent_name}, "
            f"type={self.action_type}, action={self.action_name}{status_part})"
        )


__all__ = ["ModelAgentAction"]
