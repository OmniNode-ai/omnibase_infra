# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Agent Action Model.

This module defines the model for agent action events consumed from Kafka.
Agent actions represent individual tool calls, decisions, errors, and
successes recorded during agent execution.

Design Decisions:
    - frozen=True: Immutability for thread safety
    - extra="ignore": Tolerates producer fields not in consumer schema (OMN-2986).
      The omniclaude producer (action_event_publisher.py) emits additional fields
      (action_details, debug_mode, timestamp) not present in this model. Using
      extra="ignore" prevents spurious ValidationError → DLQ routing.
    - from_attributes=True: ORM/pytest-xdist compatibility
    - raw_payload: Optional field to preserve complete payload for schema tightening
    - created_at: Defaults to ingestion time (UTC now) when absent from payload.
      Producers emit "timestamp" not "created_at"; the default factory ensures
      TTL cleanup has a valid timestamp.
    - id: Auto-generated UUID when absent from producer payload. Producers do not
      emit an "id" field; the consumer generates one at ingestion time.

Idempotency:
    Table: agent_actions
    Unique Key: id (UUID)
    Conflict Action: DO NOTHING (append-only audit log)

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> action = ModelAgentAction(
    ...     correlation_id=uuid4(),
    ...     agent_name="polymorphic-agent",
    ...     action_type="tool_call",
    ...     action_name="Bash",
    ... )
"""

import json
import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)

# Payload size limits (Phase 2 hardening - OMN-1768)
MAX_METADATA_SIZE_BYTES: int = 65_536  # 64 KB
MAX_RAW_PAYLOAD_SIZE_BYTES: int = 1_048_576  # 1 MB


class ModelAgentAction(BaseModel):
    """Agent action event model.

    Represents a single action performed by an agent, such as a tool call,
    decision, error, or success. Uses frozen=True for thread safety and
    extra="ignore" to tolerate unknown producer fields without DLQ routing.

    Schema Compatibility (OMN-2986):
        The omniclaude action_event_publisher emits these fields that are NOT in
        this model schema: action_details, debug_mode, timestamp. These are ignored
        at ingestion. The producer does NOT emit ``id`` or ``created_at`` — both
        default to auto-generated values at ingestion time.

    Attributes:
        id: Auto-generated UUID at ingestion (idempotency key, not from producer).
        correlation_id: Request correlation ID linking related actions.
        agent_name: Name of the agent that performed this action.
        action_type: Type of action (tool_call, decision, error, success).
        action_name: Specific name of the action or tool.
        created_at: Ingestion timestamp (UTC); defaults to now() when absent (TTL key).
        status: Optional status of the action (started, completed, failed).
        duration_ms: Optional duration of the action in milliseconds.
        result: Optional result summary or outcome.
        error_message: Optional error message if action failed.
        metadata: Optional additional metadata about the action.
        raw_payload: Optional complete raw payload for Phase 2 schema tightening.

    Example:
        >>> action = ModelAgentAction(
        ...     correlation_id=uuid4(),
        ...     agent_name="code-reviewer",
        ...     action_type="decision",
        ...     action_name="approve_pr",
        ...     status="completed",
        ...     duration_ms=1234,
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",  # OMN-2986: producer emits unknown fields (action_details, debug_mode, timestamp)
        from_attributes=True,
    )

    # ---- Auto-generated fields (not sent by producer) ----
    id: UUID = Field(
        default_factory=uuid4,
        description=(
            "Unique identifier for this action (idempotency key). "
            "Auto-generated at ingestion — omniclaude producer does not emit this field."
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description=(
            "Timestamp when the action was ingested (TTL key). "
            "Defaults to UTC now — omniclaude producer emits 'timestamp' not 'created_at'."
        ),
    )

    # ---- Required Fields (sent by producer) ----
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

    # ---- Payload Size Validators (Phase 2 hardening - OMN-1768) ----

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata_size(
        cls, v: dict[str, JsonType] | None
    ) -> dict[str, JsonType] | None:
        """Validate metadata does not exceed size limit.

        Truncates oversized metadata to prevent unbounded storage growth.
        Logs a warning when truncation occurs.

        Args:
            v: Metadata dictionary or None.

        Returns:
            Original metadata if within limits, truncated version if oversized.
        """
        if v is None:
            return v
        serialized = json.dumps(v)
        if len(serialized.encode("utf-8")) > MAX_METADATA_SIZE_BYTES:
            logger.warning(
                "Metadata exceeds size limit (%d bytes > %d), truncating",
                len(serialized.encode("utf-8")),
                MAX_METADATA_SIZE_BYTES,
                extra={"size_bytes": len(serialized.encode("utf-8"))},
            )
            return {
                "_truncated": True,
                "_original_size_bytes": len(serialized.encode("utf-8")),
            }
        return v

    @field_validator("raw_payload", mode="before")
    @classmethod
    def validate_raw_payload_size(
        cls, v: dict[str, JsonType] | None
    ) -> dict[str, JsonType] | None:
        """Validate raw_payload does not exceed size limit.

        Truncates oversized raw payloads to prevent unbounded storage growth.
        Logs a warning when truncation occurs.

        Args:
            v: Raw payload dictionary or None.

        Returns:
            Original payload if within limits, truncated version if oversized.
        """
        if v is None:
            return v
        serialized = json.dumps(v)
        if len(serialized.encode("utf-8")) > MAX_RAW_PAYLOAD_SIZE_BYTES:
            logger.warning(
                "Raw payload exceeds size limit (%d bytes > %d), truncating",
                len(serialized.encode("utf-8")),
                MAX_RAW_PAYLOAD_SIZE_BYTES,
                extra={"size_bytes": len(serialized.encode("utf-8"))},
            )
            return {
                "_truncated": True,
                "_original_size_bytes": len(serialized.encode("utf-8")),
            }
        return v

    # ---- Project Context (absorbed from omniclaude - OMN-2057) ----
    project_path: str | None = Field(
        default=None,
        description="Absolute path to the project being worked on.",
    )
    project_name: str | None = Field(
        default=None,
        description="Human-readable project name.",
    )
    working_directory: str | None = Field(
        default=None,
        description="Working directory where the action was executed.",
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


__all__ = [
    "MAX_METADATA_SIZE_BYTES",
    "MAX_RAW_PAYLOAD_SIZE_BYTES",
    "ModelAgentAction",
]
