# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Detection Failure Model.

This module defines the model for agent detection failure events consumed
from Kafka. Detection failures occur when the routing system cannot
determine an appropriate agent for a request.

Design Decisions:
    - extra="allow": Phase 1 flexibility - required fields typed, extras preserved
    - raw_payload: Optional field to preserve complete payload for schema tightening
    - created_at: Required for TTL cleanup job (Phase 2)

Idempotency:
    Table: agent_detection_failures
    Unique Key: correlation_id (UUID) - one failure per correlation
    Conflict Action: DO NOTHING

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> failure = ModelDetectionFailure(
    ...     correlation_id=uuid4(),
    ...     failure_reason="No matching agent pattern",
    ...     created_at=datetime.now(UTC),
    ... )
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelDetectionFailure(BaseModel):
    """Agent detection failure event model.

    Represents a failure to detect or route to an appropriate agent.
    Used for analyzing routing coverage gaps and improving agent matching.

    Attributes:
        correlation_id: Request correlation ID (idempotency key - one per correlation).
        failure_reason: Reason the detection failed.
        created_at: Timestamp when the failure was recorded (TTL key).
        request_summary: Optional summary of the request that failed routing.
        attempted_patterns: Optional list of patterns attempted during detection.
        fallback_used: Optional name of fallback agent if one was used.
        error_code: Optional error code for categorization.
        metadata: Optional additional metadata about the failure.
        raw_payload: Optional complete raw payload for Phase 2 schema tightening.

    Example:
        >>> failure = ModelDetectionFailure(
        ...     correlation_id=uuid4(),
        ...     failure_reason="Confidence below threshold (0.3 < 0.5)",
        ...     created_at=datetime.now(UTC),
        ...     attempted_patterns=["code-review", "testing", "infrastructure"],
        ...     fallback_used="polymorphic-agent",
        ... )
    """

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
    )

    # ---- Required Fields ----
    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID (idempotency key - one per correlation).",
    )
    failure_reason: str = Field(
        ...,
        description="Reason the detection failed.",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the failure was recorded (TTL key).",
    )

    # ---- Optional Fields ----
    request_summary: str | None = Field(
        default=None,
        description="Summary of the request that failed routing.",
    )
    attempted_patterns: list[str] | None = Field(
        default=None,
        description="List of patterns attempted during detection.",
    )
    fallback_used: str | None = Field(
        default=None,
        description="Name of fallback agent if one was used.",
    )
    error_code: str | None = Field(
        default=None,
        description="Error code for categorization.",
    )
    metadata: dict[str, JsonType] | None = Field(
        default=None,
        description="Additional metadata about the failure.",
    )
    raw_payload: dict[str, JsonType] | None = Field(
        default=None,
        description="Complete raw payload for Phase 2 schema tightening.",
    )


__all__ = ["ModelDetectionFailure"]
