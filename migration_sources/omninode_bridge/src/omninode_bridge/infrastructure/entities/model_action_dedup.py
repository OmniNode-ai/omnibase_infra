#!/usr/bin/env python3
"""
ModelActionDedup - Action Deduplication Entity.

Direct Pydantic representation of action_dedup_log database table.
Maps 1:1 with database schema for type-safe database operations.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelActionDedup
- UUID action identifier
- Deduplication state tracking
- Comprehensive field validation with Pydantic v2

Purpose:
Prevents duplicate processing from retries or at-least-once delivery guarantees.
Uses composite primary key (workflow_key, action_id) for uniqueness.
"""

from datetime import UTC, datetime, timedelta
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelActionDedup(BaseModel):
    """
    Entity model for action_dedup_log table.

    This model represents action deduplication records in the database.
    It maps directly to the action_dedup_log table schema.

    Database Table: action_dedup_log
    Primary Key: (workflow_key, action_id) - Composite
    Indexes: expires_at

    Purpose:
    - Prevent duplicate action processing from retries
    - Handle at-least-once delivery guarantees
    - Enable efficient TTL-based cleanup

    Example:
        >>> from uuid import uuid4
        >>> from datetime import datetime, timedelta, timezone
        >>> import hashlib
        >>> import json
        >>>
        >>> # Compute result hash
        >>> result = {"status": "completed", "items": 100}
        >>> result_hash = hashlib.sha256(
        ...     json.dumps(result, sort_keys=True).encode()
        ... ).hexdigest()
        >>>
        >>> # Create dedup record with 6-hour TTL
        >>> dedup = ModelActionDedup(
        ...     workflow_key="workflow-123",
        ...     action_id=uuid4(),
        ...     result_hash=result_hash,
        ...     processed_at=datetime.now(timezone.utc),
        ...     expires_at=datetime.now(timezone.utc) + timedelta(hours=6)
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Composite primary key components
    workflow_key: str = Field(
        ...,
        description="Workflow identifier for grouping actions",
        min_length=1,
    )

    action_id: UUID = Field(
        ...,
        description="Unique action identifier (UUID)",
    )

    # Deduplication data
    result_hash: str = Field(
        ...,
        description="SHA256 hash of action result for validation on replay",
        min_length=64,
        max_length=64,
    )

    # Temporal tracking
    processed_at: datetime = Field(
        ...,
        description="Timestamp when action was first processed",
    )

    expires_at: datetime = Field(
        ...,
        description="TTL expiration timestamp (default 6 hours), enables efficient cleanup",
    )

    @field_validator("result_hash")
    @classmethod
    def validate_result_hash_format(cls, v: str) -> str:
        """Validate that result_hash is a valid SHA256 hex string."""
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("result_hash must be a valid hexadecimal string")
        return v.lower()

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at_after_processed_at(cls, v: datetime, info) -> datetime:
        """Validate that expires_at is after processed_at."""
        processed_at = info.data.get("processed_at")
        if processed_at and v <= processed_at:
            raise ValueError("expires_at must be after processed_at")
        return v

    @classmethod
    def create_with_ttl(
        cls,
        workflow_key: str,
        action_id: UUID,
        result_hash: str,
        ttl_hours: int = 6,
        processed_at: Optional[datetime] = None,
    ) -> "ModelActionDedup":
        """
        Factory method to create a dedup record with TTL.

        Args:
            workflow_key: Workflow identifier
            action_id: Action UUID
            result_hash: SHA256 hash of result
            ttl_hours: Time-to-live in hours (default: 6)
            processed_at: Processing timestamp (default: now)

        Returns:
            ModelActionDedup instance with expires_at computed from TTL

        Example:
            >>> from uuid import uuid4
            >>> import hashlib
            >>> import json
            >>>
            >>> result = {"status": "completed"}
            >>> result_hash = hashlib.sha256(
            ...     json.dumps(result, sort_keys=True).encode()
            ... ).hexdigest()
            >>>
            >>> dedup = ModelActionDedup.create_with_ttl(
            ...     workflow_key="workflow-123",
            ...     action_id=uuid4(),
            ...     result_hash=result_hash,
            ...     ttl_hours=6
            ... )
        """
        now = processed_at or datetime.now(UTC)
        expires = now + timedelta(hours=ttl_hours)

        return cls(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
            processed_at=now,
            expires_at=expires,
        )
