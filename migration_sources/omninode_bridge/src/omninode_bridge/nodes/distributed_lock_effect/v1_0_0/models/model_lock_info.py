#!/usr/bin/env python3
"""
Lock Information Model - ONEX v2.0 Compliant.

Model for distributed lock metadata and status information.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enum_lock_status import EnumLockStatus


class ModelLockInfo(BaseModel):
    """
    Distributed lock metadata and status information.

    Contains complete lock state including ownership, timing, and status.
    Used in lock operation responses to provide lock visibility.
    """

    # === LOCK IDENTIFICATION ===

    lock_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this lock instance",
    )

    lock_name: str = Field(
        ...,
        description="Unique lock name identifier",
        min_length=1,
        max_length=255,
    )

    # === OWNERSHIP ===

    owner_id: str | None = Field(
        default=None,
        description="Current lock owner identifier (None if available)",
        max_length=255,
    )

    owner_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about lock owner (node_id, instance_id, etc.)",
    )

    # === STATUS ===

    status: EnumLockStatus = Field(
        default=EnumLockStatus.AVAILABLE,
        description="Current lock status",
    )

    # === TIMING ===

    acquired_at: datetime | None = Field(
        default=None,
        description="Timestamp when lock was acquired (None if not acquired)",
    )

    expires_at: datetime | None = Field(
        default=None,
        description="Timestamp when lock lease expires (None if not acquired)",
    )

    released_at: datetime | None = Field(
        default=None,
        description="Timestamp when lock was released (None if not released)",
    )

    lease_duration: float = Field(
        default=30.0,
        description="Lock lease duration in seconds",
        gt=0.0,
    )

    # === STATISTICS ===

    acquisition_count: int = Field(
        default=0,
        description="Total number of times lock has been acquired",
        ge=0,
    )

    extension_count: int = Field(
        default=0,
        description="Total number of times lock lease has been extended",
        ge=0,
    )

    # === METADATA ===

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when lock was first created",
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of last lock state change",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional lock metadata",
    )

    # === COMPUTED PROPERTIES ===

    def is_expired(self) -> bool:
        """
        Check if lock lease has expired.

        Returns:
            True if lock is acquired and lease has expired, False otherwise
        """
        if self.status != EnumLockStatus.ACQUIRED or self.expires_at is None:
            return False

        now = datetime.now(UTC)
        return now >= self.expires_at

    def time_until_expiration(self) -> float | None:
        """
        Calculate seconds until lock lease expires.

        Returns:
            Seconds until expiration (positive if not expired, negative if expired),
            None if lock is not acquired
        """
        if self.status != EnumLockStatus.ACQUIRED or self.expires_at is None:
            return None

        now = datetime.now(UTC)
        delta = self.expires_at - now
        return delta.total_seconds()

    def can_be_acquired(self) -> bool:
        """
        Check if lock is available for acquisition.

        Returns:
            True if lock can be acquired (AVAILABLE status or expired lease),
            False otherwise
        """
        if self.status == EnumLockStatus.AVAILABLE:
            return True

        # Check if acquired lock has expired (eligible for re-acquisition)
        if self.status == EnumLockStatus.ACQUIRED and self.is_expired():
            return True

        return False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lock_id": "123e4567-e89b-12d3-a456-426614174000",
                "lock_name": "workflow_processing_lock",
                "owner_id": "orchestrator-node-1",
                "owner_metadata": {
                    "node_id": "orchestrator-node-1",
                    "instance_id": "inst-abc123",
                },
                "status": "acquired",
                "acquired_at": "2025-10-30T12:00:00Z",
                "expires_at": "2025-10-30T12:00:30Z",
                "released_at": None,
                "lease_duration": 30.0,
                "acquisition_count": 1,
                "extension_count": 0,
                "created_at": "2025-10-30T12:00:00Z",
                "updated_at": "2025-10-30T12:00:00Z",
                "metadata": {},
            }
        }
    )


__all__ = ["ModelLockInfo"]
