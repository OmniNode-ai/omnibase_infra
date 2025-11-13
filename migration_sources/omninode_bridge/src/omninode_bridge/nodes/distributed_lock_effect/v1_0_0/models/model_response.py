#!/usr/bin/env python3
"""
Distributed Lock Response Model - ONEX v2.0 Compliant.

Response model for distributed lock operations.
"""

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .enum_lock_operation import EnumLockOperation
from .model_lock_info import ModelLockInfo


class ModelDistributedLockResponse(BaseModel):
    """
    Response model for distributed lock operations.

    Contains operation results, lock information, and execution metrics.
    """

    # === OPERATION RESULTS ===

    success: bool = Field(
        ...,
        description="Whether the lock operation succeeded",
    )

    operation: EnumLockOperation = Field(
        ...,
        description="Lock operation that was performed",
    )

    lock_info: ModelLockInfo | None = Field(
        default=None,
        description="Lock metadata and status (None for failed acquire or cleanup)",
    )

    # === CLEANUP RESULTS (for cleanup operation) ===

    cleaned_count: int = Field(
        default=0,
        description="Number of locks cleaned up (for cleanup operation)",
        ge=0,
    )

    cleaned_lock_names: list[str] = Field(
        default_factory=list,
        description="Names of locks that were cleaned up",
    )

    # === ERROR HANDLING ===

    error_message: str | None = Field(
        default=None,
        description="Error message if operation failed",
    )

    error_code: str | None = Field(
        default=None,
        description="Error code for programmatic error handling",
    )

    retry_after_seconds: float | None = Field(
        default=None,
        description="Suggested retry delay in seconds (for failed acquire)",
    )

    # === EXECUTION METRICS ===

    duration_ms: float = Field(
        ...,
        description="Total operation duration in milliseconds",
        ge=0.0,
    )

    acquire_attempts: int = Field(
        default=0,
        description="Number of lock acquisition attempts (for acquire operation)",
        ge=0,
    )

    database_query_ms: float = Field(
        default=0.0,
        description="Time spent on database queries in milliseconds",
        ge=0.0,
    )

    # === WARNINGS ===

    warnings: list[str] = Field(
        default_factory=list,
        description="Non-critical warnings during operation",
    )

    # === CORRELATION TRACKING ===

    correlation_id: UUID = Field(
        ...,
        description="UUID for correlation tracking (from request)",
    )

    execution_id: UUID = Field(
        ...,
        description="UUID for this execution instance (from request)",
    )

    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when operation completed",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "success": True,
                    "operation": "acquire",
                    "lock_info": {
                        "lock_id": "123e4567-e89b-12d3-a456-426614174000",
                        "lock_name": "workflow_processing_lock",
                        "owner_id": "orchestrator-node-1",
                        "status": "acquired",
                        "acquired_at": "2025-10-30T12:00:00Z",
                        "expires_at": "2025-10-30T12:00:30Z",
                        "lease_duration": 30.0,
                    },
                    "cleaned_count": 0,
                    "cleaned_lock_names": [],
                    "error_message": None,
                    "error_code": None,
                    "duration_ms": 45.2,
                    "acquire_attempts": 1,
                    "database_query_ms": 38.5,
                    "warnings": [],
                },
                {
                    "success": False,
                    "operation": "acquire",
                    "lock_info": None,
                    "cleaned_count": 0,
                    "cleaned_lock_names": [],
                    "error_message": "Lock already held by another owner",
                    "error_code": "LOCK_ALREADY_HELD",
                    "retry_after_seconds": 5.0,
                    "duration_ms": 25.8,
                    "acquire_attempts": 3,
                    "database_query_ms": 18.2,
                    "warnings": [],
                },
                {
                    "success": True,
                    "operation": "cleanup",
                    "lock_info": None,
                    "cleaned_count": 5,
                    "cleaned_lock_names": [
                        "workflow_lock_1",
                        "workflow_lock_2",
                        "workflow_lock_3",
                        "workflow_lock_4",
                        "workflow_lock_5",
                    ],
                    "error_message": None,
                    "error_code": None,
                    "duration_ms": 85.5,
                    "database_query_ms": 72.3,
                    "warnings": [],
                },
            ]
        }
    )


__all__ = ["ModelDistributedLockResponse"]
