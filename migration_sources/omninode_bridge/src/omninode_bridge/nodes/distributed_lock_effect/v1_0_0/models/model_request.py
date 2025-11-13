#!/usr/bin/env python3
"""
Distributed Lock Request Model - ONEX v2.0 Compliant.

Request model for distributed lock operations.
"""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enum_lock_operation import EnumLockOperation


class ModelDistributedLockRequest(BaseModel):
    """
    Request model for distributed lock operations.

    Contains operation type, lock identification, and operation-specific parameters.
    """

    # === OPERATION SPECIFICATION ===

    operation: EnumLockOperation = Field(
        ...,
        description="Lock operation type (acquire, release, extend, query, cleanup)",
    )

    # === LOCK IDENTIFICATION ===

    lock_name: str = Field(
        ...,
        description="Unique lock name identifier",
        min_length=1,
        max_length=255,
    )

    owner_id: str | None = Field(
        default=None,
        description="Lock owner identifier (required for acquire/release/extend)",
        max_length=255,
    )

    # === OPERATION PARAMETERS ===

    lease_duration: float = Field(
        default=30.0,
        description="Lock lease duration in seconds (for acquire operation)",
        gt=0.0,
        le=3600.0,  # Max 1 hour lease
    )

    extension_duration: float = Field(
        default=30.0,
        description="Lock lease extension duration in seconds (for extend operation)",
        gt=0.0,
        le=3600.0,
    )

    max_acquire_attempts: int = Field(
        default=3,
        description="Maximum lock acquisition attempts before failure",
        ge=1,
        le=10,
    )

    acquire_retry_delay: float = Field(
        default=0.1,
        description="Delay between lock acquisition retry attempts in seconds",
        ge=0.0,
        le=10.0,
    )

    max_age_seconds: float = Field(
        default=3600.0,
        description="Maximum lock age for cleanup operation in seconds",
        gt=0.0,
    )

    # === METADATA ===

    owner_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about lock owner (node_id, instance_id, etc.)",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional lock metadata",
    )

    # === CORRELATION TRACKING ===

    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="UUID for correlation tracking",
    )

    execution_id: UUID = Field(
        default_factory=uuid4,
        description="UUID for tracking this execution instance",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "operation": "acquire",
                    "lock_name": "workflow_processing_lock",
                    "owner_id": "orchestrator-node-1",
                    "lease_duration": 30.0,
                    "max_acquire_attempts": 3,
                    "owner_metadata": {
                        "node_id": "orchestrator-node-1",
                        "instance_id": "inst-abc123",
                    },
                },
                {
                    "operation": "release",
                    "lock_name": "workflow_processing_lock",
                    "owner_id": "orchestrator-node-1",
                },
                {
                    "operation": "extend",
                    "lock_name": "workflow_processing_lock",
                    "owner_id": "orchestrator-node-1",
                    "extension_duration": 30.0,
                },
                {
                    "operation": "query",
                    "lock_name": "workflow_processing_lock",
                },
                {
                    "operation": "cleanup",
                    "lock_name": "*",  # Cleanup all locks
                    "max_age_seconds": 3600.0,
                },
            ]
        }
    )


__all__ = ["ModelDistributedLockRequest"]
