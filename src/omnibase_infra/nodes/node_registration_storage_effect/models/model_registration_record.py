# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Record Model for Storage Operations.

This module provides ModelRegistrationRecord, representing a complete node
registration record for storage in backend systems (PostgreSQL, MongoDB).

Architecture:
    ModelRegistrationRecord captures all information about a registered node:
    - Identity: node_id, node_type, node_version
    - Capabilities: List of capabilities the node provides
    - Endpoints: Service discovery endpoints
    - Metadata: Additional key-value metadata
    - Timestamps: created_at, updated_at for tracking

    This model is backend-agnostic and can be serialized to any storage format.

Security:
    The metadata field should NOT contain sensitive information.
    Secrets should be stored in Vault, not in registration records.

Related:
    - NodeRegistrationStorageEffect: Effect node that stores these records
    - ProtocolRegistrationStorageHandler: Protocol for storage backends
    - ModelStorageQuery: Query model for retrieving records
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelRegistrationRecord(BaseModel):
    """Registration record for node storage operations.

    Represents a complete node registration record that can be stored in
    any backend (PostgreSQL, MongoDB). This model is capability-oriented,
    focusing on what the node does rather than implementation details.

    Immutability:
        This model uses frozen=True to ensure records are immutable
        once created, supporting safe concurrent access and comparison.

    Attributes:
        node_id: Unique identifier for the registered node.
        node_type: Type of ONEX node (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).
        node_version: Semantic version string of the node.
        capabilities: List of capability names the node provides.
        endpoints: Dict mapping endpoint type to URL.
        metadata: Additional key-value metadata (no secrets).
        created_at: Timestamp when the record was created.
        updated_at: Timestamp when the record was last updated.

    Example:
        >>> from datetime import UTC, datetime
        >>> from uuid import uuid4
        >>> record = ModelRegistrationRecord(
        ...     node_id=uuid4(),
        ...     node_type="EFFECT",
        ...     node_version="1.0.0",
        ...     capabilities=["registration.storage", "registration.storage.query"],
        ...     endpoints={"health": "http://localhost:8080/health"},
        ...     metadata={"team": "platform"},
        ...     created_at=datetime.now(UTC),
        ...     updated_at=datetime.now(UTC),
        ... )
        >>> record.node_type
        'EFFECT'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: UUID = Field(
        ...,
        description="Unique identifier for the registered node",
    )
    node_type: Literal["EFFECT", "COMPUTE", "REDUCER", "ORCHESTRATOR"] = Field(
        ...,
        description="Type of ONEX node",
    )
    node_version: str = Field(
        ...,
        description="Semantic version string of the node (e.g., '1.0.0')",
        min_length=1,
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="List of capability names the node provides",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Dict mapping endpoint type to URL (e.g., {'health': 'http://...'})",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional key-value metadata (no secrets)",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the record was created (must be timezone-aware)",
    )
    updated_at: datetime = Field(
        ...,
        description="Timestamp when the record was last updated (must be timezone-aware)",
    )

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamp_timezone_aware(cls, v: datetime) -> datetime:
        """Validate that timestamps are timezone-aware.

        Args:
            v: The timestamp value to validate.

        Returns:
            The validated timestamp.

        Raises:
            ValueError: If timestamp is naive (no timezone info).
        """
        if v.tzinfo is None:
            raise ValueError(
                "Timestamp must be timezone-aware. Use datetime.now(UTC) or "
                "datetime(..., tzinfo=timezone.utc) instead of naive datetime."
            )
        return v

    @field_validator("node_version")
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Validate that node_version follows semantic versioning pattern.

        Args:
            v: The version string to validate.

        Returns:
            The validated version string.

        Raises:
            ValueError: If version doesn't match expected pattern.
        """
        import re

        # Basic semver pattern (major.minor.patch with optional pre-release)
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$"
        if not re.match(pattern, v):
            raise ValueError(
                f"node_version must follow semantic versioning (e.g., '1.0.0'). "
                f"Got: '{v}'"
            )
        return v


__all__ = ["ModelRegistrationRecord"]
