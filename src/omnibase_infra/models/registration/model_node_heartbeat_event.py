# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Heartbeat Event Model.

This module provides ModelNodeHeartbeatEvent for periodic node heartbeat broadcasts
in the ONEX 2-way registration pattern.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelNodeHeartbeatEvent(BaseModel):
    """Event model for periodic node heartbeat broadcasts.

    Nodes publish this event periodically to indicate they are alive and
    report current health metrics. Used by the Registry node to detect
    node failures and track resource usage.

    Attributes:
        node_id: Node identifier.
        node_type: ONEX node type (EnumNodeKind).
        node_version: Semantic version of the node emitting this event.
        uptime_seconds: Node uptime in seconds (must be >= 0).
        active_operations_count: Number of active operations (must be >= 0).
        memory_usage_mb: Optional memory usage in megabytes.
        cpu_usage_percent: Optional CPU usage percentage (0-100).
        correlation_id: Request correlation ID for tracing.
        timestamp: Event timestamp.

    Example:
        >>> from datetime import UTC, datetime
        >>> from uuid import uuid4
        >>> from omnibase_core.enums import EnumNodeKind
        >>> from omnibase_core.models.primitives.model_semver import ModelSemVer
        >>> event = ModelNodeHeartbeatEvent(
        ...     node_id=uuid4(),
        ...     node_type=EnumNodeKind.EFFECT,
        ...     node_version=ModelSemVer(major=1, minor=2, patch=3),
        ...     uptime_seconds=3600.5,
        ...     active_operations_count=5,
        ...     memory_usage_mb=256.0,
        ...     cpu_usage_percent=15.5,
        ...     timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Required fields
    node_id: UUID = Field(..., description="Node identifier")
    # Design Note: node_type uses EnumNodeKind for type consistency across the codebase.
    # This aligns with ModelNodeIntrospectionEvent which also uses EnumNodeKind, ensuring
    # consistent type handling for all node-related events. The previous relaxed `str`
    # validation was speculative support for experimental/plugin nodes that never existed.
    node_type: EnumNodeKind = Field(..., description="ONEX node type")
    node_version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer(major=1, minor=0, patch=0),
        description="Semantic version of the node emitting this event",
    )

    @field_validator("node_version", mode="before")
    @classmethod
    def parse_node_version(cls, v: ModelSemVer | str) -> ModelSemVer:
        """Parse node_version from string or ModelSemVer.

        Args:
            v: Either a ModelSemVer instance or a semver string.

        Returns:
            Validated ModelSemVer instance.

        Raises:
            ValueError: If the string is not a valid semantic version.
        """
        if isinstance(v, str):
            try:
                return ModelSemVer.parse(v)
            except Exception as e:
                raise ValueError(f"node_version: {e!s}") from e
        return v

    # Health metrics
    uptime_seconds: float = Field(..., ge=0, description="Node uptime in seconds")
    active_operations_count: int = Field(
        default=0, ge=0, description="Number of active operations"
    )

    # Resource usage (optional)
    memory_usage_mb: float | None = Field(
        default=None, ge=0, description="Memory usage in megabytes"
    )
    cpu_usage_percent: float | None = Field(
        default=None, ge=0, le=100, description="CPU usage percentage (0-100)"
    )

    # Metadata
    correlation_id: UUID | None = Field(
        default=None, description="Request correlation ID for tracing"
    )
    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    timestamp: datetime = Field(..., description="Event timestamp")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_timezone_aware(cls, v: datetime) -> datetime:
        """Validate that timestamp is timezone-aware.

        Delegates to shared utility for consistent validation across all models.
        """
        return validate_timezone_aware_datetime(v)


__all__ = ["ModelNodeHeartbeatEvent"]
