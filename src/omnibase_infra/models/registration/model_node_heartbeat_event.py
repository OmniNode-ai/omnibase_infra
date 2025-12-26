# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Heartbeat Event Model.

This module provides ModelNodeHeartbeatEvent for periodic node heartbeat broadcasts
in the ONEX 2-way registration pattern.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from omnibase_core.enums import (
    EnumNodeKind,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.utils.util_semver import validate_semver as _validate_semver


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
        >>> from uuid import uuid4
        >>> from datetime import datetime, timezone
        >>> from omnibase_core.enums import EnumNodeKind
        >>> event = ModelNodeHeartbeatEvent(
        ...     node_id=uuid4(),
        ...     node_type=EnumNodeKind.EFFECT,
        ...     node_version="1.2.3",
        ...     uptime_seconds=3600.5,
        ...     active_operations_count=5,
        ...     memory_usage_mb=256.0,
        ...     cpu_usage_percent=15.5,
        ...     timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
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
    node_version: str = Field(
        default="1.0.0",
        description="Semantic version of the node emitting this event",
    )

    @field_validator("node_version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validate that node_version follows semantic versioning."""
        return _validate_semver(v, "node_version")

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


__all__ = ["ModelNodeHeartbeatEvent"]
