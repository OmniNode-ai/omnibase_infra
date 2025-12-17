# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Model.

This module provides ModelNodeRegistration for persisted node registrations
in the ONEX 2-way registration pattern.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeRegistration(BaseModel):
    """Model for persisted node registration in PostgreSQL.

    Represents a node's complete registration state for persistence.
    Created from introspection events and updated with heartbeat data.

    Attributes:
        node_id: Unique node identifier.
        node_type: ONEX node type string.
        node_version: Semantic version of the node.
        capabilities: Dictionary of node capabilities.
        endpoints: Dictionary of exposed endpoints (name -> URL).
        metadata: Additional node metadata.
        health_endpoint: URL for health check endpoint.
        last_heartbeat: Timestamp of last received heartbeat.
        registered_at: Timestamp when node was first registered.
        updated_at: Timestamp of last update.

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> now = datetime.now(UTC)
        >>> registration = ModelNodeRegistration(
        ...     node_id=uuid4(),
        ...     node_type="effect",
        ...     capabilities={"postgres": True},
        ...     endpoints={"health": "http://localhost:8080/health"},
        ...     health_endpoint="http://localhost:8080/health",
        ...     registered_at=now,
        ...     updated_at=now,
        ... )
    """

    model_config = ConfigDict(
        frozen=False,  # Mutable for updates
        extra="forbid",
        from_attributes=True,
    )

    # Identity
    node_id: UUID = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="ONEX node type")
    node_version: str = Field(
        default="1.0.0", description="Semantic version of the node"
    )

    # Capabilities and endpoints
    capabilities: dict[str, Any] = Field(
        default_factory=dict, description="Node capabilities"
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict, description="Exposed endpoints (name -> URL)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional node metadata"
    )

    # Health tracking
    health_endpoint: str | None = Field(
        default=None, description="URL for health check endpoint"
    )
    last_heartbeat: datetime | None = Field(
        default=None, description="Timestamp of last received heartbeat"
    )

    # Timestamps
    registered_at: datetime = Field(
        ..., description="Timestamp when node was first registered"
    )
    updated_at: datetime = Field(..., description="Timestamp of last update")


__all__ = ["ModelNodeRegistration"]
