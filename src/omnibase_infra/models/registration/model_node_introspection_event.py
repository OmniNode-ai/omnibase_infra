# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Introspection Event Model.

This module provides ModelNodeIntrospectionEvent for node introspection broadcasts
in the ONEX 2-way registration pattern.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata
from omnibase_infra.utils.util_semver import validate_semver as _validate_semver


class ModelNodeIntrospectionEvent(BaseModel):
    """Event model for node introspection broadcasts.

    Nodes publish this event to announce their presence, capabilities,
    and endpoints to the cluster. Used by the Registry node to maintain
    a live catalog of available nodes.

    Attributes:
        node_id: Unique node identifier.
        node_type: ONEX node type (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node emitting this event.
        capabilities: Dictionary of node capabilities.
        endpoints: Dictionary of exposed endpoints (name -> URL).
        node_role: Optional role descriptor (registry, adapter, etc).
        metadata: Additional node metadata.
        correlation_id: Required correlation ID for tracing and idempotency.
        network_id: Network/cluster identifier.
        deployment_id: Deployment/release identifier.
        epoch: Registration epoch for ordering.
        timestamp: Event timestamp.

    Example:
        >>> from uuid import uuid4
        >>> from datetime import datetime, timezone
        >>> event = ModelNodeIntrospectionEvent(
        ...     node_id=uuid4(),
        ...     node_type="effect",
        ...     node_version="1.2.3",
        ...     capabilities={"postgres": True, "read": True, "write": True},
        ...     endpoints={"health": "http://localhost:8080/health"},
        ...     correlation_id=uuid4(),
        ...     timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Required fields
    node_id: UUID = Field(..., description="Unique node identifier")
    # Design Note: node_type uses strict Literal validation because introspection
    # events are the authoritative source for node registrations persisted to
    # PostgreSQL. ModelNodeRegistration inherits this constraint to maintain
    # registry integrity. For relaxed validation supporting experimental node
    # types, see ModelNodeHeartbeatEvent which uses str.
    node_type: Literal["effect", "compute", "reducer", "orchestrator"] = Field(
        ..., description="ONEX node type"
    )
    node_version: str = Field(
        default="1.0.0",
        description="Semantic version of the node emitting this event",
    )

    @field_validator("node_version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validate that node_version follows semantic versioning."""
        return _validate_semver(v, "node_version")

    capabilities: ModelNodeCapabilities = Field(
        default_factory=ModelNodeCapabilities, description="Node capabilities"
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict, description="Exposed endpoints (name -> URL)"
    )

    @field_validator("endpoints")
    @classmethod
    def validate_endpoint_urls(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate that all endpoint values are valid URLs.

        Args:
            v: Dictionary of endpoint names to URL strings.

        Returns:
            The validated endpoints dictionary.

        Raises:
            ValueError: If any endpoint URL is invalid (missing scheme or netloc).
        """
        from urllib.parse import urlparse

        for name, url in v.items():
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL for endpoint '{name}': {url}")
        return v

    # Optional metadata
    node_role: str | None = Field(
        default=None, description="Node role (registry, adapter, etc)"
    )
    metadata: ModelNodeMetadata = Field(
        default_factory=ModelNodeMetadata, description="Additional node metadata"
    )
    correlation_id: UUID = Field(
        ..., description="Request correlation ID for tracing (required for idempotency)"
    )

    # Deployment topology
    network_id: str | None = Field(
        default=None, description="Network/cluster identifier"
    )
    deployment_id: str | None = Field(
        default=None, description="Deployment/release identifier"
    )
    epoch: int | None = Field(
        default=None,
        ge=0,
        description="Registration epoch for ordering (monotonically increasing counter)",
    )

    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    timestamp: datetime = Field(
        ...,
        description="Event timestamp",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_timezone_aware(cls, v: datetime) -> datetime:
        """Validate that timestamp is timezone-aware.

        Args:
            v: The timestamp value to validate.

        Returns:
            The validated timestamp.

        Raises:
            ValueError: If timestamp is naive (no timezone info).
        """
        if v.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware. Use datetime.now(UTC) or "
                "datetime(..., tzinfo=timezone.utc) instead of naive datetime."
            )
        return v


__all__ = ["ModelNodeIntrospectionEvent"]
