# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Model.

This module provides ModelNodeRegistration for persisted node registrations
in the ONEX 2-way registration pattern.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from urllib.parse import urlparse
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata
from omnibase_infra.utils.util_semver import validate_semver as _validate_semver


class ModelNodeRegistration(BaseModel):
    """Model for persisted node registration in PostgreSQL.

    Represents a node's complete registration state for persistence.
    Created from introspection events and updated with heartbeat data.

    Validation Design:
        This model uses **strict Literal validation** for ``node_type``, accepting
        only the canonical ONEX types: "effect", "compute", "reducer", "orchestrator".

        This differs intentionally from ``ModelNodeHeartbeatEvent``, which uses
        relaxed ``str`` validation to support experimental node types in transient
        heartbeat messages. The rationale:

        - **ModelNodeRegistration** (strict): Represents canonical node catalog
          entries persisted to PostgreSQL. Must align with the source
          ``ModelNodeIntrospectionEvent`` constraints. Invalid types would corrupt
          the registry.
        - **ModelNodeHeartbeatEvent** (relaxed): Transient operational messages
          that may include experimental or custom node types not yet in the
          canonical set.

    Attributes:
        node_id: Unique node identifier.
        node_type: ONEX node type. Uses strict Literal["effect", "compute",
            "reducer", "orchestrator"] validation to ensure registry integrity.
            See field-level design note for rationale.
        node_version: Semantic version of the node (validated against semver).
        capabilities: Structured node capabilities.
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

    See Also:
        - :class:`ModelNodeIntrospectionEvent`: Source event for registrations.
          Uses the same strict ``Literal`` validation for ``node_type``.
        - :class:`ModelNodeHeartbeatEvent`: Transient health events.
          Uses relaxed ``str`` validation to support experimental node types.
    """

    model_config = ConfigDict(
        frozen=False,  # Mutable for updates
        extra="forbid",
        from_attributes=True,
    )

    # Identity
    node_id: UUID = Field(..., description="Unique node identifier")
    # Design Note: node_type uses strict Literal validation to match the source
    # introspection event constraints. ModelNodeRegistration is created from
    # ModelNodeIntrospectionEvent data, so type constraints must align. Unlike
    # ModelNodeHeartbeatEvent (which uses relaxed str validation to support
    # experimental node types), registrations represent canonical node catalog
    # entries that require strict ONEX type compliance.
    # See ModelNodeIntrospectionEvent for source validation.
    node_type: Literal["effect", "compute", "reducer", "orchestrator"] = Field(
        ..., description="ONEX node type"
    )
    node_version: str = Field(
        default="1.0.0", description="Semantic version of the node"
    )

    @field_validator("node_version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validate that node_version follows semantic versioning."""
        return _validate_semver(v, "node_version")

    # Capabilities and endpoints
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
        for name, url in v.items():
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL for endpoint '{name}': {url}")
        return v

    metadata: ModelNodeMetadata = Field(
        default_factory=ModelNodeMetadata, description="Additional node metadata"
    )

    # Health tracking
    health_endpoint: HttpUrl | None = Field(
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
