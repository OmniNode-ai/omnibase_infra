# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Model.

This module provides ModelNodeRegistration for persisted node registrations
in the ONEX 2-way registration pattern.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata
from omnibase_infra.utils import validate_endpoint_urls_dict


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
    node_version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer(major=1, minor=0, patch=0),
        description="Semantic version of the node",
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

        Delegates to shared utility for consistent validation across all models.
        """
        return validate_endpoint_urls_dict(v)

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
