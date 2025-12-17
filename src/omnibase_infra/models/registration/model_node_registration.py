# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Model.

This module provides the Pydantic model for persisted node registration
in PostgreSQL, used by the discovery system to track registered nodes
and their capabilities.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

# Type aliases for strongly-typed capability and metadata values
# Capabilities support: lists of strings (operations/protocols), booleans (flags),
# and string dictionaries (method signatures)
CapabilityValue = list[str] | bool | dict[str, str]

# Metadata supports primitive JSON-serializable types
MetadataValue = str | int | float | bool | None


class ModelNodeRegistration(BaseModel):
    """Model for persisted node registration in PostgreSQL.

    Represents a registered node in the ONEX discovery system, including
    its type, version, capabilities, endpoints, and health status.

    Attributes:
        node_id: Unique identifier for the registered node
        node_type: Type classification of the node (e.g., 'effect', 'compute')
        node_version: Semantic version of the node (default: "1.0.0")
        capabilities: Dictionary of node capabilities and features
        endpoints: Dictionary mapping endpoint names to their URLs
        metadata: Additional metadata associated with the node
        health_endpoint: URL for the node's health check endpoint
        last_heartbeat: Timestamp of the last successful heartbeat
        registered_at: Timestamp when the node was first registered
        updated_at: Timestamp when the registration was last updated

    Example:
        >>> from omnibase_infra.models.registration import ModelNodeRegistration
        >>> from datetime import UTC, datetime
        >>> registration = ModelNodeRegistration(
        ...     node_id="node-postgres-adapter-001",
        ...     node_type="effect",
        ...     node_version="1.0.0",
        ...     capabilities={"database": True, "transactions": True},
        ...     endpoints={"api": "http://localhost:8080"},
        ...     health_endpoint="http://localhost:8080/health",
        ...     registered_at=datetime.now(UTC),
        ...     updated_at=datetime.now(UTC),
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=False,  # Allow updates for heartbeat and metadata changes
        extra="forbid",
    )

    node_id: str = Field(
        ...,
        description="Unique identifier for the registered node",
    )
    node_type: str = Field(
        ...,
        description="Type classification of the node (e.g., 'effect', 'compute')",
    )
    node_version: str = Field(
        default="1.0.0",
        description="Semantic version of the node",
    )
    capabilities: dict[str, CapabilityValue] = Field(
        default_factory=dict,
        description="Dictionary of node capabilities and features",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping endpoint names to their URLs",
    )
    metadata: dict[str, MetadataValue] = Field(
        default_factory=dict,
        description="Additional metadata associated with the node",
    )
    health_endpoint: str | None = Field(
        default=None,
        description="URL for the node's health check endpoint",
    )
    last_heartbeat: datetime | None = Field(
        default=None,
        description="Timestamp of the last successful heartbeat",
    )
    registered_at: datetime = Field(
        ...,
        description="Timestamp when the node was first registered",
    )
    updated_at: datetime = Field(
        ...,
        description="Timestamp when the registration was last updated",
    )


__all__: list[str] = ["ModelNodeRegistration", "CapabilityValue", "MetadataValue"]
