# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL intent payload model for registration orchestrator.

This module provides the typed payload model for PostgreSQL registration intents.

Design Note:
    This model uses strongly-typed ModelNodeCapabilities and ModelNodeMetadata
    instead of generic dict[str, JsonValue] to adhere to ONEX "no Any types"
    principle. The JsonValue type alias contains list[Any] and dict[str, Any]
    internally, which violates strict typing requirements.

    Using the same typed models as ModelNodeIntrospectionEvent ensures:
    1. Type safety throughout the registration pipeline
    2. Consistent validation between event source and database persistence
    3. No implicit Any types in the payload structure

Thread Safety:
    This model is fully immutable (frozen=True) with immutable field types.
    The ``endpoints`` field uses tuple of key-value pairs instead of dict
    to ensure complete immutability for thread-safe concurrent access.

    For dict-like access to endpoints, use the ``endpoints_dict`` property
    which returns a MappingProxyType (read-only view).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata


class ModelPostgresIntentPayload(BaseModel):
    """Payload for PostgreSQL registration intents.

    Contains the full node introspection data to upsert into the
    registration database. This is a typed representation of the
    data previously passed via model_dump().

    Uses strongly-typed capability and metadata models matching
    ModelNodeIntrospectionEvent for type-safe pipeline processing.

    This model is fully immutable to support thread-safe concurrent access.
    All collection fields use immutable types (tuple instead of dict).

    Attributes:
        node_id: Unique node identifier.
        node_type: ONEX node type (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node.
        capabilities: Strongly-typed node capabilities.
        endpoints: Immutable tuple of (name, URL) pairs for exposed endpoints.
            Use the ``endpoints_dict`` property for dict-like read access.
        node_role: Optional role descriptor.
        metadata: Strongly-typed node metadata.
        correlation_id: Correlation ID for distributed tracing.
        network_id: Network/cluster identifier.
        deployment_id: Deployment/release identifier.
        epoch: Registration epoch for ordering.
        timestamp: Event timestamp as ISO string.

    Example:
        >>> payload = ModelPostgresIntentPayload(
        ...     node_id=uuid4(),
        ...     node_type="effect",
        ...     endpoints={"health": "/health", "api": "/api/v1"},
        ...     correlation_id=uuid4(),
        ...     timestamp="2025-01-01T00:00:00Z",
        ... )
        >>> payload.endpoints
        (('health', '/health'), ('api', '/api/v1'))
        >>> payload.endpoints_dict["health"]
        '/health'
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    node_id: UUID = Field(..., description="Unique node identifier")
    # Design Note: node_type uses Literal validation matching ModelNodeIntrospectionEvent
    # to ensure only valid ONEX node types can be persisted to PostgreSQL.
    node_type: Literal["effect", "compute", "reducer", "orchestrator"] = Field(
        ..., description="ONEX node type"
    )
    node_version: str = Field(default="1.0.0", description="Semantic version")
    capabilities: ModelNodeCapabilities = Field(
        default_factory=ModelNodeCapabilities,
        description="Strongly-typed node capabilities",
    )
    endpoints: tuple[tuple[str, str], ...] = Field(
        default=(),
        description="Immutable tuple of (name, URL) pairs for exposed endpoints",
    )
    node_role: str | None = Field(default=None, description="Node role")
    metadata: ModelNodeMetadata = Field(
        default_factory=ModelNodeMetadata,
        description="Strongly-typed node metadata",
    )
    correlation_id: UUID = Field(..., description="Correlation ID for tracing")
    network_id: str | None = Field(default=None, description="Network identifier")
    deployment_id: str | None = Field(default=None, description="Deployment identifier")
    epoch: int | None = Field(default=None, ge=0, description="Registration epoch")
    timestamp: str = Field(..., description="Event timestamp as ISO string")

    @field_validator("endpoints", mode="before")
    @classmethod
    def _coerce_endpoints_to_tuple(cls, v: Any) -> tuple[tuple[str, str], ...]:
        """Convert dict/mapping to tuple of pairs for immutability."""
        if isinstance(v, tuple):
            return v  # type: ignore[return-value]  # Runtime validated by Pydantic
        if isinstance(v, Mapping):
            return tuple((str(k), str(val)) for k, val in v.items())
        # For unrecognized types, return empty tuple (Pydantic will validate)
        return ()

    @property
    def endpoints_dict(self) -> MappingProxyType[str, str]:
        """Return a read-only dict view of the endpoints.

        Returns:
            MappingProxyType providing dict-like read access to endpoints.
        """
        return MappingProxyType(dict(self.endpoints))


__all__ = [
    "ModelPostgresIntentPayload",
]
