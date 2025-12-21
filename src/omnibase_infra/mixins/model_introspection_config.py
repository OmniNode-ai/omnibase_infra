# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Introspection Configuration Model.

This module provides the Pydantic configuration model for the MixinNodeIntrospection
mixin, consolidating initialization parameters into a single typed configuration object.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from omnibase_infra.mixins.protocol_event_bus_like import ProtocolEventBusLike


def _coerce_to_uuid(value: str | UUID) -> UUID:
    """Convert string or UUID to UUID.

    Args:
        value: A string or UUID value.

    Returns:
        UUID representation of the value.

    Raises:
        ValueError: If string is not a valid UUID format.
    """
    if isinstance(value, UUID):
        return value
    return UUID(value)


# Type that accepts both str and UUID but stores as UUID
NodeIdType = Annotated[UUID, BeforeValidator(_coerce_to_uuid)]


class ModelIntrospectionConfig(BaseModel):
    """Configuration for node introspection initialization.

    This model groups all initialization parameters for MixinNodeIntrospection,
    providing type safety and validation for introspection configuration.

    Attributes:
        node_id: Unique identifier for this node instance (UUID).
            Strings are automatically converted to UUIDs.
        node_type: Node type classification (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).
        event_bus: Optional event bus for publishing introspection events.
            Must have ``publish_envelope()`` method if provided.
        version: Node version string (default: "1.0.0").
        cache_ttl: Cache time-to-live in seconds (default: 300.0).
        operation_keywords: Optional set of keywords to identify operation methods.
            Methods containing these keywords are reported as operations.
            If None, uses DEFAULT_OPERATION_KEYWORDS from the mixin.
        exclude_prefixes: Optional set of prefixes to exclude from capability
            discovery. Methods starting with these prefixes are filtered out.
            If None, uses DEFAULT_EXCLUDE_PREFIXES from the mixin.

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.mixins import ModelIntrospectionConfig
        >>>
        >>> # Basic configuration with UUID
        >>> config = ModelIntrospectionConfig(
        ...     node_id=uuid4(),
        ...     node_type="EFFECT",
        ... )
        >>>
        >>> # Full configuration
        >>> config = ModelIntrospectionConfig(
        ...     node_id=uuid4(),
        ...     node_type="COMPUTE",
        ...     version="2.0.0",
        ...     cache_ttl=600.0,
        ...     operation_keywords={"fetch", "upload", "download"},
        ...     exclude_prefixes={"internal_", "debug_"},
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow ProtocolEventBus duck-typed objects
    )

    node_id: NodeIdType = Field(
        ...,
        description="Unique identifier for this node instance (UUID)",
    )
    node_type: str = Field(
        ...,
        min_length=1,
        description="Node type classification (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)",
    )
    event_bus: ProtocolEventBusLike | None = Field(
        default=None,
        description="Optional event bus for publishing introspection events",
    )
    version: str = Field(
        default="1.0.0",
        description="Node version string",
    )
    cache_ttl: float = Field(
        default=300.0,
        ge=0.0,
        description="Cache time-to-live in seconds",
    )
    operation_keywords: set[str] | None = Field(
        default=None,
        description="Keywords to identify operation methods; uses mixin defaults if None",
    )
    exclude_prefixes: set[str] | None = Field(
        default=None,
        description="Prefixes to exclude from capability discovery; uses mixin defaults if None",
    )


__all__: list[str] = ["ModelIntrospectionConfig"]
