# Copyright 2025 OmniNode Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration model for node introspection initialization.

This module provides the configuration model used to initialize the
MixinNodeIntrospection mixin. Grouping parameters into a configuration
model follows ONEX patterns for reducing function parameter count.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelIntrospectionConfig(BaseModel):
    """Configuration model for introspection initialization.

    This model groups all parameters required by ``initialize_introspection()``
    into a single configuration object, following ONEX conventions for functions
    with more than 5 parameters.

    Attributes:
        node_id: Unique identifier for this node instance (UUID).
        node_type: Node type classification (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).
            Cannot be empty.
        event_bus: Optional event bus for publishing introspection events.
            Must have ``publish_envelope()`` method if provided.
        version: Node version string. Defaults to "1.0.0".
        cache_ttl: Cache time-to-live in seconds. Defaults to 300.0 (5 minutes).
        operation_keywords: Optional set of keywords to identify operation methods.
            Methods containing these keywords are reported as operations.
            If None, uses MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS.
        exclude_prefixes: Optional set of prefixes to exclude from capability
            discovery. Methods starting with these prefixes are filtered out.
            If None, uses MixinNodeIntrospection.DEFAULT_EXCLUDE_PREFIXES.

    Example:
        ```python
        from uuid import UUID, uuid4
        from omnibase_infra.models.discovery import ModelIntrospectionConfig
        from omnibase_infra.mixins import MixinNodeIntrospection

        class MyNode(MixinNodeIntrospection):
            def __init__(self, node_id: UUID, event_bus=None):
                config = ModelIntrospectionConfig(
                    node_id=node_id,
                    node_type="EFFECT",
                    event_bus=event_bus,
                    version="1.2.0",
                )
                self.initialize_introspection(config)

        # With custom operation keywords
        class MyEffectNode(MixinNodeIntrospection):
            def __init__(self, node_id: UUID | None = None, event_bus=None):
                config = ModelIntrospectionConfig(
                    node_id=node_id or uuid4(),
                    node_type="EFFECT",
                    event_bus=event_bus,
                    operation_keywords={"fetch", "upload", "download"},
                )
                self.initialize_introspection(config)
        ```

    See Also:
        MixinNodeIntrospection: The mixin that uses this configuration.
        ModelNodeIntrospectionEvent: Event model for introspection events.
    """

    node_id: UUID = Field(
        ...,
        description="Unique identifier for this node instance",
    )

    node_type: str = Field(
        ...,
        min_length=1,
        description="Node type classification (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)",
    )

    # Duck-typed event bus - accepts any object with publish_envelope() method.
    # Type annotation uses object for Pydantic runtime compatibility while
    # allowing static type checkers to infer ProtocolEventBus via duck typing.
    event_bus: object | None = Field(
        default=None,
        description="Optional event bus for publishing introspection events. "
        "Must have publish_envelope() method if provided.",
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
        description="Optional set of keywords to identify operation methods. "
        "If None, uses MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS.",
    )

    exclude_prefixes: set[str] | None = Field(
        default=None,
        description="Optional set of prefixes to exclude from capability discovery. "
        "If None, uses MixinNodeIntrospection.DEFAULT_EXCLUDE_PREFIXES.",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow arbitrary types for event_bus
        json_schema_extra={
            "examples": [
                {
                    "node_id": "550e8400-e29b-41d4-a716-446655440000",
                    "node_type": "EFFECT",
                    "event_bus": None,
                    "version": "1.0.0",
                    "cache_ttl": 300.0,
                    "operation_keywords": None,
                    "exclude_prefixes": None,
                },
                {
                    "node_id": "550e8400-e29b-41d4-a716-446655440001",
                    "node_type": "COMPUTE",
                    "event_bus": None,
                    "version": "2.1.0",
                    "cache_ttl": 120.0,
                    "operation_keywords": ["process", "transform", "analyze"],
                    "exclude_prefixes": ["_internal", "_private"],
                },
            ]
        },
    )


__all__ = ["ModelIntrospectionConfig"]
