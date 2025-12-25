# Copyright 2025 OmniNode Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration model for node introspection initialization.

This module provides the configuration model used to initialize the
MixinNodeIntrospection mixin. Grouping parameters into a configuration
model follows ONEX patterns for reducing function parameter count.

Topic Validation:
    Topics must follow ONEX naming conventions:
    - Must start with a lowercase letter
    - Can contain lowercase alphanumeric characters, dots, hyphens, and underscores
    - ONEX topics (starting with 'onex.') require a version suffix (.v1, .v2, etc.)
    - Legacy topics (not starting with 'onex.') generate a warning but are allowed
      for backward compatibility
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus

logger = logging.getLogger(__name__)

# Default topic constants following ONEX legacy conventions
# Note: These use the legacy "node." prefix for backward compatibility
DEFAULT_INTROSPECTION_TOPIC = "node.introspection"
DEFAULT_HEARTBEAT_TOPIC = "node.heartbeat"
DEFAULT_REQUEST_INTROSPECTION_TOPIC = "node.request_introspection"

# Topic validation patterns
# Matches valid topic characters: lowercase alphanumeric, dots, hyphens, underscores
TOPIC_PATTERN = re.compile(r"^[a-z][a-z0-9._-]*[a-z0-9]$|^[a-z]$")
# Invalid characters that should never appear in topic names
INVALID_TOPIC_CHARS = set("@#$%^&*()+=[]{}|\\:;\"'<>,?/! \t\n\r")
# Version suffix pattern for ONEX topics
VERSION_SUFFIX_PATTERN = re.compile(r"\.v[0-9]+$")


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
            Must implement ``ProtocolEventBus`` protocol.
        version: Node version string. Defaults to "1.0.0".
        cache_ttl: Cache time-to-live in seconds. Defaults to 300.0 (5 minutes).
        operation_keywords: Optional set of keywords to identify operation methods.
            Methods containing these keywords are reported as operations.
            If None, uses MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS.
        exclude_prefixes: Optional set of prefixes to exclude from capability
            discovery. Methods starting with these prefixes are filtered out.
            If None, uses MixinNodeIntrospection.DEFAULT_EXCLUDE_PREFIXES.
        introspection_topic: Topic for publishing introspection events.
            Defaults to "node.introspection". ONEX topics (onex.*) require
            version suffix (.v1, .v2, etc.).
        heartbeat_topic: Topic for publishing heartbeat events.
            Defaults to "node.heartbeat". ONEX topics (onex.*) require
            version suffix (.v1, .v2, etc.).
        request_introspection_topic: Topic for receiving introspection requests.
            Defaults to "node.request_introspection". ONEX topics (onex.*)
            require version suffix (.v1, .v2, etc.).

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

    # Event bus for publishing introspection events.
    # Uses `object | None` for runtime type since ProtocolEventBus is only available
    # during TYPE_CHECKING. Duck typing is enforced by the mixin at initialization.
    # The model config has arbitrary_types_allowed=True to support arbitrary objects.
    event_bus: object | None = Field(
        default=None,
        description="Optional event bus for publishing introspection events. "
        "Must implement ProtocolEventBus protocol (duck typed).",
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

    introspection_topic: str = Field(
        default=DEFAULT_INTROSPECTION_TOPIC,
        description="Topic for publishing introspection events. "
        "ONEX topics (onex.*) require version suffix (.v1, .v2, etc.).",
    )

    heartbeat_topic: str = Field(
        default=DEFAULT_HEARTBEAT_TOPIC,
        description="Topic for publishing heartbeat events. "
        "ONEX topics (onex.*) require version suffix (.v1, .v2, etc.).",
    )

    request_introspection_topic: str = Field(
        default=DEFAULT_REQUEST_INTROSPECTION_TOPIC,
        description="Topic for receiving introspection request events. "
        "ONEX topics (onex.*) require version suffix (.v1, .v2, etc.).",
    )

    @field_validator(
        "introspection_topic", "heartbeat_topic", "request_introspection_topic"
    )
    @classmethod
    def validate_topic_name(cls, v: str) -> str:
        """Validate topic name follows ONEX conventions.

        Args:
            v: Topic name to validate.

        Returns:
            Validated topic name.

        Raises:
            ValueError: If topic name is invalid.
        """
        if not v:
            raise ValueError("Topic name cannot be empty")

        # Check for invalid characters first
        invalid_found = set(v) & INVALID_TOPIC_CHARS
        if invalid_found:
            raise ValueError(f"Topic name contains invalid characters: {invalid_found}")

        # Check pattern (must start with lowercase, valid characters)
        if not TOPIC_PATTERN.match(v):
            if v[0].isupper():
                raise ValueError(f"Topic name must start with a lowercase letter: {v}")
            if v.endswith("."):
                raise ValueError(
                    f"Topic name can only lowercase alphanumeric, dot, hyphen, "
                    f"underscore, and must not end with a dot: {v}"
                )
            raise ValueError(
                f"Topic name must contain only lowercase alphanumeric, "
                f"dot, hyphen, underscore characters: {v}"
            )

        # ONEX topics require version suffix
        if v.startswith("onex."):
            if not VERSION_SUFFIX_PATTERN.search(v):
                raise ValueError(
                    f"ONEX topic must have version suffix (.v1, .v2, etc.): {v}"
                )
        # Legacy topics get a warning but are allowed
        elif not VERSION_SUFFIX_PATTERN.search(v):
            warnings.warn(
                f"Topic '{v}' does not have version suffix. "
                "Consider using ONEX format: onex.<domain>.<name>.v1",
                UserWarning,
                stacklevel=2,
            )

        return v

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
                    "introspection_topic": "node.introspection",
                    "heartbeat_topic": "node.heartbeat",
                    "request_introspection_topic": "node.request_introspection",
                },
                {
                    "node_id": "550e8400-e29b-41d4-a716-446655440001",
                    "node_type": "COMPUTE",
                    "event_bus": None,
                    "version": "2.1.0",
                    "cache_ttl": 120.0,
                    "operation_keywords": ["process", "transform", "analyze"],
                    "exclude_prefixes": ["_internal", "_private"],
                    "introspection_topic": "onex.node.introspection.published.v1",
                    "heartbeat_topic": "onex.node.heartbeat.published.v1",
                    "request_introspection_topic": "onex.registry.introspection.requested.v1",
                },
            ]
        },
    )


__all__ = [
    "ModelIntrospectionConfig",
    "DEFAULT_INTROSPECTION_TOPIC",
    "DEFAULT_HEARTBEAT_TOPIC",
    "DEFAULT_REQUEST_INTROSPECTION_TOPIC",
    "TOPIC_PATTERN",
    "VERSION_SUFFIX_PATTERN",
    "INVALID_TOPIC_CHARS",
]
