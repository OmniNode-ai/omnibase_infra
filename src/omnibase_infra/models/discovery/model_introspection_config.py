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
    - Legacy topics (not starting with 'onex.') are allowed for flexibility

See Also:
    - docs/architecture/EVENT_STREAMING_TOPICS.md: Topic naming conventions
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.contracts import ModelContractBase
from omnibase_infra.topics import (
    SUFFIX_NODE_HEARTBEAT,
    SUFFIX_NODE_INTROSPECTION,
    SUFFIX_REQUEST_INTROSPECTION,
)

if TYPE_CHECKING:
    from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus

    # Type alias for event_bus field - provides proper type during static analysis
    type _EventBusType = ProtocolEventBus | None
else:
    # At runtime, use object | None for duck typing compatibility with Pydantic
    # Pydantic cannot resolve TYPE_CHECKING-only imports, so we use object
    # The mixin enforces protocol compliance at initialization
    type _EventBusType = object | None

logger = logging.getLogger(__name__)

# Default topic constants using ONEX platform suffix constants
# These are the canonical source for introspection-related topic defaults
DEFAULT_INTROSPECTION_TOPIC = SUFFIX_NODE_INTROSPECTION
DEFAULT_HEARTBEAT_TOPIC = SUFFIX_NODE_HEARTBEAT
DEFAULT_REQUEST_INTROSPECTION_TOPIC = SUFFIX_REQUEST_INTROSPECTION

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
        operation_keywords: Optional frozenset of keywords to identify operation methods.
            Methods containing these keywords are reported as operations.
            If None, uses MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS.
        exclude_prefixes: Optional frozenset of prefixes to exclude from capability
            discovery. Methods starting with these prefixes are filtered out.
            If None, uses MixinNodeIntrospection.DEFAULT_EXCLUDE_PREFIXES.
        introspection_topic: Topic for publishing introspection events.
            Defaults to SUFFIX_NODE_INTROSPECTION (onex.evt.platform.node-introspection.v1).
            ONEX topics (onex.*) require version suffix (.v1, .v2, etc.).
        heartbeat_topic: Topic for publishing heartbeat events.
            Defaults to SUFFIX_NODE_HEARTBEAT (onex.evt.platform.node-heartbeat.v1).
            ONEX topics (onex.*) require version suffix (.v1, .v2, etc.).
        request_introspection_topic: Topic for receiving introspection requests.
            Defaults to SUFFIX_REQUEST_INTROSPECTION (onex.cmd.platform.request-introspection.v1).
            ONEX topics (onex.*) require version suffix (.v1, .v2, etc.).
        contract: Optional typed contract model for capability extraction.
            When provided, MixinNodeIntrospection extracts contract_capabilities
            using ContractCapabilityExtractor. None for legacy nodes.

    Example:
        ```python
        from uuid import UUID, uuid4
        from omnibase_core.enums import EnumNodeKind
        from omnibase_infra.models.discovery import ModelIntrospectionConfig
        from omnibase_infra.mixins import MixinNodeIntrospection

        class MyNode(MixinNodeIntrospection):
            def __init__(self, node_id: UUID, event_bus=None):
                config = ModelIntrospectionConfig(
                    node_id=node_id,
                    node_type=EnumNodeKind.EFFECT,  # Use enum directly (preferred)
                    event_bus=event_bus,
                    version="1.2.0",
                )
                self.initialize_introspection(config)

        # With custom operation keywords
        class MyEffectNode(MixinNodeIntrospection):
            def __init__(self, node_id: UUID | None = None, event_bus=None):
                config = ModelIntrospectionConfig(
                    node_id=node_id or uuid4(),
                    node_type=EnumNodeKind.EFFECT,  # Use enum directly (preferred)
                    event_bus=event_bus,
                    operation_keywords=frozenset({"fetch", "upload", "download"}),
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

    node_type: EnumNodeKind = Field(
        ...,
        description="Node type classification (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR). "
        "Accepts EnumNodeKind directly (preferred) or string (deprecated, will be coerced).",
    )

    # Event bus for publishing introspection events.
    # Uses _EventBusType which provides:
    # - ProtocolEventBus | None during static analysis (TYPE_CHECKING)
    # - object | None at runtime for Pydantic compatibility
    # Duck typing is enforced by the mixin at initialization.
    # The model config has arbitrary_types_allowed=True to support arbitrary objects.
    event_bus: _EventBusType = Field(
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

    operation_keywords: frozenset[str] | None = Field(
        default=None,
        description="Optional frozenset of keywords to identify operation methods. "
        "If None, uses MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS.",
    )

    exclude_prefixes: frozenset[str] | None = Field(
        default=None,
        description="Optional frozenset of prefixes to exclude from capability discovery. "
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

    contract: ModelContractBase | None = Field(
        default=None,
        description="Typed contract model for capability extraction. "
        "When provided, MixinNodeIntrospection will extract contract_capabilities "
        "using ContractCapabilityExtractor. None for legacy nodes without contracts.",
    )

    @field_validator("node_type", mode="before")
    @classmethod
    def validate_node_type(cls, v: object) -> EnumNodeKind:
        """Validate and coerce node_type to EnumNodeKind.

        Args:
            v: Node type value to validate. Accepts EnumNodeKind directly
               (preferred) or string (deprecated).

        Returns:
            EnumNodeKind instance.

        Raises:
            ValueError: If string node type is empty or invalid.
        """
        if isinstance(v, EnumNodeKind):
            return v
        if not isinstance(v, str):
            raise ValueError(
                f"node_type must be EnumNodeKind or str, got {type(v).__name__}"
            )
        if not v:
            raise ValueError("node_type cannot be empty")
        # Coerce string to EnumNodeKind (handles both "EFFECT" and "effect")
        try:
            return EnumNodeKind(v.lower())
        except ValueError:
            valid = ", ".join(e.value for e in EnumNodeKind)
            raise ValueError(
                f"Invalid node_type '{v}'. Must be one of: {valid}"
            ) from None

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
        # Legacy topics (not starting with 'onex.') are allowed without version suffix

        return v

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow arbitrary types for event_bus
        json_schema_extra={
            "examples": [
                # First example: Default topic configuration using suffix constants
                {
                    "node_id": "550e8400-e29b-41d4-a716-446655440000",
                    "node_type": "EFFECT",
                    "event_bus": None,
                    "version": "1.0.0",
                    "cache_ttl": 300.0,
                    "operation_keywords": None,
                    "exclude_prefixes": None,
                    "introspection_topic": "onex.evt.platform.node-introspection.v1",
                    "heartbeat_topic": "onex.evt.platform.node-heartbeat.v1",
                    "request_introspection_topic": "onex.cmd.platform.request-introspection.v1",
                },
                # Second example: Custom prefixed topics for environment isolation
                # Demonstrates adding env/namespace prefix to suffix constants
                {
                    "node_id": "550e8400-e29b-41d4-a716-446655440001",
                    "node_type": "COMPUTE",
                    "event_bus": None,
                    "version": "2.1.0",
                    "cache_ttl": 120.0,
                    "operation_keywords": ["process", "transform", "analyze"],
                    "exclude_prefixes": ["_internal", "_private"],
                    "introspection_topic": "prod.myapp.onex.evt.platform.node-introspection.v1",
                    "heartbeat_topic": "prod.myapp.onex.evt.platform.node-heartbeat.v1",
                    "request_introspection_topic": "prod.myapp.onex.cmd.platform.request-introspection.v1",
                },
            ]
        },
    )


__all__ = [
    "DEFAULT_HEARTBEAT_TOPIC",
    "DEFAULT_INTROSPECTION_TOPIC",
    "DEFAULT_REQUEST_INTROSPECTION_TOPIC",
    "INVALID_TOPIC_CHARS",
    "TOPIC_PATTERN",
    "VERSION_SUFFIX_PATTERN",
    "ModelIntrospectionConfig",
]
