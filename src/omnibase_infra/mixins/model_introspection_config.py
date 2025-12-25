# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Introspection Configuration Model.

This module provides the Pydantic configuration model for the MixinNodeIntrospection
mixin, consolidating initialization parameters into a single typed configuration object.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from omnibase_infra.mixins.protocol_event_bus_like import ProtocolEventBusLike

logger = logging.getLogger(__name__)

# Valid node types in the ONEX 4-node architecture
# Accept both uppercase (used in tests/contracts) and lowercase (EnumHandlerType values)
VALID_NODE_TYPES = frozenset(
    {
        "EFFECT",
        "COMPUTE",
        "REDUCER",
        "ORCHESTRATOR",
        "effect",
        "compute",
        "reducer",
        "orchestrator",
    }
)

# Maximum cache TTL in seconds (24 hours)
# Prevents unreasonably long cache durations that could cause stale data issues
MAX_CACHE_TTL_SECONDS = 86400.0

# Default topic names for introspection events
DEFAULT_INTROSPECTION_TOPIC = "node.introspection"
DEFAULT_HEARTBEAT_TOPIC = "node.heartbeat"
DEFAULT_REQUEST_INTROSPECTION_TOPIC = "node.request_introspection"

# Topic validation pattern
# Valid topics: lowercase alphanumeric with dots, hyphens, and underscores
# Must start with a letter and not have consecutive dots
TOPIC_VALIDATION_PATTERN = re.compile(r"^[a-z][a-z0-9._-]*[a-z0-9]$|^[a-z]$")

# Version suffix validation pattern
# Topics should end with .v followed by one or more digits (e.g., .v1, .v2, .v123)
VERSION_SUFFIX_PATTERN = re.compile(r"\.v[0-9]+$")

# ONEX topic prefix - topics starting with this prefix require version suffix
ONEX_TOPIC_PREFIX = "onex."


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
    introspection_topic: str = Field(
        default=DEFAULT_INTROSPECTION_TOPIC,
        min_length=1,
        description="Custom topic for introspection events (default: node.introspection)",
    )
    heartbeat_topic: str = Field(
        default=DEFAULT_HEARTBEAT_TOPIC,
        min_length=1,
        description="Custom topic for heartbeat events (default: node.heartbeat)",
    )
    request_introspection_topic: str = Field(
        default=DEFAULT_REQUEST_INTROSPECTION_TOPIC,
        min_length=1,
        description="Custom topic for request introspection events (default: node.request_introspection)",
    )

    @field_validator("node_type", mode="after")
    @classmethod
    def validate_node_type(cls, v: str) -> str:
        """Validate node_type against ONEX 4-node architecture types.

        Args:
            v: Node type value after Pydantic's initial validation.

        Returns:
            Validated node type string.

        Raises:
            ValueError: If node_type is not a valid ONEX node type.
        """
        if v not in VALID_NODE_TYPES:
            valid_types = ", ".join(sorted(t for t in VALID_NODE_TYPES if t.isupper()))
            raise ValueError(f"Invalid node_type '{v}'. Must be one of: {valid_types}")
        return v

    @field_validator("cache_ttl", mode="after")
    @classmethod
    def validate_cache_ttl_upper_bound(cls, v: float) -> float:
        """Validate cache_ttl has a reasonable upper bound.

        Args:
            v: Cache TTL value in seconds after Pydantic's initial validation.

        Returns:
            Validated cache TTL value.

        Raises:
            ValueError: If cache_ttl exceeds MAX_CACHE_TTL_SECONDS (24 hours).
        """
        if v > MAX_CACHE_TTL_SECONDS:
            raise ValueError(
                f"cache_ttl {v} exceeds maximum allowed value of "
                f"{MAX_CACHE_TTL_SECONDS} seconds (24 hours)"
            )
        return v

    @field_validator(
        "introspection_topic",
        "heartbeat_topic",
        "request_introspection_topic",
        mode="after",
    )
    @classmethod
    def validate_topic_format(cls, v: str) -> str:
        """Validate topic name format.

        Topic names must:
        - Start with a lowercase letter
        - Contain only lowercase alphanumeric, dots, hyphens, and underscores
        - Not contain consecutive dots
        - Not contain special characters (@, #, $, %, etc.)
        - Not contain whitespace

        Additionally:
        - ONEX topics (starting with 'onex.') MUST have version suffix (.v1, .v2, etc.)
        - Legacy topics (not starting with 'onex.') are allowed without version suffix
          but will log a warning suggesting version suffix adoption

        Args:
            v: Topic name after Pydantic's initial validation.

        Returns:
            Validated topic name.

        Raises:
            ValueError: If topic name contains invalid characters, format,
                or if ONEX topic is missing required version suffix.
        """
        # Check for whitespace
        if " " in v or "\t" in v or "\n" in v:
            raise ValueError(f"Topic '{v}' must not contain whitespace")

        # Check for consecutive dots
        if ".." in v:
            raise ValueError(f"Topic '{v}' must not contain consecutive dots")

        # Check for special characters
        invalid_chars = set("@#$%^&*+=<>[]{}|\\;:'\"")
        found_invalid = [c for c in v if c in invalid_chars]
        if found_invalid:
            raise ValueError(
                f"Topic '{v}' contains invalid characters: {found_invalid}"
            )

        # Check topic pattern
        if not TOPIC_VALIDATION_PATTERN.match(v):
            raise ValueError(
                f"Topic '{v}' must start with a lowercase letter and contain "
                "only lowercase alphanumeric characters, dots, hyphens, and underscores"
            )

        # Version suffix validation
        has_version_suffix = VERSION_SUFFIX_PATTERN.search(v) is not None
        is_onex_topic = v.startswith(ONEX_TOPIC_PREFIX)

        if is_onex_topic and not has_version_suffix:
            # ONEX topics MUST have version suffix
            raise ValueError(
                f"ONEX topic '{v}' must end with a version suffix (e.g., .v1, .v2). "
                "ONEX topics require explicit versioning for schema evolution."
            )

        if not is_onex_topic and not has_version_suffix:
            # Legacy topics: warn but allow (backwards compatibility)
            # Use warnings module for Pydantic validator compatibility
            # (logging in validators can be problematic during model construction)
            warnings.warn(
                f"Topic '{v}' does not have a version suffix (e.g., .v1). "
                "Consider adding a version suffix for better schema evolution support.",
                UserWarning,
                stacklevel=2,
            )

        return v


__all__: list[str] = [
    "ModelIntrospectionConfig",
    "VALID_NODE_TYPES",
    "MAX_CACHE_TTL_SECONDS",
    "DEFAULT_INTROSPECTION_TOPIC",
    "DEFAULT_HEARTBEAT_TOPIC",
    "DEFAULT_REQUEST_INTROSPECTION_TOPIC",
    "TOPIC_VALIDATION_PATTERN",
    "VERSION_SUFFIX_PATTERN",
    "ONEX_TOPIC_PREFIX",
]
