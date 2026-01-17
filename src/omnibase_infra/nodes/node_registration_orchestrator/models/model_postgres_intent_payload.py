# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL intent payload model for registration orchestrator.

This module provides the typed payload model for PostgreSQL registration intents.

Design Note:
    This model uses strongly-typed ModelNodeCapabilities and ModelNodeMetadata
    instead of generic dict[str, JsonType] to adhere to ONEX "no Any types"
    principle. The JsonType type alias contains list[Any] and dict[str, Any]
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

Edge Case Behavior:
    The ``endpoints`` field validator explicitly handles the following cases:
    - ``None``: Raises ValueError (invalid input, not silently ignored)
    - Empty Mapping ``{}``: Logs warning and converts to empty tuple
    - Invalid types (int, str, list, etc.): Raises ValueError
    - Tuple: Passed through as-is
    - Non-empty Mapping: Converted to tuple of (key, value) pairs
    - Non-string keys/values: Raises ValueError (strict mode)
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from types import MappingProxyType
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata

# Module-level initialization (after all imports)
logger = logging.getLogger(__name__)


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
        >>> from omnibase_core.enums import EnumNodeKind
        >>> payload = ModelPostgresIntentPayload(
        ...     node_id=uuid4(),
        ...     node_type=EnumNodeKind.EFFECT,
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
    # Design Note: node_type uses EnumNodeKind for type-safe ONEX node type validation.
    # This ensures only valid ONEX node types can be persisted to PostgreSQL.
    node_type: EnumNodeKind = Field(..., description="ONEX node type")
    node_version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer(major=1, minor=0, patch=0),
        description="Semantic version of the node",
    )
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
    def _coerce_endpoints_to_tuple(cls, v: object) -> tuple[tuple[str, str], ...]:
        """Convert dict/mapping to tuple of pairs for immutability.

        This validator ensures explicit handling of all input types rather than
        silent fallback to empty tuple, which could mask invalid input. In strict
        mode, all keys and values must already be strings - no silent type coercion.

        Args:
            v: The input value to coerce. Must be either a tuple of (key, value)
                pairs or a Mapping (dict-like object) with string keys and values.

        Returns:
            A tuple of (key, value) string pairs.

        Raises:
            ValueError: If the input is neither a tuple nor a Mapping type, or if
                any key/value is not a string. This ensures invalid input types
                are explicitly rejected rather than silently coerced.

        Edge Cases:
            - ``None``: Raises ValueError (explicit rejection)
            - Empty Mapping ``{}``: Logs warning, returns empty tuple
            - Empty tuple ``()``: Passed through (same as default)
            - Invalid types (list, int, str): Raises ValueError
            - Non-string keys/values: Raises ValueError (strict mode)
            - Non-empty Mapping with strings: Converts to tuple of (key, value) pairs

        Example:
            >>> # Valid inputs
            >>> _coerce_endpoints_to_tuple({"health": "/health"})
            (('health', '/health'),)
            >>> _coerce_endpoints_to_tuple(())
            ()
            >>> # Invalid inputs raise ValueError
            >>> _coerce_endpoints_to_tuple(None)  # Raises ValueError
            >>> _coerce_endpoints_to_tuple([])    # Raises ValueError (list not Mapping)
            >>> _coerce_endpoints_to_tuple({1: "/health"})  # Raises ValueError (non-string key)
        """
        if isinstance(v, tuple):
            # Validate tuple contents in strict mode
            for i, item in enumerate(v):
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError(
                        f"endpoints[{i}] must be a (key, value) tuple, "
                        f"got {type(item).__name__}"
                    )
                key, val = item
                if not isinstance(key, str):
                    raise ValueError(
                        f"endpoints[{i}][0] (key) must be a string, "
                        f"got {type(key).__name__}"
                    )
                if not isinstance(val, str):
                    raise ValueError(
                        f"endpoints[{i}][1] (value) must be a string, "
                        f"got {type(val).__name__}"
                    )
            return v  # type: ignore[return-value]  # NOTE: runtime type validated by Pydantic
        if isinstance(v, Mapping):
            if len(v) == 0:
                # Log warning for empty Mapping to help detect potentially missing data.
                # This is different from the default empty tuple - it's an explicit
                # empty Mapping input that gets coerced.
                warning_msg = (
                    "Empty Mapping provided for endpoints, coercing to empty tuple. "
                    "If this is intentional, consider using default=() instead."
                )
                logger.warning(warning_msg)
                warnings.warn(warning_msg, UserWarning, stacklevel=2)
                return ()
            # Validate and convert to tuple - strict mode requires string keys/values
            result: list[tuple[str, str]] = []
            for key, val in v.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"endpoints key must be a string, got {type(key).__name__}"
                    )
                if not isinstance(val, str):
                    raise ValueError(
                        f"endpoints[{key!r}] value must be a string, "
                        f"got {type(val).__name__}"
                    )
                result.append((key, val))
            return tuple(result)
        raise ValueError(
            f"endpoints must be a tuple or Mapping, got {type(v).__name__}"
        )

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
