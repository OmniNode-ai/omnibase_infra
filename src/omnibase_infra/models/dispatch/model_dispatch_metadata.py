# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Dispatch Metadata Model.

Provides strongly-typed metadata for dispatch operations, replacing raw dict usage
while maintaining extensibility for custom fields.

Design Pattern:
    ModelDispatchMetadata is a pure data model that captures common dispatch
    metadata fields with strong typing:
    - Source and target node identifiers
    - Retry information
    - Routing decision explanations

    The model uses `extra="allow"` to permit custom metadata fields beyond
    the known typed fields, enabling extensibility for domain-specific needs
    without modifying the core model.

Thread Safety:
    ModelDispatchMetadata is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from omnibase_infra.models.dispatch import ModelDispatchMetadata
    >>>
    >>> # Create metadata with known fields
    >>> metadata = ModelDispatchMetadata(
    ...     source_node="user-service",
    ...     target_node="notification-service",
    ...     routing_decision="fanout to notification handlers",
    ... )
    >>>
    >>> # Create metadata with custom fields (extra="allow")
    >>> extended = ModelDispatchMetadata(
    ...     source_node="order-service",
    ...     custom_field="custom_value",
    ...     priority="high",
    ... )
    >>> extended.model_extra["custom_field"]
    'custom_value'

See Also:
    omnibase_infra.models.dispatch.ModelDispatchContext: Context with time injection
    omnibase_infra.models.dispatch.ModelDispatchResult: Dispatch operation result
"""

from pydantic import BaseModel, ConfigDict, Field


class ModelDispatchMetadata(BaseModel):
    """
    Dispatch operation metadata with known fields and extensibility.

    Provides strongly-typed common metadata fields while allowing custom
    metadata via extra="allow" config. This replaces raw dict[str, str]
    usage with a proper Pydantic model that maintains type safety for
    known fields.

    Attributes:
        source_node: Identifier of the node that originated the dispatch.
        target_node: Identifier of the target node receiving the dispatch.
        retry_reason: Explanation for why this dispatch is a retry attempt.
        routing_decision: Description of the routing decision that was made.

    Example:
        >>> # Basic usage with known fields
        >>> meta = ModelDispatchMetadata(
        ...     source_node="event-processor",
        ...     target_node="state-reducer",
        ... )
        >>>
        >>> # With custom extensibility fields
        >>> meta = ModelDispatchMetadata(
        ...     source_node="api-gateway",
        ...     custom_trace_id="abc123",
        ...     environment="production",
        ... )
        >>> # Access custom fields via model_extra
        >>> meta.model_extra["custom_trace_id"]
        'abc123'
    """

    model_config = ConfigDict(
        frozen=True,
        extra="allow",
        from_attributes=True,
    )

    # ---- Known Dispatch Metadata Fields ----
    source_node: str | None = Field(
        default=None,
        description="Identifier of the node that originated the dispatch.",
    )
    target_node: str | None = Field(
        default=None,
        description="Identifier of the target node receiving the dispatch.",
    )
    retry_reason: str | None = Field(
        default=None,
        description="Explanation for why this dispatch is a retry attempt.",
    )
    routing_decision: str | None = Field(
        default=None,
        description="Description of the routing decision that was made.",
    )


__all__ = ["ModelDispatchMetadata"]
