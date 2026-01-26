# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Intent Storage Output Model for Intent Storage Operations.

This module provides ModelIntentStorageOutput, representing the result
of storing a classified intent in the graph database.

Architecture:
    ModelIntentStorageOutput is returned from intent.store operations
    and contains:
    - success: Whether the storage operation succeeded
    - node_id: The graph node ID of the stored intent
    - element_id: The graph element ID for the stored intent
    - labels: Graph labels applied to the node
    - properties: The stored properties (echoed back for verification)
    - correlation_id: Correlation ID for request tracing
    - error: Sanitized error message if operation failed

    This model serves as the canonical output for the intent.store operation.

Related:
    - NodeIntentStorageEffect: Effect node that produces this output
    - ModelIntentStorageInput: Input model for storage operations
    - HandlerIntent: Handler that executes storage operations
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelIntentStorageOutput(BaseModel):
    """Output model for intent storage operations.

    Contains the result of storing a classified intent as a graph node,
    including the assigned node ID and stored properties.

    Immutability:
        This model uses frozen=True to ensure outputs are immutable
        once created, enabling safe sharing and caching.

    Attributes:
        success: Whether the storage operation completed successfully.
        node_id: The graph database node ID of the stored intent.
        element_id: The graph element ID for the stored intent.
        labels: Graph labels applied to the node (e.g., ["Intent"]).
        properties: The stored properties (for verification).
        correlation_id: Correlation ID for request tracing (required).
        error: Sanitized error message if operation failed.
        duration_ms: Time taken for the operation in milliseconds.

    Example:
        >>> output = ModelIntentStorageOutput(
        ...     success=True,
        ...     node_id="123",
        ...     element_id="4:abc:123",
        ...     labels=("Intent",),
        ...     properties={"intent_type": "query", "session_id": "sess-12345"},
        ...     correlation_id=correlation_id,
        ...     duration_ms=15.2,
        ... )
        >>> if output:  # Uses custom __bool__
        ...     print(f"Stored intent: {output.node_id}")
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the storage operation completed successfully",
    )
    node_id: str | None = Field(
        default=None,
        description="The graph database node ID of the stored intent",
    )
    element_id: str | None = Field(
        default=None,
        description="The graph element ID for the stored intent",
    )
    labels: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Graph labels applied to the node",
    )
    properties: dict[str, JsonType] = Field(
        default_factory=dict,
        description="The stored properties (for verification)",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for request tracing (required for traceability)",
    )
    error: str | None = Field(
        default=None,
        description="Sanitized error message if operation failed",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Time taken for the operation in milliseconds",
        ge=0.0,
    )

    def __bool__(self) -> bool:
        """Return True if the storage operation succeeded.

        Warning:
            This is non-standard Pydantic behavior. Normally bool(model)
            returns True for any instance. This override enables idiomatic
            `if result:` checks for success validation.

            See: docs/decisions/adr-custom-bool-result-models.md

        Returns:
            True if success is True, False otherwise.
        """
        return self.success

    def has_node_id(self) -> bool:
        """Check if a node ID was assigned.

        Returns:
            True if node_id is present.
        """
        return self.node_id is not None


__all__ = ["ModelIntentStorageOutput"]
