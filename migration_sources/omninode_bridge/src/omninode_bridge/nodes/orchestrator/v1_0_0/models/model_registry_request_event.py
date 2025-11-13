#!/usr/bin/env python3
"""
Registry Request Event Model for Node Discovery.

Defines the structure for registry introspection requests,
used to trigger nodes to broadcast their capabilities.

ONEX v2.0 Compliance:
- Model-based naming: ModelRegistryRequestEvent
- Strong typing with Pydantic v2
- Integration with OnexEnvelopeV1 event publishing
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class EnumIntrospectionReason(str, Enum):
    """Reasons for requesting node introspection."""

    STARTUP = "startup"
    """Registry startup - discover all available nodes."""

    STARTUP_REBROADCAST = "startup_rebroadcast"
    """Registry startup rebroadcast - request all nodes to re-broadcast introspection."""

    RECOVERY = "recovery"
    """Registry recovery - rebuild node registry after failure."""

    REFRESH = "refresh"
    """Periodic refresh - update node capabilities and status."""

    MANUAL = "manual"
    """Manual request - admin-triggered discovery."""

    NODE_JOIN = "node_join"
    """New node joined - request introspection from all nodes."""


class ModelRegistryRequestEvent(BaseModel):
    """
    Registry request event for triggering node introspection broadcasts.

    This model represents the payload for REGISTRY_REQUEST_INTROSPECTION events,
    which are broadcast by the registry to request all nodes to respond with
    their introspection data.

    Attributes:
        registry_id: Unique identifier for the registry instance
        request_timestamp: When the introspection request was initiated
        reason: Reason for requesting introspection (startup, recovery, etc.)
        target_node_types: Optional list of specific node types to respond
        response_timeout_ms: How long nodes should wait before responding
        metadata: Additional request metadata

    Usage:
        Used as payload in OnexEnvelopeV1 for REGISTRY_REQUEST_INTROSPECTION events.
        Broadcast by registry, consumed by all nodes who respond with NODE_INTROSPECTION.
    """

    model_config = ConfigDict(extra="forbid")

    registry_id: str = Field(
        ...,
        description="Unique identifier for the registry instance requesting introspection",
    )

    request_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the introspection request was initiated",
    )

    reason: EnumIntrospectionReason = Field(
        ...,
        description="Reason for requesting introspection from nodes",
    )

    target_node_types: Optional[list[str]] = Field(
        default=None,
        description="Optional filter for specific node types (e.g., ['orchestrator', 'effect'])",
        examples=[["orchestrator", "effect"], ["reducer"], None],
    )

    response_timeout_ms: int = Field(
        default=5000,
        description="Maximum time nodes should wait before responding (milliseconds)",
        gt=0,
        le=30000,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata",
        examples=[
            {
                "registry_version": "1.0.0",
                "environment": "production",
                "region": "us-west-2",
                "expected_node_count": 10,
            }
        ],
    )

    correlation_id: Optional[UUID] = Field(
        default=None,
        description="Optional correlation ID for tracking related events",
    )

    @field_serializer("request_timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize timestamp to ISO format."""
        return value.isoformat()

    @field_serializer("correlation_id")
    def serialize_correlation_id(self, value: Optional[UUID]) -> Optional[str]:
        """Serialize correlation_id to string."""
        return str(value) if value is not None else None

    @classmethod
    def create(
        cls,
        registry_id: str,
        reason: EnumIntrospectionReason,
        target_node_types: Optional[list[str]] = None,
        response_timeout_ms: int = 5000,
        metadata: Optional[dict[str, Any]] = None,
        correlation_id: Optional[UUID] = None,
    ) -> "ModelRegistryRequestEvent":
        """
        Factory method to create a registry request event with defaults.

        Args:
            registry_id: Unique registry identifier
            reason: Reason for introspection request
            target_node_types: Optional list of node types to target
            response_timeout_ms: Response timeout in milliseconds
            metadata: Optional metadata dictionary
            correlation_id: Optional correlation ID

        Returns:
            ModelRegistryRequestEvent instance
        """
        return cls(
            registry_id=registry_id,
            reason=reason,
            target_node_types=target_node_types,
            response_timeout_ms=response_timeout_ms,
            metadata=metadata or {},
            correlation_id=correlation_id,
        )

    def should_node_respond(self, node_type: str) -> bool:
        """
        Check if a node of given type should respond to this request.

        Args:
            node_type: Type of the node checking (effect, compute, reducer, orchestrator)

        Returns:
            True if node should respond, False otherwise
        """
        if self.target_node_types is None:
            return True  # All nodes should respond
        return node_type in self.target_node_types

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for use in OnexEnvelopeV1 payload.

        Returns:
            Dictionary representation suitable for event payload
        """
        return self.model_dump(mode="json")
