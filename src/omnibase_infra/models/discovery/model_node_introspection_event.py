# Copyright 2025 OmniNode Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node introspection event model for capability discovery and reporting."""

from datetime import UTC, datetime
from typing import TypedDict
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.models.discovery.model_introspection_performance_metrics import (
    ModelIntrospectionPerformanceMetrics,
)


class CapabilitiesTypedDict(TypedDict, total=False):
    """Type-safe structure for node capabilities discovered via reflection.

    This TypedDict provides explicit typing for capability fields, eliminating
    the need for permissive `dict[str, object]` or `Any` types.

    Attributes:
        operations: List of public method names that may be operations.
            These are methods matching configured operation keywords
            (e.g., execute, handle, process).
        protocols: List of protocol/interface names implemented by the node.
            Discovered from class hierarchy (e.g., ProtocolDatabaseAdapter).
        has_fsm: Boolean indicating if node has FSM state management.
            True if state attributes like _state or current_state are found.
        method_signatures: Dict mapping public method names to their signature
            strings (e.g., {"execute": "(query: str) -> list[dict]"}).

    Example:
        >>> capabilities: CapabilitiesTypedDict = {
        ...     "operations": ["execute", "query", "batch_execute"],
        ...     "protocols": ["ProtocolDatabaseAdapter", "MixinNodeIntrospection"],
        ...     "has_fsm": True,
        ...     "method_signatures": {
        ...         "execute": "(query: str) -> list[dict]",
        ...         "query": "(sql: str, params: dict) -> list[dict]",
        ...     },
        ... }
    """

    operations: list[str]
    protocols: list[str]
    has_fsm: bool
    method_signatures: dict[str, str]


def _empty_capabilities() -> CapabilitiesTypedDict:
    """Factory function for creating an empty CapabilitiesTypedDict.

    Returns:
        An empty CapabilitiesTypedDict with default empty values.
    """
    # TypedDict with total=False allows empty dict
    return {}


class ModelNodeIntrospectionEvent(BaseModel):
    """Event model for node introspection and capability discovery.

    This model represents introspection events emitted by nodes to report their
    capabilities, endpoints, and current state. Used by MixinNodeIntrospection
    to broadcast node information to the event bus for service discovery and
    registry coordination.

    Attributes:
        node_id: Unique identifier for the node instance.
        node_type: Type classification of the node (e.g., EFFECT, COMPUTE).
        capabilities: Dictionary of node capabilities discovered via reflection.
        endpoints: Dictionary of endpoint URLs (health, api, metrics).
        current_state: Current FSM state if the node has state management.
        version: Node version string.
        reason: Reason for the introspection event (startup, shutdown, request).
        correlation_id: Required correlation ID for distributed tracing and idempotency.
        timestamp: UTC timestamp when the introspection was generated.
        performance_metrics: Optional performance metrics from the introspection operation.
            Contains timing data, cache hit/miss info, and threshold violation tracking.
            None when metrics were not captured or are unavailable.

    Example:
        ```python
        from datetime import UTC, datetime
        from uuid import uuid4

        from omnibase_infra.models.discovery import (
            ModelIntrospectionPerformanceMetrics,
            ModelNodeIntrospectionEvent,
        )

        # Basic event without performance metrics
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="EFFECT",
            capabilities={
                "operations": ["execute", "query", "batch_execute"],
                "protocols": ["ProtocolDatabaseAdapter"],
                "has_fsm": True,
            },
            endpoints={
                "health": "http://localhost:8080/health",
                "metrics": "http://localhost:8080/metrics",
            },
            current_state="connected",
            version="1.0.0",
            reason="startup",
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
        )

        # Event with performance metrics attached
        event_with_metrics = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="EFFECT",
            capabilities={"operations": ["execute"], "protocols": [], "has_fsm": False},
            endpoints={"health": "/health"},
            version="1.0.0",
            reason="request",
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            performance_metrics=ModelIntrospectionPerformanceMetrics(
                get_capabilities_ms=15.2,
                total_introspection_ms=18.5,
                cache_hit=True,
                method_count=5,
            ),
        )
        ```
    """

    node_id: UUID = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="Node type classification")

    # Capabilities discovered via reflection
    # Uses CapabilitiesTypedDict for type safety while maintaining Pydantic compatibility
    capabilities: CapabilitiesTypedDict = Field(
        default_factory=_empty_capabilities,
        description="Node capabilities discovered via reflection. "
        "Contains: operations (list[str]), protocols (list[str]), "
        "has_fsm (bool), method_signatures (dict[str, str])",
    )

    # Endpoint URLs
    #
    # Design Decision: We use dict[str, str] instead of dict[str, HttpUrl] for endpoints.
    #
    # Rationale:
    # 1. Flexibility: Internal endpoints may use non-standard schemes (e.g., "consul://",
    #    "grpc://", or relative paths like "/health") that HttpUrl rejects.
    # 2. Simplicity: Endpoint validation is deferred to the consuming services, which
    #    know their specific requirements and can apply appropriate validation.
    # 3. Performance: HttpUrl validation adds parsing overhead for every endpoint,
    #    which is unnecessary for internal service discovery where endpoints are
    #    typically constructed programmatically.
    # 4. Consistency: Other infrastructure components (e.g., Consul, Kubernetes)
    #    also use string-based endpoint formats.
    #
    # If strict URL validation is needed, consumers can wrap endpoint access with
    # validation logic or use pydantic.HttpUrl when processing individual endpoints.
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Endpoint URLs (health, api, metrics). Values are strings to "
        "support various URL schemes including internal service protocols. "
        "URL validation is deferred to consuming services.",
    )

    # State information
    current_state: str | None = Field(
        default=None,
        description="Current FSM state if applicable",
    )

    # Version information
    version: str = Field(
        default="1.0.0",
        description="Node version string",
    )

    # Event metadata
    reason: str = Field(
        default="startup",
        description="Reason for introspection event (startup, shutdown, request, heartbeat)",
    )

    # Tracing
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing (required for idempotency)",
    )

    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of introspection generation (must be explicitly provided)",
    )

    # Optional performance metrics from introspection operation
    performance_metrics: ModelIntrospectionPerformanceMetrics | None = Field(
        default=None,
        description="Optional performance metrics captured during introspection. "
        "Includes timing data, cache hit/miss info, and threshold violation tracking. "
        "None when metrics were not captured or are unavailable.",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_timezone_aware(cls, v: datetime) -> datetime:
        """Validate that timestamp is timezone-aware.

        Args:
            v: The timestamp value to validate.

        Returns:
            The validated timestamp.

        Raises:
            ValueError: If timestamp is naive (no timezone info).
        """
        if v.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware. Use datetime.now(UTC) or "
                "datetime(..., tzinfo=timezone.utc) instead of naive datetime."
            )
        return v

    # Design Decision: This model is immutable (frozen=True) because:
    # 1. Introspection events are snapshots of node state at a point in time
    # 2. Any "updates" should create new events via model_copy(update={...})
    # 3. Immutability prevents accidental state corruption in event handlers
    # 4. Frozen models are hashable, enabling use in sets/dict keys if needed
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "node_id": "550e8400-e29b-41d4-a716-446655440001",
                    "node_type": "EFFECT",
                    "capabilities": {
                        "operations": ["execute", "query", "batch_execute"],
                        "protocols": ["ProtocolDatabaseAdapter"],
                        "has_fsm": True,
                    },
                    "endpoints": {
                        "health": "http://localhost:8080/health",
                        "metrics": "http://localhost:8080/metrics",
                    },
                    "current_state": "connected",
                    "version": "1.0.0",
                    "reason": "startup",
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2025-01-15T10:30:00Z",
                    "performance_metrics": None,
                },
                {
                    "node_id": "550e8400-e29b-41d4-a716-446655440002",
                    "node_type": "COMPUTE",
                    "capabilities": {
                        "operations": ["process", "transform"],
                        "protocols": ["ProtocolComputeNode"],
                        "has_fsm": False,
                        "method_signatures": {
                            "process": "(data: bytes) -> bytes",
                        },
                    },
                    "endpoints": {
                        "health": "http://localhost:8081/health",
                    },
                    "current_state": None,
                    "version": "2.1.0",
                    "reason": "request",
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440003",
                    "timestamp": "2025-01-15T10:31:00Z",
                    "performance_metrics": {
                        "get_capabilities_ms": 12.5,
                        "discover_capabilities_ms": 8.2,
                        "get_endpoints_ms": 0.5,
                        "get_current_state_ms": 0.1,
                        "total_introspection_ms": 21.3,
                        "cache_hit": False,
                        "method_count": 15,
                        "threshold_exceeded": False,
                        "slow_operations": [],
                        "captured_at": "2025-01-15T10:31:00Z",
                    },
                },
            ]
        },
    )


__all__ = [
    "CapabilitiesTypedDict",
    "ModelIntrospectionPerformanceMetrics",
    "ModelNodeIntrospectionEvent",
]
