# Copyright 2025 OmniNode Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node introspection event model for capability discovery and reporting."""

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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
        correlation_id: Optional correlation ID for tracing.
        timestamp: UTC timestamp when the introspection was generated.

    Example:
        ```python
        event = ModelNodeIntrospectionEvent(
            node_id="node-postgres-adapter-001",
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
        )
        ```
    """

    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="Node type classification")

    # Capabilities discovered via reflection
    capabilities: dict[str, list[str] | bool | dict[str, str]] = Field(
        default_factory=dict,
        description="Node capabilities discovered via reflection",
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
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for distributed tracing",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of introspection generation",
    )

    model_config = ConfigDict(
        frozen=False,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "node_id": "node-postgres-adapter-001",
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
                }
            ]
        },
    )


__all__ = ["ModelNodeIntrospectionEvent"]
