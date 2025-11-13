#!/usr/bin/env python3
"""
Node Heartbeat Event Model for Health Monitoring.

Defines the structure for periodic node heartbeat events,
used for health monitoring and availability tracking.

ONEX v2.0 Compliance:
- Model-based naming: ModelNodeHeartbeatEvent
- Strong typing with Pydantic v2
- Integration with OnexEnvelopeV1 event publishing
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class EnumNodeHealthStatus(str, Enum):
    """Node health status values."""

    HEALTHY = "healthy"
    """Node is fully operational."""

    DEGRADED = "degraded"
    """Node is operational but with reduced performance."""

    UNHEALTHY = "unhealthy"
    """Node is experiencing issues but still running."""

    CRITICAL = "critical"
    """Node is in critical state, may fail soon."""

    RECOVERING = "recovering"
    """Node is recovering from a failure."""


class ModelNodeHeartbeatEvent(BaseModel):
    """
    Node heartbeat event for periodic health and availability reporting.

    This model represents the payload for NODE_HEARTBEAT events, which are
    broadcast periodically by nodes to indicate their health status and
    availability to the registry and monitoring systems.

    Attributes:
        node_id: Unique identifier for the node instance
        node_type: Type of node (effect, compute, reducer, orchestrator)
        health_status: Current health status of the node
        uptime_seconds: Node uptime in seconds
        last_activity_timestamp: Timestamp of last activity
        active_operations: Number of currently active operations
        resource_usage: Current resource utilization metrics
        metadata: Additional heartbeat metadata
        timestamp: When the heartbeat was generated
        correlation_id: Optional correlation ID for tracking

    Usage:
        Used as payload in OnexEnvelopeV1 for NODE_HEARTBEAT events.
        Typically broadcast on a fixed interval (e.g., every 30 seconds).
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(
        ...,
        description="Unique identifier for the node instance",
    )

    node_type: str = Field(
        ...,
        description="Type of ONEX node: effect, compute, reducer, or orchestrator",
        pattern="^(effect|compute|reducer|orchestrator)$",
    )

    health_status: EnumNodeHealthStatus = Field(
        ...,
        description="Current health status of the node",
    )

    uptime_seconds: int = Field(
        ...,
        description="Node uptime in seconds since last restart",
        ge=0,
    )

    last_activity_timestamp: datetime = Field(
        ...,
        description="Timestamp of the last meaningful activity (request processed, etc.)",
    )

    active_operations: int = Field(
        default=0,
        description="Number of currently active operations/requests",
        ge=0,
    )

    resource_usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Current resource utilization metrics",
        examples=[
            {
                "cpu_percent": 45.2,
                "memory_mb": 512,
                "memory_percent": 32.0,
                "disk_usage_percent": 68.5,
                "network_connections": 15,
                "thread_count": 20,
            }
        ],
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional heartbeat metadata",
        examples=[
            {
                "version": "1.0.0",
                "environment": "production",
                "region": "us-west-2",
                "error_count_last_minute": 0,
                "request_count_last_minute": 150,
            }
        ],
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when heartbeat was generated",
    )

    correlation_id: Optional[UUID] = Field(
        default=None,
        description="Optional correlation ID for tracking related events",
    )

    @field_serializer("last_activity_timestamp", "timestamp")
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
        node_id: str,
        node_type: str,
        health_status: EnumNodeHealthStatus,
        uptime_seconds: int,
        last_activity_timestamp: datetime,
        active_operations: int = 0,
        resource_usage: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        correlation_id: Optional[UUID] = None,
    ) -> "ModelNodeHeartbeatEvent":
        """
        Factory method to create a node heartbeat event with defaults.

        Args:
            node_id: Unique node identifier
            node_type: Type of node
            health_status: Current health status
            uptime_seconds: Node uptime in seconds
            last_activity_timestamp: Timestamp of last activity
            active_operations: Number of active operations
            resource_usage: Optional resource usage metrics
            metadata: Optional metadata dictionary
            correlation_id: Optional correlation ID

        Returns:
            ModelNodeHeartbeatEvent instance
        """
        return cls(
            node_id=node_id,
            node_type=node_type,
            health_status=health_status,
            uptime_seconds=uptime_seconds,
            last_activity_timestamp=last_activity_timestamp,
            active_operations=active_operations,
            resource_usage=resource_usage or {},
            metadata=metadata or {},
            correlation_id=correlation_id,
        )

    def is_healthy(self) -> bool:
        """
        Check if node is in a healthy state.

        Returns:
            True if node is healthy or degraded, False otherwise
        """
        return self.health_status in (
            EnumNodeHealthStatus.HEALTHY,
            EnumNodeHealthStatus.DEGRADED,
        )

    def needs_attention(self) -> bool:
        """
        Check if node needs administrative attention.

        Returns:
            True if node is unhealthy or critical, False otherwise
        """
        return self.health_status in (
            EnumNodeHealthStatus.UNHEALTHY,
            EnumNodeHealthStatus.CRITICAL,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for use in OnexEnvelopeV1 payload.

        Returns:
            Dictionary representation suitable for event payload
        """
        return self.model_dump(mode="json")
