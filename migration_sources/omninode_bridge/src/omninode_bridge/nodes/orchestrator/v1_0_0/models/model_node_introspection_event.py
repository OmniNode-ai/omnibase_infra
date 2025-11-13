#!/usr/bin/env python3
"""
Node Introspection Event Model for NodeBridgeOrchestrator.

Defines the structure for node introspection data broadcasting,
used for node discovery and capability advertisement.

ONEX v2.0 Compliance:
- Model-based naming: ModelNodeIntrospectionEvent
- Strong typing with Pydantic v2
- Integration with OnexEnvelopeV1 event publishing
"""

from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class ModelNodeIntrospectionEvent(BaseModel):
    """
    Node introspection event data for broadcasting node capabilities and metadata.

    This model represents the payload for NODE_INTROSPECTION events, which are
    broadcast by nodes to advertise their capabilities, endpoints, and metadata
    to the registry and other nodes.

    Attributes:
        node_id: Unique identifier for the node instance
        node_type: Type of node (effect, compute, reducer, orchestrator)
        capabilities: Dictionary of capabilities and their configurations
        endpoints: Dictionary of available endpoints (health, api, metrics)
        metadata: Additional node metadata (version, environment, etc.)
        timestamp: When the introspection data was generated
        correlation_id: Optional correlation ID for tracking related events

    Usage:
        Used as payload in OnexEnvelopeV1 for NODE_INTROSPECTION events.
        Typically broadcast on node startup and periodically for discovery.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(
        ...,
        description="Unique identifier for the node instance (e.g., UUID or hostname-based)",
    )

    node_type: str = Field(
        ...,
        description="Type of ONEX node: effect, compute, reducer, or orchestrator",
        pattern="^(effect|compute|reducer|orchestrator)$",
    )

    node_role: Optional[str] = Field(
        default=None,
        description="Optional role specialization within the node type (e.g., 'registry', 'adapter', 'bridge')",
        examples=["registry", "adapter", "bridge", "gateway", "proxy"],
    )

    capabilities: dict[str, Any] = Field(
        default_factory=dict,
        description="Node capabilities with configuration details",
        examples=[
            {
                "hash_generation": {"algorithm": "BLAKE3", "max_file_size_mb": 100},
                "metadata_stamping": {
                    "formats": ["markdown", "html"],
                    "batch_size": 50,
                },
                "workflow_orchestration": {"max_concurrent": 100, "timeout_ms": 5000},
            }
        ],
    )

    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Available HTTP/gRPC endpoints",
        examples=[
            {
                "health": "http://localhost:8053/health",
                "api": "http://localhost:8053/api/v1",
                "metrics": "http://localhost:9090/metrics",
                "grpc": "grpc://localhost:50051",
            }
        ],
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional node metadata",
        examples=[
            {
                "version": "1.0.0",
                "environment": "production",
                "region": "us-west-2",
                "deployed_at": "2025-10-03T10:00:00Z",
                "resource_limits": {"cpu_cores": 4, "memory_gb": 16},
            }
        ],
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when introspection data was generated",
    )

    correlation_id: Optional[UUID] = Field(
        default=None,
        description="Optional correlation ID for tracking related events",
    )

    # Network topology metadata (Phase 1a MVP - Multi-network support)
    network_id: Optional[str] = Field(
        default=None,
        description="Logical network identifier for multi-network topologies",
        examples=["omninode-network-1", "prod-cluster-us-west"],
    )

    deployment_id: Optional[str] = Field(
        default=None,
        description="Deployment instance identifier for tracking deployments",
        examples=["prod-us-west-2-001", "staging-001", "dev-001"],
    )

    epoch: Optional[int] = Field(
        default=None,
        description="Deployment epoch for blue-green and rolling deployments (integer version)",
        examples=[1, 2, 3],
        ge=1,
    )

    @field_serializer("timestamp")
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
        node_role: Optional[str] = None,
        capabilities: Optional[dict[str, Any]] = None,
        endpoints: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        correlation_id: Optional[UUID] = None,
        network_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> "ModelNodeIntrospectionEvent":
        """
        Factory method to create a node introspection event with defaults.

        Args:
            node_id: Unique node identifier
            node_type: Type of node (effect, compute, reducer, orchestrator)
            node_role: Optional role specialization (registry, adapter, bridge, etc.)
            capabilities: Optional capabilities dictionary
            endpoints: Optional endpoints dictionary
            metadata: Optional metadata dictionary
            correlation_id: Optional correlation ID
            network_id: Optional network identifier for multi-network support
            deployment_id: Optional deployment instance identifier
            epoch: Optional deployment epoch for blue-green deployments

        Returns:
            ModelNodeIntrospectionEvent instance
        """
        return cls(
            node_id=node_id,
            node_type=node_type,
            node_role=node_role,
            capabilities=capabilities or {},
            endpoints=endpoints or {},
            metadata=metadata or {},
            correlation_id=correlation_id,
            network_id=network_id,
            deployment_id=deployment_id,
            epoch=epoch,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for use in OnexEnvelopeV1 payload.

        Returns:
            Dictionary representation suitable for event payload
        """
        return self.model_dump(mode="json")
