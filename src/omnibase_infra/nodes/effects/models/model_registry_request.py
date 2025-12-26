# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry Request Model for Registry Effect Operations.

This module provides ModelRegistryRequest, representing the input to the
NodeRegistryEffect node for dual-backend registration operations.

Architecture:
    ModelRegistryRequest captures all information needed to register a node
    in both Consul and PostgreSQL backends:
    - Node identity (node_id, node_type, node_version)
    - Service discovery info (endpoints, health checks)
    - Metadata for registration record

    This model is typically created from a ModelNodeIntrospectionEvent
    or directly constructed for programmatic registration.

Related:
    - NodeRegistryEffect: Effect node that consumes this request
    - ModelRegistryResponse: Response model for registry operations
    - ModelNodeIntrospectionEvent: Source of registration data
    - OMN-954: Partial failure scenario testing
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from pydantic import BaseModel, ConfigDict, Field


class ModelRegistryRequest(BaseModel):
    """Request model for dual-backend registration operations.

    Contains all information needed to register a node in both Consul
    and PostgreSQL backends. The NodeRegistryEffect uses this request to
    execute parallel or sequential registration operations.

    Immutability:
        This model uses frozen=True to ensure requests are immutable
        once created, enabling safe concurrent access.

    Attributes:
        node_id: Unique identifier for the node being registered.
        node_type: Type of ONEX node (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node.
        correlation_id: Correlation ID for distributed tracing.
        service_name: Name for service discovery registration.
        endpoints: Dict of endpoint type to URL (e.g., {"health": "http://..."}).
        tags: List of tags for service discovery.
        metadata: Additional metadata for the registration record.
        health_check_config: Optional health check configuration for Consul.
        timestamp: When this request was created.

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_core.enums.enum_node_kind import EnumNodeKind
        >>> request = ModelRegistryRequest(
        ...     node_id=uuid4(),
        ...     node_type=EnumNodeKind.EFFECT,
        ...     node_version="1.0.0",
        ...     correlation_id=uuid4(),
        ...     service_name="onex-effect",
        ...     endpoints={"health": "http://localhost:8080/health"},
        ... )
        >>> request.node_type
        <EnumNodeKind.EFFECT: 'effect'>
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: UUID = Field(
        ...,
        description="Unique identifier for the node being registered",
    )
    node_type: EnumNodeKind = Field(
        ...,
        description="Type of ONEX node (effect, compute, reducer, orchestrator)",
    )
    node_version: str = Field(
        ...,
        description="Semantic version of the node",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for distributed tracing",
    )
    service_name: str | None = Field(
        default=None,
        description="Name for service discovery registration",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Dict of endpoint type to URL",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of tags for service discovery",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata for the registration record",
    )
    health_check_config: dict[str, str] | None = Field(
        default=None,
        description="Optional health check configuration for Consul",
    )
    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    timestamp: datetime = Field(
        ...,
        description="When this request was created (must be explicitly provided)",
    )


__all__ = ["ModelRegistryRequest"]
