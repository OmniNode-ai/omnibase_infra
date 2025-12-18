# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Record for PostgreSQL Intent.

This module provides ModelNodeRegistrationRecord, a frozen record model
suitable for use with ModelPostgresUpsertRegistrationIntent.

Architecture:
    This model extends ModelRegistrationRecordBase from omnibase_core to ensure
    protocol compliance and consistent serialization behavior. It captures the
    minimal set of fields needed for node registration persistence.

    Unlike ModelNodeRegistration (which is mutable and has complex validation),
    this record is:
    - Immutable (frozen=True)
    - Minimal (only essential fields)
    - Compliant with ProtocolRegistrationRecord

Thread Safety:
    ModelNodeRegistrationRecord is immutable after creation, making it
    thread-safe for concurrent read access.

Related:
    - ModelNodeRegistration: Full mutable registration model
    - ModelPostgresUpsertRegistrationIntent: Intent that uses this record
    - OMN-889: Infrastructure MVP
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from omnibase_core.models.intents import ModelRegistrationRecordBase
from pydantic import Field


class ModelNodeRegistrationRecord(ModelRegistrationRecordBase):
    """Frozen record for node registration in PostgreSQL.

    This is a minimal, immutable record designed for use with
    ModelPostgresUpsertRegistrationIntent. It captures the essential
    fields needed for node registration without complex validation.

    The record is serialized by the Effect layer when persisting to PostgreSQL.
    The to_persistence_dict() method (inherited from ModelRegistrationRecordBase)
    handles JSON-compatible serialization.

    Attributes:
        node_id: Unique node identifier (UUID).
        node_type: ONEX node type (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node.
        capabilities: Node capabilities as JSON-serializable dict.
        endpoints: Exposed endpoints as name -> URL mapping.
        metadata: Additional node metadata as JSON-serializable dict.
        health_endpoint: Optional URL for health check endpoint.
        registered_at: Timestamp when node was first registered.
        updated_at: Timestamp of last update.

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> now = datetime.now(UTC)
        >>> record = ModelNodeRegistrationRecord(
        ...     node_id=uuid4(),
        ...     node_type="effect",
        ...     node_version="1.0.0",
        ...     capabilities={"postgres": True},
        ...     endpoints={"health": "http://localhost:8080/health"},
        ...     metadata={"environment": "production"},
        ...     health_endpoint="http://localhost:8080/health",
        ...     registered_at=now,
        ...     updated_at=now,
        ... )
        >>> db_data = record.to_persistence_dict()
        >>> assert isinstance(db_data["node_id"], str)  # UUID serialized
    """

    # Identity
    node_id: UUID = Field(..., description="Unique node identifier")
    node_type: Literal["effect", "compute", "reducer", "orchestrator"] = Field(
        ..., description="ONEX node type"
    )
    node_version: str = Field(
        default="1.0.0", description="Semantic version of the node"
    )

    # Capabilities and endpoints (JSON-serializable dicts)
    capabilities: dict[str, Any] = Field(
        default_factory=dict, description="Node capabilities"
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict, description="Exposed endpoints (name -> URL)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional node metadata"
    )

    # Health tracking
    health_endpoint: str | None = Field(
        default=None, description="URL for health check endpoint"
    )

    # Timestamps
    registered_at: datetime = Field(
        ..., description="Timestamp when node was first registered"
    )
    updated_at: datetime = Field(..., description="Timestamp of last update")


__all__ = ["ModelNodeRegistrationRecord"]
