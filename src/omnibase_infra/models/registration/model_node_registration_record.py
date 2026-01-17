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

    Strong Typing:
    This model uses strongly-typed models for capabilities and metadata:
    - capabilities: ModelNodeCapabilities - typed capability flags
    - metadata: ModelNodeMetadata - typed deployment metadata

    These models replace the legacy dict[str, Any] pattern to comply with
    the ONEX "no Any types" rule. Both models use extra="allow" to support
    custom fields while maintaining type safety for known fields.

Thread Safety:
    ModelNodeRegistrationRecord is immutable after creation, making it
    thread-safe for concurrent read access.

Related:
    - ModelNodeRegistration: Full mutable registration model
    - ModelPostgresUpsertRegistrationIntent: Intent that uses this record
    - ModelNodeCapabilities: Strongly-typed capabilities model
    - ModelNodeMetadata: Strongly-typed metadata model
    - OMN-889: Infrastructure MVP
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.intents import ModelRegistrationRecordBase
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata


class ModelNodeRegistrationRecord(ModelRegistrationRecordBase):
    """Frozen record for node registration in PostgreSQL.

    This is a minimal, immutable record designed for use with
    ModelPostgresUpsertRegistrationIntent. It captures the essential
    fields needed for node registration without complex validation.

    The record is serialized by the Effect layer when persisting to PostgreSQL.
    The to_persistence_dict() method (inherited from ModelRegistrationRecordBase)
    handles JSON-compatible serialization.

    Strong Typing:
        Uses ModelNodeCapabilities and ModelNodeMetadata instead of dict[str, Any]
        to comply with ONEX "no Any types" rule. Both models use extra="allow"
        to support custom fields while providing type safety for known fields.

    Attributes:
        node_id: Unique node identifier (UUID).
        node_type: ONEX node type (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node.
        capabilities: Strongly-typed node capabilities (ModelNodeCapabilities).
        endpoints: Exposed endpoints as name -> URL mapping.
        metadata: Strongly-typed node metadata (ModelNodeMetadata).
        health_endpoint: Optional URL for health check endpoint.
        registered_at: Timestamp when node was first registered.
        updated_at: Timestamp of last update.

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> from omnibase_core.enums.enum_node_kind import EnumNodeKind
        >>> from omnibase_core.models.primitives.model_semver import ModelSemVer
        >>> from omnibase_infra.models.registration import (
        ...     ModelNodeCapabilities,
        ...     ModelNodeMetadata,
        ... )
        >>> now = datetime.now(UTC)
        >>> record = ModelNodeRegistrationRecord(
        ...     node_id=uuid4(),
        ...     node_type=EnumNodeKind.EFFECT,
        ...     node_version=ModelSemVer(major=1, minor=0, patch=0),
        ...     capabilities=ModelNodeCapabilities(postgres=True),
        ...     endpoints={"health": "http://localhost:8080/health"},
        ...     metadata=ModelNodeMetadata(environment="production"),
        ...     health_endpoint="http://localhost:8080/health",
        ...     registered_at=now,
        ...     updated_at=now,
        ... )
        >>> db_data = record.to_persistence_dict()
        >>> assert isinstance(db_data["node_id"], str)  # UUID serialized
    """

    # Identity
    node_id: UUID = Field(..., description="Unique node identifier")
    node_type: EnumNodeKind = Field(..., description="ONEX node type")
    node_version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer(major=1, minor=0, patch=0),
        description="Semantic version of the node",
    )

    @field_validator("node_version", mode="before")
    @classmethod
    def parse_node_version(cls, v: ModelSemVer | str) -> ModelSemVer:
        """Parse node_version from string or ModelSemVer.

        Args:
            v: Either a ModelSemVer instance or a semver string.

        Returns:
            Validated ModelSemVer instance.

        Raises:
            ValueError: If the string is not a valid semantic version.
        """
        if isinstance(v, str):
            try:
                return ModelSemVer.parse(v)
            except Exception as e:
                raise ValueError(f"node_version: {e!s}") from e
        return v

    # Capabilities and endpoints - strongly typed models
    capabilities: ModelNodeCapabilities = Field(
        default_factory=ModelNodeCapabilities,
        description="Strongly-typed node capabilities",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict, description="Exposed endpoints (name -> URL)"
    )
    metadata: ModelNodeMetadata = Field(
        default_factory=ModelNodeMetadata,
        description="Strongly-typed node metadata",
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
