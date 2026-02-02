# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Projection Model.

Provides the Pydantic model for contract projections stored in PostgreSQL.
Used by the Registry API to query registered contracts and their metadata.

The contracts table stores registered ONEX contracts with full YAML content
for replay capability and Kafka position tracking for exactly-once semantics.

Related Tickets:
    - OMN-1845: Create ProjectionReaderContract for contract/topic queries
    - OMN-1653: Contract registry state materialization
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelContractProjection(BaseModel):
    """Contract projection for Registry API queries.

    Represents a registered ONEX contract stored in PostgreSQL. This model
    maps to the contracts table created by migration 005.

    Primary Key:
        contract_id - derived natural key: node_name:major.minor.patch

    Attributes:
        contract_id: Derived natural key (e.g., "my-node:1.0.0")
        node_name: ONEX node name from contract metadata
        version_major: Semantic version major component
        version_minor: Semantic version minor component
        version_patch: Semantic version patch component
        contract_hash: SHA-256 hash of contract YAML for change detection
        contract_yaml: Full contract YAML content for replay capability
        registered_at: Timestamp when contract was first registered
        deregistered_at: Timestamp when contract was deregistered (None if active)
        last_seen_at: Timestamp of most recent heartbeat or registration event
        is_active: Whether contract is currently active (soft delete)
        last_event_topic: Kafka topic of last processed event (for dedupe)
        last_event_partition: Kafka partition of last processed event (for dedupe)
        last_event_offset: Kafka offset of last processed event (for dedupe)
        created_at: Timestamp when row was created
        updated_at: Timestamp when row was last updated

    Example:
        >>> from datetime import datetime, UTC
        >>> now = datetime.now(UTC)
        >>> projection = ModelContractProjection(
        ...     contract_id="node-registry-effect:1.0.0",
        ...     node_name="node-registry-effect",
        ...     version_major=1,
        ...     version_minor=0,
        ...     version_patch=0,
        ...     contract_hash="abc123...",
        ...     contract_yaml="name: node-registry-effect\\n...",
        ...     registered_at=now,
        ...     last_seen_at=now,
        ...     is_active=True,
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Identity
    # ONEX_EXCLUDE: pattern_validator - contract_id is a derived natural key (name:version), not UUID
    contract_id: str = Field(
        ...,
        min_length=1,
        description="Derived natural key: node_name:major.minor.patch",
    )
    # ONEX_EXCLUDE: pattern_validator - node_name is the contract name, not an entity reference
    node_name: str = Field(
        ...,
        min_length=1,
        description="ONEX node name from contract metadata",
    )
    version_major: int = Field(
        ...,
        ge=0,
        description="Semantic version major component",
    )
    version_minor: int = Field(
        ...,
        ge=0,
        description="Semantic version minor component",
    )
    version_patch: int = Field(
        ...,
        ge=0,
        description="Semantic version patch component",
    )

    # Contract content
    contract_hash: str = Field(
        ...,
        min_length=1,
        description="SHA-256 hash of contract YAML for change detection",
    )
    contract_yaml: str = Field(
        ...,
        description="Full contract YAML content for replay capability",
    )

    # Lifecycle
    registered_at: datetime = Field(
        ...,
        description="Timestamp when contract was first registered",
    )
    deregistered_at: datetime | None = Field(
        default=None,
        description="Timestamp when contract was deregistered (None if active)",
    )
    last_seen_at: datetime = Field(
        ...,
        description="Timestamp of most recent heartbeat or registration event",
    )
    is_active: bool = Field(
        default=True,
        description="Whether contract is currently active (soft delete)",
    )

    # Kafka position tracking (for exactly-once semantics)
    last_event_topic: str | None = Field(
        default=None,
        description="Kafka topic of last processed event (for dedupe)",
    )
    last_event_partition: int | None = Field(
        default=None,
        description="Kafka partition of last processed event (for dedupe)",
    )
    last_event_offset: int | None = Field(
        default=None,
        description="Kafka offset of last processed event (for dedupe)",
    )

    # Audit timestamps
    created_at: datetime | None = Field(
        default=None,
        description="Timestamp when row was created",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Timestamp when row was last updated",
    )

    @property
    def version_string(self) -> str:
        """Return semantic version as string.

        Returns:
            Semantic version string (e.g., "1.0.0")

        Example:
            >>> proj = ModelContractProjection(version_major=1, version_minor=2, version_patch=3, ...)
            >>> proj.version_string
            '1.2.3'
        """
        return f"{self.version_major}.{self.version_minor}.{self.version_patch}"


__all__: list[str] = ["ModelContractProjection"]
