# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Snapshot Model.

Provides the Pydantic model for registration snapshots used for read optimization.
Snapshots are compacted representations of registration state for efficient querying.

IMPORTANT: Snapshots are a read optimization layer and do NOT replace the immutable
event log. The source of truth remains the event log and projections. Snapshots
provide a compacted view for faster reads at the cost of historical granularity.

Thread Safety:
    This model is frozen (immutable) as snapshots are point-in-time captures.
    Create new snapshots rather than mutating existing ones.

Related Tickets:
    - OMN-947 (F2): Snapshot Publishing
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.enums import EnumInfraTransportType, EnumRegistrationState
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)

if TYPE_CHECKING:
    from omnibase_infra.models.projection.model_registration_projection import (
        ModelRegistrationProjection,
    )


class ModelRegistrationSnapshot(BaseModel):
    """Compacted registration snapshot for read optimization.

    Snapshots provide a compacted view of registration state optimized for
    read-heavy workloads. They capture the essential state needed for:
    - Service discovery (which nodes are ACTIVE)
    - Capability queries (what can each node do)
    - Status dashboards (current registration states)

    Design Notes (per F2 requirements):
        - Compacted representation: Only essential fields, no timeout tracking
        - Kafka topic compaction: Uses domain:entity_id as key for log compaction
        - Version-based ordering: snapshot_version for conflict resolution
        - Traceability: source_projection_sequence links back to source projection

    Key Differences from Projection:
        - Frozen (immutable): Snapshots are point-in-time captures
        - No timeout fields: Snapshots don't track ack/liveness deadlines
        - No event tracking: No last_applied_event_id/offset (use source_projection_sequence)
        - Compaction key: to_kafka_key() for topic compaction

    Primary Key:
        (entity_id, domain) - same composite key as projection

    Compaction Strategy:
        Kafka topic compaction with key = "{domain}:{entity_id}"
        Only the latest snapshot per entity is retained after compaction.

    Attributes:
        entity_id: Node identifier (partition key for per-entity ordering)
        domain: Domain namespace for multi-domain support (default: "registration")
        current_state: Current FSM state for workflow decisions
        node_type: ONEX node type (effect, compute, reducer, orchestrator)
        node_name: Human-readable node name (cached from introspection)
        capabilities: Node capabilities snapshot at registration time
        last_state_change_at: Timestamp when current_state last changed
        snapshot_version: Monotonically increasing version for compaction ordering
        snapshot_created_at: When this snapshot was created
        source_projection_sequence: Source projection sequence for traceability

    Example:
        >>> from datetime import datetime, timezone
        >>> from uuid import uuid4
        >>> from omnibase_core.enums import EnumNodeKind
        >>> now = datetime.now(timezone.utc)
        >>> entity_id = uuid4()
        >>> snapshot = ModelRegistrationSnapshot(
        ...     entity_id=entity_id,
        ...     current_state=EnumRegistrationState.ACTIVE,
        ...     node_type=EnumNodeKind.EFFECT,
        ...     node_name="PostgresAdapter",
        ...     last_state_change_at=now,
        ...     snapshot_version=1,
        ...     snapshot_created_at=now,
        ... )
        >>> snapshot.to_kafka_key()  # Returns 'registration:<uuid>'
        'registration:...'

    Note:
        When serialized to JSON via ``model_dump(mode="json")``, the ``node_type``
        field is serialized as its string value (e.g., ``"effect"``). When
        deserializing, both ``EnumNodeKind.EFFECT`` and ``"effect"`` are accepted.
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable - snapshots are point-in-time captures
        extra="forbid",
        from_attributes=True,
    )

    # Identity (composite key: entity_id + domain)
    entity_id: UUID = Field(
        ...,
        description="Node UUID (partition key for per-entity ordering)",
    )
    domain: str = Field(
        default="registration",
        min_length=1,
        max_length=128,
        description="Domain namespace for multi-domain support",
    )

    # FSM State
    current_state: EnumRegistrationState = Field(
        ...,
        description="Current FSM state for workflow decisions",
    )

    # Node Information (cached from introspection/registration)
    node_type: EnumNodeKind | None = Field(
        default=None,
        description="ONEX node type (cached from introspection)",
    )
    node_name: str | None = Field(
        default=None,
        max_length=256,
        description="Human-readable node name (cached from introspection)",
    )
    capabilities: ModelNodeCapabilities | None = Field(
        default=None,
        description="Node capabilities snapshot at registration",
    )

    # State Change Tracking
    last_state_change_at: datetime = Field(
        ...,
        description="Timestamp when current_state last changed",
    )

    # Snapshot Versioning (for compaction and conflict resolution)
    snapshot_version: int = Field(
        ...,
        ge=1,
        description="Monotonically increasing version for compaction ordering",
    )
    snapshot_created_at: datetime = Field(
        ...,
        description="When this snapshot was created",
    )

    # Traceability
    source_projection_sequence: int | None = Field(
        default=None,
        ge=0,
        description="Source projection sequence for traceability to projection",
    )

    @classmethod
    def from_projection(
        cls,
        projection: ModelRegistrationProjection,
        *,
        snapshot_version: int,
        snapshot_created_at: datetime,
        node_name: str | None = None,
    ) -> ModelRegistrationSnapshot:
        """Create a snapshot from a projection.

        Factory method to create a compacted snapshot from a full projection.
        Extracts essential fields and discards timeout tracking data.

        Args:
            projection: Source projection to create snapshot from
            snapshot_version: Version number for this snapshot (must be monotonically increasing)
            snapshot_created_at: Timestamp when snapshot is being created
            node_name: Optional node name (not stored in projection, must be provided externally)

        Returns:
            A new ModelRegistrationSnapshot with compacted state

        Example:
            >>> from datetime import datetime, UTC
            >>> now = datetime.now(UTC)
            >>> projection = ModelRegistrationProjection(...)
            >>> snapshot = ModelRegistrationSnapshot.from_projection(
            ...     projection,
            ...     snapshot_version=1,
            ...     snapshot_created_at=now,
            ...     node_name="PostgresAdapter",
            ... )
        """
        # Determine source sequence for traceability
        # Prefer last_applied_sequence if set, otherwise use last_applied_offset
        source_sequence = (
            projection.last_applied_sequence
            if projection.last_applied_sequence is not None
            else projection.last_applied_offset
        )

        return cls(
            entity_id=projection.entity_id,
            domain=projection.domain,
            current_state=projection.current_state,
            node_type=projection.node_type,
            node_name=node_name,
            capabilities=projection.capabilities,
            last_state_change_at=projection.updated_at,
            snapshot_version=snapshot_version,
            snapshot_created_at=snapshot_created_at,
            source_projection_sequence=source_sequence,
        )

    def to_kafka_key(self) -> str:
        """Generate Kafka compaction key for this snapshot.

        Returns a key suitable for Kafka topic compaction. The key format
        is "{domain}:{entity_id}" which ensures:
        - Per-entity compaction (only latest snapshot retained per entity)
        - Multi-domain support (entities in different domains are distinct)

        Returns:
            Compaction key in format "domain:entity_id"

        Example:
            >>> from uuid import UUID
            >>> snapshot = ModelRegistrationSnapshot(
            ...     entity_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            ...     domain="registration",
            ...     ...
            ... )
            >>> snapshot.to_kafka_key()
            'registration:550e8400-e29b-41d4-a716-446655440000'
        """
        return f"{self.domain}:{self.entity_id!s}"

    def is_newer_than(self, other: ModelRegistrationSnapshot) -> bool:
        """Check if this snapshot is newer than another.

        Compares snapshots by version number for conflict resolution
        during compaction or merge operations.

        Args:
            other: Another snapshot to compare against

        Returns:
            True if this snapshot has a higher version, False otherwise

        Raises:
            ProtocolConfigurationError: If snapshots are for different entities (entity_id + domain mismatch)

        Example:
            >>> snap_v1 = ModelRegistrationSnapshot(snapshot_version=1, ...)
            >>> snap_v2 = ModelRegistrationSnapshot(snapshot_version=2, ...)
            >>> snap_v2.is_newer_than(snap_v1)
            True
            >>> snap_v1.is_newer_than(snap_v2)
            False
        """
        if self.entity_id != other.entity_id or self.domain != other.domain:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="is_newer_than",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                f"Cannot compare snapshots for different entities: "
                f"{self.to_kafka_key()} vs {other.to_kafka_key()}",
                context=context,
            )
        return self.snapshot_version > other.snapshot_version

    def is_active(self) -> bool:
        """Check if the snapshotted entity is in ACTIVE state.

        Convenience method for service discovery queries.

        Returns:
            True if current_state is ACTIVE, False otherwise

        Example:
            >>> snapshot = ModelRegistrationSnapshot(
            ...     current_state=EnumRegistrationState.ACTIVE,
            ...     ...
            ... )
            >>> snapshot.is_active()
            True
        """
        return self.current_state.is_active()

    def is_terminal(self) -> bool:
        """Check if the snapshotted entity is in a terminal state.

        Terminal states (REJECTED, LIVENESS_EXPIRED) indicate the registration
        has ended and requires re-registration to continue.

        Returns:
            True if current_state is terminal, False otherwise

        Example:
            >>> snapshot = ModelRegistrationSnapshot(
            ...     current_state=EnumRegistrationState.LIVENESS_EXPIRED,
            ...     ...
            ... )
            >>> snapshot.is_terminal()
            True
        """
        return self.current_state.is_terminal()


__all__: list[str] = ["ModelRegistrationSnapshot"]
