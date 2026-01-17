# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Generic Snapshot Model for Point-in-Time State Capture.

Provides a generic, immutable snapshot model for capturing point-in-time state
of any entity. Snapshots support versioning, content hashing for idempotency,
and lineage tracking via parent references.

Thread Safety:
    This model is frozen (immutable) for safe sharing across threads.
    Use with_mutations() to create new snapshots from existing ones.

Related Tickets:
    - OMN-1246: ServiceSnapshot Infrastructure Primitive
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType
from omnibase_infra.models.snapshot.model_subject_ref import ModelSubjectRef


class ModelSnapshot(BaseModel):
    """Generic snapshot for point-in-time state capture.

    Snapshots are immutable captures of entity state at a specific point in time.
    They support:
    - Subject-based grouping via ModelSubjectRef
    - Sequence-based ordering within a subject
    - Content hashing for idempotency checks
    - Lineage tracking via parent_id for fork scenarios

    Design Notes:
        - Frozen (immutable): Create new snapshots via with_mutations()
        - Sequence numbers: For ordering within a subject, not globally unique
        - Content hash: Deterministic hash of data for idempotency
        - Version: For optimistic locking in storage backends

    Attributes:
        id: Unique identifier for this snapshot.
        subject: Reference to the subject this snapshot belongs to.
        data: The actual snapshot payload as JSON-compatible dict.
        sequence_number: Ordering number within the subject.
        version: Version for optimistic locking (default: 1).
        content_hash: SHA-256 hash of data for idempotency checks.
        created_at: Timestamp when snapshot was created.
        parent_id: Optional parent snapshot ID for lineage tracking.

    Example:
        >>> from uuid import uuid4
        >>> subject = ModelSubjectRef(subject_type="agent", subject_id=uuid4())
        >>> snapshot = ModelSnapshot(
        ...     subject=subject,
        ...     data={"status": "active", "config": {"timeout": 30}},
        ...     sequence_number=1,
        ... )
        >>> snapshot.content_hash is not None
        True

        >>> # Create a fork with mutations
        >>> new_snapshot = snapshot.with_mutations(
        ...     mutations={"status": "paused"},
        ...     sequence_number=2,
        ... )
        >>> new_snapshot.parent_id == snapshot.id
        True
        >>> new_snapshot.data["status"]
        'paused'
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Identity
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this snapshot",
    )
    subject: ModelSubjectRef = Field(
        ...,
        description="Reference to the subject this snapshot belongs to",
    )

    # Data
    data: dict[str, JsonType] = Field(
        default_factory=dict,
        description="The actual snapshot payload as JSON-compatible dict",
    )

    # Versioning & ordering
    sequence_number: int = Field(
        ...,
        ge=0,
        description="Ordering number within the subject",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Version for optimistic locking",
    )
    content_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of data for idempotency checks",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when snapshot was created",
    )

    # Lineage
    parent_id: UUID | None = Field(
        default=None,
        description="Parent snapshot ID for lineage/fork tracking",
    )

    def model_post_init(self, _context: object) -> None:
        """Compute content hash after initialization if not provided.

        Pydantic's frozen models don't allow attribute assignment after init,
        so we use object.__setattr__ to set the computed hash.
        """
        if self.content_hash is None:
            computed_hash = self.compute_content_hash(self.data)
            object.__setattr__(self, "content_hash", computed_hash)

    @classmethod
    def compute_content_hash(cls, data: dict[str, JsonType]) -> str:
        """Compute deterministic SHA-256 hash of snapshot data.

        Uses JSON serialization with sorted keys to ensure deterministic
        output regardless of dict key ordering.

        Args:
            data: The data dictionary to hash.

        Returns:
            Hexadecimal SHA-256 hash string.

        Example:
            >>> hash1 = ModelSnapshot.compute_content_hash({"b": 2, "a": 1})
            >>> hash2 = ModelSnapshot.compute_content_hash({"a": 1, "b": 2})
            >>> hash1 == hash2  # Same content, same hash
            True
        """
        # Use sort_keys for deterministic ordering
        json_bytes = json.dumps(data, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(json_bytes).hexdigest()

    def with_mutations(
        self,
        mutations: dict[str, JsonType],
        sequence_number: int,
        *,
        new_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> ModelSnapshot:
        """Create new snapshot with applied mutations (for fork).

        Creates a new snapshot by merging mutations into the current data.
        The new snapshot tracks lineage via parent_id pointing to this snapshot.

        Args:
            mutations: Dictionary of changes to apply to data.
            sequence_number: Sequence number for the new snapshot.
            new_id: Optional UUID for new snapshot (auto-generated if None).
            created_at: Optional creation timestamp (now if None).

        Returns:
            A new ModelSnapshot with mutations applied and parent_id set.

        Example:
            >>> from uuid import uuid4
            >>> subject = ModelSubjectRef(subject_type="config", subject_id=uuid4())
            >>> original = ModelSnapshot(
            ...     subject=subject,
            ...     data={"timeout": 30, "retries": 3},
            ...     sequence_number=1,
            ... )
            >>> updated = original.with_mutations(
            ...     mutations={"timeout": 60},
            ...     sequence_number=2,
            ... )
            >>> updated.data["timeout"]
            60
            >>> updated.data["retries"]  # Unchanged
            3
            >>> updated.parent_id == original.id
            True
        """
        # Merge mutations into existing data
        new_data = {**self.data, **mutations}

        return ModelSnapshot(
            id=new_id if new_id is not None else uuid4(),
            subject=self.subject,
            data=new_data,
            sequence_number=sequence_number,
            version=1,  # Reset version for new snapshot
            content_hash=None,  # Will be computed in model_post_init
            created_at=created_at if created_at is not None else datetime.now(UTC),
            parent_id=self.id,  # Track lineage
        )

    def to_kafka_key(self) -> str:
        """Generate Kafka compaction key for this snapshot.

        Uses the subject reference key for topic compaction, ensuring
        only the latest snapshot per subject is retained.

        Returns:
            Compaction key from subject reference.

        Example:
            >>> from uuid import UUID
            >>> subject = ModelSubjectRef(
            ...     subject_type="agent",
            ...     subject_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            ... )
            >>> snapshot = ModelSnapshot(subject=subject, data={}, sequence_number=1)
            >>> snapshot.to_kafka_key()
            'agent:550e8400-e29b-41d4-a716-446655440000'
        """
        return self.subject.to_key()

    def is_content_equal(self, other: ModelSnapshot) -> bool:
        """Check if two snapshots have identical content.

        Compares content hashes for efficient equality check without
        deep comparison of data structures.

        Args:
            other: Another snapshot to compare against.

        Returns:
            True if both snapshots have non-None content hashes that match,
            False otherwise (including when either hash is None).

        Note:
            Returns False when either snapshot has a None content_hash,
            since we cannot reliably compare content without hashes.

        Example:
            >>> from uuid import uuid4
            >>> subject = ModelSubjectRef(subject_type="test", subject_id=uuid4())
            >>> snap1 = ModelSnapshot(subject=subject, data={"a": 1}, sequence_number=1)
            >>> snap2 = ModelSnapshot(subject=subject, data={"a": 1}, sequence_number=2)
            >>> snap1.is_content_equal(snap2)
            True
        """
        # Cannot compare content if either hash is None
        if self.content_hash is None or other.content_hash is None:
            return False
        return self.content_hash == other.content_hash


__all__: list[str] = ["ModelSnapshot"]
