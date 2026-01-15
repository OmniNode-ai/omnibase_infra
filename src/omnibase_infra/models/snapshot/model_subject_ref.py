# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Subject Reference Model for Snapshot Grouping.

Provides a reference model for identifying subjects (entities) that snapshots
belong to. Enables grouping and retrieval of snapshots by subject type and ID.

Thread Safety:
    This model is frozen (immutable) for safe sharing across threads.

Related Tickets:
    - OMN-1246: ServiceSnapshot Infrastructure Primitive
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelSubjectRef(BaseModel):
    """Reference to a subject for snapshot grouping.

    A subject reference identifies the entity that a snapshot belongs to,
    enabling efficient grouping and retrieval of related snapshots.

    Attributes:
        subject_type: Type identifier for the subject (e.g., "agent", "workflow", "config").
        subject_id: Unique identifier for the subject instance.

    Example:
        >>> from uuid import uuid4
        >>> ref = ModelSubjectRef(
        ...     subject_type="agent",
        ...     subject_id=uuid4(),
        ... )
        >>> ref.to_key()
        'agent:550e8400-e29b-41d4-a716-446655440000'
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    subject_type: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Type identifier for the subject (e.g., 'agent', 'workflow', 'config')",
    )
    subject_id: UUID = Field(
        ...,
        description="Unique identifier for the subject instance",
    )

    def to_key(self) -> str:
        """Generate storage key for this subject reference.

        Returns a string key suitable for use in storage systems, caches,
        or Kafka topic compaction.

        Returns:
            Key in format "{subject_type}:{subject_id}"

        Example:
            >>> from uuid import UUID
            >>> ref = ModelSubjectRef(
            ...     subject_type="agent",
            ...     subject_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            ... )
            >>> ref.to_key()
            'agent:550e8400-e29b-41d4-a716-446655440000'
        """
        return f"{self.subject_type}:{self.subject_id!s}"


__all__: list[str] = ["ModelSubjectRef"]
