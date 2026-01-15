# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Sequence Information Model.

Provides sequence information for projection ordering and idempotency.
Supports both Kafka (partition, offset) and generic transports (sequence).

Thread Safety:
    This model is frozen (immutable) and safe for concurrent access.

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSequenceInfo(BaseModel):
    """Sequence information for projection ordering and idempotency.

    Supports both Kafka-based transports (using partition and offset) and
    generic transports (using sequence number). Per-entity monotonic ordering
    is enforced by the projector.

    Ordering Semantics:
        - Primary ordering is by `sequence` (transport-agnostic)
        - For Kafka transports, `offset` provides within-partition ordering
        - Projectors reject updates where incoming sequence <= current sequence

    Attributes:
        sequence: Monotonic sequence number (transport-agnostic).
            For Kafka, this is typically derived from offset.
            For generic transports, this is the application-provided sequence.
        partition: Kafka partition identifier (optional).
            Present only for Kafka transports. Used for debugging and
            partition-aware ordering when multiple partitions are involved.
        offset: Kafka offset within partition (optional).
            Present only for Kafka transports. Combined with partition,
            provides globally unique ordering within a topic.

    Example:
        >>> # From Kafka message
        >>> seq = ModelSequenceInfo(sequence=1000, partition="0", offset=1000)
        >>> # From generic transport
        >>> seq = ModelSequenceInfo(sequence=42)
        >>> # Staleness check
        >>> old_seq = ModelSequenceInfo(sequence=10)
        >>> new_seq = ModelSequenceInfo(sequence=20)
        >>> old_seq.is_stale_compared_to(new_seq)
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    sequence: int = Field(
        ...,
        ge=0,
        description="Monotonic sequence number (transport-agnostic)",
    )
    partition: str | None = Field(
        default=None,
        description="Kafka partition identifier (optional)",
    )
    offset: int | None = Field(
        default=None,
        ge=0,
        description="Kafka offset within partition (optional)",
    )

    def is_stale_compared_to(self, other: ModelSequenceInfo) -> bool:
        """Check if this sequence info is older (stale) compared to another.

        A sequence is considered stale if it has a lower sequence number,
        indicating it represents an earlier state that should not overwrite
        a more recent state.

        For Kafka transports, when sequences are equal, the offset is used
        as a tiebreaker within the same partition.

        Args:
            other: The sequence info to compare against (presumed current state)

        Returns:
            True if this sequence is older than other, False otherwise

        Example:
            >>> old = ModelSequenceInfo(sequence=10)
            >>> new = ModelSequenceInfo(sequence=20)
            >>> old.is_stale_compared_to(new)
            True
            >>> new.is_stale_compared_to(old)
            False
            >>> same = ModelSequenceInfo(sequence=10)
            >>> old.is_stale_compared_to(same)
            False
        """
        if self.sequence < other.sequence:
            return True
        if self.sequence == other.sequence:
            # Same sequence but lower offset is stale (Kafka tiebreaker)
            if (
                self.offset is not None
                and other.offset is not None
                and self.partition == other.partition
            ):
                return self.offset < other.offset
        return False

    def is_newer_than(self, other: ModelSequenceInfo) -> bool:
        """Check if this sequence info is newer than another.

        The inverse of is_stale_compared_to - returns True if this
        sequence should replace the other.

        Args:
            other: The sequence info to compare against

        Returns:
            True if this sequence is newer than other, False otherwise

        Example:
            >>> old = ModelSequenceInfo(sequence=10)
            >>> new = ModelSequenceInfo(sequence=20)
            >>> new.is_newer_than(old)
            True
        """
        return other.is_stale_compared_to(self)

    @classmethod
    def from_kafka(cls, partition: int, offset: int) -> ModelSequenceInfo:
        """Create sequence info from Kafka message metadata.

        Convenience factory for creating sequence info from Kafka consumer
        message metadata. Uses offset as the sequence number.

        Args:
            partition: Kafka partition number
            offset: Kafka offset within the partition

        Returns:
            ModelSequenceInfo configured for Kafka transport

        Example:
            >>> seq = ModelSequenceInfo.from_kafka(partition=0, offset=12345)
            >>> seq.sequence
            12345
            >>> seq.partition
            '0'
        """
        return cls(
            sequence=offset,
            partition=str(partition),
            offset=offset,
        )

    @classmethod
    def from_sequence(cls, sequence: int) -> ModelSequenceInfo:
        """Create sequence info from a generic sequence number.

        Convenience factory for non-Kafka transports that use
        application-provided sequence numbers.

        Args:
            sequence: Application-provided sequence number

        Returns:
            ModelSequenceInfo with only sequence set

        Example:
            >>> seq = ModelSequenceInfo.from_sequence(42)
            >>> seq.sequence
            42
            >>> seq.partition is None
            True
        """
        return cls(sequence=sequence)


__all__: list[str] = ["ModelSequenceInfo"]
