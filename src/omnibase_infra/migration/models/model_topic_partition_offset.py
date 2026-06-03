# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Single (topic, partition) committed-vs-end offset model (OMN-12623).

The atomic unit of consumer-group lag observation: one partition's committed
offset (for a group) against its log-end offset. Lag is the non-negative
difference.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelTopicPartitionOffset(BaseModel):
    """A single (topic, partition) committed-vs-end offset pair.

    ``committed_offset`` is the last offset the consumer group has committed for
    this partition (upstream ``-1`` / ``None`` is normalized to ``0`` by the
    observer, meaning no progress). ``log_end_offset`` is the high-water mark of
    the partition. Lag is the non-negative difference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = Field(..., min_length=1, description="Canonical topic name")
    partition: int = Field(..., ge=0, description="Kafka partition index")
    committed_offset: int = Field(
        ...,
        ge=0,
        description="Last committed offset for the consumer group on this partition",
    )
    log_end_offset: int = Field(
        ...,
        ge=0,
        description="High-water mark (log-end offset) of the partition",
    )

    @model_validator(mode="after")
    def _validate_offsets(self) -> ModelTopicPartitionOffset:
        if self.committed_offset > self.log_end_offset:
            raise ValueError(
                f"committed_offset ({self.committed_offset}) exceeds "
                f"log_end_offset ({self.log_end_offset}) for "
                f"{self.topic}[{self.partition}]; offsets are inconsistent."
            )
        return self

    @property
    def lag(self) -> int:
        """Un-consumed messages on this partition for the group (>= 0)."""
        return self.log_end_offset - self.committed_offset


__all__ = ["ModelTopicPartitionOffset"]
