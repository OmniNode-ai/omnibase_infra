# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Aggregated consumer-group lag model (OMN-12623).

The structured result of comparing a consumer group's committed offsets against
the log-end offsets of the partitions it consumes. Makes the drain-proof gate
decidable: a topic is *drained* for a group iff its total lag is zero.

Before this module the only Kafka admin observation available
(:class:`ProtocolKafkaAdminLike`) was group state/existence; lag was not
observable, so "block retirement until lag drains" had nothing to assert on.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.migration.models.model_topic_partition_offset import (
    ModelTopicPartitionOffset,
)


class ModelConsumerGroupLag(BaseModel):
    """Aggregated lag for one consumer group across the partitions it consumes.

    ``is_drained`` is the drain-proof gate's decision input: True iff the group
    has zero total lag (it has consumed every committed-visible message on its
    partitions).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    group_id: str = Field(..., min_length=1, description="Consumer group id")
    partition_offsets: tuple[ModelTopicPartitionOffset, ...] = Field(
        ...,
        description="Per-partition committed-vs-end offsets for this group",
    )

    @property
    def total_lag(self) -> int:
        """Sum of per-partition lag across all observed partitions."""
        return sum(po.lag for po in self.partition_offsets)

    @property
    def is_drained(self) -> bool:
        """True iff the group has fully consumed every observed partition.

        A group with no observed partitions is NOT considered drained — absence
        of evidence is not proof of drain; the gate must observe at least one
        partition to assert drain.
        """
        if not self.partition_offsets:
            return False
        return self.total_lag == 0

    def lag_for_topic(self, topic: str) -> int:
        """Total lag restricted to a single topic's partitions."""
        return sum(po.lag for po in self.partition_offsets if po.topic == topic)

    def has_partitions_for_topic(self, topic: str) -> bool:
        """True iff at least one observed partition belongs to ``topic``.

        The drain-proof gate uses this to distinguish "drained" (observed,
        zero lag) from "no evidence for this topic" (no observed partitions) —
        absence of evidence is not proof of drain.
        """
        return any(po.topic == topic for po in self.partition_offsets)


__all__ = ["ModelConsumerGroupLag"]
