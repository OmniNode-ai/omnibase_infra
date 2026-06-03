# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumer-group lag observation service (OMN-12623).

Computes :class:`ModelConsumerGroupLag` for a consumer group by combining the
group's committed offsets (``list_consumer_group_offsets``) with the partitions'
log-end offsets (``list_offsets``) via the extended
:class:`ProtocolKafkaAdminLike`.

This is the observation half of the topic-migration drain-proof gate: the gate
(:class:`ServiceDrainProofGate`) consumes the lag this service produces and
refuses to retire an old consumer group until its lag is zero.

The service is transport-agnostic at the type level: it depends only on the
structural protocol, so unit tests inject a fake admin client and no live Kafka
is required to prove the lag/gate logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.migration.models.model_consumer_group_lag import (
    ModelConsumerGroupLag,
)
from omnibase_infra.migration.models.model_topic_partition_offset import (
    ModelTopicPartitionOffset,
)

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_kafka_admin_like import (
        ProtocolKafkaAdminLike,
    )

logger = logging.getLogger(__name__)

# aiokafka's OffsetResponse uses -1 for LATEST high-water-mark requests.
_OFFSET_LATEST = -1


def _coerce_offset(value: object) -> int:
    """Normalize an offset response (or raw int) to a non-negative int.

    Accepts either a raw int or an ``OffsetAndMetadata``/``OffsetResponse``-like
    object exposing ``.offset``. Upstream sentinels (``-1`` meaning "no committed
    offset") map to ``0`` — no consumption progress — never to a negative lag.
    """
    raw: object = value.offset if hasattr(value, "offset") else value
    if not isinstance(raw, int):
        raise TypeError(f"Offset value {value!r} is not an int and has no .offset")
    return max(raw, 0)


def _require_topic_partition(tp: object) -> tuple[str, int]:
    """Extract (topic, partition) from a TopicPartition-like key or fail fast."""
    if not (hasattr(tp, "topic") and hasattr(tp, "partition")):
        raise TypeError(
            f"committed-offset key {tp!r} is not a TopicPartition "
            "(missing .topic/.partition)"
        )
    topic = tp.topic
    partition = tp.partition
    if not isinstance(topic, str) or not isinstance(partition, int):
        raise TypeError(f"TopicPartition {tp!r} has non-str topic or non-int partition")
    return topic, partition


class ServiceConsumerLagObserver:
    """Computes consumer-group lag from committed vs log-end offsets."""

    def __init__(self, admin: ProtocolKafkaAdminLike) -> None:
        self._admin = admin

    async def observe(self, group_id: str) -> ModelConsumerGroupLag:
        """Observe per-partition lag for ``group_id``.

        Queries the group's committed offsets, then the log-end offsets for the
        exact same partitions, and returns the aggregated lag.

        Raises:
            ValueError: if ``group_id`` is empty.
        """
        if not group_id:
            raise ValueError("group_id must be a non-empty consumer group id")

        committed = await self._admin.list_consumer_group_offsets(group_id)

        if not committed:
            logger.debug(
                "consumer group %s has no committed offsets; lag is unobservable",
                group_id,
            )
            return ModelConsumerGroupLag(group_id=group_id, partition_offsets=())

        end_request: dict[object, int] = dict.fromkeys(committed, _OFFSET_LATEST)
        end_offsets = await self._admin.list_offsets(end_request)

        partition_offsets: list[ModelTopicPartitionOffset] = []
        for tp, committed_meta in committed.items():
            topic, partition = _require_topic_partition(tp)
            end_meta = end_offsets.get(tp)
            if end_meta is None:
                raise ValueError(
                    f"no log-end offset returned for partition {tp!r}; cannot "
                    "compute lag for the drain-proof gate"
                )
            partition_offsets.append(
                ModelTopicPartitionOffset(
                    topic=topic,
                    partition=partition,
                    committed_offset=_coerce_offset(committed_meta),
                    log_end_offset=_coerce_offset(end_meta),
                )
            )

        partition_offsets.sort(key=lambda po: (po.topic, po.partition))
        return ModelConsumerGroupLag(
            group_id=group_id,
            partition_offsets=tuple(partition_offsets),
        )


__all__ = ["ServiceConsumerLagObserver"]
