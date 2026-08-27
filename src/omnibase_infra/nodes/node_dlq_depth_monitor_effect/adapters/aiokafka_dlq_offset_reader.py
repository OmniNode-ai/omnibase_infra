# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live ``ProtocolDlqAdminTransport`` over aiokafka (OMN-16769).

Every method the DLQ probe needs already exists on ``AIOKafkaConsumer``
(``topics`` / ``partitions_for_topic`` / ``beginning_offsets`` /
``end_offsets`` / ``offsets_for_times``), so this adapter needs no
``AIOKafkaAdminClient`` at all — unlike the lag path, which needed
``AdapterKafkaAdminLag`` (OMN-12632) precisely because the pinned admin
client lacks ``list_offsets``.

The consumer is constructed with **no ``group_id`` and no subscription**.
That is deliberate and load-bearing: a group-less consumer joins no
consumer group, is assigned nothing, and commits nothing, so this probe
cannot perturb the lag or delivery state of the very topics it is
observing. It reads metadata and offsets only.

``aiokafka`` is imported inside the constructor rather than at module
scope so that importing this node — which the contract loader does at
startup — never hard-requires the Kafka client.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_dlq_depth_monitor_effect.protocols.protocol_cluster_metadata import (
    ProtocolClusterMetadata,
)
from omnibase_infra.nodes.node_dlq_depth_monitor_effect.protocols.protocol_topic_partition import (
    ProtocolTopicPartition,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_dlq_depth_monitor_effect.protocols.protocol_dlq_admin_transport import (
        TopicPartition,
    )


class AiokafkaDlqOffsetReader:
    """Read-only offset reader backed by a group-less ``AIOKafkaConsumer``.

    NOT named ``Adapter*``: the OMN-14350 non-canonical-lifecycle ratchet
    hard-fails ``Adapter`` (and ``Client``) as type-words, and its allowlist
    is a shrink-only ratchet reserved for pre-existing residuals being burned
    down — new code does not belong on it. ``Reader`` states what this does
    and carries no lifecycle-owner connotation.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        *,
        request_timeout_ms: int = 20_000,
        client_id: str = "onex-dlq-depth-monitor",
    ) -> None:
        from aiokafka import (
            AIOKafkaConsumer,
        )
        from aiokafka.structs import (
            TopicPartition as AiokafkaTopicPartition,
        )

        self._tp_type = AiokafkaTopicPartition
        self._consumer = AIOKafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            # No group_id: join no group, commit nothing, perturb nothing.
            group_id=None,
            enable_auto_commit=False,
            client_id=client_id,
            request_timeout_ms=request_timeout_ms,
        )
        self._started = False
        # Cluster metadata snapshot captured by list_topics(). See the comment
        # on partitions_for_topic for why this cannot be read off the client.
        self._cluster: ProtocolClusterMetadata | None = None

    async def __aenter__(self) -> AiokafkaDlqOffsetReader:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.stop()

    async def start(self) -> None:
        if not self._started:
            await self._consumer.start()
            self._started = True

    async def stop(self) -> None:
        if self._started:
            await self._consumer.stop()
            self._started = False

    def _to_aiokafka(self, partition: TopicPartition) -> ProtocolTopicPartition:
        tp: ProtocolTopicPartition = self._tp_type(partition[0], partition[1])
        return tp

    async def list_topics(self) -> Sequence[str]:
        """Fetch cluster metadata and RETAIN it for partition lookups.

        ``AIOKafkaConsumer.topics()`` is implemented as
        ``(await self._client.fetch_all_metadata()).topics()`` — and
        ``fetch_all_metadata()`` returns a NEW ``ClusterMetadata`` object
        rather than updating ``client.cluster`` in place. The consumer's own
        ``partitions_for_topic()`` reads ``client.cluster``, which for a
        group-less consumer that has subscribed to nothing knows about no
        topics at all.

        Calling ``consumer.topics()`` and then
        ``consumer.partitions_for_topic()`` therefore yields the full topic
        list and zero partitions for every one of them. That was observed
        live against the .201 dev lane on 2026-08-27: 60 DLQ topics matched,
        0 observed. So the metadata object is captured here and used directly
        below instead of going back through the consumer.
        """
        # Private `_client` access is deliberate: the PUBLIC accessor
        # (`consumer.topics()`) throws this very object away, which is the
        # whole bug documented above.
        self._cluster = await self._consumer._client.fetch_all_metadata()
        return sorted(self._cluster.topics())

    async def partitions_for_topic(self, topic: str) -> Sequence[int]:
        if self._cluster is None:
            # Partition lookups are only meaningful against the metadata
            # snapshot list_topics() took; refusing here beats silently
            # reporting every topic as partition-less, which reads
            # identically to a clean bill of health.
            raise RuntimeError(
                "partitions_for_topic called before list_topics — no cluster "
                "metadata snapshot has been taken."
            )
        partitions = self._cluster.partitions_for_topic(topic)
        # None for a topic whose metadata is not resolvable. Surfacing that as
        # an empty partition set lets the handler skip the topic rather than
        # fabricate a zero-depth observation for it.
        return sorted(partitions) if partitions else []

    async def beginning_offsets(
        self, partitions: Sequence[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        raw = await self._consumer.beginning_offsets(
            [self._to_aiokafka(partition) for partition in partitions]
        )
        return {(tp.topic, tp.partition): offset for tp, offset in raw.items()}

    async def end_offsets(
        self, partitions: Sequence[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        raw = await self._consumer.end_offsets(
            [self._to_aiokafka(partition) for partition in partitions]
        )
        return {(tp.topic, tp.partition): offset for tp, offset in raw.items()}

    async def offsets_for_times(
        self, partition_timestamps: Mapping[TopicPartition, int]
    ) -> Mapping[TopicPartition, int | None]:
        raw = await self._consumer.offsets_for_times(
            {
                self._to_aiokafka(partition): timestamp
                for partition, timestamp in partition_timestamps.items()
            }
        )
        # aiokafka yields OffsetAndTimestamp | None per partition. None means
        # no record at or after the requested time — the handler normalizes
        # that to the high-water mark; it must NOT become offset 0.
        return {
            (tp.topic, tp.partition): (None if value is None else value.offset)
            for tp, value in raw.items()
        }


__all__ = ["AiokafkaDlqOffsetReader"]
