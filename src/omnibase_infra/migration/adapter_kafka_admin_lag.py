# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Admin adapter that supplies the lag surface aiokafka 0.13.0 omits (OMN-12632).

The pinned ``aiokafka==0.13.0`` ``AIOKafkaAdminClient`` exposes
``list_consumer_group_offsets`` but has **no** ``list_offsets`` method, so
``ServiceConsumerLagObserver`` — which needs both committed offsets and
per-partition log-end offsets — crashes with ``AttributeError`` against the real
runtime admin. CI never caught it because the unit tests stubbed
``list_offsets`` onto a fake admin (OMN-12623).

This adapter satisfies the full :class:`ProtocolKafkaAdminLike` surface by
delegating every committed-offset / group method to the real admin and serving
``list_offsets`` (log-end / high-water marks) through an
``AIOKafkaConsumer.end_offsets`` call, which *does* exist on 0.13.0. The observer
is wired to this adapter rather than a raw ``AIOKafkaAdminClient``, so the lag
path works against the pinned client.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.migration.protocols.protocol_kafka_lag_consumer import (
        ProtocolKafkaLagConsumer,
    )
    from omnibase_infra.protocols.protocol_kafka_admin_like import (
        ProtocolKafkaAdminLike,
    )

# aiokafka's log-end / high-water-mark request sentinel.
_OFFSET_LATEST = -1


class AdapterKafkaAdminLag:
    """``ProtocolKafkaAdminLike`` over a real admin plus a consumer for log-end offsets.

    Committed offsets and group lifecycle delegate to ``admin``; log-end offsets
    (``list_offsets``) are served via ``consumer.end_offsets`` because the pinned
    ``AIOKafkaAdminClient`` (0.13.0) has no ``list_offsets`` of its own.
    """

    def __init__(
        self,
        admin: ProtocolKafkaAdminLike,
        consumer: ProtocolKafkaLagConsumer,
    ) -> None:
        self._admin = admin
        self._consumer = consumer

    async def start(self) -> None:
        await self._admin.start()

    async def stop(self) -> None:
        await self._admin.stop()

    async def close(self) -> None:
        await self._admin.close()

    async def list_consumer_groups(
        self, broker_ids: Sequence[int] | None = None
    ) -> Sequence[tuple[str, str | None]]:
        return await self._admin.list_consumer_groups(broker_ids)

    async def describe_consumer_groups(
        self,
        group_ids: Sequence[str],
        group_coordinator_id: int | None = None,
        include_authorized_operations: bool = False,
    ) -> list[object]:
        return await self._admin.describe_consumer_groups(
            group_ids,
            group_coordinator_id=group_coordinator_id,
            include_authorized_operations=include_authorized_operations,
        )

    async def list_consumer_group_offsets(
        self,
        group_id: str,
        group_coordinator_id: int | None = None,
        partitions: Sequence[object] | None = None,
    ) -> Mapping[object, object]:
        return await self._admin.list_consumer_group_offsets(
            group_id,
            group_coordinator_id=group_coordinator_id,
            partitions=partitions,
        )

    async def list_offsets(
        self,
        topic_partitions: Mapping[object, int],
    ) -> Mapping[object, object]:
        """Serve per-partition log-end offsets via the consumer's ``end_offsets``.

        Only the high-water-mark request (``_OFFSET_LATEST``) is supported — that
        is the sole shape :class:`ServiceConsumerLagObserver` issues. Any other
        timestamp/offset target is rejected rather than silently mis-served,
        because ``end_offsets`` cannot resolve arbitrary timestamps.
        """
        unsupported = [
            tp for tp, target in topic_partitions.items() if target != _OFFSET_LATEST
        ]
        if unsupported:
            raise ValueError(
                "AdapterKafkaAdminLag.list_offsets only serves log-end "
                f"(LATEST/{_OFFSET_LATEST}) requests via end_offsets; got "
                f"non-LATEST targets for {unsupported!r}"
            )
        partitions = list(topic_partitions.keys())
        return await self._consumer.end_offsets(partitions)


__all__ = ["AdapterKafkaAdminLag"]
