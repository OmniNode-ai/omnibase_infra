# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol matching the AIOKafkaAdminClient methods used by the runtime.

Defined here rather than importing ``AIOKafkaAdminClient`` directly so that
modules using only this shape avoid the hard ``aiokafka`` import at module
parse time.

.. versionadded:: 0.39.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol


class ProtocolKafkaAdminLike(Protocol):
    """Minimal protocol for Kafka admin clients.

    Any object exposing ``start``, ``close``, ``list_consumer_groups``, and
    ``describe_consumer_groups`` satisfies this protocol, including
    ``AIOKafkaAdminClient``.
    """

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def list_consumer_groups(
        self, broker_ids: Sequence[int] | None = None
    ) -> Sequence[tuple[str, str | None]]:
        pass

    async def describe_consumer_groups(
        self,
        group_ids: Sequence[str],
        group_coordinator_id: int | None = None,
        include_authorized_operations: bool = False,
    ) -> list[object]:
        pass

    async def list_consumer_group_offsets(
        self,
        group_id: str,
        group_coordinator_id: int | None = None,
        partitions: Sequence[object] | None = None,
    ) -> Mapping[object, object]:
        """Return committed offsets for ``group_id``.

        Maps each ``TopicPartition`` the group has committed to an
        ``OffsetAndMetadata``-like object exposing ``.offset``. Added in
        OMN-12623 so consumer-group lag (committed vs log-end) is observable for
        the drain-proof migration gate; the prior protocol only exposed group
        state/existence.
        """
        ...

    async def list_offsets(
        self,
        topic_partitions: Mapping[object, int],
    ) -> Mapping[object, object]:
        """Return per-partition log-end (or timestamp-resolved) offsets.

        Maps each requested ``TopicPartition`` to an offset response exposing
        ``.offset`` (the high-water mark when ``-1`` / ``LATEST`` is requested).
        Used with :meth:`list_consumer_group_offsets` to compute lag.
        """
        ...


__all__: list[str] = ["ProtocolKafkaAdminLike"]
