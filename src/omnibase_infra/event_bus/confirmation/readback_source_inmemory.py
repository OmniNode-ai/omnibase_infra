# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Zero-infra readback surface: the in-memory bus's own history (OMN-15861).

This is the proof surface the OMN-15861 brief scopes the falsifiable version to
-- one process, no broker, no LAN. It is a *real* readback, not a stub: it
consults the bus's committed history by the exact ``(topic, partition, offset)``
coordinate the bus assigned at publish time, so a receipt whose offset does not
match a stored record is genuinely ``UNCONFIRMED``.

Bounded-history caveat (deliberate, not a bug): the in-memory bus keeps a
``deque(maxlen=max_history)``. A record evicted by newer traffic is no longer
observable and reads back as ``UNCONFIRMED``, which fails closed -- the record
stays in the outbox and is retried. That is the correct direction to be wrong in.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol, runtime_checkable

from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt

_POLL_SLICE_SECONDS = 0.01


@runtime_checkable
class ProtocolInmemoryHistorySource(Protocol):
    """The slice of the in-memory bus this source needs.

    Narrowed to one method so tests can substitute a history without standing up
    a bus, and so this module does not depend on the whole bus surface.
    """

    async def get_event_history(
        self,
        limit: int = 100,
        topic: str | None = None,
    ) -> list[object]:
        """Return recent messages, optionally filtered to ``topic``."""
        ...


class InmemoryReadbackSource:
    """Reads a coordinate back off an in-memory bus's event history.

    Args:
        bus: The in-memory bus whose history is authoritative.
        cluster: Cluster identity this bus publishes under. Compared against
            ``receipt.cluster`` so a receipt minted by a *different* in-memory
            bus in the same process cannot be confirmed here.
        history_limit: How far back to scan. Must exceed the expected in-flight
            depth, or a slow confirm behind a burst reads back ``UNCONFIRMED``.
    """

    def __init__(
        self,
        bus: ProtocolInmemoryHistorySource,
        *,
        cluster: str,
        history_limit: int = 1000,
    ) -> None:
        self._bus = bus
        self._cluster = cluster
        self._history_limit = history_limit

    @property
    def transport(self) -> EnumInfraTransportType:
        """This source can only answer for in-memory receipts."""
        return EnumInfraTransportType.INMEMORY

    async def observe(
        self,
        receipt: ModelPublishReceipt,
        *,
        deadline_seconds: float,
    ) -> bool:
        """Poll the bus history for ``receipt``'s coordinate until the deadline.

        Polling rather than a single lookup because ``publish`` appends to
        history and then fans out to subscribers; a confirmation racing that
        fan-out must be allowed to catch up, exactly as the Kafka readback loop
        allows for produce-to-fetch latency.
        """
        if receipt.cluster != self._cluster:
            return False

        deadline_at = time.monotonic() + deadline_seconds
        while True:
            history = await self._bus.get_event_history(
                limit=self._history_limit, topic=receipt.topic
            )
            for message in history:
                if self._matches(message, receipt):
                    return True
            if time.monotonic() >= deadline_at:
                return False
            await asyncio.sleep(_POLL_SLICE_SECONDS)

    @staticmethod
    def _matches(message: object, receipt: ModelPublishReceipt) -> bool:
        """Structural coordinate match against a bus message.

        The in-memory bus stores ``offset`` as ``str`` and ``partition`` as
        ``int``; both are normalised here rather than assuming either shape, so
        this keeps working if the message model tightens.
        """
        topic = getattr(message, "topic", None)
        partition = getattr(message, "partition", None)
        offset = getattr(message, "offset", None)
        if topic != receipt.topic or partition is None or offset is None:
            return False
        try:
            return int(partition) == receipt.partition and int(offset) == receipt.offset
        except (TypeError, ValueError):
            return False


__all__: list[str] = [
    "InmemoryReadbackSource",
    "ProtocolInmemoryHistorySource",
]
