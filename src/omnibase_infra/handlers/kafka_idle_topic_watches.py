# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which contract topics a live demand watch still waits on (OMN-21629).

The boot subscription demand split (OMN-18087) defers a contract whose topics
hold no record to that topic's first record. Until then its topics are absent
from the live bus registry by design, so the runtime health monitor must not
count its projection as unattached. This module is the one place that answers
"which topics are still waiting".

Each live watch owns its own set, so two sources never clear one another's
topics. A watch's set leaves the registry when the watch ends for any reason
(stop, a cancelled or dead watch task, every topic arrived), so a stopped or
dead watch masks nothing, and a start that failed is judged by the live
registry from then on and is still reported.

This is a registry, not an ONEX handler: it defines no ``handler_type`` or
``handler_category``, so the module name does not carry the ``handler_`` prefix
(the same arrangement as ``kafka_permanent_failure``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable

_WATCHES: list[set[str]] = []


def idle_watched_topics() -> frozenset[str]:
    """Contract topics some live watch in this process still awaits a record on."""
    return frozenset().union(*_WATCHES)


class IdleTopicWatch:
    """The topics one demand watch has not yet seen a first record on."""

    def __init__(self, topics: Iterable[str]) -> None:
        self._topics = set(topics)
        _WATCHES.append(self._topics)

    async def start(
        self, topic: str, on_first_record: Callable[[str], Awaitable[None]]
    ) -> None:
        """Run the topic's start, then let the live registry decide about it.

        The topic leaves the idle set after the start has run, whether or not it
        raised, so a started contract is never read as unattached before it is
        wired and a start that failed is reported.
        """
        try:
            await on_first_record(topic)
        finally:
            self._topics.discard(topic)

    def release(self) -> None:
        """The watch ended; none of its topics is idle any more. Idempotent."""
        _WATCHES[:] = [watch for watch in _WATCHES if watch is not self._topics]


__all__: list[str] = ["IdleTopicWatch", "idle_watched_topics"]
