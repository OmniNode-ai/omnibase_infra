# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Phantom-subscription alarm for the S6 core runtime (epic OMN-14717, §d).

The structural guarantee (topic-keyed unconditional dispatch) already makes the
OMN-14755 phantom class impossible for an allowlisted topic. This alarm is the
belt-and-suspenders health signal the ticket asks for: "subscribed + zero dispatch at
LAG=0 = health failure".

Design — a thin infra wrapper over the transport, NO edit to merged core:

* ``CountingTransportConsumer`` decorates the injected transport consumer (the canonical
  ``ProtocolTransportConsumer``) and counts, per topic, how many messages were POLLED (the
  topic's high-water mark advanced past the committed offset => messages existed) and how
  many were COMMITTED by ``RuntimeDispatch`` (a commit follows a successful dispatch or a
  DLQ, i.e. the message was processed — ``dispatched_count``).
* ``PhantomAlarmMonitor`` owns the counting consumer and evaluates the alarm over
  ``core_runtime_topics`` ONLY: a topic that is ASSIGNED, whose HWM ADVANCED (polled > 0),
  whose LAG has returned to 0, and whose ``dispatched_count`` is still 0 => PHANTOM =>
  flip runtime readiness FAIL + structured log ``PHANTOM_SUBSCRIPTION topic=<t>``. It does
  NOT tear the loop down (a phantom is a wiring bug to surface, not a reason to stop
  consuming).

The LAG-zero set is supplied by the caller from the kernel's existing topic/HWM readback
tooling; the counting + evaluation logic here is pure and unit-testable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

from omnibase_core.protocols.runtime.protocol_transport_consumer import (
    ProtocolTransportConsumer,
)
from omnibase_core.protocols.runtime.protocol_transport_message import (
    ProtocolTransportMessage,
)

logger = logging.getLogger(__name__)

__all__ = ["CountingTransportConsumer", "PhantomAlarmMonitor"]


class CountingTransportConsumer:
    """Transport-consumer decorator that counts polled + committed messages per topic.

    Satisfies the same pull-based ``ProtocolTransportConsumer`` surface
    (``start``/``close``/``poll``/``commit``/``nack``) and delegates every call to the
    wrapped consumer, observing topic activity as a side effect.
    """

    def __init__(self, inner: ProtocolTransportConsumer) -> None:
        self._inner = inner
        self.polled_by_topic: dict[str, int] = defaultdict(int)
        self.dispatched_by_topic: dict[str, int] = defaultdict(int)
        self.assigned_topics: set[str] = set()

    async def start(self) -> None:
        await self._inner.start()

    async def close(self) -> None:
        await self._inner.close()

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ProtocolTransportMessage]:
        batch = await self._inner.poll(max_messages=max_messages, timeout_ms=timeout_ms)
        for message in batch:
            self.polled_by_topic[message.topic] += 1
            self.assigned_topics.add(message.topic)
        return batch

    async def commit(self, message: ProtocolTransportMessage) -> None:
        await self._inner.commit(message)
        # A commit is RuntimeDispatch's terminalization signal (dispatched or DLQ'd).
        self.dispatched_by_topic[message.topic] += 1

    async def nack(self, message: ProtocolTransportMessage) -> None:
        await self._inner.nack(message)


class PhantomAlarmMonitor:
    """Owns the counting consumer and evaluates the phantom-subscription alarm (§d)."""

    def __init__(
        self,
        inner_consumer: ProtocolTransportConsumer,
        *,
        core_runtime_topics: frozenset[str],
    ) -> None:
        self._counting = CountingTransportConsumer(inner_consumer)
        self._core_runtime_topics = core_runtime_topics

    @property
    def consumer(self) -> CountingTransportConsumer:
        """The counting consumer to hand to ``RuntimeDispatch(consumer=...)``."""
        return self._counting

    def evaluate(self, *, lag_zero_topics: frozenset[str]) -> frozenset[str]:
        """Return the set of PHANTOM allowlist topics given the LAG-zero readback (§d.3).

        A topic is phantom when it is assigned, its HWM advanced (messages were polled),
        its LAG is now 0, and zero messages were dispatched (committed). Scoped strictly
        to ``core_runtime_topics``.
        """
        phantom: set[str] = set()
        for topic in self._core_runtime_topics:
            assigned = topic in self._counting.assigned_topics
            hwm_advanced = self._counting.polled_by_topic.get(topic, 0) > 0
            dispatched = self._counting.dispatched_by_topic.get(topic, 0)
            if (
                assigned
                and hwm_advanced
                and topic in lag_zero_topics
                and dispatched == 0
            ):
                phantom.add(topic)
        for topic in sorted(phantom):
            logger.error(
                "PHANTOM_SUBSCRIPTION topic=%s: assigned + HWM advanced + LAG=0 + "
                "dispatched_count=0 — a message existed and was consumed but never "
                "dispatched (OMN-14755 signature). Flipping runtime readiness FAIL; "
                "loop stays live.",
                topic,
            )
        return frozenset(phantom)

    def is_healthy(self, *, lag_zero_topics: frozenset[str]) -> bool:
        """Tri-state input: True when no allowlist topic is phantom."""
        return not self.evaluate(lag_zero_topics=lag_zero_topics)
