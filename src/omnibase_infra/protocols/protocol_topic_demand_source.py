# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol for the boot subscription demand split (OMN-18087).

The boot interleave (``subscribe_wired_contract_topics``) asks this shape which
of its subscribe topics are in use, and to report the first record that reaches
any topic that is not, so it can join a contract's consumer groups when the
contract first gets a request instead of at boot. Depending on the shape rather
than the Kafka reader keeps the interleave testable with a fake.

Related Tickets:
    - OMN-18087: Runtime boot joins every consumer group; join only the topics in use.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection
from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolTopicDemandSource(Protocol):
    """Split topics into in-use and idle, and watch the idle ones."""

    async def watch_idle_topics(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
    ) -> tuple[frozenset[str], Callable[[], Awaitable[None]]]:
        """Measure ``topics`` once and start watching the idle ones.

        Returns the topics in use (holding at least one retained record) and a
        callable that stops the watch. Every other topic is watched from the
        position it had when it was measured, so a record that arrives at any
        moment after the measurement is seen. ``on_first_record`` is awaited
        once per idle topic, when its first record arrives, and that topic is
        then no longer watched.
        """
        ...


__all__: list[str] = ["ProtocolTopicDemandSource"]
