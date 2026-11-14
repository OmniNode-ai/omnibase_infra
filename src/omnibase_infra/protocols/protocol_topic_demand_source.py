# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol for the boot subscription demand source (OMN-21318).

The boot wiring deferral (``boot_wiring_deferral``) depends on this shape
rather than the concrete ``HandlerKafkaTopicDemandSource`` so the deferral
module stays decoupled from the sibling handler at module-parse time, and a
test can substitute a fake demand source that records what was watched.

Related Tickets:
    - OMN-21318: A demand-deferred contract is not wired at boot at all.
    - OMN-18087: The boot subscription split.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection
from typing import Protocol


class ProtocolTopicDemandSource(Protocol):
    """Minimal demand-source shape the boot wiring deferral needs.

    ``watch_idle_topics`` measures the given topics once and then watches the
    idle ones: it returns the topics found in use and the stop of the watch,
    and calls ``on_first_record`` with the topic of each first record.
    """

    async def watch_idle_topics(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
    ) -> tuple[frozenset[str], Callable[[], Awaitable[None]]]:
        """Measure ``topics``; watch the idle ones for their first record."""
        ...


__all__: list[str] = ["ProtocolTopicDemandSource"]
