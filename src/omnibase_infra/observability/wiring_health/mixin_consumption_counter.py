# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Coroutine-safe consumption counter mixin for wiring health monitoring.

This module provides a mixin that tracks successful message consumption per-topic
for use in wiring health comparisons. It counts only configured topics and only
counts successful processing (not DLQ-routed messages).

Design Rationale:
    - Uses asyncio.Lock for coroutine-safe counter updates (not thread-safe)
    - Counts only topics in WIRING_HEALTH_MONITORED_TOPICS (bounded memory)
    - Only counts SUCCESSFUL consumption (after dispatch engine succeeds)
    - Does NOT count DLQ-routed messages (those are failures, not consumption)
    - Copy-on-write semantics for get_consumption_counts() (no lock held during read)

Usage:
    ```python
    from omnibase_infra.observability.wiring_health import MixinConsumptionCounter

    class EventBusSubcontractWiring(MixinConsumptionCounter):
        def __init__(self, ...):
            self._init_consumption_counter()
            # ... other init ...

        async def callback(message: ProtocolEventMessage) -> None:
            # ... dispatch to engine ...
            await self._dispatch_engine.dispatch(topic, envelope)
            # Success - record consumption
            await self._record_consumption(topic)
    ```

Concurrency Safety:
    Uses asyncio.Lock which provides coroutine-safe access within a single
    event loop. Not thread-safe. For multi-threaded usage, additional
    synchronization would be required.

See Also:
    - OMN-1895: Wiring health monitor implementation
    - topic_constants.py: WIRING_HEALTH_MONITORED_TOPICS
    - mixin_emission_counter.py: Emission side of the comparison
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from omnibase_infra.event_bus.topic_constants import WIRING_HEALTH_MONITORED_TOPICS

if TYPE_CHECKING:
    from typing import Final

_logger = logging.getLogger(__name__)


class MixinConsumptionCounter:
    """Mixin providing consumption counting for wiring health monitoring.

    Tracks successful message consumption per-topic for comparison against
    emission counts. Only tracks topics configured in WIRING_HEALTH_MONITORED_TOPICS
    to bound memory usage. Does NOT count DLQ-routed messages.

    Attributes:
        _consumption_counts: Per-topic consumption counts (topic -> count)
        _consumption_counter_lock: asyncio.Lock for coroutine-safe updates
        _consumption_monitored_topics: Set of topics to track (from constants)

    Example:
        >>> class EventBusSubcontractWiring(MixinConsumptionCounter):
        ...     def __init__(self):
        ...         self._init_consumption_counter()
        ...
        ...     async def callback(self, message):
        ...         # ... process message ...
        ...         await self._record_consumption(topic)
        ...
        >>> wiring = EventBusSubcontractWiring()
        >>> await wiring.callback(message)  # topic = "onex.evt.omniclaude.agent-match.v1"
        >>> wiring.get_consumption_counts()
        {'onex.evt.omniclaude.agent-match.v1': 1}
    """

    # Class-level constants
    _CONSUMPTION_MONITORED_TOPICS: Final[frozenset[str]] = frozenset(
        WIRING_HEALTH_MONITORED_TOPICS
    )

    def _init_consumption_counter(self) -> None:
        """Initialize consumption counter state.

        Must be called during __init__ of the class using this mixin.
        Creates the counter dict and asyncio.Lock.

        Example:
            >>> class EventBusSubcontractWiring(MixinConsumptionCounter):
            ...     def __init__(self, ...):
            ...         self._init_consumption_counter()
            ...         # ... other initialization ...
        """
        self._consumption_counts: dict[str, int] = {}
        self._consumption_counter_lock: asyncio.Lock = asyncio.Lock()

        _logger.debug(
            "Consumption counter initialized",
            extra={
                "monitored_topics": list(self._CONSUMPTION_MONITORED_TOPICS),
            },
        )

    async def _record_consumption(self, topic: str) -> None:
        """Record a successful message consumption for a topic.

        Increments the consumption counter for the given topic if it is in the
        monitored topics set. Topics not in the monitored set are ignored
        to bound memory usage.

        IMPORTANT: Only call this method after SUCCESSFUL dispatch. Do NOT call
        this for messages that were routed to DLQ or failed processing.

        This method is coroutine-safe - multiple concurrent calls will
        correctly increment the counter.

        Args:
            topic: The topic name that was successfully consumed.

        Example:
            >>> async def callback(message):
            ...     await self._dispatch_engine.dispatch(topic, envelope)
            ...     # Success - record consumption
            ...     await self._record_consumption(topic)
        """
        # Fast path: skip topics not in monitored set
        if topic not in self._CONSUMPTION_MONITORED_TOPICS:
            return

        async with self._consumption_counter_lock:
            self._consumption_counts[topic] = self._consumption_counts.get(topic, 0) + 1
            count = self._consumption_counts[topic]

        _logger.debug(
            "Recorded consumption",
            extra={
                "topic": topic,
                "count": count,
            },
        )

    def get_consumption_counts(self) -> dict[str, int]:
        """Get a snapshot of current consumption counts.

        Returns a copy of the consumption counts dict to avoid exposing
        internal state. The copy is taken without holding the lock,
        so it represents a point-in-time snapshot that may be slightly
        stale if consumptions are occurring concurrently.

        For wiring health monitoring purposes, slight staleness is
        acceptable since comparisons are periodic (Prometheus scrape interval).

        Returns:
            Dictionary mapping topic name to consumption count.
            Only includes topics that have had at least one successful consumption.

        Example:
            >>> counts = wiring.get_consumption_counts()
            >>> for topic, count in counts.items():
            ...     print(f"{topic}: {count}")
        """
        # Return a copy to avoid exposing internal state
        # dict() creates a shallow copy which is sufficient for str->int
        return dict(self._consumption_counts)

    async def reset_consumption_counts(self) -> dict[str, int]:
        """Reset consumption counts and return the previous values.

        Atomically resets all counters to zero and returns the counts
        that were reset. Useful for testing and periodic counter reset.

        Returns:
            Dictionary of consumption counts before reset.

        Example:
            >>> old_counts = await wiring.reset_consumption_counts()
            >>> print(f"Reset {sum(old_counts.values())} consumptions")
        """
        async with self._consumption_counter_lock:
            old_counts = dict(self._consumption_counts)
            self._consumption_counts.clear()

        _logger.info(
            "Consumption counts reset",
            extra={
                "reset_counts": old_counts,
                "total_reset": sum(old_counts.values()),
            },
        )

        return old_counts

    def get_consumption_monitored_topics(self) -> frozenset[str]:
        """Get the set of topics being monitored for consumption.

        Returns:
            Immutable set of topic names configured for monitoring.

        Example:
            >>> topics = wiring.get_consumption_monitored_topics()
            >>> print(f"Monitoring {len(topics)} topics")
        """
        return self._CONSUMPTION_MONITORED_TOPICS


__all__ = ["MixinConsumptionCounter"]
