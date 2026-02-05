# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Coroutine-safe emission counter mixin for EventBus wiring health monitoring.

This module provides a mixin that tracks message emissions per-topic for use
in wiring health comparisons. It counts only configured topics to avoid
unbounded memory growth from tracking all possible topics.

Design Rationale:
    - Uses asyncio.Lock for coroutine-safe counter updates (not thread-safe)
    - Counts only topics in WIRING_HEALTH_MONITORED_TOPICS (bounded memory)
    - Copy-on-write semantics for get_emission_counts() (no lock held during read)
    - Counter reset capability for testing and periodic reset

Usage:
    ```python
    from omnibase_infra.observability.wiring_health import MixinEmissionCounter

    class EventBusKafka(MixinAsyncCircuitBreaker, MixinEmissionCounter):
        def __init__(self, config):
            self._init_emission_counter()
            # ... other init ...

        async def publish(self, topic: str, key: bytes | None, value: bytes) -> None:
            # ... publish logic ...
            await self._record_emission(topic)
    ```

Concurrency Safety:
    Uses asyncio.Lock which provides coroutine-safe access within a single
    event loop. Not thread-safe. For multi-threaded usage, additional
    synchronization would be required.

See Also:
    - OMN-1895: Wiring health monitor implementation
    - topic_constants.py: WIRING_HEALTH_MONITORED_TOPICS
"""

from __future__ import annotations

import asyncio
import logging
from typing import Final

from omnibase_infra.event_bus.topic_constants import WIRING_HEALTH_MONITORED_TOPICS

_logger = logging.getLogger(__name__)


class MixinEmissionCounter:
    """Mixin providing emission counting for wiring health monitoring.

    Tracks message emissions per-topic for comparison against consumption counts.
    Only tracks topics configured in WIRING_HEALTH_MONITORED_TOPICS to bound
    memory usage.

    Attributes:
        _emission_counts: Per-topic emission counts (topic -> count)
        _emission_counter_lock: asyncio.Lock for coroutine-safe updates
        _monitored_topics: Set of topics to track (from constants)

    Example:
        >>> class MyEventBus(MixinEmissionCounter):
        ...     def __init__(self):
        ...         self._init_emission_counter()
        ...
        ...     async def publish(self, topic: str, value: bytes) -> None:
        ...         # ... publish logic ...
        ...         await self._record_emission(topic)
        ...
        >>> bus = MyEventBus()
        >>> await bus.publish("onex.evt.omniclaude.agent-match.v1", b"data")
        >>> bus.get_emission_counts()
        {'onex.evt.omniclaude.agent-match.v1': 1}
    """

    # Class-level constants
    _MONITORED_TOPICS: Final[frozenset[str]] = frozenset(WIRING_HEALTH_MONITORED_TOPICS)

    def _init_emission_counter(self) -> None:
        """Initialize emission counter state.

        Must be called during __init__ of the class using this mixin.
        Creates the counter dict and asyncio.Lock.

        Example:
            >>> class EventBusKafka(MixinEmissionCounter):
            ...     def __init__(self, config):
            ...         self._init_emission_counter()
            ...         # ... other initialization ...
        """
        self._emission_counts: dict[str, int] = {}
        self._emission_counter_lock: asyncio.Lock = asyncio.Lock()

        _logger.debug(
            "Emission counter initialized",
            extra={
                "monitored_topics": list(self._MONITORED_TOPICS),
            },
        )

    async def _record_emission(self, topic: str) -> None:
        """Record a message emission for a topic.

        Increments the emission counter for the given topic if it is in the
        monitored topics set. Topics not in the monitored set are ignored
        to bound memory usage.

        This method is coroutine-safe - multiple concurrent calls will
        correctly increment the counter.

        Args:
            topic: The topic name that was published to.

        Example:
            >>> async def publish(self, topic: str, value: bytes) -> None:
            ...     await self._kafka_producer.send(topic, value)
            ...     await self._record_emission(topic)
        """
        # Fast path: skip topics not in monitored set
        if topic not in self._MONITORED_TOPICS:
            return

        async with self._emission_counter_lock:
            self._emission_counts[topic] = self._emission_counts.get(topic, 0) + 1

        _logger.debug(
            "Recorded emission",
            extra={
                "topic": topic,
                "count": self._emission_counts.get(topic, 0),
            },
        )

    def get_emission_counts(self) -> dict[str, int]:
        """Get a snapshot of current emission counts.

        Returns a copy of the emission counts dict to avoid exposing
        internal state. The copy is taken without holding the lock,
        so it represents a point-in-time snapshot that may be slightly
        stale if emissions are occurring concurrently.

        For wiring health monitoring purposes, slight staleness is
        acceptable since comparisons are periodic (Prometheus scrape interval).

        Returns:
            Dictionary mapping topic name to emission count.
            Only includes topics that have had at least one emission.

        Example:
            >>> counts = bus.get_emission_counts()
            >>> for topic, count in counts.items():
            ...     print(f"{topic}: {count}")
        """
        # Return a copy to avoid exposing internal state
        # dict() creates a shallow copy which is sufficient for str->int
        return dict(self._emission_counts)

    async def reset_emission_counts(self) -> dict[str, int]:
        """Reset emission counts and return the previous values.

        Atomically resets all counters to zero and returns the counts
        that were reset. Useful for testing and periodic counter reset.

        Returns:
            Dictionary of emission counts before reset.

        Example:
            >>> old_counts = await bus.reset_emission_counts()
            >>> print(f"Reset {sum(old_counts.values())} emissions")
        """
        async with self._emission_counter_lock:
            old_counts = dict(self._emission_counts)
            self._emission_counts.clear()

        _logger.info(
            "Emission counts reset",
            extra={
                "reset_counts": old_counts,
                "total_reset": sum(old_counts.values()),
            },
        )

        return old_counts

    def get_monitored_topics(self) -> frozenset[str]:
        """Get the set of topics being monitored for emissions.

        Returns:
            Immutable set of topic names configured for monitoring.

        Example:
            >>> topics = bus.get_monitored_topics()
            >>> print(f"Monitoring {len(topics)} topics")
        """
        return self._MONITORED_TOPICS


__all__ = ["MixinEmissionCounter"]
