# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""In-memory store for node event bus publish topics accumulated from introspection events.

This service accumulates the ``event_bus.publish_topics`` from every
``ModelNodeIntrospectionEvent`` processed by the registration orchestrator.
It is shared between:

    1. ``HandlerNodeIntrospected`` — updates the store on each introspection event.
    2. ``HandlerCatalogRequest`` — reads the store to build a catalog response.

Thread Safety:
    Uses ``asyncio.Lock`` for coroutine-safe reads and writes. All public methods
    are ``async`` to enforce consistent lock acquisition patterns.

Design:
    - Key: ``node_id`` (UUID as string) — one entry per registered node.
    - Value: ``frozenset[str]`` of environment-qualified topic strings.
    - Topics are stored as received from ``event_bus.publish_topics`` entries.
    - Filtering to ``onex.evt.*`` is performed at read time, not write time,
      to preserve the full set for future use cases.

Related Tickets:
    - OMN-2923: Catalog responder for topic-catalog-request.v1
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


class ServiceIntrospectionTopicStore:
    """In-memory store for node event bus publish topics.

    Accumulates publish topics from introspection events so that
    ``HandlerCatalogRequest`` can assemble a complete catalog response
    without querying Kafka or PostgreSQL.

    Attributes:
        _topics_by_node: Mapping from node_id string to frozenset of publish topics.
        _lock: asyncio.Lock for coroutine-safe access.

    Example:
        >>> store = ServiceIntrospectionTopicStore()
        >>> await store.update_node("node-id-123", ["onex.evt.platform.foo.v1"])
        >>> topics = await store.get_evt_topics()
        >>> assert "onex.evt.platform.foo.v1" in topics
    """

    def __init__(self) -> None:
        """Initialize the store with an empty topic map and an asyncio lock."""
        self._topics_by_node: dict[str, frozenset[str]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def update_node(
        self,
        node_id: str,
        publish_topics: list[str],
    ) -> None:
        """Record or update the publish topics for a node.

        Called by ``HandlerNodeIntrospected`` whenever a node introspection
        event is processed. Replaces any previously stored topics for the
        same ``node_id``.

        Args:
            node_id: Unique node identifier string (UUID as str).
            publish_topics: List of topic strings from the node's
                ``event_bus.publish_topics`` configuration. Stored as-is;
                filtering to ``onex.evt.*`` happens at read time.
        """
        async with self._lock:
            self._topics_by_node[node_id] = frozenset(publish_topics)
            logger.debug(
                "IntrospectionTopicStore updated node=%s with %d topics",
                node_id,
                len(publish_topics),
            )

    async def get_evt_topics(self) -> list[str]:
        """Return sorted deduplicated list of all ``onex.evt.*`` publish topics.

        Unions all publish topics from all registered nodes, filters to those
        starting with ``onex.evt.``, and returns a sorted deduplicated list.

        Returns:
            Sorted list of ``onex.evt.*`` topic strings. Empty list when no
            nodes have registered with event bus configuration.
        """
        async with self._lock:
            all_topics: set[str] = set()
            for topics in self._topics_by_node.values():
                all_topics.update(topics)
        return sorted(t for t in all_topics if t.startswith("onex.evt."))

    async def get_node_count(self) -> int:
        """Return the number of nodes currently tracked in the store.

        Returns:
            Count of nodes that have provided event bus configuration.
        """
        async with self._lock:
            return len(self._topics_by_node)

    async def count_nodes_missing_event_bus(self) -> int:
        """Return count of nodes with no publish topics recorded.

        Nodes that sent introspection events with no ``event_bus`` configured
        will have an empty frozenset. This count is included in catalog
        responses as the ``nodes_missing_event_bus`` field.

        Returns:
            Number of nodes with zero publish topics.
        """
        async with self._lock:
            return sum(1 for topics in self._topics_by_node.values() if not topics)

    async def snapshot(self) -> tuple[list[str], int, int]:
        """Return an atomic snapshot of catalog data for catalog responses.

        Acquires the lock once to atomically read:
        - Filtered and sorted ``onex.evt.*`` topic list
        - Total node count
        - Count of nodes missing event bus configuration

        Returns:
            Tuple of (evt_topics, node_count, nodes_missing_event_bus).
        """
        async with self._lock:
            all_topics: set[str] = set()
            total_nodes = len(self._topics_by_node)
            missing_count = 0
            for topics in self._topics_by_node.values():
                all_topics.update(topics)
                if not topics:
                    missing_count += 1

        evt_topics = sorted(t for t in all_topics if t.startswith("onex.evt."))
        return evt_topics, total_nodes, missing_count


__all__: list[str] = ["ServiceIntrospectionTopicStore"]
