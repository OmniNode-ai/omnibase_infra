# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contract closure a workflow's first request wakes (OMN-26396).

A workflow is its orchestrator plus every contract the orchestrator publishes to,
and every contract those publish to, transitively. The edges are the topics the
runtime already holds on each discovered contract: a contract's published topics
(``event_bus.publish_topics``, its terminal event and its compatibility publish
topics) lead to the contracts that subscribe to them. The graph is read from the
discovered contracts the kernel hands the boot, never from a hand-written list of
names, and it needs no checkout of any package: the static validator reads the
same facts from YAML on disk for a census, which a running runtime cannot rely on.

The discovered contract carries no composition entries: a ``workflow_coordination``
node id names a step of that workflow, not a contract, so topics are the only edges.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract

__all__: list[str] = ["LazyStartClosureGraph"]


def _published_topics(contract: ModelDiscoveredContract) -> frozenset[str]:
    topics: set[str] = set(contract.compatibility_publish_topics)
    if contract.terminal_event:
        topics.add(contract.terminal_event)
    if contract.event_bus is not None:
        topics.update(contract.event_bus.publish_topics)
        if contract.event_bus.terminal_event:
            topics.add(contract.event_bus.terminal_event)
    return frozenset(topics)


def _subscribed_topics(contract: ModelDiscoveredContract) -> tuple[str, ...]:
    return () if contract.event_bus is None else contract.event_bus.subscribe_topics


class LazyStartClosureGraph:
    """Publishers to subscribers over the discovered contracts of one boot."""

    def __init__(self, contracts: Mapping[str, ModelDiscoveredContract]) -> None:
        consumers: dict[str, set[str]] = {}
        for name, contract in contracts.items():
            for topic in _subscribed_topics(contract):
                consumers.setdefault(topic, set()).add(name)
        self._consumers = {topic: tuple(sorted(n)) for topic, n in consumers.items()}
        self._published = {
            name: _published_topics(contract) for name, contract in contracts.items()
        }

    def consumers_of(self, topic: str) -> tuple[str, ...]:
        """The contracts that subscribe to ``topic``, by name."""
        return self._consumers.get(topic, ())

    def closure_of(self, root: str) -> tuple[str, ...]:
        """``root`` and every contract reachable from it, in one stable order.

        The order is by name, so two workflows that share contracts always wake
        the shared ones in the same relative order.
        """
        reached = {root}
        queue = deque((root,))
        while queue:
            for topic in sorted(self._published.get(queue.popleft(), ())):
                for consumer in self.consumers_of(topic):
                    if consumer not in reached:
                        reached.add(consumer)
                        queue.append(consumer)
        return tuple(sorted(reached))
