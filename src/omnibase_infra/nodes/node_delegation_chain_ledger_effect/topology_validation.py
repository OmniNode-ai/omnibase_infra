# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validation and canonicalization for pinned delegation topology (OMN-19729)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_declared_chain_hop import (
    ModelDeclaredChainHop,
)


def validate_declared_topology(hops: Sequence[ModelDeclaredChainHop]) -> None:
    """Refuse ambiguous, cyclic, or disconnected declared topology."""
    if not hops:
        raise ValueError("pinned topology must declare at least one hop")
    aliases: dict[str, str] = {}
    by_topic: dict[str, ModelDeclaredChainHop] = {}
    for hop in hops:
        by_topic[hop.topic] = hop
        for name in hop.topics:
            if name in aliases:
                raise ValueError(f"topology names topic {name!r} more than once")
            aliases[name] = hop.topic
    parent_by_topic: dict[str, str | None] = {}
    for hop in hops:
        if hop.parent is None:
            parent_by_topic[hop.topic] = None
            continue
        parent = aliases.get(hop.parent)
        if parent is None:
            raise ValueError(
                f"topology parent {hop.parent!r} of {hop.topic!r} is undeclared"
            )
        parent_by_topic[hop.topic] = parent
    heads = [topic for topic, parent in parent_by_topic.items() if parent is None]
    if len(heads) != 1:
        raise ValueError(f"topology must declare exactly one head, got {heads!r}")
    visited: set[str] = set()
    active: set[str] = set()

    def visit(topic: str) -> None:
        if topic in active:
            raise ValueError(f"topology contains a parent cycle at {topic!r}")
        if topic in visited:
            return
        active.add(topic)
        parent = parent_by_topic[topic]
        if parent is not None:
            visit(parent)
        active.remove(topic)
        visited.add(topic)

    for topic in by_topic:
        visit(topic)
    children: dict[str, list[str]] = defaultdict(list)
    for topic, parent in parent_by_topic.items():
        if parent is not None:
            children[parent].append(topic)
    reachable: set[str] = set()
    pending = [heads[0]]
    while pending:
        topic = pending.pop()
        if topic in reachable:
            continue
        reachable.add(topic)
        pending.extend(children[topic])
    if len(reachable) != len(by_topic):
        missing = sorted(set(by_topic) - reachable)
        raise ValueError(f"topology contains unreachable hop(s) {missing!r}")


__all__ = ["validate_declared_topology"]
