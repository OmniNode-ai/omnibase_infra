# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure evidence normalization and causal ordering for delegation replay."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from heapq import heappop, heappush
from uuid import UUID

from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_envelope_delivery_count import (
    ModelEnvelopeDeliveryCount,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_normalized_replay_evidence import (
    ModelNormalizedReplayEvidence,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_observed_envelope_evidence import (
    ModelObservedEnvelopeEvidence,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_pinned_chain_topology import (
    ModelPinnedChainTopology,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_unresolved_parent import (
    ModelUnresolvedParent,
)


def _topology_positions(topology: ModelPinnedChainTopology) -> dict[str, int]:
    return {
        topic: index for index, hop in enumerate(topology.hops) for topic in hop.topics
    }


def _causal_sort_key(
    evidence: ModelObservedEnvelopeEvidence,
    positions: dict[str, int],
    unknown_position: int,
) -> tuple[int, str, int, int, str]:
    """Pinned topology first, then durable Kafka position; never a timestamp."""
    return (
        positions.get(evidence.topic, unknown_position),
        evidence.topic,
        evidence.partition,
        evidence.kafka_offset,
        str(evidence.envelope_id),
    )


def normalize_and_topologically_order(
    evidence: Iterable[ModelObservedEnvelopeEvidence],
    topology: ModelPinnedChainTopology,
) -> ModelNormalizedReplayEvidence:
    """Collapse exact redelivery while retaining source and causal order.

    ``observed_order`` preserves raw source order. ``topological_order`` is a
    separate causal traversal tied by pinned declared-hop position and Kafka
    ``(topic, partition, offset)`` coordinates. A missing parent is recorded as
    unresolved and its child is ordered as a root so no recorded evidence is
    discarded; only identity collisions and cycles refuse the whole fold.
    """
    ordered = tuple(sorted(evidence, key=lambda item: item.observed_index))
    if len({item.observed_index for item in ordered}) != len(ordered):
        raise ValueError("evidence contains duplicate observed indexes")

    first_by_id: dict[UUID, ModelObservedEnvelopeEvidence] = {}
    delivery_count: dict[UUID, int] = {}
    for item in ordered:
        first = first_by_id.get(item.envelope_id)
        if first is None:
            first_by_id[item.envelope_id] = item
            delivery_count[item.envelope_id] = 1
            continue
        if first.identity_claim != item.identity_claim:
            raise ValueError(
                f"envelope identity collision for {item.envelope_id}: "
                "same envelope id carries conflicting evidence"
            )
        if (item.topic, item.partition, item.kafka_offset) < (
            first.topic,
            first.partition,
            first.kafka_offset,
        ):
            first_by_id[item.envelope_id] = item
        delivery_count[item.envelope_id] += 1

    collapsed = tuple(first_by_id.values())
    by_id = {item.envelope_id: item for item in collapsed}
    children: dict[UUID, list[UUID]] = defaultdict(list)
    indegree = {item.envelope_id: 0 for item in collapsed}
    unresolved_direct: list[ModelUnresolvedParent] = []
    for item in collapsed:
        parent = item.parent_envelope_id
        if parent is None:
            continue
        if parent == item.envelope_id:
            raise ValueError(f"envelope {item.envelope_id} declares itself as parent")
        if parent not in by_id:
            unresolved_direct.append(
                ModelUnresolvedParent(
                    child_envelope_id=item.envelope_id,
                    parent_envelope_id=parent,
                )
            )
            continue
        children[parent].append(item.envelope_id)
        indegree[item.envelope_id] += 1

    positions = _topology_positions(topology)
    unknown_position = len(topology.hops)

    def sort_key(envelope_id: UUID) -> tuple[int, str, int, int, str]:
        return _causal_sort_key(by_id[envelope_id], positions, unknown_position)

    cycle_indegree = dict(indegree)
    cycle_ready: list[tuple[tuple[int, str, int, int, str], UUID]] = []
    for envelope_id, degree in cycle_indegree.items():
        if degree == 0:
            heappush(cycle_ready, (sort_key(envelope_id), envelope_id))
    seen = 0
    while cycle_ready:
        _key, current = heappop(cycle_ready)
        seen += 1
        for child in children[current]:
            cycle_indegree[child] -= 1
            if cycle_indegree[child] == 0:
                heappush(cycle_ready, (sort_key(child), child))
    if seen != len(collapsed):
        raise ValueError("evidence parent relation contains a cycle")

    ready: list[tuple[tuple[int, str, int, int, str], UUID]] = []
    for envelope_id, degree in indegree.items():
        if degree == 0:
            heappush(ready, (sort_key(envelope_id), envelope_id))
    topological: list[UUID] = []
    while ready:
        _key, current = heappop(ready)
        topological.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heappush(ready, (sort_key(child), child))

    return ModelNormalizedReplayEvidence(
        envelopes=collapsed,
        observed_order=tuple(item.envelope_id for item in collapsed),
        topological_order=tuple(topological),
        delivery_counts=tuple(
            ModelEnvelopeDeliveryCount(
                envelope_id=item.envelope_id,
                delivery_count=delivery_count[item.envelope_id],
            )
            for item in collapsed
        ),
        unresolved=tuple(
            sorted(unresolved_direct, key=lambda item: sort_key(item.child_envelope_id))
        ),
    )


__all__ = ["normalize_and_topologically_order"]
