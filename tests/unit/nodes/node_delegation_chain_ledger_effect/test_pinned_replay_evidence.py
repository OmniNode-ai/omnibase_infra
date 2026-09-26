# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure Phase-2 evidence and topology seams (OMN-19729).

No database, envelope, or transport type participates in these proofs.  The
inputs are already-recorded primitive evidence and an immutable topology
snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_declared_chain_hop import (
    ModelDeclaredChainHop,
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
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.replay_evidence import (
    normalize_and_topologically_order,
)

CORRELATION = UUID("11111111-2222-3333-4444-555555555555")
HEAD = UUID("aaaaaaaa-0000-0000-0000-000000000000")
CHILD = UUID("aaaaaaaa-0000-0000-0000-000000000001")
SIBLING = UUID("aaaaaaaa-0000-0000-0000-000000000002")


def _evidence(
    envelope_id: UUID,
    topic: str,
    parent: UUID | None,
    observed_index: int,
    *,
    fingerprint: str = "a" * 64,
    partition: int = 0,
    kafka_offset: int | None = None,
) -> ModelObservedEnvelopeEvidence:
    return ModelObservedEnvelopeEvidence(
        envelope_id=envelope_id,
        topic=topic,
        parent_envelope_id=parent,
        correlation_id=CORRELATION,
        evidence_fingerprint=fingerprint,
        observed_index=observed_index,
        partition=partition,
        kafka_offset=observed_index if kafka_offset is None else kafka_offset,
    )


def _topology(*hops: ModelDeclaredChainHop) -> ModelPinnedChainTopology:
    return ModelPinnedChainTopology.create(
        topology_contract_ref="node:delegation-chain@1.0.0",
        replay_algorithm_version="1",
        hops=hops,
    )


def test_pinned_snapshot_is_canonical_and_rejects_disconnected_cycle() -> None:
    valid = (
        ModelDeclaredChainHop(topic="head", parent=None),
        ModelDeclaredChainHop(topic="child", parent="head"),
    )
    first = _topology(*valid)
    second = _topology(*valid)

    assert first.topology_sha256 == second.topology_sha256
    assert ModelPinnedChainTopology.model_validate(first.model_dump()) == first
    with pytest.raises(ValueError, match="schema_version"):
        ModelPinnedChainTopology.model_validate(
            {**first.model_dump(mode="json"), "schema_version": 2}
        )

    disconnected_cycle = (
        ModelDeclaredChainHop(topic="head", parent=None),
        ModelDeclaredChainHop(topic="left", parent="right"),
        ModelDeclaredChainHop(topic="right", parent="left"),
    )
    with pytest.raises(ValueError, match=r"unreachable|cycle"):
        _topology(*disconnected_cycle)


def test_exact_redelivery_collapses_but_conflicting_same_id_refuses() -> None:
    head = _evidence(HEAD, "head", None, 0)
    redelivery = _evidence(HEAD, "head", None, 3)
    normalized = normalize_and_topologically_order(
        (head, redelivery), _topology(ModelDeclaredChainHop(topic="head", parent=None))
    )

    assert normalized.observed_order == (HEAD,)
    assert normalized.topological_order == (HEAD,)
    assert normalized.delivery_count_by_envelope_id[HEAD] == 2

    earlier_offset_seen_later = _evidence(HEAD, "head", None, 4, kafka_offset=1)
    earliest = normalize_and_topologically_order(
        (head, earlier_offset_seen_later),
        _topology(ModelDeclaredChainHop(topic="head", parent=None)),
    )
    assert earliest.envelopes[0].kafka_offset == 0

    later_offset_seen_first = _evidence(HEAD, "head", None, 0, kafka_offset=9)
    earlier_offset_seen_second = _evidence(HEAD, "head", None, 1, kafka_offset=2)
    by_offset = normalize_and_topologically_order(
        (later_offset_seen_first, earlier_offset_seen_second),
        _topology(ModelDeclaredChainHop(topic="head", parent=None)),
    )
    assert by_offset.envelopes[0].kafka_offset == 2

    conflicting_parent = _evidence(HEAD, "head", None, 3, fingerprint="b" * 64)
    with pytest.raises(ValueError, match="identity collision"):
        normalize_and_topologically_order(
            (head, conflicting_parent),
            _topology(ModelDeclaredChainHop(topic="head", parent=None)),
        )


def test_topological_order_is_deterministic_and_preserves_observed_order() -> None:
    child = _evidence(CHILD, "child", HEAD, 0)
    sibling = _evidence(SIBLING, "sibling", HEAD, 1)
    head = _evidence(HEAD, "head", None, 2)

    topology = _topology(
        ModelDeclaredChainHop(topic="head", parent=None),
        ModelDeclaredChainHop(topic="child", parent="head"),
        ModelDeclaredChainHop(topic="sibling", parent="head"),
    )
    normalized = normalize_and_topologically_order((child, sibling, head), topology)

    assert normalized.observed_order == (CHILD, SIBLING, HEAD)
    assert normalized.topological_order == (HEAD, CHILD, SIBLING)


def test_missing_parent_is_unresolved_not_a_global_refusal() -> None:
    grandchild = _evidence(SIBLING, "grandchild", CHILD, 1)
    normalized = normalize_and_topologically_order(
        (_evidence(CHILD, "child", HEAD, 0), grandchild),
        _topology(
            ModelDeclaredChainHop(topic="head", parent=None),
            ModelDeclaredChainHop(topic="child", parent="head"),
            ModelDeclaredChainHop(topic="grandchild", parent="child"),
        ),
    )

    assert normalized.observed_order == (CHILD, SIBLING)
    assert normalized.topological_order == (CHILD, SIBLING)
    assert set(normalized.topological_order) == set(normalized.observed_order)
    assert normalized.unresolved == (
        ModelUnresolvedParent(child_envelope_id=CHILD, parent_envelope_id=HEAD),
    )


def _fixture_records(path_name: str) -> tuple[dict[str, object], ...]:
    fixture_path = Path(__file__).parents[3] / "fixtures" / "omn19726" / path_name
    return tuple(json.loads(fixture_path.read_text())["records"])


def _fixture_evidence(path_name: str) -> tuple[ModelObservedEnvelopeEvidence, ...]:
    records = _fixture_records(path_name)
    return tuple(
        _evidence(
            UUID(str(record["envelope_id"])),
            str(record["topic"]),
            (
                UUID(str(record["parent_envelope_id"]))
                if record["parent_envelope_id"] is not None
                else None
            ),
            observed_index=index,
            fingerprint=f"fixture-topic:{record['topic']}",
            partition=int(record["partition"]),
            kafka_offset=int(record["kafka_offset"]),
        )
        for index, record in enumerate(records)
    )


def _fixture_topology(
    evidence: tuple[ModelObservedEnvelopeEvidence, ...],
) -> ModelPinnedChainTopology:
    topics = tuple(dict.fromkeys(item.topic for item in evidence))
    return _topology(
        ModelDeclaredChainHop(topic=topics[0], parent=None),
        *(ModelDeclaredChainHop(topic=topic, parent=topics[0]) for topic in topics[1:]),
    )


def test_clean_lab_fixture_keeps_all_evidence_and_marks_two_unresolved() -> None:
    fixture_name = "real_branch_legacy.json"
    records = _fixture_records(fixture_name)
    evidence = _fixture_evidence(fixture_name)
    normalized = normalize_and_topologically_order(
        evidence, _fixture_topology(evidence)
    )

    envelope_ids = {str(record["envelope_id"]) for record in records}
    graded = tuple(
        record for record in records if record["stored_replay_green"] is True
    )

    assert len(graded) == 5
    assert all(
        record["parent_envelope_id"] is None
        or str(record["parent_envelope_id"]) in envelope_ids
        for record in graded
    )
    assert len(normalized.unresolved) == 2
    assert set(normalized.topological_order) == {item.envelope_id for item in evidence}


def test_conflicting_reroute_lab_fixture_refuses_reused_envelope_id() -> None:
    evidence = _fixture_evidence("real_reroute_conflict_legacy.json")

    with pytest.raises(ValueError, match="identity collision"):
        normalize_and_topologically_order(evidence, _fixture_topology(evidence))


def test_topological_ties_use_pinned_hop_then_kafka_position_not_input_order() -> None:
    topology = _topology(
        ModelDeclaredChainHop(topic="head", parent=None),
        ModelDeclaredChainHop(topic="first", parent="head"),
        ModelDeclaredChainHop(topic="second", parent="head"),
    )
    first = _evidence(CHILD, "first", HEAD, 2, partition=1, kafka_offset=9)
    second = _evidence(SIBLING, "second", HEAD, 1, partition=0, kafka_offset=1)
    head = _evidence(HEAD, "head", None, 0, partition=0, kafka_offset=0)

    from_source_order = normalize_and_topologically_order(
        (second, head, first), topology
    )
    from_permuted_input = normalize_and_topologically_order(
        (first, second, head), topology
    )

    assert from_source_order.observed_order == (HEAD, SIBLING, CHILD)
    assert from_permuted_input.observed_order == from_source_order.observed_order
    assert from_permuted_input.topological_order == (HEAD, CHILD, SIBLING)


def test_same_declared_hop_branches_use_kafka_position_not_observed_order() -> None:
    topology = _topology(
        ModelDeclaredChainHop(topic="head", parent=None),
        ModelDeclaredChainHop(topic="retry", parent="head"),
    )
    first_retry = _evidence(CHILD, "retry", HEAD, 2, partition=1, kafka_offset=9)
    second_retry = _evidence(SIBLING, "retry", HEAD, 1, partition=0, kafka_offset=1)
    head = _evidence(HEAD, "head", None, 0, partition=0, kafka_offset=0)

    normalized = normalize_and_topologically_order(
        (first_retry, head, second_retry), topology
    )

    assert normalized.observed_order == (HEAD, SIBLING, CHILD)
    assert normalized.topological_order == (HEAD, SIBLING, CHILD)


def test_parent_cycle_refuses() -> None:
    topology = _topology(
        ModelDeclaredChainHop(topic="head", parent=None),
        ModelDeclaredChainHop(topic="child", parent="head"),
    )
    with pytest.raises(ValueError, match="cycle"):
        normalize_and_topologically_order(
            (
                _evidence(HEAD, "head", CHILD, 0),
                _evidence(CHILD, "child", HEAD, 1),
            ),
            topology,
        )
