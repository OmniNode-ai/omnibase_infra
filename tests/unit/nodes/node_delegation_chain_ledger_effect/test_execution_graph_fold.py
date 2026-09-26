# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The graph fold grades only bounded recorded evidence, never stored rows."""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from omnibase_core.models.execution_graph_replay import (
    EnumExecutionGraphEdgeKind,
    EnumExecutionGraphNodeKind,
    ModelExecutionGraphSourceCursor,
    ModelExecutionGraphStoredChainAnnotation,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.execution_graph_fold import (
    DelegationExecutionGraphFold,
    ModelExecutionGraphFoldRequest,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_declared_chain_hop import (
    ModelDeclaredChainHop,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_observed_envelope_evidence import (
    ModelObservedEnvelopeEvidence,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_pinned_chain_topology import (
    ModelPinnedChainTopology,
)

CORRELATION = UUID("11111111-2222-3333-4444-555555555555")
TENANT = UUID("22222222-3333-4444-5555-666666666666")
HEAD = UUID("aaaaaaaa-0000-0000-0000-000000000000")
LEFT = UUID("aaaaaaaa-0000-0000-0000-000000000001")
RIGHT = UUID("aaaaaaaa-0000-0000-0000-000000000002")


def _version() -> ModelSemVer:
    return ModelSemVer(major=1, minor=0, patch=0)


def _record(
    envelope_id: UUID,
    topic: str,
    parent: UUID | None,
    index: int,
    *,
    event_timestamp: datetime | None = None,
) -> ModelObservedEnvelopeEvidence:
    return ModelObservedEnvelopeEvidence(
        envelope_id=envelope_id,
        topic=topic,
        parent_envelope_id=parent,
        correlation_id=CORRELATION,
        evidence_fingerprint=f"record-{index}",
        observed_index=index,
        partition=0,
        kafka_offset=index,
        event_timestamp=event_timestamp,
    )


def _request(
    *,
    stored_chain: tuple[ModelExecutionGraphStoredChainAnnotation, ...] = (),
    read_at: datetime | None = None,
) -> ModelExecutionGraphFoldRequest:
    topology = ModelPinnedChainTopology.create(
        topology_contract_ref="node:branch@1.0.0",
        replay_algorithm_version="1",
        hops=(
            ModelDeclaredChainHop(topic="head", parent=None),
            ModelDeclaredChainHop(topic="left", parent="head"),
            ModelDeclaredChainHop(topic="right", parent="head"),
        ),
    )
    return ModelExecutionGraphFoldRequest(
        correlation_id=CORRELATION,
        tenant_id=TENANT,
        bounded_evidence=(
            _record(LEFT, "left", HEAD, 0),
            _record(RIGHT, "right", HEAD, 1),
            _record(HEAD, "head", None, 2),
        ),
        topology=topology,
        topology_contract_version=_version(),
        fold_version=_version(),
        grader_version=_version(),
        verdict_reducer_version=_version(),
        source_cursors=tuple(
            ModelExecutionGraphSourceCursor(
                topic=topic, partition=0, max_kafka_offset=3
            )
            for topic in ("head", "left", "right")
        ),
        read_at=read_at or datetime(2026, 9, 26, tzinfo=UTC),
        stored_chain=stored_chain,
    )


def test_fold_uses_recorded_branch_and_pure_recomputed_grades() -> None:
    graph = DelegationExecutionGraphFold().handle(_request())

    assert graph.schema_version == 1
    assert graph.replay.order == (HEAD, LEFT, RIGHT)
    assert {node.id for node in graph.replay.nodes} == {HEAD, LEFT, RIGHT}
    assert all(
        node.kind is EnumExecutionGraphNodeKind.HOP for node in graph.replay.nodes
    )
    assert all(node.replay_green for node in graph.replay.nodes)
    assert {
        (edge.from_id.node_id, edge.to_id.node_id) for edge in graph.replay.edges
    } == {
        (HEAD, LEFT),
        (HEAD, RIGHT),
    }
    assert all(
        edge.kind is EnumExecutionGraphEdgeKind.CAUSED for edge in graph.replay.edges
    )


def test_stored_rewrite_changes_annotations_but_not_replay() -> None:
    handler = DelegationExecutionGraphFold()
    first = handler.handle(_request())
    changed = handler.handle(
        _request(
            stored_chain=(
                ModelExecutionGraphStoredChainAnnotation(
                    node_id=LEFT,
                    hop_index=9,
                    replay_green=False,
                    verifier_verdict="fail",
                ),
            ),
            read_at=datetime(2026, 9, 27, tzinfo=UTC),
        )
    )

    assert first.replay.model_dump(mode="json") == changed.replay.model_dump(
        mode="json"
    )
    assert first.annotations != changed.annotations


def test_timestamp_change_affects_only_labels() -> None:
    first_request = _request()
    changed_request = first_request.model_copy(
        update={
            "bounded_evidence": tuple(
                item.model_copy(
                    update={"event_timestamp": datetime(2040, 1, 1, tzinfo=UTC)}
                )
                for item in first_request.bounded_evidence
            )
        }
    )
    handler = DelegationExecutionGraphFold()
    first = handler.handle(first_request)
    changed = handler.handle(changed_request)

    assert first.replay.model_dump(mode="json") == changed.replay.model_dump(
        mode="json"
    )
    assert first.labels != changed.labels


def test_cursor_input_order_does_not_change_replay_bytes() -> None:
    request = _request()
    reversed_bounds = request.model_copy(
        update={"source_cursors": tuple(reversed(request.source_cursors))}
    )
    handler = DelegationExecutionGraphFold()

    assert handler.handle(request).replay.model_dump(mode="json") == handler.handle(
        reversed_bounds
    ).replay.model_dump(mode="json")


def test_reroute_evidence_is_refused_in_the_chains_only_slice() -> None:
    route_first = LEFT
    quality_gate = RIGHT
    route_repeat = UUID("aaaaaaaa-0000-0000-0000-000000000003")
    topology = ModelPinnedChainTopology.create(
        topology_contract_ref="node:reroute@1.0.0",
        replay_algorithm_version="1",
        hops=(
            ModelDeclaredChainHop(topic="head", parent=None),
            ModelDeclaredChainHop(
                topic="routing",
                parent="head",
                reroute_parents=("quality-gate",),
            ),
        ),
    )
    request = _request().model_copy(
        update={
            "topology": topology,
            "bounded_evidence": (
                _record(HEAD, "head", None, 0),
                _record(route_first, "routing", HEAD, 1),
                _record(quality_gate, "quality-gate", route_first, 2),
                _record(route_repeat, "routing", quality_gate, 3),
            ),
        }
    )

    with pytest.raises(ValueError, match="re-route display is deferred"):
        DelegationExecutionGraphFold().handle(request)
