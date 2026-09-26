# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The graph fold grades only bounded recorded evidence, never stored rows."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest

from omnibase_core.models.execution_graph_replay import (
    EnumExecutionGraphEdgeKind,
    EnumExecutionGraphNodeKind,
    ModelExecutionGraph,
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
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
    PinnedExecutionGraphTopology,
    _resolve_payload,
    build_snapshot_from_chain_contract,
)

CORRELATION = UUID("11111111-2222-3333-4444-555555555555")
TENANT = UUID("22222222-3333-4444-5555-666666666666")
HEAD = UUID("aaaaaaaa-0000-0000-0000-000000000000")
LEFT = UUID("aaaaaaaa-0000-0000-0000-000000000001")
RIGHT = UUID("aaaaaaaa-0000-0000-0000-000000000002")


def _version() -> ModelSemVer:
    return ModelSemVer(major=1, minor=0, patch=0)


def _topology(*hops: ModelDeclaredChainHop) -> PinnedExecutionGraphTopology:
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract_version": {"major": 1, "minor": 0, "patch": 0},
        "source_contract_sha256": "a" * 64,
        "chain_topology": [hop.model_dump(mode="json") for hop in hops],
        "verdict_topic": "verdict",
    }
    payload["topology_sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _resolve_payload(payload)


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
    topology = _topology(
        ModelDeclaredChainHop(topic="head", parent=None),
        ModelDeclaredChainHop(topic="left", parent="head"),
        ModelDeclaredChainHop(topic="right", parent="head"),
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
    topology = _topology(
        ModelDeclaredChainHop(topic="head", parent=None),
        ModelDeclaredChainHop(
            topic="routing",
            parent="head",
            reroute_parents=("quality-gate",),
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


def _real_branch_graph() -> ModelExecutionGraph:
    repo_root = Path(__file__).resolve().parents[4]
    fixture = json.loads(
        (repo_root / "tests/fixtures/omn19726/real_branch_legacy.json").read_text(
            encoding="utf-8"
        )
    )
    candidate = build_snapshot_from_chain_contract(
        repo_root
        / "src/omnibase_infra/nodes/node_delegation_chain_ledger_effect/contract.yaml",
        "onex.evt.omnimarket.dod-verify-completed.v1",
    )
    topology = PackagedExecutionGraphTopologyContract().resolve(candidate.version)
    records = tuple(
        record for record in fixture["records"] if record["stored_replay_green"] is True
    )
    correlation_id = UUID(fixture["correlation_id"])
    evidence = tuple(
        ModelObservedEnvelopeEvidence(
            envelope_id=UUID(record["envelope_id"]),
            topic=record["topic"],
            parent_envelope_id=(
                UUID(record["parent_envelope_id"])
                if record["parent_envelope_id"] is not None
                else None
            ),
            correlation_id=correlation_id,
            evidence_fingerprint=f"fixture-topic:{record['topic']}",
            observed_index=index,
            partition=record["partition"],
            kafka_offset=record["kafka_offset"],
        )
        for index, record in enumerate(records)
    )
    request = ModelExecutionGraphFoldRequest(
        correlation_id=correlation_id,
        tenant_id=UUID(fixture["owner_tenant_id"]),
        bounded_evidence=evidence,
        topology=topology,
        fold_version=_version(),
        grader_version=_version(),
        verdict_reducer_version=_version(),
        source_cursors=tuple(
            ModelExecutionGraphSourceCursor(
                topic=item.topic,
                partition=item.partition,
                max_kafka_offset=item.kafka_offset,
            )
            for item in evidence
        ),
        read_at=datetime(2026, 9, 26, tzinfo=UTC),
    )
    return DelegationExecutionGraphFold().handle(request)


def test_real_branch_five_hops_use_packaged_topology_stamp() -> None:
    graph = _real_branch_graph()

    assert len(graph.replay.nodes) == 5
    assert len(graph.replay.edges) == 4
    assert graph.replay.topology_version.topology_sha256 == (
        "0505ab0b163492380739a15646c0442a3ecfb54efb0acb53bc23d232640fbfd3"
    )
    assert all(node.replay_green for node in graph.replay.nodes)
    assert graph.replay.unresolved == ()
