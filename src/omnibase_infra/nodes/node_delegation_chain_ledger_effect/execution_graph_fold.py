# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure, pinned causal fold for the delegation execution-graph projection."""

from __future__ import annotations

from omnibase_core.models.execution_graph_replay import (
    EnumExecutionGraphAnchorKind,
    EnumExecutionGraphAnchorState,
    EnumExecutionGraphEdgeKind,
    EnumExecutionGraphEndpointKind,
    EnumExecutionGraphNodeKind,
    EnumExecutionGraphUnresolvedReason,
    ModelExecutionGraph,
    ModelExecutionGraphAnchor,
    ModelExecutionGraphAnnotations,
    ModelExecutionGraphEdge,
    ModelExecutionGraphEndpoint,
    ModelExecutionGraphLabel,
    ModelExecutionGraphNode,
    ModelExecutionGraphReplay,
    ModelExecutionGraphReplayPolicy,
    ModelExecutionGraphSourceRef,
    ModelExecutionGraphUnresolved,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.chain_replay import (
    assemble_replay_and_verify,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_execution_graph_fold_request import (
    ModelExecutionGraphFoldRequest,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_observed_hop import (
    ModelObservedHop,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.replay_evidence import (
    normalize_and_topologically_order,
)


class DelegationExecutionGraphFold:
    """Definition-B fold: no bus, clock, database, or mutable contract lookup."""

    def handle(self, request: ModelExecutionGraphFoldRequest) -> ModelExecutionGraph:
        """Grade the recorded parent tree in deterministic causal order."""
        normalized = normalize_and_topologically_order(
            request.bounded_evidence, request.topology
        )
        evidence_by_id = {item.envelope_id: item for item in normalized.envelopes}
        if any(
            item.correlation_id != request.correlation_id
            for item in normalized.envelopes
        ):
            raise ValueError("bounded evidence has a foreign correlation")

        hop_topics = {
            topic for hop in request.topology.declared_chain for topic in hop.topics
        }
        reroute_topics = {
            topic
            for hop in request.topology.declared_chain
            for topic in hop.reroute_parents
        } - hop_topics
        present_reroute_topics = {
            item.topic for item in normalized.envelopes
        } & reroute_topics
        if present_reroute_topics:
            raise ValueError(
                "re-route display is deferred; evidence is refused in the "
                f"chains-only slice: {sorted(present_reroute_topics)!r}"
            )
        unknown_topics = (
            {item.topic for item in normalized.envelopes} - hop_topics - reroute_topics
        )
        if unknown_topics:
            raise ValueError(
                f"bounded evidence has undeclared topics {unknown_topics!r}"
            )

        ordered = tuple(
            evidence_by_id[envelope_id] for envelope_id in normalized.topological_order
        )
        observed = tuple(
            ModelObservedHop(
                topic=item.topic,
                envelope_id=item.envelope_id,
                parent_envelope_id=item.parent_envelope_id,
                correlation_id=item.correlation_id,
            )
            for item in ordered
        )
        grades = {
            row.envelope_id: row
            for row in assemble_replay_and_verify(
                request.correlation_id, observed, request.topology.declared_chain
            )
        }

        source_refs = {
            item.envelope_id: ModelExecutionGraphSourceRef(
                topic=item.topic,
                partition=item.partition,
                kafka_offset=item.kafka_offset,
            )
            for item in ordered
        }
        nodes = tuple(
            ModelExecutionGraphNode(
                id=item.envelope_id,
                kind=EnumExecutionGraphNodeKind.HOP,
                topic=item.topic,
                partition=item.partition,
                kafka_offset=item.kafka_offset,
                parent_envelope_id=item.parent_envelope_id,
                replay_green=(
                    grades[item.envelope_id].replay_green
                    if item.envelope_id in grades
                    else None
                ),
                verifier_verdict=(
                    grades[item.envelope_id].verifier_verdict.value
                    if item.envelope_id in grades
                    else None
                ),
                source_ref=source_refs[item.envelope_id],
            )
            for item in ordered
        )
        edges = tuple(
            ModelExecutionGraphEdge(
                id=f"parent:{item.envelope_id}",
                from_id=ModelExecutionGraphEndpoint(
                    kind=EnumExecutionGraphEndpointKind.NODE,
                    node_id=item.parent_envelope_id,
                ),
                to_id=ModelExecutionGraphEndpoint(
                    kind=EnumExecutionGraphEndpointKind.NODE,
                    node_id=item.envelope_id,
                ),
                kind=EnumExecutionGraphEdgeKind.CAUSED,
                evidence_ref=source_refs[item.envelope_id],
            )
            for item in ordered
            if item.parent_envelope_id in evidence_by_id
        )
        unresolved = tuple(
            ModelExecutionGraphUnresolved(
                subject_id=item.child_envelope_id,
                reason=EnumExecutionGraphUnresolvedReason.MISSING_PARENT,
                source_ref=source_refs[item.child_envelope_id],
            )
            for item in normalized.unresolved
        )
        replay = ModelExecutionGraphReplay(
            fold_version=request.fold_version,
            topology_version=request.topology.version,
            grader_version=request.grader_version,
            verdict_reducer_version=request.verdict_reducer_version,
            policy=ModelExecutionGraphReplayPolicy(
                traversal="parent_topological",
                node_identity="envelope_id",
                redelivery="same_id_exact_redelivery_lowest_source_position",
                conflicting_identity="same_id_conflicting_parent_or_semantic_body_refuse",
                cross_topic_duplicate="same_id_across_topics_refuse",
            ),
            source_cursors=tuple(
                sorted(
                    request.source_cursors,
                    key=lambda cursor: (cursor.topic, cursor.partition),
                )
            ),
            correlation_id=request.correlation_id,
            anchor=ModelExecutionGraphAnchor(
                kind=EnumExecutionGraphAnchorKind.NONE,
                state=EnumExecutionGraphAnchorState.UNRESOLVED,
            ),
            nodes=nodes,
            edges=edges,
            order=normalized.topological_order,
            verdicts=(),
            unresolved=unresolved,
            withheld_count=request.withheld_count,
        )
        return ModelExecutionGraph(
            schema_version=1,
            replay=replay,
            labels=tuple(
                ModelExecutionGraphLabel(
                    node_id=item.envelope_id,
                    event_timestamp=item.event_timestamp,
                    ledger_written_at=item.ledger_written_at,
                )
                for item in ordered
            ),
            annotations=ModelExecutionGraphAnnotations(
                read_at=request.read_at,
                authorization_tenant_id=request.tenant_id,
                authorization_ownership_source="delegation_events",
                authorization_checked_over="full_correlation",
                stored_chain=request.stored_chain,
                stored_verdicts=request.stored_verdicts,
            ),
        )


__all__ = ["DelegationExecutionGraphFold", "ModelExecutionGraphFoldRequest"]
