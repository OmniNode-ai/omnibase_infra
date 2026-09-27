# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offset-bounded adapter from admitted ledger structure to the pure chain fold.

The selected rows and returned cursors are deterministic for a captured read.
Kafka offsets are not writer watermarks: an earlier offset may arrive later,
so append invariance is pending the separately ticketed watermark contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from omnibase_core.models.execution_graph_replay import (
    EnumExecutionGraphCursorMode,
    ModelExecutionGraphRequest,
    ModelExecutionGraphSourceCursor,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_stored_chain_annotation import (
    ModelExecutionGraphStoredChainAnnotation,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_refusal import (
    ModelExecutionGraphTerminalRefusal,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_result import (
    ModelExecutionGraphTerminalResult,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.execution_graph_fold import (
    DelegationExecutionGraphFold,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_execution_graph_fold_request import (
    ModelExecutionGraphFoldRequest,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_observed_envelope_evidence import (
    ModelObservedEnvelopeEvidence,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.replay_evidence import (
    select_bounded_evidence,
)
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    ExecutionGraphLedgerRecord,
)
from omnibase_infra.runtime.execution_graph_ownership import (
    ExecutionGraphOwnershipAdmission,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    VerifiedExecutionGraphReadAuthority,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PinnedExecutionGraphTopology,
)

_FOLD_VERSION = ModelSemVer(major=1, minor=0, patch=0)
_GRADER_VERSION = ModelSemVer(major=1, minor=0, patch=0)
_VERDICT_REDUCER_VERSION = ModelSemVer(major=1, minor=0, patch=0)


def _recorded_parent(row: ExecutionGraphLedgerRecord) -> UUID | None:
    try:
        headers: object = json.loads(row.onex_headers)
    except ValueError as exc:
        raise ValueError("invalid admitted envelope headers") from exc
    if not isinstance(headers, dict):
        raise ValueError("invalid admitted envelope headers")
    raw = headers.get("parent_message_id")
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or raw != raw.strip():
        raise ValueError("invalid admitted envelope parent")
    try:
        parent = UUID(raw)
    except ValueError as exc:
        raise ValueError("invalid admitted envelope parent") from exc
    if parent.int == 0 or str(parent) != raw:
        raise ValueError("invalid admitted envelope parent")
    return parent


def _admitted_structure(
    admission: ExecutionGraphOwnershipAdmission,
    topology: PinnedExecutionGraphTopology,
) -> tuple[ModelObservedEnvelopeEvidence, ...]:
    chain_topics = {topic for hop in topology.declared_chain for topic in hop.topics}
    reroute_topics = {
        topic for hop in topology.declared_chain for topic in hop.reroute_parents
    }
    selected = tuple(
        sorted(
            (
                row
                for row in admission.owned_rows
                if row.topic in chain_topics or row.topic in reroute_topics
            ),
            key=lambda row: (
                row.topic,
                row.partition,
                row.kafka_offset,
                str(row.ledger_entry_id),
            ),
        )
    )
    if not selected:
        raise ValueError("no admitted chain evidence")
    evidence: list[ModelObservedEnvelopeEvidence] = []
    for index, row in enumerate(selected):
        if row.envelope_id is None:
            raise ValueError("admitted envelope has no identity")
        evidence.append(
            ModelObservedEnvelopeEvidence(
                envelope_id=row.envelope_id,
                topic=row.topic,
                parent_envelope_id=_recorded_parent(row),
                correlation_id=row.correlation_id,
                evidence_fingerprint=hashlib.sha256(row.event_value).hexdigest(),
                observed_index=index,
                partition=row.partition,
                kafka_offset=row.kafka_offset,
                event_timestamp=row.event_timestamp,
                ledger_written_at=row.ledger_written_at,
            )
        )
    return tuple(evidence)


class ExecutionGraphReadFold:
    """Callable executor adapter over one admitted current evidence snapshot."""

    def __init__(
        self, *, workflow_type: str, read_clock: Callable[[], datetime]
    ) -> None:
        if not workflow_type or workflow_type != workflow_type.strip():
            raise ValueError("graph workflow_type must be canonical")
        self._workflow_type = workflow_type
        self._read_clock = read_clock
        self._fold = DelegationExecutionGraphFold()

    async def __call__(
        self,
        request: ModelExecutionGraphRequest,
        authority: VerifiedExecutionGraphReadAuthority,
        topology: PinnedExecutionGraphTopology,
        admission: ExecutionGraphOwnershipAdmission,
        stored_chain: tuple[ModelExecutionGraphStoredChainAnnotation, ...],
    ) -> ModelExecutionGraphTerminalResult:
        if (
            type(authority) is not VerifiedExecutionGraphReadAuthority
            or request != authority.request
            or admission.owner.tenant_id != authority.tenant_id
            or admission.owner.correlation_id != authority.correlation_id
            or type(topology) is not PinnedExecutionGraphTopology
        ):
            raise PermissionError("graph fold requires admitted signed ownership")

        try:
            current_evidence = _admitted_structure(admission, topology)
            if request.cursor_mode is EnumExecutionGraphCursorMode.BOUNDED:
                source_cursors = request.source_cursors or ()
                if any(
                    cursor.topic not in topology.read_set.topics
                    for cursor in source_cursors
                ):
                    raise ValueError("cursor is outside the pinned read set")
                bounded_evidence = select_bounded_evidence(
                    current_evidence, source_cursors
                )
            else:
                bounds: dict[tuple[str, int], int] = {}
                for item in current_evidence:
                    key = (item.topic, item.partition)
                    bounds[key] = max(bounds.get(key, -1), item.kafka_offset)
                source_cursors = tuple(
                    ModelExecutionGraphSourceCursor(
                        topic=topic,
                        partition=partition,
                        max_kafka_offset=max_kafka_offset,
                    )
                    for (topic, partition), max_kafka_offset in sorted(bounds.items())
                )
                bounded_evidence = current_evidence
            graph = self._fold.handle(
                ModelExecutionGraphFoldRequest(
                    correlation_id=authority.correlation_id,
                    tenant_id=authority.tenant_id,
                    bounded_evidence=bounded_evidence,
                    topology=topology,
                    fold_version=_FOLD_VERSION,
                    grader_version=_GRADER_VERSION,
                    verdict_reducer_version=_VERDICT_REDUCER_VERSION,
                    source_cursors=source_cursors,
                    read_at=self._read_clock(),
                    withheld_count=admission.withheld_count,
                    stored_chain=stored_chain,
                )
            )
        except ValueError:
            return self._refusal(
                authority,
                code="invalid_graph_evidence",
                message="The captured graph evidence cannot be folded.",
            )
        return ModelExecutionGraphTerminalResult(
            tenant_id=authority.tenant_id,
            correlation_id=authority.correlation_id,
            workflow_type=self._workflow_type,
            status="completed",
            result=graph,
        )

    def _refusal(
        self,
        authority: VerifiedExecutionGraphReadAuthority,
        *,
        code: str,
        message: str,
    ) -> ModelExecutionGraphTerminalResult:
        return ModelExecutionGraphTerminalResult(
            tenant_id=authority.tenant_id,
            correlation_id=authority.correlation_id,
            workflow_type=self._workflow_type,
            status="failed",
            refusal=ModelExecutionGraphTerminalRefusal(code=code, message=message),
        )


__all__ = ["ExecutionGraphReadFold"]
