# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure graph-fold input after request-time ownership and cursor selection."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.execution_graph_replay import (
    ModelExecutionGraphSourceCursor,
    ModelExecutionGraphStoredChainAnnotation,
    ModelExecutionGraphStoredVerdictAnnotation,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_observed_envelope_evidence import (
    ModelObservedEnvelopeEvidence,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PinnedExecutionGraphTopology,
)


class ModelExecutionGraphFoldRequest(BaseModel):
    """All semantic inputs are immutable evidence or explicit pinned versions."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    correlation_id: UUID
    tenant_id: UUID
    bounded_evidence: tuple[ModelObservedEnvelopeEvidence, ...]
    topology: PinnedExecutionGraphTopology
    fold_version: ModelSemVer
    grader_version: ModelSemVer
    verdict_reducer_version: ModelSemVer
    source_cursors: tuple[ModelExecutionGraphSourceCursor, ...]
    read_at: datetime
    stored_chain: tuple[ModelExecutionGraphStoredChainAnnotation, ...] = ()
    stored_verdicts: tuple[ModelExecutionGraphStoredVerdictAnnotation, ...] = ()
    withheld_count: int = Field(default=0, ge=0)


__all__ = ["ModelExecutionGraphFoldRequest"]
