# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure normalized evidence result for delegation graph replay (OMN-19729)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_envelope_delivery_count import (
    ModelEnvelopeDeliveryCount,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_observed_envelope_evidence import (
    ModelObservedEnvelopeEvidence,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_unresolved_parent import (
    ModelUnresolvedParent,
)


class ModelNormalizedReplayEvidence(BaseModel):
    """Deduplicated evidence with source, causal, and unresolved state retained."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    envelopes: tuple[ModelObservedEnvelopeEvidence, ...]
    observed_order: tuple[UUID, ...]
    topological_order: tuple[UUID, ...]
    delivery_counts: tuple[ModelEnvelopeDeliveryCount, ...]
    unresolved: tuple[ModelUnresolvedParent, ...]

    @property
    def delivery_count_by_envelope_id(self) -> dict[UUID, int]:
        """Convenience view; stored count records remain immutable."""
        return {item.envelope_id: item.delivery_count for item in self.delivery_counts}


__all__ = ["ModelNormalizedReplayEvidence"]
