# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for NodeDecisionStoreEffect.

Provides payload models for the two-stage write handler:
    - ModelPayloadWriteDecision: Input for Stage 1 (decision_store upsert)
    - ModelPayloadWriteConflict: Input for Stage 2 (decision_conflicts insert)
    - ModelDecisionWriteResult: Output model for both stages

Related Tickets:
    - OMN-2765: NodeDecisionStoreEffect implementation
    - OMN-2764: DB migrations (decision_store, decision_conflicts tables)
"""

from __future__ import annotations

from omnibase_infra.nodes.node_decision_store_effect.models.model_payload_write_conflict import (
    ModelPayloadWriteConflict,
)
from omnibase_infra.nodes.node_decision_store_effect.models.model_payload_write_decision import (
    ModelPayloadWriteDecision,
)

__all__: list[str] = [
    "ModelPayloadWriteConflict",
    "ModelPayloadWriteDecision",
]
