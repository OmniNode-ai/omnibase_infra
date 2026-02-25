# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handlers for NodeDecisionStoreEffect.

Two-stage write handler with structural conflict detection:
    - HandlerWriteDecision: Stage 1 (upsert) + Stage 2 (conflict detection)
    - HandlerWriteConflict: Idempotent conflict-pair insert

Also exports the structural_confidence pure function for testing.

Related Tickets:
    - OMN-2765: NodeDecisionStoreEffect implementation
    - OMN-2764: DB migrations
"""

from __future__ import annotations

from omnibase_infra.nodes.node_decision_store_effect.handlers.handler_write_conflict import (
    HandlerWriteConflict,
)
from omnibase_infra.nodes.node_decision_store_effect.handlers.handler_write_decision import (
    ACTIVE_INVARIANT_THRESHOLD,
    ALLOWED_DOMAINS,
    CONFLICT_WRITE_THRESHOLD,
    DecisionScopeKey,
    HandlerWriteDecision,
    structural_confidence,
)

__all__: list[str] = [
    "DecisionScopeKey",
    "HandlerWriteConflict",
    "HandlerWriteDecision",
    "structural_confidence",
    "ALLOWED_DOMAINS",
    "CONFLICT_WRITE_THRESHOLD",
    "ACTIVE_INVARIANT_THRESHOLD",
]
