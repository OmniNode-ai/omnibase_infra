# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""NodeDecisionStoreEffect — two-stage decision write effect node.

Implements the ONEX effect node that writes decisions to the decision_store
table using a two-stage commit pattern:

Stage 1 (committed independently):
    - Normalize + validate payload (scope_services, scope_domain, created_at)
    - Enforce supersession (superseded_by set → status=SUPERSEDED)
    - Upsert into decision_store ON CONFLICT (decision_id) DO UPDATE
    - Returns Stage1Result (was_insert, new_status, old_status)

Stage 2 (runs after Stage 1 commit — failure never rolls back Stage 1):
    - Query ACTIVE decisions in same scope_domain + scope_layer
    - Run structural_confidence() pure function against each pair
    - Write to decision_conflicts for pairs >= 0.3
    - Enforce ACTIVE invariant (demote to PROPOSED for >= 0.9 without dismissal)

Architecture:
    - node.py: Declarative node shell extending NodeEffect
    - handlers/: PostgreSQL operation handlers (write_decision, write_conflict)
    - models/: Payload models for both intent types
    - registry/: Infrastructure registry for dependency injection
    - contract.yaml: Intent routing and I/O definitions

Node Type: EFFECT_GENERIC
Purpose: Execute PostgreSQL I/O for the ONEX decision management system.

Supported Intent Types:
    - decision_store.write_decision: Two-stage upsert + conflict detection
    - decision_store.write_conflict: Idempotent conflict-pair insert

Related:
    - OMN-2765: Implementation ticket
    - OMN-2764: DB migrations (decision_store, decision_conflicts tables)
    - OMN-2763: omnibase_core store models
"""

from __future__ import annotations

from omnibase_infra.nodes.node_decision_store_effect.handlers import (
    ACTIVE_INVARIANT_THRESHOLD,
    ALLOWED_DOMAINS,
    CONFLICT_WRITE_THRESHOLD,
    DecisionScopeKey,
    HandlerWriteConflict,
    HandlerWriteDecision,
    structural_confidence,
)
from omnibase_infra.nodes.node_decision_store_effect.node import (
    NodeDecisionStoreEffect,
)
from omnibase_infra.nodes.node_decision_store_effect.registry import (
    RegistryInfraDecisionStoreEffect,
)

__all__: list[str] = [
    # Node
    "NodeDecisionStoreEffect",
    # Registry
    "RegistryInfraDecisionStoreEffect",
    # Handlers
    "HandlerWriteConflict",
    "HandlerWriteDecision",
    # Pure functions, models, and constants
    "DecisionScopeKey",
    "structural_confidence",
    "ALLOWED_DOMAINS",
    "CONFLICT_WRITE_THRESHOLD",
    "ACTIVE_INVARIANT_THRESHOLD",
]
