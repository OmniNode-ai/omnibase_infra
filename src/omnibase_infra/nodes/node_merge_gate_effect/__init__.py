# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""NodeMergeGateEffect -- merge gate decision persistence effect node.

Subscribes to ``onex.evt.platform.merge-gate-decision.v1`` Kafka events and
upserts merge gate decisions into the ``merge_gate_decisions`` PostgreSQL table.
When a QUARANTINE decision is issued, opens a Linear ticket with violation details.

Architecture:
    - node.py: Declarative node shell extending NodeEffect
    - handlers/: Merge gate upsert + Linear quarantine handler
    - models/: Payload models for merge gate events
    - registry/: Infrastructure registry for dependency injection
    - contract.yaml: Event subscription, handler routing, and I/O definitions

Node Type: EFFECT_GENERIC
Purpose: Execute PostgreSQL I/O for merge gate decision persistence.

Supported Operations:
    - merge_gate.upsert: Idempotent upsert + optional QUARANTINE Linear ticket

Related:
    - OMN-3140: Implementation ticket
"""

from __future__ import annotations

from omnibase_infra.nodes.node_merge_gate_effect.handlers import (
    HandlerUpsertMergeGate,
)
from omnibase_infra.nodes.node_merge_gate_effect.models import (
    ModelMergeGateResult,
    ModelMergeGateViolation,
)
from omnibase_infra.nodes.node_merge_gate_effect.node import (
    NodeMergeGateEffect,
)
from omnibase_infra.nodes.node_merge_gate_effect.registry import (
    RegistryInfraMergeGateEffect,
)

__all__: list[str] = [
    # Node
    "NodeMergeGateEffect",
    # Registry
    "RegistryInfraMergeGateEffect",
    # Handlers
    "HandlerUpsertMergeGate",
    # Models
    "ModelMergeGateResult",
    "ModelMergeGateViolation",
]
