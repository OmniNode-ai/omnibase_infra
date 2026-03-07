# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""NodeDeltaBundleEffect -- PR lifecycle tracking as delta bundles.

Implements the ONEX effect node that tracks PRs from merge-gate decision
through to final outcome (merged, reverted, closed). Supports fix-PR
detection via the ``stabilizes:<pr_ref>`` label convention.

Architecture:
    - node.py: Declarative node shell extending NodeEffect
    - handlers/: PostgreSQL operation handlers (write_bundle, update_outcome)
    - models/: Payload models for gate-decision and pr-outcome events
    - registry/: Infrastructure registry for dependency injection
    - contract.yaml: Intent routing and I/O definitions

Node Type: EFFECT_GENERIC
Purpose: Execute PostgreSQL I/O for delta bundle lifecycle tracking.

Related:
    - OMN-3142: Implementation ticket
    - Migration 039: delta_bundles table
"""

from __future__ import annotations

from omnibase_infra.nodes.node_delta_bundle_effect.handlers import (
    HandlerUpdateOutcome,
    HandlerWriteBundle,
    parse_stabilizes_label,
)
from omnibase_infra.nodes.node_delta_bundle_effect.node import (
    NodeDeltaBundleEffect,
)
from omnibase_infra.nodes.node_delta_bundle_effect.registry import (
    RegistryInfraDeltaBundleEffect,
)

__all__: list[str] = [
    # Node
    "NodeDeltaBundleEffect",
    # Registry
    "RegistryInfraDeltaBundleEffect",
    # Handlers
    "HandlerWriteBundle",
    "HandlerUpdateOutcome",
    # Pure functions
    "parse_stabilizes_label",
]
