# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""NodeDeltaBundleEffect -- PR lifecycle tracking as delta bundles.

Implements the ONEX effect node that tracks PRs from merge-gate decision
through to final outcome (merged, reverted, closed). Supports fix-PR
detection via the ``stabilizes:<pr_ref>`` label convention.

Architecture:
    - node.py: Declarative node shell extending NodeEffect
    - handlers/: PostgreSQL operation handlers (write_bundle, update_outcome)
    - models/: Payload models for gate-decision and pr-outcome events
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

__all__: list[str] = [
    # Node
    "NodeDeltaBundleEffect",
    # Handlers
    "HandlerWriteBundle",
    "HandlerUpdateOutcome",
    # Pure functions
    "parse_stabilizes_label",
]
