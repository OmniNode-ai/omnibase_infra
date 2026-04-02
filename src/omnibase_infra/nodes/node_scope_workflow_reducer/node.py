# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Scope workflow reducer - declarative FSM state tracker.

Tracks the scope-check workflow through states:
    pending -> reading_file -> extracting -> writing -> complete | failed

All state transition logic is driven by contract.yaml.
"""

from __future__ import annotations

from omnibase_core.nodes.node_reducer import NodeReducer


class NodeScopeWorkflowReducer(NodeReducer):
    """Declarative reducer for scope-check workflow state tracking.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeScopeWorkflowReducer"]
