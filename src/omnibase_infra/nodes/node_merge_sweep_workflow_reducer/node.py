# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Merge-sweep workflow reducer - declarative FSM state tracker.

Tracks the merge-sweep workflow through states:
    pending -> scanning -> classifying -> merging -> complete | failed

All state transition logic is driven by contract.yaml.
"""

from __future__ import annotations

from omnibase_core.nodes.node_reducer import NodeReducer


class NodeMergeSweepWorkflowReducer(NodeReducer):
    """Declarative reducer for merge-sweep workflow state tracking.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeMergeSweepWorkflowReducer"]
