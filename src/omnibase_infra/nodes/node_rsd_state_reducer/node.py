# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RSD state reducer - declarative FSM state tracker.

Tracks the RSD scoring workflow through states:
    pending -> fetching_data -> scoring -> storing -> complete | failed

All state transition logic is driven by contract.yaml.
"""

from __future__ import annotations

from omnibase_core.nodes.node_reducer import NodeReducer


class NodeRsdStateReducer(NodeReducer):
    """Declarative reducer for RSD workflow state tracking.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeRsdStateReducer"]
