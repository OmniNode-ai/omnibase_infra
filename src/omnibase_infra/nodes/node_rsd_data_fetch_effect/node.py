# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RSD data fetch effect - declarative effect node for ticket data retrieval."""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeRsdDataFetchEffect(NodeEffect):
    """Declarative effect node for fetching ticket and dependency data.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeRsdDataFetchEffect"]
