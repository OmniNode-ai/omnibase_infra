# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Sync-revert watchdog — declarative effect node.

All behavior is defined in contract.yaml - no custom logic here.
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeSyncRevertWatchdogEffect(NodeEffect):
    """Declarative effect node for the Linear sync-revert detect+correct sweep.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeSyncRevertWatchdogEffect"]
