# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""DLQ depth monitor — declarative effect node.

All behavior is defined in contract.yaml - no custom logic here.
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDlqDepthMonitorEffect(NodeEffect):
    """Declarative effect node for the read-only DLQ depth/arrival probe.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeDlqDepthMonitorEffect"]
