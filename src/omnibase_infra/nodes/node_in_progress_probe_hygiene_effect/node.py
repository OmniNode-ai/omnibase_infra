# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In-progress probe hygiene sweep — declarative effect node.

All behavior is defined in contract.yaml - no custom logic here.
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeInProgressProbeHygieneEffect(NodeEffect):
    """Declarative effect node for the in-progress probe hygiene sweep.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeInProgressProbeHygieneEffect"]
