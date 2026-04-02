# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model health effect - declarative effect node for endpoint health probing."""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeModelHealthEffect(NodeEffect):
    """Declarative effect node for probing model endpoint health.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeModelHealthEffect"]
