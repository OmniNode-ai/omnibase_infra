# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""GitHub webhook ingress -- declarative EFFECT node (OMN-19492)."""

from __future__ import annotations

from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeGitHubWebhookIngressEffect(NodeEffect):
    """Verify signed GitHub deliveries and emit PR-state and merge events."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodeGitHubWebhookIngressEffect"]
