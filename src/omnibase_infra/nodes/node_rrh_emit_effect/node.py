# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Declarative EFFECT node for RRH environment data collection.

Collects git state, runtime targets, and toolchain versions.
All behavior is defined in contract.yaml and delegated to handlers.
This node contains no custom logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeRRHEmitEffect(NodeEffect):
    """Declarative effect node for RRH environment data collection.

    Handlers:
        - ``HandlerRepoStateCollect``: branch, head_sha, is_dirty, repo_root, remote_url
        - ``HandlerRuntimeTargetCollect``: environment, kafka_broker, kubernetes_context
        - ``HandlerToolchainCollect``: pre_commit, ruff, pytest, mypy versions

    All behavior is defined in contract.yaml — no custom logic here.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
