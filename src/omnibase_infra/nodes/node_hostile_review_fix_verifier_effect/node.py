# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative effect node: verify a hostile-review fixed-in disposition (OMN-17427)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeHostileReviewFixVerifierEffect(NodeEffect):
    """Check a fixed-in commit against the PR's own git history."""

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the fix verifier effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeHostileReviewFixVerifierEffect"]
