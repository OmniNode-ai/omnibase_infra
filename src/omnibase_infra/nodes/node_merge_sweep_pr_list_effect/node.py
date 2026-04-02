# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Merge-sweep PR list effect - declarative effect node for GitHub PR listing."""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeMergeSweepPRListEffect(NodeEffect):
    """Declarative effect node for listing open PRs from GitHub.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeMergeSweepPRListEffect"]
