# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Merge-sweep auto-merge effect - enables GitHub auto-merge on PRs."""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeMergeSweepAutoMergeEffect(NodeEffect):
    """Declarative effect node for enabling GitHub auto-merge.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeMergeSweepAutoMergeEffect"]
