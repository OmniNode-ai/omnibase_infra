# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Scope manifest write effect - declarative effect node for writing JSON manifests."""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeScopeManifestWriteEffect(NodeEffect):
    """Declarative effect node for writing scope manifest JSON files.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeScopeManifestWriteEffect"]
