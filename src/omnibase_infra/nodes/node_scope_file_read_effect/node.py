# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Scope file read effect - declarative effect node for filesystem reads."""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeScopeFileReadEffect(NodeEffect):
    """Declarative effect node for reading plan files from the filesystem.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeScopeFileReadEffect"]
