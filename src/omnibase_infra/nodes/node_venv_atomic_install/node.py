# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative effect for the atomic in-place venv install."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeVenvAtomicInstall(NodeEffect):
    """Bind the atomic install operation through its contract."""
