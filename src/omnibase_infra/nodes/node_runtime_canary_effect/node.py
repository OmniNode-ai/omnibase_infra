# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative runtime canary effect; the handlers own all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeRuntimeCanaryEffect(NodeEffect):
    """Boot a dev candidate alone, wake each dormant contract once, write a receipt."""
