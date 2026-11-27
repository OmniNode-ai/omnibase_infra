# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative canary receipt delivery effect; the handlers own all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeRuntimeCanaryReceiptEffect(NodeEffect):
    """Publish a canary receipt on the bus and record a delivered one for the dev roll."""
