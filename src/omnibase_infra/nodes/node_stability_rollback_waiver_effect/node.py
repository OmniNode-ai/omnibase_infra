# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host effect; the handler owns the decision."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeStabilityRollbackWaiverEffect(NodeEffect):
    """Decide whether a stability deploy with no retained rollback point may run."""
