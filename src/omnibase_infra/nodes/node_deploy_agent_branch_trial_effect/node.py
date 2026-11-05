# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative dev-agent branch trial effect."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDeployAgentBranchTrialEffect(NodeEffect):
    """Queue a pinned trial for the trusted host supervisor."""
