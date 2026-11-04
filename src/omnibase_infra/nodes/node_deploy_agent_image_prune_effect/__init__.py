# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host-local image retention node (OMN-19998).

The timer supplies the owning effect package's contract explicitly to RuntimeLocal;
this node is not attached to shared runtime processes through onex.nodes discovery.
"""

from omnibase_infra.nodes.node_deploy_agent_image_prune_effect.node import (
    NodeDeployAgentImagePruneEffect,
)

__all__ = ["NodeDeployAgentImagePruneEffect"]
